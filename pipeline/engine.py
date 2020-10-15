import os
import numpy as np
import pandas as pd
from os.path import join, exists
from torch.utils.data import DataLoader
from torch import nn
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

from pipeline.data import MoaDataset
from pipeline.model import get_model
from pipeline.utils import split_data, print_table, SETTINGS


class Engine:
    def __init__(self, config):
        self.config = config

        # Setup project folders
        self.project_dir = join(SETTINGS['PROJECTS_DIR'], self.config['project_name'])
        if not exists(self.project_dir):
            os.mkdir(self.project_dir)

        self.ensemble_dir = join(self.project_dir, 'ensemble')
        if not exists(self.ensemble_dir):
            os.mkdir(self.ensemble_dir)

        # Load dataframes
        self.train_features = pd.read_csv(SETTINGS['TRAIN_FEATURES_PATH'])
        self.train_targets = pd.read_csv(SETTINGS['TRAIN_TARGETS_PATH'])
        self.test_features = pd.read_csv(SETTINGS['TEST_FEATURES_PATH'])

        # Split control cases
        mask = self.train_features['cp_type'] != 'ctl_vehicle'
        self.train_features_control = self.train_features[~mask]
        self.train_targets_control = self.train_targets[~mask]
        self.train_features = self.train_features[mask]
        self.train_targets = self.train_targets[mask]

        self.data_folds = []

    def preprocess(self):
        # Setup feature scaler
        features = [c for c in self.train_features.columns if '-' in c]
        all_features = np.vstack([self.train_features[features].values, self.test_features[features].values])
        scaler = StandardScaler()
        scaler.fit(all_features)

        def process_df(df):
            df.loc[:, 'cp_time'] = df.loc[:, 'cp_time'].map({24: -1, 48: 0, 72: 1})
            df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})
            df = df.drop('cp_type', axis=1)
            df.loc[:, features] = scaler.transform(df.loc[:, features])
            return df

        def label_prob(col):
            label_data = self.train_targets[col]
            label_sum = label_data.sum()
            return np.log(label_sum / (len(label_data) - label_sum))

        # Featurize cp_time and cp_dose
        self.train_features = process_df(self.train_features)
        self.test_features = process_df(self.test_features)

        # Get initial biases for NN classification layer (-ln(pos/neg))
        label_probs_path = join(SETTINGS['RAW_DATA_DIR'], 'label_probs.npy')
        if not exists(label_probs_path):
            label_probs = [label_prob(col) for col in self.train_targets.columns[1:]]
            np.save(label_probs_path, label_probs)

        # Split data into folds
        kfold = split_data(self.train_features, self.train_targets, self.config['n_folds'])
        for fold_i, ((x_train, y_train), (x_valid, y_valid)) in enumerate(kfold):
            # Setup datasets
            self.data_folds.append(((x_train, y_train), (x_valid, y_valid)))

    def postprocess(self):
        return

    def train(self):
        losses = dict(train=[], valid=[])
        # train each model one at a time
        for model_i, model_dict in enumerate(self.config['ensemble']):
            print('Training model %d...' % model_i)

            model_dir = join(self.ensemble_dir, 'model_%d' % model_i)
            if not exists(model_dir):
                os.mkdir(model_dir)
            model_dict['model_dir'] = model_dir

            model_losses = dict(train=[], valid=[])
            # Split data into k-folds and train each fold
            for fold_i in range(self.config['n_folds']):
                print('Training model %d on fold %d...' % (model_i, fold_i))
                # Get fold data
                data_fold = self.data_folds[fold_i]

                # Setup and train model
                model_dict['fold'] = fold_i
                model = get_model(model_dict, data_fold)

                if exists(model.best_weights_path):
                    print('Loading weights...')
                    model.load(model.best_weights_path)
                else:
                    model.train()

                # Gather losses
                model_losses['train'].append(model.train_metrics.min())
                model_losses['valid'].append(model.valid_metrics.min())

            avg_model_loss = float(np.mean(model_losses['valid']))
            print('Finished training model %d -- Avg validation loss: %.5f' % (model_i, avg_model_loss))

            losses['train'].append(model_losses['train'])
            losses['valid'].append(model_losses['valid'])

        # Print model summary
        print_table(losses['train'], 'Model', 'Fold', 'Model Training Summary')
        print_table(losses['valid'], 'Model', 'Fold', 'Model Validation Summary')

    def predict(self):
        for model_i, model_dict in enumerate(self.config['ensemble']):
            for fold_i in range(self.config['n_folds']):
                valid_pred_path = join(model_dict['model_dir'], 'fold-%d-valid-predictions.npy' % fold_i)
                test_pred_path = join(model_dict['model_dir'], 'fold-%d-test-predictions.npy' % fold_i)
                if exists(valid_pred_path) and exists(test_pred_path):
                    print('Previous predictions exist: skipping')
                else:
                    print('Prediction with model %d on fold %d...' % (model_i, fold_i))
                    # Get fold data
                    data_fold = self.data_folds[fold_i]

                    # Setup and train model
                    model_dict['fold'] = fold_i
                    model = get_model(model_dict, data_fold)

                    # Save validation predictions
                    avg_loss, predictions = model.validation(return_preds=True)
                    np.save(valid_pred_path, predictions)

                    # Save test predictions

                    predictions = model.predict(self.test_features)
                    predictions = 1 / (1 + np.exp(-np.array(predictions)))

                    for tta_i in range(self.config['n_tta']):
                        tta_features = self.test_features.copy()
                        tta_features.iloc[:, 1:] += np.random.normal(scale=0.25, size=tta_features.iloc[:, 1:].shape)
                        pred_i = model.predict(tta_features)
                        pred_i = 1 / (1 + np.exp(-np.array(pred_i)))
                        predictions += pred_i

                    predictions /= (self.config['n_tta'] + 1)
                    np.save(test_pred_path, predictions)

    def load(self, stage):
        all_predictions = []
        for model_i, model_dict in enumerate(self.config['ensemble']):
            model_dir = join(self.ensemble_dir, 'model_%d' % model_i)
            model_predictions = []
            for fold_i in range(self.config['n_folds']):
                model_predictions.append(np.load(join(model_dir, 'fold-%d-%s-predictions.npy' % (fold_i, stage))))
            all_predictions.append(np.array(model_predictions))
        all_predictions = np.array(all_predictions)
        return all_predictions

    def ensemble_opt(self):
        # load validation predictions from all the ensemble models for all folds
        valid_predictions = self.load('valid')

        # load validation ground truth
        valid_truth = np.vstack([self.data_folds[fold_i][1][1].values[:, 1:] for fold_i in range(self.config['n_folds'])])

        # Optimization functions
        def blend_fn(x):
            return p1 * x[0] + p2 * x[1] + p3 * x[2]

        def objective_fn(x):
            newp = blend_func(x)
            return tf.keras.losses.binary_crossentropy(L, newp).numpy().mean()

        return

    def ensemble(self):
        # 2-step process:
        #  1. Get optimal blending parameters from validation
        #  2. Blend test predictions using optimal parameters

        # Step 1
        # Skip if oom_method is mean
        if self.config['oom_method'] != 'mean':
            self.ensemble_opt()

        all_predictions = self.load('test')

        # Perform out-of-fold ensemble
        all_predictions = np.mean(all_predictions, axis=1)

        # mean out-of-model ensemble
        if self.config['oom_method'] == 'mean':
            all_predictions = np.mean(all_predictions, axis=0)
        # else:

        test_ids = self.test_features['sig_id'].values
        data = np.hstack([np.expand_dims(test_ids, 1), all_predictions])
        sub_df = pd.DataFrame(data, columns=self.train_targets.columns)
        sub_df.to_csv(join(self.project_dir, 'submission.csv'), index=False)
        print(sub_df.head())

    def run(self):
        self.preprocess()
        self.train()
        self.predict()
        self.ensemble()
        # self.postprocess()
