import os
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from apex import amp
from tqdm import tqdm
from os.path import join, exists
from skmultilearn.adapt import MLkNN
from sklearn.neighbors import KNeighborsClassifier, KDTree
import faiss

from pipeline.data import MoaDataset, get_dataloader, mlsmote
from pipeline.networks import MoaDenseNet
from pipeline.utils import log_loss, initialize_weights, Metrics


class NeuralNetModel:
    def __init__(self, model_dict, data_fold):
        self.model_dict = model_dict
        self.best_weights_path = join(self.model_dict['model_dir'], 'fold-%d-weights.pth' % self.model_dict['fold'])
        self.epoch = 0
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_metrics = Metrics()
        self.valid_metrics = Metrics()

        # Unpack data fold and initialize dataloaders
        (self.x_train, self.y_train), (self.x_valid, self.y_valid) = data_fold
        self.input_dim = len(self.x_train.columns) - 1
        self.output_dim = len(self.y_train.columns) - 1

        if model_dict['use_smote']:
            print(self.x_train.head())
            print(self.y_train.head())
            self.x_train, self.y_train = mlsmote(self.x_train, self.y_train, 30000)
            print(self.x_train.head())
            print(self.y_train.head())

        self.train_dataloader = get_dataloader(self.x_train,
                                               self.y_train,
                                               model_dict,
                                               model_dict['augmentations'],
                                               shuffle=True)
        self.valid_dataloader = get_dataloader(self.x_valid,
                                               self.y_valid,
                                               model_dict)

        if model_dict['model'] == 'MoaDenseNet':
            self.model = MoaDenseNet(
                self.input_dim,
                self.output_dim,
                model_dict['n_hidden_layer'],
                model_dict['hidden_dim'],
                model_dict['dropout'],
                model_dict['activation'],
                model_dict['normalization'],
            )

        if model_dict['use_smart_init']:
            self.model.layers = initialize_weights(self.model.layers, 'all')

        # Setup optimizer
        if model_dict['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=model_dict['learning_rate'],
                                       momentum=model_dict['momentum'])
        elif model_dict['optimizer'] == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=model_dict['learning_rate'],
                                        weight_decay=model_dict['weight_decay'])
        else:
            Exception('Optimizer not supported.')

        # Setup scheduler
        if model_dict['scheduler'] == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                  patience=3,
                                                                  threshold=0.00001)
        elif model_dict['scheduler'] == 'OneCycleLR':
            self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                           max_lr=0.01,
                                                           pct_start=0.1,
                                                           div_factor=1e3,
                                                           epochs=model_dict['n_epochs'],
                                                           steps_per_epoch=len(self.train_dataloader))
        else:
            Exception('Scheduler not supported.')

        # Save initial states of model, optimizer and scheduler
        self.init_states = dict(
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict()
        )
        self.model = self.model.cuda()

        # Setup AMP
        if model_dict['use_amp']:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")

    def train_epoch(self):
        losses = []
        self.model.train()
        for features, targets, ids in self.train_dataloader:
            features = {k: v.cuda().float() for k, v in features.items()}
            targets = targets.cuda().float()
            predictions = self.model(features)
            loss = self.criterion(predictions, targets)
            for p in self.model.parameters():
                p.grad = None
            if self.model_dict['use_amp']:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.optimizer.step()
            if self.model_dict['scheduler'] == 'OneCycleLR':
                self.scheduler.step()
            losses.append(loss.detach().cpu().numpy())
        avg_loss = float(np.mean(losses))
        self.train_metrics.add(avg_loss)
        return avg_loss

    def validation(self, return_preds=False):
        losses = []
        predictions = []
        self.model.eval()

        for features, targets, ids in self.valid_dataloader:
            features = {k: v.cuda().float() for k, v in features.items()}
            targets = targets.cuda().float()
            pred = self.model(features)
            loss = self.criterion(pred, targets)
            losses.append(loss.detach().cpu().numpy())
            if return_preds:
                predictions.extend(pred.detach().cpu().numpy())

        avg_loss = float(np.mean(losses))

        if return_preds:
            return avg_loss, predictions
        else:
            return avg_loss

    def train(self):
        print('Training NN with %d samples, validating with %d' % (len(self.x_train), len(self.x_valid)))
        print('Input_dim: %d Output_dim: %d' % (self.input_dim, self.output_dim))
        for epoch in range(self.model_dict['n_epochs']):
            self.epoch = epoch
            time0 = time.time()

            train_avg_loss = self.train_epoch()
            valid_avg_loss = self.validation()

            if self.epoch == 0 or valid_avg_loss < self.valid_metrics.min():
                # new best weights
                self.save(self.best_weights_path)
            self.valid_metrics.add(valid_avg_loss)

            if self.model_dict['scheduler'] != 'OneCycleLR':
                self.scheduler.step(valid_avg_loss)

            time1 = time.time()
            epoch_time = time1 - time0
            if self.model_dict['verbose']:
                print('Epoch %d/%d Train Loss: %.5f Valid Loss: %.5f Time: %.2f' % (
                    epoch+1, self.model_dict['n_epochs'], train_avg_loss, valid_avg_loss, epoch_time
                ))

        # restore weights from best epoch
        self.load(self.best_weights_path)

    def predict(self, test_features):
        test_dataset = MoaDataset(test_features)
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=self.model_dict['batch_size'],
                                     num_workers=self.model_dict['num_workers'],
                                     pin_memory=True)
        predictions = []
        self.model.eval()
        for features, ids in test_dataloader:
            features = {k: v.cuda().float() for k, v in features.items()}
            pred = self.model(features)
            predictions.extend(pred.detach().cpu().numpy())
        return predictions

    def reset(self):
        self.model.load_state_dict(self.init_states['model'])
        self.optimizer.load_state_dict(self.init_states['optimizer'])
        self.scheduler.load_state_dict(self.init_states['scheduler'])
        self.train_metrics.reset()
        self.valid_metrics.reset()

    def save(self, path):
        self.model.eval()
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'train_metrics': self.train_metrics.data,
            'valid_metrics': self.valid_metrics.data,
        }, path)

    def load(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self.train_metrics.load(state_dict['train_metrics'])
        self.valid_metrics.load(state_dict['valid_metrics'])


class TargetModel:
    def __init__(self, model_dict, data_fold):
        self.model_dict = model_dict
        self.best_weights_path = join(self.model_dict['model_dir'], 'fold-%d-weights.pth' % self.model_dict['fold'])
        (self.x_train, self.y_train), (self.x_valid, self.y_valid) = data_fold
        self.model = None
        self.target_cols = self.y_train.columns[1:]
        self.feature_cols = np.array([c for c in self.x_train if 'g-' in c or 'c-' in c])
        self.valid_metrics = Metrics()
        self.use_cols = {}

    def feature_selector(self, cond):
        time_controls = [-1, 0, 1]
        dose_controls = [0, 1]

        use_feats = []
        for col in self.feature_cols:
            col_data = []

            for cp_time in time_controls:
                query = cond[cond['cp_time'] == cp_time]
                query = [query[query['cp_dose'] == dose] for dose in dose_controls]
                if min([len(x) for x in query]) == 0:
                    delta = 0.0
                else:
                    centers = [q[col].mean() for q in query]
                    delta = abs(centers[0] - centers[1])
                col_data.append(delta)

            for cp_dose in dose_controls:
                query = cond[cond['cp_dose'] == cp_dose]
                centers = [query[query['cp_time'] == cp_time][col].mean() for cp_time in time_controls]
                deltas = [abs(centers[0] - centers[1]), abs(centers[1] - centers[2])]
                avg_delta = np.mean(deltas)
                col_data.append(avg_delta)

            use_feats.append(np.max(col_data))

        return np.argsort(use_feats)[::-1][:self.model_dict['n_features']]

    def train_one(self, target):
        target_mask = self.y_train[target] == 1
        if np.sum(target_mask) > self.model_dict['n_cutoff']:
            condition = self.x_train[target_mask]
            use_features = self.feature_selector(condition)
            self.use_cols[target] = np.hstack([['sig_id', 'cp_time', 'cp_dose'], self.feature_cols[use_features]])
            drop_cols = [c for c in self.x_train.columns if c not in self.use_cols[target]]
            x_train = self.x_train.drop(drop_cols, axis=1)
            y_train = self.y_train[['sig_id', target]]
            x_valid = self.x_valid.drop(drop_cols, axis=1)
            y_valid = self.y_valid[['sig_id', target]]
            data_fold = ((x_train, y_train), (x_valid, y_valid))

            save_path = join(self.model_dict['model_dir'], target)
            if not exists(save_path):
                os.mkdir(save_path)

            model_dict = self.model_dict['model']
            model_dict['model_dir'] = save_path
            model_dict['fold'] = self.model_dict['fold']
            model_dict['use_smart_init'] = False
            model = NeuralNetModel(self.model_dict['model'], data_fold)
            model.model.layers = initialize_weights(model.model.layers, target)
            model.train()

            if self.model is None:
                self.model = model

    def train(self):
        for target in self.target_cols:
            self.train_one(target)
        self.validation()

    def run(self, features):
        predictions = []
        for target in self.target_cols:
            if target not in self.use_cols.keys():
                pred = np.zeros((len(features)))
            else:
                target_features = features[self.use_cols[target]]
                self.model.load(join(self.model_dict['model_dir'], 'fold-%d-%s.pth' % (self.model_dict['fold'], target)))
                pred = self.model.predict(target_features)
                pred = pred.detach().cpu().numpy()
            predictions.append(pred)
        predictions = np.array(predictions)
        return predictions

    def validation(self, return_preds=False):
        predictions = self.run(self.x_valid)
        loss = log_loss(predictions, self.y_valid)
        self.valid_metrics.add(loss)
        if return_preds:
            return loss, predictions
        else:
            return loss

    def predict(self, test_features):
        return self.run(test_features)

    def save(self, path):
        return


class CpuKnnModel:
    def __init__(self, model_dict, data_fold):
        self.model_dict = model_dict
        (x_train, y_train), (x_valid, y_valid) = data_fold

        self.clip_threshold = model_dict['clip_threshold'] if model_dict['clip_threshold'] is not None else 1e10
        self.train_features = self.clip(x_train.values[:, 1:].astype(np.float), self.clip_threshold)
        self.valid_features = self.clip(x_valid.values[:, 1:].astype(np.float), self.clip_threshold)
        self.train_targets = y_train.values[:, 1:].astype(np.float)
        self.valid_targets = y_valid.values[:, 1:].astype(np.float)

        # Setup metric
        self.valid_metrics = Metrics()

        self.classifier = None #MLkNN(k=1)

        self.criterion = nn.BCELoss()

    def clip(self, data, threshold):
        return np.where(np.logical_and(data < threshold, data > -threshold), data, 0)

    def train(self):
        # self.classifier.fit(self.train_features, self.train_targets)
        self.validation()
        return

    def validation(self, return_preds=True):
        predictions = self.predict(self.valid_features, transform=False)
        loss = self.evaluate(predictions, self.valid_targets)
        self.valid_metrics.add(loss)
        if return_preds:
            return loss, predictions
        else:
            return loss

    def evaluate(self, prediction, target):
        prediction = torch.Tensor(prediction).cuda().float()
        target = torch.Tensor(target).cuda().float()
        loss = self.criterion(prediction, target)
        return loss.detach().cpu().numpy()

    def predict(self, test_features, transform=True):
        if transform:
            test_features = self.clip(test_features.values[1:], self.clip_threshold)

        predictions = []
        for features in tqdm(test_features):
            predictions.append(self.train_targets[np.argmin(np.sum(np.square(self.train_features - features), axis=1))])

        return predictions

    def save(self, path):
        print(dir(self.classifier))


class FaissKnnModel:
    def __init__(self, model_dict, data_fold):
        self.model_dict = model_dict
        (x_train, y_train), (x_valid, y_valid) = data_fold

        self.threshold = self.model_dict['clip_threshold']
        self.amplify = self.model_dict['amplify']
        self.max_distance = self.model_dict['max_distance']

        self.train_features = self.transform(x_train.values[:, 1:].astype(np.float))
        self.valid_features = self.transform(x_valid.values[:, 1:].astype(np.float))
        self.train_targets = y_train.values[:, 1:].astype(np.float)
        self.valid_targets = y_valid.values[:, 1:].astype(np.float)

        # Setup metric
        self.valid_metrics = Metrics()

        self.index = faiss.IndexFlatL2(self.train_features.shape[1])

    def transform(self, data):
        if self.threshold is not None:
            data = np.where(np.logical_and(data < self.threshold, data > -self.threshold), data, 0)

        if self.amplify is not None:
            data = data ** self.amplify

        return data

    def train(self):
        self.index.add(np.ascontiguousarray(self.train_features).astype('float32'))
        self.validation()

    def run(self, features):
        distances, indicies = self.index.search(np.ascontiguousarray(features).astype('float32'), 1)
        predictions = self.train_targets[indicies.flatten()]
        if self.max_distance is not None:
            predictions[np.where(distances.flatten() > self.model_dict['max_distance'])] = 0
        return predictions

    def validation(self, return_preds=True):
        predictions = self.run(self.valid_features)
        loss = log_loss(predictions, self.valid_targets)
        self.valid_metrics.add(loss)
        if return_preds:
            return loss, predictions
        else:
            return loss

    def predict(self, test_features):
        test_features = self.transform(test_features.values[:, 1:].astype(np.float))
        predictions = self.run(test_features)
        return predictions

    def save(self, path):
        return


def get_model(model_dict, data_fold):
    model_type = model_dict['type'].upper()
    if model_type == 'NN':
        return NeuralNetModel(model_dict, data_fold)
    elif model_type == 'KNN':
        return FaissKnnModel(model_dict, data_fold)
    elif model_type == 'TARGET':
        return TargetModel(model_dict, data_fold)
