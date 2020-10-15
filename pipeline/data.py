import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import faiss
from sklearn.neighbors import NearestNeighbors


# Sampling functions
def nearest_neighbour(x, neigh):
    # nbs = NearestNeighbors(n_neighbors=neigh, metric='euclidean', algorithm='kd_tree').fit(x)
    # euclidean, indices = nbs.kneighbors(x)
    index = faiss.IndexFlatL2(x.shape[1])
    index.add(np.ascontiguousarray(x).astype('float32'))
    distances, indicies = index.search(np.ascontiguousarray(x).astype('float32'), neigh)
    return indicies


def mlsmote(x, y, n_sample, neigh=5):
    x_data = x.drop('sig_id', axis=1)
    # y_data = y.drop('sig_id', axis=1)
    indices2 = nearest_neighbour(x_data.values, neigh=neigh)
    n = len(indices2)
    new_x = []
    target = []
    for _ in tqdm(range(n_sample)):
        reference = random.randint(0, n - 1)
        neighbor = random.choice(indices2[reference, 1:])
        all_point = indices2[reference]
        nn_df = y[y.index.isin(all_point)].drop('sig_id', axis=1)
        ser = nn_df.sum(axis=0, skipna=True)
        labels = np.array([1 if val > 0 else 0 for val in ser])
        target.append(np.hstack([x.iloc[reference]['sig_id'], labels]))
        ratio = random.random()
        gap = x_data.iloc[reference, :] - x_data.iloc[neighbor, :]
        features = np.array(x_data.iloc[reference, :] + ratio * gap)
        new_x.append(np.hstack([x.iloc[reference]['sig_id'], features]))
    target = np.array(target)
    new_x = pd.DataFrame(new_x, columns=x.columns)
    target = pd.DataFrame(target, columns=y.columns)
    return new_x, target


# Augmentation classes
class GaussianNoise:
    def __init__(self, std=0.05, p=0.9, row_p=0.9):
        self.std = std
        self.p = p
        self.row_p = row_p

    def __call__(self, features):
        if np.random.random() <= self.p:
            noise = np.random.normal(scale=self.std, size=features.shape)
            prob = np.random.choice([0, 1], size=features.shape, p=[1-self.row_p, self.row_p])
            features += (noise * prob)
        return features


class MixUp:
    # Averaging (mixing) multiple classes
    def __init__(self):
        return

    def __call__(self, features):
        return


class CutOut:
    # Dropout on input
    def __init__(self):
        return

    def __call__(self, features):
        return


class CutMix:
    # Cut and mix multiple classes
    def __init__(self):
        return

    def __call__(self, features):
        return


class AugmentEngine:
    word_to_class = dict(noise=GaussianNoise, mixup=MixUp, cutout=CutOut, cutmix=CutMix)

    def __init__(self, aug_list):
        self.aug_list = aug_list
        self.augmentations = [self.word_to_class[w]() for w in self.aug_list]

    def __call__(self, features):
        for augmentation in self.augmentations:
            features = augmentation(features)
        return features


# Dataset class

class MoaDataset(Dataset):
    def __init__(self, feature_df, target_df=None, augmentation_engine=None, split_features=False):
        super(MoaDataset, self).__init__()
        self.feature_df = feature_df.copy()
        self.train = type(target_df) == pd.DataFrame
        if self.train:
            self.target_df = target_df.copy()
        self.augmentation_engine = augmentation_engine
        self.split_features = split_features

        self.length = len(self.feature_df)
        self.features = self.feature_df.values[:, 1:].astype(np.float32)
        self.labels = self.feature_df['sig_id'].values

        if self.train:
            tdf = self.target_df
            self.targets = tdf.values[:, 1:].astype(np.int)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        ids = self.labels[idx]
        features = self.features[idx]

        if self.augmentation_engine is not None:
            features = self.augmentation_engine(features)

        if self.split_features:
            features_dict = dict(
                meta=features[:2],
                gene=features[2:774],
                cell=features[774:]
            )
        else:
            features_dict = dict(
                features=features
            )

        if self.train:
            y = self.targets[idx]
            return features_dict, y, ids
        else:
            return features_dict, ids


def get_dataloader(x, y, model_dict, augmentations=None, shuffle=False):
    if augmentations is not None:
        augmentations = AugmentEngine(augmentations)

    dataset = MoaDataset(x, y, augmentations)
    args = dict(
        batch_size=model_dict['batch_size'],
        num_workers=model_dict['num_workers'],
        pin_memory=True,
        shuffle=shuffle
    )
    return DataLoader(dataset, **args)
