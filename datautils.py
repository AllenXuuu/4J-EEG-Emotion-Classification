import torch
from scipy import io
import numpy as np
import random


class Dataloader(object):
    def __init__(self):
        # (15,3394,310)
        self.feature = np.stack(io.loadmat('./data/EEG_X.mat')['X'][0], 0)
        # map [-1,0,1] into [0,1,2]
        self.label = np.stack(io.loadmat('./data/EEG_Y.mat')['Y'][0], 0).squeeze(-1) + 1
        self.num_human = self.feature.shape[0]
        self.length = self.feature.shape[1]
        # self.mean = []
        # self.std = []
        # for i in range(15):
        #     train_idx = [i for i in range(self.num_human) if i != val_idx]
        #     train_feature = self.feature[train_idx]
        #     self.mean.append(np.mean(train_feature, axis=0))
        #     self.std.append(np.std(train_feature, axis=0))

    def get_train_data(self, val_idx):
        train_idx = [i for i in range(self.num_human) if i != val_idx]
        random.shuffle(train_idx)
        train_feature = self.feature[train_idx]
        train_label = self.label[train_idx]
        return train_feature, train_label

    def get_val_data(self, val_idx):

        val_idx = [val_idx]
        val_feature = self.feature[val_idx]
        val_label = self.label[val_idx]
        return val_feature, val_label

    def train_iter(self, val_idx, bz=14, shuffle_bz=True):
        train_idx = [i for i in range(self.num_human) if i != val_idx]
        if shuffle_bz:
            random.shuffle(train_idx)
        train_feature = self.feature[train_idx]
        train_label = self.label[train_idx]

        for i in range((len(train_idx) + bz - 1) // bz):
            train_feature = self.feature[train_idx[bz * i:bz * (i + 1)]]
            train_label = self.label[train_idx[bz * i:bz * (i + 1)]]
            if num_split is None:
                yield train_feature, train_label
            else:
                temporal_index = list(range(self.length))
                if shuffle_temporal:
                    random.shuffle(temporal_index)
                splits = [i * train_feature.shape[1] // num_split for i in range(num_split + 1)]
                for j in range(num_split):
                    temporal_slice = temporal_index[splits[j]:splits[j+1]]
                    yield train_feature[:, temporal_slice], train_label[:, temporal_slice]
