import torch
import glob
import os
import re
import wave
import csv
from torchvision import datasets, transforms
import torch.nn as nn
from base import BaseDataLoader
import pyroomacoustics as pra
import numpy as np
from data_loader.preprocessing import *
import matplotlib.pyplot as plt
from scipy.io import wavfile
import tempfile
import boto3

s3 = boto3.resource('s3')

class GSPDataset(torch.utils.data.Dataset):
    def __init__(self, pre_training, normalization):
        super().__init__()
        self.data = []
        self.pre_training = pre_training
        self.normalization = normalization
        self.load()

    def load(self):
        if self.pre_training:
            with tempfile.TemporaryFile() as temp:
                res = s3.Object(bucket_name="wg3-1", key="mita/simulation_data/train_random_100000.npz").download_fileobj(temp)
                temp.seek(0)
                dataset = np.load(temp)
                dataset_dict = dict(dataset)
                X, y = dataset_dict["X"], dataset_dict["X"]
                X, y = np.array(X), np.array(y)
                X = self.data_normalization(X)
                y = self.data_normalization(y)
        else:
            with tempfile.TemporaryFile() as temp:
                res = s3.Object(bucket_name="wg3-1", key="mita/simulation_data/test_random_1000.npz").download_fileobj(temp)
                temp.seek(0)
                dataset = np.load(temp)
                dataset_dict = dict(dataset)
                X, y = dataset_dict["X"], dataset_dict["X"]
                X, y = np.array(X), np.array(y)
                X = self.data_normalization(X)
                y = self.data_normalization(y)

        print("X.shape:", X.shape)
        print("y.shape:", y.shape)

        for i in range(X.shape[0]):
            # データを float32 型で準備
            data_tensor = torch.tensor(X[i].astype("float32"))
            # # ターゲットを float32 型で準備
            target_tensor = torch.tensor(y[i].astype("float32"))
            self.data.append((data_tensor, target_tensor))

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]
    
    def data_normalization(self, x):
        if self.normalization == "standard":
            _mean = np.mean(x)
            _std = np.std(x)
            x = (x - _mean) / _std
        elif self.normalization == "minmax":
            _min = np.min(x)
            _max = np.max(x)
            x = (x - _min) / (_max - _min)
        else:
            pass
        return x

class GSPDataLoader(BaseDataLoader):
    """
    Tongue data loading demo using BaseDataLoader
    """
    def __init__(self, batch_size, shuffle=True, validation_split=0.2, num_workers=1, pre_training=False, normalization="standard"):
        self.dataset = GSPDataset(pre_training, normalization)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        print("Dataset size: ", len(self.dataset))