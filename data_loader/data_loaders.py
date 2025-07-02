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
from data_loader.pre_processing import *
import matplotlib.pyplot as plt
from scipy.io import wavfile
import tempfile
import boto3

s3 = boto3.resource('s3')

class GSPDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, normalization, is_positioning):
        super().__init__()
        self.data = []
        self.data_path = data_path
        self.normalization = normalization
        self.is_positioning = is_positioning
        self.load()

    def load(self):
        if self.is_positioning:
            with tempfile.TemporaryFile() as temp:
                res = s3.Object(bucket_name="wg3-1", key=self.data_path).download_fileobj(temp)
                temp.seek(0)
                dataset = np.load(temp)
                dataset_dict = dict(dataset)
                X, y = dataset_dict["X"], dataset_dict["pos"]
                X, y = np.array(X), np.array(y)
                X = self.data_normalization(X)
        else:
            with tempfile.TemporaryFile() as temp:
                res = s3.Object(bucket_name="wg3-1", key=self.data_path).download_fileobj(temp)
                temp.seek(0)
                dataset = np.load(temp)
                dataset_dict = dict(dataset)
                X, y = dataset_dict["X"], dataset_dict["X"]
                X, y = np.array(X), np.array(y)
                X = self.data_normalization(X)
                y = self.data_normalization(y)

        print("data_loader {} loaded".format(self.data_path))
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
    def __init__(self, data_path, batch_size, shuffle=True, validation_split=0.2, num_workers=1, normalization="standard", is_positioning=False):
        self.dataset = GSPDataset(data_path, normalization, is_positioning)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        print("Dataset size: ", len(self.dataset))

class GSPMeasurementDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, normalization):
        super().__init__()
        self.data = []
        self.data_path = data_path
        self.normalization = normalization
        self.load()

    def load(self):
        with tempfile.TemporaryFile() as temp:
            res = s3.Object(bucket_name="wg3-1", key=self.data_path).download_fileobj(temp)
            temp.seek(0)
            dataset = np.load(temp)
            dataset_dict = dict(dataset)
            X, y = dataset_dict["X"], dataset_dict["pos"]
            X, y = np.array(X), np.array(y)
            X = self.data_normalization(X)

        print("data_loader {} loaded".format(self.data_path))
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

class GSPMeasurementDataLoader(BaseDataLoader):
    """
    Tongue data loading demo using BaseDataLoader
    """
    def __init__(self, data_path, batch_size, shuffle=True, validation_split=0.2, num_workers=1, normalization="standard"):
        self.dataset = GSPMeasurementDataset(data_path, normalization)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        print("Dataset size: ", len(self.dataset))