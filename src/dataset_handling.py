'''
This is the data module for the model
'''
from typing import Tuple
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import torch
import pandas as pd

import numpy as np
    

class RealDataset(Dataset):
    def __init__(self,
                 file_path: Path,
                 lookback: int,
                 verbose: bool = True
                 ) -> None:
        '''
        Load the dataset from a given file containing the data stream.

        Arguments:
            - `file_path`: the path of the file containing the data stream
            - `lookback`: length of the sequence to extract from the data stream
            - `transform`: optional transformation to be done on the data
        '''
        super().__init__()

        xy = np.loadtxt(file_path, delimiter=",", dtype=np.float32)

        # initialize parameters
        self.n_samples: int = xy.shape[0]
        try:
            self.data_dim: int = xy.shape[1]
        except:
            self.data_dim: int = 1
        self.lookback: int = lookback
        self.n_seq: int = int(self.n_samples / lookback)

        # transform data
        scaler = MinMaxScaler(feature_range=(-1,1)) # preserves the data distribution
        xy = xy.reshape(self.n_samples, self.data_dim) # needed when data_dim == 1
        scaler.fit(xy)
        self.x = torch.from_numpy( # <- (n_samples, data_dim)
            scaler.transform(xy)
            ).type(torch.float32
            )

        if verbose:
            print(f"Loaded dataset with {self.n_samples} samples of dimension {self.data_dim}, resulted in {self.n_seq} sequences of length {lookback}.")

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.x[index:index+self.lookback]
        target = self.x[index+1:index+self.lookback+1]
        return sequence, target

    def __len__(self) -> int:
        return self.n_seq
    
    def get_all_sequences(self) -> torch.Tensor:
        return self.x
    
    def get_whole_stream(self) -> torch.Tensor:
        return self.x.reshape(self.n_samples, self.data_dim)
    

class PrivilegedDataset(Dataset):
    def __init__(self,
                 file_path: Path,
                 lookback:int,
                 privileged_lookback:int,
                 verbose: bool = True
                 ) -> None:
        '''
        Load the dataset from a given file containing the data stream.

        Arguments:
            - `file_path`: the path of the file containing the data stream
            - `lookback`: length of the sequence to extract from the data stream
        '''
        super().__init__()

        xy = np.loadtxt(file_path, delimiter=",", dtype=np.float32)

        # initialize parameters
        self.n_samples: int = xy.shape[0]
        try:
            self.data_dim: int = xy.shape[1]
        except:
            self.data_dim: int = 1
        self.lookback: int = lookback
        self.pi_lookback: int = privileged_lookback
        self.n_seq: int = int(self.n_samples / lookback)

        # transform data
        scaler = MinMaxScaler(feature_range=(-1,1)) # preserves the data distribution
        xy = xy.reshape(self.n_samples, self.data_dim) # needed when data_dim == 1
        scaler.fit(xy)
        self.x = torch.from_numpy( # <- (n_samples, data_dim)
            scaler.transform(xy)
            ).type(torch.float32
            )

        if verbose:
            print(f"Loaded dataset with {self.n_samples} samples of dimension {self.data_dim}, resulted in {self.n_seq} sequences of length {lookback}.")

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.x[index:index+self.lookback]
        target   = self.x[index+1:index+self.lookback+1]
        # [min, max, mu, var]
        summary_statistics = torch.zeros(self.lookback, self.data_dim*4)
        for i in range(index, index+self.lookback):
            tmp  = self.x[max(0, i-self.pi_lookback):i+1]
            m    = torch.min(tmp, dim=0).values
            M    = torch.max(tmp, dim=0).values
            mean = torch.mean(tmp, dim=0)
            var  = torch.var(tmp, dim=0).nan_to_num(nan=0.0)
            summary = torch.cat((m,M,mean,var), dim=0)
            summary_statistics[i-index] += summary
        return sequence, target, summary_statistics

    def __len__(self) -> int:
        return self.n_seq
    
    def get_all_sequences(self) -> torch.Tensor:
        return self.x
    
    def get_whole_stream(self) -> torch.Tensor:
        return self.x.reshape(self.n_samples, self.data_dim)

    def get_all_pi(self) -> torch.Tensor:
        pi = torch.zeros(self.n_seq, self.lookback, self.data_dim*4)
        for index in range(len(self)):
            summary_statistics = torch.zeros(self.lookback, self.data_dim*4)
            for i in range(index, index+self.lookback):
                tmp  = self.x[max(0, i-self.pi_lookback):i+1]
                m    = torch.min(tmp, dim=0).values
                M    = torch.max(tmp, dim=0).values
                mean = torch.mean(tmp, dim=0)
                var  = torch.var(tmp, dim=0).nan_to_num(nan=0.0)
                summary = torch.cat((m,M,mean,var), dim=0)
                summary_statistics[i-index] += summary
            pi[index] += summary_statistics
        return pi

def train_test_split(X, split: float=0.7, train_file_name: str="./datasets/training.csv", test_file_name: str="./datasets/testing.csv"):
    '''
    This function takes a tensor and saves it as two different csv files according to the given split parameter.

    Arguments:
    - `X`: the tensor containing the data, dimensions must be ( num_samples, sample_dim )
    - `split`: the perchentage of samples to keep for training
    - `train_file_name`: name of the .csv file that will contain the training set
    - `test_file_name`: name of the .csv file that will contain the testing set
    '''
    assert(split > 0 and split < 1)
    delimiter = int( X.shape[0] * split )

    # Train
    df = pd.DataFrame(X[:delimiter])
    df.to_csv(train_file_name, index=False, header=False)

    # Test
    df = pd.DataFrame(X[delimiter:])
    df.to_csv(test_file_name, index=False, header=False)

