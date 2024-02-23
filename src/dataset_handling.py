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
            - `privileged_lookback`: lenght of the sequence from which to compute the summary statistics
            - `verbose`: if to print informations
        '''
        super().__init__()

        xy = np.loadtxt(file_path, delimiter=",", dtype=np.float32)

        # initialize parameters
        try:
            self.data_dim: int = xy.shape[1]
        except:
            self.data_dim: int = 1
        self.lookback: int = lookback
        self.pi_lookback: int = privileged_lookback
        self.n_seq: int = int(xy.shape[0] / lookback)
        self.n_samples = xy.shape[0]

        # transform data
        scaler = MinMaxScaler(feature_range=(-1,1)) # preserves the data distribution
        xy = xy.reshape(self.n_samples, self.data_dim) # needed when data_dim == 1
        scaler.fit(xy)
        self.x = torch.from_numpy( # <- (n_samples, data_dim)
            scaler.transform(xy)
            ).type(torch.float32
            )

        self.n_samples: int = self.get_whole_stream().size()[0]

        if verbose:
            print(f"Loaded dataset with {self.n_samples} samples of dimension {self.data_dim}, resulted in {self.n_seq} sequences of length {lookback}.")

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.x[index*self.lookback:index*self.lookback+self.lookback]
        target   = self.x[index*self.lookback+1:index*self.lookback+self.lookback+1]
        # [min, max, mu, var]
        summary_statistics = torch.zeros(self.lookback, self.data_dim*4)
        for i in range(index*self.lookback, index*self.lookback+self.lookback):
            tmp  = self.x[max(0, i-self.pi_lookback):i+1]
            m    = torch.min(tmp, dim=0).values
            M    = torch.max(tmp, dim=0).values
            mean = torch.mean(tmp, dim=0)
            var  = torch.var(tmp, dim=0).nan_to_num(nan=0.0)
            summary = torch.cat((m,M,mean,var), dim=0)
            summary_statistics[i-index*self.lookback] += summary
        return sequence, target, summary_statistics

    def __len__(self) -> int:
        return self.n_seq
    
    def get_all_sequences(self) -> torch.Tensor:
        seqs = torch.zeros(self.n_seq, self.lookback, self.data_dim)
        for i in range(len(self)):
            sequence = self.x[i*self.lookback:i*self.lookback+self.lookback]
            seqs[i] += sequence
        return seqs
    
    def get_whole_stream(self) -> torch.Tensor:
        return self.get_all_sequences().reshape(-1, self.data_dim)

    def get_all_pi(self) -> torch.Tensor:
        pi = torch.zeros(self.n_seq, self.lookback, self.data_dim*4)
        for index in range(len(self)):
            # [min, max, mu, var]
            summary_statistics = torch.zeros(self.lookback, self.data_dim*4)
            for i in range(index*self.lookback, index*self.lookback+self.lookback):
                tmp  = self.x[max(0, i-self.pi_lookback):i+1]
                m    = torch.min(tmp, dim=0).values
                M    = torch.max(tmp, dim=0).values
                mean = torch.mean(tmp, dim=0)
                var  = torch.var(tmp, dim=0).nan_to_num(nan=0.0)
                summary = torch.cat((m,M,mean,var), dim=0)
                summary_statistics[i-index*self.lookback] += summary
            pi[index] += summary_statistics
        return pi


class FeedDataset(Dataset):
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
            - `privileged_lookback`: lenght of the sequence from which to compute the summary statistics
            - `verbose`: if to print informations
        '''
        super().__init__()

        xy = np.loadtxt(file_path, delimiter=",", dtype=np.float32)

        # initialize parameters
        try:
            self.data_dim: int = xy.shape[1]
        except:
            self.data_dim: int = 1
        self.lookback: int = lookback
        self.pi_lookback: int = privileged_lookback
        self.n_seq: int = xy.shape[0]-self.lookback-1
        self.n_samples = xy.shape[0]

        # transform data
        scaler = MinMaxScaler(feature_range=(-1,1)) # preserves the data distribution
        xy = xy.reshape(self.n_samples, self.data_dim) # needed when data_dim == 1
        scaler.fit(xy)
        self.x = torch.from_numpy( # <- (n_samples, data_dim)
            scaler.transform(xy)
            ).type(torch.float32
            )

        self.n_samples: int = self.get_whole_stream().size()[0]

        if verbose:
            print(f"Loaded dataset with {xy.shape[0]} samples of dimension {self.data_dim}, resulted in {self.n_seq} sequences of length {lookback}.")

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.x[index:index+self.lookback].reshape(self.lookback*self.data_dim)
        target   = self.x[index+self.lookback].reshape(self.data_dim)
        # [min, max, mu, var]
        tmp  = self.x[max(0, index-self.pi_lookback):index+1]
        m    = torch.min(tmp, dim=0).values
        M    = torch.max(tmp, dim=0).values
        mean = torch.mean(tmp, dim=0)
        var  = torch.var(tmp, dim=0).nan_to_num(nan=0.0)
        summary = torch.cat((m,M,mean,var), dim=0)
        # shape = ( sequence, data )
        summary.reshape(self.data_dim*4)
        return sequence, target, summary

    def __len__(self) -> int:
        return self.n_seq
    
    def get_all_sequences(self) -> torch.Tensor:
        seqs = torch.zeros(self.n_seq, self.lookback, self.data_dim)
        for i in range(len(self)):
            sequence = self.x[i:i+self.lookback]
            seqs[i] += sequence
        return seqs.reshape(-1, self.lookback*self.data_dim)
    
    def get_whole_stream(self) -> torch.Tensor:
        return self.x.reshape(-1)[:-(self.lookback+1)]

    def get_all_pi(self) -> torch.Tensor:
        pi = torch.zeros(self.n_seq, self.data_dim*4)
        for index in range(len(self)):
            # [min, max, mu, var]
            tmp  = self.x[max(0, index-self.pi_lookback):index+1]
            m    = torch.min(tmp, dim=0).values
            M    = torch.max(tmp, dim=0).values
            mean = torch.mean(tmp, dim=0)
            var  = torch.var(tmp, dim=0).nan_to_num(nan=0.0)
            summary = torch.cat((m,M,mean,var), dim=0)
            # shape = ( sequence, data )
            summary.reshape(self.data_dim*4)
            pi[index] += summary
        return pi


def train_test_split(X, split: float=0.7, train_file_name: str="./datasets/training.csv", test_file_name: str="./datasets/testing.csv"):
    '''
    This function takes a tensor and saves it as two different csv files according to the given split parameter.

    Arguments:
    - `X`: the array numpy containing the data, dimensions must be ( num_samples, sample_dim )
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



def refactor_dataset(dataset_path:str="./datasets/data857489168.csv",
                     new_dataset_path:str="./datasets/sensor-data.csv"):
    '''
    Refactors the dataset to be compatible with numpy.
    '''
    dataset = pd.read_csv(dataset_path,
                          sep=", ",
                          engine="python")
    
    dataset = dataset.pivot(index='SampleTime',
                            columns='SensorID',
                            values='Value')
    # TODO: rimuovi questo
    dataset = dataset['2']
    dataset.to_csv(new_dataset_path,
                   index=False) # TODO: header at false



## TESTING AREA
if __name__ == '__main__':
    from hyperparameters import Config
    import utilities as ut

    hparams = Config()
    dataset_path = "./datasets/sensor-data-2.csv"
    train_dataset_path = f"./datasets/{hparams.train_file_name}"
    val_dataset_path   = f"./datasets/{hparams.val_file_name}"
    test_dataset_path  = f"./datasets/{hparams.test_file_name}"

    train_path, val_path, test_path = ut.generate_data()
    dataset = FeedDataset(file_path=train_path,
                          lookback=hparams.lookback,
                          privileged_lookback=hparams.privileged_lookback
                          )

    print(len(dataset))

    # dataset = pd.read_csv("./datasets/sensor-data.csv")
    # dataset = dataset['2']
    # dataset.to_csv(dataset_path, index=False, header=False)


    # # ignore sensors with too many nulls
    # #

    # # Train & Test
    # train_test_split(X=np.loadtxt(dataset_path, delimiter=",", dtype=np.float32),
    #                 split=hparams.train_test_split,
    #                 train_file_name=train_dataset_path,
    #                 test_file_name=test_dataset_path    
    #                 )

    # # Train & Validation
    # train_test_split(X=np.loadtxt(train_dataset_path, delimiter=",", dtype=np.float32),
    #                 split=hparams.train_val_split,
    #                 train_file_name=train_dataset_path,
    #                 test_file_name=val_dataset_path    
    #                 )