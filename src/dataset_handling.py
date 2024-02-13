'''
This is the data module for the model
'''
from typing import Optional, Callable, Tuple
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import torch
import pandas as pd

import numpy as np
    

class RealDataset(Dataset):
    def __init__(self,
                 file_path: Path,
                 seq_len: int,
                 verbose: bool = True
                 ) -> None:
        '''
        Load the dataset from a given file containing the data stream.

        Arguments:
            - `file_path`: the path of the file containing the data stream
            - `seq_len`: length of the sequence to extract from the data stream
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
        self.seq_len: int = seq_len
        self.n_seq: int = int(self.n_samples / seq_len)

        # transform data
        scaler = MinMaxScaler(feature_range=(-1,1)) # preserves the data distribution
        xy = xy.reshape(self.n_samples, self.data_dim) # needed when data_dim == 1
        scaler.fit(xy)
        self.x = torch.from_numpy( # <- (n_samples, data_dim)
            scaler.transform(xy)
            ).type(torch.float32
            )

        if verbose:
            print(f"Loaded dataset with {self.n_samples} samples of dimension {self.data_dim}, resulted in {self.n_seq} sequences of length {seq_len}.")

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.x[index:index+self.seq_len]
        target = self.x[index+1:index+self.seq_len+1]
        return sequence, target

    def __len__(self) -> int:
        return self.n_seq
    
    def get_all_sequences(self) -> torch.Tensor:
        return self.x
    
    def get_whole_stream(self) -> torch.Tensor:
        return self.x.reshape(self.n_samples, self.data_dim)
    


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

