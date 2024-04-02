'''
This is the data module for the model
'''
from typing import Tuple
from torch.utils.data import Dataset
from pathlib import Path
import torch
import pandas as pd
from hyperparameters import Config
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as spc
from scipy.spatial.distance import squareform
    

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
        xy = xy.reshape(self.n_samples, self.data_dim) # needed when data_dim == 1
        self.x = torch.from_numpy( # <- (n_samples, data_dim)
            xy
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
        xy = xy.reshape(self.n_samples, self.data_dim) # needed when data_dim == 1
        self.x = torch.from_numpy( # <- (n_samples, data_dim)
            xy
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
        xy = xy.reshape(self.n_samples, self.data_dim) # needed when data_dim == 1
        self.x = torch.from_numpy( # <- (n_samples, data_dim)
            xy).type(torch.float32)

        self.n_samples: int = self.get_all_targets().size()[0]

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
    
    def get_all_targets(self) -> torch.Tensor:
        targs = torch.zeros(self.n_seq, self.data_dim)
        for i in range(len(self)):
            target = self.x[i+self.lookback].reshape(self.data_dim)
            targs[i] += target
        return targs.reshape(-1, self.data_dim)

    def get_all_sequences(self) -> torch.Tensor:
        seqs = torch.zeros(self.n_seq, self.lookback, self.data_dim)
        for i in range(len(self)):
            sequence = self.x[i:i+self.lookback]
            seqs[i] += sequence
        return seqs.reshape(-1, self.lookback*self.data_dim)
    
    def get_whole_stream(self) -> torch.Tensor:
        return self.x.reshape(-1,self.data_dim)[:-(self.lookback+1)]

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


def train_test_split(X,
                     split:float=0.7,
                     train_file_name:str="./datasets/training.csv",
                     test_file_name: str="./datasets/testing.csv"
                     ):
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

    Arguments:
        - `dataset_path`: the path to the original dataset
        - `new_dataset_path`: where to save the refactored csv file
    '''
    dataset = pd.read_csv(dataset_path)
    
    dataset = dataset.pivot(index='SampleTime',
                            columns='SensorID',
                            values='Value')
    # TODO: rimuovi questo
    #dataset = dataset['2']
    dataset.to_csv(new_dataset_path,
                   index=False) # TODO: header at false


def clean_dataset(null_threshold=1,dataset_path:str="./datasets/sensor-data.csv",
                     new_dataset_path:str="./datasets/sensor_data_cleaned.csv"):
    
    df = pd.read_csv(dataset_path)

    # Drop all columns with too many NULL
    col_count = 0
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count >= null_threshold:
            df = df.drop(columns=col)
            col_count += 1
    print("Numbers of cols dropped",col_count)

    #probably unneeded
    # drop column if it starts with some nulls and has nulls after some real values
    if null_threshold > 1:
        col_count = 0
        def get_first_non_null(column):
            for index,value in enumerate(column):
                if pd.notna(value):
                    return index
        for col in df.columns:
            index = get_first_non_null(col)
            to_delete = df[col].iloc[index:].isnull().any()
            if to_delete:
                print("Nulls after the start",df[col].iloc[index:].isnull().sum())
                print("Deleting column",col)
                df = df.drop(columns=col)
                col_count += 1
        print("Columns dropped",col_count)
    df.to_csv(new_dataset_path,
                   index=False) # TODO: header at false


def select_sensor(sensor=2, hparams:Config=Config(), do_validation:bool=False):
    '''
    Saves a csv with only the realizations of the chosen sensor.

    Arguments:
        - `sensor`: the sensor to isolate
        - `do_validation`: if to further split the training set into a validation set
        - `hparams`: hyperparameters
    '''
    dataset_path = f"./datasets/sensor_data_{sensor}.csv"
    train_dataset_path = f"./datasets/{hparams.train_file_name}"
    val_dataset_path   = f"./datasets/{hparams.val_file_name}"
    test_dataset_path  = f"./datasets/{hparams.test_file_name}"

    dataset = pd.read_csv("./datasets/sensor_data_cleaned.csv")
    try:
        dataset = dataset[str(sensor)]
    except:
        raise ValueError
    dataset.to_csv(dataset_path, index=False, header=False)

    # Train & Test
    train_test_split(X=np.loadtxt(dataset_path, delimiter=",", dtype=np.float32),
                     split=hparams.train_test_split,
                     train_file_name=train_dataset_path,
                     test_file_name=test_dataset_path    
                     )

    # Train & Validation
    if do_validation:
        train_test_split(X=np.loadtxt(train_dataset_path, delimiter=",", dtype=np.float32),
                        split=hparams.train_val_split,
                        train_file_name=train_dataset_path,
                        test_file_name=val_dataset_path    
                        )
        print(f"Sensor {sensor} data saved in files:\n\t- {train_dataset_path}\n\t- {val_dataset_path}\n\t- {test_dataset_path}")
    else:
        print(f"Sensor {sensor} data saved in files:\n\t- {train_dataset_path}\n\t- {test_dataset_path}")


def select_sensors(sensors=[2,992], hparams:Config=Config(), do_validation:bool=False):
    '''
    Saves a csv with only the realizations of the multiple chosen sensors.

    Arguments:
        - `sensors`: the sensors to isolate, if it's None it will get all the sensors
        - `do_validation`: if to further split the training set into a validation set
        - `hparams`: hyperparameters
    '''
    dataset_path = "./datasets/sensor_data_multi.csv"
    train_dataset_path = f"./datasets/{hparams.train_file_name}"
    val_dataset_path   = f"./datasets/{hparams.val_file_name}"
    test_dataset_path  = f"./datasets/{hparams.test_file_name}"

    dataset = pd.read_csv("./datasets/sensor_data_cleaned.csv")

    try:
        if sensors != None:
            sensor_list = [str(i) for i in sensors]
            dataset = dataset[sensor_list]
    except:
        raise ValueError
    dataset.to_csv(dataset_path, index=False, header=False)

    # Train & Test
    train_test_split(X=np.loadtxt(dataset_path, delimiter=",", dtype=np.float32),
                     split=hparams.train_test_split,
                     train_file_name=train_dataset_path,
                     test_file_name=test_dataset_path    
                     )

    # Train & Validation
    if do_validation:
        train_test_split(X=np.loadtxt(train_dataset_path, delimiter=",", dtype=np.float32),
                        split=hparams.train_val_split,
                        train_file_name=train_dataset_path,
                        test_file_name=val_dataset_path    
                        )
        print(f"Sensors {sensors} data saved in files:\n\t- {train_dataset_path}\n\t- {val_dataset_path}\n\t- {test_dataset_path}")
    else:
        print(f"Sensors {sensors} data saved in files:\n\t- {train_dataset_path}\n\t- {test_dataset_path}")


def select_sensors_diff(sensors=[2,992], hparams:Config=Config(), do_validation:bool=False):
    '''
    Saves a csv with only the realizations of the multiple chosen sensors.

    Arguments:
        - `sensors`: the sensors to isolate, if it's None it will get all the sensors
        - `do_validation`: if to further split the training set into a validation set
        - `hparams`: hyperparameters
    '''
    print("ATTENTION: Using differential dataset.")
    dataset_path = "./datasets/sensor_data_multi.csv"
    train_dataset_path = f"./datasets/{hparams.train_file_name}"
    val_dataset_path   = f"./datasets/{hparams.val_file_name}"
    test_dataset_path  = f"./datasets/{hparams.test_file_name}"

    dataset = pd.read_csv("./datasets/sensor_data_cleaned.csv").diff().iloc[1:]

    try:
        if sensors != None:
            sensor_list = [str(i) for i in sensors]
            dataset = dataset[sensor_list]
    except:
        raise ValueError
    dataset.to_csv(dataset_path, index=False, header=False)

    # Train & Test
    train_test_split(X=np.loadtxt(dataset_path, delimiter=",", dtype=np.float32),
                     split=hparams.train_test_split,
                     train_file_name=train_dataset_path,
                     test_file_name=test_dataset_path    
                     )

    # Train & Validation
    if do_validation:
        train_test_split(X=np.loadtxt(train_dataset_path, delimiter=",", dtype=np.float32),
                        split=hparams.train_val_split,
                        train_file_name=train_dataset_path,
                        test_file_name=val_dataset_path    
                        )
        print(f"Sensors {sensors} data saved in files:\n\t- {train_dataset_path}\n\t- {val_dataset_path}\n\t- {test_dataset_path}")
    else:
        print(f"Sensors {sensors} data saved in files:\n\t- {train_dataset_path}\n\t- {test_dataset_path}")


def corr_heatmap(correlation,
                 save_pic:bool=True,
                 show_pic:bool=True,
                 pic_name:str="correlation-heatmap"
                 ) -> None:
    '''
    Saves a picture of the correlation matrix as a heatmap.
    '''
    plt.figure()
    sns.heatmap(correlation,
                cmap='RdBu',
                annot=False,
                vmin=-1,
                vmax=1
                )
    if save_pic:
        plt.savefig(f"./img/{pic_name}.png",dpi=300)
    if show_pic:
        plt.show()


def cluster_sensors(correlation,
                    threshold:float=1.0,
                    show_pic:bool=True,
                    save_pic:bool=True
                    ) -> np.ndarray:
    '''
    Given the correlation matrix, assigns the sensors to clusters and returns the labels.
    '''
    if show_pic or save_pic:
        plt.figure()
    pdist = spc.distance.pdist(correlation)
    Z = spc.linkage(pdist, method='complete')

    spc.dendrogram(Z,
               no_labels=True,
               orientation='top',
               leaf_rotation=90,
               color_threshold=threshold*pdist.max()
               )
    if save_pic:
        plt.savefig("./img/dendrogram.png", dpi=300)
    if show_pic:
        plt.show()
    
    # Clusterize the data
    labels = spc.fcluster(Z, threshold*pdist.max(), criterion='distance')

    # Show the cluster
    return labels


## TESTING AREA
if __name__ == '__main__':
    hparams = Config()
    # Refactor original sesor dataset
    # refactor_dataset()
    # clean_dataset()


    # Check sensor covariance
    df = pd.read_csv("./datasets/sensor_data_cleaned.csv")
    n_sensors = min(hparams.n_sensors, 526)


    # Check sensor correlation
    correlation = df.iloc[:,:n_sensors].corr()
    #corr_heatmap(correlation)
    

    # Cluster the sensors
    labels = cluster_sensors(correlation,
                             threshold=hparams.clustering_threshold,
                             show_pic=False
                             )
    n_clusters = labels.max()
    print(f"The clusters of the first {n_sensors} sensors are:", n_clusters)

    clusters = dict() 
    for sensor_idx in range(len(labels)):
        cluster = labels[sensor_idx]
        sensor_id = int(correlation.columns[sensor_idx])
        if cluster in clusters.keys():
            clusters[cluster].append(sensor_id)
        else:
            clusters[cluster] = [sensor_id]

    cluster = min(hparams.cluster_selected, n_clusters)
    for i in range(1,n_clusters+1):
        print(i,":", clusters[i])
    select_sensors_diff(sensors=[81])#clusters[cluster])

    # Save funky pictures of the clusters  
    # for i in range(1,n_clusters+1):
    #     new_corr = df[[str(i) for i in clusters[i]]].corr()
    #     corr_heatmap(new_corr, pic_name=f"correlation-cluster-{i}")