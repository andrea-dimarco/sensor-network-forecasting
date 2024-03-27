'''
Fancy class to hold data
'''
from dataclasses import dataclass

@dataclass
class Config:

    ## Training parameters
    model_type: bool = 'SSF' # . . . . . Which model to use: PSF, SSF or FFSF
    load_model: bool = False #. . . . . . Whether to load the model or train a new one
    n_epochs: int = 15001 #. . . . . . . Number of epochs of training
    val_frequency: int = 50 #. . . . . . How often to perform a validation step
    seed: int = 11 # . . . . . . . . . . Global Seed
    batch_size: int = 128 #. . . . . . . Amount of samples in each batch

    lr: float = 0.01 # . . . . . . . . . adam: learning rate
    b1: float = 0.75 # . . . . . . . . . adam: decay of first order momentum of gradient
    b2: float = 0.90 # . . . . . . . . . adam: decay of first order momentum of gradient

    decay_start: float = 1.0 # . . . . . Starting decay factor for the schedulers
    decay_end: float   = 0.90 #. . . . . Ending decay factor for the scheduler
    
    patience: int = 10**3 #. . . . . . . Patience for the early stopping callback
    log_images: int =  1 # . . . . . . . Number of images to log


    ## Data loading
    #. . . . . . . . . . . . . . . . . . Datasets file names
    n_sensors: int = 600 # . . . . . . . Number of sensors to consider when creating csv
    cluster_selected: int = 11 #. . . . . Cluster to select for training
    clustering_threshold: float = 0.321 # Threshold for the hierarchical clustering 
    train_file_name = "training.csv"
    test_file_name  = "testing.csv"
    val_file_name   = "validating.csv"
    dataset_name: str = 'real' # . . . . Which dataset to use
                               # . . . . . . real: gets the samples from csv files
                               # . . . . . . sine: runs independent sine processes wih random phases
                               # . . . . . . iid: samples iids from a multivariate
                               # . . . . . . cov: samples iids from a multivariate
                               # . . . . . . . . . with random covariance matrix
                               # . . . . . . wien: runs a number or wiener processes 
                               # . . . . . . . . . with random mutual correlations
    train_test_split: float = 0.8 #. . . Split between training and testing samples
    train_val_split: float = 0.8 # . . . Split between training and validating samples
    data_dim: int =  1 # . . . . . . . . Dimension of one sample
    num_samples: int = 10**4 # . . . . . Number of samples to generate (if any)


    ## Network parameters
    hidden_dim: int = 21 #. . . . . . . Dimension of the hidden layers for the lstm
    num_layers: int = 1 #. . . . . . . . Number of layers for the generator
    lookback: int = 200 #. . . . . . . . Length of the input sequences
    privileged_lookback: int = 5 # . . . Length of the privileged lookback


    ## Testing phase
    plot_horizon: int = 1000 # . . . . . How many samples to plot when testing
    alpha: float = 0.1 # . . . . . . . . Parameter for the Anomaly Detector
    h: float = 10 #. . . . . . . . . . . Parameter for the Anomaly Detector