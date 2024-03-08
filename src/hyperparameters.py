'''
Fancy class to hold data
'''

from dataclasses import dataclass

@dataclass
class Config:

    ## Training parameters
    model_type: bool = 'FFSF' # . . . . . Whether to use PSF, SSF or FFSF
    n_epochs: int = 10**1 #. . . . . . . Number of epochs of training
    seed: int = 42 # . . . . . . . . . . Global Seed
    batch_size: int = 32 # . . . . . . . Amount of samples in each batch

    lr: float = 0.01 # . . . . . . . . . adam: learning rate
    b1: float = 0.75 # . . . . . . . . . adam: decay of first order momentum of gradient
    b2: float = 0.90 # . . . . . . . . . adam: decay of first order momentum of gradient

    decay_start: float = 1.0 # . . . . . Starting decay factor for the schedulers
    decay_end: float   = 0.1 # . . . . . Ending decay factor for the scheduler
    
    patience: int = 10**4 #. . . . . . . Patience for the early stopping callback

    log_images: int =  1 # . . . . . . . Number of images to log


    ## Data loading
    #. . . . . . . . . . . . . . . . . . Datasets file names
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
    train_test_split: float = 0.7 #. . . Split between training and testing samples
    train_val_split: float = 0.8 # . . . Split between training and validating samples
    data_dim: int =  1 # . . . . . . . . Dimension of one sample
    num_samples: int = 10**4 # . . . . . Number of samples to generate (if any)


    ## Network parameters
    hidden_dim: int = 20 # . . . . . . . Dimension of the hidden layers for the lstm
    num_layers: int = 1 #. . . . . . . . Number of layers for the generator
    lookback: int = 20 # . . . . . . . . Length of the input sequences
    privileged_lookback: int = 100 # . . Length of the privileged lookback


    ## Testing phase
    plot_horizon: int = 2000 #. . . . . How many samples to plot when testing
    alpha: float = 0.1 # . . . . . . . . Parameter for the Anomaly Detector
    h: float = 10 #. . . . . . . . . . . Parameter for the Anomaly Detector