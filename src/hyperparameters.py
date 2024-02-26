'''
Fancy class to hold data
'''

from dataclasses import dataclass

@dataclass
class Config:

    ## Training parameters
    model_type: bool = 'PSF' #. . . . . Whether to use PSF, SSF or FFSF
    n_epochs: int = 10**3 #. . . . . . . Number of epochs of training
    batch_size: int = 32 # . . . . . . . Amount of samples in each batch

    lr: float = 0.01 # . . . . . . . . . adam: learning rate
    b1: float = 0.75 # . . . . . . . . . adam: decay of first order momentum of gradient
    b2: float = 0.90 # . . . . . . . . . adam: decay of first order momentum of gradient

    log_images: int =  1 # . . . . . . . Number of images to log


    ## Data loading
    #. . . . . . . . . . . . . . . . . . Datasets file names
    train_file_name = "training.csv"
    test_file_name  = "testing.csv"
    val_file_name   = "validating.csv"
    data_dim: int =  1 # . . . . . . . . Dimension of one sample
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
    num_samples: int = 10**4 # . . . . . Number of samples to generate (if any)
    lookback: int  = 9 # . . . . . . . . Length of the input sequences
    privileged_lookback: int = 100 # . . Length of the privileged lookback


    ## Network parameters
    hidden_dim: int = 20 # . . . . . . . Dimension of the hidden layers for the lstm
    num_layers: int = 1 #. . . . . . . . Number of layers for the generator


    ## Testing phase
    alpha: float = 0.1 # . . . . . . . . Parameter for the Anomaly Detector
    h: float = 10 #. . . . . . . . . . . Parameter for the Anomaly Detector
    plot_horizon: int = 100 #. . . . . . How many samples to plot when testing