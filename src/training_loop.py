
from forecasting_model import SSF
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from torch import cuda
from pytorch_lightning import Trainer
import wandb
from hyperparameters import Config
import numpy as np
import torch
import matplotlib.pyplot as plt
from data_generation import wiener_process
from dataset_handling import train_test_split, RealDataset
from numpy import loadtxt, float32

import warnings
warnings.filterwarnings("ignore")


def generate_data(datasets_folder="./datasets/"):
    hparams = Config()

    # Generate and store the dataset as requested
    dataset_path = f"{datasets_folder}{hparams.dataset_name}_generated_stream.csv"
    wiener_process.save_wiener_process(p=hparams.data_dim, N=hparams.num_samples, file_path=dataset_path)
    print(f"The {hparams.dataset_name} dataset has been succesfully created and stored into:\n\t- {dataset_path}")
    
    train_dataset_path = f"{datasets_folder}{hparams.dataset_name}_training.csv"
    val_dataset_path   = f"{datasets_folder}{hparams.dataset_name}_testing.csv"

    train_test_split(X=loadtxt(dataset_path, delimiter=",", dtype=float32),
                    split=hparams.train_test_split,
                    train_file_name=train_dataset_path,
                    test_file_name=val_dataset_path    
                    )
    print(f"The {hparams.dataset_name} dataset has been split successfully into:\n\t- {train_dataset_path}\n\t- {val_dataset_path}")


def train(datasets_folder="./datasets/"):

    torch.multiprocessing.set_sharing_strategy('file_system')

    # Parameters
    hparams = Config()

    if hparams.dataset_name in ['sine', 'wien', 'iid', 'cov']:
        train_dataset_path = f"{datasets_folder}{hparams.dataset_name}_training.csv"
        val_dataset_path  = f"{datasets_folder}{hparams.dataset_name}_testing.csv"

    elif hparams.dataset_name == 'real':
        train_dataset_path = hparams.train_file_path
        val_dataset_path  = hparams.test_file_path
    else:
        raise ValueError("Dataset not supported.")

    # Instantiate the model
    timegan = SSF(hparams=hparams,
                    train_file_path=train_dataset_path,
                    val_file_path=val_dataset_path,
                    plot_losses=False
                    )

    # Define the logger -> https://www.wandb.com/articles/pytorch-lightning-with-weights-biases.
    wandb_logger = WandbLogger(project="SSF PyTorch (2024)", log_model=True)

    wandb_logger.experiment.watch(timegan, log='all', log_freq=500)

    # Define the trainer
    trainer = Trainer(logger=wandb_logger,
                    max_epochs=hparams.n_epochs,
                    val_check_interval=1.0
                    )

    # Start the training
    trainer.fit(timegan)

    # Log the trained model
    trainer.save_checkpoint('SSF-checkpoint.pth')
    wandb.save('SSF-wandb.pth')
    with torch.no_grad():
        timegan.eval()
        dataset = RealDataset(file_path=train_dataset_path,seq_len=hparams.seq_len)
        train_plot = np.ones_like(dataset.get_whole_stream()[:,0]) * np.nan
        y_pred = timegan(dataset.get_all_sequences())[:-hparams.seq_len,0]
        #y_pred = y_pred[:, -1, :]
        train_plot[hparams.seq_len:dataset.get_whole_stream().size()[0]] = y_pred
        plt.plot(dataset.get_whole_stream()[:,0])
        plt.plot(train_plot, c='r')
        plt.show()


import random
import pytorch_lightning as pl
def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True # Can have performance impact
    torch.backends.cudnn.benchmark = False

    _ = pl.seed_everything(seed)


### Testing Area
datasets_folder = "./datasets/"
set_seed(69)
generate_data(datasets_folder)
train(datasets_folder)


