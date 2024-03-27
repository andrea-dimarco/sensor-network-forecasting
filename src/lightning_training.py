
import torch
import wandb
import numpy as np
import dataset_handling as dh
import matplotlib.pyplot as plt
from lightning_modules.psf import PSF
from lightning_modules.ssf import SSF
from lightning_modules.ffsf import FFSF
from hyperparameters import Config
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import warnings
warnings.filterwarnings("ignore")


def train(datasets_folder="./datasets/", hparams:Config=Config()):
    '''
    Allocate and Train the model.
    '''
    torch.multiprocessing.set_sharing_strategy('file_system')

    if hparams.dataset_name in ['sine', 'wien', 'iid', 'cov']:
        train_dataset_path = f"{datasets_folder}{hparams.dataset_name}_training.csv"
        val_dataset_path   = f"{datasets_folder}{hparams.dataset_name}_validating.csv"
        test_dataset_path  = f"{datasets_folder}{hparams.dataset_name}_testing.csv"

    elif hparams.dataset_name == 'real':
        train_dataset_path = str(datasets_folder) + hparams.train_file_name
        val_dataset_path  = str(datasets_folder) + hparams.val_file_name
        test_dataset_path = str(datasets_folder) + hparams.test_file_name
    else:
        raise ValueError("Dataset not supported.")
    

    # get model dimension
    data_dim = dh.RealDataset(file_path=train_dataset_path, lookback=hparams.lookback).data_dim

    # Instantiate the model
    if hparams.model_type == 'FFSF':
        model = FFSF(hparams=hparams,
                        train_file_path=train_dataset_path,
                        val_file_path=test_dataset_path,
                        data_dim=data_dim,
                        plot_losses=False
                        )
    elif hparams.model_type == 'PSF':
        model = PSF(hparams=hparams,
                        train_file_path=train_dataset_path,
                        val_file_path=test_dataset_path,
                        data_dim=data_dim,
                        plot_losses=False
                        )
    elif hparams.model_type == 'SSF':
        model = SSF(hparams=hparams,
                        train_file_path=train_dataset_path,
                        val_file_path=test_dataset_path,
                        data_dim=data_dim,
                        plot_losses=False
                        )
        
    wandb_logger = WandbLogger(project=f"{hparams.model_type} PyTorch (2024)", log_model=True)       
    wandb_logger.experiment.watch(model, log='all', log_freq=500)

    # Early stopper
    early_stop_callback = EarlyStopping(monitor="val_loss", 
                                        patience=hparams.patience,
                                        mode="min",
                                        verbose=False,
                                        strict=False
                                        )

    # Define the trainer
    trainer = Trainer(logger=wandb_logger,
                    max_epochs=hparams.n_epochs,
                    callbacks=[early_stop_callback]
                    )

    # Start the training
    trainer.fit(model)

    # Log the trained model
    torch.save(model.state_dict(), f"./{hparams.model_type}-model.pth")

    # Validate the model
    ut.validate_model(model, train_dataset_path, val_dataset_path)#test_dataset_path)



### Testing Area
# DEPRECATED!!
if __name__ == '__main__':
    import utilities as ut
    ut.set_seed(Config().seed)
    ut.generate_data()
    train()