
import torch
import wandb
import numpy as np
import dataset_handling as dh
import matplotlib.pyplot as plt
from privileged_model import PSF
from forecasting_model import SSF
from hyperparameters import Config
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger

import warnings
warnings.filterwarnings("ignore")


def train(datasets_folder="./datasets/"):

    torch.multiprocessing.set_sharing_strategy('file_system')

    # Parameters
    hparams = Config()

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

    # Instantiate the model
    if hparams.use_pi:
        model = PSF(hparams=hparams,
                        train_file_path=train_dataset_path,
                        val_file_path=val_dataset_path,
                        plot_losses=False
                        )
        wandb_logger = WandbLogger(project="PSF PyTorch (2024)", log_model=True)
    else:
        model = SSF(hparams=hparams,
                        train_file_path=train_dataset_path,
                        val_file_path=val_dataset_path,
                        plot_losses=False
                        )
        wandb_logger = WandbLogger(project="SSF PyTorch (2024)", log_model=True)    

    wandb_logger.experiment.watch(model, log='all', log_freq=500)

    # Define the trainer
    trainer = Trainer(logger=wandb_logger,
                    max_epochs=hparams.n_epochs
                    )

    # Start the training
    trainer.fit(model)

    # Log the trained model
    if hparams.use_pi:
        torch.save(model.state_dict(), "./pi-model.pth")
    else:
        torch.save(model.state_dict(), "./model.pth")

    # Validate the model
    with torch.no_grad():
        model.eval()
        model.cpu()
        dataset = dh.PrivilegedDataset(file_path=train_dataset_path,
                                       lookback=hparams.lookback,
                                       privileged_lookback=hparams.privileged_lookback
                                       )
        print("Loaded real testing dataset.")
        synth_plot = np.ones_like(dataset.get_whole_stream()) * np.nan
        if hparams.use_pi:
            y_pred = model(dataset.get_all_sequences(), dataset.get_all_pi()
                        ).reshape(-1,hparams.data_dim)[hparams.lookback:]
        else:
            y_pred = model(dataset.get_all_sequences()
                        ).reshape(-1,hparams.data_dim)[hparams.lookback:]
        synth_plot[hparams.lookback:dataset.n_samples] = y_pred
        print("Predictions done.")
        # only plot the first dimension
        horizon = min(hparams.plot_horizon, dataset.n_samples)
        plt.plot(dataset.get_whole_stream()[:horizon,0])
        plt.plot(synth_plot[:horizon,0], c='r')

        print("Plot done.")
        plt.savefig("forecasting-plot.png")
        plt.show()


### Testing Area
if __name__ == '__main__':
    import utilities as ut
    ut.set_seed(69)
    ut.generate_data()
    train()


