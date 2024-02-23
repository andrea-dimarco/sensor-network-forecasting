
import torch
import wandb
import numpy as np
import dataset_handling as dh
import matplotlib.pyplot as plt
from privileged_model import PSF
from forecasting_model import SSF
from feedforward_net import FFSF
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
    if hparams.model_type == 'FFSF':
        model = FFSF(hparams=hparams,
                        train_file_path=train_dataset_path,
                        val_file_path=val_dataset_path,
                        plot_losses=False
                        )
    elif hparams.model_type == 'PSF':
        model = PSF(hparams=hparams,
                        train_file_path=train_dataset_path,
                        val_file_path=val_dataset_path,
                        plot_losses=False
                        )
    elif hparams.model_type == 'SSF':
        model = SSF(hparams=hparams,
                        train_file_path=train_dataset_path,
                        val_file_path=val_dataset_path,
                        plot_losses=False
                        )
        
    wandb_logger = WandbLogger(project=f"{hparams.model_type} PyTorch (2024)", log_model=True)       
    wandb_logger.experiment.watch(model, log='all', log_freq=500)

    # Define the trainer
    trainer = Trainer(logger=wandb_logger,
                    max_epochs=hparams.n_epochs
                    )

    # Start the training
    trainer.fit(model)

    # Log the trained model
    torch.save(model.state_dict(), f"./{hparams.model_type}-model.pth")

    # Validate the model
    with torch.no_grad():
        model.eval()
        model.cpu()
        if hparams.model_type == 'FFSF':
            dataset = dh.FeedDataset(file_path=train_dataset_path,
                                        lookback=hparams.lookback,
                                        privileged_lookback=hparams.privileged_lookback
                                        )
        elif hparams.model_type == 'PSF':
            dataset = dh.PrivilegedDataset(file_path=train_dataset_path,
                                        lookback=hparams.lookback,
                                        privileged_lookback=hparams.privileged_lookback
                                        )
        elif hparams.model_type == 'SSF':
            dataset = dh.RealDataset(file_path=train_dataset_path,
                                    lookback=hparams.lookback
                                    )
        else:
            raise ValueError
        
        print("Loaded testing dataset.")
        synth_plot = np.zeros_like(dataset.get_whole_stream()) * np.nan

        if hparams.model_type == 'PSF':
            y_pred = model(dataset.get_all_sequences(), dataset.get_all_pi()
                        ).reshape(-1,hparams.data_dim)
            
        elif hparams.model_type == 'SSF':
            y_pred = model(dataset.get_all_sequences()
                        ).reshape(-1,hparams.data_dim)

        elif hparams.model_type == "FFSF":
            y_pred = model(dataset.get_all_sequences(), dataset.get_all_pi()
                        ).reshape(-1)


        if hparams.model_type in ['SSF', 'PSF']:
            synth_plot[hparams.lookback:] = y_pred[hparams.lookback:]
        elif hparams.model_type == 'FFSF':
            synth_plot = y_pred

        print("Predictions done.")
        # only plot the first dimension
        horizon = min(hparams.plot_horizon, dataset.n_samples)
        if hparams.model_type in ['PSF', 'SSF']:
            plt.plot(dataset.get_whole_stream()[:horizon,0])
            plt.plot(synth_plot[:horizon,0], c='r')
        else:
            plt.plot(dataset.get_all_targets()[:horizon])
            plt.plot(synth_plot[:horizon], c='r')

        print("Plot done.")
        plt.savefig(f"{hparams.model_type}-forecasting-plot.png")
        plt.show()


### Testing Area
if __name__ == '__main__':
    import utilities as ut
    ut.set_seed(69)
    ut.generate_data()
    train()


