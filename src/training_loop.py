
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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import warnings
warnings.filterwarnings("ignore")


def train(datasets_folder="./datasets/", hparams:Config=Config()):

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
                        val_file_path=val_dataset_path,
                        data_dim=data_dim,
                        plot_losses=False
                        )
    elif hparams.model_type == 'PSF':
        model = PSF(hparams=hparams,
                        train_file_path=train_dataset_path,
                        val_file_path=val_dataset_path,
                        data_dim=data_dim,
                        plot_losses=False
                        )
    elif hparams.model_type == 'SSF':
        model = SSF(hparams=hparams,
                        train_file_path=train_dataset_path,
                        val_file_path=val_dataset_path,
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
    validate_model(model, train_dataset_path, val_dataset_path)#test_dataset_path)


def validate_model(model:SSF|PSF|FFSF, train_dataset_path:str, test_dataset_path:str, hparams:Config=Config()) -> None:
    '''
    Plots a graph with the predictions on the training set and on the test set.
    '''
    with torch.no_grad():
        model.eval()
        model.cpu()

        # On the TRAINING set
        if hparams.model_type == 'FFSF':
            dataset_train = dh.FeedDataset(file_path=train_dataset_path,
                                        lookback=hparams.lookback,
                                        privileged_lookback=hparams.privileged_lookback
                                        )
        elif hparams.model_type == 'PSF':
            dataset_train = dh.PrivilegedDataset(file_path=train_dataset_path,
                                        lookback=hparams.lookback,
                                        privileged_lookback=hparams.privileged_lookback
                                        )
        elif hparams.model_type == 'SSF':
            dataset_train = dh.RealDataset(file_path=train_dataset_path,
                                    lookback=hparams.lookback
                                    )
        else:
            raise ValueError
        horizon_train = min(int(hparams.plot_horizon/2), dataset_train.n_samples)
        print("Loaded training dataset.")

        synth_plot_train = np.ones((dataset_train.n_samples, dataset_train.data_dim)) * np.nan

        if hparams.model_type == 'PSF':
            y_pred = model(dataset_train.get_all_sequences(), dataset_train.get_all_pi()
                        ).reshape(-1,hparams.data_dim)
            
        elif hparams.model_type == 'SSF':
            y_pred = model(dataset_train.get_all_sequences()
                        ).reshape(-1,hparams.data_dim)

        elif hparams.model_type == "FFSF":
            y_pred = model(dataset_train.get_all_sequences(), dataset_train.get_all_pi()
                        ).reshape(-1)

        if hparams.model_type in ['SSF', 'PSF']:
            synth_plot_train[hparams.lookback:] = y_pred[hparams.lookback:]
        elif hparams.model_type == 'FFSF':
            synth_plot_train = y_pred

        print("Predictions on training set done.")


        # On the TEST set
        if hparams.model_type == 'FFSF':
            dataset_test = dh.FeedDataset(file_path=test_dataset_path,
                                        lookback=hparams.lookback,
                                        privileged_lookback=hparams.privileged_lookback
                                        )
        elif hparams.model_type == 'PSF':
            dataset_test = dh.PrivilegedDataset(file_path=test_dataset_path,
                                        lookback=hparams.lookback,
                                        privileged_lookback=hparams.privileged_lookback
                                        )
        elif hparams.model_type == 'SSF':
            dataset_test = dh.RealDataset(file_path=test_dataset_path,
                                    lookback=hparams.lookback
                                    )
        else:
            raise ValueError
        horizon_test = min(int(hparams.plot_horizon/2), dataset_test.n_samples)
        print("Loaded testing dataset.")
        
        synth_plot_test = np.ones((dataset_train.n_samples+dataset_test.n_samples, dataset_test.data_dim)) * np.nan
        
        plot_test = np.ones((dataset_train.n_samples+dataset_test.n_samples, dataset_test.data_dim)) * np.nan
        if hparams.model_type == 'FFSF':
            plot_test = plot_test.reshape(-1)
            plot_test[dataset_train.n_samples:] = dataset_test.get_all_targets()
            synth_plot_test = synth_plot_test.reshape(-1)
        else:
            plot_test[dataset_train.n_samples:] = dataset_test.get_whole_stream()

        if hparams.model_type == 'PSF':
            y_pred = model(dataset_test.get_all_sequences(), dataset_test.get_all_pi()
                        ).reshape(-1,hparams.data_dim)
            
        elif hparams.model_type == 'SSF':
            y_pred = model(dataset_test.get_all_sequences()
                        ).reshape(-1,hparams.data_dim)

        elif hparams.model_type == "FFSF":
            y_pred = model(dataset_test.get_all_sequences(), dataset_test.get_all_pi()
                        ).reshape(-1)

        if hparams.model_type in ['SSF', 'PSF']:
            synth_plot_test[dataset_train.n_samples+hparams.lookback:] = y_pred[hparams.lookback:]
        elif hparams.model_type == 'FFSF':
            synth_plot_test[dataset_train.n_samples:] = y_pred

        print("Predictions on testing set done.")

        plt.figure(figsize=(50,20),dpi=300)
        plt.grid(True)
        plt.minorticks_on()
        # Only plot the first dimension
        if hparams.model_type in ['PSF', 'SSF']:
            plt.plot(dataset_train.get_whole_stream()[:horizon_train,0], c='b')
            #print(dataset_train.get_whole_stream()[:10])
            plt.plot(synth_plot_train[:horizon_train,0], c='r')

            plt.plot(plot_test[dataset_train.n_samples-horizon_test : dataset_train.n_samples+horizon_test,0], c='b')
            plt.plot(synth_plot_test[dataset_train.n_samples-horizon_test : dataset_train.n_samples+horizon_test,0], c='g')

        else:
            plt.plot(dataset_train.get_all_targets()[:horizon_train], c='b')
            plt.plot(synth_plot_train[:horizon_train], c='r')

            plt.plot(plot_test[dataset_train.n_samples : dataset_train.n_samples+horizon_test], c='b')
            plt.plot(synth_plot_test[dataset_train.n_samples : dataset_train.n_samples+horizon_test], c='g')

        print("Plot done.")
        plt.savefig(f"img/{hparams.model_type}-{hparams.n_epochs}-e-{hparams.hidden_dim}-hs-{hparams.seed}-seed.png")
        plt.show()



### Testing Area
if __name__ == '__main__':
    import utilities as ut
    ut.set_seed(Config().seed)
    ut.generate_data()
    train()


