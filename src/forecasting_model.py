

# Libraries
from typing import Sequence, List, Dict, Tuple, Union, Mapping

from dataclasses import asdict
from pathlib import Path

from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from torch import optim

import wandb
import pytorch_lightning as pl

# My modules
import dataset_handling as dh
import utilities as ut
from hyperparameters import Config
from cell import Cell

'''
Single Sensor Forecasting
'''
class SSF(pl.LightningModule):
    def __init__(self,
        hparams: Union[Dict, Config],
        train_file_path: Path,
        val_file_path: Path,
        plot_losses: bool=False
    ) -> None:
        '''
        The TimeGAN model.

        Arguments:
            - `hparams`: dictionary that contains all the hyperparameters
            - `train_file_path`: Path to the folder that contains the training stream
            - `val_file_path`: Path to the file that contains the testing stream
            - `plot_losses`: Saves the losses in the `loss_history` and `val_loss_history`
        '''
        super().__init__()
        self.save_hyperparameters(asdict(hparams) if not isinstance(hparams, Mapping) else hparams)

        # Dataset paths
        self.train_file_path = train_file_path
        self.val_file_path  = val_file_path

        # loss criteria
        self.reconstruction_loss = torch.nn.MSELoss()

        # Expected shapes 
        self.data_dim = hparams.data_dim
        self.seq_len = hparams.seq_len

        if plot_losses:
            self.plot_losses = True
            self.loss_history = []
            self.val_loss_history = []
        else:
            self.plot_losses = False
            self.loss_history = None
            self.val_loss_history = None

        # Initialize Modules
        # For 1 sensor
        # input.shape = ( batch_size, seq_len, data_dim )
        self.cell = Cell(input_size=hparams.data_dim,
                         output_size=hparams.data_dim,
                         num_layers=hparams.num_layers,
                         module_type=hparams.module_type)
        # Forward pass cache to avoid re-doing some computation
        self.fake = None

        # It avoids wandb logging when lighting does a sanity check on the validation
        self.is_sanity = True


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Takes the noise and generates a batch of sequences

        Arguments:
            - `Z`: input of the forward pass with shape [batch, seq_len, noise_dim]

        Returns:
            - the translated image with shape [batch, seq_len, data_dim]
        '''
        return self.cell(x)


    def train_dataloader(self) -> DataLoader:
        '''
        Create the train set DataLoader

        Returns:
            - `train_loader`: the train set DataLoader
        '''
        train_loader = DataLoader(
            dh.RealDataset(
                file_path=self.train_file_path,
                seq_len=self.seq_len
            ),
            batch_size=self.hparams["batch_size"],
            shuffle=True,
            num_workers=self.hparams["n_cpu"],
            pin_memory=True,
        )
        return train_loader
        

    def val_dataloader(self) -> DataLoader:
        '''
        Create the validation set DataLoader.

        It is deterministic.
        It does not shuffle and does not use random transformation on each image.
        
        Returns:
            - `val_loader`: the validation set DataLoader
        '''
        val_loader = DataLoader(
            dh.RealDataset(
                file_path=self.val_file_path,
                seq_len=self.seq_len
            ),
            batch_size=self.hparams["batch_size"],
            shuffle=False,
            num_workers=self.hparams["n_cpu"],
            pin_memory=True,
        )
        return val_loader


    def configure_optimizers(self
    ) -> Tuple[optim.Optimizer, optim.Optimizer, optim.Optimizer, optim.Optimizer, optim.Optimizer]:
    #) -> Tuple[Sequence[optim.Optimizer], Sequence[Dict[str, Any]]]:
        '''
        Instantiate the optimizers and schedulers.

        We have five optimizers (and relative schedulers):

        - `optim`: optimzer for the Embedder
        
        - `lr_scheduler`: learning rate scheduler for the Embedder

        Each scheduler implements a linear decay to 0 after `self.hparams.decay_epoch`

        Returns:
            - the optimizers
            - the schedulers for the optimizers
        '''

        # Optimizers
        optim = torch.optim.Adam(self.cell.parameters(recurse=True),
                                  lr=self.hparams.lr,
                                  betas=(self.hparams.b1, self.hparams.b2))

        # linear decay scheduler
        #assert(self.hparams["n_epochs"] > self.hparams["decay_epoch"]), "Decay must start BEFORE the training ends!"
        #linear_decay = lambda epoch: float(1.0 - max(0, epoch-self.hparams["decay_epoch"]) / (self.hparams["n_epochs"]-self.hparams["decay_epoch"]))
        

        # Schedulers 
        # lr_scheduler_E = torch.optim.lr_scheduler.LinearLR(
        #     E_optim,
        #     start_factor=1.0,
        #     end_factor=0.1
        # )
        # return (
        #     [E_optim, D_optim, G_optim, S_optim, R_optim],
        #     [
        #         {"scheduler": lr_scheduler_E, "interval": "epoch", "frequency": 1},
        #         {"scheduler": lr_scheduler_D, "interval": "epoch", "frequency": 1},
        #         {"scheduler": lr_scheduler_G, "interval": "epoch", "frequency": 1},
        #         {"scheduler": lr_scheduler_S, "interval": "epoch", "frequency": 1},
        #         {"scheduler": lr_scheduler_R, "interval": "epoch", "frequency": 1}
        #     ]
        # )
        return optim


    def loss_f(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Reconstruction loss
        '''
        x_hat = self.cell(x)

        return self.reconstruction_loss(x, x_hat)
   

    def training_step(self,
                      batch: torch.Tensor, batch_idx: int
    ) -> torch.Tensor:
        '''
        Implements a single training step

        The parameter `optimizer_idx` identifies with optimizer "called" this training step,
        this way we can change the behaviour of the training depending on which optimizer
        is currently performing the optimization

        Arguments:
            - `batch`: current training batch
            - `batch_idx`: the index of the batch being processed

        Returns:
            - the total loss for the current training step, together with other information for the
                  logging and possibly the progress bar
        '''
        # Process the batch
        loss = self.loss_f(batch)

        # Log results
        loss_dict = { "loss": loss }
        self.log_dict(loss_dict)

        return loss


    def validation_step(self,
                        batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int,
    ) -> Dict[str, Union[torch.Tensor,Sequence[wandb.Image]]]:
        '''
        Implements a single validation step

        In each validation step some translation examples are produced and a
        validation loss that uses the cycle consistency is computed

        Arguments:
            - `batch`: the current validation batch

        Returns:
            - the loss and example images
        '''
        # Process the batch
        loss = self.loss_f(batch)


        # visualize result
        image = self.get_image_examples(batch[0], self.cell(batch)[0])

        # Validation loss
        self.log("val_loss", loss)

        return { "val_loss": loss, "image": image }


    def get_image_examples(self,
                           real: torch.Tensor, fake: torch.Tensor,
                           fake_label:str="Synthetic"):
        '''
        Given real and "fake" translated images, produce a nice coupled images to log

        Arguments:
            - `real`: the real sequence with shape [seq_len, data_dim]
            - `fake`: the fake sequence with shape [seq_len, data_dim]

        Returns:
            - A sequence of wandb.Image to log and visualize the performance
        '''
        example_images = []
        couple = ut.compare_sequences(real=real, fake=fake, save_img=False, show_graph=False, fake_label=fake_label)

        example_images.append(
            wandb.Image(couple, mode="RGB")
        )
        return example_images


    def validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        '''
        Implements the behaviouir at the end of a validation epoch

        Currently it gathers all the produced examples and log them to wandb,
        limiting the logged examples to `hparams["log_images"]`.

        Then computes the mean of the losses and returns it.
        Updates the progress bar label with this loss.

        Arguments:
            - outputs: a sequence that aggregates all the outputs of the validation steps

        Returns:
            - The aggregated validation loss and information to update the progress bar
        '''
        images = []

        for x in outputs:
            images.extend(x["image"])

        images = images[: self.hparams.log_images]

        if not self.is_sanity:  # ignore if it not a real validation epoch. The first one is not.
            self.logger.experiment.log(
                {"images": images },
                step=self.global_step,
            )
        self.is_sanity = False

        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log_dict({"val_loss": avg_loss})
        return {"val_loss": avg_loss}
    

    def plot(self):
        if self.plot_losses and len(self.loss_history)>0:
            import numpy as np
            L = np.asarray(self.loss_history)
            ut.plot_processes(samples=L, save_picture=False, show_plot=True)