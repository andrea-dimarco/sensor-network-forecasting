import warnings
warnings.filterwarnings("ignore")

import time
import torch
import torch.nn as nn
import torch.optim as optim
import dataset_handling as dh
import matplotlib.pyplot as plt
import torch.utils.data as data
from hyperparameters import Config
import utilities as ut

    

class FFSF(nn.Module):
    def __init__(self,
                 data_dim:int,
                 hidden_dim:int,
                 lookback:int,
                 num_layers:int=1) -> None:
        '''
        The Single Sensor Forecasting model.

        Arguments:
            - `data_dim`: dimension of the single sample 
            - `hidden_dim`: hidden dimension
            - `num_layers`: number of linear layers
            - `lookback`: how many backward steps to look
        '''
        super().__init__()
        assert(data_dim > 0)
        assert(hidden_dim > 0)
        assert(num_layers > 0)
        assert(lookback > 0)

        self.pi_dim = data_dim*4
        # Initialize Modules
        # input = ( batch_size, lookback*data_dim )
        model = [
            nn.Linear(in_features=lookback*data_dim, out_features=hidden_dim),
            nn.ReLU(inplace=True)
        ]
        for i in range(num_layers-1):
            model += [
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                nn.ReLU(inplace=True)
            ]
        self.feed = nn.Sequential(*model)
        self.fc = nn.Linear(in_features=hidden_dim+self.pi_dim,
                            out_features=data_dim)
        
        # init weights
        self.feed.apply(ut.init_weights)
        self.fc.apply(ut.init_weights)
    
    def forward(self, x:torch.Tensor, p:torch.Tensor) -> torch.Tensor:
        '''
        Takes the noise and generates a batch of sequences

        Arguments:
            - `x`: input of the forward pass with shape [batch, lookback, data_dim]

        Returns:
            - the predicted sequences [batch, lookback, data_dim]
        '''
        # x    = (batch, lookback*data)
        x = self.feed(x)
        # x   = (batch, hidden)
        # p   = (batch, 4)
        x = torch.cat((x,p), dim=1)
        # x   = (batch, hidden+4)
        x = self.fc(x)
        # x   = (batch, data)
        return x


def train_model(train_file_path="./datasets/training.csv",
                val_file_path="./datasets/testing.csv",
                plot_loss:bool=False,
                loss_fn=nn.MSELoss(),
                val_frequency:int=100
                ):
    '''
    Instanciates and trains the model.

    Arguments:
        - `X_train`: train Tensor [n_sequences, lookback, data_dim]
        - `y_train`: the targets
        - `X_val`: seques for validation
        - `y_val`: targets for validation
        - `plot_loss`: if to plot the loss or not
        - `loss_fn`: the loss function to use
        - `val_frequency`: after how many epochs to run a validation epoch
    '''
    hparams = Config()
    train_dataset = dh.FeedDataset(file_path=train_file_path,
                                   lookback=hparams.lookback,
                                   privileged_lookback=hparams.privileged_lookback
                                   )
    val_dataset = dh.FeedDataset(file_path=val_file_path,
                                   lookback=hparams.lookback,
                                   privileged_lookback=hparams.privileged_lookback
                                   )
    data_dim = train_dataset.data_dim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}.")
    input_size = data_dim
    hidden_size = hparams.hidden_dim
    batch_size = hparams.batch_size
    n_epochs = hparams.n_epochs
    num_layers = hparams.num_layers

    model = FFSF(data_dim=input_size,
                    hidden_dim=hidden_size,
                    num_layers=num_layers,
                    lookback=hparams.lookback
                    ).to(device=device)
    print("Parameters count: ", ut.count_parameters(model))
    optimizer = optim.Adam(model.parameters(),
                           lr=hparams.lr,
                           betas=(hparams.b1, hparams.b2)
                           )
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                     start_factor=hparams.decay_start,
                                                     end_factor=hparams.decay_end,
                                                     total_iters=n_epochs
                                                     )
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True
                                   )
    
    # Required for CUDA
    TRAINING_SET = train_dataset.get_all_sequences().to(device=device)
    TRAINING_PI = train_dataset.get_all_pi().to(device=device)
    TRAINING_TARGETS = train_dataset.get_all_targets().to(device=device)
    VALIDATION_SET = val_dataset.get_all_sequences().to(device=device)
    VALIDATION_PI = val_dataset.get_all_pi().to(device=device)
    VALIDATION_TARGETS = val_dataset.get_all_targets().to(device=device)

    print("Begin Training")
    loss_history = []
    start_time = time.time()
    for epoch in range(n_epochs):
        # Training step
        model.train()
        for X_batch, y_batch, p_batch in train_loader:
            y_pred = model(X_batch.to(device=device), p_batch.to(device=device))
            loss = loss_fn(y_pred, y_batch.to(device=device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation step
        if epoch % val_frequency == 0:
            model.eval()
            with torch.no_grad():
                y_pred = model(TRAINING_SET, TRAINING_PI)
                train_loss = torch.sqrt(loss_fn(y_pred, TRAINING_TARGETS))
                y_pred = model(VALIDATION_SET, VALIDATION_PI)
                val_loss = torch.sqrt(loss_fn(y_pred, VALIDATION_TARGETS))
                if plot_loss:
                    loss_history.append(val_loss.item())
            end_time = time.time()
            print("Epoch %d/%d: train_loss=%.4f, val_loss=%.4f, lr=%.4f, elapsed_time=%.2fs" % (epoch, n_epochs, train_loss, val_loss, optimizer.param_groups[0]["lr"], end_time-start_time))
            start_time = time.time()
        lr_scheduler.step()
    
    # Save loss plot
    if plot_loss:
        plt.plot(loss_history, label="val_loss")
        plt.savefig(f"img/loss-{n_epochs}-e.png")

    # Log the trained model
    torch.save(model.state_dict(), f"./models/FFSF-{hidden_size}-hidden-{data_dim}-input-{num_layers}-layer.pth")
    return model


if __name__ == '__main__':
    ut.set_seed(42)
    hparams = Config()
    datasets_folder = "./datasets/"

    # dataset path
    if hparams.dataset_name in ['sine', 'wien', 'iid', 'cov']:
        train_dataset_path = f"{datasets_folder}{hparams.dataset_name}_training.csv"
        test_dataset_path  = f"{datasets_folder}{hparams.dataset_name}_testing.csv"
    elif hparams.dataset_name == 'real':
        train_dataset_path = str(datasets_folder) + hparams.train_file_name
        test_dataset_path = str(datasets_folder) + hparams.test_file_name
    else:
        raise ValueError("Dataset not supported.")
    
    # train model
    model = train_model(train_file_path=train_dataset_path,
                        val_file_path=test_dataset_path
                        )
    
    # plot graph
    ut.validate_model(model=model,
                   train_dataset_path=train_dataset_path,
                   test_dataset_path=test_dataset_path,
                   model_type='FFSF'
                   )

    