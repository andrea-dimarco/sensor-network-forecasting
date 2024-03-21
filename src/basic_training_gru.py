import warnings
warnings.filterwarnings("ignore")


import torch
import random
import time
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch.utils.data as data
from torchsummary import summary
from hyperparameters import Config
from utilities import validate_model
from data_generation.wiener_process import multi_dim_wiener_process


class GSF(nn.Module):
    def __init__(self,
                 data_dim:int,
                 hidden_dim:int,
                 num_layers:int=1
                 ) -> None:
        '''
        The Single Sensor Forecasting model.

        Arguments:
            - `data_dim`: dimension of the single sample 
            - `hidden_dim`: hidden dimension
            - `num_layers`: number of gru layers
        '''
        super().__init__()
        assert(data_dim > 0)
        assert(hidden_dim > 0)
        assert(num_layers > 0)

        # Initialize Modules
        # input = ( batch_size, lookback, data_dim )
        self.gru = nn.GRU(input_size=data_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True
                            )
        self.fc = nn.Linear(in_features=hidden_dim,
                            out_features=data_dim
                            )
        
        # init weights
        self.fc.apply(init_weights)
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.xavier_uniform_(self.gru.__getattr__(p))
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        '''
        Takes the noise and generates a batch of sequences

        Arguments:
            - `x`: input of the forward pass with shape [batch, lookback, data_dim]

        Returns:
            - the predicted sequences [batch, lookback, data_dim]
        '''
        x, _ = self.gru(x)
        x = self.fc(x)
        return x
    

def init_weights(m):
    '''
    Initialized the weights of the nn.Sequential block
    '''
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.zeros_(m.bias)


def create_dataset(dataset, lookback:int):
    """
    Transform a time series into a prediction dataset
    
    Args:
        `dataset`: A numpy array of time series, first dimension is the time steps
        `lookback`: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X).type(torch.float32), torch.tensor(y).type(torch.float32)


def set_seed(seed=0):
    '''
    Sets the global seed
    
    Arguments:
        - `seed`: the seed to be set
    '''
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True # Can have performance impact
    torch.backends.cudnn.benchmark = False

    _ = pl.seed_everything(seed)


def get_data(verbose=True):
    '''
    Gets and returns the datasets as torch.Tensors
    '''
    hparams = Config()
    lookback = hparams.lookback
    dataset_name = hparams.dataset_name

    # LOAD DATASET
    if dataset_name == 'wien':
        dataset = multi_dim_wiener_process(p=hparams.data_dim, N=hparams.num_samples)

    elif dataset_name == 'real':
        dataset_path = "./datasets/sensor_data_multi.csv"
        dataset = np.loadtxt(dataset_path, delimiter=",", dtype=np.float32)
        n_samples = dataset.shape[0]

    else:
        raise ValueError

    n_samples = dataset.shape[0]
    try:
        data_dim = dataset.shape[1]
    except:
        data_dim = 1
        dataset = dataset.reshape(n_samples, data_dim)

    train_size = int(n_samples*hparams.train_test_split)
    train = dataset[:train_size]
    test = dataset[train_size:]


    # SPLIT DATASET
    X_train, y_train = create_dataset(train, lookback=lookback)
    X_test, y_test = create_dataset(test, lookback=lookback)


    # CONVERT TO TORCH
    dataset = torch.from_numpy(dataset).type(torch.float32)
    train = torch.from_numpy(train).type(torch.float32)
    test = torch.from_numpy(test).type(torch.float32)


    # RETURN RESULTS
    if verbose:
        print(f"Training Features: {X_train.size()}, Training Targets {y_train.size()}")
        print(f"Testing Features: {X_test.size()}, Testing Targets {y_test.size()}")
        print(f"Shape: ( num_sequences, lookback, data_dim )")

    return X_train, y_train, X_test, y_test


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(X_train:torch.Tensor,
                y_train:torch.Tensor,
                X_val:torch.Tensor,
                y_val:torch.Tensor,
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
    try:
        data_dim = X_train.size()[2]
    except:
        data_dim = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}.")
    input_size = data_dim
    hidden_size = hparams.hidden_dim
    batch_size = hparams.batch_size
    n_epochs = hparams.n_epochs
    num_layers = hparams.num_layers

    model = GSF(data_dim=input_size,
                hidden_dim=hidden_size,
                num_layers=num_layers,
                ).to(device=device)
    print("Parameters count: ", count_parameters(model))

    optimizer = optim.Adam(model.parameters(),
                           lr=hparams.lr,
                           betas=(hparams.b1, hparams.b2)
                           )
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                     start_factor=hparams.decay_start,
                                                     end_factor=hparams.decay_end,
                                                     total_iters=n_epochs
                                                     )
    train_loader = data.DataLoader(data.TensorDataset(X_train, y_train),
                                   shuffle=True,
                                   batch_size=batch_size
                                   )
    # requred for cuda
    TRAINING_SET = X_train.to(device=device)
    TRAINING_TARGETS = y_train.to(device=device)
    VALIDATION_SET = X_val.to(device=device)
    VALIDATION_TARGETS = y_val.to(device=device)

    print("Begin Training")
    loss_history = []
    start_time = time.time()
    for epoch in range(n_epochs):
        # Training step
        model.train()
        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch.to(device=device))
            loss = loss_fn(y_pred, y_batch.to(device=device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation step
        if epoch % val_frequency == 0:
            model.eval()
            with torch.no_grad():
                y_pred = model(TRAINING_SET)
                train_loss = torch.sqrt(loss_fn(y_pred, TRAINING_TARGETS))
                y_pred = model(VALIDATION_SET)
                val_loss = torch.sqrt(loss_fn(y_pred, VALIDATION_TARGETS))
                if plot_loss:
                    loss_history.append(val_loss.item())
            end_time = time.time()
            print("Epoch %d/%d: train_loss=%.4f, val_loss=%.4f, lr=%.4f, elapsed_time=%.4fs" % (epoch, n_epochs, train_loss, val_loss, optimizer.param_groups[0]["lr"], end_time-start_time))
            start_time = time.time()
        lr_scheduler.step()
    
    # Save loss plot
    if plot_loss:
        plt.plot(loss_history, label="val_loss")
        plt.savefig(f"img/loss-{n_epochs}-e.png")

    # Log the trained model
    torch.save(model.state_dict(), f"./GSF-model.pth")
    return model


def load_model(data_dim:int=526,
               hidden_dim:int=1500,
               num_layers:int=1
               ) -> GSF:
    '''
    Rturns the pretrained model.

    Arguments:
        - `data_dim`: dimension of one sample
        - `hidden_dim`: hidden dimension of the model
        - `num_layers`: number of concatenated gru networks
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GSF(data_dim=data_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                )
    model.load_state_dict(torch.load(f"./GSF-{hidden_dim}-hidden-{data_dim}-input-{num_layers}-layer.pth", map_location=device))
    print(f"GSF model with input_dim={data_dim}, hidden_dim={hidden_dim}, num_layers={num_layers} has been loaded.")
    return model


if __name__ == '__main__':
    # setup
    hparams = Config()
    set_seed(hparams.seed)

    if hparams.load_model:
        model = load_model()

    else:
        X_train, y_train, X_test, y_test = get_data()
        model = train_model(X_train=X_train,
                            y_train=y_train,
                            X_val=X_test,
                            y_val=y_test,
                            val_frequency=100
                            )
        del X_train, y_train, X_test, y_test 
    
    # Validation
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
    
    # plot graph
    validate_model(model=model,
                   train_dataset_path=train_dataset_path,
                   test_dataset_path=test_dataset_path,
                   lookback=10,
                   model_type='GSF'
                   )

    