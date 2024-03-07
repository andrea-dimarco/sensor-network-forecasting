import warnings
warnings.filterwarnings("ignore")


import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch.utils.data as data
from hyperparameters import Config
from data_generation.wiener_process import multi_dim_wiener_process


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
        dataset_path = "./datasets/sensor_data_2.csv"
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

    return dataset, X_train, y_train, X_test, y_test


class SSF(nn.Module):
    def __init__(self,
                 data_dim,
                 hidden_dim,
                 num_layers=1) -> None:
        '''
        The Single Sensor Forecasting model.

        Arguments:
            - `data_dim`: dimension of the single sample 
            - `hidden_dim`: hidden dimension
            - `num_layers`: number of lstm layers
        '''
        super().__init__()
        assert(data_dim > 0)
        assert(hidden_dim > 0)
        assert(num_layers > 0)

        # Initialize Modules
        # input = ( batch_size, lookback, data_dim )
        self.lstm = nn.LSTM(input_size=data_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True
                            )
        self.fc = nn.Linear(in_features=hidden_dim,
                            out_features=data_dim
                            )
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        '''
        Takes the noise and generates a batch of sequences

        Arguments:
            - `x`: input of the forward pass with shape [batch, lookback, data_dim]

        Returns:
            - the predicted sequences [batch, lookback, data_dim]
        '''
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x
    

def train_model(X_train:torch.Tensor,
                y_train:torch.Tensor,
                X_val:torch.Tensor,
                y_val:torch.Tensor
                ):
    '''
    Instanciates and trains the model.
    '''
    hparams = Config()
    try:
        data_dim = X_train.size()[2]
    except:
        data_dim = 1

    input_size = data_dim
    hidden_size = hparams.hidden_dim
    batch_size = hparams.batch_size
    n_epochs = hparams.n_epochs
    num_layers = hparams.num_layers
    val_frequency = 3

    loss_fn = nn.L1Loss()
    model = SSF(data_dim=input_size,
                hidden_dim=hidden_size,
                num_layers=num_layers
                )
    optimizer = optim.Adam(model.parameters())
    train_loader = data.DataLoader(data.TensorDataset(X_train, y_train),
                                   shuffle=True,
                                   batch_size=batch_size)
    
    print("Begin Training")

    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation step
        if epoch % val_frequency != 0:
            continue
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train)
            train_rmse = np.sqrt(loss_fn(y_pred, y_train))
            y_pred = model(X_val)
            test_rmse = np.sqrt(loss_fn(y_pred, y_val))
        print("Epoch %d/%d: train RMSE %.4f, val RMSE %.4f" % (epoch, n_epochs-1, train_rmse, test_rmse))
    
    # Log the trained model
    torch.save(model.state_dict(), f"./{hparams.model_type}-model.pth")
    return model


def validate(model,
             dataset:torch.Tensor,
             X_train:torch.Tensor,
             X_test:torch.Tensor
             ):
    '''
    Plots model's prediction on train and test datasets, against the ground truths.
    '''
    #TODO: This does not work
    hparams = Config()
    lookback = hparams.lookback
    train_size = X_train.size()[0]
    test_size = X_test.size()[0]
    dataset_size = dataset.size()[0]
    data_dim = X_test.size()[2]

    with torch.no_grad():
        # shift train predictions for plotting
        train_plot = np.ones((dataset_size, data_dim)) * np.nan
        print("X_TRAIN SHAPE:", X_train.shape)
        y_pred_train = model(X_train)[:-1:]
        print("Y_PRED SHAPE:", y_pred_train.shape)
        train_plot[lookback:train_size] = y_pred_train

        # shift test predictions for plotting
        test_plot = np.ones((dataset_size, data_dim)) * np.nan
        y_pred_test = model(X_test)[:-1:]
        test_plot[train_size+lookback:dataset_size] = y_pred_test
    # plot
    plt.plot(dataset, c='b')
    plt.plot(train_plot, c='r')
    plt.plot(test_plot, c='g')
    plt.savefig(f"./{hparams.model_type}-{hparams.n_epochs}-e-{hparams.hidden_dim}-hs-{hparams.seed}-seed.png",dpi=300)
    plt.show()



if __name__ == '__main__':
    set_seed(42)
    dataset, X_train, y_train, X_test, y_test = get_data()
    model = train_model(X_train=X_train,
                        y_train=y_train,
                        X_val=X_test,
                        y_val=y_test
                        )

    # TODO: fix this workaround, maybe
    # validate(model=model,
    #          dataset=dataset,
    #          X_train=X_train,
    #          X_test=X_test
    #          )
    from lightning_training import validate_model
    hparams = Config()
    datasets_folder = "./datasets/"
    if hparams.dataset_name in ['sine', 'wien', 'iid', 'cov']:
        train_dataset_path = f"{datasets_folder}{hparams.dataset_name}_training.csv"
        test_dataset_path  = f"{datasets_folder}{hparams.dataset_name}_testing.csv"

    elif hparams.dataset_name == 'real':
        train_dataset_path = str(datasets_folder) + hparams.train_file_name
        test_dataset_path = str(datasets_folder) + hparams.test_file_name
    else:
        raise ValueError("Dataset not supported.")
    validate_model(model=model,
                   train_dataset_path=train_dataset_path,
                   test_dataset_path=test_dataset_path
                   )

    