import warnings
warnings.filterwarnings("ignore")


import numpy as np
import torch
'''
Utility
'''
def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X).type(torch.float32), torch.tensor(y).type(torch.float32)

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
set_seed(69)

'''
Dataset Generation
'''
from hyperparameters import Config
from data_generation.wiener_process import multi_dim_wiener_process
from sklearn.preprocessing import MinMaxScaler
if True:
    hparams = Config()
    data_dim = hparams.data_dim
    n_samples = hparams.num_samples
    lookback = hparams.seq_len
    train_size = int(n_samples*0.7)

    dataset = multi_dim_wiener_process(p=data_dim, N=n_samples)
    scaler = MinMaxScaler(feature_range=(-1,1)) # preserves the data distribution
    dataset = dataset.reshape(n_samples, data_dim) # needed when data_dim == 1
    scaler.fit(dataset)
    dataset = scaler.transform(dataset)

    train = dataset[:train_size]
    test = dataset[train_size:]

    X_train, y_train = create_dataset(train, lookback=lookback)
    X_test, y_test = create_dataset(test, lookback=lookback)

    dataset = torch.from_numpy(dataset).type(torch.float32)
    train = torch.from_numpy(train).type(torch.float32)
    test = torch.from_numpy(test).type(torch.float32)

    print(f"Training Features: {X_train.size()}, Training Targets {y_train.size()}")
    print(f"Testing Features: {X_train.size()}, Testing Targets {y_train.size()}")
    print(f"Shape: ( num_sequences, seq_len, data_dim )")



'''
Define Model
'''
import torch.nn as nn
class Forecaster(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size,
                                output_size)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
    

'''
Training
'''
import torch.optim as optim
import torch.utils.data as data

if True:
    input_size = data_dim
    hidden_size = hparams.hidden_dim#50
    output_size = data_dim
    batch_size = hparams.batch_size
    n_epochs = hparams.n_epochs
    val_frequency = 3

    loss_fn = nn.MSELoss()
    model = Forecaster(input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=output_size,
                    num_layers=1
                    )
    optimizer = optim.Adam(model.parameters())
    train_loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=batch_size)
    
    print("Begin Training")

    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        if epoch % val_frequency != 0:
            continue
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train)
            train_rmse = np.sqrt(loss_fn(y_pred, y_train))
            y_pred = model(X_test)
            test_rmse = np.sqrt(loss_fn(y_pred, y_test))
        print("Epoch %d/%d: train RMSE %.4f, test RMSE %.4f" % (epoch, n_epochs-1, train_rmse, test_rmse))



'''
Validation
'''
import matplotlib.pyplot as plt
if True:
    with torch.no_grad():
        # shift train predictions for plotting
        train_plot = np.ones_like(dataset) * np.nan
        y_pred = model(X_train)
        y_pred = y_pred[:, -1, :]
        train_plot[lookback : train.size()[0]] = model(X_train)[:, -1, :]

        # shift test predictions for plotting
        test_plot = np.ones_like(dataset) * np.nan
        test_plot[train_size+lookback:dataset.size()[0]] = model(X_test)[:, -1, :]
    # plot
    plt.plot(dataset, c='b')
    plt.plot(train_plot, c='r')
    plt.plot(test_plot, c='g')
    plt.show()