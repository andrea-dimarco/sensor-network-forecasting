import warnings
warnings.filterwarnings("ignore")

import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch.utils.data as data
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from hyperparameters import Config
from data_generation.wiener_process import multi_dim_wiener_process
import utilities as ut

    

class FFSD(nn.Module):
    def __init__(self,
                 data_dim:int,
                 hidden_dim:int,
                 lookback:int,
                 num_layers:int=1,
                 discretization:int=2) -> None:
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

        self.data_dim = data_dim
        self.discretization = discretization
        self.lookback = lookback
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
        self.fc = nn.Linear(in_features=hidden_dim,
                            out_features=(discretization*2+1)*data_dim
                            )
        self.softmax = nn.Softmax(dim=2)
        
        # init weights
        self.feed.apply(init_weights)
        self.fc.apply(init_weights)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        '''
        Takes the noise and generates a batch of sequences

        Arguments:
            - `x`: input of the forward pass with shape [batch, lookback, data_dim]

        Returns:
            - the predicted sequences [batch, lookback, data_dim]
        '''
        batch = x.size()[0]
        # x = (batch, lookback*data)
        x = self.feed(x)
        # x = (batch, hidden)
        x = self.fc(x)
        # x = ( batch, data_dim, discretization )
        x = x.reshape(batch, self.data_dim, self.discretization*2+1)
        x = self.softmax(x)
        return x


def init_weights(m):
    '''
    Initialized the weights of the nn.Sequential block
    '''
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.zeros_(m.bias)


def create_dataset(dataset:np.ndarray, lookback:int, discretization:int=Config().discretization):
    """
    Transform a time series into a prediction dataset
    
    Args:
        `dataset`: A numpy array of time series, first dimension is the time steps
        `lookback`: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback].reshape(-1) # ( lookback*data_dim )
        # TODO: is this index correct
        step = dataset[i+lookback] # ( data_dim )
        # TODO: compute std and mean on the whole dataset
        std = feature.std()
        mu = feature.mean()
        target = [] # ( data_dim, discretization )
        for sensor in step:
            classification = [0 for i in range(discretization*2+1)] # ( discretization )
            c = int((sensor-mu)/std)
            # lower bound
            c = max(c, -discretization)
            # upper bound
            c = min(c, discretization)
            classification[c+discretization] = 1
            # discretization = 2
            # [1, 0, 0, 0, 0] (-inf)*std < x < (-2)*std
            # [0, 1, 0, 0, 0]   (-2)*std < x < (-1)*std
            # [0, 0, 1, 0, 0]   (-1)*std < x < (+0)*std
            # [0, 0, 1, 0, 0]   (+0)*std < x < (+1)*std
            # [0, 0, 0, 1, 0]   (+1)*std < x < (+2)*std
            # [0, 0, 0, 0, 1]   (+2)*std < x < (+inf)*std
            target.append(classification)
        X.append(feature) 
        y.append(target)
    X = torch.tensor(X).type(torch.float32) # ( n_seq, lookback*data_dim )
    y = torch.tensor(y).type(torch.float32) # ( n_seq, data_dim, discretization )
    return X, y


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
        print(f"Training Features: {X_train.size()}, Training Targets: {y_train.size()}")
        print(f"Testing Features: {X_test.size()}, Testing Targets: {y_test.size()}")
        print(f"Shape Train: ( num_sequences, lookback*data_dim )\nShape Test: ( num_sequences, data_dim, discretization )")

    return X_train, y_train, X_test, y_test


def train_model(X_train:torch.Tensor,
                y_train:torch.Tensor,
                X_val:torch.Tensor,
                y_val:torch.Tensor,
                lookback:int=Config().lookback,
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
    discretization = hparams.discretization

    model = FFSD(data_dim=input_size,
                hidden_dim=hidden_size,
                num_layers=num_layers,
                lookback=lookback,
                discretization=discretization
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
    train_loader = data.DataLoader(data.TensorDataset(X_train, y_train),
                                   shuffle=True,
                                   batch_size=batch_size
                                   )

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
                y_pred = model(X_train)
                train_loss = loss_fn(y_pred, y_train)
                y_pred = model(X_val)
                val_loss = loss_fn(y_pred, y_val)
                if plot_loss:
                    loss_history.append(val_loss.item())
            end_time = time.time()
            print("Epoch %d/%d: train_loss=%.4f, val_loss=%.4f, lr=%.4f, elapsed_time=%.2fs" % (epoch, n_epochs, train_loss, val_loss, optimizer.param_groups[0]["lr"], end_time-start_time))
            start_time = time.time()
        lr_scheduler.step()
    
    # Save loss plot
    if plot_loss:
        plt.plot(loss_history, label="validation loss")
        plt.savefig(f"img/loss-{n_epochs}-e.png")

    # Log the trained model
    torch.save(model.state_dict(), f"./models/FFSD-{hidden_size}-hidden-{data_dim}-input-{num_layers}-layer-{discretization}-disc.pth")
    return model


def load_model(data_dim:int=526,
               hidden_dim:int=Config().hidden_dim,
               num_layers:int=Config().num_layers,
               discretization:int=Config().discretization
               ) -> FFSD:
    '''
    Rturns the pretrained model.

    Arguments:
        - `data_dim`: dimension of one sample
        - `hidden_dim`: hidden dimension of the model
        - `num_layers`: number of concatenated gru networks
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FFSD(data_dim=data_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                discretization=discretization
                )
    model.load_state_dict(torch.load(f"./models/FFSD-{hidden_dim}-hidden-{data_dim}-input-{num_layers}-layer-{discretization}-disc.pth", map_location=device))
    print(f"FFSD model with input_dim={data_dim}, hidden_dim={hidden_dim}, num_layers={num_layers} has been loaded.")
    return model


def validate_model(model:FFSD,
                   X_train:torch.Tensor,
                   y_train:torch.Tensor,
                   X_test:torch.Tensor,
                   y_test:torch.Tensor,
                   hparams:Config=Config()
                   ) -> None:
    '''
    Plot funky graph.
    '''
    print("Begin validation.")
    model.eval()
    model.cpu()
    discretization = model.discretization
    with torch.no_grad():
        # TRAINING PREDICTIONS
        y_pred_train = model(X_train) # ( batch, data_dim, discr )
        n_samples_train = y_pred_train.size()[0]
        data_dim = y_pred_train.size()[2]
        _, y_pred_refactored_train = torch.max(y_pred_train, 2) # ( batch, data_dim )
        y_pred_refactored_train -= discretization

        # refactor training set
        _, train_refactored = torch.max(y_train, 2) # ( batch, data_dim )
        train_refactored -= discretization
        print("Validation on training set done.")

        # TESTING PREDICTIONS
        y_pred_test = model(X_test) # ( n_samples, data_dim, discr )
        n_samples_test = y_pred_test.size()[0]
        data_dim = y_pred_test.size()[2]
        _, y_pred_refactored_test = torch.max(y_pred_test, 2) # ( batch, data_dim )
        y_pred_refactored_test -= discretization

        # refactor testing set
        _, test_refactored = torch.max(y_test, 2) # ( batch, data_dim )
        test_refactored -= discretization
        print("Validation on test set done.")

        fig, ax = plt.subplots()
        ax.grid(which = "major", linewidth = 1)
        ax.grid(which = "minor", linewidth = 0.2)
        
        fig.set_size_inches(18.5, 10.5)
        # Only plot the first dimension
        horizon_train = min(int(hparams.plot_horizon/2), n_samples_train)
        horizon_test = min(int(hparams.plot_horizon/2), n_samples_test)

        timesteps = [i for i in range(hparams.plot_horizon)]

        plot_train_targets = np.zeros((horizon_train+horizon_test, data_dim)) * np.nan
        plot_train_targets[:horizon_train] = train_refactored[:horizon_train]
        
        plot_train_preds = np.zeros((horizon_train+horizon_test, data_dim))* np.nan
        plot_train_preds[:horizon_train] = y_pred_refactored_train[:horizon_train]
        
        plot_test_targets = np.zeros((horizon_train+horizon_test, data_dim)) * np.nan
        plot_test_targets[horizon_train:] = test_refactored[:horizon_test]
        
        plot_test_preds = np.zeros((horizon_train+horizon_test, data_dim)) * np.nan
        plot_test_preds[horizon_train:] = y_pred_refactored_test[:horizon_test]

        plt.scatter(y=plot_train_targets[:,0], x=timesteps, c='b')
        plt.scatter(y=plot_train_preds[:,0], x=timesteps, c='g')

        plt.scatter(y=plot_test_targets[:,0], x=timesteps, c='b')
        plt.scatter(y=plot_test_preds[:,0], x=timesteps, c='r')

        plt.grid(True)
        plt.yticks([i for i in range(-discretization,discretization+1)])

        print("Plot done.")
        plt.savefig(f"img/FFSD-{Config().n_epochs}-e-{Config().hidden_dim}-hs-{Config().num_layers}-layers-{Config().seed}-seed.png",dpi=300)
        plt.show()

        return (y_pred_refactored_test[:,0], test_refactored[:,0])
        #return (y_pred_refactored_train, train_refactored)
    

def show_confusion_matrix(actual: torch.Tensor,predicted: torch.Tensor):
    '''
    Display confusion matrix.
    '''
    cm = confusion_matrix(actual,predicted,normalize='pred')
    discretization = Config.discretization
    labels = [i for i in range(-discretization,discretization+1)]
    sns.heatmap(cm * 100, 
                annot=True,
                fmt='g', 
                xticklabels=labels,
                yticklabels=labels)
    plt.ylabel('Prediction',fontsize=13)
    plt.xlabel('Actual',fontsize=13)
    plt.title('Confusion Matrix',fontsize=17)

    plt.savefig(f"img/FFSD_confusion_matrix.png")
    plt.show()
    f1_val = f1_score(actual,predicted,average=None,labels=labels)
    precision_val = precision_score(actual,predicted,average=None,labels=labels)
    print("F1 score:  ", f1_val)
    print("Precision: ", precision_val)


if __name__ == '__main__':
    # setup
    hparams = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(hparams.seed)

    X_train, y_train, X_test, y_test = get_data()

    if hparams.load_model:
        model = load_model(data_dim=X_train.size()[2])
    
    else:
        model = train_model(X_train=X_train.to(device=device),
                            y_train=y_train.to(device=device),
                            X_val=X_test.to(device=device),
                            y_val=y_test.to(device=device),
                            val_frequency=hparams.val_frequency,
                            lookback=hparams.lookback,
                            plot_loss=True
                            )
        
    # Validation
    actual,predicted = validate_model(model=model,
                   X_train=X_train,
                   y_train=y_train,
                   X_test=X_test,
                   y_test=y_test
                   )    
    
    show_confusion_matrix(actual,predicted)