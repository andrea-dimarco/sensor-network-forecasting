import warnings
warnings.filterwarnings("ignore")


import time
import torch
import numpy as np
import torch.nn as nn
import utilities as ut
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.utils.data as data
from hyperparameters import Config
from data_generation.wiener_process import multi_dim_wiener_process
import os

class SSD(nn.Module):
    def __init__(self,
                 data_dim:int,
                 hidden_dim:int,
                 discretization:int,
                 num_layers:int=2
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
        self.data_dim = data_dim
        self.discretization = discretization
        # input = ( batch_size, lookback, data_dim, discretization )
        self.gru = nn.GRU(input_size=data_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True
                            )
        self.fc = nn.Linear(in_features=hidden_dim,
                            out_features=(discretization*2+1)*data_dim
                            )
        
        self.softmax = nn.Softmax(dim=3)
        
        # init weights
        self.fc.apply(ut.init_weights)
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
        # x = ( batch, lookback, data_dim )
        batch = x.size()[0]
        x, _ = self.gru(x) # (batch, lookback, hidden )
        x = self.fc(x).reshape(batch, -1, self.data_dim, self.discretization*2+1)
        x = self.softmax(x) # ( batch, lookback, data_dim, discretization)
        return x
    

def create_dataset(dataset:np.ndarray,
                   lookback:int,
                   discretization:int=Config().discretization):
    """
    Transform a time series into a prediction dataset
    
    Args:
        `dataset`: A numpy array of time series, first dimension is the time steps
        `lookback`: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback] # ( lookback, data_dim )
        
        future_step = dataset[i+1:i+lookback+1]
        std = feature.std()
        mu = feature.mean()
        target = [] # ( lookback, data_dim, discretization )
        for step in future_step:
            sensors_classification = []
            for sensor in step:
                classification = [0 for i in range(discretization*2+1)]
                c = int((sensor-mu)/std)
                # lower bound
                c = max(c, -discretization)
                # upper bound
                c = min(c, discretization)
                classification[c+discretization] = 1
                sensors_classification.append(classification)
                # discretization = 2
                # [1, 0, 0, 0, 0] (-inf)*std < x < (-2)*std
                # [0, 1, 0, 0, 0]   (-2)*std < x < (-1)*std
                # [0, 0, 1, 0, 0]   (-1)*std < x < (+0)*std
                # [0, 0, 1, 0, 0]   (+0)*std < x < (+1)*std
                # [0, 0, 0, 1, 0]   (+1)*std < x < (+2)*std
                # [0, 0, 0, 0, 1]   (+2)*std < x < (+inf)*std
            target.append(sensors_classification)
        X.append(feature)
        y.append(target)
    X = torch.tensor(X).type(torch.float32)
    y = torch.tensor(y).type(torch.float32)
    return X, y


def get_data(verbose=False
             ) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
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


def train_model(X_train:torch.Tensor,
                y_train:torch.Tensor,
                X_val:torch.Tensor,
                y_val:torch.Tensor,
                plot_loss:bool=False,
                loss_fn=nn.BCELoss(),
                val_frequency:int=100,
                experiment_dir: str= "img/",
                verbose:bool=False
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
    device = ut.get_device()
    if verbose: print(f"Using device {device}.")
    input_size = data_dim
    hidden_size = hparams.hidden_dim
    batch_size = hparams.batch_size
    n_epochs = hparams.n_epochs
    num_layers = hparams.num_layers
    discretization = hparams.discretization

    model = SSD(data_dim=input_size,
                hidden_dim=hidden_size,
                num_layers=num_layers,
                discretization=discretization
                ).to(device=device)
    if verbose: print("Parameters count: ", ut.count_parameters(model))
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

    if verbose: print("Begin Training")
    loss_history = []
    start_time = time.time()
    for epoch in range(n_epochs):
        # Training step
        model.train()
        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch.to(device=device))
            loss = loss_fn(y_pred, y_batch.to(device))
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
            if verbose: print("Epoch %d/%d: train_loss=%.4f, val_loss=%.4f, lr=%.4f, elapsed_time=%.2fs" % (epoch, n_epochs, train_loss, val_loss, optimizer.param_groups[0]["lr"], end_time-start_time))
            start_time = time.time()
        lr_scheduler.step()
    
    # Save loss plot
    if plot_loss:
        plt.plot(loss_history, label="validation loss")
        plt.savefig(f"{experiment_dir}/loss-{n_epochs}-e.png")

    # Log the trained model
    torch.save(model.state_dict(), f"./models/SSD-{hidden_size}-hidden-{data_dim}-input-{num_layers}-layer-{discretization}-disc.pth")
    return model


def load_model(data_dim:int=526,
               hidden_dim:int=Config().hidden_dim,
               num_layers:int=Config().num_layers,
               discretization:int=Config().discretization
               ) -> SSD:
    '''
    Rturns the pretrained model.

    Arguments:
        - `data_dim`: dimension of one sample
        - `hidden_dim`: hidden dimension of the model
        - `num_layers`: number of concatenated gru networks
    '''
    device = ut.get_device()
    model = SSD(data_dim=data_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                discretization=discretization
                )
    model.load_state_dict(torch.load(f"./models/SSD-{hidden_dim}-hidden-{data_dim}-input-{num_layers}-layer-{discretization}-disc.pth", map_location=device))
    print(f"SSD model with input_dim={data_dim}, hidden_dim={hidden_dim}, num_layers={num_layers} has been loaded.")
    return model


def validate_model(experiment_dir:str,
                   model:SSD,
                   X_train:torch.Tensor,
                   y_train:torch.Tensor,
                   X_test:torch.Tensor,
                   y_test:torch.Tensor,
                   lookback:int=Config().lookback,
                   hparams:Config=Config(),
                   show_plot: bool=False,
                   verbose: bool=False
                   ) -> None:
    '''
    Plot funky graph.
    '''
    if verbose:print("Begin validation.")
    model.eval()
    model.cpu()
    discretization = model.discretization
    with torch.no_grad():
        # TRAINING PREDICTIONS
        y_pred_train = model(X_train) # ( n_samples, lookback, data_dim, discr )
        n_samples_train = y_pred_train.size()[0]
        data_dim = y_pred_train.size()[2]
        _, y_pred_refactored_train = torch.max(y_pred_train, 3) # ( batch, lookback, discretizaton, pred )
        y_pred_refactored_train -= discretization
        y_pred_refactored_train = y_pred_refactored_train[:,-1,:]

        # refactor training set
        _, train_refactored = torch.max(y_train, 3) # ( batch, lookback, discretizaton, pred )
        train_refactored -= discretization
        train_refactored = train_refactored[:,-1,:]
        if verbose: print("Validation on training set done.")

        # TESTING PREDICTIONS
        y_pred_test = model(X_test) # ( n_samples, lookback, data_dim, discr )
        n_samples_test = y_pred_test.size()[0]
        data_dim = y_pred_test.size()[2]
        _, y_pred_refactored_test = torch.max(y_pred_test, 3) # ( batch, lookback, discretizaton, pred )
        y_pred_refactored_test -= discretization
        y_pred_refactored_test = y_pred_refactored_test[:,-1,:]

        # refactor testing set
        _, test_refactored = torch.max(y_test, 3) # ( batch, lookback, discretizaton, pred )
        test_refactored -= discretization
        test_refactored = test_refactored[:,-1,:]
        if verbose: print("Validation on test set done.")

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


        if verbose: print("Plot done.")
        if show_plot:
            plt.scatter(y=plot_train_targets[:,0], x=timesteps, c='b')
            plt.scatter(y=plot_train_preds[:,0], x=timesteps, c='g')

            plt.scatter(y=plot_test_targets[:,0], x=timesteps, c='b')
            plt.scatter(y=plot_test_preds[:,0], x=timesteps, c='r')

            plt.grid(True)
            plt.yticks([i for i in range(-discretization,discretization+1)])

            plt.savefig(f"{experiment_dir}/SSD-{Config().n_epochs}-e-{Config().hidden_dim}-hs-{Config().num_layers}-layers-{Config().seed}-seed.png",dpi=300)
            plt.show()

        return (y_pred_refactored_test[:,0], test_refactored[:,0])
        #return (y_pred_refactored_train, train_refactored)


def execute_training(sensor:int=None,
                     verbose:bool=False
                     ) -> None:
    # setup
    hparams = Config()
    device = ut.get_device()
    if sensor:
        experiment_dir = ut.initialize_experiment_directory(sensor,is_adversary=False)
    else:
        experiment_dir = "./"

    X_train, y_train, X_test, y_test = get_data()

    if hparams.load_model:
        model = load_model(data_dim=X_train.size()[2])
    
    else:
        train_start_time = time.time()
        model = train_model(X_train=X_train.to(device=device),
                            y_train=y_train.to(device=device),
                            X_val=X_test.to(device=device),
                            y_val=y_test.to(device=device),
                            val_frequency=hparams.val_frequency,
                            plot_loss=False,
                            verbose=verbose
                            )
        train_end_time = time.time()
        train_time = train_end_time - train_start_time
        if sensor:
            log = open(os.path.join(experiment_dir, "log.txt"), "a")
            log.write("Parameters count: "+ str(ut.count_parameters(model)))
            log.write("\n")
            log.write("Training time: " + str(train_time))
            log.write("\n")
            log.close()
    # Validation
    actual, predicted = validate_model(experiment_dir=experiment_dir,
                                       model=model,
                                       X_train=X_train,
                                       y_train=y_train,
                                       X_test=X_test,
                                       y_test=y_test
                                       )    
    
    ut.show_summary_statistics(experiment_dir=experiment_dir,
                               actual=actual,
                               predicted=predicted,
                               model_name='SSD'
                               )


if __name__ == '__main__':
    execute_training(verbose=True)