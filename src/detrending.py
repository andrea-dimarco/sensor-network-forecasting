import torch
import utilities as ut
from hyperparameters import Config
from ssf_training import SSF

def load_model(data_dim:int=526,
               hidden_dim:int=1500,
               num_layers:int=1
               ) -> SSF:
    '''
    Rturns the pretrained model.

    Arguments:
        - `data_dim`: dimension of one sample
        - `hidden_dim`: hidden dimension of the model
        - `num_layers`: number of concatenated lstm networks
    '''
    model = SSF(data_dim=data_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                )
    model.load_state_dict(torch.load(f"./SSF-{hidden_dim}-hidden-{data_dim}-input-{num_layers}-layer.pth"))
    print(f"SSF model with input_dim={data_dim}, hidden_dim={hidden_dim}, num_layers={num_layers} has been loaded.")
    return model

if __name__ == '__main__':
    hparams = Config()
    ut.set_seed(hparams.seed)

    # Get the model
    model = load_model()

    # Get the Anomaly Detector
    # Train Anomaly Detector
    # Stream of values
        # Compute prediction absolute error
        # Query Anomaly Detector