
from torch import nn, Tensor, zeros
from torch.nn.init import normal_, xavier_uniform_, zeros_


class Cell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, var=0.02, num_layers=3,
                 module_type='gru', normalize=False) -> None:
        '''
        The Embedder maps the input sequence to a lower dimensionality representation.
        Args:
            - module_type: what module to use between RNN, GRU and LSTM
            - input_size: dimensionality of one sample
            - hidden_size: dimensionality of the sample returned by the module
            - num_layers: depth of the module
            - output_size: size of the final output
            - normalize: whether to normalize the samples or not
        '''
        assert(module_type in ['rnn', 'gru', 'lstm'])

        super().__init__()
        # input.shape = ( batch_size, seq_len, feature_size )
        if module_type == 'rnn':
            self.module = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        elif module_type == 'gru':
            self.module = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        elif module_type == 'lstm':
            self.module = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        else:
            assert(False)

        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x: Tensor) -> Tensor:
        '''
        Forward pass
        '''
        x, _ = self.module(x) # shape = ( batch_size, seq_len, output_size )
        x = self.fc(x)

        return x



def init_weights(m):
    '''
    Initialized the weights of the nn.Sequential block
    '''
    if isinstance(m, nn.Linear):
        xavier_uniform_(m.weight)
        if hasattr(m, "bias") and m.bias is not None:
            zeros_(m.bias)