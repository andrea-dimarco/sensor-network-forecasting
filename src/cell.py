
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
        self.module_type = module_type
        self.num_layers = num_layers
        self.num_final_layers = int(num_layers/3+1)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.normalize = normalize
        
        # input.shape = ( batch_size, seq_len, feature_size )
        if self.module_type == 'rnn':
            self.module = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        elif self.module_type == 'gru':
            self.module = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        elif self.module_type == 'lstm':
            self.module = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        else:
            assert(False)

        self.fc = nn.Linear(hidden_size, output_size)

        # Normalization
        if self.normalize:
            self.norm = nn.InstanceNorm1d(output_size, affine=True)
        else:
            self.norm = None


    def forward(self, x: Tensor) -> Tensor:
        '''
        Forward pass
        '''
        batch_size = x.size()[0]

        if self.module_type == 'lstm':
            out, _ = self.module(x) # shape = ( batch_size, seq_len, output_size )
        else:
            out, _ = self.module(x) # shape = ( batch_size, seq_len, output_size )

        if self.normalize:
            # required shape (batch_size, output_size, seq_len )
            out = self.norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        #out = out[:,-1,:] # only consider the last output of each sequence
        out = self.fc(out)

        return out



def init_weights(m):
    '''
    Initialized the weights of the nn.Sequential block
    '''
    if isinstance(m, nn.Linear):
        xavier_uniform_(m.weight)
        if hasattr(m, "bias") and m.bias is not None:
            zeros_(m.bias)