from itertools import chain

import torch
import torch.nn as nn

from blox.tensor.ops import concat_inputs
from blox.torch.subnetworks import HiddenStatePredictorModel
from blox.torch.recurrent_modules import ZeroLSTMCellInitializer, MLPLSTMCellInitializer


class SumTreeHiddenStatePredictorModel(HiddenStatePredictorModel):
    """ A HiddenStatePredictor for tree morphologies. Averages parents' hidden states """

    def forward(self, hidden1, hidden2, *inputs):
        hidden_state = hidden1 + hidden2
        return super().forward(hidden_state, *inputs)


class LinTreeHiddenStatePredictorModel(HiddenStatePredictorModel):
    """ A HiddenStatePredictor for tree morphologies. Averages parents' hidden states """
    def build_network(self):
        super().build_network()
        self.projection = nn.Linear(self.get_state_dim() * 2, self.get_state_dim())

    def forward(self, hidden1, hidden2, *inputs):
        hidden_state = self.projection(concat_inputs(hidden1, hidden2))
        return super().forward(hidden_state, *inputs)


class SplitLinTreeHiddenStatePredictorModel(HiddenStatePredictorModel):
    """ A HiddenStatePredictor for tree morphologies. Averages parents' hidden states """
    def build_network(self):
        super().build_network()
        split_state_size = int(self.get_state_dim() / (self._hp.n_lstm_layers * 2))
        
        if self._hp.use_conv_lstm:
            projection = lambda: nn.Conv2d(split_state_size * 2, split_state_size, kernel_size=3, padding=1)
        else:
            projection = lambda: nn.Linear(split_state_size * 2, split_state_size)
        
        self.projections = torch.nn.ModuleList([projection() for _ in range(self._hp.n_lstm_layers*2)])

    def forward(self, hidden1, hidden2, *inputs):
        chunked_hidden1 = list(chain(*[torch.chunk(h, 2, 1) for h in torch.chunk(hidden1, self._hp.n_lstm_layers, 1)]))
        chunked_hidden2 = list(chain(*[torch.chunk(h, 2, 1) for h in torch.chunk(hidden2, self._hp.n_lstm_layers, 1)]))
        chunked_projected = [projection(concat_inputs(h1, h2))
                             for projection, h1, h2 in zip(self.projections, chunked_hidden1, chunked_hidden2)]
        hidden_state = torch.cat(chunked_projected, dim=1)
        return super().forward(hidden_state, *inputs)


def build_tree_lstm(hp, input_dim, output_dim):
    if hp.tree_lstm == 'sum':
        cls = SumTreeHiddenStatePredictorModel
    elif hp.tree_lstm == 'linear':
        cls = LinTreeHiddenStatePredictorModel
    elif hp.tree_lstm == 'split_linear':
        cls = SplitLinTreeHiddenStatePredictorModel
    else:
        raise ValueError("don't know this TreeLSTM type")

    subgoal_pred = cls(hp, input_dim, output_dim)
    lstm_initializer = get_lstm_initializer(hp, subgoal_pred)
    
    return subgoal_pred, lstm_initializer


def get_lstm_initializer(hp, cell):
    if hp.lstm_init == 'zero':
        return ZeroLSTMCellInitializer(hp, cell)
    elif hp.lstm_init == 'mlp':
        return MLPLSTMCellInitializer(hp, cell, 2 * hp.nz_enc + hp.nz_vae)
    else:
        raise ValueError('dont know lstm init type {}!'.format(hp.lstm_init))
