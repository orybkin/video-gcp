import torch
from torch.distributions import OneHotCategorical
import torch.nn as nn

from blox import batch_apply, AttrDict
from blox.tensor.ops import remove_spatial, broadcast_final
from blox.torch.losses import CELogitsLoss
from blox.torch.recurrent_modules import BaseProcessingLSTM
from blox.torch.subnetworks import SeqEncodingModule, Predictor


class AttnKeyEncodingModule(SeqEncodingModule):
    def build_network(self, input_size, hp):
        self.net = Predictor(hp, input_size, hp.nz_attn_key, num_layers=1)

    def forward(self, seq):
        return batch_apply(self.net, seq.contiguous())


class RecurrentPolicyModule(SeqEncodingModule):
    def __init__(self, hp, input_size, output_size, add_time=True):
        super().__init__(hp, False)
        self.hp = hp
        self.output_size = output_size
        self.net = BaseProcessingLSTM(hp, input_size, output_size)

    def build_network(self, input_size, hp):
        pass

    def forward(self, seq):
        sh = list(seq.shape)
        seq = seq.view(sh[:2] + [-1])
        proc_seq = self.run_net(seq)
        proc_seq = proc_seq.view(sh[:2] + [self.output_size] + sh[3:])
        return proc_seq


class LengthPredictorModule(nn.Module):
    """Predicts the length of a segment given start and goal image encoding of that segment."""
    def __init__(self, hp):
        super().__init__()
        self._hp = hp
        self.p = Predictor(hp, hp.nz_enc * 2, hp.max_seq_len)

    def forward(self, e0, eg):
        """Returns the logits of a OneHotCategorical distribution."""
        output = AttrDict()
        output.seq_len_logits = remove_spatial(self.p(e0, eg))
        output.seq_len_pred = OneHotCategorical(logits=output.seq_len_logits)
        
        return output
    
    def loss(self, inputs, model_output):
        losses = AttrDict()
        losses.len_pred = CELogitsLoss(self._hp.length_pred_weight)(model_output.seq_len_logits, inputs.end_ind)
        return losses


class ActionConditioningWrapper(nn.Module):
    def __init__(self, hp, net):
        super().__init__()
        self.net = net
        self.ac_net = Predictor(hp, hp.nz_enc + hp.n_actions, hp.nz_enc)

    def forward(self, input, actions):
        net_outputs = self.net(input)
        padded_actions = torch.nn.functional.pad(actions, (0, 0, 0, net_outputs.shape[1] - actions.shape[1], 0, 0))
        # TODO quite sure the concatenation is automatic
        net_outputs = batch_apply(self.ac_net, torch.cat([net_outputs, broadcast_final(padded_actions, input)], dim=2))
        return net_outputs
