import torch.nn as nn
import torch
import torch.nn.functional as F

from blox import AttrDict
from blox.torch.ops import make_one_hot, apply_linear
from blox.torch.subnetworks import Predictor, MultiheadAttention
from gcp.prediction.models.tree.inference import Inference


class AttentiveInference(Inference):
    def __init__(self, hp, q):
        super().__init__(hp, q)
        self.attention = Attention(hp)
    
    def forward(self, inputs, e_l, e_r, start_ind, end_ind, timestep=None):
        assert timestep is None
        
        output = AttrDict()
        if self.deterministic:
            output.q_z = self.q(e_l)
            return output
        
        values = inputs.inf_enc_seq
        keys = inputs.inf_enc_key_seq
        
        # Get (initial) attention key
        query_input = [e_l, e_r]
        
        e_tilde, output.gamma = self.attention(values, keys, query_input, start_ind, end_ind, inputs)
        output.q_z = self.q(e_l, e_r, e_tilde)
        return output


class Attention(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self._hp = hp
        time_cond_length = self._hp.max_seq_len if self._hp.one_hot_attn_time_cond else 1
        input_size = hp.nz_enc * 2
        self.query_net = Predictor(hp, input_size, hp.nz_attn_key)
        self.attention_layers = nn.ModuleList([MultiheadAttention(hp) for _ in range(hp.n_attention_layers)])
        self.predictor_layers = nn.ModuleList([Predictor(hp, hp.nz_enc, hp.nz_attn_key, num_layers=2)
                                               for _ in range(hp.n_attention_layers)])
        self.out = nn.Linear(hp.nz_enc, hp.nz_enc)

    def forward(self, values, keys, query_input, start_ind, end_ind, inputs):
        """
        Performs multi-layered, multi-headed attention.

        Note: the query can have a different batch size from the values/keys. In that case, the query is interpreted as
        multiple queries, i.e. the values are tiled to match the query tensor size.
        
        :param values: tensor batch x length x dim_v
        :param keys: tensor batch x length x dim_k
        :param query_input: input to the query network, batch2 x dim_k
        :param start_ind:
        :param end_ind:
        :param inputs:
        :param timestep: specify the timestep of the attention directly. tensor batch2 x 1
        :param attention_weights:
        :return:
        """

        query = self.query_net(*query_input)
        s_ind, e_ind = (torch.floor(start_ind), torch.ceil(end_ind)) if self._hp.mask_inf_attention \
                                                                     else (inputs.start_ind, inputs.end_ind)
        
        # Reshape values, keys, inputs if not enough dimensions
        mult = int(query.shape[0] / keys.shape[0])
        tile = lambda x: x[:, None][:, [0] * mult].reshape((-1,) + x.shape[1:])
        values = tile(values)
        keys = tile(keys)
        s_ind = tile(s_ind)
        e_ind = tile(e_ind)
        
        # Attend
        norm_shape_k = query.shape[1:]
        norm_shape_v = values.shape[2:]
        raw_attn_output, att_weights = None, None
        for attention, predictor in zip(self.attention_layers, self.predictor_layers):
            raw_attn_output, att_weights = attention(query, keys, values, s_ind, e_ind)
            x = F.layer_norm(raw_attn_output, norm_shape_v)
            query = F.layer_norm(predictor(x) + query, norm_shape_k)  # skip connections around attention and predictor

        return apply_linear(self.out, raw_attn_output, dim=1), att_weights     # output non-normalized output of final attention layer
