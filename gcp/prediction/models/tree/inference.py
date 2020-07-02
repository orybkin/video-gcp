import torch.nn as nn

from blox import AttrDict
from blox.tensor.ops import batchwise_index
from blox.torch.losses import KLDivLoss2
from blox.torch.variational import FixedPrior


class Inference(nn.Module):
    def __init__(self, hp, q):
        super().__init__()
        self._hp = hp
        self.q = q
        self.deterministic = isinstance(self.q, FixedPrior)
    
    def forward(self, inputs, e_l, e_r, start_ind, end_ind, timestep):
        assert timestep is not None
        output = AttrDict(gamma=None)
        
        if self.deterministic:
            output.q_z = self.q(e_l)
            return output
        
        values = inputs.inf_enc_seq
        keys = inputs.inf_enc_key_seq
        
        mult = int(timestep.shape[0] / keys.shape[0])
        if mult > 1:
            timestep = timestep.reshape(-1, mult)
            result = batchwise_index(values, timestep.long())
            e_tilde = result.reshape([-1] + list(result.shape[2:]))
        else:
            e_tilde = batchwise_index(values, timestep[:, 0].long())
        
        output.q_z = self.q(e_l, e_r, e_tilde)
        return output
    
    def loss(self, q_z, p_z, weights=1):
        if q_z.mu.numel() == 0:
            return {}
        
        return AttrDict(kl=KLDivLoss2(self._hp.kl_weight, breakdown=1, free_nats_per_dim=self._hp.free_nats)(
            q_z, p_z, weights=weights, log_error_arr=True))
