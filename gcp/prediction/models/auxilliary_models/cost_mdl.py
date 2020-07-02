from contextlib import contextmanager

import numpy as np
import torch

from blox import AttrDict
from blox.tensor.ops import batchwise_index
from blox.torch.losses import L2Loss
from blox.torch.subnetworks import Predictor
from gcp.prediction.models.auxilliary_models.base_model import BaseModel
from gcp.prediction.training.checkpoint_handler import CheckpointHandler


class CostModel(BaseModel):
    def __init__(self, params, logger):
        super().__init__(logger)
        self._hp = self._default_hparams()
        params.update({'use_convs': False})
        self.override_defaults(params)  # override defaults with config file
        self.postprocess_params()
        self.build_network()

        if self._hp.cost_fcn is not None:
            self._gt_cost_fcn = self._hp.cost_fcn(True)

    def _default_hparams(self):
        # put new parameters in here:
        default_dict = {
            'nz_enc': 128,  # number of dimensions in encoder-latent space
            'nz_mid': 128,  # number of hidden units in fully connected layer
            'n_processing_layers': 3,  # Number of layers in MLPs
            'checkpt_path': None,
            'load_epoch': None,  # checkpoint epoch which should be loaded, if 'none' loads latest
            'cost_fcn': None,   # holds ground truth cost function
            'use_path_dist_cost': False,   # if True, uses fast path distance cost computation, ignores cost_fcn
        }

        # misc params
        default_dict.update({
            'use_skips': False,
            'dense_rec_type': None,
            'device': None,
            'randomize_length': False,
        })

        # add new params to parent params
        parent_params = super()._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params
    
    def build_network(self, build_encoder=True):
        self.cost_pred = Predictor(self._hp, self._hp.nz_enc * 2, 1, detached=True)

    def forward(self, inputs):
        """Forward pass at training time."""
        # randomly sample start and end state, compute GT cost
        start, end, gt_cost = self._fast_path_dist_cost(inputs) if self._hp.use_path_dist_cost \
                                else self._general_cost(inputs)

        # compute cost estimate
        cost_pred = self.cost_pred(torch.cat([start, end], dim=-1))

        output = AttrDict(
            cost=cost_pred,
            cost_target=gt_cost,
        )
        return output

    def loss(self, inputs, outputs, add_total=True):
        return AttrDict(
            cost_estimation=L2Loss(1.0)(outputs.cost, outputs.cost_target)
        )

    def log_outputs(self, outputs, inputs, losses, step, log_images, phase):
        super()._log_losses(losses, step, log_images, phase)

    @contextmanager
    def val_mode(self):
        yield

    def _fast_path_dist_cost(self, inputs):
        """Vectorized computation of path distance cost."""
        # sample start goal indices
        batch_size = inputs.end_ind.shape[0]
        start_idx = torch.rand((batch_size,), device=inputs.end_ind.device) * (inputs.end_ind.float() - 1)
        end_idx = torch.rand((batch_size,), device=inputs.end_ind.device) * (inputs.end_ind.float() - (start_idx + 1)) + (start_idx + 1)
        start_idx, end_idx = start_idx.long(), end_idx.long()

        # get start goal latents
        start = batchwise_index(inputs.model_enc_seq, start_idx).detach()
        end = batchwise_index(inputs.model_enc_seq, end_idx).detach()

        # compute path distance cost
        cum_diff = torch.cumsum(torch.norm(inputs.traj_seq[:, 1:] - inputs.traj_seq[:, :-1], dim=-1), dim=1)
        cum_diff = torch.cat((torch.zeros((batch_size, 1), dtype=cum_diff.dtype, device=cum_diff.device),
                              cum_diff), dim=1)    # prepend 0
        gt_cost = batchwise_index(cum_diff, end_idx) - batchwise_index(cum_diff, start_idx)

        return start, end, gt_cost[:, None].detach()

    def _general_cost(self, inputs):
        """Computes cost with generic cost function, not vectorized."""
        batch_size = inputs.end_ind.shape[0]

        start, end, gt_cost = [], [], []
        for b in range(batch_size):
            start_idx = np.random.randint(0, inputs.end_ind[b].cpu().numpy(), 1)[0]
            end_idx = np.random.randint(start_idx + 1, inputs.end_ind[b].cpu().numpy() + 1, 1)[0]
            start.append(inputs.model_enc_seq[b, start_idx])
            end.append(inputs.model_enc_seq[b, end_idx])
            gt_cost.append(self._gt_cost_fcn([inputs.traj_seq[b, start_idx:end_idx+1].data.cpu().numpy()],
                                             inputs.traj_seq[b, end_idx].data.cpu().numpy()))
        start, end = torch.stack(start).detach(), torch.stack(end).detach()  # no gradients in encoder
        gt_cost = torch.tensor(np.stack(gt_cost), device=start.device, requires_grad=False).float()
        return start, end, gt_cost

    @property
    def input_dim(self):
        return self._hp.nz_enc


class TestTimeCostModel(CostModel):
    def __init__(self, params, logger):
        super().__init__(params, logger)
        if torch.cuda.is_available():
            self.cuda()
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        assert self._hp.checkpt_path is not None
        load_epoch = self._hp.load_epoch if self._hp.load_epoch is not None else 'latest'
        weights_file = CheckpointHandler.get_resume_ckpt_file(load_epoch, self._hp.checkpt_path)
        success = CheckpointHandler.load_weights(weights_file, self, submodule_name='cost_mdl')
        if not success: raise ValueError("Could not load checkpoint from {}!".format(weights_file))

    def forward(self, inputs):
        for k in inputs:
            if not isinstance(inputs[k], torch.Tensor):
                inputs[k] = torch.Tensor(inputs[k])
            if not inputs[k].device == self.device:
                inputs[k] = inputs[k].to(self.device) 

        return self.cost_pred(inputs['enc1'], inputs['enc2'])

