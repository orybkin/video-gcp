import torch
import torch.nn as nn

from blox import AttrDict
from blox.torch.core import ar2ten
from blox.torch.losses import L2Loss
from blox.torch.ops import like
from gcp.rec_planner_utils.matching import TemperatureMatcher


class FractionMatcher(TemperatureMatcher):
    def __call__(self, inputs, subgoal, left_parent, right_parent):
        super().build_network()
        
        timesteps = self.comp_timestep(left_parent.timesteps, right_parent.timesteps, subgoal.fraction)
        return AttrDict(timesteps=timesteps)

    @staticmethod
    def comp_timestep(t_l, t_r, frac):
        return t_l + (t_r - t_l) * frac

    def get_w(self, pad_mask, inputs, model_output, log=False):
        """ Match according to the fraction """
        self.apply_tree(model_output.tree, inputs)
        timesteps = model_output.tree.bf.timesteps
        
        gt_timesteps = like(torch.arange, timesteps)(pad_mask.shape[1])[None, None]
        scaled_dists = (gt_timesteps - timesteps[..., None])**2 / self.temp
        if self._hp.leaf_nodes_only:        # only use leaf nodes -> set distance for all other to inf
            scaled_dists[:, :2**(self._hp.hierarchy_levels - 1) - 1] = float("Inf")
        gt_match_dists = nn.Softmax(dim=1)(-scaled_dists)
        return gt_match_dists

    def loss(self, inputs, model_output):
        losses = super().loss(inputs, model_output)
        tree = model_output.tree

        if self._hp.supervise_fraction_weight > 0.0:
            key_frac = ar2ten(self.criterion.get_index(inputs.demo_seq), self._hp.device).float() / (
            self._hp.max_seq_len - 1)
    
            weights = 1.0
            if self._hp.supervise_fraction == 'top_index':
                estimates = tree.bf.fraction[:, 0]
                targets = key_frac.float()
            elif self._hp.supervise_fraction == 'balanced':
                estimates = tree.bf.fraction
                targets = torch.ones_like(estimates) * 0.5
            elif self._hp.supervise_fraction == 'index+balanced':
                estimates = tree.bf.fraction
                targets = torch.cat([key_frac.float()[:, None], torch.ones_like(estimates[:, 1:]) * 0.5], 1)
        
                weights = targets.new_ones([1, targets.shape[1]])
                weights[:, 0] = self._hp.max_seq_len
    
            losses.supervise_fraction = L2Loss(self._hp.supervise_fraction_weight)(estimates, targets, weights=weights)

        return losses
