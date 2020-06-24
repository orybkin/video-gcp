import numpy as np
import torch
from blox import AttrDict
from blox.basic_types import listdict2dictlist

from gcp.models.cost_mdl import TestTimeCostModel


class CostFcn:
    """Base class to define CEM cost functions."""
    def __init__(self, dense_cost, final_step_weight=1.0, *unused_args):
        self._dense_cost = dense_cost
        self._final_step_weight = final_step_weight

    def __call__(self, cem_outputs, goal):
        cost_per_step = self._compute(cem_outputs, goal)
        for i in range(len(cost_per_step)):
            cost_per_step[i][-1] *= self._final_step_weight
        if self._dense_cost:
            return np.array([np.sum(c) for c in cost_per_step])
        else:
            return np.array([c[-1] for c in cost_per_step])

    def _compute(self, cem_outputs, goal):
        raise NotImplementedError


class ImageCost:
    """Provides method to split off image and latent sequence from input sequence."""
    def _split_state_rollout(self, rollouts):
        """Splits off latents from states in joined rollouts."""
        def reshape_to_image(flat):
            if len(flat.shape) != 2:
                import pdb; pdb.set_trace()
            assert len(flat.shape) == 2
            res = int(np.sqrt(flat.shape[1] / 3))   # assumes 3-channel image
            return flat.reshape(flat.shape[0], 3, res, res)
        return listdict2dictlist([AttrDict(image_rollout=reshape_to_image(r[..., :-self.input_dim]),
                                           latent_rollout=r[..., -self.input_dim:]) for r in rollouts])


class EuclideanDistance(CostFcn):
    """Euclidean distance between vals and goal."""
    def _compute(self, cem_outputs, goal):
        euclid_dists = [np.linalg.norm(cem_output - goal[None], axis=-1) for cem_output in cem_outputs]
        return euclid_dists


class EuclideanPathLength(CostFcn):
    """Euclidean length of the whole path to the goal."""
    def _compute(self, cem_outputs, goal):
        assert self._dense_cost     # need dense cost for path length computation
        return [np.linalg.norm(np.concatenate([cem_output[1:], goal[None]]) - cem_output, axis=-1)
                        for cem_output in cem_outputs]


class StepPathLength(CostFcn):
    """Cost is equivalent to number of steps in path."""
    def _compute(self, cem_outputs, goal):
        path_lengths = [cem_output.shape[0] for cem_output in cem_outputs]
        return [np.concatenate((np.zeros(cem_output.shape[0]-1), np.array([path_length]))) 
                        for cem_output, path_length in zip(cem_outputs, path_lengths)]


class L2ImageCost(CostFcn, ImageCost):
    """Cost is equivalent to L2 distance in image space."""
    LATENT_SIZE = 128        # TODO: make this configurable

    def _compute(self, cem_outputs, goal_raw):
        image_sequences = self._split_state_rollout(cem_outputs).image_rollout
        goal = goal_raw.transpose(0, 3, 1, 2) * 2 - 1.0
        return [np.sqrt(np.sum((seq - goal)**2, axis=(1, 2, 3))) for seq in image_sequences]

    @property
    def input_dim(self):
        return self.LATENT_SIZE


class LearnedCostEstimate:
    """Uses learned network to estimate cost between to latent states."""
    def __init__(self, config):
        self.net = TestTimeCostModel(params=config, logger=None)

    def __call__(self, start_enc, goal_enc):
        if isinstance(start_enc, np.ndarray):
            # compute cost for single start goal pair
            return self.net(AttrDict(enc1=start_enc, enc2=goal_enc)).data.cpu().numpy()
        elif isinstance(start_enc, list):
            # compute summed cost for sequence
            costs = []
            for seq, goal in zip(start_enc, goal_enc):
                seq_input = torch.cat((torch.tensor(seq).to(self.net.device), torch.tensor(goal).to(self.net.device)))
                cost_per_step = self.net(AttrDict(enc1=seq_input[:-1], enc2=seq_input[1:]))
                costs.append(cost_per_step.sum().data.cpu().numpy())
            return np.array(costs)
        else:
            raise ValueError("Dimensionality of input to learned cost function not supported!")

    @property
    def input_dim(self):
        return self.net.input_dim


class ImageLearnedCostEstimate(LearnedCostEstimate, ImageCost):
    pass


class ImageWrappedLearnedCostFcn(LearnedCostEstimate, ImageCost):
    """Shallow wrapper around LearnedCostEstimate that unpacks image input."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, start_enc, goal_enc):
        start_enc = self._split_state_rollout(start_enc).latent_rollout
        goal_enc = [start_enc[-1] for _ in range(len(start_enc))]   # HACK that only works for goal-cond prediction!
        return super().__call__(start_enc, goal_enc)
