import numpy as np
import torch
import torch.nn as nn

from blox import AttrDict, batch_apply2, rmap
from blox.tensor.ops import batchwise_index
from blox.torch.losses import PenaltyLoss
from blox.torch.modules import DummyModule, ExponentialDecayUpdater
from blox.torch.ops import cdist
from blox.torch.ops import like, ten2ar, list2ten
from gcp.rec_planner_utils import global_params


class BaseMatcher(DummyModule):
    def __init__(self, hp, criterion=None, decoder=None):
        super().__init__()
        self._hp = hp
        self.criterion = criterion
        self.decoder = decoder

        self.build_network()

    def get_init_inds(self, inputs):
        return inputs.start_ind.float()[:, None], inputs.end_ind.float()[:, None]

    def apply_tree(self, tree, inputs):
        # recursive_add_dim = make_recursive(lambda x: add_n_dims(x, n=1, dim=1))
        start_ind, end_ind = self.get_init_inds(inputs)
        tree.apply_fn(
            {}, fn=self, left_parents=AttrDict(timesteps=start_ind), right_parents=AttrDict(timesteps=end_ind))

    def loss(self, inputs, model_output):
        # TODO this should be somewhere else
        losses = AttrDict()

        if self._hp.top_bias != 1.0 and WeightsHacker.can_get_index():
            losses.n_top_bias_nodes = PenaltyLoss(self._hp.supervise_match_weight) \
                (1 - WeightsHacker.get_n_top_bias_nodes(inputs.demo_seq, model_output.tree.bf.match_dist))

        if WeightsHacker.can_get_d2b(inputs):
            dist = WeightsHacker.distance2bottleneck(inputs, model_output)
            losses.distance_to_bottleneck_1 = PenaltyLoss(0)(dist[0])
            losses.distance_to_bottleneck_2 = PenaltyLoss(0)(dist[1])
            losses.distance_to_bottleneck_3 = PenaltyLoss(0)(dist[2])

        if self._hp.log_d2b_3x3maze:
            top_nodes = model_output.tree.bf.images[:, :self._hp.n_top_bias_nodes].reshape(-1, 2)
    
            def get_bottleneck_states():
                if inputs.demo_seq_states.max() > 13.5:
                    scale = 45  # 5x5 maze
                else:
                    scale = 27  # 3x3 maze
                start = -0.5 * scale
                end = 0.5 * scale
                
                doors_x = torch.linspace(start, end, self._hp.log_d2b_3x3maze + 1).to(self._hp.device)[1:-1]
                doors_y = torch.linspace(start, end, self._hp.log_d2b_3x3maze * 2 + 1).to(self._hp.device)[1:-1:2]
                n_x = doors_x.shape[0]
                n_y = doors_y.shape[0]
                doors_x = doors_x.repeat(n_y)
                doors_y = doors_y.repeat_interleave(n_x)
                doors = torch.stack([doors_x, doors_y], 1)
                # And the other way around
                return torch.cat([doors, doors.flip(1)], 0)
    
            doors = get_bottleneck_states()
            dist = cdist(top_nodes, doors)
            avg_dist = dist.min(-1).values.mean()
    
            losses.distance_to_doors = PenaltyLoss(self._hp.supervise_match_weight)(avg_dist)

        return losses
    
    def reconstruction_loss(self, inputs, outputs, weights):
        losses = AttrDict()
        
        outputs.soft_matched_estimates = self.criterion.get_soft_estimates(outputs.gt_match_dists, outputs.tree.bf.images)
        losses.update(self.criterion.loss(
            outputs, inputs.demo_seq, weights, inputs.pad_mask, self._hp.dense_img_rec_weight, self.decoder.log_sigma))
        
        return losses
    
    def get_matched_sequence(self, tree, key):
        latents = tree.bf[key]
        indices = tree.bf.match_dist.argmax(1)
        # Two-dimensional indexing
        matched_sequence = rmap(lambda x: batchwise_index(x, indices), latents)
    
        return matched_sequence
    
    
class TemperatureMatcher(BaseMatcher):
    """ A simple class that creates a temperature parameter for Matchers """
    def build_network(self):
        self.temp = nn.Parameter(self._hp.matching_temp * torch.ones(1))
        if not self._hp.learn_matching_temp:
            self.temp.requires_grad_(False)

        if self._hp.matching_temp_tenthlife != -1:
            assert not self._hp.learn_matching_temp
            self.matching_temp_updater = ExponentialDecayUpdater(
                self.temp, self._hp.matching_temp_tenthlife, min_limit=self._hp.matching_temp_min)


class WeightsHacker():
    @staticmethod
    def hack_weights(weights):
        if abs(global_params.hp.leaves_bias) > 0.0:
            w_1 = weights[:, :-global_params.hp.max_seq_len]
            w_2 = weights[:, -global_params.hp.max_seq_len:] * (1 - global_params.hp.leaves_bias)
            weights = torch.cat([w_1, w_2], 1)
        
        if global_params.hp.top_bias != 1.0:
            w_1 = weights[:, :global_params.hp.n_top_bias_nodes] * global_params.hp.top_bias
            w_2 = weights[:, global_params.hp.n_top_bias_nodes:]
            weights = torch.cat([w_1, w_2], 1)
        
        return weights

    @staticmethod
    def hack_weights_df(weights):
        # TODO implement bf2df for indices and use here
        
        if global_params.hp.top_bias != 1.0:
            n_top_bias_layers = np.int(np.log2(global_params.hp.n_top_bias_nodes + 1))
            depth = np.int(np.log2(weights.shape[1] + 1))
            m = torch.ones(weights.shape[:2], device=weights.device)
            for l in range(n_top_bias_layers):
                m[:, 2**(depth - l - 1) - 1 :: 2**(depth - l)] = global_params.hp.top_bias
            weights = weights * m[:, :, None]
        return weights
    
    @staticmethod
    def get_n_top_bias_nodes(targets, weights):
        """ Return the probability that the downweighted nodes match the noisy frame"""
        inds = WeightsHacker.get_index(targets)
        
        noise_frames = batchwise_index(weights, inds, 2)
        n = noise_frames.mean(0)[:global_params.hp.n_top_bias_nodes].sum() / global_params.hp.top_bias
        
        return n
    
    @staticmethod
    def can_get_index():
        return 'dataset_conf' in global_params.data_conf and 'dataset_class' in global_params.data_conf.dataset_conf \
              and global_params.data_conf.dataset_conf.dataset_class == PointMassDataset

    @staticmethod
    def can_get_d2b(inputs):
        if 'actions' in inputs and inputs.actions.shape[2] == 3 and \
              torch.equal(torch.unique(inputs.actions[:, :, 2]), like(list2ten, inputs.actions)([-1, 1])):
            # Looks like sawyer
            return True
        else:
            return False

    @staticmethod
    def distance2bottleneck(inputs, outputs):
        dists = []
        for i in range(inputs.actions.shape[0]):
            gripper = inputs.actions[i, :, -1]
            picks = (gripper[1:] == gripper[:-1] + 2).nonzero()[:, 0]
            places = (gripper[1:] == gripper[:-1] - 2).nonzero()[:, 0]
            bottlenecks = torch.cat([picks, places], -1)
            
            top_inds = outputs.tree.bf.match_dist[i, :3].argmax(-1)
            # top_inds = torch.from_numpy(np.asarray([20, 40, 60])).long().cuda()
            
            def closest_point_distance(a, b):
                mat = torch.abs(a[:, None] - b[None, :])
                ind = mat.argmin(-1)
                return torch.abs(a - b[ind])
            
            dist = closest_point_distance(top_inds, bottlenecks)
            dists.append(dist)
            
        return torch.mean(torch.stack(dists).float(), 0)
    

