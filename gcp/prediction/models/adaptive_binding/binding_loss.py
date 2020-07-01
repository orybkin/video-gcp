import numpy as np
import torch
import torch.nn as nn

from blox import AttrDict
from blox.tensor.ops import batchwise_index
from blox.torch.losses import L2Loss, PenaltyLoss
from blox.torch.ops import batch_cdist, like, list2ten
from gcp.prediction.utils.visualization import draw_frame
from gcp.prediction import global_params


class LossAveragingCriterion(nn.Module):
    def __init__(self, hp):
        self._loss = L2Loss
        self._hp = hp
        super().__init__()
    
    def loss(self, outputs, targets, weights, pad_mask, weight, log_sigma):
        predictions = outputs.tree.bf.images
        gt_match_dists = outputs.gt_match_dists
        
        # Compute likelihood
        loss_val = batch_cdist(predictions, targets, reduction='sum')
        
        log_sigmas = log_sigma - WeightsHacker.hack_weights(torch.ones_like(loss_val)).log()
        n = np.prod(predictions.shape[2:])
        loss_val = 0.5 * loss_val * torch.pow(torch.exp(-log_sigmas), 2) + n * (log_sigmas + 0.5 * np.log(2 * np.pi))
        
        # Weigh by matching probability
        match_weights = gt_match_dists
        match_weights = match_weights * pad_mask[:, None]  # Note, this is now unnecessary since both tree models handle it already
        loss_val = loss_val * match_weights * weights
        
        losses = AttrDict()
        losses.dense_img_rec = PenaltyLoss(weight, breakdown=2)(loss_val, log_error_arr=True, reduction=[-1, -2])
        
        # if self._hp.top_bias > 0.0:
        #     losses.n_top_bias_nodes = PenaltyLoss(
        #         self._hp.supervise_match_weight)(1 - WeightsHacker.get_n_top_bias_nodes(targets, weights))
        
        return losses

    def get_soft_estimates(self, gt_match_dists, vals):
        """ This function is only used to produce visualization now. Move it. """
    
        # soft_matched_estimates = torch.sum(add_n_dims(gt_match_dists, len(vals.shape)-2) * vals[:, :, None], dim=1)
        def soft_average(values):
            """ Averages per-node values to compute per-frame values """
        
            return torch.einsum('int, in...->it...', gt_match_dists, values).detach()
    
        soft_matched_estimates = soft_average(vals)
    
        # Mark top nodes
        if self._hp.use_convs:
            color = torch.zeros(vals.shape[:2]).to(vals.device)
            color[:, :3] = 0.5
            color_t = soft_average(color)
            soft_matched_estimates = draw_frame(soft_matched_estimates, color_t)
    
        return soft_matched_estimates


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
                m[:, 2 ** (depth - l - 1) - 1:: 2 ** (depth - l)] = global_params.hp.top_bias
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
