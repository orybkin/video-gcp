import torch
import torch.nn as nn
import numpy as np

from blox.torch.ops import batch_cdist, broadcast_final, batchwise_assign
from blox import AttrDict
from blox.torch.losses import L2Loss, PenaltyLoss

from gcp.rec_planner_utils.matching import WeightsHacker
from gcp.rec_planner_utils.vis_utils import draw_frame


class FramesAveragingCriterion(nn.Module):
    def __init__(self, hp):
        self._loss = L2Loss
        self._hp = hp
        super().__init__()
    
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
    
    def loss(self, matcher_output, targets, pad_mask, weight, log_sigma):
        return self._loss(weight)(matcher_output.soft_matched_est, targets, weights=broadcast_final(pad_mask, targets),
                                  log_error_arr=True)


class LossAveragingCriterion(FramesAveragingCriterion):
    def __init__(self, hp):
        super().__init__(hp)
    
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


class ExpectationCriterion(LossAveragingCriterion):
    def _loss(self, estimates, targets, weights):
        loss = torch.exp(-torch.nn.MSELoss(reduction='none')(estimates, targets))
        return -torch.log(torch.mean(loss * weights))


class LossFramesAveragingCriterion(LossAveragingCriterion):
    def loss(self, matcher_output, targets, pad_mask):
        return (LossAveragingCriterion.loss(self, matcher_output, targets, pad_mask) +
                FramesAveragingCriterion.loss(self, matcher_output, targets, pad_mask)) / 2.0


def get_n_OOO(tree):
    raise NotImplementedError("Deprecated! Not updated to layer-wise tree computation!")
    
    def prob_ooo_right(right_child, parent):
        prob_left = torch.cumsum(right_child,
                                 dim=1)  # The number of frames that the child covers to the left of current
        
        pad = torch.zeros(parent.shape[0], 1, device=parent.device)
        n_w_shifted = torch.cat((pad, parent[:, 1:]), dim=1)  # Number of frames parent covers at each position
        
        return ((prob_left * n_w_shifted).sum(dim=1)).mean().item()
    
    prob_ooo = 0
    for node in tree:
        if node.depth > 1:
            s0 = node.s0.subgoal.match_dist
            s1 = node.s0.subgoal.match_dist
            n = node.subgoal.match_dist
            prob_ooo += prob_ooo_right(s1, n) + prob_ooo_right(s0.flip(dims=[1]), n.flip(dims=[1]))
    
    return prob_ooo
