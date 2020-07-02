import numpy as np
import torch
import torch.nn as nn

from blox import batch_apply, AttrDict
from blox.torch.dist import normalize
from blox.torch.losses import BCELogitsLoss, PenaltyLoss
from blox.torch.modules import ExponentialDecayUpdater
from blox.torch.ops import batch_cdist, cdist
from blox.torch.subnetworks import Predictor
from gcp.prediction.models.adaptive_binding.probabilistic_dtw import soft_dtw
from gcp.prediction.models.tree.frame_binding import BaseBinding
from gcp.prediction.utils.tree_utils import depthfirst2breadthfirst
from gcp.prediction.models.adaptive_binding.binding_loss import LossAveragingCriterion, WeightsHacker


class AdaptiveBinding(BaseBinding):
    def build_network(self):
        self.temp = nn.Parameter(self._hp.matching_temp * torch.ones(1))
        if not self._hp.learn_matching_temp:
            self.temp.requires_grad_(False)

        if self._hp.matching_temp_tenthlife != -1:
            assert not self._hp.learn_matching_temp
            self.matching_temp_updater = ExponentialDecayUpdater(
                self.temp, self._hp.matching_temp_tenthlife, min_limit=self._hp.matching_temp_min)
            
        self.distance_predictor = Predictor(self._hp, self._hp.nz_enc * 2, 1, spatial=False)

        self.criterion = LossAveragingCriterion(self._hp)
    
    def get_w(self, pad_mask, inputs, model_output, log=False):
        """ Matches according to the dynamic programming-based posterior. """
        
        # TODO add a prior over w - this is currently trained as an AE.
        # (oleg) it seems that a uniform prior wouldn't change the computation
        # A prior on specific edges would change it somewhat similarly to weighting the cost (but not exactly with p)
    
        # Get cost matrix
        tree = model_output.tree

        if self._hp.matching_type == 'dtw_image':
            imgs = tree.df.images
            cost_matrix = batch_cdist(imgs, inputs.traj_seq, reduction='mean')
        elif self._hp.matching_type == 'dtw_latent':
            img_latents = tree.df.e_g_prime
            cost_matrix = batch_cdist(img_latents, inputs.enc_traj_seq, reduction='mean')
        
        # TODO remove the detachment to propagate the gradients!
        cost_matrix = WeightsHacker.hack_weights_df(cost_matrix)
        w_matrix = soft_dtw(cost_matrix.detach() / self.temp, inputs.end_ind)
        
        # TODO write this up
        # (oleg) There is some magic going on here. To define a likelihood, we define a mixture model for each frame
        # that consists of the nodes and the respective weights. We normalize the weights for it to be a distribution.
        # Then, we invoke Jensen's!
        # Since we expect all elements in the mixture to be either x or have zero probability, the bound is tight.
        w_matrix = normalize(w_matrix, 1)
        
        return depthfirst2breadthfirst(w_matrix)
    
    def prune_sequence(self, inputs, outputs, key='images'):
        seq = getattr(outputs.tree.df, key)
        latent_seq = outputs.tree.df.e_g_prime
        
        distances = batch_apply(self.distance_predictor,
                                latent_seq[:, :-1].contiguous(), latent_seq[:, 1:].contiguous())[..., 0]
        outputs.distance_predictor = AttrDict(distances=distances)

        # distance_predictor outputs true if the two frames are too close
        close_frames = torch.sigmoid(distances) > self._hp.learned_pruning_threshold
        # Add a placeholder for the first frame
        close_frames = torch.cat([torch.zeros_like(close_frames[:, [0]]), close_frames], 1)
        
        pruned_seq = [seq[i][~close_frames[i]] for i in range(seq.shape[0])]
        
        return pruned_seq
    
    def loss(self, inputs, outputs):
        losses = super().loss(inputs, outputs)

        if self._hp.top_bias != 1.0 and WeightsHacker.can_get_index():
            losses.n_top_bias_nodes = PenaltyLoss(self._hp.supervise_match_weight) \
                (1 - WeightsHacker.get_n_top_bias_nodes(inputs.traj_seq, outputs.tree.bf.match_dist))

        if WeightsHacker.can_get_d2b(inputs):
            dist = WeightsHacker.distance2bottleneck(inputs, outputs)
            losses.distance_to_bottleneck_1 = PenaltyLoss(0)(dist[0])
            losses.distance_to_bottleneck_2 = PenaltyLoss(0)(dist[1])
            losses.distance_to_bottleneck_3 = PenaltyLoss(0)(dist[2])

        if self._hp.log_d2b_3x3maze:
            top_nodes = outputs.tree.bf.images[:, :self._hp.n_top_bias_nodes].reshape(-1, 2)
    
            def get_bottleneck_states():
                if inputs.traj_seq_states.max() > 13.5:
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
            
        if 'distance_predictor' in outputs:
            df_match_dists = outputs.tree.df.match_dist
            best_matching = df_match_dists.argmax(-1)
            targets = best_matching[:, 1:] == best_matching[:, :-1]  # 1 if frames are too close, i.e. if best matching gt is the same
            losses.distance_predictor = BCELogitsLoss()(outputs.distance_predictor.distances, targets.float())
        
        return losses

    def reconstruction_loss(self, inputs, outputs, weights):
        losses = AttrDict()
    
        outputs.soft_matched_estimates = self.criterion.get_soft_estimates(outputs.gt_match_dists,
                                                                           outputs.tree.bf.images)
        losses.update(self.criterion.loss(
            outputs, inputs.traj_seq, weights, inputs.pad_mask, self._hp.dense_img_rec_weight, self.decoder.log_sigma))
    
        return losses
