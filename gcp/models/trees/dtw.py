import torch

from blox import batch_apply, AttrDict
from blox.torch.dist import normalize
from blox.torch.losses import BCELogitsLoss
from blox.torch.ops import batch_cdist
from blox.torch.subnetworks import Predictor
from gcp.rec_planner_utils.matching import TemperatureMatcher, WeightsHacker
from gcp.rec_planner_utils.recplan_dtw import soft_dtw
from gcp.rec_planner_utils.tree_utils import depthfirst2breadthfirst


class DTWMatcher(TemperatureMatcher):
    def build_network(self):
        super().build_network()
        
        self.distance_predictor = Predictor(self._hp, self._hp.nz_enc * 2, 1, spatial=False)
    
    def get_w(self, pad_mask, inputs, model_output, log=False):
        """ Matches according to the dynamic programming-based posterior. """
        
        # TODO add a prior over w - this is currently trained as an AE.
        # (oleg) it seems that a uniform prior wouldn't change the computation
        # A prior on specific edges would change it somewhat similarly to weighting the cost (but not exactly with p)
    
        # Get cost matrix
        tree = model_output.tree

        if self._hp.matching_type == 'dtw_image':
            imgs = tree.get_leaf_nodes().images if self._hp.leaf_nodes_only else tree.df.images
            cost_matrix = batch_cdist(imgs, inputs.demo_seq, reduction='mean')
        elif self._hp.matching_type == 'dtw_latent':
            img_latents = tree.get_leaf_nodes().e_g_prime if self._hp.leaf_nodes_only else tree.df.e_g_prime
            cost_matrix = batch_cdist(img_latents, inputs.enc_demo_seq, reduction='mean')
        
        # TODO remove the detachment to propagate the gradients!
        cost_matrix = WeightsHacker.hack_weights_df(cost_matrix)
        w_matrix = soft_dtw(cost_matrix.detach() / self.temp, inputs.end_ind)
        
        # TODO write this up
        # (oleg) There is some magic going on here. To define a likelihood, we define a mixture model for each frame
        # that consists of the nodes and the respective weights. We normalize the weights for it to be a distribution.
        # Then, we invoke Jensen's!
        # Since we expect all elements in the mixture to be either x or have zero probability, the bound is tight.
        w_matrix = normalize(w_matrix, 1)

        # fill in complete weight matrix if DTW was over leaf nodes only
        if self._hp.leaf_nodes_only:
            w_part = w_matrix
            w_matrix = torch.zeros((w_matrix.shape[0], tree.size,) + w_matrix.shape[2:], device=w_matrix.device)
            w_matrix[:, ::2] = w_part
        
        return depthfirst2breadthfirst(w_matrix)
    
    def prune_sequence(self, inputs, outputs, key='images'):
        seq = getattr(outputs.tree.df, key)
        latent_seq = outputs.tree.df.e_g_prime
        
        distances = batch_apply([latent_seq[:, :-1].contiguous(), latent_seq[:, 1:].contiguous()],
                                self.distance_predictor, separate_arguments=True)[..., 0]
        outputs.distance_predictor = AttrDict(distances=distances)

        # distance_predictor outputs true if the two frames are too close
        close_frames = torch.sigmoid(distances) > self._hp.learned_pruning_threshold
        # Add a placeholder for the first frame
        close_frames = torch.cat([torch.zeros_like(close_frames[:, [0]]), close_frames], 1)
        
        pruned_seq = [seq[i][~close_frames[i]] for i in range(seq.shape[0])]
        
        return pruned_seq
    
    def loss(self, inputs, outputs):
        losses = super().loss(inputs, outputs)
        
        if 'distance_predictor' in outputs:
            df_match_dists = outputs.tree.df.match_dist
            best_matching = df_match_dists.argmax(-1)
            targets = best_matching[:, 1:] == best_matching[:, :-1]  # 1 if frames are too close, i.e. if best matching gt is the same
            losses.distance_predictor = BCELogitsLoss()(outputs.distance_predictor.distances, targets.float())
        return losses

