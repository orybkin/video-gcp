import torch

from blox import AttrDict, batch_apply
from blox.torch.ops import make_one_hot, broadcast_final
from blox.torch.subnetworks import Predictor
from blox.torch.losses import BCELogitsLoss

from gcp.rec_planner_utils.matching import BaseMatcher


class BalancedMatcher(BaseMatcher):
    def build_network(self):
        self.existence_predictor = Predictor(self._hp, self._hp.nz_enc, 1, spatial=False)
    
    def __call__(self, inputs, subgoal, left_parent, right_parent):
        timesteps = self.comp_timestep(left_parent.timesteps, right_parent.timesteps)
        c_n_prime = make_one_hot(timesteps.long(), self._hp.max_seq_len)

        # TODO implement the alternative of not doing this. Then would need to renormalize
        c_n_prime[left_parent.timesteps.long() == timesteps, :] = 0
        c_n_prime[right_parent.timesteps.long() == timesteps, :] = 0
        
        return AttrDict(timesteps=timesteps.long(), c_n_prime=c_n_prime.float())

    @staticmethod
    def comp_timestep(t_l, t_r, *unused_args):
        return (t_l + t_r) / 2
    
    def get_w(self, pad_mask, inputs, model_output, log=False):
        """ Match to the middle frame between parents """
        self.apply_tree(model_output.tree, inputs)
        match_dists = model_output.tree.bf.c_n_prime
        
        # Leaves-only
        if self._hp.leaf_nodes_only:
            n_leafs = 2**(self._hp.hierarchy_levels - 1)
            max_seq_len = match_dists.shape[-1]
            assert n_leafs >= max_seq_len     # need at least as many leafs as frames in sequence
            match_dists = torch.zeros_like(match_dists)
            end_ind = torch.argmax(pad_mask * torch.arange(max_seq_len, dtype=torch.float, device=pad_mask.device), 1)
            for b in range(match_dists.shape[0]):
                leaf_idxs = torch.linspace(0, n_leafs-1, end_ind[b]+1, device=match_dists.device).long()
                one_hot_leaf_idxs = make_one_hot(leaf_idxs, n_leafs)
                match_dists[b, -n_leafs:, :end_ind[b]+1] = one_hot_leaf_idxs.transpose(0, 1)
        return match_dists

    def get_init_inds(self, model_output):
        # TODO this actually mostly gets passed 'inputs'
        return torch.zeros_like(model_output.end_ind[:, None]) - 1, \
               model_output.end_ind[:, None] + 1

    def prune_sequence(self, inputs, outputs, key='images'):
        seq = getattr(outputs.tree.df, key)
        latent_seq = outputs.tree.df.e_g_prime
    
        existence = batch_apply(latent_seq, self.existence_predictor)[..., 0]
        outputs.existence_predictor = AttrDict(existence=existence)
    
        existing_frames = torch.sigmoid(existence) > 0.5
        pruned_seq = [seq[i][existing_frames[i]] for i in range(seq.shape[0])]
    
        return pruned_seq

    def loss(self, inputs, outputs):
        losses = super().loss(inputs, outputs)
    
        if 'existence_predictor' in outputs:
            losses.existence_predictor = BCELogitsLoss()(
                outputs.existence_predictor.existence, outputs.tree.df.match_dist.sum(2).float())
        return losses

    def reconstruction_loss(self, inputs, outputs, weights):
        """ Balanced tree can have a simpler loss version which doesn't use cdist """
        tree = outputs.tree
    
        estimates = self.get_matched_sequence(tree, 'distr')
        outputs.soft_matched_estimates = self.get_matched_sequence(tree, 'images')
        targets = inputs.demo_seq

        weights = broadcast_final(weights * inputs.pad_mask, targets)
        losses = self.decoder.nll(estimates, targets, weights, log_error_arr=True)
    
        return losses
