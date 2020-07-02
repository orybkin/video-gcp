import torch

from blox import AttrDict, batch_apply, rmap
from blox.tensor.ops import batchwise_index
from blox.torch.losses import BCELogitsLoss
from blox.torch.modules import DummyModule
from blox.torch.ops import make_one_hot, broadcast_final
from blox.torch.subnetworks import Predictor


class BaseBinding(DummyModule):
    def __init__(self, hp, decoder=None):
        super().__init__()
        self._hp = hp
        self.decoder = decoder

        self.build_network()

    def get_init_inds(self, inputs):
        return inputs.start_ind.float()[:, None], inputs.end_ind.float()[:, None]

    def apply_tree(self, tree, inputs):
        # recursive_add_dim = make_recursive(lambda x: add_n_dims(x, n=1, dim=1))
        start_ind, end_ind = self.get_init_inds(inputs)
        tree.apply_fn(
            {}, fn=self, left_parents=AttrDict(timesteps=start_ind), right_parents=AttrDict(timesteps=end_ind))
    
    def get_matched_sequence(self, tree, key):
        latents = tree.bf[key]
        indices = tree.bf.match_dist.argmax(1)
        # Two-dimensional indexing
        matched_sequence = rmap(lambda x: batchwise_index(x, indices), latents)
    
        return matched_sequence


class BalancedBinding(BaseBinding):
    # TODO almost all of this class is no longer needed I think
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
    
    def get_w(self, pad_mask, inputs, outputs, log=False):
        """ Match to the middle frame between parents """
        self.apply_tree(outputs.tree, inputs)
        match_dists = outputs.tree.bf.c_n_prime
        return match_dists

    def get_init_inds(self, outputs):
        # TODO this actually mostly gets passed 'inputs'
        return torch.zeros_like(outputs.end_ind[:, None]) - 1, \
               outputs.end_ind[:, None] + 1

    def prune_sequence(self, inputs, outputs, key='images'):
        seq = getattr(outputs.tree.df, key)
        latent_seq = outputs.tree.df.e_g_prime
    
        existence = batch_apply(self.existence_predictor, latent_seq)[..., 0]
        outputs.existence_predictor = AttrDict(existence=existence)
    
        existing_frames = torch.sigmoid(existence) > 0.5
        existing_frames[:, 0] = 1
        pruned_seq = [seq[i][existing_frames[i]] for i in range(seq.shape[0])]
    
        return pruned_seq

    def loss(self, inputs, outputs):
        losses = AttrDict()
    
        if 'existence_predictor' in outputs:
            losses.existence_predictor = BCELogitsLoss()(
                outputs.existence_predictor.existence, outputs.tree.df.match_dist.sum(2).float())
        return losses

    def reconstruction_loss(self, inputs, outputs, weights=1):
        """ Balanced tree can have a simpler loss version which doesn't use cdist """
        tree = outputs.tree
    
        estimates = self.get_matched_sequence(tree, 'distr')
        outputs.soft_matched_estimates = self.get_matched_sequence(tree, 'images')
        targets = inputs.traj_seq

        weights = broadcast_final(weights * inputs.pad_mask, targets)
        losses = self.decoder.nll(estimates, targets, weights, log_error_arr=True)
    
        return losses
