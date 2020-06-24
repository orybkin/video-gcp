from blox import AttrDict
from blox.torch.dist import normalize
from blox.torch.ops import mask_out, make_one_hot, batch_cdist
from gcp.rec_planner_utils.matching import BaseMatcher


class TAPMatcher(BaseMatcher):
    def __call__(self, _, subgoal, left_parent, right_parent):
        """ Match to the most similar ground truth image """
        dist = subgoal.tap_distance
        mask_out(dist, left_parent.timestep, right_parent.timestep, 0)
        timestep = dist.argmax(1)
    
        # If no space, match to the left
        mask = (dist == 0).all(1)
        timestep[mask] = left_parent.timestep[mask]
    
        c_n_prime = make_one_hot(timestep, self._hp.max_seq_len)
        return AttrDict(timestep=timestep, c_n_prime=c_n_prime.float())
    
    def get_w(self, pad_mask, inputs, model_output, log=False):
        """ Matches to the closest gt image. """
        tree = model_output.tree
        
        # Get distance matrix
        tree.bf.tap_distance = batch_cdist(tree.bf.images, inputs.demo_seq)
        # Compute matches
        self.apply_tree(tree, inputs)
        w_matrix = normalize(tree.bf.c_n_prime.clone(), 1)
        
        return w_matrix
    
    def get_init_inds(self, inputs):
        return inputs.start_ind[:, None] - 1, inputs.end_ind[:, None] + 1