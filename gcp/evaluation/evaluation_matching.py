import numpy as np
import torch
from blox.torch.ops import cdist
from blox import AttrDict
from blox.utils import PriorityQueue
from gcp.models.trees.balanced import BalancedMatcher

try:
    print("\nUsing fast C-version of DTW!\n")
    import gcp.evaluation.cutils as cutils
    from gcp.evaluation.dtw_utils import c_dtw as dtw
except:
    print("\nC-version of DTW not compiled! Falling back to slower Numpy version!\n")
    from gcp.evaluation.dtw_utils import basic_dtw as dtw


def torch2np(tensor):
    return tensor.detach().cpu().numpy()


class BaseEvalMatcher:
    def __init__(self, hp):
        self._hp = hp
        self._leaf_only = self._hp.leaf_nodes_only
    
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("This function needs to be implemented by the subclass!")
    
    def _init_tree(self, tree, inputs):
        bsize = tree.end_ind.shape[0]
        for node in tree:
            node.selected = np.zeros(bsize, dtype=bool)
    
    def _collect_sequence(self, tree, inputs, i_ex):
        if self._leaf_only:
            return tree.get_leaf_nodes().images[i_ex]
        else:
            return torch.stack([node.subgoal.images[i_ex] for node in tree.depth_first_iter()])
    
    @staticmethod
    def _check_output(output_seq, length):
        if output_seq.shape[0] != length:
            print("Expected output length {}, got sequence of length {}. Repeating last frame".format(length,
                                                                                                      output_seq.shape[
                                                                                                          0]))
            output_seq = torch.cat((output_seq, output_seq[-1:].expand(length - output_seq.shape[0], -1, -1, -1)))
            return output_seq
        return output_seq


class GreedyExistEvalMatcher(BaseEvalMatcher):
    def __call__(self, tree, inputs, length, i_ex):
        """Perform greedy search over the tree prioritizing with respect to existence probability."""
        if tree.selected is None:
            self._init_tree(tree)
        root = tree
        
        p_queue = PriorityQueue()
        p_queue.push(root, torch2np(root.subgoal.p_n_hat[i_ex]))
        
        for i in range(length):
            node = p_queue.pop()
            node.selected[i_ex] = True
            
            s0 = node.s0
            s1 = node.s1
            if s0.subgoal is not None:
                p_queue.push(s0, torch2np(s0.subgoal.p_n_hat[i_ex]))
            if s1.subgoal is not None:
                p_queue.push(s1, torch2np(s1.subgoal.p_n_hat[i_ex]))
        
        gen_images = self._collect_sequence(tree, inputs, i_ex)
        gen_images = self._check_output(gen_images, length)
        return gen_images, None


class GreedyL2EvalMatcher(BaseEvalMatcher):
    def _init_tree(self, tree, inputs):
        super()._init_tree(tree, inputs)
        tree.min_l2_match(np.zeros_like(inputs.end_ind.cpu().numpy()), inputs.end_ind.cpu().numpy() + 1,
                                  inputs.demo_seq,
                                  np.asarray(np.ones_like(inputs.end_ind.cpu().numpy()), dtype=np.uint8))
    
    @staticmethod
    def _get_best_filler_seq(partial_gt_seq, frame_1, frame_2):
        frames = torch.stack((frame_1, frame_2), dim=0)
        gt_seg_len = partial_gt_seq.shape[0]
        l2_distances = torch.nn.MSELoss(reduction='none')(partial_gt_seq[None], frames[:, None]) \
            .view(2, gt_seg_len, -1).mean(-1)
        frame_choices = torch.argmin(l2_distances, dim=0)
        frames = frames[frame_choices]
        return [t[0] for t in torch.split(frames, 1)]
    
    @staticmethod
    def _collect_sequence(tree, inputs, i_ex):
        sel_list = []
        
        def maybe_fill(prev_matched_step, prev_matched_img, matched_step, matched_img):
            diff = matched_step - prev_matched_step
            if diff > 1:  # add missing nodes
                sel_list.extend(
                    GreedyL2EvalMatcher._get_best_filler_seq(inputs.demo_seq[i_ex, prev_matched_step + 1:matched_step],
                                                             prev_matched_img, matched_img))
        
        prev_matched_step, prev_matched_img = -1, None
        for node in tree:  # iterate through the tree
            if node.selected[i_ex]:
                matched_step, matched_img = node.match_eval_idx[i_ex], node.subgoal.images[i_ex]
                prev_matched_img = matched_img if prev_matched_img is None else prev_matched_img  # fill with first predicted image
                maybe_fill(prev_matched_step, prev_matched_img, matched_step, matched_img)
                sel_list.append(matched_img)
                prev_matched_step, prev_matched_img = matched_step, matched_img
        maybe_fill(prev_matched_step, prev_matched_img, inputs.end_ind[i_ex] + 1,
                   prev_matched_img)  # fill with last predicted image
        return torch.stack(sel_list, dim=0)
    
    def __call__(self, tree, inputs, length, i_ex):
        """Perform greedy minimal-L2-matchings from the root of the tree."""
        if tree.selected is None:
            self._init_tree(tree, inputs)
        gen_images = self._collect_sequence(tree, inputs, i_ex)
        gen_images = self._check_output(gen_images, length)
        return gen_images, None


class DTWEvalMatcher(BaseEvalMatcher):
    def __call__(self, model_output, inputs, length, i_ex, targets=None, estimates=None):
        """ Match """
        if estimates is None:
            estimates = self._collect_sequence(model_output.tree, inputs, i_ex)
        
        if targets is None:
            targets = inputs.demo_seq[i_ex, :inputs.end_ind[i_ex] + 1]

        return self.get_single_matches(targets, estimates)
    
    @staticmethod
    def get_single_matches(targets, estimates):
        # Get dtw
        matrix = cdist(estimates, targets, reduction='mean').data.cpu().numpy()
        # matrix = ((estimates[:, None] - targets[None]) ** 2).mean(dim=[2, 3, 4]).data.cpu().numpy()
        # norm = lambda x,targets: torch.nn.MSELoss()(x,targets).data.cpu().numpy()
        d, cost_matrix, path = dtw(matrix)
        
        # Get best matches for gt frames
        match_matrix = np.zeros_like(cost_matrix)
        match_matrix[:, :] = np.inf
        match_matrix[path[0], path[1]] = cost_matrix[path[0], path[1]]
        inds = np.argmin(match_matrix, axis=0)
        gen_images = estimates[inds]
        matching_output = AttrDict(targets=targets, estimates=estimates, matching_path=path, gen_images=gen_images)
        return gen_images, matching_output
    
    @staticmethod
    def vis_matching(matching_output):
        """
        Visualizes the DTW matching path between GT and predicted sequence
        :param matching_output: Dict that gets returned in 'get_single_matches'
                (targets: n_t, channels, res, res; dtype: torch.Tensor)
                (estimates: n_n, channels, res, res; dtype: torch.Tensor)
                ([2x(max(n_t, n_n))]; dtype: ndarray)
                (gen_images: n_t, channels, res, res; dtype: torch.Tensor)
        """
        n_t, channels, res, _ = matching_output.targets.shape
        n_n = matching_output.estimates.shape[0]
        img = -torch.ones((channels, res * (n_t + 1), res * (n_n + 2)), dtype=matching_output.targets.dtype)
        img[:, res:, :res] = matching_output.targets.transpose(0, 1).contiguous().view(channels, n_t * res, res)
        img[:, res:, res:2 * res] = matching_output.gen_images.transpose(0, 1).contiguous().view(channels, n_t * res,
                                                                                                 res)
        img[:, :res, 2 * res:] = matching_output.estimates.permute(1, 2, 0, 3).contiguous().view(channels, res,
                                                                                                 n_n * res)
        for pn, pt in zip(*matching_output.matching_path):
            img[:, (pt + 1) * res: (pt + 2) * res, (pn + 2) * res: (pn + 3) * res] = 1.0
        return img.permute(1, 2, 0).data.cpu().numpy()


class BalancedEvalMatcher(BaseEvalMatcher):
    def __init__(self, hp):
        self.matcher = BalancedMatcher(hp)
        super().__init__(hp)

    def __call__(self, model_output, inputs, length, i_ex, name=None, targets=None, estimates=None):
        start_ind, end_ind = self.matcher.get_init_inds(model_output)
        if i_ex == 0:
            # Only for the first element
            model_output.tree.compute_matching_dists({}, matching_fcn=self.matcher,
                                                left_parents=AttrDict(timesteps=start_ind),
                                                right_parents=AttrDict(timesteps=end_ind))

        name = 'images' if name is None else name
        estimates = torch.stack([node.subgoal[name][i_ex] for node in model_output.tree.depth_first_iter()])
        leave = torch.stack([node.subgoal.c_n_prime[i_ex] for node in model_output.tree.depth_first_iter()]).byte().any(1)
        return estimates[leave], None
    
    def get_all_samples(self, model_output, inputs, length, name=None, targets=None, estimates=None):
        start_ind, end_ind = self.matcher.get_init_inds(model_output)
        
        # Only for the first element
        model_output.tree.compute_matching_dists({}, matching_fcn=self.matcher,
                                            left_parents=AttrDict(timesteps=start_ind),
                                            right_parents=AttrDict(timesteps=end_ind))

        name = 'images' if name is None else name
        estimates = torch.stack([node.subgoal[name] for node in model_output.tree.depth_first_iter()])
        leave = torch.stack([node.subgoal.c_n_prime for node in model_output.tree.depth_first_iter()]).byte().any(-1)

        pruned_seqs = [estimates[:, i][leave[:, i]] for i in range(leave.shape[1])]
        
        return pruned_seqs, None
    
    
class BalancedPrunedDTWMatcher():
    def __init__(self, hp):
        self.pruning_matcher = BalancedEvalMatcher(hp)
        self.dtw_matcher = DTWEvalMatcher(hp)

    def __call__(self, model_output, inputs, length, i_ex, targets=None, estimates=None):
        
        pruned_sequence = self.pruning_matcher(model_output, inputs, length, i_ex)
        warped_sequence = self.dtw_matcher(model_output, inputs, length, i_ex, estimates=pruned_sequence[0])
        
        return warped_sequence
    
    @staticmethod
    def vis_matching(matching_output):
        return DTWEvalMatcher.vis_matching(matching_output)
        
