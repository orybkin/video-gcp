import numpy as np
import torch
from blox.torch.ops import slice_tensor, reduce_dim
from blox.torch import porch
from blox.basic_types import map_dict, listdict2dictlist
from blox.tensor.ops import batch_apply, make_recursive_list, map_recursive


class SubgoalTreeLayer:
    def __init__(self, parent=None):
        if isinstance(parent, map):
            raise ValueError("This happens when splitting across multiple GPUs, will get caught above")
        self.child_layer = None
        self.subgoals = None
        self.depth = None
        self.pruned = None
        # self.parent_layer = parent
        self.selected = None
        self.match_eval_idx = None

    def produce_tree_cont_time(self, inputs, layerwise_inputs, start_inds, end_inds, left_parents, right_parents, producer, depth):
        """no done mask checks, assumes start_ind never None.
            all input tensors are of shape [batch, num_parent_nodes, ...]
        """
        self.depth = depth
        if depth == 0:
            return

        # slice out inputs for this layer

        layer_inputs = map_recursive(lambda x: depthfirst2layers(reduce_dim(x, dim=1))[-depth].contiguous(), layerwise_inputs)

        out = batch_apply([layer_inputs, start_inds.float(), end_inds.float(), left_parents, right_parents],
                          lambda *x: producer.produce_subgoal(inputs, *x, depth=depth), separate_arguments=True)
        self.subgoals, left_parents, right_parents = out

        self.child_layer = SubgoalTreeLayer(self)
        self.child_layer.produce_tree_cont_time(inputs,
                                                layerwise_inputs,
                                                rec_interleave([start_inds.float(), self.subgoals.ind.clone()]),
                                                rec_interleave([self.subgoals.ind.clone(), end_inds.float()]),
                                                rec_interleave([left_parents, self.subgoals]),
                                                rec_interleave([self.subgoals, right_parents]),
                                                producer, depth - 1)

    def compute_matching_dists(self, inputs, matching_fcn, left_parents, right_parents):
        """Computes the distribution of matches of subgoals to ground truth frames."""
        self.apply_fn(inputs, matching_fcn, left_parents, right_parents)
        
    def apply_fn(self, inputs, fn, left_parents, right_parents):
        """ Recursively applies fn to the tree.
        
        :param inputs:
        :param fn: a function that takes in (inputs, subgoal, left_parent, right_parent) and outputs a dict
        :param left_parents:
        :param right_parents:
        :return:
        """
        
        if self.depth == 0:
            return
        assert self.subgoals is not None      # need subgoal info to match to ground truth sequence

        self.subgoals.update(batch_apply([inputs, self.subgoals, left_parents, right_parents],
                                         fn, separate_arguments=True, unshape_inputs=True))
        self.child_layer.apply_fn(rec_interleave([inputs, inputs]),
                                  fn,
                                  rec_interleave([left_parents, self.subgoals]),
                                  rec_interleave([self.subgoals, right_parents]))

    def __iter__(self):
        """Layer-wise iterator."""
        if self.subgoals is None:
            return
        yield self
        if self.child_layer is not None:
            for l in self.child_layer:
                yield l

    def depth_first_iter(self, current_node=0):
        """Depth-first subgoal iterator."""
        if self.subgoals is None or self.child_layer is None:
            return
        for n in self.child_layer.depth_first_iter(2*current_node):
            yield n
        self.subgoal = map_recursive(lambda x: x[:, current_node], self.subgoals)
        yield self
        for n in self.child_layer.depth_first_iter(2*current_node+1):
            yield n

    def get_attr_df(self, attr):
        # TODO make this faster
        return torch.stack([node.subgoal[attr] for node in self.depth_first_iter()], 1)
    
    def set_attr_df(self, **kwargs):
        # TODO check
        for name, value in kwargs:
            split = self.split_by_layer_df(value, 1)
            for chunk, node in zip(split, self):
                node[name] = chunk
        
    def get_attr_bf(self, attr):
        return porch.cat([node.subgoals[attr] for node in self], 1)

    def set_attr_bf(self, **kwargs):
        start = 0
        for i, node in enumerate(self):
            node.subgoals.update(map_recursive(lambda x: x[:,start:start+2**i].contiguous(), kwargs))
            start += 2**i

    def get_leaf_nodes(self):
        if self.depth == 0:
            raise ValueError("Depth 0 tree does not have leaf nodes!")
        elif self.depth == 1:
            return self.subgoals
        else:
            return self.child_layer.get_leaf_nodes()

    @staticmethod
    def cat(*argv):
        tree = SubgoalTreeLayer()
        for attr, val in argv[0].__dict__.items():
            if val is None or np.isscalar(val):
                tree.__dict__[attr] = val
            elif attr == 'subgoals':
                tree.__dict__[attr] = map_dict(concat, listdict2dictlist([d.subgoals for d in argv]))
            elif attr == 'child_layer':
                tree.__dict__[attr] = SubgoalTreeLayer.cat(*[d.child_layer for d in argv])
            else:
                raise ValueError("Cannot handle data type {} during tree concatenation!".format(type(val)))
        return tree

    @staticmethod
    def reduce(*argv):
        """Called inside result gathering for multi-GPU processing"""
        return SubgoalTreeLayer.cat(*argv)

    @staticmethod
    def split_by_layer_df(vals, dim):
        return depthfirst2layers(vals, dim)
        # """Splits depth-first vals into N lists along dimension dim, each containing vals for the corresp. layer."""
        # depth = int(np.log2(vals.shape[dim]) + 1)
        # output = [[] for _ in range(depth)]     # one list per layer
        #
        # def get_elem(l_idx, r_idx, d):
        #     if l_idx == r_idx - 1: return
        #     idx = int((r_idx - l_idx) / 2) + l_idx
        #     output[d].append(vals[:, idx])
        #     get_elem(l_idx, idx, d + 1)
        #     get_elem(idx, r_idx, d + 1)
        #
        # get_elem(-1, vals.shape[dim], 0)
        # return output

    @staticmethod
    def split_by_layer_bf(vals, dim):
        """Splits breadth-first vals into N arrays along dimension dim, each containing vals for the corresp. layer."""
        depth = int(np.log2(vals.shape[dim]) + 1)
        output = []     # one list per layer
        current_idx = 0
        for d in range(depth):
            output.append(vals[:, current_idx : current_idx + int(2**d)])
            current_idx += int(2**d)
        return output

    @property
    def bf(self):
        return AccessWrapper(self, 'bf')
    
    @property
    def df(self):
        return AccessWrapper(self, 'df')

    @property
    def size(self):
        return int(2**self.depth - 1)


class AccessWrapper():
    def __init__(self, obj, type):
        super().__setattr__('tree', obj)
        super().__setattr__('type', type)
        
    def __getattr__(self, item):
        if self.type == 'bf':
            return self.tree.get_attr_bf(item)
        elif self.type == 'df':
            return self.tree.get_attr_df(item)

    def __setattr__(self, key, value):
        if self.type == 'bf':
            return self.tree.set_attr_bf(**{key: value})
        elif self.type == 'df':
            return self.tree.set_attr_df(**{key: value})

    def __getitem__(self, item):
        return getattr(self, item)
        
    def __setitem__(self, key, value):
        return setattr(self, key, value)


def interleave(t1, t2):
    if t1 is None or t2 is None: return None
    assert t1.shape == t2.shape     # can only interleave vectors of equal shape
    return torch.stack((t1, t2), dim=2).view(t1.shape[0], 2*t1.shape[1], *t1.shape[2:])


rec_interleave = make_recursive_list(interleave)


def concat(*argv):
    if argv[0][0] is None: return None
    device = argv[0][0].device
    return torch.cat([v.to(device) for v in argv[0]], dim=0)


def depthfirst2breadthfirst(tensor, dim=1):
    """ Converts a sequence represented depth first to breadth first """
    return torch.cat(depthfirst2layers(tensor, dim), dim)


def depthfirst2layers(tensor, dim=1):
    """ Converts a sequence represented depth first to a list of layers """
    len = tensor.shape[dim]
    depth = np.int(np.log2(len + 1))
    
    slices = []
    for i in range(depth):
        slices.append(slice_tensor(tensor, 0, 2, dim))
        tensor = slice_tensor(tensor, 1, 2, dim)
    
    return list(reversed(slices))


def ind_df2bf(df_indices, depth):
    """ Transforms indices for a depth-first array such that the same elements can be retrieved from the corresponding
    breadth-first array """
    df_indices = (df_indices + 1).byte()  # starting from 1
    bf_indices = torch.zeros_like(df_indices)
    for i in range(depth):
        mask = (df_indices % (2**i) == 0) & (df_indices % (2**(i+1)) > 0)  # if in layer i from the bottom
        bf_indices[mask] = df_indices[mask] // (2**(i+1)) + (2**(depth - i - 1) - 1)  # order in layer + layer position
        
    return bf_indices


def ind_bf2df(bf_indices, depth):
    """ Transforms indices for a breadth-first array such that the same elements can be retrieved from the corresponding
    depth-first array """
    bf_indices = (bf_indices + 1).byte()  # starting from 1
    df_indices = torch.zeros_like(bf_indices)
    for i in range(depth):
        mask = (bf_indices >= 2 ** i) & (bf_indices < 2 ** (i + 1))  # if in layer i from the top
        ib = depth - i - 1  # layer from the bottom
        # order in layer * layer position
        df_indices[mask] = (bf_indices[mask] - 2**i) * (2**(ib+1)) + (2**ib) - 1
    
    return df_indices


class SubgoalTreeNode:
    pass
    # """ Represents the binary tree of subgoals """
    #
    # def __init__(self, parent=None):
    #     self.s0 = None
    #     self.s1 = None
    #     self.subgoal = None
    #     self.done_mask = None
    #     self.depth = None
    #     self.pruned = None
    #     # self.parent = parent
    #     self.selected = None
    #     self.match_eval_idx = None
    #
    # def produce_tree(self, inputs, start_ind, end_ind, e_0, e_g, producer, depth):
    #     """ Recursively applies the subgoal producer """
    #
    #     raise NotImplementedError("This is deprecated. We should unify the two functions")
    #
    #     self.depth = depth
    #     self.start_ind = start_ind
    #     self.end_ind = end_ind
    #     self.e_0 = e_0
    #     self.e_g = e_g
    #     if start_ind is not None:
    #         self.done_mask = start_ind == (end_ind - 1)  # Special cases where the sequence is empty
    #
    #     if depth == 0:
    #         return
    #
    #     self.subgoal = producer.produce_subgoal(inputs, start_ind, end_ind, e_0, e_g)
    #
    #     if start_ind is not None:
    #         s0_end_ind = self.subgoal.ind.clone()
    #         s0_end_ind[self.done_mask] = self.end_ind[self.done_mask]  # Change so that end_ind > start_ind
    #     else:
    #         s0_end_ind = None
    #     self.s0 = SubgoalTreeNode(self)
    #     self.s0.produce_tree(inputs, start_ind, s0_end_ind, e_0, self.subgoal.e_g_prime, producer, depth - 1)
    #
    #     if start_ind is not None:
    #         s1_start_ind = self.subgoal.ind.clone()
    #         s1_start_ind[self.done_mask] = self.start_ind[self.done_mask]  # Change so that end_ind > start_ind
    #     else:
    #         s1_start_ind = None
    #     self.s1 = SubgoalTreeNode(self)
    #     self.s1.produce_tree(inputs, s1_start_ind, end_ind, self.subgoal.e_g_prime, e_g, producer, depth - 1)
    #
    # def produce_tree_cont_time(self, inputs, start_ind, end_ind, left_parent, right_parent, producer, depth):
    #     """no done mask checks, assumes start_ind never None."""
    #     self.depth = depth
    #     self.start_ind = start_ind.float()
    #     self.end_ind = end_ind.float()
    #     self.e_0 = left_parent.e_g_prime
    #     self.e_g = right_parent.e_g_prime
    #
    #     if depth == 0:
    #         return
    #
    #     self.subgoal = producer.produce_subgoal(inputs, start_ind.float(), end_ind.float(), left_parent, right_parent)
    #
    #     self.s0 = SubgoalTreeNode(self)
    #     self.s0.produce_tree_cont_time(inputs, start_ind.float(), self.subgoal.ind.clone(), left_parent,
    #                                    self.subgoal, producer, depth - 1)
    #     self.s1 = SubgoalTreeNode(self)
    #     self.s1.produce_tree_cont_time(inputs, self.subgoal.ind.clone(), end_ind.float(), self.subgoal,
    #                                    right_parent, producer, depth - 1)
    #
    # def compute_matching_dists(self, inputs, matching_fcn, left_parent, right_parent):
    #     """Computes the distribution of matches of subgoals to ground truth frames."""
    #     if self.depth == 0:
    #         return
    #     assert self.subgoal is not None      # need subgoal info to match to ground truth sequence
    #
    #     self.subgoal.update(matching_fcn(inputs, self.subgoal, left_parent, right_parent))
    #
    #     self.s0.compute_matching_dists(inputs, matching_fcn, left_parent, self.subgoal)
    #     self.s1.compute_matching_dists(inputs, matching_fcn, self.subgoal, right_parent)
    #
    # def compute_action_matching_dists(self, l_dist, r_dist, end_index, detach=False):
    #     if self.depth == 0:
    #         return
    #     assert self.subgoal is not None     # need a subgoal for matching
    #     match_dist = self.subgoal.match_dist.detach() if detach else self.subgoal.match_dist
    #     if l_dist is None:
    #         self.subgoal.a_l_match_dist = torch.cat((match_dist[:, :1], torch.zeros_like(match_dist[:, 1:-1])), dim=1)
    #     else:
    #         self.subgoal.a_l_match_dist = l_dist[:, :-1] * match_dist[:, 1:]
    #     if r_dist is None:
    #         self.subgoal.a_r_match_dist = torch.zeros_like(match_dist[:, :-1])
    #         batchwise_assign(self.subgoal.a_r_match_dist, end_index-1, batchwise_index(match_dist, end_index))
    #     else:
    #         self.subgoal.a_r_match_dist = r_dist[:, 1:] * match_dist[:, :-1]
    #
    #     self.s0.compute_action_matching_dists(l_dist=l_dist, r_dist=match_dist, end_index=end_index, detach=detach)
    #     self.s1.compute_action_matching_dists(l_dist=match_dist, r_dist=r_dist, end_index=end_index, detach=detach)
    #
    # def compute_parent_probs(self, l_dist, r_dist, detach=False):
    #     if self.depth == 0:
    #         return
    #     assert self.subgoal is not None     # need a subgoal for matching
    #     p_n = self.subgoal.p_n.detach() if detach else self.subgoal.p_n
    #     self.subgoal.p_n_parents = torch.ones_like(p_n)      # compute prob of both parents existing
    #     if l_dist is not None:
    #         self.subgoal.p_n_parents = self.subgoal.p_n_parents * l_dist
    #     if r_dist is not None:
    #         self.subgoal.p_n_parents = self.subgoal.p_n_parents * r_dist
    #
    #     self.s0.compute_parent_probs(l_dist=l_dist, r_dist=p_n, detach=detach)
    #     self.s1.compute_parent_probs(l_dist=p_n, r_dist=r_dist, detach=detach)
    #
    # def compute_action_parent_probs(self, l_dist, r_dist, target_tmpl="p_a_{}_parents", source="p_n", detach=False):
    #     if self.depth == 0:
    #         return
    #     assert self.subgoal is not None     # need a subgoal for matching
    #     dist = self.subgoal[source].detach() if detach else self.subgoal[source]
    #     target_l, target_r = target_tmpl.format('l'), target_tmpl.format('r')
    #     self.subgoal[target_l], self.subgoal[target_r] = dist, dist  # compute prob of both parents existing
    #     if l_dist is not None:
    #         self.subgoal[target_l] = self.subgoal[target_l] * l_dist
    #     if r_dist is not None:
    #         self.subgoal[target_r] = self.subgoal[target_r] * r_dist
    #
    #     self.s0.compute_action_parent_probs(l_dist=l_dist, r_dist=dist, target_tmpl=target_tmpl, source=source, detach=detach)
    #     self.s1.compute_action_parent_probs(l_dist=dist, r_dist=r_dist, target_tmpl=target_tmpl, source=source, detach=detach)
    #
    # def node_exist_prune(self, l_parent_pruned, r_parent_pruned, exist_thresh, hierarchical=True):
    #     """Removes node+children if node's existence probability is below the given threshold."""
    #     if self.depth == 0:
    #         self.pruned = np.ones(l_parent_pruned.shape, dtype=bool)
    #         return
    #     self.pruned = self.subgoal.p_n_hat.data.cpu().numpy() < exist_thresh
    #     if hierarchical:
    #         self.pruned = self.pruned | l_parent_pruned | r_parent_pruned
    #     self.s0.node_exist_prune(l_parent_pruned, self.pruned, exist_thresh)
    #     self.s1.node_exist_prune(self.pruned, r_parent_pruned, exist_thresh)
    #
    # def action_exist_prune(self, parent_l_not_exist, parent_r_not_exist, exist_thresh, parent_exist_thresh):
    #     """Removes node+children if node's joint action existence probability is below the given threshold."""
    #     if self.depth == 0:
    #         self.pruned = np.ones(self.start_ind.shape, dtype=bool)
    #         return self.pruned
    #     self_not_exist = (self.subgoal.p_n_hat.data.cpu().numpy() < parent_exist_thresh) | parent_l_not_exist | parent_r_not_exist
    #     s0_pruned = self.s0.action_exist_prune(parent_l_not_exist, self_not_exist, exist_thresh, parent_exist_thresh)
    #     s1_pruned = self.s1.action_exist_prune(self_not_exist, parent_r_not_exist, exist_thresh, parent_exist_thresh)
    #     self.pruned = (self.subgoal.p_a_l_hat.data.cpu().numpy() < exist_thresh) | \
    #                   (self.subgoal.p_a_r_hat.data.cpu().numpy() < exist_thresh) | \
    #                     parent_l_not_exist | parent_r_not_exist
    #     self.pruned = self.pruned & (s0_pruned & s1_pruned)     # can only prune of both children are pruned
    #     return self.pruned
    #
    # def min_l2_match(self, l_parent_idxs, r_parent_idxs, gt_seq, parent_selected):
    #     if self.depth == 0: return
    #     assert self.selected is not None    # can only start matching of 'selected' variable is initialized
    #     batch, seg_len = gt_seq.shape[:2]
    #     l2_distance = torch.nn.MSELoss(reduction='none')(self.subgoal.images[:, None], gt_seq)\
    #                                                      .view(batch, seg_len, -1).mean(-1)
    #     inds = np.repeat(np.arange(seg_len, dtype=l_parent_idxs.dtype)[None], batch, axis=0)
    #     mask = (inds >= r_parent_idxs[:, None]) | (inds < l_parent_idxs[:, None])
    #     l2_distance[torch.tensor(np.asarray(mask, dtype=np.uint8))] = torch.tensor(float("Inf"), device=l2_distance.device)
    #     self.match_eval_idx = torch.argmin(l2_distance, dim=1).data.cpu().numpy()
    #     self.selected = parent_selected & (~batchwise_index(mask, self.match_eval_idx))
    #     self.s0.min_l2_match(l_parent_idxs, self.match_eval_idx, gt_seq, self.selected)
    #     self.s1.min_l2_match(self.match_eval_idx+1, r_parent_idxs, gt_seq, self.selected)
    #
    # def __iter__(self):
    #     """ implement the iterator protocol over the subgoal children  """
    #     if self.subgoal is None:
    #         return
    #
    #     if self.s0 is not None:
    #         for v in self.s0:
    #             yield v
    #     yield self
    #     if self.s1 is not None:
    #         for v in self.s1:
    #             yield v
    #
    # def full_tree(self):
    #     """ implement the iterator protocol over all children """
    #     if self.s0 is not None:
    #         for v in self.s0.full_tree():
    #             yield v
    #     yield self
    #     if self.s1 is not None:
    #         for v in self.s1.full_tree():
    #             yield v
