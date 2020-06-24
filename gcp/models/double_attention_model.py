from blox import AttrDict, rmap, batch_apply
from blox.tensor.ops import add_n_dims, map_recursive, batchwise_index
from blox.torch.ops import make_one_hot
from blox.torch.dist import normalize
from gcp.models.base_hierarchical_planner import HierarchicalPlanner
from gcp.models.hedge import SHPredModule
from gcp.rec_planner_utils.tree_utils import SubgoalTreeLayer

""" This module contains the double pass DTW model which uses the first pass to get the matching and in the second pass
produces the latents from the correct matched frames. This was used for debugging. Likely contains bugs - didn't work
too well. """

class DoubleAttentionModel(HierarchicalPlanner):
    def predict_sequence(self, inputs, outputs, start_ind, end_ind, phase):
        filtered_inputs = self.one_step_planner._filter_inputs_for_model(inputs, phase)
        layerwise_inputs = self.filter_layerwise_inputs(inputs)
        start_node, end_node = self._create_initial_nodes(inputs)
        
        outputs.tree = tree = root = SubgoalTreeLayer()
        tree_inputs = [layerwise_inputs, start_ind, end_ind, start_node, end_node]
        tree_inputs = rmap(lambda x: add_n_dims(x, n=1, dim=1), tree_inputs)
        tree_inputs = [filtered_inputs] + tree_inputs
        
        self.produce_tree(root, tree, tree_inputs, inputs, outputs)
        
        outputs.dense_rec = self.dense_rec(root, inputs)
        
        # add pruned reconstruction if necessary
        if not outputs.dense_rec and self._hp.matching_type == 'balanced':
            outputs.pruned_prediction = self.dense_rec.get_all_samples_with_len(
                outputs.end_ind, outputs, inputs, pruning_scheme='basic')[0]
        
        return outputs

    def produce_tree(self, root, tree, tree_inputs, inputs, outputs):
        # Produce the tree to get the matching
        root.produce_tree_cont_time(*tree_inputs, self.one_step_planner, self._hp.hierarchy_levels)
    
        if not self.one_step_planner._sample_prior:
            tree.set_attr_bf(**self.decoder.decode_seq(inputs, tree.bf.e_g_prime))
            tree.bf.match_dist = outputs.gt_match_dists = self.one_step_planner.matcher.get_w(inputs.pad_mask, inputs,
                                                                                              outputs)
        
            # Run the tree the second time attending to the matched frames
            # TODO pass an indicator variable for the predictive model
            tree_inputs[1].attention_weights = add_n_dims(normalize(tree.df.match_dist, -1), n=1, dim=1)
            root.produce_tree_cont_time(*tree_inputs, self.one_step_planner, self._hp.hierarchy_levels)


class AutoencoderModel(DoubleAttentionModel):
    """ This model works as an autoencoder by reconstructing from the matched latents """
    def produce_tree(self, root, tree, tree_inputs, inputs, outputs):
        # Produce the tree to get the matching
        root.produce_tree_cont_time(*tree_inputs, self.one_step_planner, self._hp.hierarchy_levels)
        
        if not self.one_step_planner._sample_prior:
            tree.set_attr_bf(**self.decoder.decode_seq(inputs, tree.bf.e_g_prime))
            tree.bf.match_dist = outputs.gt_match_dists = self.one_step_planner.matcher.get_w(inputs.pad_mask, inputs,
                                                                                              outputs)

            matched_index = tree.bf.match_dist.argmax(-1)
            tiled_enc_demo_seq = inputs.enc_demo_seq[:, None].repeat_interleave(matched_index.shape[1], 1)
            matched_latents = batch_apply([tiled_enc_demo_seq, matched_index], lambda pair: batchwise_index(pair[0], pair[1]))
            
            tree.bf.e_g_prime = matched_latents


class AutoAttentionModel(DoubleAttentionModel):
    """ This model works as an autoencoder by reconstructing from the matched latents, however the latents
    are produced using the attention code. Used to test attention code for bugs """
    def produce_tree(self, root, tree, tree_inputs, inputs, outputs):
        # Produce the tree to get the matching
        root.produce_tree_cont_time(*tree_inputs, self.one_step_planner, self._hp.hierarchy_levels)
        
        if not self.one_step_planner._sample_prior:
            tree.set_attr_bf(**self.decoder.decode_seq(inputs, tree.bf.e_g_prime))
            tree.bf.match_dist = outputs.gt_match_dists = self.one_step_planner.matcher.get_w(inputs.pad_mask, inputs,
                                                                                              outputs)
            
            # Use attention
            tree_inputs[1].attention_weights = add_n_dims(normalize(tree.df.match_dist, -1), n=1, dim=1)
            root.produce_tree_cont_time(*tree_inputs, self.one_step_planner, self._hp.hierarchy_levels)
            
            # Autoencode
            tree.bf.e_g_prime = tree.bf.e_tilde
