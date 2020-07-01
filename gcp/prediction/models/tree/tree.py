import torch
import torch.distributions
from gcp.prediction.models.tree.untied_layers_tree import UntiedLayersTree
from gcp.prediction.models.tree.tree_module import TreeModule
from gcp.prediction.models.tree.tree_dense_rec import TreeDenseRec

from blox import AttrDict, rmap
from blox.basic_types import subdict
from blox.tensor.ops import add_n_dims
from gcp.prediction.models.base_gcp import BaseGCPModel
from gcp.prediction.utils.tree_utils import SubgoalTreeLayer


class TreeModel(BaseGCPModel):
    def build_network(self, build_encoder=True):
        super().build_network(build_encoder)
        
        cls = TreeModule
        if self._hp.untied_layers:
            cls = UntiedLayersTree
        self.tree_module = cls(self._hp, self.decoder)

        self.dense_rec = TreeDenseRec(
            hp=self._hp, input_size=self._hp.nz_enc, output_size=self._hp.nz_enc, decoder=self.decoder)
        
    def _create_initial_nodes(self, inputs):
        start_node, end_node = AttrDict(e_g_prime=inputs.e_0, images=inputs.I_0), \
                               AttrDict(e_g_prime=inputs.e_g, images=inputs.I_g)
        if not self._hp.attentive_inference:
            start_match_timestep, end_match_timestep = self.tree_module.binding.get_init_inds(inputs)
            start_node.update(AttrDict(match_timesteps=start_match_timestep))
            end_node.update(AttrDict(match_timesteps=end_match_timestep))
        if self._hp.tree_lstm:
            start_node.hidden_state, end_node.hidden_state = None, None
        return start_node, end_node

    def filter_layerwise_inputs(self, inputs):
        layerwise_input_keys = ['z']  # these inputs are assumed to be depth-first inputs per node in dim 1
        layerwise_inputs = subdict(inputs, layerwise_input_keys, strict=False)
        return layerwise_inputs

    def predict_sequence(self, inputs, outputs, start_ind, end_ind, phase):
        layerwise_inputs = self.filter_layerwise_inputs(inputs)
        start_node, end_node = self._create_initial_nodes(inputs)

        outputs.tree = root = SubgoalTreeLayer()
        tree_inputs = [layerwise_inputs, start_ind, end_ind, start_node, end_node]
        tree_inputs = rmap(lambda x: add_n_dims(x, n=1, dim=1), tree_inputs)
        tree_inputs = [inputs] + tree_inputs

        root.produce_tree(*tree_inputs, self.tree_module, self._hp.hierarchy_levels)
        outputs.dense_rec = self.dense_rec(root, inputs)

        if 'traj_seq' in inputs and phase == 'train':
            # compute matching between nodes & frames of input sequence, needed for loss and inv mdl etc.
            self.tree_module.compute_matching(inputs, outputs)

        # TODO the binding has to move to this class
        outputs.pruned_prediction = self.tree_module.binding.prune_sequence(inputs, outputs)

        # add pruned reconstruction if necessary
        if not outputs.dense_rec and self._hp.matching_type == 'balanced':
            # TODO this has to be unified with the balanced tree case
            outputs.pruned_prediction = self.dense_rec.get_all_samples_with_len(
                outputs.end_ind, outputs, inputs, pruning_scheme='basic')[0]

        return outputs

    def get_predicted_pruned_seqs(self, inputs, outputs):
        return self.tree_module.binding.prune_sequence(inputs, outputs, 'e_g_prime')

    def loss(self, inputs, outputs, log_error_arr=False):
        losses = super().loss(inputs, outputs, log_error_arr)

        losses.update(self.tree_module.loss(inputs, outputs))

        return losses

    def log_outputs(self, outputs, inputs, losses, step, log_images, phase):
        super().log_outputs(outputs, inputs, losses, step, log_images, phase)

        if outputs.tree.subgoals is None:
            outputs.tree.subgoals = AttrDict()

        if log_images:
            dataset = self._hp.dataset_class
            if self._hp.use_convs:
                self._logger.log_hierarchy_image(outputs, inputs, "hierarchical_splits", step, phase)
                self._logger.log_rows_gif([outputs.pruned_prediction], 'pruned_seq_gif', step, phase)
                
                if 'match_dist' in outputs.tree.subgoals:
                    # Any model that has matching
                    self._logger.log_gt_match_overview(outputs, inputs, "match_overview", step, phase)
    
                    if 'soft_matched_estimates' in outputs:
                        self._logger.log_loss_gif(outputs.soft_matched_estimates, inputs.traj_seq,
                                                  'gt_target_gif', step, phase)

                if phase == 'val' and 'images' in outputs.tree.subgoals:
                    self._logger.log_val_tree(outputs, inputs, "output_tree", step, phase)
                if 'pixel_copy_mask' in outputs.tree.subgoals:
                    self._logger.log_balanced_tree(outputs, "pixel_copy_mask", "pixel_copy_masks", step, phase)
                if 'gamma' in outputs.tree.subgoals and outputs.tree.subgoals.gamma is not None:
                    self._logger.log_attention_overview(outputs, inputs, "attention_masks", step, phase)
                if outputs.dense_rec:
                    self._logger.log_pruned_pred(outputs, inputs, "pruned_pred", step, phase)
                    if outputs.tree.pruned is not None:
                        self._logger.log_pruned_tree(outputs, "pruned_tree", step, phase)

            log_prior_images = False
            if log_prior_images:
                # Run the model N times
                with torch.no_grad(), self.val_mode():
                    rows = list([self(inputs).pruned_prediction for i in range(4)])
                    self._logger.log_rows_gif(rows, "prior_samples", step, phase)
                    
                    for i in range(4):
                        if 'regressed_state' in outputs:
                            out = self(inputs, 'test')
                            self._logger.log_dataset_specific_trajectory(outputs, inputs,
                                                                         "prior_regressed_state_topdown_" + str(i),
                                                                         step, phase, dataset,
                                                                         predictions=out.regressed_state,
                                                                         end_inds=inputs.end_ind)
