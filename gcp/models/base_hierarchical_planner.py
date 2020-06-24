from blox import AttrDict, rmap
from blox.basic_types import subdict
from blox.tensor.ops import add_n_dims
import torch
import torch.distributions

from gcp.models.gcp_model import GCPModel
from gcp.models.hedge import SHPredModule
from gcp.models.untied_layers_tree import UntiedLayersInpainter
from gcp.rec_planner_utils.tree_utils import SubgoalTreeLayer


class HierarchicalPlanner(GCPModel):
    def build_network(self, build_encoder=True):
        super().build_network(build_encoder)
        
        cls = SHPredModule
        if self._hp.untied_layers:
            cls = UntiedLayersInpainter
        self.one_step_planner = cls(self._hp, self.decoder)

        self.dense_rec = self.one_step_planner._get_dense_rec_class()(**self._get_rec_class_args())

    def _create_initial_nodes(self, inputs):
        start_node, end_node = AttrDict(e_g_prime=inputs.enc_e_0, images=inputs.I_0), \
                               AttrDict(e_g_prime=inputs.enc_e_g, images=inputs.I_g)
        if self._hp.forced_attention or self._hp.timestep_cond_attention or self._hp.supervise_attn_weight > 0.0:
            start_match_timestep, end_match_timestep = self.one_step_planner.matcher.get_init_inds(inputs)
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
        filtered_inputs = self.one_step_planner._filter_inputs_for_model(inputs, phase)
        layerwise_inputs = self.filter_layerwise_inputs(inputs)
        start_node, end_node = self._create_initial_nodes(inputs)

        outputs.tree = root = SubgoalTreeLayer()
        tree_inputs = [layerwise_inputs, start_ind, end_ind, start_node, end_node]
        tree_inputs = rmap(lambda x: add_n_dims(x, n=1, dim=1), tree_inputs)
        tree_inputs = [filtered_inputs] + tree_inputs

        root.produce_tree_cont_time(*tree_inputs, self.one_step_planner, self._hp.hierarchy_levels)
        outputs.dense_rec = self.dense_rec(root, inputs)

        if 'demo_seq' in inputs and phase == 'train':
            # compute matching between nodes & frames of input sequence, needed for loss and inv mdl etc.
            self.one_step_planner.compute_matching(inputs, outputs)

        # TODO the matcher has to move to this class
        outputs.pruned_prediction = self.one_step_planner.matcher.prune_sequence(inputs, outputs)

        # add pruned reconstruction if necessary
        if not outputs.dense_rec and self._hp.matching_type == 'balanced':
            # TODO this has to be unified with the balanced tree case
            outputs.pruned_prediction = self.dense_rec.get_all_samples_with_len(
                outputs.end_ind, outputs, inputs, pruning_scheme='basic')[0]

        return outputs

    def get_predicted_pruned_seqs(self, inputs, outputs):
        return self.one_step_planner.matcher.prune_sequence(inputs, outputs, 'e_g_prime')

    def loss(self, inputs, outputs, log_error_arr=False):
        losses = super().loss(inputs, outputs, log_error_arr)

        losses.update(self.one_step_planner.loss(inputs, outputs))

        return losses

    def log_outputs(self, model_output, inputs, losses, step, log_images, phase):
        super().log_outputs(model_output, inputs, losses, step, log_images, phase)

        if model_output.tree.subgoals is None:
            model_output.tree.subgoals = AttrDict()
            # if 'c_n_prime' in model_output.tree.subgoals:
            # The model with parent integration
            # if 'c_n' in model_output.tree.subgoals:
            #     self._logger.log_match_dists(model_output.tree, "match_dists", step, phase)

        if log_images:
            if self._hp.use_convs:
                self._logger.log_hierarchy_image(model_output, inputs, "hierarchical_splits", step, phase)
                self._logger.log_rows_gif([model_output.pruned_prediction], 'pruned_seq_gif', step, phase)
                
                if 'match_dist' in model_output.tree.subgoals:
                    # Any model that has matching
                    self._logger.log_gt_match_overview(model_output, inputs, "match_overview", step, phase)
    
                    if 'soft_matched_estimates' in model_output:
                        self._logger.log_loss_gif(model_output.soft_matched_estimates, inputs.demo_seq,
                                                  'gt_target_gif', step, phase)
                        
                    if 'a_l' in model_output.tree.subgoal and inputs.actions.shape[-1] == 2:
                        self._logger.log_gt_action_match_overview(model_output, inputs, self._hp,
                                                                  "action_match_overview", step, phase)

                if phase == 'val' and 'images' in model_output.tree.subgoals:
                    self._logger.log_val_tree(model_output, inputs, "output_tree", step, phase)
                if 'pixel_copy_mask' in model_output.tree.subgoals:
                    self._logger.log_balanced_tree(model_output, "pixel_copy_mask", "pixel_copy_masks", step, phase)
                if 'gamma' in model_output.tree.subgoals and model_output.tree.subgoals.gamma is not None:
                    self._logger.log_attention_overview(model_output, inputs, "attention_masks", step, phase)
                if model_output.dense_rec:
                    self._logger.log_pruned_pred(model_output, inputs, "pruned_pred", step, phase)
                    if model_output.tree.pruned is not None:
                        self._logger.log_pruned_tree(model_output, "pruned_tree", step, phase)

            log_prior_images = False
            if log_prior_images:
                # Run the model N times
                with torch.no_grad(), self.val_mode():
                    rows = list([self(inputs).pruned_prediction for i in range(4)])
                    self._logger.log_rows_gif(rows, "prior_samples", step, phase)
                    
                    for i in range(4):
                        if 'regressed_state' in model_output:
                            out = self(inputs, 'test')
                            self._logger.log_maze_topdown(model_output, inputs, "prior_regressed_state_topdown_" + str(i),
                                                          step, phase, predictions=out.regressed_state,
                                                          end_inds=inputs.end_ind)


class HierarchicalPlannerTest(HierarchicalPlanner):
    pass

