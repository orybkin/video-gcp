import os
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence as pad_sequence

from blox import batch_apply, AttrDict, optional, rmap
from blox.basic_types import subdict
from blox.tensor.core import find_tensor
from blox.tensor.ops import remove_spatial
from blox.torch.dist import ProbabilisticModel
from blox.torch.encoder_decoder import Encoder, DecoderModule
from blox.torch.layers import BaseProcessingNet
from blox.torch.losses import L2Loss
from blox.torch.modules import Identity, LinearUpdater
from blox.torch.subnetworks import ConvSeqEncodingModule, RecurrentSeqEncodingModule, BidirectionalSeqEncodingModule, \
    Predictor
from gcp.prediction.models.auxilliary_models.misc import AttnKeyEncodingModule, LengthPredictorModule, \
    ActionConditioningWrapper
from gcp.prediction import global_params
from gcp.prediction.hyperparameters import get_default_gcp_hyperparameters
from gcp.prediction.models.auxilliary_models.base_model import BaseModel
from gcp.prediction.models.auxilliary_models.cost_mdl import CostModel
from gcp.prediction.models.auxilliary_models.inverse_mdl import InverseModel
from gcp.prediction.utils import visualization


class BaseGCPModel(BaseModel):
    def __init__(self, params, logger):
        super().__init__(logger)
        self._hp = self._default_hparams()
        self.override_defaults(params)  # override defaults with config file
        self.postprocess_params()
        visualization.PARAMS.hp = self._hp
        assert self._hp.batch_size != -1   # make sure that batch size was overridden
        if self._hp.regress_actions or self._hp.action_conditioned_pred:
            assert self._hp.n_actions != -1     # make sure action dimensionality was overridden

        self.build_network()
        self._use_pred_length = False
        self._inv_mdl_full_seq = False

    @contextmanager
    def val_mode(self, pred_length=True):
        """Sets validation parameters. To be used like: with model.val_mode(): ...<do something>..."""
        self.call_children('switch_to_prior', ProbabilisticModel)
        self._use_pred_length = pred_length
        self._inv_mdl_full_seq = True
        yield
        self.call_children('switch_to_inference', ProbabilisticModel)
        self._use_pred_length = False
        self._inv_mdl_full_seq = False

    def postprocess_params(self):
        super().postprocess_params()
        if self._hp.action_activation is None:
            pass
        elif self._hp.action_activation == 'sigmoid':
            self._hp.action_activation = torch.sigmoid
        elif self._hp.action_activation == 'tanh':
            self._hp.action_activation = torch.tanh
        else:
            raise ValueError('Action activation {} not supported!'.format(self._hp.action_activation))
        
        global_params.hp = self._hp
    
    def _default_hparams(self):
        parent_params = super()._default_hparams()
        default_dict = get_default_gcp_hyperparameters()
        
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def build_network(self, build_encoder=True):
        if build_encoder:
            self.encoder = Encoder(self._hp)
        self.decoder = DecoderModule(self._hp,  # infer actions in decoder if not using SH-Pred model
                               regress_actions=self._hp.regress_actions and self._hp.one_step_planner is not 'sh_pred')

        self.build_inference()

        if self._hp.regress_length:
            self.length_pred = LengthPredictorModule(self._hp)

        if self._hp.attach_inv_mdl:
            self.inv_mdl = InverseModel(self._hp.inv_mdl_params, self._logger)

        if self._hp.attach_cost_mdl:
            self.cost_mdl = CostModel(self._hp.cost_mdl_params, self._logger)

        if self._hp.attach_state_regressor:
            self.state_regressor = BaseProcessingNet(self._hp.nz_enc, self._hp.nz_mid, self._hp.state_dim,
                                                     self._hp.n_processing_layers, self._hp.fc_builder)

        if self._hp.separate_cnn_start_goal_encoder:
            from blox.torch.layers import LayerBuilderParams
            from tensorflow.contrib.training import HParams
            with_conv_hp = HParams()
            for k in self._hp.values().keys():
                with_conv_hp.add_hparam(k, self._hp.values()[k])
            #with_conv_hp = self._hp 
            with_conv_hp.set_hparam('use_convs', True)
            with_conv_hp.set_hparam('input_nc', 3)
            with_conv_hp.set_hparam('builder', LayerBuilderParams(
                            True, self._hp.use_batchnorm, self._hp.normalization, self._hp.predictor_normalization))
            self.start_goal_enc = Encoder(with_conv_hp)

    def build_inference(self):
        if self._hp.states_inference:
            self.inf_encoder = Predictor(self._hp, self._hp.nz_enc + 2, self._hp.nz_enc)
        elif self._hp.act_cond_inference:
            self.inf_encoder = ActionConditioningWrapper(self._hp, self.build_temporal_inf_encoder())
        else:
            self.inf_encoder = self.build_temporal_inf_encoder()
    
        self.inf_key_encoder = nn.Sequential(self.build_temporal_inf_encoder(),
                                             AttnKeyEncodingModule(self._hp, add_time=False))
    
        if self._hp.kl_weight_burn_in is not None:
            target_kl_weight = self._hp.kl_weight
            delattr(self._hp, 'kl_weight')
            del self._hp._hparam_types['kl_weight']
            self._hp.add_hparam('kl_weight', nn.Parameter(torch.zeros(1)))
            self._hp.kl_weight.requires_grad_(False)
            self.kl_weight_updater = LinearUpdater(
                self._hp.kl_weight, self._hp.kl_weight_burn_in, target_kl_weight, name='kl_weight')

    def build_temporal_inf_encoder(self):
        if self._hp.seq_enc == 'none':
            return Identity()
        elif self._hp.seq_enc == 'conv':
            return ConvSeqEncodingModule(self._hp)
        elif self._hp.seq_enc == 'lstm':
            return RecurrentSeqEncodingModule(self._hp)
        elif self._hp.seq_enc == 'bi-lstm':
            return BidirectionalSeqEncodingModule(self._hp)
            
    def forward(self, inputs, phase='train'):
        """
        forward pass at training time
        :param
            images shape = batch x time x height x width x channel
            pad mask shape = batch x time, 1 indicates actual image 0 is padded
        :return:
        """
        outputs = AttrDict()
        inputs.reference_tensor = find_tensor(inputs)
        self.optional_preprocessing(inputs)
    
        # Get features
        self.run_encoder(inputs)
        # Optionally predict sequence length
        end_ind = self.get_end_ind(inputs, outputs)
        # Generate sequences
        outputs.update(self.predict_sequence(inputs, outputs, inputs.start_ind, end_ind, phase))
        # Predict other optional values
        outputs.update(self.run_auxilliary_models(inputs, outputs, phase))
    
        return outputs

    def optional_preprocessing(self, inputs):
        if self._hp.non_goal_conditioned:
            # Set goal to zeros
            if 'traj_seq' in inputs:
                inputs.traj_seq[torch.arange(inputs.traj_seq.shape[0]), inputs.end_ind] = 0.0
                inputs.traj_seq_images[torch.arange(inputs.traj_seq.shape[0]), inputs.end_ind] = 0.0
            inputs.I_g = torch.zeros_like(inputs.I_g)
            if "I_g_image" in inputs:
                inputs.I_g_image = torch.zeros_like(inputs.I_g_image)
            if inputs.I_0.shape[-1] == 5: # special hack for maze
                inputs.I_0[..., -2:] = 0.0
                if "traj_seq" in inputs:
                    inputs.traj_seq[..., -2:] = 0.0

        if self._hp.train_on_action_seqs:
            # swap in actions if we want to train action sequence decoder
            inputs.traj_seq = torch.cat([inputs.actions, torch.zeros_like(inputs.actions[:, :1])], dim=1)

        if 'start_ind' not in inputs:
            inputs.start_ind = torch.zeros(self._hp.batch_size, dtype=torch.long, device=inputs.reference_tensor.device)

    def run_encoder(self, inputs):
        if 'traj_seq' in inputs:
            # Encode sequence
            if not 'enc_traj_seq' in inputs:
                inputs.enc_traj_seq, inputs.skips = batch_apply(self.encoder, inputs.traj_seq)
                if 'skips' in inputs:
                    inputs.skips = rmap(lambda s: s[:, 0], inputs.skips)  # only use start image activations

            # Encode sequence for inference and attention
            if self._hp.act_cond_inference:
                inputs.inf_enc_seq = self.inf_encoder(inputs.enc_traj_seq, inputs.actions)
            elif self._hp.states_inference:
                inputs.inf_enc_seq = batch_apply(self.inf_encoder,
                                                 inputs.enc_traj_seq, inputs.traj_seq_states[..., None, None])
            else:
                inputs.inf_enc_seq = self.inf_encoder(inputs.enc_traj_seq)
            inputs.inf_enc_key_seq = self.inf_key_encoder(inputs.enc_traj_seq)
            
        # Encode I_0, I_g
        if self._hp.separate_cnn_start_goal_encoder:
            e_0, inputs.skips = self.start_goal_enc(inputs.I_0_image)
            inputs.e_0 = remove_spatial(e_0)
            inputs.e_g = remove_spatial(self.start_goal_enc(inputs.I_g_image)[0])
        else:
            inputs.e_0, inputs.skips = self.encoder(inputs.I_0)
            inputs.e_g = self.encoder(inputs.I_g)[0]
            
        # Encode actions
        if self._hp.action_conditioned_pred:
            inputs.enc_action_seq = batch_apply(self.action_encoder, inputs.actions)
            
    def get_end_ind(self, inputs, outputs):
        # TODO clean this up. outputs.end_ind is not currently used anywhere
        end_ind = inputs.end_ind if 'end_ind' in inputs else None
        
        if self._hp.regress_length:
            # predict total sequence length
            outputs.update(self.length_pred(inputs.e_0, inputs.e_g))
            if self._use_pred_length and (self._hp.length_pred_weight > 0 or end_ind is None):
                end_ind = torch.clamp(torch.argmax(outputs.seq_len_pred.sample().long(), dim=1), min=2)   # min pred seq len needs to be >= 3 for planning
                if self._hp.action_conditioned_pred or self._hp.non_goal_conditioned:
                    # don't use predicted length when action conditioned
                    end_ind = torch.ones_like(end_ind) * (self._hp.max_seq_len - 1)

        outputs.end_ind = end_ind
        return end_ind
    
    def predict_sequence(self, inputs, outputs, start_ind, end_ind, phase):
        raise NotImplementedError("Needs to be implemented in the child class")

    def run_auxilliary_models(self, inputs, outputs, phase):
        aux_outputs = AttrDict()
        
        if self.prune_sequences:
            if phase == 'train':
                inputs.model_enc_seq = self.get_matched_pruned_seqs(inputs, outputs)
            else:
                inputs.model_enc_seq = self.get_predicted_pruned_seqs(inputs, outputs)
            inputs.model_enc_seq = pad_sequence(inputs.model_enc_seq, batch_first=True)
            if len(inputs.model_enc_seq.shape) == 5:
                inputs.model_enc_seq = inputs.model_enc_seq[..., 0, 0]
                
            if inputs.model_enc_seq.shape[1] == 0:
                # This happens usually in the beginning of training where the pruning is inaccurate
                return aux_outputs
        
            if self._hp.attach_inv_mdl and phase == 'train':
                aux_outputs.update(self.inv_mdl(inputs, full_seq=self._inv_mdl_full_seq or self._hp.train_inv_mdl_full_seq))
            if self._hp.attach_state_regressor:
                regressor_inputs = inputs.model_enc_seq
                if not self._hp.supervised_decoder:
                    regressor_inputs = regressor_inputs.detach()
                    aux_outputs.regressed_state = batch_apply(self.state_regressor, regressor_inputs)
            if self._hp.attach_cost_mdl and self._hp.run_cost_mdl and phase == 'train':
                # There is an issue here since SVG doesn't output a latent for the first imagge
                # Beyong conceptual problems, this breaks if end_ind = 199
                aux_outputs.update(self.cost_mdl(inputs))
                
        return aux_outputs

    def loss(self, inputs, outputs, log_error_arr=False):
        losses = AttrDict()
        
        # Length prediction loss
        if self._hp.regress_length:
            losses.update(self.length_pred.loss(inputs, outputs))
        
        # Dense Reconstruction loss
        losses.update(self.dense_rec.loss(inputs, outputs.dense_rec, log_error_arr))

        # Inverse Model loss
        if self._hp.attach_inv_mdl:
            losses.update(self.inv_mdl.loss(inputs, outputs, add_total=False))

        # Cost model loss
        if self._hp.attach_cost_mdl and self._hp.run_cost_mdl:
            losses.update(self.cost_mdl.loss(inputs, outputs))

        # State regressor cost
        if self._hp.attach_state_regressor:
            reg_len = outputs.regressed_state.shape[1]
            losses.state_regression = L2Loss(1.0)(outputs.regressed_state, inputs.traj_seq_states[:, :reg_len],
                                                  weights=inputs.pad_mask[:, :reg_len][:, :, None])

        # Negative Log-likelihood (upper bound)
        if 'dense_img_rec' in losses and 'kl' in losses:
            losses.nll = AttrDict(value=losses.dense_img_rec.value + losses.kl.value, weight=0.0)

        return losses
    
    def get_total_loss(self, inputs, losses):
        # compute total loss
        ## filtering is important when some losses are nan
        ## the unsqueeze is important when some of the weights or losses are 1-dim tensors.
        # TODO use the function from blox
        total_loss = torch.stack([loss[1].value[None] * loss[1].weight for loss in
                                  filter(lambda x: x[1].weight > 0, losses.items())]).sum()
        total_loss = total_loss / torch.prod(torch.tensor(inputs.traj_seq.shape[1:]))
        if torch.isnan(total_loss).any():
            import pdb; pdb.set_trace()
        return AttrDict(value=total_loss)

    def log_outputs(self, outputs, inputs, losses, step, log_images, phase):
        super().log_outputs(outputs, inputs, losses, step, log_images, phase)
        if self._hp.attach_inv_mdl:
            self.inv_mdl.log_outputs(outputs, inputs, losses, step, log_images, phase)
        
        if log_images:
            dataset = self._hp.dataset_class
            if 'regressed_state' in outputs:
                self._logger.log_dataset_specific_trajectory(outputs, inputs, "regressed_state_topdown", step, phase,
                                                             dataset, predictions=outputs.regressed_state,
                                                             end_inds=inputs.end_ind)
    
            if 'regressed_state' in outputs and self._hp.attach_inv_mdl and 'actions' in outputs:
                if len(outputs.actions.shape) == 3:
                    actions = outputs.actions
                else:
                    # Training, need to get the action sequence
                    actions = self.inv_mdl(inputs, full_seq=True).actions
                    
                cum_action_traj = torch.cat((outputs.regressed_state[:, :1], actions), dim=1).cumsum(1)
                self._logger.log_dataset_specific_trajectory(outputs, inputs, "action_traj_topdown", step, phase, dataset,
                                                             predictions=cum_action_traj, end_inds=inputs.end_ind)
    
            if not self._hp.use_convs:
                self._logger.log_dataset_specific_trajectory(outputs, inputs, "prediction_topdown", step, phase, dataset)
                
                # TODO remove this
                if self._hp.log_states_2d:
                    self._logger.log_states_2d(outputs, inputs, "prediction_states_2d", step, phase)
                    
                if self._hp.train_on_action_seqs:
                    action_seq = outputs.dense_rec.images
                    cum_action_seq = torch.cumsum(action_seq, dim=1)
                    self._logger.log_dataset_specific_trajectory(outputs, inputs, "cum_action_prediction_topdown", step,
                                                                 phase, dataset, predictions=cum_action_seq,
                                                                 end_inds=inputs.end_ind)

        if self._hp.dump_encodings:
            os.makedirs(self._logger._log_dir + '/stored_data/', exist_ok=True)
            torch.save(subdict(inputs, ['enc_traj_seq', 'traj_seq', 'traj_seq_states', 'actions']),
                       self._logger._log_dir + '/stored_data/encodings_{}'.format(step))

        if self._hp.dump_encodings_inv_model:
            os.makedirs(self._logger._log_dir + '/stored_data_inv_model/', exist_ok=True)
            torch.save(subdict(inputs, ['model_enc_seq', 'traj_seq_states', 'actions']),
                       self._logger._log_dir + '/stored_data_inv_model/encodings_{}.th'.format(step))

    @property
    def prune_sequences(self):
        return self._hp.attach_inv_mdl or (self._hp.attach_cost_mdl and self._hp.run_cost_mdl) \
               or self._hp.attach_state_regressor

    def get_predicted_pruned_seqs(self, inputs, outputs):
        raise NotImplementedError

    def get_matched_pruned_seqs(self, inputs, outputs):
        name = 'encodings' if outputs.dense_rec else 'e_g_prime'  # for SVG vs tree
        if 'dtw' in self._hp.matching_type:
            # use precomputed matching dists for pruning
            matched_latents = self.tree_module.binding.get_matched_sequence(outputs.tree, 'e_g_prime')
            
            batch, time = inputs.traj_seq.shape[:2]
            model_enc_seq = [matched_latents[i_ex, :inputs.end_ind[i_ex] + 1] for i_ex in range(batch)]
            model_enc_seq = model_enc_seq
        else:
            # batched collection for SVG and balanced tree
            model_enc_seq = self.dense_rec.get_all_samples_with_len(
                inputs.end_ind, outputs, inputs, 'basic', name=name)[0]
        return model_enc_seq

