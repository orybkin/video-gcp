import numpy as np

import gcp.models.trees.balanced
import gcp.models.trees.dtw
import gcp.models.trees.fraction
import gcp.models.trees.tap
import gcp.rec_planner_utils.matching as matching
import torch
import torch.nn as nn
from blox.torch.ops import like, ar2ten
from blox import AttrDict
from blox.basic_types import subdict
from blox.torch.dist import safe_entropy, Gaussian, ProbabilisticModel
from blox.torch.variational import setup_variational_inference, AttentiveInference
from blox.torch.losses import L2Loss, PenaltyLoss, CELogitsLoss
from blox.torch.modules import ExponentialDecayUpdater, DummyModule, ConstantUpdater
from blox.torch.recurrent_modules import ZeroLSTMCellInitializer, MLPLSTMCellInitializer
from blox.torch.subnetworks import GeneralizedPredictorModel, Predictor, SumTreeHiddenStatePredictorModel,\
    LinTreeHiddenStatePredictorModel, SplitLinTreeHiddenStatePredictorModel, Attention
from gcp.evaluation.evaluation_matching import DTWEvalMatcher, BalancedEvalMatcher, \
    BalancedPrunedDTWMatcher
from gcp.rec_planner_utils.recplan_losses import FramesAveragingCriterion, \
    LossAveragingCriterion, ExpectationCriterion, LossFramesAveragingCriterion
from gcp.rec_planner_utils.tree_utils import SubgoalTreeLayer
from gcp.rec_planner_utils.tree_utils import depthfirst2breadthfirst


class SHPredModule(nn.Module, ProbabilisticModel):
    def __init__(self, hp, decoder):
        nn.Module.__init__(self)
        ProbabilisticModel.__init__(self)
        self._hp = hp
        self.decoder = decoder
        self.predict_fraction = False

        assert not self._hp.teacher_forcing  # cannot do teacher forcing in SH-Pred model

        self.build_network()

    def build_network(self):
        hp = self._hp

        q, self.prior = setup_variational_inference(self._hp, self._hp.nz_enc, self._hp.nz_enc * 2)
        self.inference = AttentiveInference(self._hp, q, Attention(self._hp))

        # todo clean this up with subclassing?
        pred_inp_dim = hp.nz_enc * 2 + hp.nz_vae
        if self._hp.var_inf is '2layer':
            pred_inp_dim = pred_inp_dim + hp.nz_vae2
        if self._hp.context_every_step:
            pred_inp_dim = pred_inp_dim + hp.nz_enc * 2
            
        if hp.tree_lstm:
            if hp.tree_lstm == 'sum':
                cls = SumTreeHiddenStatePredictorModel
            elif hp.tree_lstm == 'linear':
                cls = LinTreeHiddenStatePredictorModel
            elif hp.tree_lstm == 'split_linear':
                cls = SplitLinTreeHiddenStatePredictorModel
            else:
                raise ValueError("don't know this TreeLSTM type")
                
            self.subgoal_pred = cls(hp, input_dim=pred_inp_dim, output_dim=hp.nz_enc)
            self.lstm_initializer = self._get_lstm_initializer(self.subgoal_pred)
        else:
            self.subgoal_pred = GeneralizedPredictorModel(hp, input_dim=pred_inp_dim, output_dims=[hp.nz_enc],
                                                          activations=[None])

        # TODO this can be moved into matcher
        self.criterion = LossAveragingCriterion(self._hp)
        self.build_matcher()
        
        if self.predict_fraction:
            # TODO implement the inference side version of this
            # TODO put this inside the matcher
            input_size = hp.nz_enc * 2 if hp.timestep_cond_attention else hp.nz_enc * 3
            self.fraction_pred = Predictor(hp, input_size, output_dim=1, spatial=False,
                                           final_activation=nn.Sigmoid())

        if self._hp.regress_index:
            self.index_predictor = Predictor(
                self._hp, self._hp.nz_enc * 2, self._hp.max_seq_len, detached=False, spatial=False)

    # @lazy_property
    # def index_predictor(self):
    #     """ Using lazy property is convenient for this since it enforces that this is only constructed if needed """
    #     p = Predictor(self._hp, self._hp.nz_enc * 2 + 1, 1, detached=False, spatial=False)
    #     p.to(self._hp.device)  # TODO streamline this
    #     # TODO this doesn't work with restoring parameters
    #     # TODO this doesn't work with optimizer either
    #     # The only way I see this working is if we 'construct' the network explicitly in the training loop by calling
    #     # forward.
    #     # On the flip side, if we do this, we can get rid of specifying input dims for most tensors.
    #     return p

    def build_matcher(self):
        # TODO this has to be specified with classes, not strings
        hp = self._hp
        
        if self._hp.matching_type == 'fraction':
            matcher_class = gcp.models.trees.fraction.FractionMatcher
            self.predict_fraction = True
        elif self._hp.matching_type == 'balanced':
            matcher_class = gcp.models.trees.balanced.BalancedMatcher
        elif self._hp.matching_type == 'tap':
            matcher_class = gcp.models.trees.tap.TAPMatcher
        elif 'dtw' in self._hp.matching_type:
            matcher_class = gcp.models.trees.dtw.DTWMatcher
        else:
            raise NotImplementedError
        
        self.matcher = matcher_class(hp, self.criterion, self.decoder)

    def _get_dense_rec_class(self):
        if self._hp.dense_rec_type == 'node_prob' or self._hp.dense_rec_type == 'none':
            return SHDenseRec
        elif self._hp.dense_rec_type == 'constant':
            return ConstantDenseRec

    def _filter_inputs_for_model(self, inputs, phase):
        keys = ['I_0', 'I_g', 'skips', 'start_ind', 'end_ind', 'enc_e_0', 'enc_e_g', 'z']
        if phase == 'train': keys += ['inf_enc_seq', 'inf_enc_key_seq']
        return subdict(inputs, keys, strict=False)

    def _get_lstm_initializer(self, cell):
        if self._hp.lstm_init == 'zero':
            return ZeroLSTMCellInitializer(self._hp, cell)
        elif self._hp.lstm_init == 'mlp':
            return MLPLSTMCellInitializer(self._hp, cell, 2 * self._hp.nz_enc + self._hp.nz_vae)
        else:
            raise ValueError('dont know lstm init type {}!'.format(self._hp.lstm_init))

    def produce_subgoal(self, inputs, layerwise_inputs, start_ind, end_ind, left_parent, right_parent, depth=None):
        """
        Divides the subsequence by producing a subgoal inside it.
         This function represents one step of recursion of the model
        """
        subgoal = AttrDict()
        batch_size = start_ind.shape[0]

        e_l = left_parent.e_g_prime
        e_r = right_parent.e_g_prime

        subgoal.p_z = self.prior(e_l, e_r)

        if 'z' in layerwise_inputs:
            z = layerwise_inputs.z
            if self._hp.prior_type == 'learned':    # reparametrize if learned prior is used
                z = subgoal.p_z.reparametrize(z)
        elif self._sample_prior:
            z = subgoal.p_z.sample()
        else:
            ## Inference
            if (self._hp.timestep_cond_attention or self._hp.forced_attention):
                subgoal.fraction = self.fraction_pred(e_l, e_r)[..., -1] if self.predict_fraction else None
                subgoal.match_timesteps = self.matcher.comp_timestep(left_parent.match_timesteps,
                                                                     right_parent.match_timesteps,
                                                                     subgoal.fraction[:,
                                                                     None] if subgoal.fraction is not None else None)
                subgoal.update(self.inference(inputs, e_l, e_r, start_ind, end_ind, subgoal.match_timesteps.float()))
            else:
                subgoal.update(self.inference(
                    inputs, e_l, e_r, start_ind, end_ind, attention_weights=layerwise_inputs.safe.attention_weights))
                
            z = subgoal.q_z.sample()

        ## Predict the next node
        pred_input = [e_l, e_r, z]
        if self._hp.context_every_step:
            mult = int(z.shape[0] / inputs.enc_e_0.shape[0])
            pred_input += [inputs.enc_e_0.repeat_interleave(mult, 0),
                           inputs.enc_e_g.repeat_interleave(mult, 0)]
        
        if self._hp.tree_lstm:
            if left_parent.hidden_state is None and right_parent.hidden_state is None:
                left_parent.hidden_state, right_parent.hidden_state = self.lstm_initializer(e_l, e_r, z)
                if self._hp.lstm_warmup_cycles > 0:
                    for _ in range(self._hp.lstm_warmup_cycles):
                        left_parent.hidden_state, __ = \
                            self.subgoal_pred(left_parent.hidden_state, right_parent.hidden_state, e_l, e_r, z)
                        right_parent.hidden_state = left_parent.hidden_state.clone()
                        
            subgoal.hidden_state, subgoal.e_g_prime = \
                self.subgoal_pred(left_parent.hidden_state, right_parent.hidden_state, *pred_input)
        else:
            subgoal.e_g_prime_preact = self.subgoal_pred(*pred_input)
            subgoal.e_g_prime = torch.tanh(subgoal.e_g_prime_preact)

        ## Additional predicted values
        if self.predict_fraction and not self._hp.timestep_cond_attention:
            subgoal.fraction = self.fraction_pred(e_l, e_r, subgoal.e_g_prime)[..., -1]     # remove unnecessary dim
            
        # add attention target if trained with attention supervision
        if self._hp.supervise_attn_weight > 0.0:
            frac = subgoal.fraction[:, None] if 'fraction' in subgoal and subgoal.fraction is not None else None
            subgoal.match_timesteps = self.matcher.comp_timestep(left_parent.match_timesteps,
                                                                 right_parent.match_timesteps,
                                                                 frac)

        subgoal.ind = (start_ind + end_ind) / 2     # gets overwritten w/ argmax of matching at training time (in loss)
        return subgoal, left_parent, right_parent

    def loss(self, inputs, model_output):
        if model_output.tree.depth == 0:
            return {}

        losses = AttrDict()

        if not 'gt_matching_dists' in model_output:     # potentially already computed in forward pass
            self.compute_matching(inputs, model_output)

        losses.update(self.get_node_loss(inputs, model_output))

        losses.update(self.get_extra_losses(inputs, model_output))
        
        # Explaining loss
        losses.update(self.matcher.loss(inputs, model_output))
        
        # entropy penalty
        losses.entropy = PenaltyLoss(weight=self._hp.entropy_weight)(model_output.entropy)

        return losses

    def compute_matching(self, inputs, model_output):
        """ Match the tree nodes to ground truth and compute relevant values """
        tree = model_output.tree
    
        # compute matching distributions
        if 'gt_match_dists' in model_output:
            gt_match_dists = model_output.gt_match_dists
        else:
            gt_match_dists = self.matcher.get_w(inputs.pad_mask, inputs, model_output, log=True)
            
        tree.bf.match_dist = model_output.gt_match_dists = gt_match_dists

        # compute additional vals
        model_output.entropy = safe_entropy(model_output.gt_match_dists, dim=-1)
        # probability of the node existing
        tree.bf.p_n = model_output.p_n = torch.sum(model_output.gt_match_dists, dim=2).clamp(0, 1)

    def get_node_loss(self, inputs, outputs):
        """ Reconstruction and KL divergence loss """
        losses = AttrDict()
        tree = outputs.tree
        
        # Weight of the loss
        kl_weights = weights = 1
        if self._hp.equal_weight_layer:
            top = 2 ** (self._hp.hierarchy_levels - 1)
            # For each layer, divide the weight by the number of elements in the layer
            weights = np.concatenate([np.full((2 ** l,), top / (2 ** l)) for l in range(self._hp.hierarchy_levels)])
            weights = torch.from_numpy(weights).to(self._hp.device).float()[None, :, None]
            kl_weights = weights[..., None, None]
        
        losses.update(self.matcher.reconstruction_loss(inputs, outputs, weights))
        
        losses.update(self.inference.loss(tree.bf.q_z, tree.bf.p_z, weights=kl_weights))

        return losses
    
    def get_extra_losses(self, inputs, model_output):
        losses = AttrDict()

        if self._hp.supervise_attn_weight > 0.0:
            raise NotImplementedError("This code is wrong: it needs logits, not distributions as input")
            
            gammas = model_output.tree.bf.gamma
            gammas = gammas.view(-1, gammas.shape[-1])
            targets = model_output.tree.bf.match_timesteps.long().view(-1)
            losses.forced_attention = CELogitsLoss(self._hp.supervise_attn_weight)(gammas, targets)

        if self._hp.regress_index:
            raise NotImplementedError
            key_index = ar2ten(self.criterion.get_index(inputs.demo_seq), self._hp.device)
            predicted_index = self.index_predictor(
                inputs.enc_e_0, inputs.enc_e_g)  # , make_one_hot(key_index, self._hp.max_seq_len).float())
            losses.regress_index = CELogitsLoss(100)(predicted_index, key_index)

        return losses

    @staticmethod
    def _log_outputs(model_output, inputs, losses, step, log_images, phase, logger):
        if log_images:
            # Log layerwise loss
            layerwise_keys = ['dense_img_rec', 'kl'] & losses.keys()
            for name, loss in subdict(losses, layerwise_keys).items():
                if len(loss.error_mat.shape) > 2:   # reduce to two dimensions
                    loss.error_mat = loss.error_mat.mean([i for i in range(len(loss.error_mat.shape))][2:])
                layerwise_loss = SubgoalTreeLayer.split_by_layer_bf(loss.error_mat, dim=1)
                layerwise_loss = torch.tensor([l[l != 0].mean() for l in layerwise_loss])
                logger.log_graph(layerwise_loss, '{}_{}'.format(name, 'loss_layerwise'), step, phase)

    def log_outputs_stateful(self, step, log_images, phase, logger):
        if phase == 'train':
            if hasattr(self, 'temp'):
                logger.log_scalar(self.temp, 'matching_temp', step, phase)
                

class SHDenseRec(DummyModule):
    def __init__(self, hp, *_, decoder, **__):
        super().__init__()
        self._hp = hp
        self.eval_matcher = None
        self.decoder = decoder

    def get_sample_with_len(self, i_ex, length, model_output, inputs, pruning_scheme, name=None):
        """Perform evaluation matching, return dense sequence of specified length."""
        if self.eval_matcher is None:
            self.eval_matcher = self._get_eval_matcher(pruning_scheme)
        return self.eval_matcher(model_output, inputs, length, i_ex, name)
    
    def get_all_samples_with_len(self, length, model_output, inputs, pruning_scheme, name=None):
        """Perform evaluation matching, return dense sequence of specified length."""
        if self.eval_matcher is None:
            self.eval_matcher = self._get_eval_matcher(pruning_scheme)
            
        if hasattr(self.eval_matcher, 'get_all_samples'):
            return self.eval_matcher.get_all_samples(model_output, inputs, length, name)
        else:
            return [self.eval_matcher(model_output, inputs, length, i_ex, name) for i_ex in range(model_output.end_ind.shape[0])]
 
    def _get_eval_matcher(self, pruning_scheme):
        if pruning_scheme == 'dtw':
            return DTWEvalMatcher(self._hp)
        if pruning_scheme == 'pruned_dtw':
            assert self._hp.matching_type == 'balanced'
            return BalancedPrunedDTWMatcher(self._hp)
        if pruning_scheme == 'basic':
            assert self._hp.matching_type == 'balanced'
            return BalancedEvalMatcher(self._hp)
        else:
            raise ValueError("Eval pruning scheme {} not currently supported!".format(pruning_scheme))
        
    def forward(self, tree, inputs):
        decoded_seq = self.decoder.decode_seq(inputs, tree.bf.e_g_prime)
        tree.set_attr_bf(**decoded_seq)
        return AttrDict()


class ConstantDenseRec(SHDenseRec):
    """ Dummy dense rec for debugging the binding """
    def get_sequence(self, inputs):
        # Create a sequence of indices and add zero elements
        seq = like(torch.eye, inputs.reference_tensor)(self._hp.max_seq_len, dtype=torch.float32)
        missing_length = (2 ** self._hp.hierarchy_levels - 1) - self._hp.max_seq_len
    
        def merge_seqs(seq1, seq2):
            mode = "shuffle"
            if mode == "last":
                seq = torch.cat([seq1, seq2], 0)
            elif mode == "second_to_last":
                seq = torch.cat([seq1[:-1], seq2, seq1[-1:]], 0)
            elif mode == "shuffle":
                assert (seq2 == 0).all()
                np.random.seed(11)
                ids = np.random.choice(self._hp.max_seq_len + missing_length, self._hp.max_seq_len, replace=False)
                ids.sort()
                seq = seq1.new_zeros(self._hp.max_seq_len + missing_length, seq1.shape[1])
                seq[ids] = seq1
    
            return seq

        seq = merge_seqs(seq, seq.new_zeros(missing_length, self._hp.max_seq_len))
        seq = depthfirst2breadthfirst(seq, dim=0)
        seq = seq[None].repeat_interleave(self._hp.batch_size, 0)
        return seq

    def forward(self, tree, inputs):
        seq = self.get_sequence(inputs)
        tree.set_attr_bf(images=seq, actions=None)
        return AttrDict()
