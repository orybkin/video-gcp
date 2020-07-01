import numpy as np
import torch
import torch.nn as nn

import gcp.prediction.models.adaptive_binding.adaptive
import gcp.prediction.models.tree.frame_binding
from blox import AttrDict
from blox.basic_types import subdict
from blox.torch.dist import safe_entropy, ProbabilisticModel
from blox.torch.losses import PenaltyLoss
from blox.torch.subnetworks import GeneralizedPredictorModel, Predictor
from gcp.prediction.models.tree.tree_lstm import build_tree_lstm
from blox.torch.variational import setup_variational_inference
from gcp.prediction.models.adaptive_binding.attentive_inference import AttentiveInference
from gcp.prediction.utils.tree_utils import SubgoalTreeLayer
from gcp.prediction.models.tree.inference import Inference


class TreeModule(nn.Module, ProbabilisticModel):
    def __init__(self, hp, decoder):
        nn.Module.__init__(self)
        ProbabilisticModel.__init__(self)
        self._hp = hp
        self.decoder = decoder

        self.build_network()

    def build_network(self):
        hp = self._hp

        q, self.prior = setup_variational_inference(self._hp, self._hp.nz_enc, self._hp.nz_enc * 2)
        if self._hp.attentive_inference:
            self.inference = AttentiveInference(self._hp, q)
        else:
            self.inference = Inference(self._hp, q)

        # todo clean this up with subclassing?
        pred_inp_dim = hp.nz_enc * 2 + hp.nz_vae
        if self._hp.context_every_step:
            pred_inp_dim = pred_inp_dim + hp.nz_enc * 2
            
        if hp.tree_lstm:
            self.subgoal_pred, self.lstm_initializer = build_tree_lstm(hp, pred_inp_dim, hp.nz_enc)
        else:
            self.subgoal_pred = GeneralizedPredictorModel(hp, input_dim=pred_inp_dim, output_dims=[hp.nz_enc],
                                                          activations=[None])

        self.build_binding()
        
        if self._hp.regress_index:
            self.index_predictor = Predictor(
                self._hp, self._hp.nz_enc * 2, self._hp.max_seq_len, detached=False, spatial=False)

    def build_binding(self):
        # TODO this has to be specified with classes, not strings
        hp = self._hp
        
        if self._hp.matching_type == 'balanced':
            binding_class = gcp.prediction.models.tree.frame_binding.BalancedBinding
        elif 'dtw' in self._hp.matching_type:
            binding_class = gcp.prediction.models.adaptive_binding.adaptive.AdaptiveBinding
        else:
            raise NotImplementedError
        
        self.binding = binding_class(hp, self.decoder)

    def produce_subgoal(self, inputs, layerwise_inputs, start_ind, end_ind, left_parent, right_parent, depth=None):
        """
        Divides the subsequence by producing a subgoal inside it.
         This function represents one step of recursion of the model
        """
        subgoal = AttrDict()

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
            if self._hp.attentive_inference:
                subgoal.update(self.inference(inputs, e_l, e_r, start_ind, end_ind))
            else:
                subgoal.match_timesteps = self.binding.comp_timestep(left_parent.match_timesteps,
                                                                     right_parent.match_timesteps)
                subgoal.update(self.inference(inputs, e_l, e_r, start_ind, end_ind, subgoal.match_timesteps.float()))
                
            z = subgoal.q_z.sample()

        ## Predict the next node
        pred_input = [e_l, e_r, z]
        if self._hp.context_every_step:
            mult = int(z.shape[0] / inputs.e_0.shape[0])
            pred_input += [inputs.e_0.repeat_interleave(mult, 0),
                           inputs.e_g.repeat_interleave(mult, 0)]
        
        if self._hp.tree_lstm:
            if left_parent.hidden_state is None and right_parent.hidden_state is None:
                left_parent.hidden_state, right_parent.hidden_state = self.lstm_initializer(e_l, e_r, z)
                
            subgoal.hidden_state, subgoal.e_g_prime = \
                self.subgoal_pred(left_parent.hidden_state, right_parent.hidden_state, *pred_input)
        else:
            subgoal.e_g_prime_preact = self.subgoal_pred(*pred_input)
            subgoal.e_g_prime = torch.tanh(subgoal.e_g_prime_preact)

        subgoal.ind = (start_ind + end_ind) / 2     # gets overwritten w/ argmax of matching at training time (in loss)
        return subgoal, left_parent, right_parent

    def loss(self, inputs, outputs):
        if outputs.tree.depth == 0:
            return {}

        losses = AttrDict()

        losses.update(self.get_node_loss(inputs, outputs))

        # Explaining loss
        losses.update(self.binding.loss(inputs, outputs))
        
        # entropy penalty
        losses.entropy = PenaltyLoss(weight=self._hp.entropy_weight)(outputs.entropy)

        return losses

    def compute_matching(self, inputs, outputs):
        """ Match the tree nodes to ground truth and compute relevant values """
        tree = outputs.tree
    
        # compute matching distributions
        if 'gt_match_dists' in outputs:
            gt_match_dists = outputs.gt_match_dists
        else:
            gt_match_dists = self.binding.get_w(inputs.pad_mask, inputs, outputs, log=True)
            
        tree.bf.match_dist = outputs.gt_match_dists = gt_match_dists

        # compute additional vals
        outputs.entropy = safe_entropy(outputs.gt_match_dists, dim=-1)
        # probability of the node existing
        tree.bf.p_n = outputs.p_n = torch.sum(outputs.gt_match_dists, dim=2).clamp(0, 1)

    def get_node_loss(self, inputs, outputs):
        """ Reconstruction and KL divergence loss """
        losses = AttrDict()
        tree = outputs.tree
        
        losses.update(self.binding.reconstruction_loss(inputs, outputs))
        losses.update(self.inference.loss(tree.bf.q_z, tree.bf.p_z))

        return losses

    @staticmethod
    def _log_outputs(outputs, inputs, losses, step, log_images, phase, logger):
        if log_images:
            # Log layerwise loss
            layerwise_keys = ['dense_img_rec', 'kl'] & losses.keys()
            for name, loss in subdict(losses, layerwise_keys).items():
                if len(loss.error_mat.shape) > 2:   # reduce to two dimensions
                    loss.error_mat = loss.error_mat.mean([i for i in range(len(loss.error_mat.shape))][2:])
                layerwise_loss = SubgoalTreeLayer.split_by_layer_bf(loss.error_mat, dim=1)
                layerwise_loss = torch.tensor([l[l != 0].mean() for l in layerwise_loss])
                logger.log_graph(layerwise_loss, '{}_{}'.format(name, 'loss_layerwise'), step, phase)
