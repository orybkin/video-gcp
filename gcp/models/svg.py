import torch
import torch.nn as nn

from blox.tensor.ops import broadcast_final, get_dim_inds
from blox import AttrDict
from blox.torch.dist import Gaussian, ProbabilisticModel
from blox.torch.losses import KLDivLoss2
from blox.torch.models.vrnn import VRNNCell

from gcp.evaluation.evaluation_matching import DTWEvalMatcher
from gcp.models.gcp_model import GCPModel
from blox.torch.layers import BaseProcessingNet


class SVGRecModule(nn.Module):
    """
    """
    
    def __init__(self, hp, input_size, output_size, decoder):
        # TODO make test time version
        assert input_size == output_size
        super().__init__()
        self._hp = hp
        self.decoder = decoder
        
        context_size = 0
        if hp.context_every_step:
            context_size += hp.nz_enc * 2
        if hp.action_conditioned_pred:
            context_size += hp.nz_enc
            
        self.lstm = VRNNCell(hp, input_size, context_size, hp.nz_enc * 2).make_lstm()
        self.eval_matcher = DTWEvalMatcher(hp)
        if hp.skip_from_parents:
            raise NotImplementedError("SVG doesn't support skipping from parents")
    
    def forward(self, root, inputs):
        lstm_inputs = AttrDict()
        initial_inputs = AttrDict(x=inputs.enc_e_0)
        context = torch.cat([inputs.enc_e_0, inputs.enc_e_g], dim=1)
        static_inputs = AttrDict()

        if 'enc_demo_seq' in inputs:
            lstm_inputs.x_prime = inputs.enc_demo_seq[:, 1:]
        if 'z' in inputs:
            lstm_inputs.z = inputs.z
        if self._hp.context_every_step:
            static_inputs.context = context
        if self._hp.action_conditioned_pred:
            assert 'enc_action_seq' in inputs   # need to feed actions for action conditioned predictor
            lstm_inputs.update(more_context=inputs.enc_action_seq)
        
        self.lstm.cell.init_state(initial_inputs.x, context, lstm_inputs.get('more_context', None))
        # Note: the last image is also produced. The actions are defined as going to the image
        outputs = self.lstm(inputs=lstm_inputs,
                            initial_inputs=initial_inputs,
                            static_inputs=static_inputs,
                            length=self._hp.max_seq_len - 1)
        outputs.encodings = outputs.pop('x')
        outputs.update(self.decoder.decode_seq(inputs, outputs.encodings))
        outputs.images = torch.cat([inputs.I_0[:, None], outputs.images], dim=1)
        return outputs
    
    def loss(self, inputs, model_output, log_error_arr=False):
        losses = self.decoder.loss(inputs, model_output, extra_action=False, log_error_arr=log_error_arr)
        
        # TODO don't place loss on the final image
        weights = broadcast_final(inputs.pad_mask[:, 1:], model_output.p_z.mu)
        losses.kl = KLDivLoss2(self._hp.kl_weight, breakdown=1, free_nats_per_dim=self._hp.free_nats)\
            (model_output.q_z, model_output.p_z, weights=weights, log_error_arr=log_error_arr)
        
        return losses
    
    def get_sample_with_len(self, i_ex, len, model_output, inputs, pruning_scheme, name=None):
        """
        :param i_ex:  example index
        :param input_seq:
        :param model_output:
        :return gen_images
        """
        
        if pruning_scheme == 'dtw':
            # Cut the first image off for DTW - it is the GT image
            targets = inputs.demo_seq[i_ex, 1:inputs.end_ind[i_ex] + 1]
            estimates = model_output.dense_rec.images[i_ex, 1:inputs.end_ind[i_ex] + 1]
            images, matching_output = self.eval_matcher(None, None, None, None, targets=targets, estimates=estimates)
            # TODO clean up
            # Add the first image back (eval cuts it off)..
            return torch.cat([model_output.dense_rec.images[i_ex, [1]], images], dim=0), matching_output
        elif pruning_scheme == 'basic':
            if name is None:
                return model_output.dense_rec.images[i_ex, :len], None
            elif name == 'encodings':
                # TODO fix this. This is necessary because the Hierarchical model outputs the first latent too.
                # This concatenates the first encoder latent to compensate
                return torch.cat((inputs.enc_e_0[i_ex][None], model_output.dense_rec[name][i_ex]), 0)[:len], None
            else:
                return model_output.dense_rec[name][i_ex, :len], None

    def get_all_samples_with_len(self, end_idxs, model_output, inputs, pruning_scheme, name=None):
        return [self.get_sample_with_len(b, end_idxs[b]+1, model_output, inputs, pruning_scheme, name=name)[0]
                for b in range(end_idxs.shape[0])], None
            

class SVGModel(GCPModel):
    def build_network(self, build_encoder=True):
        super().build_network(build_encoder)
        
        self.dense_rec = SVGRecModule(**self._get_rec_class_args())
        
        if self._hp.action_conditioned_pred:
            self.action_encoder = BaseProcessingNet(self._hp.n_actions, self._hp.nz_mid, self._hp.nz_enc,
                                                    self._hp.n_processing_layers, self._hp.fc_builder)
            
    def predict_sequence(self, inputs, outputs, start_ind, end_ind, phase):
        outputs = AttrDict(dense_rec=self.dense_rec(None, inputs))
        return outputs

    def log_outputs(self, model_output, inputs, losses, step, log_images, phase):
        super().log_outputs(model_output, inputs, losses, step, log_images, phase)

        if log_images:
            if model_output.dense_rec and self._hp.use_convs:
                self._logger.log_dense_gif(model_output, inputs, "dense_rec", step, phase)
                
                log_prior_images = False
                if log_prior_images:
                    # Run the model N times
                    with torch.no_grad(), self.val_mode():
                        rows = list([self(inputs).dense_rec.images for i in range(4)])
                    self._logger.log_rows_gif(rows, "prior_samples", step, phase)

    def get_predicted_pruned_seqs(self, inputs, model_output):
        return [seq[:end_ind+1] for seq, end_ind in zip(model_output.dense_rec.encodings, model_output.end_ind)]
