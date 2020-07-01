import torch
import torch.nn as nn

from blox import AttrDict
from blox.tensor.ops import broadcast_final
from blox.torch.layers import BaseProcessingNet
from blox.torch.losses import KLDivLoss2
from blox.torch.models.vrnn import VRNNCell
from gcp.evaluation.evaluation_matching import DTWEvalBinding
from gcp.prediction.models.base_gcp import BaseGCPModel


class SequentialRecModule(nn.Module):

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
        self.eval_binding = DTWEvalBinding(hp)
        if hp.skip_from_parents:
            raise NotImplementedError("SVG doesn't support skipping from parents")
    
    def forward(self, root, inputs):
        lstm_inputs = AttrDict()
        initial_inputs = AttrDict(x=inputs.e_0)
        context = torch.cat([inputs.e_0, inputs.e_g], dim=1)
        static_inputs = AttrDict()

        if 'enc_traj_seq' in inputs:
            lstm_inputs.x_prime = inputs.enc_traj_seq[:, 1:]
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
    
    def loss(self, inputs, outputs, log_error_arr=False):
        losses = self.decoder.loss(inputs, outputs, extra_action=False, log_error_arr=log_error_arr)
        
        # TODO don't place loss on the final image
        weights = broadcast_final(inputs.pad_mask[:, 1:], outputs.p_z.mu)
        losses.kl = KLDivLoss2(self._hp.kl_weight, breakdown=1, free_nats_per_dim=self._hp.free_nats)\
            (outputs.q_z, outputs.p_z, weights=weights, log_error_arr=log_error_arr)
        
        return losses
    
    def get_sample_with_len(self, i_ex, len, outputs, inputs, pruning_scheme, name=None):
        """
        :param i_ex:  example index
        :param input_seq:
        :param outputs:
        :return gen_images
        """
        
        if pruning_scheme == 'dtw':
            # Cut the first image off for DTW - it is the GT image
            targets = inputs.traj_seq[i_ex, 1:inputs.end_ind[i_ex] + 1]
            estimates = outputs.dense_rec.images[i_ex, 1:inputs.end_ind[i_ex] + 1]
            images, matching_output = self.eval_binding(None, None, None, None, targets=targets, estimates=estimates)
            # TODO clean up
            # Add the first image back (eval cuts it off)..
            return torch.cat([outputs.dense_rec.images[i_ex, [1]], images], dim=0), matching_output
        elif pruning_scheme == 'basic':
            if name is None:
                return outputs.dense_rec.images[i_ex, :len], None
            elif name == 'encodings':
                # TODO fix this. This is necessary because the Hierarchical model outputs the first latent too.
                # This concatenates the first encoder latent to compensate
                return torch.cat((inputs.e_0[i_ex][None], outputs.dense_rec[name][i_ex]), 0)[:len], None
            else:
                return outputs.dense_rec[name][i_ex, :len], None

    def get_all_samples_with_len(self, end_idxs, outputs, inputs, pruning_scheme, name=None):
        return [self.get_sample_with_len(b, end_idxs[b] + 1, outputs, inputs, pruning_scheme, name=name)[0]
                for b in range(end_idxs.shape[0])], None
            

class SequentialModel(BaseGCPModel):
    def build_network(self, build_encoder=True):
        super().build_network(build_encoder)
        
        self.dense_rec = SequentialRecModule(
            hp=self._hp, input_size=self._hp.nz_enc, output_size=self._hp.nz_enc, decoder=self.decoder)
        
        if self._hp.action_conditioned_pred:
            self.action_encoder = BaseProcessingNet(self._hp.n_actions, self._hp.nz_mid, self._hp.nz_enc,
                                                    self._hp.n_processing_layers, self._hp.fc_builder)
            
    def predict_sequence(self, inputs, outputs, start_ind, end_ind, phase):
        outputs = AttrDict(dense_rec=self.dense_rec(None, inputs))
        return outputs

    def log_outputs(self, outputs, inputs, losses, step, log_images, phase):
        super().log_outputs(outputs, inputs, losses, step, log_images, phase)

        if log_images:
            if outputs.dense_rec and self._hp.use_convs:
                self._logger.log_dense_gif(outputs, inputs, "dense_rec", step, phase)
                
                log_prior_images = False
                if log_prior_images:
                    # Run the model N times
                    with torch.no_grad(), self.val_mode():
                        rows = list([self(inputs).dense_rec.images for i in range(4)])
                    self._logger.log_rows_gif(rows, "prior_samples", step, phase)

    def get_predicted_pruned_seqs(self, inputs, outputs):
        return [seq[:end_ind+1] for seq, end_ind in zip(outputs.dense_rec.encodings, outputs.end_ind)]
