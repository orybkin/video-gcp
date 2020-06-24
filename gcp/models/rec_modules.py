import torch
import torch.nn as nn
from blox.tensor.ops import batchwise_assign, broadcast_final
from blox import AttrDict
from blox.torch.losses import L2Loss
from blox.torch.recurrent_modules import ReinitLSTMCell

# Note: this module is no longer used


class RecBase(nn.Module):
    """ Base module for dense reconstruction. Handles skip connections loss, and action decoding

    """
    
    def __init__(self, hp, decoder):
        super().__init__()
        self._hp = hp
        self.decoder = decoder
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def _dense_decode(self, inputs, encodings, seq_len):
        return self.decoder.decode_seq(inputs, encodings)
    
    def loss(self, inputs, model_output, extra_action=True, first_image=True):
        return self.decoder.loss(inputs, model_output, extra_action, first_image)


class ResetRnnRecBase(RecBase):
    """ Base module for reset-based dense reconstruction. Handles the ReinitLSTM. Can be modified by overriding
    the function that decides the inputs to the LSTM.
    
    """
    def __init__(self, hp, input_size, output_size, hidden_size, n_layers, decoder):
        super().__init__(hp, decoder)
        self.lstm = ReinitLSTMCell(hp, input_size, output_size, hidden_size, n_layers, self._hp.nz_enc * 2).make_lstm()

    def _get_lstm_inputs(self, root, inputs):
        raise NotImplementedError("Needs to be implemented in base class!")

    def forward(self, root, inputs):
        # TODO implement stopping probability prediction
        # TODO make the low-level network not predict subgoals
        batch_size, time = self._hp.batch_size, self._hp.max_seq_len
        outputs = AttrDict()

        lstm_inputs = self._get_lstm_inputs(root, inputs)
        lstm_outputs = self.lstm(lstm_inputs, time)
        outputs.encodings = torch.stack(lstm_outputs, dim=1)
        outputs.update(self._dense_decode(inputs, outputs.encodings, time))
        return outputs


class DiscreteDenseRecModule(ResetRnnRecBase):
    """ Peforms dense reconstruction on a subgoal tree. The inputs to the LSTM are the start and the end of the segment
    for each frame.
    
    """
    
    def _get_lstm_inputs(self, root, inputs):
        """
        :param root:
        :return:
        """
        device = inputs.reference_tensor.device
        batch_size, time = self._hp.batch_size, self._hp.max_seq_len
        fullseq_shape = [batch_size, time] + list(inputs.enc_e_0.shape[1:])
        lstm_inputs = AttrDict()

        e_0s = torch.zeros(fullseq_shape, dtype=torch.float32, device=device)
        e_gs = torch.zeros(fullseq_shape, dtype=torch.float32, device=device)
        reset_indicator = torch.zeros((batch_size, time), dtype=torch.uint8, device=device)
        for segment in root.full_tree():  # traversing the tree in breadth-first order.
            if segment.depth == 0:  # if leaf-node
                e_0, e_g = segment.e_0, segment.e_g
                
                # TODO iterating over batch must be gone
                for ex in range(self._hp.batch_size):
                    e_0s[ex, segment.start_ind[ex]:segment.end_ind[ex]] = e_0[ex]
                    e_gs[ex, segment.start_ind[ex]:segment.end_ind[ex]] = e_g[ex]
                    
        lstm_inputs.cell_input = e_gs
        lstm_inputs.reset_indicator = reset_indicator
        lstm_inputs.reset_input = torch.cat([e_gs, e_0s], dim=2)
        
        # TODO compute the latent variable with another LSTM
        #      TODO the lstm has to observe timesteps
        #      TODO aggregate the latent variables in the loop above
        # TODO add the latent variable to reset inputs
        return lstm_inputs


class Soft2FramesRecModule(ResetRnnRecBase):
    """ Peforms dense reconstruction on a subgoal tree via linear interpolation within segments."""

    def _get_lstm_inputs(self, root, inputs):
        """
        :param root:
        :return:
        """
        device = inputs.reference_tensor.device
        batch_size, time = self._hp.batch_size, self._hp.max_seq_len
        fullseq_shape = [batch_size, time] + list(inputs.enc_e_0.shape[1:])
        lstm_inputs = AttrDict()
        
        # collect start and end indexes and values of all segments
        e_0s = torch.zeros(fullseq_shape, dtype=torch.float32, device=device)
        e_gs = torch.zeros(fullseq_shape, dtype=torch.float32, device=device)
        start_inds, end_inds = torch.zeros((batch_size, time), dtype=torch.float32, device=device), \
                               torch.zeros((batch_size, time), dtype=torch.float32, device=device)
        reset_indicator = torch.zeros((batch_size, time), dtype=torch.uint8, device=device)
        for segment in root.full_tree():  # traversing the tree in breadth-first order.
            if segment.depth == 0:  # if leaf-node
                start_ind = torch.ceil(segment.start_ind).type(torch.LongTensor)
                end_ind = torch.floor(segment.end_ind).type(torch.LongTensor)
                batchwise_assign(reset_indicator, start_ind, 1)

                # TODO iterating over batch must be gone
                for ex in range(self._hp.batch_size):
                    if start_ind[ex] > end_ind[ex]: continue   # happens if start and end floats have no int in between
                    e_0s[ex, start_ind[ex]:end_ind[ex]+1] = segment.e_0[ex]     # +1 for including end_ind frame
                    e_gs[ex, start_ind[ex]:end_ind[ex]+1] = segment.e_g[ex]
                    start_inds[ex, start_ind[ex]:end_ind[ex]+1] = segment.start_ind[ex]
                    end_inds[ex, start_ind[ex]:end_ind[ex]+1] = segment.end_ind[ex]

        # perform linear interpolation
        time_steps = torch.arange(time, dtype=torch.float, device=device)
        inter = (time_steps - start_inds) / (end_inds - start_inds + 1e-7)
        
        lstm_inputs.reset_indicator = reset_indicator
        lstm_inputs.cell_input = (e_gs - e_0s) * broadcast_final(inter, e_gs) + e_0s
        lstm_inputs.reset_input = torch.cat([e_gs, e_0s], dim=2)
        
        return lstm_inputs


class SoftNFramesRecModule(RecBase):
    """ A module for dense reconstruction.
    The module performs latent interpolation over all latents by constructing a distribution using temporal distance
    to the reconstructed frame as the energy function.
    
    Note: this class does not reset the LSTM.
    TODO unify the class with the reset-based class and with averaging-based losses
    """
    def __init__(self, hp, decoder):
        super().__init__(hp, decoder)

    def forward(self, root, inputs):
        outputs = AttrDict()
        # TODO implement soft interpolation

        sg_times, sg_encs = [], []
        for segment in root:
            sg_times.append(segment.subgoal.ind)
            sg_encs.append(segment.subgoal.e_g_prime)
        sg_times = torch.stack(sg_times, dim=1)
        sg_encs = torch.stack(sg_encs, dim=1)

        # compute time difference weights
        seq_length = self._hp.max_seq_len
        target_ind = torch.arange(end=seq_length, dtype=sg_times.dtype)
        time_diffs = torch.abs(target_ind[None, None, :] - sg_times[:, :, None])
        weights = nn.functional.softmax(-time_diffs, dim=-1)

        # compute weighted sum outputs
        weighted_sg = weights[:, :, :, None, None, None] * sg_encs.unsqueeze(2).repeat(1, 1, seq_length, 1, 1, 1)
        outputs.encodings = torch.sum(weighted_sg, dim=1)
        outputs.update(self._dense_decode(inputs, outputs.encodings, seq_length))
        return outputs