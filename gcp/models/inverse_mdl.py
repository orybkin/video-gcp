from contextlib import contextmanager

import numpy as np
import torch
from blox.tensor.ops import broadcast_final, batch_apply, batchwise_index, remove_spatial
from blox import AttrDict
from blox.basic_types import map_dict
from blox.torch.losses import L2Loss
from blox.torch.subnetworks import Predictor
from blox.torch.encoder_decoder import Encoder
from gcp.models.base_model import BaseModel
from gcp.rec_planner_utils.checkpointer import CheckpointHandler


class InverseModel(BaseModel):
    def __init__(self, params, logger):
        super().__init__(logger)
        self._hp = self._default_hparams()
        self.override_defaults(params)  # override defaults with config file
        self.postprocess_params()
        assert self._hp.n_actions != -1     # make sure action dimensionality was overridden
        self.build_network()

        # load only the encoder params during training
        if self._hp.enc_params_checkpoint is not None:
            assert self._hp.build_encoder   # provided checkpoint but did not build encoder
            self._load_weights([
                (self.encoder, 'encoder', self._hp.enc_params_checkpoint),
            ])
        self.detach_enc = not self._hp.finetune_enc

    def _default_hparams(self):
        # put new parameters in here:
        default_dict = {
            'ngf': 4,  # number of feature maps in shallowest level
            'nz_enc': 128,  # number of dimensions in encoder-latent space
            'nz_mid': 128,  # number of hidden units in fully connected layer
            'n_processing_layers': 3,  # Number of layers in MLPs
            'temp_dist': 1,  # sample temporal distances between 1 and temp_dist, regress only first action
            'enc_params_checkpoint': None,   # specify pretrained encoder weights to load for training
            'take_first_tstep': False,   # take only first and second time step, no shuffling.
            'use_states': False,
            'aggregate_actions': False,           # when taking two images that are more than one step apart sum the actions along that
            'pred_states': False,
            'finetune_enc': False,
            'checkpt_path': None,
            'build_encoder': True,      # if False, does not build an encoder, assumes that inputs are encoded from model
            'add_lstm_state_enc': False,    # if True, expects lstm state as additional encoded input
            'log_topdown_maze': False,
            'train_full_seq': False,
            'train_im0_enc': True,  # If True, the first frame latent is passed in as `enc_demo_seq`
        }

        # loss weights
        default_dict.update({
            'action_rec_weight': 1.0,
            'state_rec_weight': 1.0,
        })

        # misc params
        default_dict.update({
            'use_skips': False,
            'dense_rec_type': None,
            'device': None,
            'randomize_length': False,
        })

        # add new params to parent params
        parent_params = super()._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params
    
    def build_network(self, build_encoder=True):
        if self._hp.build_encoder:
            self.encoder = Encoder(self._hp)
        input_sz = self._hp.nz_enc * 3 if self._hp.add_lstm_state_enc else self._hp.nz_enc * 2
        self.action_pred = Predictor(self._hp, input_sz, self._hp.n_actions)

    def sample_offsets(self, end_ind):
        """
         # sample temporal distances between 1 and temp_dist, regress only first action
        :return:  None, call by reference
        """
        bs = end_ind.shape[0]
        if self._hp.take_first_tstep:
            t0 = torch.zeros(bs, device=self._hp.device).long()
            t1 = torch.ones_like(t0)
        else:
            t0 = np.zeros(bs)
            for b in range(bs):
                assert end_ind[b].cpu().numpy() >= self._hp.temp_dist
                t0[b] = np.random.randint(0, end_ind[b].cpu().numpy() - self._hp.temp_dist + 1, 1)
            delta_t = np.random.randint(1, self._hp.temp_dist + 1, bs)
            t1 = t0 + delta_t
            t0 = torch.tensor(t0, device=self._hp.device, dtype=torch.long)
            t1 = torch.tensor(t1, device=self._hp.device, dtype=torch.long)
        return t0, t1

    def index_input(self, input, t, aggregate=False, t1=None):
        if aggregate:
            assert t1 is not None       # need end time step for aggregation
            selected = torch.zeros_like(input[:, 0])
            for b in range(input.shape[0]):
                selected[b] = torch.sum(input[b, t[b]:t1[b]], dim=0)
        else:
            selected = batchwise_index(input, t)
        return selected

    def full_seq_forward(self, inputs):
        if 'model_enc_seq' in inputs:
            enc_seq_1 = inputs.model_enc_seq[:, 1:]
            if self._hp.train_im0_enc and 'enc_demo_seq' in inputs:
                enc_seq_0 = inputs.enc_demo_seq.reshape(inputs.enc_demo_seq.shape[:2] + (self._hp.nz_enc,))[:, :-1]
                enc_seq_0 = enc_seq_0[:, :enc_seq_1.shape[1]]
            else:
                enc_seq_0 = inputs.model_enc_seq[:, :-1]
        else:
            enc_seq = batch_apply([inputs.demo_seq], self.encoder)
            enc_seq_0, enc_seq_1 = enc_seq[:, :-1], enc_seq[:, 1:]

        if self.detach_enc:
            enc_seq_0 = enc_seq_0.detach()
            enc_seq_1 = enc_seq_1.detach()

        actions_pred = batch_apply(torch.cat([enc_seq_0, enc_seq_1], dim=2), self.action_pred)
       
        output = AttrDict()
        output.actions = actions_pred  #remove_spatial(actions_pred)
        if 'actions' in inputs:
            output.action_targets = inputs.actions
            output.pad_mask = inputs.pad_mask
        return output

    def forward(self, inputs, full_seq=None):
        """
        forward pass at training time
        :arg full_seq: if True, outputs actions for the full sequence, expects input encodings
        """
        if full_seq is None:
            full_seq = self._hp.train_full_seq
        
        if full_seq:
            return self.full_seq_forward(inputs)

        t0, t1 = self.sample_offsets(inputs.norep_end_ind if 'norep_end_ind' in inputs else inputs.end_ind)
        im0 = self.index_input(inputs.demo_seq, t0)
        im1 = self.index_input(inputs.demo_seq, t1)
        if 'model_enc_seq' in inputs:
            if self._hp.train_im0_enc and 'enc_demo_seq' in inputs:
                enc_im0 = self.index_input(inputs.enc_demo_seq, t0).reshape(inputs.enc_demo_seq.shape[:1] + (self._hp.nz_enc,))
            else:
                enc_im0 = self.index_input(inputs.model_enc_seq, t0)
            enc_im1 = self.index_input(inputs.model_enc_seq, t1)
        else:
            assert self._hp.build_encoder       # need encoder if no encoded latents are given
            enc_im0 = self.encoder.forward(im0)[0]
            enc_im1 = self.encoder.forward(im1)[0]

        if self.detach_enc:
            enc_im0 = enc_im0.detach()
            enc_im1 = enc_im1.detach()

        selected_actions = self.index_input(inputs.actions, t0, aggregate=self._hp.aggregate_actions, t1=t1)
        selected_states = self.index_input(inputs.demo_seq_states, t0)

        if self._hp.pred_states:
            actions_pred, states_pred = torch.split(self.action_pred(enc_im0, enc_im1), 2, 1)
        else:
            actions_pred = self.action_pred(enc_im0, enc_im1)

        output = AttrDict()
        output.actions = remove_spatial(actions_pred)
        output.action_targets = selected_actions
        output.state_targets = selected_states
        output.img_t0, output.img_t1 = im0, im1
        
        return output

    def loss(self, inputs, model_output, add_total=True):
        losses = AttrDict()

        # subgoal reconstruction loss
        n_action_output = model_output.actions.shape[1]
        loss_weights = broadcast_final(model_output.pad_mask[:, :n_action_output], inputs.actions) if 'pad_mask' in model_output else 1
        losses.action_reconst = L2Loss(self._hp.action_rec_weight)(model_output.actions, model_output.action_targets[:, :n_action_output], weights=loss_weights)
        if self._hp.pred_states:
            losses.state_reconst = L2Loss(self._hp.state_rec_weight)(model_output.states, model_output.state_targets)

        return losses

    def log_outputs(self, model_output, inputs, losses, step, log_images, phase):
        super()._log_losses(losses, step, log_images, phase)
        if log_images and len(inputs.demo_seq.shape) == 5:
            self._logger.log_pred_actions(model_output, inputs, 'pred_actions', step, phase)
        if self._hp.pred_states:
            self._logger.log_pred_states(model_output, inputs, 'pred_states', step, phase)

        if log_images:
            if len(model_output.actions.shape) == 3:
                actions = model_output.actions
            else:
                # Training, need to get the action sequence
                actions = self(inputs, full_seq=True).actions
    
            cum_action_traj = torch.cat((inputs.demo_seq_states[:, :1], actions), dim=1).cumsum(1)
            self._logger.log_maze_topdown(model_output, inputs, "action_traj_topdown", step, phase,
                                          predictions=cum_action_traj, end_inds=inputs.end_ind)
            
            cum_action_traj = torch.cat((inputs.demo_seq_states[:, :1], inputs.actions), dim=1).cumsum(1)
            self._logger.log_maze_topdown(model_output, inputs, "action_traj_gt_topdown", step, phase,
                                          predictions=cum_action_traj, end_inds=inputs.end_ind)

    def run_single(self, enc_latent_img0, model_latent_img1):
        """Runs inverse model on first input encoded by encoded and second input produced by model."""
        assert self._hp.train_im0_enc   # inv model needs to be trained from
        return remove_spatial(self.action_pred(enc_latent_img0, model_latent_img1))

    @contextmanager
    def val_mode(self, *args, **kwargs):
        yield


class TestTimeInverseModel(InverseModel):
    def __init__(self, params, logger):
        super().__init__(params, logger)
        if torch.cuda.is_available():
            self.cuda()
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        assert self._hp.checkpt_path is not None
        weights_file = CheckpointHandler.get_resume_ckpt_file('latest', self._hp.checkpt_path)
        success = CheckpointHandler.load_weights(weights_file, self)
        if not success: raise ValueError("Could not load checkpoint from {}!".format(weights_file))

    def forward(self, inputs):
        for k in inputs:
            if not isinstance(inputs[k], torch.Tensor):
                inputs[k] = torch.Tensor(inputs[k])
            if not inputs[k].device == self.device:
                inputs[k] = inputs[k].to(self.device) 

        enc_im0 = self.encoder.forward(inputs['img_t0'])[0]
        enc_im1 = self.encoder.forward(inputs['img_t1'])[0]
        return remove_spatial(self.action_pred(enc_im0, enc_im1))


class FromStatesInverseModel(InverseModel):
    def __init__(self, params, logger):
        super().__init__(params, logger)

    def get_timesteps(self, inputs):
        """
         # sample temporal distances between 1 and temp_dist, regress only first action
        :return:  None, call by reference
        """

        t0 = np.zeros(self._hp.batch_size)
        for b in range(self._hp.batch_size):
            t0[b] = np.random.randint(0, abs(inputs.end_ind[b].cpu().numpy() - self._hp.temp_dist), 1)
        delta_t = np.random.randint(1, self._hp.temp_dist + 1, self._hp.batch_size)
        t1 = t0 + delta_t
        t0 = torch.tensor(t0, device=inputs.demo_seq_states.device, dtype=torch.long)
        t1 = torch.tensor(t1, device=inputs.demo_seq_states.device, dtype=torch.long)
        inputs.state_t0 = batchwise_index(inputs.demo_seq_states, t0)
        inputs.state_t1 = batchwise_index(inputs.demo_seq_states, t1)
        inputs.selected_action = batchwise_index(inputs.actions, t0)

    def build_network(self, build_encoder=True):
        self.action_pred = Predictor(self._hp, self._hp.state_dim*2, self._hp.n_actions, 3)

    def forward(self, inputs):
        self.get_timesteps(inputs)
        actions_pred = self.action_pred(inputs.state_t0[:,:,None, None], inputs.state_t1[:,:,None, None])
        output = AttrDict()
        output.actions = torch.squeeze(actions_pred)
        return output

    def log_outputs(self, model_output, inputs, losses, step, log_images, phase):
        super()._log_losses(losses, step, phase)


class EarlyFusionInverseModel(InverseModel):
    def __init__(self, params, logger):
        super().__init__(params, logger)

    def build_network(self, build_encoder=True):
        self._hp.input_nc = 6
        self.encoder = Encoder(self._hp)

        if self._hp.pred_states:
            outdim = self._hp.n_actions + self._hp.state_dim
        else:
            outdim = self._hp.n_actions
        self.action_pred = Predictor(self._hp, self._hp.nz_enc, outdim, 3)

    def forward(self, inputs):
        self.get_timesteps(inputs)
        enc = self.encoder.forward(torch.cat([inputs.img_t0, inputs.img_t1], dim=1))[0]
        output = AttrDict()
        out = self.action_pred(enc)
        if self._hp.pred_states:
            output.actions, output.states = torch.split(torch.squeeze(out), [2,2], 1)
        else:
            output.actions = torch.squeeze(out)
        return output



