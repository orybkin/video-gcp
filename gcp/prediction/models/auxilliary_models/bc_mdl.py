from contextlib import contextmanager

import torch

from blox import AttrDict
from blox.tensor.ops import broadcast_final, batch_apply, remove_spatial
from blox.torch.encoder_decoder import Encoder
from blox.torch.losses import L2Loss
from blox.torch.subnetworks import Predictor
from gcp.prediction.models.auxilliary_models.misc import RecurrentPolicyModule
from gcp.prediction.models.auxilliary_models.base_model import BaseModel
from gcp.prediction.training.checkpoint_handler import CheckpointHandler


class BehavioralCloningModel(BaseModel):
    def __init__(self, params, logger):
        super().__init__(logger)
        self._hp = self._default_hparams()
        self.override_defaults(params)  # override defaults with config file
        self.postprocess_params()
        assert self._hp.n_actions != -1     # make sure action dimensionality was overridden
        self.build_network()

        # load only the encoder params during training
        if self._hp.enc_params_checkpoint is not None:
            self._load_weights([
                (self.encoder, 'encoder', self._hp.enc_params_checkpoint),
            ])
        self.detach_enc = self._hp.enc_params_checkpoint is not None and not self._hp.finetune_enc

    def _default_hparams(self):
        # put new parameters in here:
        default_dict = {
            'ngf': 4,  # number of feature maps in shallowest level
            'nz_enc': 32,  # number of dimensions in encoder-latent space
            'nz_mid': 32,       # number of hidden units in fully connected layer
            'nz_mid_lstm': 32,
            'n_lstm_layers': 1,
            'n_processing_layers': 3,  # Number of layers in MLPs
            'reactive': True,   # if False, adds recurrent cell to policy
            'enc_params_checkpoint': None,   # specify pretrained encoder weights to load for training
            'finetune_enc': False,
            'checkpt_path': None,
            'train_first_action_only': False,     # if True, only trains on initial action of sequence
            'n_conv_layers': None,  # Number of conv layers. Can be of format 'n-<int>' for any int for relative spec'n_conv_layers': None,  # Number of conv layers. Can be of format 'n-<int>' for any int for relative spec
        }

        # misc params
        default_dict.update({
            'use_skips': False,
            'dense_rec_type': None,
            'device': None,
            'randomize_length': False,
            'randomize_start': False,
        })

        # add new params to parent params
        parent_params = super()._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params
    
    def build_network(self, build_encoder=True):
        self.encoder = Encoder(self._hp)
        if not self._hp.reactive:
            self.policy = RecurrentPolicyModule(self._hp, 2*self._hp.nz_enc, self._hp.n_actions)
        else:
            self.policy = Predictor(self._hp, 2*self._hp.nz_enc, self._hp.n_actions)

    def forward(self, inputs, phase='train'):
        """
        forward pass at training time
        """
        if not 'enc_traj_seq' in inputs:
            enc_traj_seq, _ = self.encoder(inputs.traj_seq[:, 0]) if self._hp.train_first_action_only \
                                    else batch_apply(self.encoder, inputs.traj_seq)
            if self._hp.train_first_action_only: enc_traj_seq = enc_traj_seq[:, None]
            enc_traj_seq = enc_traj_seq.detach() if self.detach_enc else enc_traj_seq

        enc_goal, _ = self.encoder(inputs.I_g)
        n_dim = len(enc_goal.shape)
        fused_enc = torch.cat((enc_traj_seq, enc_goal[:, None].repeat(1, enc_traj_seq.shape[1], *([1]*(n_dim-1)))), dim=2)
        #fused_enc = torch.cat((enc_traj_seq, enc_goal[:, None].repeat(1, enc_traj_seq.shape[1], 1, 1, 1)), dim=2)

        if self._hp.reactive:
            actions_pred = batch_apply(self.policy, fused_enc)
        else:
            policy_output = self.policy(fused_enc) 
            actions_pred = policy_output

        # remove last time step to match ground truth if training on full sequence
        actions_pred = actions_pred[:, :-1]	if not self._hp.train_first_action_only else actions_pred

        output = AttrDict()
        output.actions = remove_spatial(actions_pred) if len(actions_pred.shape) > 3 else actions_pred
        return output

    def loss(self, inputs, outputs):
        losses = AttrDict()

        # action prediction loss
        n_actions = outputs.actions.shape[1]
        losses.action_reconst = L2Loss(1.0)(outputs.actions, inputs.actions[:, :n_actions],
                                            weights=broadcast_final(inputs.pad_mask[:, :n_actions], inputs.actions))

        # compute total loss
        #total_loss = torch.stack([loss[1].value * loss[1].weight for loss in losses.items()]).sum()
        #losses.total = AttrDict(value=total_loss)
        # losses.total = total_loss*torch.tensor(np.nan)   # for checking if backprop works
        return losses

    def get_total_loss(self, inputs, losses):
        total_loss = torch.stack([loss[1].value * loss[1].weight for loss in losses.items()]).sum()
        return AttrDict(value=total_loss)

    def log_outputs(self, outputs, inputs, losses, step, log_images, phase):
        super().log_outputs(outputs, inputs, losses, step, log_images, phase)
        if log_images and self._hp.use_convs:
            self._logger.log_pred_actions(outputs, inputs, 'pred_actions', step, phase)

    @contextmanager
    def val_mode(self, *args, **kwargs):
        yield

    @property
    def has_image_input(self):
        return self._hp.use_convs


class TestTimeBCModel(BehavioralCloningModel):
    def __init__(self, params, logger):
        super().__init__(params, logger) 
        if torch.cuda.is_available(): 
            self.cuda()
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        if not self._hp.reactive: self.init_hidden_var = self.init_hidden_var.to(self.device)

        assert self._hp.checkpt_path is not None
        weights_file = CheckpointHandler.get_resume_ckpt_file('latest', self._hp.checkpt_path)
        success = CheckpointHandler.load_weights(weights_file, self)
        if not success: raise ValueError("Could not load checkpoint from {}!".format(weights_file))

    def forward(self, inputs):
        for k in inputs:
            if inputs[k] is None:
                continue
            if not isinstance(inputs[k], torch.Tensor):
                inputs[k] = torch.Tensor(inputs[k])
            if not inputs[k].device == self.device:
                inputs[k] = inputs[k].to(self.device)

        enc, _ = self.encoder(inputs['I_0'])
        enc_goal, _ = self.encoder(inputs['I_g'])
        fused_enc = torch.cat((enc, enc_goal), dim=1)
        if self._hp.reactive:
            action_pred = self.policy(fused_enc)
            hidden_var = None
        else:
            hidden_var = self.init_hidden_var if inputs.hidden_var is None else inputs.hidden_var
            policy_output = self.policy(fused_enc[:, None], hidden_var)
            action_pred, hidden_var = policy_output.output, policy_output.hidden_state[:, 0]
        if self._hp.use_convs:
            return remove_spatial(action_pred if len(action_pred.shape)==4 else action_pred[:, 0]), hidden_var
        else:
            return action_pred, hidden_var

