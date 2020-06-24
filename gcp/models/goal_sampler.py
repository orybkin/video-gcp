from contextlib import contextmanager

import numpy as np
import torch
from blox.tensor.ops import broadcast_final, batch_apply, batchwise_index, remove_spatial
from blox import AttrDict
from blox.basic_types import map_dict
from blox.torch.losses import L2Loss, NLL
from blox.torch.subnetworks import Predictor
from blox.torch.encoder_decoder import Encoder, DecoderModule
from blox.torch.variational import CVAE
from blox.torch.dist import Gaussian
from gcp.models.base_model import BaseModel
from gcp.rec_planner_utils.checkpointer import CheckpointHandler


class GoalSampler(BaseModel):
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

    def _default_hparams(self):
        # put new parameters in here:
        default_dict = {
            'ngf': 4,  # number of feature maps in shallowest level
            'nz_enc': 32,  # number of dimensions in encoder-latent space
            'nz_mid': 32,  # number of hidden units in fully connected layer
            'n_processing_layers': 3,  # Number of layers in MLPs
            'enc_params_checkpoint': None,   # specify pretrained encoder weights to load for training
            'build_encoder': True,      # if False, does not build an encoder, assumes that inputs are encoded from model
            'device': None,
        }

        # Dataset params
        default_dict.update({
            'randomize_length': False,
            'randomize_start': False,
        })
        
        # VAE params
        default_dict.update({
            'learn_sigma': False,   # This parameter is not used!
            'log_sigma': 0,   # This parameter is not used!
            'nz_vae': 32,
            'prior_type': 'learned',
            'var_inf': 'standard',
            'kl_weight': 1,
        })

        # decoder params
        default_dict.update({
            'use_skips': False,
            'dense_rec_type': None,
            'decoder_distribution': 'gaussian',
            'add_weighted_pixel_copy': False,
            'pixel_shift_decoder': False,
            'initial_sigma': 1,
            'learn_beta': True,
            'dense_img_rec_weight': 1,
        })

        # add new params to parent params
        parent_params = super()._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params
    
    def build_network(self, build_encoder=True):
        assert self._hp.build_encoder
        if self._hp.build_encoder:
            self.encoder = Encoder(self._hp)
            self.decoder = DecoderModule(self._hp, regress_actions=False)
            
        self.net = CVAE(self._hp, x_dim=self._hp.nz_enc, cond_dim=self._hp.nz_enc)

    def sample_target(self, seq, end_inds, repeats):
        """
        
        :param seq:
        :param end_inds:
        :param repeats: how many times to sample from each sequence
        :return:
        """
        # Note: it would be easier to implement it using randomized length
        
        def get_random_ints(min_i, max_i):
            index = torch.rand(max_i.shape + (repeats,), device=max_i.device)
            index = (min_i + index * (max_i[:, None].float() - min_i)).floor()
            return index
        
        index = get_random_ints(1, end_inds + 1)
        target = batchwise_index(seq, index.long())
        
        return target

    def forward(self, inputs, full_seq=None):
        outputs = AttrDict()
        
        n_repeats = 32
        if inputs.I_0.shape[0] == inputs.demo_seq.shape[0]:
            # Repeat the I_0 so that the visualization is correct
            # This should only be done once per batch!!
            inputs.I_0 = inputs.I_0.repeat_interleave(n_repeats, 0)
        
        I_target = self.sample_target(inputs.demo_seq, inputs.end_ind, n_repeats)
        inputs.I_target = I_target = I_target.reshape((-1,) + I_target.shape[2:])

        # Note, the decoder supports skips and pixel copying!
        e_0, _ = self.encoder(inputs.I_0)
        e_g, _ = self.encoder(I_target)
        
        outputs.update(self.net(e_g, e_0))
        outputs.e_target_prime = e_target_prime = outputs.pop('mu')
        outputs.I_target_prime = self.decoder(e_target_prime)
        
        return outputs

    def loss(self, inputs, outputs, add_total=True):
        losses = self.net.loss(inputs, outputs)
        losses.update(self.decoder.nll(outputs.I_target_prime.distr, inputs.I_target))

        return losses

    def log_outputs(self, outputs, inputs, losses, step, log_images, phase):
        super()._log_losses(losses, step, log_images, phase)

        if log_images:
            
            I_0 = inputs.I_0[:8]
            I_target_prime = outputs.I_target_prime.images[:8]
            I_target = inputs.I_target[:8]
            
            reconstructions = torch.cat([I_0, I_target_prime, I_target], 0)
            self._logger.log_image_grid(reconstructions, 'prediction', step, phase, nrow=8)
            
            with self.prior_mode():
                n_samples = 4
                samples = [self(inputs).I_target_prime.images[:8] for i in range(n_samples)]
                
            samples = torch.cat([I_0] + samples, 0)
            self._logger.log_image_grid(samples, 'samples', step, phase, nrow=8)

    def decode(self, z, cond):
        e, _ = self.encoder(cond)
        return self.decoder(self.net.gen(z, e)).images

    @contextmanager
    def val_mode(self, *args, **kwargs):
        yield

    @property
    def nz_vae(self):
        return self._hp.nz_vae

    #@property
    #def device(self):
    #    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
