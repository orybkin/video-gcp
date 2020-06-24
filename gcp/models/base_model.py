import os
from contextlib import contextmanager

import torch
import torch.nn as nn
from blox.torch.layers import LayerBuilderParams
from blox import AttrDict
from blox.torch.modules import Updater
from blox.torch.models import base as bloxm
from tensorflow.contrib.training import HParams


class BaseModel(bloxm.BaseModel):
    def __init__(self, logger):
        super().__init__()
        self._hp = None
        self._logger = logger

    @contextmanager
    def val_mode(self):
        """Sets validation parameters. To be used like: with model.val_mode(): ...<do something>..."""
        raise NotImplementedError("Need to implement val_mode context manager in subclass!")

    def step(self):
        self.call_children('step', Updater)

    def override_defaults(self, policyparams):
        for name, value in policyparams.items():
            print('overriding param {} to value {}'.format(name, value))
            if value == getattr(self._hp, name):
                raise ValueError("attribute is {} is identical to default value!!".format(name))
            self._hp.set_hparam(name, value)

    def _default_hparams(self):
        # Data Dimensions
        default_dict = AttrDict({
            'batch_size': -1,
            'max_seq_len': -1,
            'n_actions': -1,
            'state_dim': -1,
            'img_sz': 32,  # image resolution
            'input_nc': 3,  # number of input feature maps
            'n_conv_layers': None,  # Number of conv layers. Can be of format 'n-<int>' for any int for relative spec
        })
        
        # Network params
        default_dict.update({
            'use_convs': True,
            'use_batchnorm': True,  # TODO deprecate
            'normalization': 'batch',
            'predictor_normalization': 'group',
        })

        # Misc params
        default_dict.update({
            'filter_repeated_tail': False,      # whether to remove repeated states from the dataset
            'rep_tail': False,
            'dataset_class': None,
            'standardize': None,
            'split': None,
            'subsampler': None,
            'subsample_args': None,
            'checkpt_path': None,
        })
        
        # add new params to parent params
        parent_params = HParams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def postprocess_params(self):
        if not self._hp.use_convs:
            # self._hp.input_nc = self._hp.img_sz ** 2 * self._hp.input_nc
            self._hp.input_nc = self._hp.state_dim
        self._hp.add_hparam('builder', LayerBuilderParams(
            self._hp.use_convs, self._hp.use_batchnorm, self._hp.normalization, self._hp.predictor_normalization))
        self._hp.add_hparam('fc_builder', LayerBuilderParams(
            False, self._hp.use_batchnorm, self._hp.normalization, self._hp.predictor_normalization))

    def build_network(self):
        raise NotImplementedError("Need to implement this function in the subclass!")

    def forward(self, inputs):
        raise NotImplementedError("Need to implement this function in the subclass!")

    def loss(self, model_output, inputs):
        raise NotImplementedError("Need to implement this function in the subclass!")

    def log_outputs(self, model_output, inputs, losses, step, log_images, phase):
        # Log generally useful outputs
        self._log_losses(losses, step, log_images, phase)

        if phase == 'train':
            self.log_gradients(step, phase)
            
        for module in self.modules():
            if hasattr(module, '_log_outputs'):
                module._log_outputs(model_output, inputs, losses, step, log_images, phase, self._logger)

            if hasattr(module, 'log_outputs_stateful'):
                module.log_outputs_stateful(step, log_images, phase, self._logger)
            
    def _log_losses(self, losses, step, log_images, phase):
        for name, loss in losses.items():
            self._logger.log_scalar(loss.value, name + '_loss', step, phase)
            if 'breakdown' in loss and log_images:
                self._logger.log_graph(loss.breakdown, name + '_breakdown', step, phase)

    def _load_weights(self, weight_loading_info):
        """
        Loads weights of submodels from defined checkpoints + scopes.
        :param weight_loading_info: list of tuples: [(model_handle, scope, checkpoint_path)]
        """

        def get_filtered_weight_dict(checkpoint_path, scope):
            if os.path.isfile(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self._hp.device)
                filtered_state_dict = {}
                remove_key_length = len(scope) + 1      # need to remove scope from checkpoint key
                for key, item in checkpoint['state_dict'].items():
                    if key.startswith(scope):
                        filtered_state_dict[key[remove_key_length:]] = item
                if not filtered_state_dict:
                    raise ValueError("No variable with scope '{}' found in checkpoint '{}'!".format(scope, checkpoint_path))
                return filtered_state_dict
            else:
                raise ValueError("Cannot find checkpoint file '{}' for loading '{}'.".format(checkpoint_path, scope))

        print("")
        for loading_op in weight_loading_info:
            print(("=> loading '{}' from checkpoint '{}'".format(loading_op[1], loading_op[2])))
            filtered_weight_dict = get_filtered_weight_dict(checkpoint_path=loading_op[2],
                                                            scope=loading_op[1])
            loading_op[0].load_state_dict(filtered_weight_dict)
            print(("=> loaded '{}' from checkpoint '{}'".format(loading_op[1], loading_op[2])))
        print("")

    def log_gradients(self, step, phase):
        grad_norms = list([torch.norm(p.grad.data) for p in self.parameters() if p.grad is not None])
        if len(grad_norms) == 0:
            return
        grad_norms = torch.stack(grad_norms)

        self._logger.log_scalar(grad_norms.mean(), 'gradients/mean_norm', step, phase)
        self._logger.log_scalar(grad_norms.max(), 'gradients/max_norm', step, phase)
    
    def get_total_loss(self, inputs, losses):
        # compute total loss
        ## filtering is important when some losses are nan
        ## the unsqueeze is important when some of the weights or losses are 1-dim tensors.
        # TODO use the function from blox
        total_loss = torch.stack([loss[1].value[None] * loss[1].weight for loss in
                                  filter(lambda x: x[1].weight > 0, losses.items())]).sum()
        if torch.isnan(total_loss).any():
            import pdb; pdb.set_trace()
        return AttrDict(value=total_loss)
