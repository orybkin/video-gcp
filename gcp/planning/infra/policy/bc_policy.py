import numpy as np
import torch

from blox import AttrDict
from gcp.planning.infra.policy.policy import Policy
from gcp.prediction.models.auxilliary_models.bc_mdl import TestTimeBCModel


class BehavioralCloningPolicy(Policy):
    """
    Behavioral Cloning Policy
    """
    def __init__(self, ag_params, policyparams, gpu_id, ngpu, conversion_fcns=None, n_rooms=None):
        super(BehavioralCloningPolicy, self).__init__()

        self._hp = self._default_hparams()
        self.override_defaults(policyparams)

        self.log_dir = ag_params.log_dir
        self._hp.params['batch_size'] = 1
        # self._hp.params['n_actions'] = self._hp.params.n_actions  # todo get this from env!
        self.policy = TestTimeBCModel(self._hp.params, None)
        self.policy.eval()
        self.hidden_var = None      # stays None for non-recurrent policy

    def reset(self):
        super().reset()
        self.hidden_var = None

    def _default_hparams(self):
        default_dict = {
            'params': {},
            'checkpt_path': None,
            'model': None,
            'logger': None,
        }

        parent_params = super()._default_hparams()
        parent_params.ncam = 1
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def act(self, t=None, i_tr=None, images=None, state=None, goal=None, goal_image=None):
        # Note: goal_image provides n (2) images starting from the last images of the trajectory
        self.t = t
        self.i_tr = i_tr
        self.goal_image = goal_image

        if self.policy.has_image_input:
            inputs = AttrDict(
                I_0=self._preprocess_input(images[t]),
                I_g=self._preprocess_input(goal_image[-1] if len(goal_image.shape) > 4 else goal_image),
                hidden_var=self.hidden_var
            )
        else:
            current = state[-1:, :2]
            goal = goal[-1:, :2] #goal_state = np.concatenate([state[-1:, -2:], state[-1:, 2:]], axis=-1)
            inputs = AttrDict(
                I_0=current,
                I_g=goal,
                hidden_var=self.hidden_var
            )

        actions, self.hidden_var = self.policy(inputs)

        output = AttrDict()
        output.actions = actions.data.cpu().numpy()[0]
        return output

    @staticmethod
    def _preprocess_input(input):
        assert len(input.shape) == 4    # can currently only handle inputs with 4 dims
        if input.max() > 1.0: 
            input = input / 255.
        if input.min() >= 0.0:
            input = 2*input - 1.0
        if input.shape[-1] == 3:
            input = input.transpose(0, 3, 1, 2)
        return input

    @property
    def default_action(self):
        return np.zeros(self.policy._hp.n_actions)

    def log_outputs_stateful(self, logger=None, global_step=None, phase=None, dump_dir=None, exec_seq=None, goal=None, index=None, env=None, goal_pos=None, traj=None, topdown_image=None):
        logger.log_video(np.transpose(exec_seq, [0, 3, 1, 2]), 'control/traj{}_'.format(index), global_step, phase)
        goal_img = np.transpose(goal, [2, 0, 1])[None]
        goal_img = torch.tensor(goal_img)
        logger.log_images(goal_img, 'control/traj{}_goal'.format(index), global_step, phase)

class BehavioralCloningPolicy_RegressionState(BehavioralCloningPolicy):
    def act(self, t=None, i_tr=None, images=None, regression_state=None, goal=None, goal_image=None):
        return super().act(t, i_tr, images, regression_state, goal, goal_image)

import cv2

def resize_image(im):
    return cv2.resize(im.squeeze(), (64, 64), interpolation=cv2.INTER_AREA)


