import numpy as np
import torch
from gcp.rec_planner_utils import vis_utils
import copy
import pdb
import imageio
import copy
from gcp.infra.policy.prm_policy.prm_policy import PrmPolicy
from gcp.infra.policy.policy import Policy
from gcp.models.base_hierarchical_planner import HierarchicalPlannerTest
from blox import AttrDict
from gcp.rec_planner_utils.checkpointer import CheckpointHandler
from gcp.models.inverse_mdl import TestTimeInverseModel
from gcp.infra.utils.im_utils import npy_to_gif
from gcp.rec_planner_utils.vis_utils import unstack
from PIL import Image


class PRMHierarchicalPlannerPolicy(Policy):
    """
    Hierachical Plannner Policy, uses actions from high-level planner
    """
    def __init__(self, ag_params, policyparams, gpu_id, ngpu):
        super(PRMHierarchicalPlannerPolicy, self).__init__()

        self._hp = self._default_hparams()
        self.override_defaults(policyparams)
        self.verbose = self._hp.verbose

        self.log_dir = ag_params.log_dir

        self.max_seq_len = ag_params.T

        self.prm_policy = PrmPolicy(ag_params, self._hp.params, gpu_id, ngpu)

        env_type, env_params = ag_params.env
        self.pred_env = env_type(env_params, self._reset_state)

        self.current_action = None
        self.action_plan = None
        self.planner_outputs = []
        self.image_plan = None

        self.img_t0_history = []
        self.img_t1_history = []

    def reset(self):
        super().reset()
        self.current_action = None
        self.action_plan = None
        self.planner_outputs = []
        self.img_t0_history = []
        self.img_t1_history = []

    def _default_hparams(self):
        default_dict = {
            'replan_interval': 1,
            'params': {},  # parameters of highlevel planner
            'verbose' :False,
            'max_dump_rollouts' :5,  # max number of rollouts to dump
            'max_planning_steps': 20,
        }

        parent_params = super()._default_hparams()
        parent_params.ncam = 1
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def act(self, t=None, i_tr=None, goal=None, qpos_full=None, images=None):
        # Note: goal_image provides n (2) images starting from the last images of the trajectory

        self.t = t
        self.i_tr = i_tr
        self.log_dir_verb = self.log_dir + '/verbose/traj{}'.format(self.i_tr)
        output = AttrDict()

        if self.image_plan is None or (t % self._hp.replan_interval == 0) or self.current_action >= self.image_plan.shape[0]:
            self.image_plan = self._plan(qpos_full, goal)

        output.actions = self.get_action(images[t])
        self.current_action = self.current_action + 1
        return output

    def get_action(self, current_image):
        """
        :param current_image:  the current observation
        :return:  actions
        """
        raise NotImplementedError

    def _plan(self, current_state, goal):
        print("planning at t{}".format(self.t))

        state = copy.deepcopy(current_state)
        self.pred_env.reset(state)

        image_plan = []
        tpred = 0
        while tpred < self._hp.max_planing_steps:
            action = self.prm_policy.act(tpred, self.t, qpos_full=None, goal=None)
            obs = self.pred_env.step(action)
            image_plan.append(obs['images'])
            state = obs['qpos_full']

            tpred += 1

        self.planner_outputs.append((self.t, image_plan))

    def log_outputs_stateful(self, logger=None, global_step=None, phase=None, dump_dir=None, exec_seq=None, goal=None, index=None, env=None, goal_pos=None, traj=None, topdown_image=None):
        """
        :param logger:  neeeded for tensorboard logs
        :param global_step: the global_step used when training
        :param phase:
        :param dump_dir:
        :param exec_seq:
        :param goal:
        :param index:
        :return:
        """
        imageio.imwrite(self.log_dir_verb + '/goalim.png', self.goal_image[0])
        imt0t1_history = [np.concatenate([im0, im1], axis=0) for im0, im1 in zip(self.img_t0_history, self.img_t1_history)]

        # store plan overview in TB
        if index < self._hp.max_dump_rollouts:   # log for max 5 rollouts
            vis_utils.PARAMS.hp = self.planner._hp
            logger.log_planning_overview(self.planner_outputs, exec_seq, goal, 'control/planning_overview_{}'.format(index), global_step, phase)
            imt0t1_history = np.transpose(np.stack(imt0t1_history, 0), [0, 3, 1, 2])
            logger.log_video(imt0t1_history, 'control/planning_overview_{}'.format(index), global_step, phase)
            logger.log_video(topdown_image.transpose(0, 3, 1, 2), 'control/planning_overview_{}_topdown'.format(index), global_step, phase)
            logger.flush()

    def planner2npy_img(self, img):
        img = img.detach().cpu().numpy().squeeze()
        img = ((img + 1 ) / 2 *255).astype(np.uint8)
        if len(img.shape) == 4:
            img = np.transpose(img, [0, 2, 3, 1])
            return unstack(img)
        elif len(img.shape) == 3:
            return np.transpose(img, [1, 2, 0])