import numpy as np
import os
import torch
import imageio
import copy
from blox import AttrDict
from gcp.infra.policy.policy import Policy
from gcp.models.base_hierarchical_planner import HierarchicalPlannerTest
from gcp.models.inverse_mdl import TestTimeInverseModel
from gcp.infra.utils.im_utils import npy_to_gif
from gcp.rec_planner_utils.checkpointer import CheckpointHandler
from gcp.rec_planner_utils.vis_utils import unstack
from gcp.rec_planner_utils.vis_utils import plot_val_tree
from PIL import Image


class HierarchicalPlannerPolicy(Policy):
    """
    Hierachical Plannner Policy, uses actions from high-level planner
    """
    def __init__(self, ag_params, policyparams, gpu_id=None, ngpu=None, conversion_fcns=None, n_rooms=None):
        super(HierarchicalPlannerPolicy, self).__init__()

        self._hp = self._default_hparams()
        self.override_defaults(policyparams)
        self.verbose = self._hp.verbose

        self.log_dir = ag_params.log_dir
        self._hp.params['batch_size'] = 1
        if 'max_seq_len' not in self._hp.params:
            self._hp.params['max_seq_len'] = ag_params.T
        print(self._hp.params['max_seq_len'])
        model = policyparams['model_cls'] if 'model_cls' in policyparams else HierarchicalPlannerTest
        self.planner = model(self._hp.params, None)
        assert self.planner._hp.img_sz == ag_params.image_width

        self.max_seq_len = ag_params.T

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')
        self.planner.to(self.device)

        self.planner.device = torch.device('cuda')
        self.planner._hp.device = self.planner.device

        try:
            # If policyparams.model exists, load the weights
            self.planner.load_state_dict(AttrDict(policyparams).model.state_dict())
        except AttributeError:
            load_epoch = 'latest' if self._hp.load_epoch is None else self._hp.load_epoch
            weights_file = CheckpointHandler.get_resume_ckpt_file(load_epoch, self._hp.checkpt_path)
            success = CheckpointHandler.load_weights(weights_file, self.planner, strict=False)
        self.planner.eval()

        self.current_exec_step = None
        self.image_plan = None
        self.action_plan = None
        self.planner_outputs = []
        self.image_plan = None
        self.num_replans = 0

        self.img_t0_history = []
        self.img_t1_history = []

    def reset(self):
        super().reset()
        self.current_exec_step = None
        self.action_plan = None
        self.image_plan = None
        self.num_replans = 0
        self.planner_outputs = []
        self.img_t0_history = []
        self.img_t1_history = []

    def _default_hparams(self):
        default_dict = {
            'replan_interval': 1,
            'replan_if_deviated': False,
            'deviation_threshold': None,
            'params': {},
            'checkpt_path': None,
            'verbose': False,
            'model': None,
            'max_dump_rollouts': 5,   #max number of rollouts to dump
            'load_epoch': None,
            'logger': None,
            'num_max_replans': 10,
            'model_cls': None,
        }

        parent_params = super()._default_hparams()
        parent_params.ncam = 1
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def act(self, t=None, i_tr=None, images=None, goal_image=None):
        # Note: goal_image provides n (2) images starting from the last images of the trajectory

        self.t = t
        self.i_tr = i_tr
        self.goal_image = goal_image
        self.log_dir_verb = self.log_dir + '/verbose/traj{}'.format(self.i_tr)
        output = AttrDict()

        if self.image_plan is None \
              or self.image_plan.shape[0] - 1 <= self.current_exec_step \
              or (t % self._hp.replan_interval == 0 and self.num_replans < self._hp.num_max_replans)\
              or (self._hp.replan_if_deviated and self._deviated(images[t], self.image_plan[self.current_exec_step]) and \
                  self.num_replans < self._hp.num_max_replans):
            self._plan(images[t], goal_image, t)
            self.num_replans += 1

        output.actions = self.get_action(images[t])
        self.current_exec_step = self.current_exec_step + 1
        return output

    def get_action(self, current_image):
        assert self.action_plan is not None     # need to attach inverse model to planner to get actions!
        action = self.action_plan[self.current_exec_step]

        # log current and goal image
        img_t0 = self._preprocess_input(current_image)
        img_t1 = self._preprocess_input(self.image_plan[self.current_exec_step][None])
        self.img_t0_history.append(copy.deepcopy(self.planner2npy_img(img_t0)))
        self.img_t1_history.append(copy.deepcopy(self.planner2npy_img(img_t1)))

        return action

    def _plan(self, image, goal_image, step):
        print("planning at t{}".format(self.t))
        input_dict = AttrDict(I_0=self._env2planner(image), I_g=self._env2planner(goal_image),
                              start_ind=torch.Tensor([0]).long(),
                              end_ind=torch.Tensor([self._hp.params['max_seq_len'] - 1]).long())
        with self.planner.val_mode():
            planner_output = self.planner(input_dict)
            # perform pruning for the balanced tree
            image_plan, _ = self.planner.dense_rec.get_sample_with_len(
                0, self._hp.params['max_seq_len'], planner_output, input_dict, 'basic')

        # first image is copy of the initial frame -> omit
        self.image_plan = image_plan[1:]
        self.action_plan = planner_output.actions.detach().cpu().numpy()[0] if 'actions' in planner_output else None

        planner_output.dense_rec = AttrDict(images=image_plan[None])
        self.planner_outputs.append((step, planner_output))
        self.current_exec_step = 0

        if self.verbose:
            npy_to_gif(self.planner2npy_img(planner_output.dense_rec.images[0]),
                       self.log_dir_verb + '/plan_t{}'.format(self.t, step))

    def log_outputs_stateful(self, logger=None, global_step=None, phase=None, dump_dir=None, exec_seq=None, goal=None,
                             index=None, env=None, goal_pos=None, traj=None, topdown_image=None):
        """
        :param logger:  neeeded for tensorboard logs
        """
        imageio.imwrite(self.log_dir_verb + '/goalim.png', self.goal_image[0, 0] if len(self.goal_image.shape)>4 else self.goal_image[0])

        assert self._hp.params['batch_size'] == 1   # the logging function does not currently support batches > 1!
        # dump predicted trees to directory
        tree_save_dir = os.path.join(dump_dir, 'verbose', 'pred_trees', 'it_{}'.format(global_step), 'run_{}'.format(index))
        if not os.path.exists(tree_save_dir):
            os.makedirs(tree_save_dir)
        for plan_step, planner_output in self.planner_outputs:
            if 'subgoal' in planner_output.tree and planner_output.tree.subgoal is not None:
                im = np.asarray(plot_val_tree(planner_output, None, n_logged_samples=1)[0] * 255, dtype=np.uint8)
                imageio.imwrite(os.path.join(tree_save_dir, 'planned_tree_step_{}.png'.format(plan_step)), im)

        imt0t1_history = [np.concatenate([im0[0], im1], axis=0) for im0, im1 in zip(self.img_t0_history, self.img_t1_history)]
        npy_to_gif(imt0t1_history, self.log_dir + '/verbose/traj{}/imt0t1_history'.format(self.i_tr))

        # store plan overview in TB
        if index < self._hp.max_dump_rollouts:   # log for max 5 rollouts
            logger.log_planning_overview(self.planner_outputs, exec_seq, goal, 'control/planning_overview_{}'.format(index), global_step, phase)
            imt0t1_history = np.transpose(np.stack(imt0t1_history, 0), [0, 3, 1, 2])
            logger.log_video(imt0t1_history, 'control/planning_overview_{}'.format(index), global_step, phase)

    def planner2npy_img(self, img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy().squeeze()
        img = ((img + 1)/2*255).astype(np.uint8)
        if len(img.shape) == 4:
            img = np.transpose(img, [0, 2, 3, 1])
            return unstack(img)
        elif len(img.shape) == 3:
            return np.transpose(img, [1, 2, 0])

    def _env2planner(self, img):
        """Converts images to the [-1...1] range of the hierarchical planner."""
        if img.dtype == np.uint8:
            img = img / 255.0
        if len(img.shape) == 5:
            img = img[0]
        img = np.transpose(img, [0, 3, 1, 2])
        return torch.from_numpy(img * 2 - 1.0).float().to(self.device)

    def _planner2env(self, action):
        """Converts model output tensor to numpy array."""
        if action is not None and action.shape[0] > 0:
            return action.data.cpu().numpy()
        else:
            return self.default_action[:, None]  # Failed to plan

    def _deviated(self, state, target_state):
        return np.linalg.norm(state[:target_state.shape[0]] - target_state) > self._hp.deviation_threshold

    @property
    def default_action(self):
        return np.zeros(self.planner._hp.n_actions)

    @staticmethod
    def _preprocess_input(input):
        assert len(input.shape) == 4  # can currently only handle inputs with 4 dims
        if input.max() > 1.0:
            input = input / 255.
        if input.min() >= 0.0:
            input = 2 * input - 1.0
        if input.shape[-1] == 3:
            input = input.transpose(0, 3, 1, 2)
        return input
