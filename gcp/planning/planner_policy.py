import torch
import numpy as np

from blox import AttrDict
from gcp.planning.infra.policy.policy import Policy
from gcp.prediction.models.tree.tree import TreeModel
from gcp.prediction.training.checkpoint_handler import CheckpointHandler
from gcp.planning.cem.cem_planner import ImageCEMPlanner, CEMPlanner
from gcp.planning.cem.cem_simulator import GCPSimulator, \
    GCPImageSimulator, ActCondGCPImageSimulator


class PlannerPolicy(Policy):
    """Policy that uses predictive planning algorithm to devise plan, and then follows it."""
    def __init__(self, ag_params, policyparams, gpu_id=None, ngpu=None, conversion_fcns=None, n_rooms=None):
        """
        :param ag_params: Agent parameters for infrastructure
        :param policyparams: Parameters for the policy, including model parameters
        :param gpu_id: unused arg (to comply with infrastructure definition)
        :param ngpu: unused arg (to comply with infrastructure definition)
        :param conversion_fcns: unused arg (to comply with infrastructure definition)
        :param n_rooms: unused arg (to comply with infrastructure definition)
        """
        super(PlannerPolicy, self).__init__()

        self._hp = self._default_hparams()
        self.override_defaults(policyparams)
        self.verbose = self._hp.verbose
        self.log_dir = ag_params.log_dir
        self._hp.params['batch_size'] = 1
        self.max_seq_len = ag_params.T
        if 'max_seq_len' not in self._hp.params:
            self._hp.params['max_seq_len'] = ag_params.T

        # create planner predictive model
        model = policyparams['model_cls'] if 'model_cls' in policyparams else TreeModel
        self.planner = model(self._hp.params, None)
        assert self.planner._hp.img_sz == ag_params.image_width

        # move planner model to device
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')
        self.planner.to(self.device)
        self.planner.device = torch.device('cuda')
        self.planner._hp.device = self.planner.device

        # load weights for predictive model
        load_epoch = 'latest' if self._hp.load_epoch is None else self._hp.load_epoch
        weights_file = CheckpointHandler.get_resume_ckpt_file(load_epoch, self._hp.checkpt_path)
        CheckpointHandler.load_weights(weights_file, self.planner, strict=False)
        self.planner.eval()

        self.current_exec_step = None
        self.image_plan = None
        self.action_plan = None
        self.planner_outputs = []
        self.num_replans = 0

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
            'params': {},               # parameters for predictive model
            'model_cls': None,  # class for predictive model
            'checkpt_path': None,       # checkpoint path for predictive model
            'load_epoch': None,         # epoch that weigths should be loaded from
            'logger': None,
            'verbose': False,           # whether verbose planning outputs are logged
            'max_dump_rollouts': 5,     # max number of rollouts to dump
            'replan_interval': 1,       # interval at which replanning is triggered
            'num_max_replans': 10,      # maximum number of replannings per episode
        }

        parent_params = super()._default_hparams()
        parent_params.ncam = 1
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def act(self, t=None, i_tr=None, images=None, goal_image=None):
        """
        Triggers planning if no plan is made yet / last plan is completely executed. Then executes current plan.
        :param t: current time step in task execution
        :param i_tr: index of currently executed task
        :param images: images of so-far executed trajectory
        :param goal_image: goal-image that should be planned towards
        """
        self.t = t
        self.i_tr = i_tr
        self.goal_image = goal_image
        self.log_dir_verb = self.log_dir + '/verbose/traj{}'.format(self.i_tr)
        output = AttrDict()

        if self.image_plan is None \
              or self.image_plan.shape[0] - 1 <= self.current_exec_step \
              or (t % self._hp.replan_interval == 0 and self.num_replans < self._hp.num_max_replans):
            self._plan(images[t], goal_image, t)
            self.num_replans += 1

        output.actions = self.get_action(images[t])
        self.current_exec_step = self.current_exec_step + 1
        return output

    def get_action(self, current_image):
        assert self.action_plan is not None     # need to attach inverse model to planner to get actions!
        action = self.action_plan[self.current_exec_step]
        return action

    def _plan(self, image, goal_image, step):
        """Runs planning algorithm to obtain image and action plans."""
        raise NotImplementedError

    def log_outputs_stateful(self, logger=None, global_step=None, phase=None, dump_dir=None, exec_seq=None, goal=None,
                             index=None, env=None, goal_pos=None, traj=None, topdown_image=None):
        """Logs planner outputs for visualization."""
        raise NotImplementedError


class CEMPolicy(PlannerPolicy):
    """Implements a simple CEM planning policy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hp.cem_params.update({'max_seq_len': self._hp.params['max_seq_len']})
        self._cem_simulator = self.simulator_type(self.planner, append_latent=True)
        self._cem_planner = self._hp.cem_planner(self._hp.cem_params, self._cem_simulator)

    def _default_hparams(self):
        default_dict = {
            'cem_planner': None,
            'cem_params': {},
        }
        parent_params = super()._default_hparams()
        parent_params.ncam = 1
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def act(self, t=None, i_tr=None, state=None, goal=None, regression_state=None, images=None, run_super=False):
        """Plans a state trajectory with CEM, output actions are delta-states."""
        if run_super:
            return super().act(t, i_tr, state, goal)
        self._images = images[:, 0]
        self._states = state
        return super().act(t, i_tr, state, goal)

    def _plan(self, state, goal, step):
        """Runs CEM with planner model to generate state/action plan."""
        # run CEM to get state plan
        input_goal = goal[-1] if len(goal.shape) > 1 else goal
        self.image_plan, action_plan, _, self.plan_cost = self._cem_planner(state, input_goal)
        self.current_exec_step = 0  # reset internal execution counter used to index plan

        # compute action plan as difference between states
        self.action_plan = self.image_plan[1:] - self.image_plan[:-1]

    def log_outputs_stateful(self, logger=None, global_step=None, phase=None, dump_dir=None, exec_seq=None, goal=None,
                             index=None, env=None, goal_pos=None, traj=None, topdown_image=None):
        # self._cem_simulator.dump_logs()
        self._cem_planner.log_verbose(logger, global_step, phase, self.i_tr, dump_dir)

        # log executed image sequence
        executed_traj = self._images.astype(np.float32) / 255
        logger.log_video(executed_traj.transpose(0, 3, 1, 2), "elite_trajs_{}_test/execution".format(self.i_tr),
                         global_step, phase)

    def get_action(self, current_image):
        assert self.action_plan is not None  # need to attach inverse model to planner to get actions!
        if self.action_plan.size < 1:
            return 0.05 * np.random.rand(2, )
        action = self.action_plan[self.current_exec_step]
        return action

    @property
    def simulator_type(self):
        return GCPSimulator

    @property
    def planner_type(self):
        return CEMPlanner


class ImageCEMPolicy(CEMPolicy):
    """CEM planning policy for image-based tasks. Uses inverse model to follow plan"""

    def _default_hparams(self):
        default_dict = {
            'closed_loop_execution': False,  # if True, will execute state plan in closed loop
            'act_cond': False,  # if action-conditioned simulator should be used
        }
        parent_params = super()._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def act(self, t=None, i_tr=None, state=None, images=None, goal_image=None):
        self._images = images[:, 0]
        self._states = state
        return super().act(t, i_tr, images, goal_image, run_super=True)

    def _plan(self, state, goal, step):
        """Planner directly outputs action plan via inverse model."""
        self.image_plan, self.action_plan, self.latent_plan, self.plan_cost = self._cem_planner(state, goal)
        self.current_exec_step = 0
        self._cem_planner.hack_add_state(self._states[-1].copy())

    def get_action(self, current_image):
        """Executes plan, optional closed-loop by re-inferring actions with the inverse model."""
        if self._hp.closed_loop_execution:
            return self._infer_action(current_image, self.latent_plan[self.current_exec_step + 1])
        else:
            return super().get_action(current_image)

    def _infer_action(self, current_img, target_latent):
        """Uses inverse model to infer closed loop execution action."""
        img = torch.tensor(current_img, device=self.device, dtype=torch.float32)
        enc_img0 = self.planner.encoder(self._cem_simulator._env2planner(img))[0][:, :, 0, 0]
        return self.planner.inv_mdl.run_single(
            enc_img0, torch.tensor(target_latent[None], device=self.device))[0].data.cpu().numpy()

    @property
    def simulator_type(self):
        return GCPImageSimulator if not self._hp.act_cond else ActCondGCPImageSimulator

    @property
    def planner_type(self):
        return ImageCEMPlanner

