import torch
import tqdm
import numpy as np

from blox import AttrDict
from blox import batch_apply
from gcp.planning.hierarchical_planner_policy import HierarchicalPlannerPolicy
from gcp.planning.cem_policy.utils.cem_planner import ImageCEMPlanner, CEMPlanner
from gcp.planning.cem_policy.utils.cem_simulator import GCPSimulator,\
    GCPImageSimulator, ActCondGCPImageSimulator
from gcp.models.goal_sampler import GoalSampler
from gcp.rec_planner_utils.checkpointer import CheckpointHandler


class CEMPolicy(HierarchicalPlannerPolicy):
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
        self.current_exec_step = 0      # reset internal execution counter used to index plan
    
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
            return 0.05 * np.random.rand(2,)
        action = self.action_plan[self.current_exec_step]
        return action

    @property
    def simulator_type(self):
        return GCPSimulator

    @property
    def planner_type(self):
        return CEMPlanner


class ImageCEMPolicy(CEMPolicy):
    """CEM planning policy for image-based tasks."""

    def _default_hparams(self):
        default_dict = {
            'closed_loop_execution': False,     # if True, will execute state plan in closed loop
            'act_cond': False,                  # if action-conditioned simulator should be used
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
        # make it such that planner outputs action plan too! -> for now this change can be breaking
        self.image_plan, self.action_plan, self.latent_plan, self.plan_cost = self._cem_planner(state, goal)
        self.current_exec_step = 0
        self._cem_planner.hack_add_state(self._states[-1].copy())

    def get_action(self, current_image):
        if self._hp.closed_loop_execution:
            return self._infer_action(current_image, self.latent_plan[self.current_exec_step+1])
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


if __name__ == "__main__":
    from gcp.infra.envs.miniworld_env.utils.multiroom2d_layout import define_layout

    layout = define_layout(5, texture_dir="../assets/textures")

    import numpy as np
    from contextlib import contextmanager
    from gcp.train_planner_model import ModelTrainer
    from gcp.planning.cem_policy.utils.sampler import SimpleHierarchicalCEMSampler

    # test CEM with dummy values
    class DummyModel:
        def __init__(self):
            self.dense_rec = AttrDict(
                get_sample_with_len=self.fcn,
            )

        @contextmanager
        def val_mode(self):
            pass; yield; pass

        def __call__(self, *args, **kwargs):
            return np.zeros(64)

        def fcn(self, unused_arg, length, reference, *args, **kwargs):
            import torch
            return torch.rand((reference.shape[0], length, 2,)), None

    trainer = ModelTrainer()
    model = trainer.model_test       # comment out train command at end of __init__ for this to work

    sim = GCPSimulator(model, append_latent=False)
    params = AttrDict(horizon=80, action_dim=32, verbose=True, n_level_hierarchy=7, sampler=SimpleHierarchicalCEMSampler)
    planner = CEMPlanner(params, sim)

    start, end = np.random.rand(2), np.random.rand(2)
    output = planner(start, end)
    planner.log_verbose(trainer.logger, 10, 'val', 0)
