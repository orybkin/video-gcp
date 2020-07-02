import numpy as np
import copy
import os
import torch
import pickle as pkl
from collections import defaultdict
from blox import AttrDict
from blox.utils import ParamDict
from blox.basic_types import listdict2dictlist

from gcp.planning.cem.cost_fcn import EuclideanPathLength, LearnedCostEstimate
from gcp.planning.cem.sampler import FlatCEMSampler, HierarchicalTreeCEMSampler


class CEMPlanner:
    """Generic CEM planner."""
    def __init__(self, hp, simulator):
        self._hp = self._default_hparams().overwrite(hp)
        self._simulator = simulator
        self._cost_fcn = self._build_cost()
        self._sampler = self._build_sampler()
        self._logs = []

    def _default_hparams(self):
        default_dict = ParamDict(
            horizon=None,        # CEM optimization horizon (i.e. how many sequential actions get optimized)
            action_dim=None,     # dimensionality of the actions that are optimized
            n_iters=1,           # number of CEM iterations
            batch_size=64,       # number of rollouts per iteration
            max_rollout_bs=100,  # maximum batch size for rollout (splits 'batch_size' if too large)
            elite_frac=0.1,      # percentage of 'best' trajectories
        )
        # cost params
        default_dict.update(ParamDict(
            cost_fcn=EuclideanPathLength,
            dense_cost=False,
            final_step_cost_weight=1.0,
        ))
        # sampler params
        default_dict.update(ParamDict(
            sampler=FlatCEMSampler,
            sampler_clip_val=float("Inf"),
            initial_std=3e-1,
        ))
        # misc
        default_dict.update(ParamDict(
            verbose=False,                  # whether to visualize planning procedure (for debugging)
            dump_planning_data=False,       # whether to dump raw planning data
            use_delta_state_actions=False,  # if True, uses delta between inferred states as action plan
            use_inferred_actions=True,      # if True, uses model-inferred actions for action plan
            max_seq_len=None,               # used for model during rollout
        ))
        return default_dict

    def __call__(self, state, goal_state):
        logs = []
        self._sampler.init()
        for cem_iter in range(self._hp.n_iters):
            # sample actions
            samples = self._sampler.sample(self._hp.batch_size)

            # rollout simulator
            rollouts = self._rollout(state, goal_state, samples)

            best_rollouts, best_rollouts_states, best_scores, best_samples, elite_idxs = \
                self._get_best_rollouts(rollouts, goal_state, samples)

            # refit action distribution
            self._sampler.fit(best_samples, best_scores)

            # store all logs
            logs.append(AttrDict(
                elite_rollouts=copy.deepcopy(best_rollouts),
                elite_scores=best_scores,
                dists=self._sampler.get_dists(),
                goal_state=goal_state,
                elite_states=copy.deepcopy(best_rollouts_states),
            ))

        # perform final rollout with best actions
        final_rollouts = self._rollout(state, goal_state, best_samples)
        logs.append(AttrDict(
            elite_rollouts=copy.deepcopy(self._maybe_split_image(final_rollouts.predictions)),
            elite_scores=best_scores,
            dists=self._sampler.get_dists(),
            goal_state=goal_state,
            elite_states=copy.deepcopy(final_rollouts.states),
        ))

        # extract output action plan
        best_actions = self._get_action_plan(final_rollouts, best_samples)

        # save logs
        self._logs.append(logs)

        return final_rollouts.predictions[0], best_actions[0], final_rollouts.latents[0], best_scores[0]

    def log_verbose(self, logger, step, phase, i_tr, dump_dir):
        if self._hp.dump_planning_data:
            os.makedirs(os.path.join(dump_dir, "planning"), exist_ok=True)
            with open(os.path.join(dump_dir, "planning/traj{}_raw_data.pkl".format(i_tr)), "wb") as F:
                pkl.dump(self._logs, F)

        self._logs = []

    def _build_cost(self):
        return self._hp.cost_fcn(self._hp.dense_cost, self._hp.final_step_cost_weight)

    def _build_sampler(self):
        return self._hp.sampler(self._hp.sampler_clip_val,
                                self._hp.max_seq_len,
                                self._hp.action_dim,
                                self._hp.initial_std)

    def _rollout(self, state, goal, samples):
        output = defaultdict(list)
        for i in range(max(samples.shape[0] // self._hp.max_rollout_bs, 1)):
            sim_output = self._simulator.rollout(state, goal,
                                                 samples[i * self._hp.max_rollout_bs: (i + 1) * self._hp.max_rollout_bs],
                                                 self._hp.max_seq_len)
            output = self._join_dicts(sim_output, output)
        return AttrDict({key: self._cap_to_horizon(output[key]) for key in output})

    def _get_best_rollouts(self, rollouts, goal_state, samples):
        # compute rollout scores
        scores = self._cost_fcn(rollouts.predictions, goal_state)

        # get idxs of best rollouts
        full_elite_idxs = scores.argsort()
        elite_idxs = full_elite_idxs[:int(self._hp.batch_size * self._hp.elite_frac)]

        best_rollouts, best_rollouts_states = \
            [rollouts.predictions[idx] for idx in elite_idxs], [rollouts.states[idx] for idx in elite_idxs],
        best_scores, best_samples = scores[elite_idxs], samples[elite_idxs]
        return self._maybe_split_image(best_rollouts), best_rollouts_states, best_scores, best_samples, elite_idxs

    def _maybe_split_image(self, rollout):
        if hasattr(self._cost_fcn, "_split_state_rollout"):
            # separate latent and image in case that latent got attached to rollout
            return self._cost_fcn._split_state_rollout(rollout).image_rollout
        return rollout

    def _get_action_plan(self, final_rollouts, best_samples):
        if self._hp.use_delta_state_actions:
            return [b[1:] - b[:-1] for b in final_rollouts.states]
        elif self._hp.use_inferred_actions:
            return final_rollouts.actions
        else:
            return best_samples

    def _cap_to_horizon(self, input):
        if self._hp.horizon is not None:
            return [elem[:self._hp.horizon] for elem in input]
        else:
            return input

    @property
    def append_latent(self):
        return self._sampler.append_latent

    @staticmethod
    def _join_dicts(d1, d2):
        return AttrDict({key: d1[key] + d2[key] for key in d1})


class HierarchicalCEMPlanner(CEMPlanner):
    """CEM planner for hierarchical optimization."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._hp.sampling_rates_per_layer is not None:
            assert self._hp.n_iters == len(self._hp.sampling_rates_per_layer) + 1

    def _default_hparams(self):
        default_dict = super()._default_hparams()
        # general params
        default_dict.update(ParamDict(
            horizon=None,       # for GCP we do not need to define horizon
        ))
        # cost params
        default_dict.update(ParamDict(
            cost_fcn=LearnedCostEstimate,
            cost_config={},     # cost function for subgoal optimization
            LL_cost_fcn=None,   # if None cost_fcn is used
        ))
        # sampler params
        default_dict.update(ParamDict(
            sampler=HierarchicalTreeCEMSampler,
            n_level_hierarchy=None,
            sampling_rates_per_layer=None,
            n_ll_samples=5,
        ))
        return default_dict

    def _build_cost(self):
        cost_fcn = self._hp.cost_fcn(self._hp.cost_config)
        self._ll_cost_fcn = cost_fcn if self._hp.LL_cost_fcn is None \
            else self._hp.LL_cost_fcn(self._hp.dense_cost, self._hp.final_step_cost_weight)
        return cost_fcn

    def _build_sampler(self):
        return self._hp.sampler(self._hp.sampler_clip_val,
                                self._hp.max_seq_len,
                                self._hp.action_dim,
                                self._hp.initial_std,
                                n_level_hierarchy=self._hp.n_level_hierarchy,
                                sampling_rates_per_layer=self._hp.sampling_rates_per_layer,
                                subgoal_cost_fcn=self._cost_fcn,
                                ll_cost_fcn=self._ll_cost_fcn,
                                n_ll_samples=self._hp.n_ll_samples)

    def _get_best_rollouts(self, rollouts, goal_state, samples):
        if not isinstance(self._sampler, HierarchicalTreeCEMSampler):
            # in case we use non-hierarchical optimization with tree-based model
            return super()._get_best_rollouts(rollouts, goal_state, samples)
        best_rollouts, best_scores = self._sampler.optimize(rollouts.predictions, goal_state)
        best_samples = self._sampler.sample(self._hp.batch_size)
        elite_idxs = np.arange(len(best_rollouts))      # dummy value
        return best_rollouts, rollouts.states, best_scores, best_samples, elite_idxs


class ImageCEMPlanner(CEMPlanner):
    def log_verbose(self, logger, step, phase, i_tr, dump_dir):
        if self._hp.verbose:
            for replan_idx, replan_log in enumerate(self._logs):
                for cem_iter_idx, iter_log in enumerate(replan_log):

                    # visualize all plans in order
                    plan_stack = []
                    for plan in iter_log.elite_rollouts:
                        time, c, h, w = plan.shape
                        plan = np.clip((plan+1) / 2, 0, 1.0)
                        if time < self._hp.horizon:
                            plan = np.concatenate((plan, np.ones((self._hp.horizon - time, c, h, w))))
                        plan_stack.append(plan)
                    plan_stack = np.array(plan_stack)
                    n_plans = plan_stack.shape[0]
                    log_img = torch.tensor(plan_stack.transpose(0, 2, 3, 1, 4)
                                           .reshape(n_plans, c, h, self._hp.horizon*w)
                                           .transpose(1, 0, 2, 3).reshape(c, h*n_plans, self._hp.horizon*w))
                    logger.log_images(log_img[None],
                            "elite_trajs_{}_test/plan_r{}_iter{}_overview".format(i_tr, replan_idx, cem_iter_idx),
                            step, phase)

                    if 'elite_states' in iter_log:
                        logger.log_single_topdown_traj(iter_log.elite_states[0],
                                                       "elite_trajs_{}_test/plan_r{}_iter{}_z_inferStateTraj".
                                                       format(i_tr, replan_idx, cem_iter_idx), step, phase)
                        logger.log_multiple_topdown_trajs(iter_log.elite_states,
                                                       "elite_trajs_{}_test/plan_r{}_iter{}_z_inferStateTrajDist".
                                                       format(i_tr, replan_idx, cem_iter_idx), step, phase)
                    if 'goal_state' in iter_log:
                        logger.log_images(torch.tensor(iter_log['goal_state'].transpose(0, 3, 1, 2)),
                                          "elite_trajs_{}_test/plan_r{}_iter{}_z_goal".
                                          format(i_tr, replan_idx, cem_iter_idx), step, phase)
        super().log_verbose(logger, step, phase, i_tr, dump_dir)

    def hack_add_state(self, state):
        self._logs[-1][-1].state = state.copy()


class HierarchicalImageCEMPlanner(HierarchicalCEMPlanner, ImageCEMPlanner):
    def log_verbose(self, *args, **kwargs):
        ImageCEMPlanner.log_verbose(self, *args, **kwargs)



