import numpy as np
from blox import AttrDict
from blox.basic_types import listdict2dictlist


class CEMSampler:
    """Defines interface for sampler used in CEM optimization loop."""
    def __init__(self, clip_val, n_steps, action_dim, initial_std):
        self._clip_val = clip_val
        self._n_steps = n_steps
        self._action_dim = action_dim
        self._initial_std = initial_std
        self.init()

    def init(self):
        """Initialize the sampling distributions."""
        raise NotImplementedError

    def sample(self, n_samples):
        """Sample n_samples from the sampling distributions."""
        raise NotImplementedError

    def fit(self, data, scores):
        """Refits distributions to data."""
        raise NotImplementedError

    def get_dists(self):
        """Returns a representation of the current sampling distributions."""
        raise NotImplementedError


class FlatCEMSampler(CEMSampler):
    """Samples flat arrays from Gaussian distributions."""
    def init(self):
        """Initialize the sampling distributions."""
        self.mean = np.zeros((self._n_steps, self._action_dim))
        self.std = self._initial_std * np.ones((self._n_steps, self._action_dim))

    def sample(self, n_samples):
        raw_actions = np.random.normal(loc=self.mean, scale=self.std, size=(n_samples, self._n_steps, self._action_dim))
        return np.clip(raw_actions, -self._clip_val, self._clip_val)

    def fit(self, data, scores):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

    def get_dists(self):
        return AttrDict(mean=self.mean, std=self.std)


class PDDMSampler(FlatCEMSampler):
    """Samples correlated noise, uses path integral formulation to fit it."""
    BETA = 0.5      # noise correlation factor
    GAMMA = 1.0     # reward weighting factor

    def sample(self, n_samples):
        noise = np.random.normal(loc=np.zeros_like(self.mean), scale=self.std,
                                 size=(n_samples, self._n_steps, self._action_dim))
        correlated_noise, n_i = [], np.zeros((n_samples, self._action_dim))
        for i in range(noise.shape[1]):
            u_i = noise[:, i]
            n_i = self.BETA * u_i + (1 - self.BETA) * n_i
            correlated_noise.append(n_i)
        correlated_noise = np.stack(correlated_noise, axis=1)
        return np.clip(correlated_noise + self.mean[None], -self._clip_val, self._clip_val)

    def fit(self, actions, scores):
        """Assumes that scores are better the lower (ie cost function output)."""
        self.mean = np.sum(actions * np.exp(-self.GAMMA * scores)[:, None, None], axis=0) \
                    / np.sum(np.exp(-self.GAMMA * scores))


class SimpleHierarchicalCEMSampler(FlatCEMSampler):
    """Hierarchical CEM sampler that treats all levels equally."""
    def __init__(self, *args, n_level_hierarchy, **kwargs):
        self._n_layer_hierarchy = n_level_hierarchy
        super().__init__(*args)
        self._n_steps = 2**n_level_hierarchy - 1


class FlexibleHierarchicalSampler(SimpleHierarchicalCEMSampler):
    """Allows for different sampling rates of different layers in the hierarchy."""
    def __init__(self, *args, sampling_rates_per_layer, subgoal_cost_fcn, ll_cost_fcn, n_ll_samples, **kwargs):
        self._sampling_rates_per_layer = sampling_rates_per_layer
        self._subgoal_cost_fcn = subgoal_cost_fcn
        self._ll_cost_fcn = ll_cost_fcn
        self._n_ll_samples = n_ll_samples
        super().__init__(*args, **kwargs)
        assert self._n_layer_hierarchy >= len(sampling_rates_per_layer)     # not enough layers in tree

    def init(self):
        self._segment_tree = ParallelOptTrajectorySegmentTree(self._action_dim,
                                                              self._sampling_rates_per_layer.copy(),
                                                              self._n_layer_hierarchy,
                                                              self._subgoal_cost_fcn,
                                                              self._ll_cost_fcn,
                                                              self._n_ll_samples)

    def sample(self, n_samples):
        raw_actions = self._segment_tree.sample()
        return np.clip(raw_actions, -self._clip_val, self._clip_val)

    def optimize(self, rollouts, goal):
        best_rollout, best_cost = self._segment_tree.optimize(rollouts, goal)
        if (best_rollout[-1] != goal).any():    # this can happen if too few frames on right tree side
            best_rollout = np.concatenate((best_rollout, goal[None]))
        return [best_rollout], best_cost

    def fit(*args, **kwargs):
        """Does not currently support refitting distributions."""
        pass

    def get_dists(self):
        return AttrDict(mean=0., std=1.)    # dummy values

    @property
    def append_latent(self):
        return False        # do not append latents in rollouts

    @property
    def fully_optimized(self):
        return self._segment_tree.fully_optimized


class SequentialHierarchicalSampler(FlexibleHierarchicalSampler):
    """Optimizes the layers of the hierarchy sequentially, starting from the top."""
    def init(self):
        self._segment_tree = SequentialOptTrajectorySegmentTree(self._action_dim,
                                                                self._sampling_rates_per_layer.copy(),
                                                                self._n_layer_hierarchy,
                                                                self._subgoal_cost_fcn,
                                                                self._ll_cost_fcn,
                                                                self._n_ll_samples)

    @property
    def append_latent(self):
        return True     # we need latent rollouts to compute subgoal costs


class ImageSequentialHierarchicalSampler(SequentialHierarchicalSampler):
    """Optimizes the layers of the hierarchy sequentially, starting from the top."""
    def init(self):
        self._segment_tree = ImageSequentialOptTrajectorySegmentTree(self._action_dim,
                                                                     self._sampling_rates_per_layer.copy(),
                                                                     self._n_layer_hierarchy,
                                                                     self._subgoal_cost_fcn,
                                                                     self._ll_cost_fcn,
                                                                     self._n_ll_samples)

    def optimize(self, rollouts, goal):
        best_rollout, best_cost = self._segment_tree.optimize(rollouts, goal)
        if (best_rollout[-1] != goal[0].transpose(2, 0, 1)).any():    # can happen if too few frames on right tree side
            best_rollout = np.concatenate((best_rollout, goal.transpose(0, 3, 1, 2)))
        if not hasattr(best_cost, "__len__"):
            best_cost = [best_cost]         # need to return array-shaped cost, no scalar
        return [best_rollout], best_cost


class ParallelOptTrajectorySegmentTree:
    """Holds sampling distributions for subgoal + reference to lower-level segments.
       Samples and optimizes for all layers in parallel."""
    def __init__(self, action_dim, sampling_rates, depth, subgoal_cost_fcn, ll_cost_fcn, final_layer_samples):
        self._action_dim = action_dim
        self._depth = depth
        self._subgoal_cost_fcn = subgoal_cost_fcn
        self._ll_cost_fcn = ll_cost_fcn
        self._is_optimized = False  # indicates whether this layer of the tree is already optimized
        self._opt_z = None  # holds optimal subgoal once optimized
        self._latest_z_samples = None  # holds last-sampled z samples for subgoal opt
        if sampling_rates:
            # not yet at bottom-most layer
            self._n_samples = sampling_rates.pop(0)
            self._n_latents = 1
            self._children = [[type(self)(action_dim, sampling_rates.copy(), depth - 1,
                                          self._subgoal_cost_fcn, self._ll_cost_fcn,
                                          final_layer_samples)
                               for _ in range(self._n_samples)] for _ in range(2)]
        else:
            # final layer, create remaining samples
            self._n_samples = final_layer_samples
            self._n_latents = 2**depth - 1
            self._children = None

        self.mean = np.zeros((self._n_latents, self._action_dim))
        self.std = np.ones((self._n_latents, self._action_dim))

    def sample(self):
        """Samples latents from all segments of the tree, returns concatenated result."""
        # sample own latents
        z = self._sample()

        if self._children is not None:
            # sample children's latents and concatenate
            samples = []
            for child_left, child_right, z_i in zip(self._children[0], self._children[1], z):
                z_left, z_right = child_left.sample(), child_right.sample()
                assert z_left.shape == z_right.shape    # latent tree needs to be balanced
                samples.append(np.concatenate([z_left, np.tile(z_i[0], (z_left.shape[0], 1, 1)), z_right], axis=1))
            z = np.concatenate(samples)

        self._latest_z_samples = z.copy()
        return z

    def optimize(self, all_rollouts, goal):
        """Returns best sequence and cost."""
        if self._children is None:
            best_rollout, best_cost, best_idx = self._best_of_n(all_rollouts, goal, self._ll_cost_fcn)
            self._opt_z = self._latest_z_samples[best_idx]
            self._is_optimized = True
        else:
            per_latent_rollouts = np.array_split(all_rollouts, self._n_samples)
            best_costs, best_rollouts = [], []
            for child_left, child_right, rollouts in zip(self._children[0], self._children[1], per_latent_rollouts):
                rollouts = [r for r in rollouts]  # convert from array of arrays to list

                # filter too short rollouts that don't need hierarchical expansion, replace with dummy
                short_rollouts = []
                for r_idx, r in enumerate(rollouts):
                    if r.shape[0] < 3:
                        short_rollouts.append(r)
                        rollouts[r_idx] = self._make_dummy_seq(r[0])

                # expand hierarchically
                subgoal_inds = [int(np.floor(r.shape[0] / 2)) for r in rollouts]
                subgoal = rollouts[0][subgoal_inds[0]]  # across batch dimension all of the subgoals are identical
                best_rollout_left, best_cost_left = \
                    child_left.optimize([r[:si] for r, si in zip(rollouts, subgoal_inds)], subgoal)
                best_rollout_right, best_cost_right = \
                    child_right.optimize([r[si:] for r, si in zip(rollouts, subgoal_inds)], goal)
                best_rollout = np.concatenate([best_rollout_left, best_rollout_right])
                best_cost = best_cost_left + best_cost_right

                # check whether too short trajectories are better, if so: replace results
                if short_rollouts:
                    best_rollout_short, best_cost_short, _ = self._best_of_n(short_rollouts, goal, self._ll_cost_fcn)
                    if best_cost_short < best_cost or np.isnan(best_cost):
                        best_rollout, best_cost = best_rollout_short, best_cost_short

                # dump best results for this latent
                best_rollouts.append(best_rollout); best_costs.append(best_cost)

            best_cost_idx = np.argmin(np.array(best_costs))
            best_rollout, best_cost = best_rollouts[best_cost_idx], best_costs[best_cost_idx]
        return best_rollout, best_cost

    def _sample(self):
        return np.random.normal(loc=self.mean, scale=self.std, size=(self._n_samples, self._n_latents, self._action_dim))

    @staticmethod
    def _best_of_n(all_rollouts, goal, cost_fcn):
        """Computes best sequence out of N flat sequences (input is list of seqs)."""
        cost = cost_fcn(all_rollouts, goal)
        best_cost_idx = np.argmin(cost)
        best_rollout, best_cost = all_rollouts[best_cost_idx], cost[best_cost_idx]
        return best_rollout, best_cost, best_cost_idx

    @staticmethod
    def _make_dummy_seq(reference_array):
        return np.stack([np.ones_like(reference_array) * float("inf"),  # fill with dummy w/ max cost
                         np.zeros_like(reference_array),
                         np.ones_like(reference_array) * float("inf")])

    @property
    def fully_optimized(self):
        if self._children is not None:
            return not (not self._is_optimized or
                        np.any([not c.fully_optimized for c in self._children[0]]) or
                        np.any([not c.fully_optimized for c in self._children[1]]))
        else:
            return self._is_optimized


class SequentialOptTrajectorySegmentTree(ParallelOptTrajectorySegmentTree):
    """Holds sampling distributions for subgoal + reference to lower-level segments.
       Samples and optimizes one layer at a time."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_optimized = False      # indicates whether this layer of the tree is already optimized
        self._opt_z = None      # holds optimal subgoal once optimized
        self._latest_z_samples = None   # holds last-sampled z samples for subgoal opt
        self._dummy_env = None      # used for visualization purposes

    def sample(self, below_opt_layer=False):
        """Samples latents from all segments of the tree, returns concatenated result.
            below_opt_layer indicates whether the layer is below the layer that is currently getting optimized."""
        # sample own latents
        if self._is_optimized:
            z = self._opt_z.copy()[None]
        else:
            z = self._sample()[:1] if below_opt_layer else self._sample()   # if below only sample single sample
            self._latest_z_samples = z.copy()
        next_below_opt_layer = (not self._is_optimized and not below_opt_layer) \
                                    or below_opt_layer  # in the first case the current layer is getting optimized
        if self._children is not None:
            # sample children's latents and concatenate
            samples = []
            for child_left, child_right, z_i in zip(self._children[0], self._children[1], z):
                z_left, z_right = child_left.sample(next_below_opt_layer), child_right.sample(next_below_opt_layer)
                assert z_left.shape == z_right.shape    # latent tree needs to be balanced
                samples.append(np.concatenate([z_left, np.tile(z_i[0], (z_left.shape[0], 1, 1)), z_right], axis=1))
            z = np.concatenate(samples)

        return z

    def optimize(self, all_rollouts, goal):
        """Optimizes subgoal in all layers but the last, optimizes full rollout in last layer."""
        if not self._is_optimized and self._children is not None:
            # perform subgoal optimization - first split off latents from concatenated rollouts
            rollouts = self._split_state_rollout(all_rollouts)

            # prepare start + goal + subgoal arrays
            starts, start_latents = np.stack([r[0] for r in rollouts.state_rollout]), \
                                    np.stack([r[0] for r in rollouts.latent_rollout])
            subgoals = np.stack([r[int(np.floor(r.shape[0] / 2))] for r in rollouts.state_rollout])
            subgoal_latents = np.stack([r[int(np.floor(r.shape[0] / 2))] for r in rollouts.latent_rollout])
            goals = np.stack([self._split_state_rollout([goal[None]]).state_rollout[0][0] \
                                     if goal.shape[-1] == all_rollouts[0].shape[-1] else goal
                                     for _ in rollouts.state_rollout])
            goal_latents = np.stack([self._split_state_rollout([goal[None]]).latent_rollout[0][0] \
                                     if goal.shape[-1] == all_rollouts[0].shape[-1] else r[-1]
                                     for r in rollouts.latent_rollout])

            # compute pairwise cost
            to_cost, from_cost = self._subgoal_cost_fcn(start_latents, subgoal_latents), \
                                 self._subgoal_cost_fcn(subgoal_latents, goal_latents)
            total_cost = to_cost + from_cost

            # find optimal subgoal
            opt_z_idx = np.argmin(total_cost)
            self._opt_z = self._latest_z_samples[opt_z_idx]

            # construct output rollout + output_cost,
            outputs = [starts[opt_z_idx]]
            if (subgoals[opt_z_idx] != outputs[-1]).any():  # they can be the same if sequence is too short
                outputs.append(subgoals[opt_z_idx])
            if not goal.shape[-1] == all_rollouts[0].shape[-1]:     # only append very final goal once
                if goals[opt_z_idx].shape == outputs[-1].shape:
                    outputs.append(goals[opt_z_idx])
                else:
                    outputs.append(goals[opt_z_idx][0].transpose(2, 0, 1))  # for image-based CEM
            output_rollout = np.stack(outputs)
            output_cost = total_cost[opt_z_idx]

            self._children = [c[:1] for c in self._children]    # remove children for other samples, not necessary
            self._n_samples = 1
            self._is_optimized = True

            # log all options to output
            if self._depth == 8:
                self._log_all_subgoal_plans(starts, subgoals, goals, to_cost, from_cost)
            return output_rollout, output_cost
        else:
            return super().optimize(all_rollouts, goal)

    def _best_of_n(self, all_rollouts, goal, cost_fcn, run_super=False):
        if run_super:
            return super()._best_of_n(all_rollouts, goal, cost_fcn)
        rollouts = self._split_state_rollout(all_rollouts).state_rollout
        state_goal = self._split_state_rollout([goal]).state_rollout[0] if goal.shape[-1] == all_rollouts[0].shape[-1] \
                        else goal
        return super()._best_of_n(rollouts, state_goal, cost_fcn)

    def _split_state_rollout(self, rollouts):
        """Splits off latents from states in joined rollouts."""
        return listdict2dictlist([AttrDict(state_rollout=r[..., :-self._subgoal_cost_fcn.input_dim],
                                           latent_rollout=r[..., -self._subgoal_cost_fcn.input_dim:]) for r in rollouts])

    def _log_all_subgoal_plans(self, starts, subgoals, goals, to_cost, from_cost):
        for s, sg, g, tc, fc in zip(starts, subgoals, goals, to_cost, from_cost):
           r = np.stack([s, sg, g])
           c = tc + fc
           self._log_subgoal_plan(r, c)

    def _log_subgoal_plan(self, rollout, cost):
        from gcp.infra.envs.miniworld_env.multiroom3d.multiroom3d_env import Multiroom3dEnv
        import time
        import cv2, os
        if self._dummy_env is None:
            self._dummy_env = Multiroom3dEnv({'n_rooms': 25}, no_env=True)
        im = self._dummy_env.render_top_down(rollout)
        name = "subgoal_{}_{}.png".format(cost, time.time())
        cv2.imwrite(os.path.join("/parent/tmp", name), im*255.)


class ImageSequentialOptTrajectorySegmentTree(SequentialOptTrajectorySegmentTree):
    def _best_of_n(self, all_rollouts, goal, cost_fcn, *unused_args, **unused_kwargs):
        rollouts = self._split_state_rollout(all_rollouts).latent_rollout

        # this is a hack because we sometimes dont have the goal encoding
        if len(goal.shape) > 2:
            state_goal = [r[-1:] for r in rollouts]
        else:
            state_goal = [self._split_state_rollout([goal[None]]).latent_rollout[0] for _ in rollouts]
        best_latent_rollout, best_cost, best_idx = super()._best_of_n(rollouts, state_goal, cost_fcn, run_super=True)
        return self._split_state_rollout(all_rollouts).state_rollout[best_idx], best_cost, best_idx

    def _split_state_rollout(self, rollouts):
        """Splits off latents from states in joined rollouts."""
        def reshape_to_image(flat):
            if len(flat.shape) != 2:
                import pdb; pdb.set_trace()
            assert len(flat.shape) == 2
            res = int(np.sqrt(flat.shape[1] / 3))   # assumes 3-channel image
            return flat.reshape(flat.shape[0], 3, res, res)
        return listdict2dictlist([AttrDict(state_rollout=reshape_to_image(r[..., :-self._subgoal_cost_fcn.input_dim]),
                                           latent_rollout=r[..., -self._subgoal_cost_fcn.input_dim:]) for r in rollouts])

    def _log_all_subgoal_plans(self, starts, subgoals, goals, to_cost, from_cost):
        import cv2, time, os
        img_stack = []
        for s, sg, g, tc, fc in zip(starts, subgoals, goals, to_cost, from_cost):
            if len(g.shape) == 4:
                g = g[0].transpose(2, 0, 1) * 2 - 1
            img_strip = (np.concatenate((s, sg, g), axis=2).transpose(1, 2, 0) + 1) / 2
            img_strip = cv2.resize(img_strip[..., ::-1], (0,0), fx=4, fy=4)
            img_strip = cv2.putText(img_strip, "{:.2f}".format(float(fc+tc)),
                                    (10,img_strip.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
            img_strip = cv2.putText(img_strip, "{:.2f}".format(float(fc+tc)),
                                    (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1.0, 1.0, 1.0), 2)
            img_stack.append(img_strip)
        name = "subgoals_{}.png".format(time.time())
        cv2.imwrite(os.path.join("/parent/tmp", name), np.concatenate(img_stack, axis=0) * 255)


if __name__ == "__main__":

    sampler = PDDMSampler(1.0, 100, 4, initial_std=0.1)
    s = sampler.sample(10)
    sampler.fit(s, np.random.rand(10))

    # sampler = SequentialHierarchicalSampler(1.0, 0, 128, n_level_hierarchy=7, sampling_rates_per_layer=[2, 5, 10])
    # sampler.init()
    # samples = sampler.sample(2*5*10)
    #
    # from gcp.infra.policy.cem_policy.utils.cost_fcn import EuclideanPathLength
    # cost_fcn = EuclideanPathLength(dense_cost=True, final_step_weight=1.0)
    # goal = np.random.rand(2)
    # rollouts = [np.random.rand(5, 2+128) for _ in range(samples.shape[0])]
    # # rollouts = [np.random.rand(5, 2 + 128)] + [np.random.rand(5, 2 + 128) for _ in range(1)]
    # best_traj, best_cost = sampler.optimize(rollouts, cost_fcn, goal)
    #
    # samples = sampler.sample(2 * 5 * 10)
    # rollouts = [np.random.rand(5, 2 + 128) for _ in range(samples.shape[0])]
    # best_traj, best_cost = sampler.optimize(rollouts, cost_fcn, goal)
    #
    # samples = sampler.sample(2 * 5 * 10)
    # rollouts = [np.random.rand(5, 2 + 128) for _ in range(samples.shape[0])]
    # best_traj, best_cost = sampler.optimize(rollouts, cost_fcn, goal)
    #
    # samples = sampler.sample(2 * 5 * 10)
    # rollouts = [np.random.rand(5, 2 + 128) for _ in range(samples.shape[0])]
    # best_traj, best_cost = sampler.optimize(rollouts, cost_fcn, goal)
    # x = 0


