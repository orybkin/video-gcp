import numpy as np

from blox import AttrDict
from blox.basic_types import listdict2dictlist


class HierarchicalTreeLatentOptimizer:
    """Optimizes latent distributions for GCP-tree layers recursively, one layer at a time.
       After N layers have been optimized hierarchically, all remaining layers are jointly optimized as 'segments'
       connecting between the planned subgoals."""
    def __init__(self, latent_dim, sampling_rates, depth, subgoal_cost_fcn, ll_cost_fcn, final_layer_samples):
        """
        :param latent_dim: dimensionality of optimized latent
        :param sampling_rates: per-layer sampling rates (except for last layer) as list
        :param depth: depth of GCP tree model who's latents are getting optimized
        :param subgoal_cost_fcn: cost function for estimating cost of sampled predictions
        :param ll_cost_fcn: cost function for estimating cost of dense prediction in last layer
        :param final_layer_samples: number of samples for optimizing last layer's dense trajectory predictions
        """
        self._latent_dim = latent_dim
        self._depth = depth
        self._subgoal_cost_fcn = subgoal_cost_fcn
        self._ll_cost_fcn = ll_cost_fcn
        self._is_optimized = False          # indicates whether this layer of the tree is already optimized
        self._opt_z = None                  # holds optimal subgoal latent once optimized
        self._latest_z_samples = None       # holds last-sampled z samples for subgoal optimization
        self._dummy_env = None              # used for visualization purposes
        if sampling_rates:
            # not yet at bottom-most layer
            self._n_samples = sampling_rates.pop(0)
            self._n_latents = 1
            self._children = [[type(self)(latent_dim, sampling_rates.copy(), depth - 1,
                                          self._subgoal_cost_fcn, self._ll_cost_fcn,
                                          final_layer_samples)
                               for _ in range(self._n_samples)] for _ in range(2)]
        else:
            # final layer, create remaining samples for non-hierarchical 'segment' optimization
            self._n_samples = final_layer_samples
            self._n_latents = 2**depth - 1
            self._children = None

        self.mean = np.zeros((self._n_latents, self._latent_dim))
        self.std = np.ones((self._n_latents, self._latent_dim))

    def sample(self, below_opt_layer=False):
        """Samples latents from all layers of the tree, returns concatenated result.
           Samples N latents for layer that's currently getting optimized, only 1 latent for all layers above and below.
        :param below_opt_layer: indicates whether layer is below the layer that is currently getting optimized."""
        # sample current layer's latents
        if self._is_optimized:      # if layer is already optimized --> sample optimized subgoal latent
            z = self._opt_z.copy()[None]
        else:
            # sample N latents if this is currently optimized layer,
            # if below optimized layer sample only single latent (since decoding won't be used for optimization)
            z = self._sample()[:1] if below_opt_layer else self._sample()
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
        """Optimizes subgoal in all layers sequentially, optimizes full rollout in last layer."""
        if self._children is None:      # final layer --> optimize dense segment
            return self._optimize_segment(all_rollouts, goal)
        elif not self._is_optimized:    # non-final layer, not optimized --> optimize subgoal
            return self._optimize_subgoal(all_rollouts, goal)
        else:                           # non-final layer, already optimized --> recurse
            return self._recurse_optimization(all_rollouts, goal)

    def _optimize_segment(self, all_rollouts, goal):
        """Optimizes final-layer 'segment' between subgoals."""
        best_rollout, best_cost, best_idx = self._best_of_n_segments(all_rollouts, goal, self._ll_cost_fcn)
        self._opt_z = self._latest_z_samples[best_idx]
        self._is_optimized = True
        return best_rollout, best_cost

    def _optimize_subgoal(self, all_rollouts, goal):
        """Optimizes subgoal latent by minimizing pairwise cost estimate with both parents."""
        # first split off latents from concatenated rollouts
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

        # construct output rollout + output_cost
        outputs = [starts[opt_z_idx]]
        if (subgoals[opt_z_idx] != outputs[-1]).any():  # they can be the same if sequence is too short
            outputs.append(subgoals[opt_z_idx])
        if not goal.shape[-1] == all_rollouts[0].shape[-1]:  # only append very final goal once
            if goals[opt_z_idx].shape == outputs[-1].shape:
                outputs.append(goals[opt_z_idx])
            else:
                outputs.append(goals[opt_z_idx][0].transpose(2, 0, 1))  # for image-based CEM
        output_rollout = np.stack(outputs)
        output_cost = total_cost[opt_z_idx]

        # remove children for all but optimal latent, indicate layer is optimized for subsequent optimization passes
        self._children = [c[:1] for c in self._children]
        self._n_samples = 1
        self._is_optimized = True

        # (optional) log all options to output for debugging, here in first layer of 8-layer GCP only
        # if self._depth == 8:
        #     self._log_all_subgoal_plans(starts, subgoals, goals, to_cost, from_cost)
        return output_rollout, output_cost

    def _recurse_optimization(self, all_rollouts, goal):
        """Splits sequence around subgoal and optimizes both parts independently.
           Handles edge cases of sub-trajetories too short for further recursion."""
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
                best_rollout_short, best_cost_short, _ = self._best_of_n_segments(short_rollouts, goal, self._ll_cost_fcn)
                if best_cost_short < best_cost or np.isnan(best_cost):
                    best_rollout, best_cost = best_rollout_short, best_cost_short

            # dump best results for this latent
            best_rollouts.append(best_rollout); best_costs.append(best_cost)

        best_cost_idx = np.argmin(np.array(best_costs))
        best_rollout, best_cost = best_rollouts[best_cost_idx], best_costs[best_cost_idx]
        return best_rollout, best_cost

    def _sample(self):
        return np.random.normal(loc=self.mean, scale=self.std, size=(self._n_samples, self._n_latents, self._latent_dim))

    def _best_of_n_segments(self, all_rollouts, goal, cost_fcn):
        """Applies dense cost function to segment samples, returns min-cost segment + min-cost + idx."""
        all_rollouts_opt, goal_opt = self._prep_segment_opt_inputs(all_rollouts, goal)
        cost = cost_fcn(all_rollouts_opt, goal_opt)
        best_cost_idx = np.argmin(cost)
        return self._split_state_rollout(all_rollouts).state_rollout[best_cost_idx], cost[best_cost_idx], best_cost_idx

    def _prep_segment_opt_inputs(self, all_rollouts, goal):
        """Splits off input to cost function from combined inputs (rollouts are concat of both state and latent)"""
        rollouts = self._split_state_rollout(all_rollouts).state_rollout
        state_goal = self._split_state_rollout([goal]).state_rollout[0] if goal.shape[-1] == all_rollouts[0].shape[-1] \
            else goal
        return rollouts, state_goal

    def _split_state_rollout(self, rollouts):
        """Splits off latents from states in joined rollouts."""
        return listdict2dictlist([AttrDict(state_rollout=r[..., :-self._subgoal_cost_fcn.input_dim],
                                           latent_rollout=r[..., -self._subgoal_cost_fcn.input_dim:]) for r in rollouts])

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

    def _log_all_subgoal_plans(self, starts, subgoals, goals, to_cost, from_cost):
        for s, sg, g, tc, fc in zip(starts, subgoals, goals, to_cost, from_cost):
           r = np.stack([s, sg, g])
           c = tc + fc
           self._log_subgoal_plan(r, c)

    def _log_subgoal_plan(self, rollout, cost):
        from gcp.planning.infra.envs.miniworld_env.multiroom3d.multiroom3d_env import Multiroom3dEnv
        import time
        import cv2, os
        if self._dummy_env is None:
            self._dummy_env = Multiroom3dEnv({'n_rooms': 25}, no_env=True)
        im = self._dummy_env.render_top_down(rollout)
        name = "subgoal_{}_{}.png".format(cost, time.time())
        cv2.imwrite(os.path.join("/parent/tmp", name), im*255.)


class ImageHierarchicalTreeLatentOptimizer(HierarchicalTreeLatentOptimizer):
    def _prep_segment_opt_inputs(self, all_rollouts, goal):
        rollouts = self._split_state_rollout(all_rollouts).latent_rollout
        if len(goal.shape) > 2:     # in case we dont have goal encoding use final state of rollout
            state_goal = [r[-1:] for r in rollouts]
        else:
            state_goal = [self._split_state_rollout([goal[None]]).latent_rollout[0] for _ in rollouts]
        return rollouts, state_goal

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
