import numpy as np

from blox import AttrDict
from gcp.planning.tree_optimizer import HierarchicalTreeLatentOptimizer, ImageHierarchicalTreeLatentOptimizer


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


class SimpleTreeCEMSampler(FlatCEMSampler):
    """CEM sampler for tree-GCPs that optimizes all levels at once with same number of samples
       (ie like flat CEM sampler)."""
    def __init__(self, *args, n_level_hierarchy, **kwargs):
        self._n_layer_hierarchy = n_level_hierarchy
        super().__init__(*args)
        self._n_steps = 2**n_level_hierarchy - 1


class HierarchicalTreeCEMSampler(SimpleTreeCEMSampler):
    """Tree-GCP CEM sampler that optimizes the layers of the hierarchy sequentially, starting from the top."""
    def __init__(self, *args, sampling_rates_per_layer, subgoal_cost_fcn, ll_cost_fcn, n_ll_samples, **kwargs):
        self._sampling_rates_per_layer = sampling_rates_per_layer
        self._subgoal_cost_fcn = subgoal_cost_fcn
        self._ll_cost_fcn = ll_cost_fcn
        self._n_ll_samples = n_ll_samples
        super().__init__(*args, **kwargs)
        assert self._n_layer_hierarchy >= len(sampling_rates_per_layer)     # not enough layers in tree

    def init(self):
        self._optimizer = HierarchicalTreeLatentOptimizer(self._action_dim,
                                                          self._sampling_rates_per_layer.copy(),
                                                          self._n_layer_hierarchy,
                                                          self._subgoal_cost_fcn,
                                                          self._ll_cost_fcn,
                                                          self._n_ll_samples)

    def sample(self, n_samples):
        raw_actions = self._optimizer.sample()
        return np.clip(raw_actions, -self._clip_val, self._clip_val)

    def optimize(self, rollouts, goal):
        best_rollout, best_cost = self._optimizer.optimize(rollouts, goal)
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
        return True  # we need latent rollouts to compute subgoal costs

    @property
    def fully_optimized(self):
        return self._optimizer.fully_optimized


class ImageHierarchicalTreeCEMSampler(HierarchicalTreeCEMSampler):
    """Hierarchical GCP-tree CEM sampler for image prediction GCPs."""
    def init(self):
        self._optimizer = ImageHierarchicalTreeLatentOptimizer(self._action_dim,
                                                               self._sampling_rates_per_layer.copy(),
                                                               self._n_layer_hierarchy,
                                                               self._subgoal_cost_fcn,
                                                               self._ll_cost_fcn,
                                                               self._n_ll_samples)

    def optimize(self, rollouts, goal):
        best_rollout, best_cost = self._optimizer.optimize(rollouts, goal)
        if (best_rollout[-1] != goal[0].transpose(2, 0, 1)).any():    # can happen if too few frames on right tree side
            best_rollout = np.concatenate((best_rollout, goal.transpose(0, 3, 1, 2)))
        if not hasattr(best_cost, "__len__"):
            best_cost = [best_cost]         # need to return array-shaped cost, no scalar
        return [best_rollout], best_cost
