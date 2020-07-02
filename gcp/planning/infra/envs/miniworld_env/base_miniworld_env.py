from gcp.planning.infra.envs.base_env import BaseEnv


class BaseMiniworldEnv(BaseEnv):
    def __init__(self, hp):
        self._hp = hp
        self._ncam = 1
        self.num_objects = None

        self._goal = None
        self._goaldistances = []
        self._initial_shortest_dist, self._final_shortest_dist = None, None
        self._full_traj = []

    def reset(self):
        self._goaldistances = []
        self._initial_shortest_dist, self._final_shortest_dist = None, None
        self._full_traj = []

    def set_goal(self, goal):
        self._goal = goal

    def add_goal_dist(self, goal_dist):
        self._goaldistances.append(goal_dist)

    def valid_rollout(self):
        return True     # no invalid states in miniworld env

    def eval(self):
        assert self._initial_shortest_dist is not None and self._final_shortest_dist is not None    # need to be set by subclass before eval!

        stats = {}
        stats['improvement'] = self._initial_shortest_dist - self._final_shortest_dist
        stats['initial_dist'] = self._initial_shortest_dist
        stats['final_dist'] = self._final_shortest_dist
        stats['all_goal_distances'] = self._goaldistances
        stats['full_traj'] = self._full_traj
        stats['goal'] = self._goal
        # TODO add success rate
        return stats

    def get_reset_from_obs(self, obs_dict):
        reset_state = {}
        reset_state['state'] = obs_dict['state'][0]
        reset_state['qpos_full'] = obs_dict['qpos_full'][0]
        return reset_state

    @staticmethod
    def default_ncam():
        return 1

    @property
    def ncam(self):
        return self._ncam

    @property
    def cam_res(self):
        return self._frame_height, self._frame_width



