""" This file defines an agent for the MuJoCo simulator environment. """
import copy
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from gcp.infra.policy.policy import get_policy_args
from gcp.infra.utils.im_utils import resize_store, npy_to_gif
from tensorflow.contrib.training import HParams


class Image_Exception(Exception):
    def __init__(self):
        pass


class Environment_Exception(Exception):
    def __init__(self):
        pass


class GeneralAgent(object):
    """
    All communication between the algorithms and MuJoCo is done through
    this class.
    """
    N_MAX_TRIALS = 100

    def __init__(self, hyperparams, start_goal_list=None):

        self._hp = self._default_hparams()
        self.override_defaults(hyperparams)

        self.T = self._hp.T
        self._start_goal_list = start_goal_list
        self._goal = None
        self._goal_seq = None
        self._goal_image = None
        self._demo_images = None
        self._reset_state = None
        self._setup_world(0)

    def override_defaults(self, config):
        """
        :param config:  override default valus with config dict
        :return:
        """
        for name, value in config.items():
            print('overriding param {} to value {}'.format(name, value))
            if value == getattr(self._hp, name):
                raise ValueError("attribute is {} is identical to default value!!".format(name))
            if name in self._hp and self._hp.get(name) is None:   # don't do a type check for None default values
                setattr(self._hp, name, value)
            else: self._hp.set_hparam(name, value)

    def _default_hparams(self):
        default_dict = {
            'T': None,
            'adim': None,
            'sdim': None,
            'ncam': 1,
            'rejection_sample': False,   # repeatedly attemp to collect a trajectory if error occurs
            'type': None,
            'env': None,
            'image_height': 48,
            'image_width': 64,
            'nchannels': 3,
            'data_save_dir': '',
            'log_dir': '',
            'make_final_gif': True,   # whether to make final gif
            'make_final_gif_freq': 1,   # final gif, frequency
            'make_final_gif_pointoverlay': False,
            'gen_xml': (True, 1),  # whether to generate xml, and how often
            'start_goal_confs': None,
            'show_progress': False,
            'state_resets': False,   # reset the simluator state zeroing velocities according to policies replan frequency
            'do_not_save_images': False  # dataset savers will not save images if True
        }
        # add new params to parent params
        parent_params = HParams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def _setup_world(self, itr):
        """
        Helper method for handling setup of the MuJoCo world.
        Args:
            filename: Path to XML file containing the world information.
        """
        env_type, env_params = self._hp.env
        if self._start_goal_list is not None:
            env_params['init_pos'] = self._start_goal_list[itr, 0]
            env_params['goal_pos'] = self._start_goal_list[itr, 1]
        self.env = env_type(env_params, self._reset_state)

        self._hp.adim = self.adim = self.env.adim
        self._hp.sdim = self.sdim = self.env.sdim
        self._hp.ncam = self.ncam = self.env.ncam
        self.num_objects = self.env.num_objects

    def sample(self, policy, i_traj):
        """
        Runs a trial and constructs a new sample containing information
        about the trial.
        """
        self.i_traj = i_traj
        if self._hp.gen_xml[0]:
            if i_traj % self._hp.gen_xml[1] == 0 and i_traj > 0:
                self._setup_world(i_traj)

        traj_ok, obs_dict, policy_outs, agent_data = False, None, None, None
        i_trial = 0
        imax = self.N_MAX_TRIALS

        while not traj_ok and i_trial < imax:
            i_trial += 1
            try:
                agent_data, obs_dict, policy_outs = self.rollout(policy, i_trial, i_traj)
                traj_ok = agent_data['traj_ok']
            except Image_Exception:
                traj_ok = False
            if not traj_ok:
                print('traj_ok: ', traj_ok)

        print('needed {} trials'.format(i_trial))

        if self._hp.make_final_gif or self._hp.make_final_gif_pointoverlay:
            if i_traj % self._hp.make_final_gif_freq == 0:
                self.save_gif(i_traj, self._hp.make_final_gif_pointoverlay)
                # self.plot_endeff_traj(obs_dict)

        self._reset_state = None # avoid reusing the same reset state

        return agent_data, obs_dict, policy_outs

    def _post_process_obs(self, env_obs, agent_data, initial_obs=False, stage=None):
        """
        Handles conversion from the environment observations, to agent observation
        space. Observations are accumulated over time, and images are resized to match
        the given image_heightximage_width dimensions.

        Original images from cam index 0 are added to buffer for saving gifs (if needed)

        Data accumlated over time is cached into an observation dict and returned. Data specific to each
        time-step is returned in agent_data

        :param env_obs: observations dictionary returned from the environment
        :param initial_obs: Whether or not this is the first observation in rollout
        :return: obs: dictionary of observations up until (and including) current timestep
        """
        agent_img_height = self._hp.image_height
        agent_img_width = self._hp.image_width

        if stage is not None:
            env_obs['stage'] = stage

        if initial_obs:
            T = self._hp.T + 1
            self._agent_cache = {}

            for k in env_obs:
                if k == 'images':
                    if 'obj_image_locations' in env_obs:
                        self.traj_points = []
                    self._agent_cache['images'] = np.zeros((T, self._hp.ncam, agent_img_height, agent_img_width, self._hp.nchannels), dtype=np.uint8)
                elif isinstance(env_obs[k], np.ndarray):
                    obs_shape = [T] + list(env_obs[k].shape)
                    self._agent_cache[k] = np.zeros(tuple(obs_shape), dtype=env_obs[k].dtype)
                else:
                    self._agent_cache[k] = []
            self._cache_cntr = 0

        t = self._cache_cntr
        self._cache_cntr += 1

        point_target_width = agent_img_width

        obs = {}

        if self._hp.show_progress:
            plt.imshow(env_obs['images'][0])
            path = self._hp.log_dir + '/verbose/traj{}/'.format(self.i_traj)
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(path + '/im{}.png'.format(t))

        for k in env_obs:
            if k == 'images':
                resize_store(t, self._agent_cache['images'], env_obs['images'])
                self.gif_images_traj.append(self._agent_cache['images'][t,0])  # only take first camera
            elif k == 'obj_image_locations':
                self.traj_points.append(copy.deepcopy(env_obs['obj_image_locations'][0]))  # only take first camera
                env_obs['obj_image_locations'] = np.round((env_obs['obj_image_locations'] *
                                                           point_target_width / env_obs['images'].shape[2])).astype(
                    np.int64)
                self._agent_cache['obj_image_locations'][t] = env_obs['obj_image_locations']
            elif isinstance(env_obs[k], np.ndarray):
                self._agent_cache[k][t] = env_obs[k]
            else:
                self._agent_cache[k].append(env_obs[k])
            obs[k] = self._agent_cache[k][:self._cache_cntr]

            if k == 'topdown_image':
                self.topdown_images.append((self._agent_cache['topdown_image'][t]*255).astype(np.uint8))  # only take first camera

        if 'obj_image_locations' in env_obs:
            agent_data['desig_pix'] = env_obs['obj_image_locations']
        if self._goal_image is not None:
            agent_data['goal_image'] = self._goal_image

        if self._goal is not None:
            agent_data['goal'] = self._goal
        if self._demo_images is not None:
            agent_data['demo_images'] = self._demo_images
        if self._reset_state is not None:
            agent_data['reset_state'] = self._reset_state
            obs['reset_state'] = self._reset_state
        return obs

    def _required_rollout_metadata(self, agent_data, traj_ok, t, i_tr):
        """
        Adds meta_data such as whether the goal was reached and the total number of time steps
        into the agent dictionary that is MANDATORY for later parts of pipeline
        :param agent_data: Agent data dictionary
        :param traj_ok: Whether or not rollout succeeded
        :return: None
        """
        agent_data['term_t'] = t - 1
        if self.env.has_goal():
            agent_data['goal_reached'] = self.env.goal_reached()
        agent_data['traj_ok'] = traj_ok

    def rollout(self, policy, i_trial, i_traj):
        """
        Rolls out policy for T timesteps
        :param policy: Class extending abstract policy class. Must have act method (see arg passing details)
        :param i_trial: Rollout attempt index (increment each time trajectory fails rollout)
        :return: - agent_data: Dictionary of extra statistics/data collected by agent during rollout
                 - obs: dictionary of environment's observations. Each key maps to that values time-history
                 - policy_ouputs: list of policy's outputs at each timestep.
                 Note: tfrecord saving assumes all keys in agent_data/obs/policy_outputs point to np arrays or primitive int/float
        """
        self._init()

        agent_data, policy_outputs = {}, []

        # Take the sample.
        t = 0
        done = self._hp.T <= 0
        initial_env_obs, self._reset_state = self.env.reset(self._reset_state)
        obs = self._post_process_obs(initial_env_obs, agent_data, True, stage=0)
        policy.reset()

        while not done:
            """
            Every time step send observations to policy, acts in environment, and records observations

            Policy arguments are created by
                - populating a kwarg dict using get_policy_arg
                - calling policy.act with given dictionary

            Policy returns an object (pi_t) where pi_t['actions'] is an action that can be fed to environment
            Environment steps given action and returns an observation
            """

            if self._hp.state_resets:   # reset the simulator to state so that mujoco-based cem-planning can strat from there.
                if t % policy.replan_interval == 0 and t != 0:
                    print('_____')
                    print('gen_ag: performing state reset ')
                    self.env.qpos_reset(obs['qpos_full'][t], obs['qvel_full'][t])

                    new_obs = self.env._get_obs()
                    print('qpos of t ', new_obs['qpos'])
                    print('qvel of t', new_obs['qvel'])
                    print('_____')

            pi_t = policy.act(**get_policy_args(policy, obs, t, i_traj, agent_data))
            policy_outputs.append(pi_t)

            if 'done' in pi_t:
                done = pi_t['done']
            try:
                obs = self._post_process_obs(self.env.step(pi_t['actions']), agent_data)
                # obs = self._post_process_obs(self.env.step(copy.deepcopy(pi_t['actions']), stage=stage), agent_data, stage=pi_t['policy_index'])
            except Environment_Exception as e:
                print(e)
                return {'traj_ok': False}, None, None


            if (self._hp.T - 1) == t or obs['env_done'][-1]:   # environements can include the tag 'env_done' in the observations to signal that time is over
                done = True
            t += 1
            print('t', t)

        traj_ok = self.env.valid_rollout()
        if self._hp.rejection_sample:
            if self._hp.rejection_sample > i_trial:
                assert self.env.has_goal(), 'Rejection sampling enabled but env has no goal'
                traj_ok = self.env.goal_reached()
            print('goal_reached', self.env.goal_reached())

        self._required_rollout_metadata(agent_data, traj_ok, t, i_trial)
        obs.update(self.env.add_extra_obs_info())

        return agent_data, obs, policy_outputs

    def save_gif(self, i_traj, overlay=False):
        if self.traj_points is not None and overlay:
            colors = [tuple([np.random.randint(0, 256) for _ in range(3)]) for __ in range(self.num_objects)]
            for pnts, img in zip(self.traj_points, self.gif_images_traj):
                for i in range(self.num_objects):
                    center = tuple([int(np.round(pnts[i, j])) for j in (1, 0)])
                    cv2.circle(img, center, 4, colors[i], -1)

        file_path = self._hp.log_dir
        # plt.switch_backend('tkagg')
        # plt.imshow(self.gif_images_traj[0])
        # plt.show()
        npy_to_gif(self.gif_images_traj, file_path + '/verbose/traj{}/video'.format(i_traj)) # todo make extra folders for each run?

        if False: #len(self.topdown_images) > 0:
            npy_to_gif(self.topdown_images, file_path + '/verbose/traj{}/topdownvideo'.format(i_traj))

    def plot_endeff_traj(self, obs_dict):
        endeff_pos = obs_dict['regression_state'][:,:3]
        xpos = endeff_pos[:,0]
        zpos = endeff_pos[:,2]
        plt.switch_backend('TkAgg')
        plt.plot(xpos, zpos)
        plt.show()

    def _init(self):
        """
        Set the world to a given model
        """
        self.gif_images_traj, self.topdown_images, self.traj_points = [], [], None


