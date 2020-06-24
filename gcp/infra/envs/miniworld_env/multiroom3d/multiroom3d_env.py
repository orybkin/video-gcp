import cv2
import copy
import gym
import numpy as np
from blox import AttrDict
from gcp.infra.envs.miniworld_env.base_miniworld_env import BaseMiniworldEnv
from gcp.infra.envs.miniworld_env.utils.multiroom2d_layout import define_layout, draw_layout_overview, \
                                                                                        default_texture_dir
from gcp.infra.envs.miniworld_env.utils.sampling_fcns import RoomSampler2d
import numbers


def fcn_apply(fcn, arg):
    return lambda: fcn(arg())


class Multiroom3dEnv(BaseMiniworldEnv):
    def __init__(self, hp, reset_state=None, no_env=False, crop_window=None):
        self._hp = self._default_hparams()
        for name, value in hp.items():
            print('setting param {} to value {}'.format(name, value))
            self._hp.set_hparam(name, value)
        super().__init__(self._hp)

        self._texture_dir = default_texture_dir()
        self._rooms_per_side = int(np.sqrt(self._hp.n_rooms))
        self._layout = define_layout(self._rooms_per_side, self._texture_dir)
        self._topdown_render_scale = 256       # table_size * scale = px size of output render img
        self._static_img_topdown = draw_layout_overview(self._rooms_per_side,
                                                        self._topdown_render_scale,
                                                        texture_dir=self._texture_dir)
        self._crop_window = crop_window
        if crop_window is not None:
            # top-down rendering will get cropped -> pad static background
            padded_bg = np.zeros((self._static_img_topdown.shape[0] + 2 * crop_window,
                                  self._static_img_topdown.shape[1] + 2 * crop_window, 3), dtype=self._static_img_topdown.dtype)
            padded_bg[crop_window:-crop_window, crop_window:-crop_window] = self._static_img_topdown
            self._static_img_topdown = padded_bg

        self._adim, self._sdim = 2, 3
        if not no_env:
            import gym_miniworld  # KEEP, important for registering environment
            self.env = gym.make("MiniWorld-Multiroom3d-v0",
                            obs_height=self._hp.obs_height,
                            obs_width=self._hp.obs_width,
                            rooms_per_side=self._rooms_per_side,
                            doors=self._layout.doors,
                            heading_smoothing=self._hp.heading_smoothing,
                            layout_params=AttrDict(room_size=self._layout.room_size,
                                                   door_size=self._layout.door_size,
                                                   textures=self._layout.textures))

        # Define the sample_*_state method by looking up the function with the corresponding name
        self.state_sampler = RoomSampler2d(self._rooms_per_side)
        self.current_pos = None
        self.goal_pos = None
        self.prm_policy = None  # used to compute shortest distance between pos and goal

    def _default_hparams(self):
        default_dict = {
            'obs_height': 300,
            'obs_width': 400,
            'goal_pos': None,
            'init_pos': None,
            'n_rooms': 9,
            'heading_smoothing': 0.2,   # how much of new angle is taken into average
        }
        parent_params = super()._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def reset(self, reset_state):
        super().reset()

        if reset_state is None:
            start_pos = self.env.mj2mw(self.state_sampler.sample(self._hp.init_pos))
            start_angle = 2 * np.pi * np.random.rand()
            goal_pos = self.env.mj2mw(self.state_sampler.sample(self._hp.goal_pos))
        else:
            start_pos = reset_state[:2]
            start_angle = reset_state[2]
            goal_pos = reset_state[-2:]

        reset_state = AttrDict(start_pos=start_pos,
                               start_angle=start_angle,
                               goal=goal_pos)

        img_obs = self.env.reset(reset_state)
        self.goal_pos = goal_pos
        qpos_full = np.concatenate((start_pos, np.array([start_angle])))

        obs = AttrDict(images=np.expand_dims(img_obs, axis=0),      # add camera dimension
                       qpos_full=qpos_full,
                       goal=goal_pos,
                       env_done=False,
                       state=np.concatenate((qpos_full, goal_pos)),
                       topdown_image=self.render_pos_top_down(qpos_full, self.goal_pos)
                        )
        self._post_step(start_pos)
        self._initial_shortest_dist = self.comp_shortest_dist(start_pos, goal_pos)
        return obs, reset_state

    def get_reset_from_obs(self, obs_dict):
        return obs_dict['state'][0]

    def get_goal_from_obs(self, obs_dict):
        self.goal = obs_dict['goal'][-1]
        return self.goal

    def step(self, action):
        img_obs, reward, done, agent_pos = self.env.step(action)
        obs = AttrDict(images=np.expand_dims(img_obs, axis=0),  # add camera dimension
                       qpos_full=agent_pos,
                       goal=self.goal_pos,
                       env_done=done,
                       state=np.concatenate((agent_pos, self.goal_pos)),
                       topdown_image=self.render_pos_top_down(agent_pos, self.goal_pos)
                       )
        self._post_step(agent_pos)
        return obs

    def _post_step(self, agent_pos):
        self.current_pos = agent_pos
        self.add_goal_dist(self.comp_shortest_dist(self.current_pos[:2], self.goal_pos))
        self._full_traj.append(agent_pos)

    def eval(self):
        self._final_shortest_dist = self.comp_shortest_dist(self.current_pos[:2], self.goal_pos)
        return super().eval()

    def comp_shortest_dist(self, p1, p2):
        """Uses PRM to get the shortest distance between two points within the maze."""
        if self.prm_policy is None:
            from gcp.infra.policy.prm_policy.prm_policy import PrmPolicy
            self.prm_policy = PrmPolicy(None, AttrDict(n_samples_per_room=200), None, None, **self.env_policy_params())
        dist, _ = self.prm_policy.compute_shortest_path(p1, p2)
        return dist

    def env_policy_params(self):
        def transform_plan(state_plan, action_plan):
            state_plan = self.env.mj2mw(state_plan)
            action_plan = state_plan[:, 1:] - state_plan[:, :-1]
            return state_plan, action_plan

        conversion_fcns = AttrDict(transform_plan=transform_plan,
                                   env2prm=self.env.mw2mj,
                                   prm2env=self.env.mj2mw)
        return {'conversion_fcns': conversion_fcns, 'n_rooms': self._hp.n_rooms}

    def render_top_down(self, traj, background=None, goal=None, line_thickness=4, color=(1.0, 0, 0), mark_pts=False):
        """Renders a state trajectory in a top-down view."""
        if isinstance(color[0], numbers.Number):
            color = [color] * (traj.shape[0] - 1)
         
        img = self._static_img_topdown.copy() if background is None else background.copy()
        traj = traj.copy()  # very important!!!
        if goal is not None:
            goal = goal.copy()
            if traj.shape[1] == 5 or traj.shape[1] == 2: goal = goal[:2]; goal[1] *= -1
            if traj.max() > 1.0 or traj.min() < -1.0: goal = goal / 27.0
            goal = goal + 0.5 * self._layout.table_size
        if traj.shape[1] == 5 or traj.shape[1] == 2: traj = traj[:, :2]; traj[:, 1] *= -1
        if traj.max() > 1.0 or traj.min() < -1.0: traj = traj / 27.0    # scale from miniworld env to [-1...1]
        traj = traj + 0.5 * self._layout.table_size
        for i in range(traj.shape[0] - 1):
            cv2.line(img, (int(traj[i, 0] * self._topdown_render_scale),
                           img.shape[0] - int(traj[i, 1] * self._topdown_render_scale)),
                          (int(traj[i+1, 0] * self._topdown_render_scale),
                           img.shape[0] - int(traj[i+1, 1] * self._topdown_render_scale)),
                           color[i], line_thickness)
            if mark_pts and i > 0 and i < (traj.shape[0] - 2):
                cv2.line(img, (int(traj[i, 0] * self._topdown_render_scale),
                               img.shape[0] - int(traj[i, 1] * self._topdown_render_scale)),
                              (int(traj[i, 0] * self._topdown_render_scale),
                               img.shape[0] - int(traj[i, 1] * self._topdown_render_scale)),
                         (1.0, 0, 0), int(3*line_thickness))

        # print start+end position
        img = self.render_pos_top_down(traj[0], traj[-1], background=img, mirror_scale=False)
        if goal is not None:
            img = self.render_pos_top_down(traj[0], goal, background=img, mirror_scale=False, large_goal=True)
        return img

    def render_pos_top_down(self,
                            current_pose,
                            goal_pos,
                            background=None,
                            mirror_scale=True,
                            large_goal=False):
        """Renders a state trajectory in a top-down view."""
        img = self._static_img_topdown.copy() if background is None else background.copy()

        def convert_sim2topdown(pos, img_shape):
            pos = pos.copy()  # very important !!!!!
            if mirror_scale:
                pos[1] *= -1
                pos = pos / 27.0    # scale from miniworld env to [-1...1]
                pos = pos + 0.5 * self._layout.table_size
            return (int(pos[0] * self._topdown_render_scale), img_shape[0] - int(pos[1] * self._topdown_render_scale))

        curr_pos = convert_sim2topdown(current_pose, img.shape)
        goal_pos = convert_sim2topdown(goal_pos, img.shape)

        if self._crop_window is not None:
            # we need to correct for the too large size of img.shape above, therefore -2*self._crop_window
            curr_pos = (curr_pos[0] + self._crop_window, curr_pos[1] + self._crop_window - 2*self._crop_window)
            goal_pos = (goal_pos[0] + self._crop_window, goal_pos[1] + self._crop_window - 2*self._crop_window)

        cv2.line(img, curr_pos, curr_pos, (0.0, 0, 1.0), 10)
        cv2.line(img, goal_pos, goal_pos, (0.0, 1.0, 0), 10 if not large_goal else 20)

        if self._crop_window is not None:
            # catch rounding errors
            curr_pos = (max(self._crop_window, curr_pos[0]), max(self._crop_window, curr_pos[1]))
            lower, upper = np.asarray(curr_pos) - self._crop_window, np.asarray(curr_pos) + self._crop_window
            img = img[lower[1]:upper[1], lower[0]:upper[0]]

        return img

    @property
    def adim(self):
        return self._adim

    @property
    def sdim(self):
        return self._sdim


class TopdownMultiroom3dEnv(Multiroom3dEnv):
    """Image observations are rendered topdown in a window around the agent."""
    def __init__(self, hp, reset_state=None, no_env=False, crop_window=None):
        assert "crop_window" in hp      # need to specify the crop window for topdown rendering
        temp_hp = copy.deepcopy(hp)
        crop_window = temp_hp.pop("crop_window")
        super().__init__(temp_hp, reset_state, no_env, crop_window=crop_window)

    def reset(self, reset_state):
        obs, reset_state = super().reset(reset_state)
        obs.images = np.asarray(255*obs.topdown_image.copy(), dtype=np.uint8)[None]
        return obs, reset_state

    def step(self, action):
        obs = super().step(action)
        obs.images = np.asarray(255*obs.topdown_image.copy(), dtype=np.uint8)[None]
        return obs

