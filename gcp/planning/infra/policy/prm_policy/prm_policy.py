import queue

import numpy as np
from blox import AttrDict
from gcp.planning.infra.envs.miniworld_env.utils.multiroom2d_layout import define_layout
from gcp.planning.infra.envs.miniworld_env.utils.sampling_fcns import RoomSampler2d
from gcp.planning.infra.policy.policy import Policy
from gcp.planning.infra.policy.prm_policy.prm import PRM_planning
from scipy import interpolate


class PrmPolicy(Policy):
    """
    Cross Entropy Method Stochastic Optimizer
    """
    VAR_SAMPLING_RATES = [30, 300]

    def __init__(self, ag_params, policyparams, gpu_id, ngpu, conversion_fcns=None, n_rooms=None):
        super(PrmPolicy, self).__init__()

        self._hp = self._default_hparams()
        policyparams['n_rooms'] = n_rooms
        self.override_defaults(policyparams)

        self._rooms_per_side = int(np.sqrt(self._hp.n_rooms))
        self.layout = define_layout(self._rooms_per_side)
        self.state_sampler = RoomSampler2d(self._rooms_per_side, sample_wide=self.layout.non_symmetric)
        self.plan_params = AttrDict(n_knn=self._hp.n_knn,
                                    max_edge_len=self._hp.max_edge_len,
                                    cost_fcn=lambda d: d ** self._hp.cost_power)

        self.current_action = None
        self.state_plan = None
        self.action_plan = None
        self.convert = conversion_fcns      # AttrDict containing env2prm, transform_plan
        self._room_plan = None

    def reset(self):
        self.current_action = None
        self.state_plan = None
        self.action_plan = None
        self._room_plan = None

    def _default_hparams(self):
        default_dict = {
            'n_samples_per_room': 50,  # number of sample_points in first try, then gets increased
            'n_samples_per_door': 3,   # number of samples per door
            'n_knn': 10,  # number of edge from one sampled point
            'max_edge_len': 0.1,  # Maximum edge length (in layout units)
            'replan_eps': 0.05,     # distance btw planned and executed state that triggers replan, in % of table size
            'max_planning_retries': 2,   # maximum number of replans before inverting the last action
            'cost_power': 2,    # power on the distance for cost function
            'bottleneck_sampling': True,    # sample explicitly in bottlenecks to ease planning
            'use_var_sampling': False,  # if True, uses variable PRM sampling rates for different rooms
            'subsample_factor': 1.0,    # how much to subsample the plan in state space
            'max_traj_length': None,     # maximum length of planned trajectory
            'smooth_trajectory': False,     # if True, uses spline interpolation to smooth trajectory
            'sample_door_center': False,     # if True, samples door samples in center position of door
            'use_scripted_path': False,     # if True, uses scripted waypoints to construct path
            'straight_through_door': False,     # if True, crosses through door in a straight line
            'n_rooms': None,   # number of rooms in the layout
            'use_fallback_plan': True,  # if True executes fallback plan if planning fails
            'logger': None,     # dummy variable
        }

        parent_params = super()._default_hparams()
        parent_params.ncam = 1
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def act(self, t=None, i_tr=None, qpos_full=None, goal=None):
        self.i_tr = i_tr
        output = AttrDict()

        if self.action_plan is None or \
                self._check_deviate(qpos_full[t, :2],
                                    self.state_plan[:, min(self.current_action, self.state_plan.shape[1]-1)]):
            self._plan(qpos_full[t], goal[t], t)
            self.current_action = 0

        done = False
        if self.current_action < self.action_plan.shape[1]:
            output.actions = self.action_plan[:, self.current_action]
        else:   # if required number of steps > planned steps
            done = True
            output.actions = np.zeros(2)
        self.current_action = self.current_action + 1
        output.done = done
        return output

    def _sample_uniform(self):
        px, py = [], []
        for _ in range(self._hp.n_samples_per_room * self._hp.n_rooms):
            p = self.state_sampler.sample()
            px.append(p[0]); py.append(p[1])
        return px, py

    def _sample_per_room(self, room_path):
        px, py = [], []
        room_path = range(self._hp.n_rooms) if room_path is None else room_path
        for room in room_path:
            n_samples = int(np.random.choice(PrmPolicy.VAR_SAMPLING_RATES, 1)) if self._hp.use_var_sampling \
                            else self._hp.n_samples_per_room
            for _ in range(n_samples):
                p = self.state_sampler.sample(room)
                px.append(p[0]); py.append(p[1])
        return px, py

    def _sample_per_door(self, room_path=None):
        doors = self.layout if room_path is None else \
            [(min(room_path[i], room_path[i+1]), max(room_path[i], room_path[i+1])) for i in range(len(room_path) - 1)]
        if not doors: return [], []
        samples = np.asarray([[self.state_sampler.sample_door(d[0], d[1], self._hp.sample_door_center)
                               for _ in range(self._hp.n_samples_per_door)]
                              for d in doors]).transpose(2, 0, 1).reshape(2, -1)
        return samples[0], samples[1]

    def _sample_points(self, room_path=None):
        px, py = self._sample_per_room(room_path)
        if self._hp.bottleneck_sampling:
            dx, dy = self._sample_per_door(room_path)
            px.extend(dx); py.extend(dy)
        return [px, py]

    def _check_deviate(self, pos, target_pos):
        print(np.linalg.norm(pos - target_pos))
        return np.linalg.norm(pos - target_pos) > self._hp.replan_eps

    def _plan(self, agent_pos, goal_pos, t):
        ## UNCOMMENT for random exploration policcy
        #from gcp.infra.policy.cem.utils.sampler import PDDMSampler
        #sampler = PDDMSampler(clip_val=float("Inf"), n_steps=self._hp.max_traj_length - t, action_dim=2, initial_std=2)
        #self.action_plan = sampler.sample(1)[0].transpose(1, 0)  #(np.random.rand(2, self._hp.max_traj_length - t) - 0.5) * 2 * 2#* 6e-1
        #self.state_plan = agent_pos[:2][:, None].repeat(self.action_plan.shape[1], axis=1) + np.cumsum(self.action_plan, axis=1)
        #return self.action_plan, True

        if self.convert is not None:
            pos = self.convert.env2prm(agent_pos[:2])
            goal_pos = self.convert.env2prm(goal_pos)
        else:
            pos = agent_pos[:2]

        length, path = self.compute_shortest_path(pos, goal_pos, transform_pose=False)
        if self._hp.use_scripted_path:
            planned_x, planned_y = [p[0] for p in path], [p[1] for p in path]
            success = True
        else:
            sx, sy = pos[0], pos[1]
            gx, gy = goal_pos[0], goal_pos[1]
            ox, oy = self.layout.ox, self.layout.oy
            if self._room_plan is None:
                room_path = self.plan_room_seq(self.layout.coords2ridx(*pos),
                                               self.layout.coords2ridx(*goal_pos), self.layout.doors)
                print("Planned room sequence with {} rooms!".format(len(room_path)))
                self._room_plan = room_path
            else:
                room_path = self._room_plan
                print("Reused existing room plan!")

            for _ in range(self._hp.max_planning_retries):
                pts = self._sample_points(room_path)
                planned_x, planned_y, success = PRM_planning(sx, sy, gx, gy, ox, oy, self.layout.robot_size, self.plan_params,
                                                             self._hp.n_samples_per_room * self._hp.n_rooms, pts)
                if success: break     # when planning is successful

        if not success:
            if self._hp.use_fallback_plan:
                print("Did not find a plan in {} tries!".format(self._hp.max_planning_retries))
                self._fallback_plan()
            return None, False

        n_steps = min(int(length * 20), self._hp.max_traj_length - t) # min(int(self._hp.subsample_factor * len(planned_x)), self._hp.max_traj_length)
        try:
            if self._hp.max_traj_length is not None: n_steps = min(n_steps, self._hp.max_traj_length - t)
            tck, u = interpolate.splprep([planned_x, planned_y], s=0.0)
            x_i, y_i = interpolate.splev(np.linspace(0, 1, n_steps), tck)
            # x_i, y_i = planned_x, planned_y
            self.state_plan = np.stack((x_i, y_i))
        except TypeError:
            print("Could not interpolate!")     # this happens if duplicate values in plan
            self._fallback_plan()
            return None, False
            #self.state_plan = np.array([planned_x, planned_y])
        self.action_plan = self.state_plan[:, 1:] - self.state_plan[:, :-1]

        raw_plan = self.state_plan.copy()
        if self.convert is not None:
            self.state_plan, self.action_plan = self.convert.transform_plan(self.state_plan, self.action_plan)
        return raw_plan, True

    def _fallback_plan(self):
        if self.action_plan is not None:
            self.action_plan = -2 * self.action_plan[:, self.current_action-1:]    # TODO: come up with a better fallback solution!)
        else:
            self.action_plan = self.state_plan = 0.02 * np.random.rand(2, 1)

    def compute_shortest_path(self, p1, p2, transform_pose=True, straight_through_door=False):
        if self.convert is not None and transform_pose:
            p1, p2 = self.convert.env2prm(p1), self.convert.env2prm(p2)
        if (np.stack((p1,p2)) < -0.5).any() or (np.stack((p1,p2)) > 0.5).any():
            return 10., []      # coordinates invalid
        room_path = plan_room_seq(self.layout.coords2ridx(p1[0], p1[1]),
                                  self.layout.coords2ridx(p2[0], p2[1]), self.layout.doors)
        waypoints = [p1]
        for n in range(len(room_path)-1):
            # input rooms must be in ascending order
            if straight_through_door:
                waypoints.extend(self.state_sampler.get_door_path(room_path[n], room_path[n + 1]))
            else:
                waypoints.append(self.state_sampler.get_door_pos(min(room_path[n], room_path[n+1]),
                                                                 max(room_path[n], room_path[n+1])))
        waypoints.append(p2)
        waypoints = np.array(waypoints)
        length = np.linalg.norm(waypoints[1:] - waypoints[:-1], axis=1).sum()
        return length, waypoints

    def plan_room_seq(self, *args, **kwargs):
        if self.layout.multimodal:
            return plan_room_seq_multimodal(*args, **kwargs)
        else:
            return plan_room_seq(*args, **kwargs)

    def avg_step_length(self, px, py):
        return np.mean(np.sqrt((np.array(px[1:]) - np.array(px[:-1]))**2 + (np.array(py[1:]) - np.array(py[:-1]))**2))


def plan_room_seq(start, goal, doors):
    """Implements a breadth-first room search to find the sequence of rooms that reaches the goal."""
    frontier = queue.Queue()
    visited = []

    def expand(node):
        if node.room == goal: return node
        visited.append(node.room)
        neighbors = []
        for d in doors:
            if d[0] == node.room and d[1] not in visited:
                neighbors.append(d[1])
            elif d[1] == node.room and d[0] not in visited:
                neighbors.append(d[0])
        [frontier.put(AttrDict(room=neighbor, parent=node)) for neighbor in neighbors]
        return expand(frontier.get())

    linked_path = expand(AttrDict(room=start, parent=None))
    room_path = []

    def collect(node):
        room_path.append(node.room)
        if node.parent is None: return
        collect(node.parent)

    collect(linked_path)
    return room_path[::-1]


def plan_room_seq_multimodal(start, goal, doors):
    """Finds all paths between start and goal that visit each room at most once. Returns one of them at random."""
    frontier = queue.Queue()
    goal_nodes = []

    def collect_path(start_node):
        room_path = []
        def collect(node):
            room_path.append(node.room)
            if node.parent is None: return
            collect(node.parent)
        collect(start_node)
        return room_path

    def expand(node):
        if node.room == goal:
            goal_nodes.append(node)
        else:
            neighbors = []
            for d in doors:
                if d[0] == node.room and d[1] not in collect_path(node):
                    neighbors.append(d[1])
                elif d[1] == node.room and d[0] not in collect_path(node):
                    neighbors.append(d[0])
            [frontier.put(AttrDict(room=neighbor, parent=node)) for neighbor in neighbors]
        if frontier.empty(): return
        expand(frontier.get())

    # collect list of all possible paths, is sorted by length (short to long)
    expand(AttrDict(room=start, parent=None))

    # sample one of the possible paths at random
    return collect_path(np.random.choice(goal_nodes))[::-1]


if __name__ == "__main__":
    layout = define_layout(3)
    room_seq = plan_room_seq_multimodal(0, 8, layout.doors)
    print(room_seq)



