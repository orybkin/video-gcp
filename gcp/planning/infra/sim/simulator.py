import os
import os.path
import sys

from tensorflow.contrib.training import HParams

from gcp.planning.infra.agent.utils.hdf5_saver import HDF5Saver
from gcp.planning.infra.agent.utils.raw_saver import RawSaver
from gcp.prediction.utils.logger import HierarchyLogger

sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
import numpy as np


class Sim(object):
    """ Main class to run algorithms and experiments. """

    def __init__(self, config, gpu_id=0, ngpu=1):
        self._start_goal_list = config.pop('start_goal_list') if 'start_goal_list' in config else None
        self._hp = self._default_hparams()
        self.override_defaults(config)
        self._hp.agent['log_dir'] = self._hp.log_dir
        self._hp.n_rooms = self._hp.agent['env'][1]['n_rooms'] if 'n_rooms' in self._hp.agent['env'][1] else None
        self.agent = self._hp.agent['type'](self._hp.agent, self._start_goal_list)
        self.agentparams = self._hp.agent

        self._record_queue = self._hp.record_saver
        self._counter = self._hp.counter

        if self._hp.logging_conf is None:
            self.logger = HierarchyLogger(self._hp.log_dir + '/verbose', self._hp, self._hp.agent['T'])
            self._hp.logging_conf = {'logger':self.logger, 'global_step':-1, 'phase':'test'}

        self._hp.policy['logger'] = self.logger
        self.policy = self._hp.policy['type'](self.agent._hp, self._hp.policy, gpu_id, ngpu,
                                              **self.agent.env.env_policy_params())

        self.trajectory_list = []
        self.im_score_list = []
        try:
            os.remove(self._hp.agent['image_dir'])
        except:
            pass

        self.savers = []
        if 'hdf5' in self._hp.save_format:
            self.savers.append(HDF5Saver(self._hp.data_save_dir, self.agent.env._hp, self.agent._hp,
                                     traj_per_file=self._hp.traj_per_file, offset=self._hp.start_index,
                                         split_train_val_test=self._hp.split_train_val_test))
        if 'raw' in self._hp.save_format:
            self.savers.append(RawSaver(self._hp.data_save_dir))

        self.logging_conf = self._hp.logging_conf

    def override_defaults(self, config):
        """
        :param config:  override default valus with config dict
        :return:
        """
        for name, value in config.items():
            print('overriding param {} to value {}'.format(name, value))
            if value == getattr(self._hp, name):
                raise ValueError("attribute {} is identical to default value!!".format(name))
            if name in self._hp and self._hp.get(name) is None:   # don't do a type check for None default values
                setattr(self._hp, name, value)
            else: self._hp.set_hparam(name, value)

    def _default_hparams(self):
        default_dict = {
            'save_format': ['hdf5', 'raw'],
            'save_data': True,
            'agent': {},
            'policy': {},
            'start_index': -1,
            'end_index': -1,
            'ntraj': -1,
            'gpu_id': -1,
            'current_dir': '',
            'record_saver': None,
            'counter': None,
            'traj_per_file': 10,
            'data_save_dir': '',
            'log_dir': '',
            'result_dir': '',
            'split_train_val_test': True,
            'logging_conf': None,   # only needed for training loop
        }
        # add new params to parent params
        parent_params = HParams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def run(self):
        if self._counter is None:
            for i in range(self._hp.start_index, self._hp.end_index+1):
                self.take_sample(i)
        else:
            itr = self._counter.ret_increment()
            while itr < self._hp.ntraj:
                print('taking sample {} of {}'.format(itr, self._hp.ntraj))
                self.take_sample(itr)
                itr = self._counter.ret_increment()

    def take_sample(self, index):
        """
        :param index:  run a single trajectory with index
        :return:
        """
        self.policy.reset()
        agent_data, obs_dict, policy_out = self.agent.sample(self.policy, index)
        if self._hp.save_data:
            self.save_data(index, agent_data, obs_dict, policy_out)
        if self.logging_conf is not None and 'goal_image' in agent_data and 'images' in obs_dict:
            goal = agent_data['goal_image'][-1, 0] if len(agent_data['goal_image'].shape)>4 else agent_data['goal_image'][-1]
            if 'goal_pos' in obs_dict:
                goal_pos = obs_dict['goal'][-1, :] if isinstance(obs_dict['goal'], np.ndarray) else obs_dict['goal']
            else:
                goal_pos = None
            topdown_image = obs_dict['topdown_image'] if 'topdown_image' in obs_dict else None
            self.policy.log_outputs_stateful(**self.logging_conf, dump_dir=self._hp.log_dir,
                                             exec_seq=obs_dict['images'][:, 0], goal=goal, goal_pos=goal_pos,
                                             index=index, topdown_image=topdown_image, env=self.agent.env)     # [:, 0] for cam0
        return agent_data

    def save_data(self, itr, agent_data, obs_dict, policy_outputs):
        if self._record_queue is not None:  # if using a queue to save data
            self._record_queue.put((agent_data, obs_dict, policy_outputs))
        else:
            for saver in self.savers: # if directly saving data
                saver.save_traj(itr, agent_data, obs_dict, policy_outputs)
