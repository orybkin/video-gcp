import matplotlib
import numpy as np

matplotlib.use('agg')
from gcp.planning.infra.datasets.save_util.record_saver import HDF5SaverBase
from blox import AttrDict


def pad_traj_timesteps(traj, max_num_actions):
    """
    pad images and actions with zeros
    :param traj:
    :param max_num_actions:
    :return:
    """

    if 'images' in traj:
        im_shape = traj.images.shape
    ac_shape = traj.actions.shape

    if ac_shape[0] < max_num_actions:
        if 'images' in traj:
            zeros = np.zeros([max_num_actions - im_shape[0] + 1, im_shape[1], im_shape[2], im_shape[3], im_shape[4]], dtype=np.uint8)
            traj.images = np.concatenate([traj.images, zeros])

        if len(ac_shape) > 1:
            zeros = np.zeros([max_num_actions - ac_shape[0], ac_shape[1]])
        else:
            zeros = np.zeros([max_num_actions - ac_shape[0]])
        traj.actions = np.concatenate([traj.actions, zeros])

    if 'images' in traj:
        assert traj.images.shape[0] == max_num_actions + 1
    assert traj.actions.shape[0] == max_num_actions

    return traj


def get_pad_mask(action_len, max_num_actions):
    """
     create a 0/1 mask with 1 where there are images and 0 where there is padding
    :param action_len:  the number of actions in trajectory
    :param max_num_actions:  maximum number of actions allowed
    :return:
    """
    if action_len < max_num_actions:
        mask = np.concatenate([np.ones(action_len + 1), np.zeros(max_num_actions - action_len)])
    elif action_len == max_num_actions:
        mask = np.ones(max_num_actions + 1)
    else:
        raise ValueError

    assert mask.shape[0] == max_num_actions + 1

    return mask


class HDF5Saver(HDF5SaverBase):
    def __init__(self, save_dir, envparams, agentparams, traj_per_file,
                 offset=0, split=(0.90, 0.05, 0.05), split_train_val_test=True):

        self.do_not_save_images = agentparams.do_not_save_images
        if hasattr(agentparams, 'max_num_actions'):
            self.max_num_actions = envparams.max_num_actions
        else:
            self.max_num_actions = agentparams.T

        super().__init__(save_dir, traj_per_file, offset, split, split_train_val_test)

    def _save_manifests(self, agent_data, obs, policy_out):
        pass

    def make_traj(self, agent_data, obs, policy_out):
        traj = AttrDict()

        if not self.do_not_save_images:
            traj.images = obs['images']
        traj.states = obs['state']
        
        action_list = [action['actions'] for action in policy_out]
        traj.actions = np.stack(action_list, 0)
        
        traj.pad_mask = get_pad_mask(traj.actions.shape[0], self.max_num_actions)
        traj = pad_traj_timesteps(traj, self.max_num_actions)

        if 'robosuite_xml' in obs:
            traj.robosuite_xml = obs['robosuite_xml'][0]
        if 'robosuite_env_name' in obs:
            traj.robosuite_env_name = obs['robosuite_env_name'][0]
        if 'robosuite_full_state' in obs:
            traj.robosuite_full_state = obs['robosuite_full_state']

        # minimal state that contains all information to position entities in the env
        if 'regression_state' in obs:
            traj.regression_state = obs['regression_state']

        return traj

    def save_traj(self, itr, agent_data, obs, policy_out):
        traj = self.make_traj(agent_data, obs, policy_out)
        self._save_traj(traj)
