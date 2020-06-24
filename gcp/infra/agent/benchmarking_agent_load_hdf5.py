from .general_agent import GeneralAgent
from gcp.infra.utils.im_utils import resize_store
import pickle as pkl
import numpy as np
import cv2
import PIL
import pdb
from PIL import Image
import os
import shutil
import glob


from robosuite.utils.mjcf_utils import postprocess_model_xml
from gcp.infra.agent.benchmarking_agent import BenchmarkAgent


class BenchmarkAgentLoadHDF5(BenchmarkAgent):
    def __init__(self, hyperparams, start_goal_list=None):
        super.__init__(hyperparams, start_goal_list)

    def _load_raw_data(self, itr):
        """
        doing the reverse of save_raw_data
        :param itr:
        :return:
        """
        traj = self._start_goal_confs + 'itr.hdf5'
        self._demo_images = demo_images.astype(np.float32)/255.
        self._goal_image = self._demo_images[-1]

        with open('{}/obs_dict.pkl'.format(traj), 'rb') as file:
            obs_dict.update(pkl.load(file))


        self._goal = self.env.get_goal_from_obs(obs_dict)
        reset_state = self.get_reset_state(obs_dict)

        if os.path.exists(traj  + '/robosuite.xml'):
            with open(traj  + '/robosuite.xml', "r") as model_f:
                model_xml = model_f.read()

            xml = postprocess_model_xml(model_xml)
            reset_state['robosuite_xml'] = xml

        return reset_state

    def get_reset_state(self, obs_dict):
        return self.env.get_reset_from_obs(obs_dict)


