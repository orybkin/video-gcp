from .general_agent import GeneralAgent
import pickle as pkl
import numpy as np
import cv2
import PIL
from PIL import Image
import os
import glob


class BenchmarkAgent(GeneralAgent):
    N_MAX_TRIALS = 1    # only allow one trial per trajectory when benchmarking
    def __init__(self, hyperparams, start_goal_list=None):
        self._start_goal_confs = hyperparams.get('start_goal_confs', None)
        self.ncam = hyperparams['env'][1].get('ncam', hyperparams['env'][0].default_ncam()) # check if experiment has ncam set, otherwise get env default

        GeneralAgent.__init__(self, hyperparams, start_goal_list)
        self._is_robot_bench = 'robot_name' in self._hp.env[1]
        if not self._is_robot_bench:
            self._hp.gen_xml = (True, 1)        # this was = 1 but that did not work?!

    def _setup_world(self, itr):
        old_ncam = self.ncam
        GeneralAgent._setup_world(self, itr)
        if self._start_goal_confs is not None:
            self._reset_state = self._load_raw_data(itr)
        assert old_ncam == self.ncam, """Environment has {} cameras but benchmark has {}. 
                                            Feed correct ncam in agent_params""".format(self.ncam, old_ncam)

    def _required_rollout_metadata(self, agent_data, traj_ok, t, i_itr):
        GeneralAgent._required_rollout_metadata(self, agent_data, traj_ok, t, i_itr)

        if self._start_goal_confs is not None:
            agent_data.update(self.env.eval())

    def _init(self):
        return GeneralAgent._init(self)

    def _load_raw_data(self, itr):
        """
        doing the reverse of save_raw_data
        :param itr:
        :return:
        """
        if 'robot_name' in self._hp.env[1]:   # robot experiments don't have a reset state
            return None

        if 'iex' in self._hp:
            itr = self._hp.iex

        ngroup = 1000
        igrp = itr // ngroup
        group_folder = '{}/traj_group{}'.format(self._start_goal_confs, igrp)
        traj_folder = group_folder + '/traj{}'.format(itr)

        print('reading from: ', traj_folder)
        num_files = len(glob.glob("{}/images0/*.png".format(traj_folder)))
        assert num_files > 0, " no files found!"

        obs_dict = {}
        demo_images = np.zeros([num_files, self.ncam, self._hp.image_height, self._hp.image_width, 3])
        for t in [0, num_files-1]: #range(num_files):
            for c in range(self.ncam):
                image_file = '{}/images{}/im_{}.png'.format(traj_folder, c, t)
                if not os.path.isfile(image_file):
                    raise ValueError("Can't find goal image: {}".format(image_file))
                img = cv2.imread(image_file)[..., ::-1]
                if img.shape[0] != self._hp.image_height or img.shape[1] != self._hp.image_width:
                    img = Image.fromarray(img)
                    img = img.resize((self._hp.image_height, self._hp.image_width), PIL.Image.BILINEAR)
                    img = np.asarray(img, dtype=np.uint8)
                demo_images[t, c] = img
        self._demo_images = demo_images.astype(np.float32)/255.

        self._goal_image = self._demo_images[-1]

        with open('{}/obs_dict.pkl'.format(traj_folder), 'rb') as file:
            obs_dict.update(pkl.load(file))

        self._goal = self.env.get_goal_from_obs(obs_dict)
        reset_state = self.get_reset_state(obs_dict)

        if os.path.exists(traj_folder + '/robosuite.xml'):
            with open(traj_folder + '/robosuite.xml', "r") as model_f:
                model_xml = model_f.read()

            from robosuite.utils.mjcf_utils import postprocess_model_xml
            xml = postprocess_model_xml(model_xml)
            reset_state['robosuite_xml'] = xml

        return reset_state

    def get_reset_state(self, obs_dict):
        return self.env.get_reset_from_obs(obs_dict)


