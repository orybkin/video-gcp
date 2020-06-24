import os
import shutil
import pickle as pkl
import cv2

import copy

class RawSaver():
    def __init__(self, save_dir, ngroup=1000):
        self.save_dir = save_dir
        self.ngroup = ngroup

    def save_traj(self, itr, agent_data=None, obs_dict=None, policy_outputs=None):

        igrp = itr // self.ngroup
        group_folder = self.save_dir + '/raw/traj_group{}'.format(igrp)
        if not os.path.exists(group_folder):
            os.makedirs(group_folder)

        traj_folder = group_folder + '/traj{}'.format(itr)
        if os.path.exists(traj_folder):
            print('trajectory folder {} already exists, deleting the folder'.format(traj_folder))
            shutil.rmtree(traj_folder)

        os.makedirs(traj_folder)
        print('writing: ', traj_folder)

        if 'robosuite_xml' in obs_dict:
            save_robosuite_xml(traj_folder + '/robosuite.xml', obs_dict['robosuite_xml'][-1])

        if 'images' in obs_dict:
            images = obs_dict['images'].copy()
            T, n_cams = images.shape[:2]
            for i in range(n_cams):
                os.mkdir(traj_folder + '/images{}'.format(i))
            for t in range(T):
                for i in range(n_cams):
                    cv2.imwrite('{}/images{}/im_{}.png'.format(traj_folder, i, t), images[t, i, :, :, ::-1])

        if agent_data is not None:
            with open('{}/agent_data.pkl'.format(traj_folder), 'wb') as file:
                pkl.dump(agent_data, file)
        if obs_dict is not None:
            with open('{}/obs_dict.pkl'.format(traj_folder), 'wb') as file:
                obs_dict_cpy = copy.deepcopy(obs_dict)
                if 'topdown_image' in obs_dict_cpy:
                    obs_dict_cpy.pop('topdown_image') # don't save topdown image, takes too much memory!
                pkl.dump(obs_dict_cpy, file)
        if policy_outputs is not None:
            with open('{}/policy_out.pkl'.format(traj_folder), 'wb') as file:
                pkl.dump(policy_outputs, file)




import os
import xml.dom.minidom
import xml.etree.ElementTree as ET
import io

def save_robosuite_xml(fname, xml_str, pretty=False):
    with open(fname, "w") as f:
        if pretty:
            # TODO: get a better pretty print library
            parsed_xml = xml.dom.minidom.parseString(xml_str)
            xml_str = parsed_xml.toprettyxml(newl="")
        f.write(xml_str)