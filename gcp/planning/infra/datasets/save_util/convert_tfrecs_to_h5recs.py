# import ipdb
import glob

import numpy as np
import gcp.planning.infra.datasets.save_util.configs.TAP_3obj_push as config
import tensorflow as tf
from blox import AttrDict
from blox.basic_types import str2int
from gcp.planning.infra.agent.utils.hdf5_saver import HDF5SaverBase
from tqdm import tqdm

phase = 'train'

count = 0
H = config.precrop_frame_ht
W = config.precrop_frame_wd
C = 3


class TAPMaker(HDF5SaverBase):
    def __init__(self, save_dir, offset=0, split=(0.90, 0.05, 0.05)):
        super().__init__(save_dir, traj_per_file=1, offset=offset, split=split)
        
        self.filenames = sorted(glob.glob(config.tfrecord_dir + phase + '/*.tfrecord?'))
        self.max_seq_len = 80
    
    def get_traj(self, string_record):
        example = tf.train.Example()
        example.ParseFromString(string_record)
        images = []
        
        numbered_keys = filter(lambda x: str2int(x.split('/')[0]) is not None, example.features.feature.keys())
        image_keys = filter(lambda x: 'image_view0/encoded' in x, numbered_keys)
        indices = np.array(list([str2int(x.split('/')[0]) for x in image_keys]))
        length = np.max(indices) + 1
        
        for i in range(length):
            key = '{}/image_view0/encoded'.format(i)
            val = np.frombuffer(example.features.feature[key].bytes_list.value[0], dtype=np.uint8)
            val = val.reshape(H, W, C)
            images.append(val)
        
        pad_mask = np.ones(len(images))
        images = np.array(images)
        
        return AttrDict(images=images, pad_mask=pad_mask)
    
    def make_phase(self, filenames, phase):
        for fn in filenames:
            record_iterator = tf.python_io.tf_record_iterator(path=fn)
            for i, string_record in enumerate(tqdm(record_iterator)):
                traj = self.get_traj(string_record)
                self.save_hdf5([traj], phase)
                
    # def save_hdf5(self, traj_list, phase):
    #     traj = traj_list[0]
    #     with h5py.File(config.h5record_dir + 'hdf5/' + phase + '/traj_{0:06d}'.format(count) + '.h5', 'w') as F:
    #         F['traj0/pad_mask'] = traj.pad_mask
    #         F['traj0/images'] = traj.images
    #         F['traj_per_file'] = 1


if __name__ == '__main__':
    maker = TAPMaker(config.h5record_dir)
    
    maker.make_dataset()