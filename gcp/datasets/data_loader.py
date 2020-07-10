import glob
import imp
import os
import random

import h5py
import numpy as np
import torch.utils.data as data
from torchvision.transforms import Resize

from blox import AttrDict
from blox.basic_types import map_dict
from blox.torch.training import RepeatedDataLoader
from blox.vis import resize_video
from gcp.prediction import global_params


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed + 2)


class BaseVideoDataset(data.Dataset):
    def __init__(self, data_dir, mpar, data_conf, phase, shuffle=True, dataset_size=-1):
        """
        :param data_dir: path to data directory
        :param mpar: model parameters used to determine output resolution etc
        :param data_conf: dataset config
        :param phase: string indicating whether 'train'/'val'/'test'
        :param shuffle: whether to shuffle within batch, set to False for computing metrics
        :param dataset_size: (optional) if not full dataset should be used, specifies number of used sequences
        """

        self.phase = phase
        self.data_dir = data_dir
        self.spec = data_conf.dataset_spec
        self.data_conf = data_conf
        self.dataset_size = dataset_size
        
        self.shuffle = shuffle and phase == 'train'
        self.use_states = not mpar.use_convs
        self.img_sz = mpar.img_sz
        self.device = mpar.device
        
        if shuffle:
            self.n_worker = 4
        else:
            self.n_worker = 1
        if global_params.debug:
            self.n_worker = 0

        self.filenames = None
        self.states_mean = None
        self.states_std = None

    def process_data_dict(self, data_dict):
        if 'images' in data_dict:
            data_dict.traj_seq_images = data_dict.pop('images')
        if 'states' in data_dict:
            data_dict.traj_seq_states = data_dict.pop('states')
            if self.states_mean is not None:
                data_dict.traj_seq_states = self.standardize(data_dict.traj_seq_states)
                data_dict.traj_seq_states_mean = self.states_mean
                data_dict.traj_seq_states_std = self.states_std

        if 'traj_seq_images' in data_dict and len(data_dict.traj_seq_images.shape) > 1:    # some datasets don't have images
            data_dict.traj_seq_images = self.preprocess_images(data_dict.traj_seq_images)
        data_dict.traj_seq = data_dict.traj_seq_states if self.use_states else data_dict.traj_seq_images

        if 'start_ind' not in data_dict:
            data_dict.start_ind = 0
        if 'end_ind' not in data_dict:
            data_dict.end_ind = self.spec['max_seq_len'] - 1
        if 'pad_mask' not in data_dict:
            data_dict.pad_mask = np.ones(self.spec['max_seq_len'], dtype=np.float32)
            
        data_dict.I_0 = data_dict.traj_seq[0]
        data_dict.I_g = data_dict.traj_seq[data_dict.end_ind]
        if 'traj_seq_images' in data_dict:
            data_dict.I_0_image = data_dict.traj_seq_images[0]
            data_dict.I_g_image = data_dict.traj_seq_images[data_dict.end_ind]

    def get_data_loader(self, batch_size, n_repeat):
        print('len {} dataset {}'.format(self.phase, len(self)))
        assert self.device in ['cuda', 'cpu']  # Otherwise the logic below is wrong
        return RepeatedDataLoader(self, batch_size=batch_size, shuffle=self.shuffle, num_workers=self.n_worker,
                                  drop_last=True, n_repeat=n_repeat, pin_memory=self.device == 'cuda',
                                  worker_init_fn=lambda x: np.random.seed(np.random.randint(65536) + x))

    def preprocess_images(self, images):
        return images

    @staticmethod
    def visualize(*args, **kwargs):
        pass

    def standardize(self, states):
        return (states - self.states_mean)/(1e-6 + self.states_std)

    @staticmethod
    def get_dataset_spec(data_dir):
        return imp.load_source('dataset_spec', os.path.join(data_dir, 'dataset_spec.py')).dataset_spec


class VarLenVideoDataset(BaseVideoDataset):
    """Variable length video dataset"""
    def __init__(self, data_dir, mpar, data_conf, phase, shuffle=True, dataset_size=-1):
        """
        :param data_dir: path to data directory
        :param mpar: model parameters used to determine output resolution etc
        :param data_conf: dataset config
        :param phase: string indicating whether 'train'/'val'/'test'
        :param shuffle: whether to shuffle within batch, set to False for computing metrics
        :param dataset_size: (optional) if not full dataset should be used, specifies number of used sequences
        """
        super().__init__(data_dir, mpar, data_conf, phase, shuffle, dataset_size)

        print('loading files from', self.data_dir)
        self.filenames = self._get_filenames()
        self.traj_per_file = self.get_traj_per_file(self.filenames[0])

        self.randomize_length = mpar.randomize_length
        self.randomize_start = mpar.randomize_start
        self.transform = Resize([mpar.img_sz, mpar.img_sz])
        self.flatten_im = False

        if 'states_mean' in self.spec:
            self.states_mean = self.spec['states_mean']
            self.states_std = self.spec['states_std']

    def _get_filenames(self):
        raise NotImplementedError("Needs to be implemented in sub-class!")

    def get_traj_per_file(self, path):
        with h5py.File(path, 'r') as F:
            return F['traj_per_file'].value

    def __getitem__(self, index):
        file_index = index // self.traj_per_file
        path = self.filenames[file_index]

        try:
            with h5py.File(path, 'r') as F:
                ex_index = index % self.traj_per_file   # get the index
                key = 'traj{}'.format(ex_index)

                # Fetch data into a dict
                if key + '/images' in F.keys():
                    data_dict = AttrDict(images=(F[key + '/images'].value))
                else:
                    data_dict = AttrDict()
                for name in F[key].keys():
                    if name in ['states', 'actions', 'pad_mask']:
                        data_dict[name] = F[key + '/' + name].value.astype(np.float32)

                # Make length consistent
                end_ind = np.argmax(data_dict.pad_mask * np.arange(data_dict.pad_mask.shape[0], dtype=np.float32), 0)
                start_ind = np.random.randint(0, end_ind - 1) if self.randomize_start else 0
                start_ind, end_ind, data_dict = self.sample_max_len_video(data_dict, start_ind, end_ind)

                # Randomize length
                if self.randomize_length:
                    end_ind = self._randomize_length(start_ind, end_ind, data_dict)

                # Collect data into the format the model expects
                data_dict.end_ind = end_ind
                data_dict.start_ind = start_ind

                self.process_data_dict(data_dict)
        except:
            raise ValueError("Problem when loading file from {}".format(path))

        return data_dict
    
    def sample_max_len_video(self, data_dict, start_ind, end_ind):
        """ This function processes data tensors so as to have length equal to max_seq_len
        by sampling / padding if necessary """
        extra_length = (end_ind - start_ind + 1) - self.spec['max_seq_len']
        if self.phase == 'train':
            offset = max(0, int(np.random.rand() * (extra_length + 1))) + start_ind
        else:
            offset = 0 
        
        data_dict = map_dict(lambda tensor: self._maybe_pad(tensor, offset, self.spec['max_seq_len']), data_dict)
        if 'actions' in data_dict:
            data_dict.actions = data_dict.actions[:-1]
        end_ind = min(end_ind - offset, self.spec['max_seq_len'] - 1)

        return 0, end_ind, data_dict        # start index gets 0 by design
    
    def _randomize_length(self, start_ind, end_ind, data_dict):
        """ This function samples part of the input tensors so that the length of the result
        is uniform between 1 and max """
        
        length = 3 + int(np.random.rand() * (end_ind - 2))  # The length of the seq is from 2 to total length
        chop_length = int(np.random.rand() * (end_ind + 1 - length))  # from 0 to the reminder
        end_ind = length - 1
        pad_mask = np.logical_and((np.arange(self.spec['max_seq_len']) <= end_ind),
                                  (np.arange(self.spec['max_seq_len']) >= start_ind)).astype(np.float32)
    
        # Chop off the beginning of the arrays
        def pad(array):
            array = np.concatenate([array[chop_length:], np.repeat(array[-1:], chop_length, 0)], 0)
            array[end_ind + 1:] = 0
            return array
    
        for key in filter(lambda key: key != 'pad_mask', data_dict):
            data_dict[key] = pad(data_dict[key])
        data_dict.pad_mask = pad_mask
        
        return end_ind

    def preprocess_images(self, images):
        # Resize video
        if len(images.shape) == 5:
            images = images[:, 0]  # Number of cameras, used in RL environments
        assert images.dtype == np.uint8, 'image need to be uint8!'
        images = resize_video(images, (self.img_sz, self.img_sz))
        images = np.transpose(images, [0, 3, 1, 2])  # convert to channel-first
        images = images.astype(np.float32) / 255 * 2 - 1
        assert images.dtype == np.float32, 'image need to be float32!'
        if self.flatten_im:
            images = np.reshape(images, [images.shape[0], -1])
        return images

    def _split_with_percentage(self, frac, filenames):
        assert sum(frac.values()) <= 1.0    # fractions cannot sum up to more than 1
        assert self.phase in frac
        if self.phase == 'train':
            start, end = 0, frac['train']
        elif self.phase == 'val':
            start, end = frac['train'], frac['train'] + frac['val']
        else:
            start, end = frac['train'] + frac['val'], frac['train'] + frac['val'] + frac['test']
        start, end = int(len(filenames) * start), int(len(filenames) * end)
        return filenames[start:end]

    @staticmethod
    def _maybe_pad(val, offset, target_length):
        """Pads / crops sequence to desired length."""
        val = val[offset:]
        len = val.shape[0]
        if len > target_length:
            return val[:target_length]
        elif len < target_length:
            return np.concatenate((val, np.zeros([int(target_length - len)] + list(val.shape[1:]), dtype=val.dtype)))
        else:
            return val

    @staticmethod
    def _shuffle_with_seed(arr, seed=2):
        rng = random.Random()
        rng.seed(seed)
        rng.shuffle(arr)
        return arr

    def __len__(self):
        if self.dataset_size != -1:
            return self.dataset_size

        return len(self.filenames) * self.traj_per_file


class FolderSplitVarLenVideoDataset(VarLenVideoDataset):
    """Splits in train/val/test using given folder structure."""

    def _get_filenames(self):
        filenames = sorted(glob.glob(os.path.join(self.data_dir, 'hdf5', self.phase + '/*')))
        if not filenames:
            raise RuntimeError('No filenames found in {}'.format(self.data_dir))
        filenames = self._shuffle_with_seed(filenames)
        return filenames


class GlobalSplitVarLenVideoDataset(VarLenVideoDataset):
    """Splits in train/val/test using global percentages."""

    def _get_filenames(self):
        filenames = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".h5") and not file == 'dataset_info.h5':
                    filenames.append(os.path.join(root, file))

        if not filenames:
            raise RuntimeError('No filenames found in {}'.format(self.data_dir))
        filenames = self._shuffle_with_seed(filenames)
        filenames = self._split_with_percentage(self.spec.split, filenames)
        return filenames


class MazeGlobalSplitVarLenVideoDataset(GlobalSplitVarLenVideoDataset):
    def process_data_dict(self, data_dict):
        if 'states' in data_dict:
            data_dict['states'] = data_dict['states'][..., :2]  # only use x,y position states
        return super().process_data_dict(data_dict)


class MazeTopRenderedGlobalSplitVarLenVideoDataset(MazeGlobalSplitVarLenVideoDataset):
    def __init__(self, data_dir, mpar, data_conf, phase, shuffle=True, dataset_size=-1):
        from gcp.planning.infra.envs.miniworld_env.multiroom3d.multiroom3d_env import Multiroom3dEnv
        super().__init__(data_dir, mpar, data_conf, phase, shuffle, dataset_size)
        assert 'n_rooms' in data_conf       # need to add this in config file!
        self._crop_window_px = data_conf.crop_window
        self._render_env = Multiroom3dEnv({'n_rooms': data_conf['n_rooms']}, no_env=True,
                                          crop_window=self._crop_window_px)

    def process_data_dict(self, data_dict):
        # replace images with topdown rendered images -> first render, then resize to scale
        if "images" in data_dict:
            assert "states" in data_dict and "end_ind" in data_dict
            rendered_imgs = np.zeros((data_dict.images.shape[0], 1, self._crop_window_px*2, self._crop_window_px*2, 3),
                                     dtype=data_dict.images.dtype)
            for t in range(data_dict.end_ind + 1):
                raw_img = self._render_env.render_pos_top_down(data_dict.states[t, :2],
                                                               data_dict.states[data_dict.end_ind, :2],)
                rendered_imgs[t, 0] = np.asarray(raw_img * 255, dtype=rendered_imgs.dtype)
            data_dict.images = rendered_imgs
        return super().process_data_dict(data_dict)
