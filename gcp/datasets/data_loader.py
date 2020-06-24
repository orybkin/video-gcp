import glob
import imp
import os
import random
import copy

import h5py
import numpy as np
import torch.utils.data as data
from blox.torch.training import RepeatedDataLoader
from blox import AttrDict
from blox.basic_types import map_dict, maybe_retrieve
from blox.vis import resize_video
from gcp.rec_planner_utils import global_params
from torchvision.transforms import Resize
from gcp.datasets.data_utils import TargetLengthSubsampler


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed + 2)


class BaseVideoDataset(data.Dataset):
    def __init__(self, data_dir, mpar, data_conf, phase, shuffle=True, dataset_size=-1):
        """

        :param data_dir:
        :param mpar:
        :param data_conf:
        :param phase:
        :param shuffle: whether to shuffle within batch, set to False for computing metrics
        :param dataset_size:
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
            data_dict.demo_seq_images = data_dict.pop('images')
        if 'states' in data_dict:
            data_dict.demo_seq_states = data_dict.pop('states')
            if self.states_mean is not None:
                data_dict.demo_seq_states = self.standardize(data_dict.demo_seq_states)
                data_dict.demo_seq_states_mean = self.states_mean
                data_dict.demo_seq_states_std = self.states_std

        if 'demo_seq_images' in data_dict and len(data_dict.demo_seq_images.shape) > 1:    # some datasets don't have images
            data_dict.demo_seq_images = self.preprocess_images(data_dict.demo_seq_images)
        data_dict.demo_seq = data_dict.demo_seq_states if self.use_states else data_dict.demo_seq_images

        if 'start_ind' not in data_dict:
            data_dict.start_ind = 0
        if 'end_ind' not in data_dict:
            data_dict.end_ind = self.spec['max_seq_len'] - 1
        if 'pad_mask' not in data_dict:
            data_dict.pad_mask = np.ones(self.spec['max_seq_len'], dtype=np.float32)
        
        data_dict.I_0 = data_dict.demo_seq[0]
        data_dict.I_g = data_dict.demo_seq[data_dict.end_ind]
        if 'demo_seq_images' in data_dict:
            data_dict.I_0_image = data_dict.demo_seq_images[0]
            data_dict.I_g_image = data_dict.demo_seq_images[data_dict.end_ind]

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

    def get_dataset_stats(self):
        n_standardize = 1000
        states_cat = []
        for i in range(n_standardize):
            path = self.filenames[i]
            with h5py.File(path, 'r') as F:

                ex_index = i % self.traj_per_file  # get the index
                key = 'traj{}'.format(ex_index)

                states = F[key + '/regression_state'].value.astype(np.float32)
                if 'regression_state' in F[key].keys():
                    states = F[key + '/regression_state'].value.astype(np.float32)

                states_cat.append(states)
        states_cat = np.concatenate(states_cat, 0)

        self.states_mean = np.mean(states_cat, 0)
        self.states_std = np.std(states_cat, 0)
        print('mean', self.states_mean)
        print('std', self.states_std)


class VarLenVideoDataset(BaseVideoDataset):
    """
    Variable length video dataset
    """

    def __init__(self, data_dir, mpar, data_conf, phase, shuffle=True, dataset_size=-1):
        """

        :param data_dir:
        :param mpar:
        :param data_conf:
        :param phase:
        :param shuffle: whether to shuffle within batch, set to False for computing metrics
        :param dataset_size:
        """
        super().__init__(data_dir, mpar, data_conf, phase, shuffle, dataset_size)

        print('loading files from', self.data_dir)
        self.filenames = self._get_filenames()
        self.traj_per_file = self.get_traj_per_file(self.filenames[0])

        self.randomize_length = mpar.randomize_length
        self.randomize_start = mpar.randomize_start
        self.transform = Resize([mpar.img_sz, mpar.img_sz])
        self.flatten_im = False
        self.filter_repeated_tail = False
        self.filter_repeated_tail = maybe_retrieve(self.spec, 'filter_repeated_tail')
        subsampler_class = maybe_retrieve(self.spec, 'subsampler')
        if subsampler_class is not None:
            subsample_args = maybe_retrieve(self.spec, 'subsample_args')
            assert subsample_args is not None  # need to specify subsampler args dict
            self.subsampler = subsampler_class(**subsample_args)
        else:
            self.subsampler = None
        self.repeat_tail = phase == "train" and ("rep_tail" in self.spec and self.spec.rep_tail)
        if self.repeat_tail:
            print("\n!!! Repeating Final Frame for all Sequences!!!\n")

        if 'states_mean' in self.spec:
            self.states_mean = self.spec['states_mean']
            self.states_std = self.spec['states_std']
        # self.get_dataset_stats()

    def _get_filenames(self):
        raise NotImplementedError("Needs to be implemented in sub-class!")

    def get_traj_per_file(self, path):
        with h5py.File(path, 'r') as F:
            return F['traj_per_file'].value

    def __getitem__(self, index):
        if 'one_datum' in self.data_conf and self.data_conf.one_datum:
            index = 1
        
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

                # remove spurious states at end of trajectory
                if self.filter_repeated_tail:
                    data_dict = self._filter_tail(data_dict)

                # maybe subsample seqs
                if self.subsampler is not None:
                    data_dict = self._subsample_data(data_dict)

                if 'robosuite_full_state' in F[key].keys():
                    data_dict.robosuite_full_state = F[key + '/robosuite_full_state'].value
                if 'regression_state' in F[key].keys():
                    data_dict.states = F[key + '/regression_state'].value.astype(np.float32)

                # Make length consistent
                end_ind = np.argmax(data_dict.pad_mask * np.arange(data_dict.pad_mask.shape[0], dtype=np.float32), 0)
                start_ind = np.random.randint(0, end_ind - 1) if self.randomize_start else 0
                start_ind, end_ind, data_dict = self.sample_max_len_video(data_dict, start_ind, end_ind)

                # Randomize length
                if self.randomize_length:
                    end_ind = self._randomize_length(start_ind, end_ind, data_dict)

                # repeat last frame until end of sequence
                data_dict.norep_end_ind = end_ind
                if self.repeat_tail:
                    data_dict, end_ind = self._repeat_tail(data_dict, end_ind)

                # Collect data into the format the model expects
                data_dict.end_ind = end_ind
                data_dict.start_ind = start_ind

                # for roboturk env rendering
                if 'robosuite_env_name' in F[key].keys():
                    data_dict.robosuite_env_name = F[key + '/robosuite_env_name'].value
                if 'robosuite_xml' in F[key].keys():
                    data_dict.robosuite_xml = F[key + '/robosuite_xml'].value

                self.process_data_dict(data_dict)
        except: # KeyError:
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

    def _subsample_data(self, data_dict):
        idxs = None
        for key in data_dict:
            data_dict[key], idxs = self.subsampler(data_dict[key], states=data_dict['states'], idxs=idxs)
        return data_dict
    
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
    def _repeat_tail(data_dict, end_ind):
        data_dict.images[end_ind:] = data_dict.images[end_ind][None]
        if 'states' in data_dict:
            data_dict.states[end_ind:] = data_dict.states[end_ind][None]
        data_dict.pad_mask = np.ones_like(data_dict.pad_mask)
        end_ind = data_dict.pad_mask.shape[0] - 1
        return data_dict, end_ind

    def __len__(self):
        if self.dataset_size != -1:
            return self.dataset_size
            
        return len(self.filenames) * self.traj_per_file

    @staticmethod
    def _filter_tail(data_dict):
        """Filters repeated states at the end of trajectory."""
        abs_action = np.linalg.norm(data_dict.actions, axis=-1)
        n_non_zero = np.trim_zeros(abs_action, 'b').size     # trim trailing zeros
        for key in data_dict:
            if key == 'actions': continue
            data_dict[key][n_non_zero+1:] = np.zeros_like(data_dict[key][n_non_zero+1:])
        return data_dict

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
    def get_dataset_spec(data_dir):
        return imp.load_source('dataset_spec', os.path.join(data_dir, 'dataset_spec.py')).dataset_spec

    @staticmethod
    def _shuffle_with_seed(arr, seed=1):
        rng = random.Random()
        # rng.seed(seed)
        rng.seed(2)
        rng.shuffle(arr)
        return arr


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


class LengthFilteredVarLenVideoDataset(VarLenVideoDataset):
    """Splits in train/val/test using length-dependent percentages."""

    def _get_filenames(self):
        with h5py.File(os.path.join(self.data_dir, "dataset_info.h5"), 'r') as F:
            name, length = F['name'][()], F['length'][()]

        filenames = []
        for length_range, frac in self.spec.split:
            files = list(set(name[length >= length_range[0]]) & set(name[length <= length_range[1]]))
            if not files: print("No sequences found in length range {}!".format(length_range)); continue
            files = self._shuffle_with_seed(files)
            files = self._split_with_percentage(frac, files)
            filenames.extend(files)

        if not filenames:
            raise RuntimeError('No filenames found in {}'.format(self.data_dir))
        filenames = self._shuffle_with_seed(filenames)
        filenames = [os.path.join(self.data_dir, f.decode()) for f in filenames]
        return filenames


class RoomFilteredVarLenVideoDataset(VarLenVideoDataset):
    """Splits in train/val/test using room-dependent percentages."""

    def _get_filenames(self):
        with h5py.File(os.path.join(self.data_dir, "dataset_info.h5"), 'r') as F:
            name, room_from, room_to = F['name'][()], F['from_room'][()], F['to_room'][()]

        filenames = []
        for room_combis, frac in self.spec.split:
            for room_combi in room_combis:
                files = list(set(name[room_from == room_combi[0]]) & set(name[room_to == room_combi[1]]))
                if not files: print("No sequences found in room combination {}!".format(room_combi)); continue
                files = self._shuffle_with_seed(files)
                files = self._split_with_percentage(frac, files)
                filenames.extend(files)

        if not filenames:
            raise RuntimeError('No filenames found in {}'.format(self.data_dir))
        filenames = self._shuffle_with_seed(filenames)
        filenames = [os.path.join(self.data_dir, f.decode()) for f in filenames]
        return filenames

    @staticmethod
    def filter_low_visitation_rooms(info_file_path, frac):
        """Sorts sequence by total visitation count of the visited rooms, filters most rare fraction."""
        with h5py.File(info_file_path, 'r') as F:
            room_from, room_to, rooms_visited = F['from_room'][()], F['to_room'][()], F['rooms_visited'][()]
        total_visits = rooms_visited.sum(axis=0)
        per_seq_score = (rooms_visited * total_visits[None]).sum(axis=1)
        n_seqs = int(per_seq_score.shape[0] * frac)
        filtered_idxs = np.argsort(per_seq_score)[n_seqs:]
        select_rf, select_rt = room_from[filtered_idxs], room_to[filtered_idxs]
        filtered_room_pairs = [(rf, rt) for rf, rt in zip(select_rf, select_rt)]
        filtered_room_pairs = [(e[0], e[1]) for e in np.unique(np.asarray(filtered_room_pairs), axis=0)]
        return filtered_room_pairs


class MazeGlobalSplitVarLenVideoDataset(GlobalSplitVarLenVideoDataset):
    def process_data_dict(self, data_dict):
        if 'states' in data_dict:
            data_dict['states'] = data_dict['states'][..., :2]  # only use x,y position states
        return super().process_data_dict(data_dict)


class MazeSubsampledGlobalSplitVarLenVideoDataset(MazeGlobalSplitVarLenVideoDataset):
    def __init__(self, data_dir, mpar, data_conf, phase, shuffle=True, dataset_size=-1):
        self._target_len = data_conf.dataset_spec['max_seq_len']
        self._final_subsampler = TargetLengthSubsampler(self._target_len)
        dc = copy.deepcopy(data_conf)
        dc.dataset_spec['max_seq_len'] = data_conf.data_loading_len
        super().__init__(data_dir, mpar, dc, phase, shuffle, dataset_size)

    def process_data_dict(self, data_dict):
        idxs = None
        seq_len = data_dict['end_ind'] + 1
        for key in data_dict:
            data_dict[key], idxs = self._final_subsampler(data_dict[key], seq_len, idxs=idxs)
        if 'end_ind' in data_dict and data_dict['end_ind'] >= self._target_len:
            data_dict['end_ind'] = self._target_len - 1
        return super().process_data_dict(data_dict)


class MazeStepFilteredGlobalSplitDataset(MazeGlobalSplitVarLenVideoDataset):
    """Filters trajectories with too high average step length."""
    def _get_filenames(self):
        filenames = super()._get_filenames()
        with h5py.File(os.path.join(self.data_dir, "dataset_info.h5"), 'r') as F:
            name, step_length = F['name'][()], F['avg_step_size'][()]
        filter_files = list(set(name[step_length <= 2]))
        filter_files = [os.path.join(self.data_dir, f.decode()) for f in filter_files]
        filenames = [f for f in filenames if f in filter_files]
        return filenames


class MazeTopRenderedGlobalSplitVarLenVideoDataset(MazeGlobalSplitVarLenVideoDataset):
    def __init__(self, data_dir, mpar, data_conf, phase, shuffle=True, dataset_size=-1):
        from gcp.infra.envs.miniworld_env.multiroom3d.multiroom3d_env import Multiroom3dEnv
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


class MazeTopRenderedSubsampledGlobalSplitVarLenVideoDataset(MazeSubsampledGlobalSplitVarLenVideoDataset):
    def __init__(self, data_dir, mpar, data_conf, phase, shuffle=True, dataset_size=-1):
        from gcp.infra.envs.miniworld_env.multiroom3d.multiroom3d_env import Multiroom3dEnv
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


class MazeTopRenderedFullGlobalSplitVarLenVideoDataset(MazeGlobalSplitVarLenVideoDataset):
    """Renders maze location top-down showing full maze with circles for start/goal on black background."""
    RENDER_RES = 256

    def __init__(self, data_dir, mpar, data_conf, phase, shuffle=True, dataset_size=-1):
        from gcp.infra.envs.miniworld_env.multiroom3d.multiroom3d_env import Multiroom3dEnv
        super().__init__(data_dir, mpar, data_conf, phase, shuffle, dataset_size)
        assert 'n_rooms' in data_conf       # need to add this in config file!
        self._render_env = Multiroom3dEnv({'n_rooms': data_conf['n_rooms']}, no_env=True)

    def process_data_dict(self, data_dict):
        # replace images with topdown rendered images -> first render, then resize to scale
        if "images" in data_dict:
            assert "states" in data_dict and "end_ind" in data_dict
            rendered_imgs = np.zeros((data_dict.images.shape[0], 1, self.RENDER_RES, self.RENDER_RES, 3),
                                     dtype=data_dict.images.dtype)
            for t in range(data_dict.end_ind + 1):
                raw_img = self._render_env.render_pos_top_down(data_dict.states[t, :2],
                                                               data_dict.states[data_dict.end_ind, :2],
                                                               background=rendered_imgs[t, 0])
                rendered_imgs[t, 0] = np.asarray(raw_img * 255, dtype=rendered_imgs.dtype)
            data_dict.images = rendered_imgs
        return super().process_data_dict(data_dict)


class MazeTopRenderedFullSubsampledGlobalSplitVarLenVideoDataset(MazeSubsampledGlobalSplitVarLenVideoDataset):
    """Renders maze location top-down showing full maze with circles for start/goal on black background."""
    RENDER_RES = 256

    def __init__(self, data_dir, mpar, data_conf, phase, shuffle=True, dataset_size=-1):
        from gcp.infra.envs.miniworld_env.multiroom3d.multiroom3d_env import Multiroom3dEnv
        super().__init__(data_dir, mpar, data_conf, phase, shuffle, dataset_size)
        assert 'n_rooms' in data_conf       # need to add this in config file!
        self._render_env = Multiroom3dEnv({'n_rooms': data_conf['n_rooms']}, no_env=True)

    def process_data_dict(self, data_dict):
        # replace images with topdown rendered images -> first render, then resize to scale
        if "images" in data_dict:
            assert "states" in data_dict and "end_ind" in data_dict
            rendered_imgs = np.zeros((data_dict.images.shape[0], 1, self.RENDER_RES, self.RENDER_RES, 3),
                                     dtype=data_dict.images.dtype)
            for t in range(data_dict.end_ind + 1):
                raw_img = self._render_env.render_pos_top_down(data_dict.states[t, :2],
                                                               data_dict.states[data_dict.end_ind, :2],
                                                               background=rendered_imgs[t, 0])
                rendered_imgs[t, 0] = np.asarray(raw_img * 255, dtype=rendered_imgs.dtype)
            data_dict.images = rendered_imgs
        return super().process_data_dict(data_dict)



