import glob
import itertools
import os

import cv2
import numpy as np
from blox import AttrDict
from gcp.planning.infra.agent.utils.hdf5_saver import HDF5SaverBase
from tqdm import tqdm


def process_frame(frame_in, frame_output_size=(64, 64)):
    """Standardizes input frame width and height, and removes dummy channels.
    """
    
    # NB: OpenCV takes new size as (X, Y), not (Y, X)!!!
    frame_out = cv2.resize(
        frame_in.astype(np.float32),
        (frame_output_size[1], frame_output_size[0])).astype(
        frame_in.dtype)
    
    if frame_out.shape[-1] == 1:
        # Chop off redundant dimensions
        frame_out = frame_out[..., 0]
    elif frame_out.shape[-1] == 3:
        # Convert OpenCV's BGR to RGB
        frame_out = frame_out[..., ::-1]
    else:
        raise ValueError("Unexpected frame shape!")
    
    return frame_out

def read_video(video_path, n_downsample=8):
    """Reads a video from a file and returns as an array."""
    assert os.path.isfile(video_path)
    cap = cv2.VideoCapture(video_path)
    
    all_frames = []
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if i % n_downsample == 0:
                all_frames.append(process_frame(frame))
        else:
            cap.release()
        i += 1
    
    return np.asarray(all_frames)


class H36Maker(HDF5SaverBase):
    def __init__(self, data_dir, folders, save_dir, traj_per_file, offset=0, split=(0.90, 0.05, 0.05), n_downsample=8):
        super().__init__(save_dir, traj_per_file, offset, split)
        
        data_folders = list([data_dir + '/' + folder + '/Videos' for folder in folders])
        listlist_names = list([glob.glob(folder + '/*') for folder in data_folders])
        self.filenames = list(itertools.chain(*listlist_names))
        self.max_seq_len = 1500
        self.n_downsample = n_downsample
    
    def get_traj(self, name):
        vid = read_video(name, self.n_downsample)
        print(vid.shape)
        # pad_mask = np.concatenate([np.ones(vid.shape[0]), np.zeros(self.max_seq_len - vid.shape[0])])
        pad_mask = np.ones(vid.shape[0])
        
        return AttrDict(images=vid, pad_mask=pad_mask)
    
    def make_phase(self, filenames, phase):
        traj_list = []
        
        for name in tqdm(filenames):
            traj = self.get_traj(name)
            traj_list.append(traj)

            if len(traj_list) == self.traj_per_file:
                self.save_hdf5(traj_list, phase)
                traj_list = []


if __name__ == '__main__':
    folders_list = [['S1', 'S5', 'S6', 'S7'], ['S8'], ['S9', 'S11']]
    phase_list = ['train', 'val', 'test']
    
    for i in range(3):
        maker = H36Maker('/parent/nfs/kun1/users/oleg/h36m', folders_list[i],
                         '/workspace/recplan_data/h36m_long', 10, split=(1, 0, 0), n_downsample=1)
        
        maker.make_phase(maker.filenames, phase_list[i])
