import random
import torch.utils.data as data
import numpy as np
from PIL import Image
import glob
import h5py
import random
import torch
import matplotlib.pyplot as plt
from PIL import Image

def get_traj_per_file(file):
    with h5py.File(file, 'r') as F:
        return len(list(F.keys()))

class Video_Dataset(data.Dataset):
    def __init__(self,
                 data_dir,
                 phase
                 ):

        self.data_dir = data_dir + '/' + phase
        self.train_val_split = 0.95

        self.filenames = sorted(glob.glob(self.data_dir + '/*.h5'))
        if not self.filenames:
            raise RuntimeError('No filenames found')
        random.seed(1)
        random.shuffle(self.filenames)

        self.traj_per_file = get_traj_per_file(self.filenames[0])

    def __getitem__(self, index):
        file_index = index//self.traj_per_file
        path = self.filenames[file_index]

        in_file_ind = index % self.traj_per_file

        with h5py.File(path, 'r') as F:
            images = np.asarray(F['traj{}/images'.format(in_file_ind)])
            actions = np.asarray(F['traj{}/actions'.format(in_file_ind)])
        return images, actions

    def __len__(self):
        return len(self.filenames)*self.traj_per_file


def make_data_loader(data_dir, phase):
    """
    :param data_dir:
    :param phase: either train, val or test
    :return: dataset iterator
    """
    d = Video_Dataset(data_dir, phase)
    return torch.utils.data.DataLoader(d, batch_size=3, shuffle=True,num_workers=1, pin_memory=True)

if __name__ == '__main__':

    data_dir = '/mnt/sda1/recursive_planning_data/sim/cartgripper/multi_block_2d'
    d = Video_Dataset(data_dir, phase='train')
    loader = torch.utils.data.DataLoader(d, batch_size=3, shuffle=True,num_workers=1, pin_memory=True)

    deltat = []
    for i_batch, sample_batched in enumerate(loader):

        images, actions = sample_batched

        print(actions)
        for t in range(images.shape[1]):
            plt.imshow(images[0,t,0])
            plt.show()
        # images = images.numpy().transpose((0, 1, 3, 4, 2))
        # file = '/'.join(str.split(config['agent']['data_save_dir'], '/')[:-1]) + '/example'
        # comp_single_video(file, images)

