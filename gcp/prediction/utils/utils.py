import datetime
import os
from functools import partial

import dload
import numpy as np
import torch
from blox import AttrDict
from blox.tensor.ops import batchwise_index


def select_e_0_e_g(seq, start_ind, end_ind):
    e_0 = batchwise_index(seq, start_ind)
    e_g = batchwise_index(seq, end_ind)
    return e_0, e_g


def get_end_ind(pad_mask):
    """
    :param pad_mask: torch tensor with 1 where there is an actual image and zeros where there's padding
    pad_mask has shape batch_size x max_seq_len
    :return:
    """
    max_seq_len = pad_mask.shape[1]
    end_ind = torch.argmax(pad_mask * torch.arange(max_seq_len, dtype=torch.float, device=pad_mask.device), 1)

    return end_ind


def get_pad_mask(end_ind, max_seq_len):
    """
    :param pad_mask: torch tensor with 1 where there is an actual image and zeros where there's padding
    pad_mask has shape batch_size x max_seq_len
    :return:
    """
    use_torch = isinstance(end_ind, torch.Tensor)
    
    if use_torch:
        arange_fn = partial(torch.arange, dtype=torch.long, device=end_ind.device)
    else:
        arange_fn = np.arange
    
    pad_mask = arange_fn(max_seq_len) <= end_ind[:, None]
    
    if use_torch:
        pad_mask = pad_mask.float()
    else:
        pad_mask = pad_mask.astype(np.float)
    
    return pad_mask


def datetime_str():
    return datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")


def make_path(exp_dir, conf_path, prefix, make_new_dir):
    # extract the subfolder structure from config path
    path = conf_path.split('experiments/', 1)[1]
    if make_new_dir:
        prefix += datetime_str()
    base_path = os.path.join(exp_dir, path)
    return os.path.join(base_path, prefix) if prefix else base_path


def set_seeds(seed=0):
    """Sets all seeds and disables non-determinism in cuDNN backend."""
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def get_dataset_path(dataset_name):
    """Returns path to dataset."""
    return os.path.join(os.environ["GCP_DATA_DIR"], dataset_name)


def download_data(dataset_name):
    """Downloads data if not yet existent."""
    DATA_URLs = AttrDict(
        nav_9rooms='https://www.seas.upenn.edu/~oleh/datasets/gcp/nav_9rooms.zip',
        nav_25rooms='https://www.seas.upenn.edu/~oleh/datasets/gcp/nav_25rooms.zip',
        sawyer='https://www.seas.upenn.edu/~oleh/datasets/gcp/sawyer.zip',
        h36m='https://www.seas.upenn.edu/~oleh/datasets/gcp/h36m.zip',
    )
    if dataset_name not in DATA_URLs:
        raise ValueError("Dataset identifier {} is not known!".format(dataset_name))
    if not os.path.exists(get_dataset_path(dataset_name)):
        print("Downloading dataset from {} to {}.".format(DATA_URLs[dataset_name], os.environ["GCP_DATA_DIR"]))
        print("This may take a few minutes...")
        dload.save_unzip(DATA_URLs[dataset_name], os.environ["GCP_DATA_DIR"], delete_after=True)
        print("...Done!")
