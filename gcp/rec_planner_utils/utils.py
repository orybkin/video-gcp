from functools import partial

import numpy as np
import torch
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