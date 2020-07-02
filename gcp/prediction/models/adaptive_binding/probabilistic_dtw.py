from math import floor

import numpy as np
import torch
from blox.tensor import ndim
from blox.torch.ops import batchwise_assign, batchwise_index
from blox.utils import timing
from torch import nn


def fast_gak(C, transition, begin_inds=0):
    """
    Computes the global alignment kernel (Cuturi'07, A kernel for time series based on global alignments).
    This version parallelizes the computation across the diagonal
    This version is able to process batched sequences of variable length by using begin_inds

    :param C: the cost matrix
    :return: the kernel value and the matrix of intermediate values (kernel values for smaller sequences)
    """
    """
    Computes Dynamic Time Warping (DTW) of two sequences.
    This version multiplies the cost instead of adding it. This can be used if the costs represent exponential values
    exp_dtw = exp(basic_dtw(log(C)))

    :param C: the cost matrix ... x n_x x n_y
    :param aggr: the function used for aggregating candidates.
    :param transition: if 'nohor', horizontal transitions are excluded, i.e. one x only matches one y
    (but not the other way around)
    :return: the minimum distance and the accumulated cost matrix
    """
    r, c = C.shape[-2:]
    D = torch.full_like(C, -np.inf)
    batchwise_assign(D[:, 0], begin_inds, batchwise_index(C[:, 0], begin_inds))
    
    assert transition == 'nohor'
    assert r >= c
    
    # impossible = (1 - torch.ones_like(D).tril()).byte()
    # impossible += (1 - torch.ones_like(D).triu(-diff)).byte()
    
    # Iterate over diagonals
    for i in range(1, r + c):
        ids = torch.arange(i + 1).flip(0)
        jds = torch.arange(i + 1)
        # exclude boundary violations
        ids = ids[max(0, i - r + 1):c]
        jds = jds[max(0, i - r + 1):c]
        
        # ToDo clean this up?
        ids = ids.flip(0)
        jds = jds.flip(0)
        
        skip = D[..., ids - 1, jds]
        step = D[..., ids - 1, jds - 1]
        
        # # Allow horizontal transitions where the y sequence ended
        # # jds[-1] is the column index of the lowest element on the diagonal
        # repeat = torch.full_like(skip, -np.inf)
        # allow_mask = jds[-1] > end_inds
        # repeat[allow_mask, -1] = D[allow_mask, ids[-1], jds[-1] - 1]
        
        # Recursion
        add = torch.logsumexp(ndim.stack([skip, step], -1), -1)
        new_cost = C[..., ids, jds] + add
        mask = D[..., ids, jds] != -np.inf
        new_cost[mask] = D[..., ids, jds][mask]  # If D was specified, leave as is (for begin_inds)
        
        # TODO change to simple array indexing once the backward pass for it is implemented
        mask = ids2mask(D, ids, jds)
        if hasattr(mask, 'bool'):
            mask = mask.bool()
        D = D.masked_scatter(mask, (new_cost).flatten())
    return D


def ids2mask(mat, ids, jds):
    mask = torch.zeros_like(mat, dtype=torch.uint8)
    mask[..., ids, jds] = 1
    return mask


def soft_dtw(C, end_inds=None):
    """
    Computes the expected edge frequencies. See https://www.overleaf.com/read/jksjyppbrdgn
    
    :param C: the cost matrix, n_x x n_y
    :param end_inds: the end indices for the sequences y. The sequences y will be assumed to have the length as per the
    end index and the remainer of the frames will not be matched
    :return: the matrix with expected edge frequencies
    """
    C = (-C).double()
    batch, r, c = C.shape
    if end_inds is None:
        end_inds = torch.full([batch], c - 1, dtype=torch.long)
    
    # mask = torch.zeros_like(C)
    # if end_inds is not None:
    #     for i, ind in enumerate(end_inds):
    #         mask[i, :-1, end_inds[i]:] = 1
    #     C[mask.byte()] = -np.inf

    # Compute forward-backward
    comb_C = torch.cat([C, ndim.flip(C, [-1, -2])], 0)
    # The backward begins with end indices, not (-1,-1)
    comb_begin_inds = torch.cat([torch.zeros_like(end_inds), c - end_inds - 1], 0)

    accum = fast_gak(comb_C, transition='nohor', begin_inds=comb_begin_inds)
    
    forward = accum[:batch]
    backward = ndim.flip(accum[batch:], [-1, -2])

    # Compute expected matrix
    z = batchwise_index(forward[:, -1], end_inds)[:, None, None]
    e = forward + backward - C
    e[C == -np.inf] = -np.inf
    w = (e - z).exp()
    
    if not equal(w.sum(2).max(), 1, eps=1e-2):
        print('warning: dtw is not stable with these cost values')
        import pdb; pdb.set_trace()

    return w.float()


def equal(n, m, eps):
    return m - eps < n < m + eps
