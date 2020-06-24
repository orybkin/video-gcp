from math import floor

import numpy as np
import torch
from blox.tensor import ndim
from blox.torch.ops import batchwise_assign, batchwise_index
from blox.utils import timing
from torch import nn


# ------------------ In development ------------------


def basic_dtw(C):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.

    :param C: the cost matrix
    :return: the minimum distance and the accumulated cost matrix
    """
    r, c = C.shape
    D = C.copy()
    
    D[0] = np.cumsum(C[0])
    D[:, 0] = np.cumsum(C[:, 0])
    
    for i in range(1, r):
        for j in range(1, c):
            candidates = [D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]]
            D[i, j] += min(candidates)
    return D


def exp_dtw(C, aggr=min):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.
    This version multiplies the cost instead of adding it. This can be used if the costs represent exponential values
    exp_dtw = exp(basic_dtw(log(C)))

    :param C: the cost matrix
    :param aggr: the function used for aggregating candidates.
    :param transition: if 'nohor', horizontal transitions are excluded, i.e. one x only matches one y
    (but not the other way around)
    :return: the minimum distance and the accumulated cost matrix
    """
    r, c = C.shape
    D = C.copy()
    
    D[0] = np.cumprod(C[0])
    D[:, 0] = np.cumprod(C[:, 0])
    
    for i in range(1, r):
        for j in range(1, c):
            candidates = [D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]]
            D[i, j] *= aggr(candidates)
    return D


# @torch.jit.script
# def extra_fast_gak(C):
#    """
#    Computes the global alignment kernel (Cuturi'07, A kernel for time series based on global alignments).
#    This version parallelizes across the diagonal and uses torch.jit. This should be as fast as it gets.
#
#    :param C: the cost matrix
#    :return: the kernel value and the matrix of intermediate values (kernel values for smaller sequences)
#    """
#    """
#    Computes Dynamic Time Warping (DTW) of two sequences.
#    This version multiplies the cost instead of adding it. This can be used if the costs represent exponential values
#    exp_dtw = exp(basic_dtw(log(C)))
#
#    :param C: the cost matrix ... x n_x x n_y
#    :param aggr: the function used for aggregating candidates.
#    :param transition: if 'nohor', horizontal transitions are excluded, i.e. one x only matches one y
#    (but not the other way around)
#    :return: the minimum distance and the accumulated cost matrix
#    """
#    r, c = C.shape[-2:]
#    diff = r - c
#    D = C.clone()
#
#    assert r >= c
#
#    ids = torch.arange(c - 1).long()
#    jds = torch.arange(1, c).long()
#    D[:, ids, jds] = torch.ones(1, device=D.device, dtype=D.dtype) * (-np.inf)
#
#    # Iterate over diagonals
#    for i in range(r+c):
#        if i > 0:
#            ids = torch.arange(i + 1).flip(0).long()
#            jds = torch.arange(i + 1).long()
#            # exclude boundary violations
#            # ids = ids[max(0, i - r + 1):c]
#            # jds = jds[max(0, i - r + 1):c]
#            # exclude the impossible elements and boundary violations. necessary in the jit implementation.
#            ids = ids[int(max(max(0, i - r + 1), floor((i - diff) / 2))):int(min(c, floor(i/2) + 1))]
#            jds = jds[int(max(max(0, i - r + 1), floor((i - diff) / 2))):int(min(c, floor(i/2) + 1))]
#
#            skip = D[:, ids - 1, jds]
#
#            # TODO this can be reduced with the extended array trick
#            if i >= r:
#                step = D[:, ids - 1, jds - 1]
#            else:
#                step = D[:, ids[1:] - 1, jds[1:] - 1]
#                step = torch.cat([-torch.ones_like(skip[..., :1]) * np.inf, step], -1)
#
#            # Recursion
#            D[:, ids, jds] += torch.logsumexp(torch.stack([skip, step], -1), -1)
#    return D

def gak(C, transition):
    """
    Computes the global alignment kernel (Cuturi'07, A kernel for time series based on global alignments).

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
    diff = r - c
    D = ndim.copy(C)
    
    if transition == 'nohor':
        assert r >= c
        impossible = (1 - torch.ones_like(D).tril()).byte()
        impossible += (1 - torch.ones_like(D).triu(-diff)).byte()
        D[impossible] = - np.inf

    jrange = range(0, c)
    for i in range(0, r):
        if transition == 'nohor':
            jrange = range(max(i - diff, 0), min(i + 1, c))
            
        for j in jrange:
            candidates = []
            if i == 0 and j == 0:
                continue
            if i >= 0 and j >= 0:
                candidates += [D[..., i - 1, j - 1]]
            if i >= 0:
                candidates += [D[..., i - 1, j]]
            if j >= 0 and not transition == 'nohor':
                # TODO for some reason 'nohor' causes nan gradients for the excluded elements
                candidates += [D[..., i, j - 1]]
        
            D[..., i, j] += torch.logsumexp(ndim.stack(candidates, -1), -1)
            
    return D


def fast_gak_nograd(C, transition):
    """
    Computes the global alignment kernel (Cuturi'07, A kernel for time series based on global alignments).
    This version parallelizes the computation across the diagonal.
    The backward pass does not work for this version

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
    diff = r - c
    D = ndim.copy(C)
    
    assert transition == 'nohor'
    assert r >= c
    
    impossible = (1 - torch.ones_like(D).tril()).byte()
    impossible += (1 - torch.ones_like(D).triu(-diff)).byte()
    D[impossible] = - np.inf
    
    # Iterate over diagonals
    for i in range(1, r + c):
        ids = torch.arange(i + 1).flip(0)
        jds = torch.arange(i + 1)
        # exclude boundary violations
        ids = ids[max(0, i - r + 1):c]
        jds = jds[max(0, i - r + 1):c]
        # exclude the impossible elements and boundary violations. doesn't seem to speed up computation.
        # ids = ids[max(0, i - r + 1, floor((i - diff) / 2)):min(c, floor((i)/2)+1)]
        # jds = jds[max(0, i - r + 1, floor((i - diff) / 2)):min(c, floor((i)/2)+1)]
        
        skip = D[..., ids - 1, jds]
        
        # TODO this can be reduced with the extended array trick
        if i >= r:
            step = D[..., ids - 1, jds - 1]
        else:
            step = D[..., ids[1:] - 1, jds[1:] - 1]
            step = torch.cat([-torch.ones_like(skip[..., :1])*np.inf, step], -1)

        # Recursion
        D[..., ids, jds] += torch.logsumexp(ndim.stack([skip, step], -1), -1)
    return D


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


if __name__ == "__main__":
    # Basic DTW
    # np.random.seed(1)
    # basic_dtw(np.zeros((4, 5)))
    # D = basic_dtw(np.random.rand(4, 5))
    # print(D[-1,-1])
    #
    # # exp dtw
    # np.random.seed(1)
    # D = exp_dtw(np.exp(np.random.rand(4, 5)))
    # print(np.log(D[-1, -1]))
    # print('these two numbers should be the same')
    #
    # # soft dtw
    np.random.seed(2)
    cost = np.random.rand(1, 5, 4)
    # ww = soft_dtw(cost)
    # print(ww)

    torch_cost = torch.from_numpy(cost)
    end_inds = torch.from_numpy(np.asarray([2]))
    ww = soft_dtw(torch_cost)
    print(ww)
    ww = soft_dtw(torch_cost, end_inds)
    ww = soft_dtw(torch.cat([torch_cost, torch_cost[:, :, [0]]], 2), torch.from_numpy(np.asarray([3])))
    print(ww)
    
    import pdb; pdb.set_trace()
    # print('these two arrays should be the same')
    
    # x = nn.Parameter(torch.zeros(5), requires_grad=True)
    # i = torch.arange(1, 3)
    # y = x.clone()
    # z = x.clone()
    # # y[1] += 1
    # # y[[0]] = y[[1]]
    # # z[[1]] = y[[1]]
    # # y[[0]] = z[[1]]
    # mask = torch.zeros_like(x, dtype=torch.bool)
    # mask1 = mask.clone()
    # mask[[1,2]] = 1
    # mask1[[2,3]] = 1
    # y = y.masked_scatter(mask, y[mask1])
    # y.sum().backward()
    # print(x.grad)


    # with torch.autograd.set_detect_anomaly(True):
    #     # x = nn.Parameter(torch_cost[:, :3, :3], requires_grad=True)
    #     x = nn.Parameter(torch_cost, requires_grad=True)
    #     # x = nn.Parameter(torch.ones(3)*-np.inf, requires_grad=True)
    #     y = soft_dtw(x)
    #     # y = torch.logsumexp(x, 0)
    #     print(y)
    #     y[:, 2, 1].backward()
    #     print(x.grad)
    #     import pdb; pdb.set_trace()

    # This is a very simple test for learning a cost matrix that penalizes a certain path.
    from blox.torch.training import MiniTrainer

    cost = nn.Parameter(torch.zeros((1, 5, 4)), requires_grad=True)
    def step(i):
        print('cost 2,1: ', cost[:, 2, 1])
        return soft_dtw(cost)[:, 2, 1]
    trainer = MiniTrainer(None, step, [cost])

    with timing('training time: '):
        trainer.train(100)
    print(cost)
    
    
    import pdb; pdb.set_trace()
    
