import numpy as np
import torch
from scipy.spatial.distance import cdist
from math import isinf
try:
    import gcp.evaluation.cutils as cutils
except:
    pass


def dtw_dist(x, y, dist=None, warp=1, w=np.inf, s=1.0):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.

    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    :param int warp: how many shifts are computed.
    :param int w: window size limiting the maximal distance between indices of matched entries |i,j|.
    :param float s: weight applied on off-diagonal moves of the path. As s gets larger, the warping path is increasingly biased towards the diagonal
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    r, c = len(x), len(y)
    D1 = np.zeros((r, c))
    for i in range(r):
        for j in range(c):
            # if (isinf(w) or (max(0, i - w) <= j <= min(c, i + w))):
            D1[i, j] = dist(x[i], y[j])

    return dtw(D1, warp, w, s)


def dtw(inp_D0, warp=1, w=np.inf, s=1.0):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.

    :param int warp: how many shifts are computed.
    :param int w: window size limiting the maximal distance between indices of matched entries |i,j|.
    :param float s: weight applied on off-diagonal moves of the path. As s gets larger, the warping path is increasingly biased towards the diagonal
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    r, c = inp_D0.shape
    assert w >= abs(r - c)
    assert s > 0
    if not isinf(w):
        D0 = np.full((r + 1, c + 1), np.inf)
        for i in range(1, r + 1):
            D0[i, max(1, i - w):min(c + 1, i + w + 1)] = 0
        D0[0, 0] = 0
    else:
        D0 = np.zeros((r + 1, c + 1))
        D0[0, 1:] = np.inf
        D0[1:, 0] = np.inf

    D1 = D0[1:, 1:]  # view
    D0[1:, 1:] = inp_D0  # TODO to support w, this needs to be modified to mask the assignment.
    C = D1.copy()
    jrange = range(c)
    for i in range(r):
        if not isinf(w):
            jrange = range(max(0, i - w), min(c, i + w + 1))
        for j in jrange:
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                i_k = min(i + k, r)
                j_k = min(j + k, c)
                min_list += [D0[i_k, j] * s, D0[i, j_k] * s]
            D1[i, j] += min(min_list)
    if r == 1:
        path = np.zeros(c), range(c)
    elif c == 1:
        path = range(r), np.zeros(r)
    else:
        path = _traceback(D0)
    return D1[-1, -1] / sum(D1.shape), C, D1, path

def basic_dtw(C):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.

    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.

    :param C: the cost matrix
    :return:
    """
    r, c = C.shape

    D = np.zeros((r + 1, c + 1))
    D[0, 1:] = np.inf
    D[1:, 0] = np.inf
    D[1:, 1:] = C
    for i in range(r):
        for j in range(c):
            candidates = [D[i, j], D[i + 1, j], D[i, j + 1]]
            D[i + 1, j + 1] += min(candidates)
    path = _traceback(D)
    return D[-1, -1] / (r + c), D[1:, 1:], path

def c_dtw(C):
    """
        Computes Dynamic Time Warping (DTW) of two sequences efficiently in C.

        Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.

        :param C: the cost matrix
        :return:
        """
    r, c = C.shape

    D = np.zeros((r + 1, c + 1))
    D[0, 1:] = np.inf
    D[1:, 0] = np.inf
    D[1:, 1:] = C
    cutils.min_cumsum(D)
    path = _traceback(D)
    return D[-1, -1] / (r + c), D[1:, 1:], path


def batched_dtw(C, end_ind):
    b, r, c = C.shape
    D = np.zeros((b, r + 1, c + 1))
    D[:, 0, 1:] = np.inf
    D[:, 1:, 0] = np.inf
    D[:, 1:, 1:] = C
    for i in range(r):
       for j in range(c):
           candidates = [D[:, i, j], D[:, i + 1, j], D[:, i, j + 1]]
           D[:, i + 1, j + 1] += np.min(np.stack(candidates), axis=0)
    paths, path_lengths = _batched_traceback(D, end_ind)
    return D[np.arange(b), -1, end_ind+1] / (r + end_ind+1), D[:, 1:, 1:], paths, path_lengths


def torch_dtw(C, end_ind):
    b, r, c = C.shape
    D = torch.zeros((b, r + 1, c + 1))
    D[:, 0, 1:] = torch.Tensor([float("Inf")])
    D[:, 1:, 0] = torch.Tensor([float("Inf")])
    D[:, 1:, 1:] = C

    for i in range(r):
        for j in range(c):
            candidates = [D[:, i, j], D[:, i + 1, j], D[:, i, j + 1]]
            D[:, i + 1, j + 1].add_(torch.min(torch.stack(candidates), dim=0).values)

    paths, path_lengths = _torched_traceback(D, end_ind)
    return D[torch.arange(b), -1, (end_ind.float()+1).long()] / (r + end_ind.float()+1), D[:, 1:, 1:], paths, path_lengths


def accelerated_dtw(x, y, dist=None, inp_D0=None, warp=1):
    """
    Computes Dynamic Time Warping (DTW) of two sequences in a faster way.
    Instead of iterating through each element and calculating each distance,
    this uses the cdist function from scipy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)

    :param array x: N1*M array
    :param array y: N2*M array
    :param string or func dist: distance parameter for cdist. When string is given, cdist uses optimized functions for the distance metrics.
    If a string is passed, the distance function can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.
    :param int warp: how many shifts are computed.
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    if np.ndim(x) == 1:
        x = x.reshape(-1, 1)
    if np.ndim(y) == 1:
        y = y.reshape(-1, 1)
    r, c = len(x), len(y)
    D0 = np.zeros((r + 1, c + 1))
    D0[0, 1:] = np.inf
    D0[1:, 0] = np.inf
    D1 = D0[1:, 1:]
    if inp_D0 is not None:
        D0[1:, 1:] = inp_D0
    else:
        D0[1:, 1:] = cdist(x, y, dist)
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                min_list += [D0[min(i + k, r), j],
                             D0[i, min(j + k, c)]]
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = np.zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), np.zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1] / sum(D1.shape), C, D1, path


def _traceback(D):
    # TODO I suspect this doesn't work with fancy stuff (w, s, warp)
    i, j = np.array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = np.argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        elif tb == 2:
            j -= 1
        else:
            raise ValueError

        p.insert(0, i)
        q.insert(0, j)
    return np.array(p), np.array(q)


def _batched_traceback(D, end_ind):
    b, r, c = D.shape
    i, j = np.asarray(np.ones((b,)) * (r - 2), dtype=int), end_ind
    p, q = [i.copy()], [j.copy()]
    path_lengths = np.zeros_like(i)
    cnt = 0
    while (i > 0).any() or (j > 0).any():
        cnt += 1
        path_lengths[(i == 0) & (j == 0) & (path_lengths == 0)] = cnt
        tb = np.argmin(np.stack((D[np.arange(b), i, j], D[np.arange(b), i, j + 1], D[np.arange(b), i + 1, j])), axis=0)
        i[(tb == 0) & (i > 0)] -= 1
        j[(tb == 0) & (j > 0)] -= 1
        i[(tb == 1) & (i > 0)] -= 1
        j[(tb == 2) & (j > 0)] -= 1
        p.insert(0, i.copy())
        q.insert(0, j.copy())
    return (np.array(p), np.array(q)), path_lengths


def _torched_traceback(D, end_ind):
    b, r, c = D.shape
    i, j = (torch.ones((b,)) * (r - 2)).long(), end_ind
    p, q = [i.clone()], [j.clone()]
    path_lengths = torch.zeros_like(i)
    cnt = 0
    while (i > 0).any() or (j > 0).any():
        cnt += 1
        path_lengths[(i == 0) & (j == 0) & (path_lengths == 0)] = cnt
        tb = torch.argmin(
            torch.stack((D[torch.arange(b), i, j], D[torch.arange(b), i, j + 1], D[torch.arange(b), i + 1, j])), dim=0)
        i[(tb == 0) & (i > 0)] -= 1
        j[(tb == 0) & (j > 0)] -= 1
        i[(tb == 1) & (i > 0)] -= 1
        j[(tb == 2) & (j > 0)] -= 1
        p.insert(0, i.clone())
        q.insert(0, j.clone())
    return (torch.stack(p), torch.stack(q)), path_lengths


if __name__ == "__main__":
    b, r, c = 8, 1024, 1000
    min_length = int(c - 1)

    EPS = 1e-5
    import numpy as np
    import time
    np.random.seed(40)
    DD = np.random.rand(b, r, c)
    end_ind = min_length + np.asarray(np.random.rand(b) * (c - min_length - 1), dtype=int)
    
    dd, dd2, pp, pp2, t1, t2 = [], [], [], [], 0.0, 0.0
    for D, i in zip(DD, end_ind):
        s = time.time()
        d, cost_matrix, acc_cost_matrix, path = dtw(D[:, :i+1])
        t1 += time.time() - s
        dd.append(d); pp.append(path)
        s = time.time()
        d2, acc_cost_matrix_2, path_2 = c_dtw(D[:, :i+1])
        t2 += time.time() - s
        dd2.append(d2); pp2.append(path_2)
    print("DTW: {}".format(t1))
    print("C DTW: {}".format(t2))
    
    def check(cond, name):
        print("{}: PASS".format(name)) if cond else print("{}: FAIL".format(name))

    check(not np.any((np.array(dd) - dd2) > EPS), "Distance")
    check(not np.any(np.concatenate([(np.array(pp[i][0]) - np.array(pp2[i][0])) > EPS for i in range(b)])) and \
          not np.any(np.concatenate([(np.array(pp[i][1]) - np.array(pp2[i][1])) > EPS for i in range(b)])), "Paths")

