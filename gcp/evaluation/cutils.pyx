# distutils: language = c++
# distutils: extra_compile_args = -O3 -w -DNDEBUG -std=c++11 -DEIGEN_NO_MALLOC
# distutils: extra_link_args = -fopenmp

###############################################################################
## Author: Andrew Miller <acm@seas.harvard.edu>
###############################################################################
from __future__ import division
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport fmin
from cython cimport view
np.import_array()

# TYPEDEFS
ctypedef np.float_t FLOAT_t
ctypedef np.ulong_t INDEX_t

@cython.boundscheck(False)
@cython.wraparound(False)
def min_cumsum(FLOAT_t[:,::1] D):
    cdef INDEX_t r, c, i, j
    r = D.shape[0] - 1
    c = D.shape[1] - 1
    for i in range(r):
        for j in range(c):
            D[i+1, j+1] += fmin(D[i, j], fmin(D[i, j+1], D[i+1, j]))
    return D