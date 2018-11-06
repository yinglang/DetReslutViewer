# coding=utf-8
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float32_t, ndim=2] _bbox_insect(
        np.ndarray[np.float32_t, ndim=1] box,
        np.ndarray[np.float32_t, ndim=2] boxes):
    cdef int N = boxes.shape[0]
    cdef np.ndarray[np.float32_t, ndim=1] t_box
    for i in range(N):
        t_box = boxes[i]

