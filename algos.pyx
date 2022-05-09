# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import cython
from cython.parallel cimport prange, parallel
cimport numpy
import numpy
import math

def floyd_warshall(adjacency_matrix):

    (nrows, ncols) = adjacency_matrix.shape
    assert nrows == ncols
    cdef unsigned int n = nrows

    adj_mat_copy = adjacency_matrix.astype(long, order='C', copy=True)
    assert adj_mat_copy.flags['C_CONTIGUOUS']
    cdef numpy.ndarray[long, ndim=2, mode='c'] M = adj_mat_copy
    cdef numpy.ndarray[long, ndim=2, mode='c'] path = numpy.zeros([n, n], dtype=numpy.int64)

    cdef unsigned int i, j, k
    cdef long M_ij, M_ik, cost_ikkj
    cdef long* M_ptr = &M[0,0]
    cdef long* M_i_ptr
    cdef long* M_k_ptr

    # set unreachable nodes distance to 510
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i][j] = 0
            elif M[i][j] == 0:
                M[i][j] = 510
#510是不是太大了
    # floyed algo
    for k in range(n):
        M_k_ptr = M_ptr + n*k
        for i in range(n):
            M_i_ptr = M_ptr + n*i
            M_ik = M_i_ptr[k]
            for j in range(n):
                cost_ikkj = M_ik + M_k_ptr[j]
                M_ij = M_i_ptr[j]
                if M_ij > cost_ikkj:
                    M_i_ptr[j] = cost_ikkj
                    path[i][j] = k

    # set unreachable path to 510
    for i in range(n):
        for j in range(n):
            if M[i][j] >= 510:
                path[i][j] = 510
                M[i][j] = 510

    return M, path


def get_all_edges(path, i, j):
    cdef unsigned int k = path[i][j]
    if k == 0:
        return []
    else:
        return get_all_edges(path, i, k) + [k] + get_all_edges(path, k, j)


def gen_edge_input(max_dist, path, edge_feat):

    (nrows, ncols) = path.shape
    assert nrows == ncols
    cdef unsigned int n = nrows
    cdef unsigned int max_dist_copy = max_dist

    path_copy = path.astype(long, order='C', casting='safe', copy=True)
    edge_feat_copy = edge_feat.astype(long, order='C', casting='safe', copy=True)
    assert path_copy.flags['C_CONTIGUOUS']
    assert edge_feat_copy.flags['C_CONTIGUOUS']

    cdef numpy.ndarray[long, ndim=4, mode='c'] edge_fea_all = numpy.zeros([n, n, max_dist_copy, edge_feat.shape[-1]], dtype=numpy.int64)
    cdef unsigned int i, j, k, num_path, cur

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if path_copy[i][j] == 510:
                continue

            path = [i] + get_all_edges(path_copy, i, j) + [j]
            #num_path = len(path) - 1 原本的代码，我想加一个距离限制
            num_path = min(len(path) - 1,max_dist_copy)#
            for k in range(num_path):
                edge_fea_all[i, j, k, :] = edge_feat_copy[path[k], path[k+1], :]

    return edge_fea_all


def easy_bin(x, bin):
    x = float(x)
    cnt = 0
    if math.isinf(x):
        return 17
    if math.isnan(x):
        return 18

    while True:
        if cnt == len(bin):
            return cnt
        if x > bin[cnt]:
            cnt += 1
        else:
            return cnt


def bin_rel_pos_3d_1(rel_pos_3d, noise=False):
    (nrows, ncols) = rel_pos_3d.shape
    assert nrows == ncols
    cdef unsigned int n = nrows

    rel_pos_3d_copy = rel_pos_3d.astype(float, order='C', casting='safe', copy=True)
    assert rel_pos_3d_copy.flags['C_CONTIGUOUS']

    cdef numpy.ndarray[long, ndim=2, mode='c'] rel_pos_new = numpy.zeros([n, n], dtype=numpy.int64)
    cdef float t
    for i in range(n):
        for j in range(n):
            t = rel_pos_3d_copy[i][j]
            if noise:
                t += numpy.random.laplace(0.001994, 0.031939)

            rel_pos_new[i][j] = easy_bin(t,[0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0])
    
    return rel_pos_new

