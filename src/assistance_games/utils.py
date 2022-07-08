import numpy as np
import os

import sparse

_ASSETS = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets')


def get_asset(name):
    return os.path.join(_ASSETS, name)


def sample_distribution(p):
    p = force_dense(p)
    return np.random.choice(np.arange(len(p)), p=p)


def uniform_simplex_sample(n):
    """Uniform sample of n-dim point in 0 <= x_i <= 1, sum_i x_i = 1.
    """
    arr = np.random.rand(n-1)
    arr = np.concatenate([[0, 1], arr])
    arr.sort()
    diffs = arr[1:] - arr[:-1]
    vals = diffs[np.random.permutation(n)]
    return vals


def dict_to_sparse(d, shape):
    points = []
    values = []
    for point, value in d.items():
        points.append(point)
        values.append(value)

    coords = list(zip(*points))

    return sparse.COO(coords, values, shape=shape)


def force_sparse(arr):
    if isinstance(arr, sparse.COO):
        return arr
    else:
        return sparse.COO(arr)


def force_dense(arr):
    if isinstance(arr, sparse.COO):
        return arr.todense()
    else:
        return arr
