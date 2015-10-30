import numpy as np
from scipy import sparse
from sklearn.metrics import euclidean_distances
from numpy.testing import assert_array_almost_equal
from pastis.optimization import mds


def estimate_X_test():
    n = 50
    random_state = np.random.RandomState(42)
    X_true = random_state.rand(n, 3)
    dis = euclidean_distances(X_true)
    alpha, beta = -3., 1.

    counts = beta * dis ** alpha
    counts = np.triu(counts)
    counts[np.arange(len(counts)), np.arange(len(counts))] = 0
    counts = sparse.coo_matrix(counts)

    X = mds.estimate_X(counts, random_state=random_state)
    assert_array_almost_equal(dis,
                              euclidean_distances(X), 2)
