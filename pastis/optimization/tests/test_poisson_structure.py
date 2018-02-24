import numpy as np
from sklearn.metrics import euclidean_distances
from numpy.testing import assert_array_almost_equal
from scipy import sparse
from pastis.optimization import poisson_structure


def test_estimate_X_with_diag():
    n = 25
    random_state = np.random.RandomState(42)
    X_true = random_state.rand(n, 3)
    dis = euclidean_distances(X_true)
    alpha, beta = -3., 1.

    counts = beta * dis ** alpha
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts[np.arange(len(counts)), np.arange(len(counts))] = 5
    counts = np.triu(counts)
    counts = sparse.coo_matrix(counts)

    X = poisson_structure.estimate_X(counts, random_state=random_state)


def test_estimate_X_ini():
    n = 25
    random_state = np.random.RandomState(42)
    X_true = random_state.rand(n, 3)
    dis = euclidean_distances(X_true)
    alpha, beta = -3., 1.
    ini = list(np.random.randn(n * 3))

    counts = beta * dis ** alpha
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts[np.arange(len(counts)), np.arange(len(counts))] = 5
    counts = np.triu(counts)
    counts = sparse.coo_matrix(counts)

    X = poisson_structure.estimate_X(counts, random_state=random_state,
                                     ini=ini)
