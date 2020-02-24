import sys
import pytest
import numpy as np
from sklearn.metrics import euclidean_distances
from numpy.testing import assert_array_almost_equal
from scipy import sparse

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 6), reason="Requires python3.6 or higher")

if sys.version_info[0] >= 3:
    from pastis.optimization import poisson
    from pastis.optimization.counts import _format_counts


def test_pastis_poisson_haploid():
    lengths = np.array([20])
    ploidy = 1
    seed = 42
    alpha, beta = -3., 1.

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    X_true = random_state.rand(n * ploidy, 3)
    dis = euclidean_distances(X_true)
    dis[dis == 0] = np.inf
    counts = beta * dis ** alpha
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts = np.triu(counts, 1)
    counts = sparse.coo_matrix(counts)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=None)

    obj = poisson.objective(
        X=X_true, counts=counts, alpha=alpha, lengths=lengths, bias=None)

    assert obj < 1e-6


def test_pastis_poisson_diploid_unambig():
    lengths = np.array([20])
    ploidy = 2
    seed = 42
    alpha, beta = -3., 1.

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    X_true = random_state.rand(n * ploidy, 3)
    dis = euclidean_distances(X_true)
    dis[dis == 0] = np.inf
    counts = beta * dis ** alpha
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts = np.triu(counts, 1)
    counts = sparse.coo_matrix(counts)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=None)

    obj = poisson.objective(
        X=X_true, counts=counts, alpha=alpha, lengths=lengths, bias=None)

    assert obj < 1e-6


def test_pastis_poisson_diploid_ambig():
    lengths = np.array([20])
    ploidy = 2
    seed = 42
    alpha, beta = -3., 1.

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    X_true = random_state.rand(n * ploidy, 3)
    dis = euclidean_distances(X_true)
    dis[dis == 0] = np.inf
    counts = beta * dis ** alpha
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts = counts[:n, :n] + counts[n:, n:] + counts[:n, n:] + counts[n:, :n]
    counts = np.triu(counts, 1)
    counts = sparse.coo_matrix(counts)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=None)

    obj = poisson.objective(
        X=X_true, counts=counts, alpha=alpha, lengths=lengths, bias=None)

    assert obj < 1e-6


def test_pastis_poisson_diploid_partially_ambig():
    lengths = np.array([20])
    ploidy = 2
    seed = 42
    alpha, beta = -3., 1.

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    X_true = random_state.rand(n * ploidy, 3)
    dis = euclidean_distances(X_true)
    dis[dis == 0] = np.inf
    counts = beta * dis ** alpha
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts = counts[:, :n] + counts[:, n:]
    np.fill_diagonal(counts[:n, :], 0)
    np.fill_diagonal(counts[n:, :], 0)
    counts = sparse.coo_matrix(counts)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=None)

    obj = poisson.objective(
        X=X_true, counts=counts, alpha=alpha, lengths=lengths, bias=None)

    assert obj < 1e-6
