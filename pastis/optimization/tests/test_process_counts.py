import sys
import pytest
import numpy as np
from sklearn.metrics import euclidean_distances
from scipy import sparse

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 6), reason="Requires python3.6 or higher")

if sys.version_info[0] >= 3:
    from pastis.optimization import counts as process_counts


def test_3d_indices_haploid():
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

    row3d, col3d = process_counts._counts_indices_to_3d_indices(
        counts=counts, n=n, ploidy=ploidy)

    assert np.array_equal(counts.row, row3d)
    assert np.array_equal(counts.col, col3d)


def test_3d_indices_diploid_unambig():
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

    row3d, col3d = process_counts._counts_indices_to_3d_indices(
        counts=counts, n=n, ploidy=ploidy)

    assert np.array_equal(counts.row, row3d)
    assert np.array_equal(counts.col, col3d)


def test_3d_indices_diploid_ambig():
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

    row3d, col3d = process_counts._counts_indices_to_3d_indices(
        counts=counts, n=n, ploidy=ploidy)

    row3d_true = np.concatenate([np.tile(counts.row, 2), np.tile(counts.row, 2) + n])
    col3d_true = np.tile(np.concatenate([counts.col, counts.col + n]), 2)
    assert np.array_equal(row3d_true, row3d)
    assert np.array_equal(col3d_true, col3d)


def test_3d_indices_diploid_partially_ambig():
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

    row3d, col3d = process_counts._counts_indices_to_3d_indices(
        counts=counts, n=n, ploidy=ploidy)

    row3d_true = np.tile(counts.row, 2)
    col3d_true = np.concatenate([counts.col, counts.col + n])
    assert np.array_equal(row3d_true, row3d)
    assert np.array_equal(col3d_true, col3d)
