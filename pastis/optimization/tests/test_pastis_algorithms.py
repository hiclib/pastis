import sys
import pytest
import numpy as np
from sklearn.metrics import euclidean_distances
from numpy.testing import assert_array_almost_equal
from scipy import sparse

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 6), reason="Requires python3.6 or higher")

if sys.version_info[0] >= 3:
    from pastis.optimization import pastis_algorithms


def test_pastis_poisson_diploid_unambig():
    lengths = np.array([25])
    ploidy = 2
    seed = 42
    bcc_lambda = 0
    hsc_lambda = 0
    hsc_r = 0
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

    struct_, infer_var = pastis_algorithms.pastis_poisson(
        counts, lengths, ploidy, outdir=None, alpha=alpha, seed=seed,
        normalize=False, filter_threshold=0, bcc_lambda=bcc_lambda,
        hsc_lambda=hsc_lambda, hsc_r=hsc_r, print_freq=None, history_freq=None,
        save_freq=None)


def test_pastis_poisson_diploid_ambig():
    lengths = np.array([25])
    ploidy = 2
    seed = 42
    bcc_lambda = 0
    hsc_lambda = 0
    hsc_r = 0
    alpha, beta = -3., 1.

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    X_true = random_state.rand(n * ploidy, 3)
    dis = euclidean_distances(X_true)
    dis[dis == 0] = np.inf
    counts = beta * dis ** alpha
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts = (counts[:n, :n] + counts[n:, n:] + counts[:n, n:] + counts[n:, :n])
    counts = np.triu(counts, 1)
    counts = sparse.coo_matrix(counts)

    struct_, infer_var = pastis_algorithms.pastis_poisson(
        counts, lengths, ploidy, outdir=None, alpha=alpha, seed=seed,
        normalize=False, filter_threshold=0, bcc_lambda=bcc_lambda,
        hsc_lambda=hsc_lambda, hsc_r=hsc_r, print_freq=None, history_freq=None,
        save_freq=None)


def test_pastis_poisson_diploid_partially_ambig():
    lengths = np.array([25])
    ploidy = 2
    seed = 42
    bcc_lambda = 0
    hsc_lambda = 0
    hsc_r = 0
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

    struct_, infer_var = pastis_algorithms.pastis_poisson(
        counts, lengths, ploidy, outdir=None, alpha=alpha, seed=seed,
        normalize=False, filter_threshold=0, bcc_lambda=bcc_lambda,
        hsc_lambda=hsc_lambda, hsc_r=hsc_r, print_freq=None, history_freq=None,
        save_freq=None)
