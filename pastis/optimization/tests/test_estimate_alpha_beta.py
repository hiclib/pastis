import sys
import pytest
import numpy as np
from sklearn.metrics import euclidean_distances
from numpy.testing import assert_array_almost_equal
from scipy import sparse

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 6), reason="Requires python3.6 or higher")

if sys.version_info[0] >= 3:
    from pastis.optimization import estimate_alpha_beta
    from pastis.optimization.counts import _format_counts


def test_estimate_alpha_beta_haploid():
    lengths = np.array([20])
    ploidy = 1
    seed = 42
    alpha_true, beta_true = -3., 2.

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    X_true = random_state.rand(n * ploidy, 3)
    dis = euclidean_distances(X_true)
    dis[dis == 0] = np.inf
    counts = beta_true * dis ** alpha_true
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts = np.triu(counts, 1)
    counts = sparse.coo_matrix(counts)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta_true)

    alpha, obj, converged, _ = estimate_alpha_beta.estimate_alpha(
        X=X_true, counts=counts, alpha_init=alpha_true, lengths=lengths)

    beta = list(estimate_alpha_beta._estimate_beta(
        X=X_true, counts=counts, alpha=alpha, lengths=lengths,
        verbose=False).values())[0]

    assert converged
    assert obj < -1e4
    assert_array_almost_equal(alpha_true, alpha, decimal=5)
    assert_array_almost_equal(beta_true, beta, decimal=3)


def test_estimate_alpha_beta_haploid_biased():
    lengths = np.array([20])
    ploidy = 1
    seed = 42
    alpha_true, beta_true = -3., 3.

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    X_true = random_state.rand(n * ploidy, 3)
    dis = euclidean_distances(X_true)
    dis[dis == 0] = np.inf
    counts = beta_true * dis ** alpha_true
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts = np.triu(counts, 1)

    bias = 0.1 + random_state.rand(n)
    counts *= bias.reshape(-1, 1) * bias.reshape(-1, 1).T
    counts = sparse.coo_matrix(counts)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta_true)

    alpha, obj, converged, _ = estimate_alpha_beta.estimate_alpha(
        X=X_true, counts=counts, alpha_init=alpha_true, lengths=lengths,
        bias=bias)

    beta = list(estimate_alpha_beta._estimate_beta(
        X=X_true, counts=counts, alpha=alpha, lengths=lengths, bias=bias,
        verbose=False).values())[0]

    assert converged
    assert obj < -1e3
    assert_array_almost_equal(alpha_true, alpha, decimal=5)
    assert_array_almost_equal(beta_true, beta, decimal=1)


def test_estimate_alpha_beta_diploid_unambig():
    lengths = np.array([20])
    ploidy = 2
    seed = 42
    alpha_true, beta_true = -3., 2.

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    X_true = random_state.rand(n * ploidy, 3)
    dis = euclidean_distances(X_true)
    dis[dis == 0] = np.inf
    counts = beta_true * dis ** alpha_true
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts = np.triu(counts, 1)
    counts = sparse.coo_matrix(counts)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta_true)

    alpha, obj, converged, _ = estimate_alpha_beta.estimate_alpha(
        X=X_true, counts=counts, alpha_init=alpha_true, lengths=lengths)

    beta = list(estimate_alpha_beta._estimate_beta(
        X=X_true, counts=counts, alpha=alpha, lengths=lengths,
        verbose=False).values())[0]

    assert converged
    assert obj < -1e4
    assert_array_almost_equal(alpha_true, alpha, decimal=5)
    assert_array_almost_equal(beta_true, beta, decimal=3)


def test_estimate_alpha_beta_diploid_unambig_biased():
    lengths = np.array([20])
    ploidy = 2
    seed = 42
    alpha_true, beta_true = -3., 2.

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    X_true = random_state.rand(n * ploidy, 3)
    dis = euclidean_distances(X_true)
    dis[dis == 0] = np.inf
    counts = beta_true * dis ** alpha_true
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts = np.triu(counts, 1)

    bias = 0.1 + random_state.rand(n)
    counts *= np.tile(bias, 2).reshape(-1, 1) * np.tile(bias, 2).reshape(-1, 1).T
    counts = sparse.coo_matrix(counts)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta_true)

    alpha, obj, converged, _ = estimate_alpha_beta.estimate_alpha(
        X=X_true, counts=counts, alpha_init=alpha_true, lengths=lengths,
        bias=bias)

    beta = list(estimate_alpha_beta._estimate_beta(
        X=X_true, counts=counts, alpha=alpha, lengths=lengths, bias=bias,
        verbose=False).values())[0]

    #assert converged
    assert obj < -1e4
    assert_array_almost_equal(alpha_true, alpha, decimal=5)
    assert_array_almost_equal(beta_true, beta, decimal=1)


def test_estimate_alpha_beta_diploid_ambig():
    lengths = np.array([20])
    ploidy = 2
    seed = 42
    alpha_true, beta_true = -3., 4.

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    X_true = random_state.rand(n * ploidy, 3)
    dis = euclidean_distances(X_true)
    dis[dis == 0] = np.inf
    counts = beta_true * dis ** alpha_true
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts = counts[:n, :n] + counts[n:, n:] + counts[:n, n:] + counts[n:, :n]
    counts = np.triu(counts, 1)
    counts = sparse.coo_matrix(counts)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta_true)

    alpha, obj, converged, _ = estimate_alpha_beta.estimate_alpha(
        X=X_true, counts=counts, alpha_init=alpha_true, lengths=lengths)

    beta = list(estimate_alpha_beta._estimate_beta(
        X=X_true, counts=counts, alpha=alpha, lengths=lengths,
        verbose=False).values())[0]

    assert converged
    assert obj < -1e4
    assert_array_almost_equal(alpha_true, alpha, decimal=5)
    assert_array_almost_equal(beta_true, beta, decimal=3)


def test_estimate_alpha_beta_diploid_ambig_biased():
    lengths = np.array([20])
    ploidy = 2
    seed = 42
    alpha_true, beta_true = -3., 5.

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    X_true = random_state.rand(n * ploidy, 3)
    dis = euclidean_distances(X_true)
    dis[dis == 0] = np.inf
    counts = beta_true * dis ** alpha_true
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts = counts[:n, :n] + counts[n:, n:] + counts[:n, n:] + counts[n:, :n]
    counts = np.triu(counts, 1)

    bias = 0.1 + random_state.rand(n)
    counts *= bias.reshape(-1, 1) * bias.reshape(-1, 1).T
    counts = sparse.coo_matrix(counts)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta_true)

    alpha, obj, converged, _ = estimate_alpha_beta.estimate_alpha(
        X=X_true, counts=counts, alpha_init=alpha_true, lengths=lengths,
        bias=bias)

    beta = list(estimate_alpha_beta._estimate_beta(
        X=X_true, counts=counts, alpha=alpha, lengths=lengths, bias=bias,
        verbose=False).values())[0]

    assert converged
    assert obj < -1e4
    assert_array_almost_equal(alpha_true, alpha, decimal=5)
    assert_array_almost_equal(beta_true, beta, decimal=3)


def test_estimate_alpha_beta_diploid_partially_ambig():
    lengths = np.array([20])
    ploidy = 2
    seed = 42
    alpha_true, beta_true = -3., 4.

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    X_true = random_state.rand(n * ploidy, 3)
    dis = euclidean_distances(X_true)
    dis[dis == 0] = np.inf
    counts = beta_true * dis ** alpha_true
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts = counts[:, :n] + counts[:, n:]
    np.fill_diagonal(counts[:n, :], 0)
    np.fill_diagonal(counts[n:, :], 0)
    counts = sparse.coo_matrix(counts)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta_true)

    alpha, obj, converged, _ = estimate_alpha_beta.estimate_alpha(
        X=X_true, counts=counts, alpha_init=alpha_true, lengths=lengths)

    beta = list(estimate_alpha_beta._estimate_beta(
        X=X_true, counts=counts, alpha=alpha, lengths=lengths,
        verbose=False).values())[0]

    assert converged
    assert obj < -1e4
    assert_array_almost_equal(alpha_true, alpha, decimal=5)
    assert_array_almost_equal(beta_true, beta, decimal=3)


def test_estimate_alpha_beta_diploid_partially_ambig_biased():
    lengths = np.array([20])
    ploidy = 2
    seed = 42
    alpha_true, beta_true = -3., 4.

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    X_true = random_state.rand(n * ploidy, 3)
    dis = euclidean_distances(X_true)
    dis[dis == 0] = np.inf
    counts = beta_true * dis ** alpha_true
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts = counts[:, :n] + counts[:, n:]
    np.fill_diagonal(counts[:n, :], 0)
    np.fill_diagonal(counts[n:, :], 0)

    bias = 0.1 + random_state.rand(n)
    counts *= np.tile(bias, 2).reshape(-1, 1)
    counts *= bias.reshape(-1, 1).T
    counts = sparse.coo_matrix(counts)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta_true)

    alpha, obj, converged, _ = estimate_alpha_beta.estimate_alpha(
        X=X_true, counts=counts, alpha_init=alpha_true, lengths=lengths,
        bias=bias)

    beta = list(estimate_alpha_beta._estimate_beta(
        X=X_true, counts=counts, alpha=alpha, lengths=lengths, bias=bias,
        verbose=False).values())[0]

    assert converged
    assert obj < -1e4
    assert_array_almost_equal(alpha_true, alpha, decimal=5)
    assert_array_almost_equal(beta_true, beta, decimal=3)
