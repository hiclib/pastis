import sys
import pytest
import numpy as np
from sklearn.metrics import euclidean_distances
from scipy import sparse

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 6), reason="Requires python3.6 or higher")

if sys.version_info[0] >= 3:
    from pastis.optimization import poisson
    from pastis.optimization.counts import _format_counts


def test_poisson_objective_haploid():
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
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta)

    obj = poisson.objective(
        X=X_true, counts=counts, alpha=alpha, lengths=lengths)

    assert obj < -1e4


def test_poisson_objective_haploid_biased():
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

    bias = 0.1 + random_state.rand(n)
    counts *= bias.reshape(-1, 1) * bias.reshape(-1, 1).T
    counts = sparse.coo_matrix(counts)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta)

    obj = poisson.objective(
        X=X_true, counts=counts, alpha=alpha, lengths=lengths, bias=bias)

    assert obj < -1e3


def test_poisson_objective_diploid_unambig():
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
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta)

    obj = poisson.objective(
        X=X_true, counts=counts, alpha=alpha, lengths=lengths)

    assert obj < -1e4


def test_poisson_objective_diploid_unambig_biased():
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

    bias = 0.1 + random_state.rand(n)
    counts *= np.tile(bias, 2).reshape(-1, 1) * np.tile(bias, 2).reshape(-1, 1).T
    counts = sparse.coo_matrix(counts)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta)

    obj = poisson.objective(
        X=X_true, counts=counts, alpha=alpha, lengths=lengths, bias=bias)

    assert obj < -1e4


def test_poisson_objective_diploid_ambig():
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
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta)

    obj = poisson.objective(
        X=X_true, counts=counts, alpha=alpha, lengths=lengths)

    assert obj < -1e4


def test_poisson_objective_diploid_ambig_biased():
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

    bias = 0.1 + random_state.rand(n)
    counts *= bias.reshape(-1, 1) * bias.reshape(-1, 1).T
    counts = sparse.coo_matrix(counts)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta)

    obj = poisson.objective(
        X=X_true, counts=counts, alpha=alpha, lengths=lengths, bias=bias)

    assert obj < -1e4


def test_poisson_objective_diploid_partially_ambig():
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
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta)

    obj = poisson.objective(
        X=X_true, counts=counts, alpha=alpha, lengths=lengths)

    assert obj < -1e4


def test_poisson_objective_diploid_partially_ambig_biased():
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

    bias = 0.1 + random_state.rand(n)
    counts *= np.tile(bias, 2).reshape(-1, 1)
    counts *= bias.reshape(-1, 1).T
    counts = sparse.coo_matrix(counts)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta)

    obj = poisson.objective(
        X=X_true, counts=counts, alpha=alpha, lengths=lengths, bias=bias)

    assert obj < -1e4
