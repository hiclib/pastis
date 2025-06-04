import sys
import pytest
import numpy as np
from sklearn.metrics import euclidean_distances
from numpy.testing import assert_array_almost_equal, assert_allclose
from scipy import sparse

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 6), reason="Requires python3.6 or higher")

if sys.version_info[0] >= 3:
    from pastis.optimization import estimate_alpha_beta
    from pastis.optimization.counts import _format_counts, NullCountsMatrix
    from pastis.optimization.constraints import Constraints
    from pastis.optimization.constraints import _mean_interhomolog_counts


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


def test_estimate_alpha_beta_diploid_combo():
    lengths = np.array([20])
    ploidy = 2
    seed = 42
    alpha_true, beta_true_single = -3., 4.
    ratio_ambig, ratio_pa, ratio_ua = [1 / 3] * 3
    bias = None

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    X_true = random_state.rand(n * ploidy, 3)
    dis = euclidean_distances(X_true)
    dis[dis == 0] = np.inf
    poisson_intensity = dis ** alpha_true

    ambig_counts = ratio_ambig * beta_true_single * poisson_intensity
    ambig_counts[np.isnan(ambig_counts) | np.isinf(ambig_counts)] = 0
    ambig_counts = ambig_counts[:n, :n] + ambig_counts[n:, n:] + ambig_counts[:n, n:] + ambig_counts[n:, :n]
    ambig_counts = np.triu(ambig_counts, 1)
    ambig_counts = sparse.coo_matrix(ambig_counts)

    pa_counts = ratio_pa * beta_true_single * poisson_intensity
    pa_counts[np.isnan(pa_counts) | np.isinf(pa_counts)] = 0
    pa_counts = pa_counts[:, :n] + pa_counts[:, n:]
    np.fill_diagonal(pa_counts[:n, :], 0)
    np.fill_diagonal(pa_counts[n:, :], 0)
    pa_counts = sparse.coo_matrix(pa_counts)

    ua_counts = ratio_ua * beta_true_single * poisson_intensity
    ua_counts[np.isnan(ua_counts) | np.isinf(ua_counts)] = 0
    ua_counts = np.triu(ua_counts, 1)
    ua_counts = sparse.coo_matrix(ua_counts)

    counts_raw = [ambig_counts, pa_counts, ua_counts]
    beta_true = np.array([ratio_ambig, ratio_pa, ratio_ua]) * beta_true_single
    counts = _format_counts(
        counts=counts_raw, lengths=lengths, ploidy=ploidy, beta=beta_true)

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


def test_estimate_alpha_beta_diploid_mhs_constraint():
    lengths = np.array([30])
    ploidy = 2
    seed = 42
    true_interhomo_dis = np.array([5.])
    alpha_true, beta_true = -3., 4.
    mhs_lambda = 1

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()

    '''X_true = np.zeros((n * ploidy, 3), dtype=float)
    for i in range(X_true.shape[0]):
        X_true[i:, random_state.choice([0, 1, 2])] += 1'''

    X_true = random_state.rand(n * ploidy, 3)

    X_true[n:] -= X_true[n:].mean(axis=0)
    X_true[:n] -= X_true[:n].mean(axis=0)
    begin = end = 0
    for i in range(len(lengths)):
        end += lengths[i]
        X_true[begin:end, 0] += true_interhomo_dis[i]
        begin = end

    dis = euclidean_distances(X_true)
    dis[dis == 0] = np.inf
    counts = beta_true * dis ** alpha_true
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts = np.triu(counts, 1)
    counts = sparse.coo_matrix(counts)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta_true)
    null_counts = [NullCountsMatrix(
        counts=counts, lengths=lengths, ploidy=ploidy, multiscale_factor=1)]

    mhs_k = _mean_interhomolog_counts(counts, lengths=lengths)

    constraint = Constraints(
        counts, lengths=lengths, ploidy=ploidy, multiscale_factor=1,
        constraint_lambdas={'mhs': mhs_lambda},
        constraint_params={'mhs': mhs_k})
    constraint.check()

    alpha, obj, converged, _ = estimate_alpha_beta.estimate_alpha(
        X=X_true, counts=null_counts, alpha_init=alpha_true, lengths=lengths,
        constraints=constraint)

    beta = list(estimate_alpha_beta._estimate_beta(
        X=X_true, counts=counts, alpha=alpha, lengths=lengths,
        verbose=False).values())[0]

    print(alpha, obj, converged, beta)

    assert converged
    assert obj < 1e-6
    assert_allclose(alpha_true, alpha, rtol=1e-2)
    assert_allclose(beta_true, beta, rtol=1e-2)
