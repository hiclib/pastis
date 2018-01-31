import numpy as np
from scipy import sparse
from sklearn.metrics import euclidean_distances
from numpy.testing import assert_array_almost_equal
from pastis.optimization import poisson_model


def test_poisson_exp():
    random_state = np.random.RandomState(seed=42)
    n = 50
    X = random_state.rand(n, 3)
    counts = euclidean_distances(X)**(-3)
    counts[np.isinf(counts) | np.isnan(counts)] = 0
    eps = poisson_model.poisson_exp(X, counts, -2)
    assert eps < 1e-6


def test_poisson_exp_biased():
    random_state = np.random.RandomState(seed=42)
    n = 50
    bias = 0.1 + random_state.rand(n)
    bias = bias.reshape(n, 1)
    X = random_state.rand(n, 3)
    counts = euclidean_distances(X)**(-3)
    counts *= bias * bias.T
    counts[np.isinf(counts) | np.isnan(counts)] = 0
    eps = poisson_model.poisson_exp(X, counts, -2, bias=bias)
    assert eps < 1e-6


def test_poisson_exp_sparse():
    random_state = np.random.RandomState(seed=42)
    n = 50
    X = random_state.rand(n, 3)
    counts = euclidean_distances(X)**(-3)
    counts[np.isinf(counts) | np.isnan(counts)] = 0
    counts_dense = counts
    counts_sparse = sparse.coo_matrix(np.triu(counts))
    exp_dense = poisson_model.poisson_exp(X, counts_dense, -3,
                                          use_empty_entries=False)
    exp_sparse = poisson_model.poisson_exp(X, counts_sparse, -3,
                                           use_empty_entries=False)
    assert_array_almost_equal(exp_dense, exp_sparse)


def test_poisson_exp_sparse_biased():
    random_state = np.random.RandomState(seed=42)
    n = 50
    X = random_state.rand(n, 3)
    bias = 0.1 + random_state.rand(n)
    bias = bias.reshape(n, 1)

    counts = euclidean_distances(X)**(-3)
    counts[np.isinf(counts) | np.isnan(counts)] = 0
    counts *= bias * bias.T

    counts_dense = counts
    counts_sparse = sparse.coo_matrix(np.triu(counts))
    exp_dense = poisson_model.poisson_exp(X, counts_dense, -3, bias=bias,
                                          use_empty_entries=False)
    exp_sparse = poisson_model.poisson_exp(X, counts_sparse, -3, bias=bias,
                                           use_empty_entries=False)
    assert_array_almost_equal(exp_dense, exp_sparse)


def test_gradient_poisson_exp_sparse():
    random_state = np.random.RandomState(seed=42)
    n = 50
    X = random_state.rand(n, 3)
    counts = euclidean_distances(X)**(-3)
    counts[np.isinf(counts) | np.isnan(counts)] = 0
    counts_dense = counts
    counts_sparse = sparse.coo_matrix(np.triu(counts))
    grad_dense = poisson_model.gradient_poisson_exp(X, counts_dense, -3,
                                                    beta=1,
                                                    use_empty_entries=False)
    grad_sparse = poisson_model.gradient_poisson_exp(X, counts_sparse, -3,
                                                     beta=1,
                                                     use_empty_entries=False)
    assert_array_almost_equal(grad_dense, grad_sparse)


def test_gradient_poisson_exp_sparse_biased():
    random_state = np.random.RandomState(seed=42)
    n = 50
    X = random_state.rand(n, 3)
    bias = 0.1 + random_state.rand(n)
    bias = bias.reshape(n, 1)

    counts = euclidean_distances(X)**(-3)
    counts[np.isinf(counts) | np.isnan(counts)] = 0
    counts *= bias * bias.T

    counts_dense = counts
    counts_sparse = sparse.coo_matrix(np.triu(counts))
    grad_dense = poisson_model.gradient_poisson_exp(X, counts_dense, -3,
                                                    bias=bias,
                                                    beta=1,
                                                    use_empty_entries=False)
    grad_sparse = poisson_model.gradient_poisson_exp(X, counts_sparse, -3,
                                                     bias=bias,
                                                     beta=1,
                                                     use_empty_entries=False)
    assert_array_almost_equal(np.array([0]), grad_dense)
    assert_array_almost_equal(grad_dense, grad_sparse)


def test_estimate_alpha_beta():
    n = 100
    random_state = np.random.RandomState(seed=42)
    X = random_state.rand(n, 3)
    dis = euclidean_distances(X)
    beta_true, alpha_true = 2., -3.
    counts = beta_true * dis ** alpha_true
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts = np.triu(counts)

    alpha, beta = poisson_model.estimate_alpha_beta(
        sparse.coo_matrix(counts), X)
    assert_array_almost_equal(alpha_true, alpha, 5)
    assert_array_almost_equal(beta_true, beta, 5)


def test_estimate_alpha_beta_biased():
    n = 100
    random_state = np.random.RandomState(seed=42)
    X = random_state.rand(n, 3)
    dis = euclidean_distances(X)
    bias = 0.1 + random_state.rand(n)
    bias = bias.reshape(n, 1)
    beta_true, alpha_true = 2., -3.5
    counts = beta_true * dis ** alpha_true
    counts *= bias * bias.T
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts = np.triu(counts)

    alpha, beta = poisson_model.estimate_alpha_beta(
        sparse.coo_matrix(counts), X, bias=bias)
    assert_array_almost_equal(alpha_true, alpha, 5)
    assert_array_almost_equal(beta_true, beta, 4)

