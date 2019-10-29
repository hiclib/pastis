import numpy as np
from scipy import sparse
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_almost_equal
from sklearn.metrics import euclidean_distances
from pastis.optimization import negative_binomial


def test_negative_binomial_obj_sparse():
    n = 10
    random_state = np.random.RandomState(42)
    X = random_state.rand(n, 3)
    counts = random_state.randn(n, n)
    counts = np.triu(counts)
    counts[np.arange(len(counts)), np.arange(len(counts))] = 0
    # obj_dense = negative_binomial.negative_binomial_obj(X, counts)
    obj_sparse = negative_binomial.negative_binomial_obj(
        X, sparse.coo_matrix(counts))

    # assert_almost_equal(obj_dense, obj_sparse)


def test_negative_binomial_obj_sparse_biased():
    random_state = np.random.RandomState(seed=42)
    n = 50
    X = random_state.rand(n, 3)
    bias = 0.1 + random_state.rand(n)
    bias = bias.reshape(n, 1)

    counts = euclidean_distances(X)**(-3)
    counts[np.isinf(counts) | np.isnan(counts)] = 0
    counts *= bias * bias.T

    counts_dense = np.triu(counts)
    counts_sparse = sparse.coo_matrix(np.triu(counts))
    #exp_dense = negative_binomial.negative_binomial_obj(
    #    X, counts_dense, -3, bias=bias)
    exp_sparse = negative_binomial.negative_binomial_obj(
        X, counts_sparse, -3, bias=bias)
    #assert_almost_equal(exp_dense, exp_sparse)


def test_negative_binomial_grad_sparse():
    n = 10
    random_state = np.random.RandomState(seed=42)
    X = random_state.rand(n, 3)

    dis = euclidean_distances(X)
    alpha, beta = -3, 1

    counts = beta * dis ** alpha
    counts = np.triu(counts)
    counts[np.arange(len(counts)), np.arange(len(counts))] = 0
    counts = sparse.coo_matrix(counts)

    # grad_dense = negative_binomial.negative_binomial_gradient(X, counts)
    grad_sparse = negative_binomial.negative_binomial_gradient(
        X, sparse.coo_matrix(counts))
    # assert_array_almost_equal(grad_dense, grad_sparse)
    assert_array_almost_equal(np.zeros(grad_sparse.shape),
                              grad_sparse)


def test_negative_binomial_grad_biased():
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
    # grad_dense = negative_binomial.negative_binomial_gradient(
    #    X, counts_dense, -3,
    #    bias=bias,
    #    beta=1)
    grad_sparse = negative_binomial.negative_binomial_gradient(
        X, counts_sparse, -3,
        bias=bias,
        beta=1)
    assert_array_almost_equal(np.array([0, 0.]), grad_sparse)
    # assert_array_almost_equal(grad_dense, grad_sparse)


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
    counts[np.arange(len(counts)), np.arange(len(counts))] = 0
    counts = sparse.coo_matrix(np.triu(counts))

    alpha, beta = negative_binomial.estimate_alpha_beta(
        counts, X, bias=bias, random_state=random_state)
    assert_almost_equal(alpha_true, alpha, 3)
    assert_almost_equal(beta_true, beta, 3)

    alpha, beta = negative_binomial.estimate_alpha_beta(
        sparse.coo_matrix(counts), X, bias=bias,
        random_state=random_state)
    assert_almost_equal(alpha_true, alpha, 5)
    assert_almost_equal(beta_true, beta, 5)


def test_estimate_alpha_beta():
    n = 100
    random_state = np.random.RandomState(seed=42)
    X = 5 * random_state.rand(n, 3)
    dis = euclidean_distances(X)
    beta_true, alpha_true = 2., -3.5
    counts = beta_true * dis ** alpha_true
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts[np.arange(len(counts)), np.arange(len(counts))] = 0
    counts = sparse.coo_matrix(np.triu(counts))

    alpha, beta = negative_binomial.estimate_alpha_beta(
        sparse.coo_matrix(counts), X, random_state=random_state)
    assert_almost_equal(alpha_true, alpha, 3)
    assert_almost_equal(beta_true, beta, 4)
