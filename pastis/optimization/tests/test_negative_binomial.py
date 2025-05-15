import numpy as np
from scipy import sparse
from nose.tools import assert_almost_equal
from numpy.testing import assert_array_almost_equal
from sklearn.metrics import euclidean_distances
from pastis.optimization import negative_binomial
from scipy.optimize import check_grad


def test_negative_binomial_obj_sparse():
    n = 10
    random_state = np.random.RandomState(42)
    X = random_state.rand(n, 3)
    counts = random_state.randn(n, n)
    counts = np.triu(counts)
    counts[np.arange(len(counts)), np.arange(len(counts))] = 0
    obj_dense = negative_binomial.negative_binomial_obj(X, counts)
    obj_sparse = negative_binomial.negative_binomial_obj(
        X, sparse.coo_matrix(counts))

    assert_almost_equal(obj_dense, obj_sparse)


def test_negative_binomial_obj_sparse_biased():
    random_state = np.random.RandomState(seed=42)
    n = 50
    X = random_state.rand(n, 3)
    bias = 0.1 + random_state.rand(n)
    bias = bias.reshape(n, 1)

    counts = euclidean_distances(X)**(-3)
    counts[np.isinf(counts) | np.isnan(counts)] = 0
    counts *= bias * bias.T

    counts_dense = np.triu(counts, 1)
    counts_sparse = sparse.coo_matrix(np.triu(counts, 1))
    exp_dense = negative_binomial.negative_binomial_obj(
        X, counts_dense, -3, bias=bias)
    exp_sparse = negative_binomial.negative_binomial_obj(
        X, counts_sparse, -3, bias=bias)
    assert_almost_equal(exp_dense, exp_sparse)


def test_negative_binomial_grad_dense():
    n = 10
    random_state = np.random.RandomState(seed=42)
    X = random_state.rand(n, 3)
    dis = euclidean_distances(X)
    alpha, beta = -3, 1

    counts = beta * dis ** alpha
    grad_dense = negative_binomial.negative_binomial_gradient(
        X, counts, use_zero_counts=True)
    # Checking that the gradient at the solution is 0
    assert_array_almost_equal(grad_dense, 0)


def test_negative_binomial_grad_sparse():
    n = 10
    random_state = np.random.RandomState(seed=42)
    X = random_state.rand(n, 3)

    dis = euclidean_distances(X)
    alpha, beta = -3, 1

    counts = beta * dis ** alpha

    grad_dense = negative_binomial.negative_binomial_gradient(
        X, counts)
    grad_sparse = negative_binomial.negative_binomial_gradient(
        X, sparse.coo_matrix(np.triu(counts, 1)))
    assert_array_almost_equal(grad_dense, grad_sparse)
    assert_array_almost_equal(np.zeros(grad_dense.shape), grad_dense)


def test_negative_binomial_grad_biased():
    random_state = np.random.RandomState(seed=42)
    n = 50
    X = random_state.rand(n, 3)
    bias = 0.1 + random_state.rand(n)
    bias = bias.reshape(n, 1)

    counts = euclidean_distances(X)**(-3)
    counts[np.isinf(counts) | np.isnan(counts)] = 0
    counts *= bias.T * bias

    counts_dense = counts
    counts_sparse = sparse.coo_matrix(np.triu(counts, 1))
    grad_dense = negative_binomial.negative_binomial_gradient(
        X, counts_dense,
        bias=bias)
    grad_sparse = negative_binomial.negative_binomial_gradient(
        X, counts_sparse,
        bias=bias)
    assert_array_almost_equal(grad_dense, grad_sparse)
    assert_array_almost_equal(np.array([0, 0.]), grad_dense)


def test_estimate_alpha_beta_biased():
    n = 100
    random_state = np.random.RandomState(seed=42)
    X = random_state.rand(n, 3)
    dis = euclidean_distances(X)
    bias = 0.1 + random_state.rand(n)
    bias = bias.reshape(n, 1)
    beta_true, alpha_true = 1., -3.5
    counts = beta_true * dis ** alpha_true
    counts *= bias * bias.T
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts[np.arange(len(counts)), np.arange(len(counts))] = 0
    counts = sparse.coo_matrix(np.triu(counts))

    alpha, beta = negative_binomial.estimate_alpha_beta(
        counts, X, bias=bias, random_state=random_state,
        use_zero_entries=True, infer_beta=False)

    assert_almost_equal(alpha_true, alpha, 3)
    assert_almost_equal(beta_true, beta, 3)

    alpha, beta = negative_binomial.estimate_alpha_beta(
        sparse.coo_matrix(counts), X, bias=bias,
        random_state=random_state, infer_beta=False,
        use_zero_entries=True)
    assert_almost_equal(alpha_true, alpha, 3)
    assert_almost_equal(beta_true, beta, 4)


def test_estimate_alpha_beta():
    n = 100
    random_state = np.random.RandomState(seed=42)
    X = 5 * random_state.rand(n, 3)
    dis = euclidean_distances(X)
    beta_true, alpha_true = 1., -3.5
    counts = beta_true * dis ** alpha_true
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts[np.arange(len(counts)), np.arange(len(counts))] = 0
    counts = sparse.coo_matrix(np.triu(counts))

    alpha, beta = negative_binomial.estimate_alpha_beta(
        sparse.coo_matrix(counts), X,
        random_state=random_state, infer_beta=False,
        use_zero_entries=True)
    assert_almost_equal(alpha_true, alpha, 3)


def test_estimate_alpha_beta_dispersion():
    n = 100
    random_state = np.random.RandomState(seed=42)
    X = 5 * random_state.rand(n, 3)
    dis = euclidean_distances(X)
    beta_true, alpha_true = 1., -3.5
    counts = beta_true * dis ** alpha_true
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts[np.arange(len(counts)), np.arange(len(counts))] = 0
    counts = sparse.coo_matrix(np.triu(counts))

    alpha, beta = negative_binomial.estimate_alpha_beta(
        counts, X, random_state=random_state,
        infer_beta=False, use_zero_entries=True)
    assert_almost_equal(alpha_true, alpha, 3)
    # assert_almost_equal(beta_true, beta, 4)


def test_check_grad():
    random_state = np.random.RandomState(42)
    n = 30
    X = random_state.rand(n, 3)
    dis = euclidean_distances(X) ** -3
    udata = (n, 3, sparse.coo_matrix(np.triu(dis, 1)), X,
             None, None, False, None, True)
    init = np.array([-3, 1])
    eps = check_grad(negative_binomial.eval_f,
                     negative_binomial.eval_grad_f,
                     init,
                     udata)
    assert eps < 1e-3

    udata = (n, 3, dis, X, None, None, True, None, True)
    init = np.array([-3, 1])
    eps = check_grad(negative_binomial.eval_f,
                     negative_binomial.eval_grad_f,
                     init,
                     udata)
    assert eps < 1e-3

    bias = 0.1 + random_state.rand(n)
    bias = bias.reshape(n, 1)

    udata = (n, 3, dis, X, None, bias, True, None, True)
    init = np.array([-3, 1])
    eps = check_grad(negative_binomial.eval_f,
                     negative_binomial.eval_grad_f,
                     init,
                     udata)
    assert eps < 1e-3
