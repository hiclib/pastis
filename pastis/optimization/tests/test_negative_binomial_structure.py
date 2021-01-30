import numpy as np
from scipy import sparse
from sklearn.metrics import euclidean_distances
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_almost_equal
from scipy.optimize import check_grad

from pastis import _dispersion as dispersion
from pastis.optimization import negative_binomial_structure


def test_negative_binomial_obj_dense():

    lengths = np.array([5, 5])
    n = lengths.sum()

    random_state = np.random.RandomState(42)
    X = random_state.rand(n, 3)
    dis = euclidean_distances(X)
    alpha, beta = -3, 1

    counts = beta * dis ** alpha
    counts = np.triu(counts)
    counts[np.arange(len(counts)), np.arange(len(counts))] = 0

    obj = negative_binomial_structure.negative_binomial_obj(
        X, counts, alpha=alpha, beta=beta)

    obj_ = negative_binomial_structure.negative_binomial_obj(
        random_state.rand(*X.shape),
        counts, alpha=alpha, beta=beta)
    assert(obj < obj_)

    obj = negative_binomial_structure.negative_binomial_obj(
        X, counts, alpha=alpha, beta=beta, use_zero_counts=True)

    obj_ = negative_binomial_structure.negative_binomial_obj(
        random_state.rand(*X.shape),
        counts, alpha=alpha, beta=beta, use_zero_counts=True)
    assert(obj < obj_)


def test_negative_binomial_obj_sparse():
    n = 10
    random_state = np.random.RandomState(42)
    X = random_state.rand(n, 3)
    dis = euclidean_distances(X)
    alpha, beta = -3, 1

    counts = beta * dis ** alpha
    counts = np.triu(counts)
    counts[np.arange(len(counts)), np.arange(len(counts))] = 0
    counts = sparse.coo_matrix(counts)

    obj = negative_binomial_structure.negative_binomial_obj(
        X, counts, alpha=alpha, beta=beta)

    # Create a random structure
    rand_X = random_state.rand(*X.shape)

    # Compute the objective function for both sparse and dense
    obj_sparse = negative_binomial_structure.negative_binomial_obj(
        rand_X,
        counts, alpha=alpha, beta=beta)
    obj_dense = negative_binomial_structure.negative_binomial_obj(
        rand_X,
        counts.toarray(), alpha=alpha, beta=beta)

    # The objective for a random point should be larger than for the solution
    assert(obj < obj_sparse)
    # The objective in sparse and dense should be equal
    assert_almost_equal(obj_sparse, obj_dense)


def test_negative_binomial_obj_sparse_dispersion_biased():
    n = 10
    random_state = np.random.RandomState(42)
    X = random_state.rand(n, 3)
    dis = euclidean_distances(X)
    alpha, beta = -3, 1

    counts = beta * dis ** alpha
    _, mean, variance, _ = dispersion.compute_mean_variance(
        counts**2,
        np.array([counts.shape[0]]))
    mean, variance = mean[:-1], variance[:-1]

    d = dispersion.ExponentialDispersion()
    d.fit(mean, variance)

    counts = np.triu(counts)
    counts[np.arange(len(counts)), np.arange(len(counts))] = 0
    counts = sparse.coo_matrix(counts)

    obj_sparse = negative_binomial_structure.negative_binomial_obj(
        X, counts, dispersion=d, alpha=alpha, beta=beta)
    obj_dense = negative_binomial_structure.negative_binomial_obj(
        X, counts.toarray(), dispersion=d, alpha=alpha, beta=beta)

    obj_ = negative_binomial_structure.negative_binomial_obj(
        random_state.rand(*X.shape),
        counts, dispersion=d, alpha=alpha, beta=beta)
    assert(obj_sparse < obj_)
    assert_almost_equal(obj_sparse, obj_dense, 6)


def test_negative_binomial_obj_dense_weights():
    n = 50
    lengths = np.array([10, 20, 20, 10])
    random_state = np.random.RandomState(42)
    X = random_state.rand(n, 3)
    dis = euclidean_distances(X)
    alpha, beta = -3, 1

    counts = beta * dis ** alpha
    counts = np.triu(counts)
    counts[np.arange(len(counts)), np.arange(len(counts))] = 0

    obj = negative_binomial_structure.negative_binomial_obj(
        X, counts, alpha=alpha, beta=beta, lengths=lengths)

    obj_ = negative_binomial_structure.negative_binomial_obj(
        random_state.rand(*X.shape),
        counts, alpha=alpha, beta=beta, lengths=lengths)
    assert(obj < obj_)

    obj = negative_binomial_structure.negative_binomial_obj(
        X, counts, alpha=alpha, beta=beta, use_zero_counts=True,
        lengths=lengths)

    obj_ = negative_binomial_structure.negative_binomial_obj(
        random_state.rand(*X.shape),
        counts, alpha=alpha, beta=beta, use_zero_counts=True, lengths=lengths)
    assert(obj < obj_)

    obj = negative_binomial_structure.negative_binomial_obj(
        X, counts, alpha=alpha, beta=beta, weights=1.)

    obj_ = negative_binomial_structure.negative_binomial_obj(
        random_state.rand(*X.shape),
        counts, alpha=alpha, beta=beta, weights=1.)
    assert(obj < obj_)

    obj = negative_binomial_structure.negative_binomial_obj(
        X, counts, alpha=alpha, beta=beta, use_zero_counts=True,
        weights=1.)

    obj_ = negative_binomial_structure.negative_binomial_obj(
        random_state.rand(*X.shape),
        counts, alpha=alpha, beta=beta, use_zero_counts=True, weights=1.)
    assert(obj < obj_)


def test_negative_binomial_obj_weighted_sparse():
    lengths = np.array([10, 20, 20, 10])
    n = lengths.sum()
    random_state = np.random.RandomState(42)
    X = random_state.rand(n, 3)
    dis = euclidean_distances(X)
    alpha, beta = -3, 1

    counts = beta * dis ** alpha
    counts = np.triu(counts)
    counts[np.arange(len(counts)), np.arange(len(counts))] = 0

    obj_dense = negative_binomial_structure.negative_binomial_obj(
        X, counts, alpha=alpha, beta=beta, lengths=lengths, weights=0.1)

    obj_sparse = negative_binomial_structure.negative_binomial_obj(
        X, sparse.coo_matrix(counts),
        alpha=alpha, beta=beta, lengths=lengths, weights=0.1)
    assert_almost_equal(obj_dense, obj_sparse)


def test_negative_binomial_gradient_dense():
    n = 10
    random_state = np.random.RandomState(42)
    X = random_state.rand(n, 3)
    dis = euclidean_distances(X)
    alpha, beta = -3, 1

    counts = beta * dis ** alpha
    counts = np.triu(counts)
    counts[np.arange(len(counts)), np.arange(len(counts))] = 0
    counts = sparse.coo_matrix(counts)

    gradient_dense = negative_binomial_structure.negative_binomial_gradient(
        X, counts.toarray())
    assert_array_almost_equal(np.zeros(gradient_dense.shape), gradient_dense)


def test_negative_binomial_gradient_sparse():
    n = 10
    random_state = np.random.RandomState(42)
    X = random_state.rand(n, 3)
    dis = euclidean_distances(X)
    alpha, beta = -3, 1

    counts = beta * dis ** alpha
    counts = np.triu(counts)
    counts[np.arange(len(counts)), np.arange(len(counts))] = 0
    counts = sparse.coo_matrix(counts)

    gradient_sparse = negative_binomial_structure.negative_binomial_gradient(
        X, counts)

    rand_X = random_state.randn(*X.shape)
    assert_array_almost_equal(np.zeros(gradient_sparse.shape), gradient_sparse)

    rand_grad_sparse = negative_binomial_structure.negative_binomial_gradient(
        rand_X, counts)
    rand_grad_dense = negative_binomial_structure.negative_binomial_gradient(
        rand_X, counts.toarray())
    assert_array_almost_equal(rand_grad_sparse, rand_grad_dense)

    lengths = np.array([5, 5])
    rand_grad_sw = negative_binomial_structure.negative_binomial_gradient(
        rand_X, counts, lengths=lengths, weights=0.1)
    rand_grad_dw = negative_binomial_structure.negative_binomial_gradient(
        rand_X, counts.toarray(), lengths=lengths, weights=0.1)
    assert_array_almost_equal(rand_grad_sw, rand_grad_dw)


def test_negative_binomial_gradient_sparse_dispersed():
    n = 10
    random_state = np.random.RandomState(42)
    X = random_state.rand(n, 3)
    dis = euclidean_distances(X)
    alpha, beta = -3, 1

    fdis = beta * dis**alpha
    fdis[np.isinf(fdis)] = 0
    dispersion_estimated = fdis + fdis ** 2
    p = fdis / (fdis + dispersion_estimated)

    # counts = random_state.negative_binomial(dispersion, 1 - p)
    # counts = np.triu(counts)
    counts = np.triu(fdis)
    counts[np.arange(len(counts)), np.arange(len(counts))] = 0
    counts = sparse.coo_matrix(counts, dtype=float)

    _, mean, variance, _ = dispersion.compute_mean_variance(
        counts,
        np.array([counts.shape[0]]))
    mean, variance = mean[:-1], variance[:-1]
    d = dispersion.ExponentialDispersion()
    d.fit(mean, variance)

    gradient_sparse = negative_binomial_structure.negative_binomial_gradient(
        X, counts, dispersion=d)
    gradient_dense = negative_binomial_structure.negative_binomial_gradient(
        X, counts.toarray(), dispersion=d)

    assert_array_almost_equal(gradient_dense, gradient_sparse)
    assert_array_almost_equal(
       np.zeros(gradient_sparse.shape), gradient_sparse, -5)


def test_estimate_X():
    n = 25
    random_state = np.random.RandomState(42)
    X_true = random_state.rand(n, 3)
    dis = euclidean_distances(X_true)
    alpha, beta = -3, 1

    counts = beta * dis ** alpha
    counts = np.triu(counts)
    counts[np.arange(len(counts)), np.arange(len(counts))] = 0
    counts = sparse.coo_matrix(counts)

    X = negative_binomial_structure.estimate_X(counts, alpha, beta,
                                               factr=10,
                                               random_state=random_state)
    assert_array_almost_equal(dis,
                              euclidean_distances(X), 2)

    X = negative_binomial_structure.estimate_X(counts.toarray(), alpha, beta,
                                               factr=10,
                                               use_zero_entries=True,
                                               random_state=random_state)

    # FIXME need to rescale the distance to minimize the mean squared error
    # between the two
    # estimated_distances = euclidean_distances(X)
    # error = ((dis - euclidean_distances(X))**2).mean()
    # assert error < 1e-2


def test_estimate_X_biased_dispersion():
    n = 50
    random_state = np.random.RandomState(42)
    X_true = random_state.rand(n, 3) * 10
    dis = euclidean_distances(X_true)
    alpha, beta = -3, 1

    fdis = beta * dis ** alpha
    fdis[np.isinf(fdis)] = 1
    disp = fdis + fdis ** 2
    p = fdis / (fdis + disp)

    counts = random_state.negative_binomial(disp, 1 - p)
    counts = np.triu(counts)
    counts[np.arange(len(counts)), np.arange(len(counts))] = 0
    counts = sparse.coo_matrix(counts, dtype=np.float)

    lengths = np.array([counts.shape[0]])
    _, mean, variance, _ = dispersion.compute_mean_variance(counts, lengths)
    mean, variance = mean[:-1], variance[:-1]
    d = dispersion.ExponentialDispersion()
    d.fit(mean, variance)

    X = negative_binomial_structure.estimate_X(counts, alpha, beta,
                                               dispersion=d,
                                               random_state=random_state)
    # XXX Need to rescale the distance before computing the error.
    # estimated_distances = euclidean_distances(X)
    # distance_error = ((dis - estimated_distances)**2).mean()
    # assert distance_error < 1e-2


def test_check_grad():
    n = 50
    random_state = np.random.RandomState(42)
    X_true = random_state.rand(n, 3)
    dis = euclidean_distances(X_true)
    alpha, beta = -3, 1

    fdis = beta * dis ** alpha
    fdis[np.isinf(fdis)] = 1
    disp = fdis + fdis ** 2
    p = fdis / (fdis + disp)

    counts = random_state.negative_binomial(disp, 1 - p)
    counts = np.triu(counts, 1)
    counts[np.arange(len(counts)), np.arange(len(counts))] = 0
    user_data = (n, counts, alpha, beta, None, None, False, None, None)

    eps = check_grad(negative_binomial_structure.eval_f,
                     negative_binomial_structure.eval_grad_f,
                     random_state.randn(*X_true.shape).flatten(),
                     user_data)

    user_data = (n, sparse.coo_matrix(counts),
                 alpha, beta, None, None, False, None, None)

    eps_ = check_grad(negative_binomial_structure.eval_f,
                      negative_binomial_structure.eval_grad_f,
                      random_state.randn(*X_true.shape).flatten(),
                      user_data)

    assert eps < 1e-1
    assert eps_ < 1e-1
