import numpy as np
from scipy import sparse
from sklearn.metrics import euclidean_distances
from numpy.testing import assert_array_almost_equal
from pastis.optimization import negative_binomial_structure


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

    obj_ = negative_binomial_structure.negative_binomial_obj(
        random_state.rand(*X.shape),
        counts, alpha=alpha, beta=beta)
    assert(obj < obj_)


def test_negative_binomial_obj_sparse_dispersion_biased():
    n = 10
    random_state = np.random.RandomState(42)
    X = random_state.rand(n, 3)
    dis = euclidean_distances(X)
    alpha, beta = -3, 1

    counts = beta * dis ** alpha

    return True
    from minorswing import dispersion
    mean, variance = dispersion.compute_mean_variance(
        counts**2,
        np.array([counts.shape[0]]))
    mean, variance = mean[:-1], variance[:-1]
    d = dispersion.Dispersion()
    d.fit(mean, variance)

    counts = np.triu(counts)
    counts[np.arange(len(counts)), np.arange(len(counts))] = 0
    counts = sparse.coo_matrix(counts)

    obj = negative_binomial_structure.negative_binomial_obj(
        X, counts, dispersion=d, alpha=alpha, beta=beta)

    obj_ = negative_binomial_structure.negative_binomial_obj(
        random_state.rand(*X.shape),
        counts, dispersion=d, alpha=alpha, beta=beta)
    assert(obj < obj_)


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
    assert_array_almost_equal(np.zeros(gradient_sparse.shape), gradient_sparse)


def test_negative_binomial_gradient_sparse_dispersed():
    n = 10
    random_state = np.random.RandomState(42)
    X = random_state.rand(n, 3)
    dis = euclidean_distances(X)
    alpha, beta = -3, 1

    fdis = beta * dis**alpha
    fdis[np.isinf(fdis)] = 1
    dispersion = fdis + fdis ** 2
    p = fdis / (fdis + dispersion)

    counts = random_state.negative_binomial(dispersion, 1 - p)
    counts = np.triu(counts)
    counts[np.arange(len(counts)), np.arange(len(counts))] = 0
    counts = sparse.coo_matrix(counts, dtype=float)
    return True
    # from minorswing import dispersion
    mean, variance = dispersion.compute_mean_variance(
        counts,
        np.array([counts.shape[0]]))
    mean, variance = mean[:-1], variance[:-1]
    d = dispersion.DispersionPolynomial()
    d.fit(mean, variance)

    gradient_sparse = negative_binomial_structure.negative_binomial_gradient(
        X, counts, dispersion=d)
    #assert_array_almost_equal(np.zeros(gradient_sparse.shape), gradient_sparse)


def test_estimate_X():
    n = 50
    random_state = np.random.RandomState(42)
    X_true = random_state.rand(n, 3)
    dis = euclidean_distances(X_true)
    alpha, beta = -3, 1

    counts = beta * dis ** alpha
    counts = np.triu(counts)
    counts[np.arange(len(counts)), np.arange(len(counts))] = 0
    counts = sparse.coo_matrix(counts)

    X = negative_binomial_structure.estimate_X(counts, alpha, beta,
                                               random_state=random_state)
    assert_array_almost_equal(dis,
                              euclidean_distances(X), 2)


def test_estimate_X_biased_dispersion():
    n = 50
    random_state = np.random.RandomState(42)
    X_true = random_state.rand(n, 3)
    dis = euclidean_distances(X_true)
    alpha, beta = -3, 1

    fdis = beta * dis ** alpha
    fdis[np.isinf(fdis)] = 1
    dispersion = fdis + fdis ** 2
    p = fdis / (fdis + dispersion)

    counts = random_state.negative_binomial(dispersion, 1 - p)
    counts = np.triu(counts)
    counts[np.arange(len(counts)), np.arange(len(counts))] = 0
    counts = sparse.coo_matrix(counts, dtype=np.float)

    lengths = np.array([counts.shape[0]])
    return True
    from minorswing import dispersion
    mean, variance = dispersion.compute_mean_variance(counts, lengths)
    mean, variance = mean[:-1], variance[:-1]
    d = dispersion.DispersionPolynomial()
    d.fit(mean, variance)

    X = negative_binomial_structure.estimate_X(counts, alpha, beta,
                                               dispersion=d,
                                               random_state=random_state)
    #assert_array_almost_equal(dis, euclidean_distances(X))
