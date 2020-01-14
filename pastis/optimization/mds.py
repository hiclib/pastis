import numpy as np
import warnings
from scipy import optimize
from scipy import sparse
from sklearn.utils import check_random_state
from sklearn.metrics import euclidean_distances
from sklearn.isotonic import IsotonicRegression

from .utils import compute_wish_distances


def MDS_obj(X, distances):
    if sparse.issparse(distances):
        return MDS_obj_sparse(X, distances)
    else:
        return MDS_obj_dense(X, distances)


def MDS_obj_dense(X, distances):
    X = X.reshape(-1, 3)
    dis = euclidean_distances(X)
    X = X.flatten()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obj = 1. / distances ** 2 * (dis - distances) ** 2
    return obj[np.invert(np.isnan(obj) | np.isinf(obj))].sum()


def MDS_obj_sparse(X, distances):
    X = X.reshape(-1, 3)
    dis = np.sqrt(((X[distances.row] - X[distances.col])**2).sum(axis=1))
    return ((dis - distances.data)**2 / distances.data**2).sum()


def MDS_gradient(X, distances):
    if sparse.issparse(distances):
        return MDS_gradient_sparse(X, distances)
    else:
        return MDS_gradient_dense(X, distances)


def MDS_gradient_dense(X, distances):
    X = X.reshape(-1, 3)
    m, n = X.shape
    tmp = X.repeat(m, axis=0).reshape((m, m, n))
    dif = tmp - tmp.transpose(1, 0, 2)
    dis = euclidean_distances(X).repeat(3, axis=1).flatten()
    distances = distances.repeat(3, axis=1).flatten()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grad = 2 * dif.flatten() * (dis - distances) / dis / distances**2
    grad[(distances == 0) | np.isnan(grad)] = 0
    X = X.flatten()
    return grad.reshape((m, m, n)).sum(axis=1).flatten()


def MDS_gradient_sparse(X, distances):
    X = X.reshape(-1, 3)
    dis = np.sqrt(((X[distances.row] - X[distances.col])**2).sum(axis=1))

    grad = 2 * ((dis - distances.data) / dis /
                distances.data**2)[:, np.newaxis] * (
        X[distances.row] - X[distances.col])
    grad_ = np.zeros(X.shape)

    for i in range(X.shape[0]):
        grad_[i] += grad[distances.row == i].sum(axis=0)
        grad_[i] -= grad[distances.col == i].sum(axis=0)

    X = X.flatten()
    return grad_.flatten()


def estimate_X(counts, alpha=-3., beta=1., ini=None,
               verbose=0,
               use_zero_entries=False,
               precompute_distances=False,
               bias=None,
               random_state=None, type="MDS2",
               factr=1e12,
               maxiter=10000):
    n = counts.shape[0]

    random_state = check_random_state(random_state)
    if ini is None or ini == "random":
        ini = 1 - 2 * random_state.rand(n * 3)
    if not precompute_distances or precompute_distances == "auto":
        distances = compute_wish_distances(counts, alpha=alpha, beta=beta,
                                           bias=bias)
    else:
        if bias is not None:
            counts /= bias
            counts /= bias.T
        distances = counts
    results = optimize.fmin_l_bfgs_b(
        MDS_obj, ini.flatten(),
        MDS_gradient,
        (distances, ),
        iprint=verbose,
        factr=factr,
        maxiter=maxiter)
    return results[0].reshape(-1, 3)


class MDS(object):
    """
    """
    def __init__(self, alpha=-3., beta=1.,
                 max_iter=5000, random_state=None, n_init=1, n_jobs=1,
                 precompute_distances="auto", bias=None,
                 init=None, verbose=False, factr=1e12):
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.random_state = check_random_state(random_state)
        self.precompute_distances = precompute_distances
        self.n_init = n_init
        self.n_jobs = n_jobs
        self.init = init
        self.verbose = verbose
        self.bias = bias
        self.factr = factr

    def fit(self, counts, lengths=None):
        """

        """
        if not sparse.isspmatrix_coo(counts):
            counts = sparse.coo_matrix(counts)

        X_ = estimate_X(counts,
                        alpha=self.alpha,
                        beta=self.beta,
                        ini=self.init,
                        verbose=self.verbose,
                        precompute_distances=self.precompute_distances,
                        use_zero_entries=False,
                        random_state=self.random_state,
                        bias=self.bias,
                        factr=self.factr,
                        maxiter=self.max_iter)
        return X_


class NMDS(object):
    """
    """
    def __init__(self, alpha=-3., beta=1.,
                 max_iter=5000, random_state=None, n_init=1, n_jobs=1,
                 precompute_distances="auto", bias=None,
                 init=None, verbose=False, max_iter_outer=5, factr=1e12):
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.random_state = check_random_state(random_state)
        self.precompute_distances = precompute_distances
        self.n_init = n_init
        self.n_jobs = n_jobs
        self.init = init
        self.verbose = verbose
        self.bias = bias
        self.factr = factr
        self.max_iter_outer = max_iter_outer

    def fit(self, counts, lengths=None):
        """

        """
        if not sparse.isspmatrix_coo(counts):
            counts = sparse.coo_matrix(counts)

        for i in range(self.max_iter_outer):
            if i == 0:
                X = estimate_X(counts,
                               alpha=self.alpha,
                               beta=self.beta,
                               ini=self.init,
                               verbose=self.verbose,
                               use_zero_entries=False,
                               random_state=self.random_state,
                               bias=self.bias,
                               factr=self.factr,
                               maxiter=self.max_iter)
            else:
                ir = IsotonicRegression()
                dis = np.sqrt(
                    ((X[counts.row] - X[counts.col])**2).sum(axis=1))
                wish_distances = ir.fit_transform(
                    1. / counts.data,
                    dis)
                X = estimate_X(sparse.coo_matrix((wish_distances,
                                                  (counts.row,
                                                   counts.col))),
                               alpha=self.alpha,
                               beta=self.beta,
                               ini=X,
                               verbose=self.verbose,
                               use_zero_entries=False,
                               precompute_distances="precomputed",
                               random_state=self.random_state,
                               factr=self.factr,
                               maxiter=self.max_iter)
        print("writing wish distances")
        return X
