import numpy as np
from scipy import optimize
from scipy import sparse
from sklearn.utils import check_random_state
from sklearn.metrics import euclidean_distances
from utils import compute_wish_distances


def MDS_obj(X, distances):
    X = X.reshape(-1, 3)
    dis = euclidean_distances(X)
    X = X.flatten()
    return ((dis - distances)**2).sum()


def MDS_obj_sparse(X, distances):
    X = X.reshape(-1, 3)
    dis = np.sqrt(((X[distances.row] - X[distances.col])**2).sum(axis=1))
    return ((dis - distances.data)**2 / distances.data**2).sum()


def MDS_gradient(X, distances):
    X = X.reshape(-1, 3)
    m, n = X.shape
    tmp = X.repeat(m, axis=0).reshape((m, m, n))
    dif = tmp - tmp.transpose(1, 0, 2)
    dis = euclidean_distances(X).repeat(3, axis=1).flatten()
    distances = distances.repeat(3, axis=1).flatten()
    grad = dif.flatten() * (dis - distances) / dis / distances.data**2
    grad[(distances == 0) | np.isnan(grad)] = 0
    X = X.flatten()
    return grad.reshape((m, m, n)).sum(axis=1).flatten()


def MDS_gradient_sparse(X, distances):
    X = X.reshape(-1, 3)
    dis = np.sqrt(((X[distances.row] - X[distances.col])**2).sum(axis=1))

    grad = ((dis - distances.data) /
            dis / distances.data**2)[:, np.newaxis] * (
        X[distances.row] - X[distances.col])
    grad_ = np.zeros(X.shape)

    for i in range(X.shape[0]):
        grad_[i] += grad[distances.row == i].sum(axis=0)
        grad_[i] -= grad[distances.col == i].sum(axis=0)

    X = X.flatten()
    return grad_.flatten()


def estimate_X(counts, alpha=-3., beta=1., ini=None,
               verbose=0,
               bias=None,
               use_zero_entries=False,
               random_state=None, type="MDS2",
               maxiter=10000):
    n = counts.shape[0]

    random_state = check_random_state(random_state)
    if ini is None or ini == "random":
        ini = 1 - 2 * random_state.rand(n * 3)

    distances = compute_wish_distances(counts, alpha=alpha, beta=beta,
                                       bias=bias)
    results = optimize.fmin_l_bfgs_b(
        MDS_obj_sparse, ini.flatten(),
        MDS_gradient_sparse,
        (distances, ),
        iprint=verbose,
        maxiter=maxiter)
    return results[0].reshape(-1, 3)


class MDS(object):
    """
    """
    def __init__(self, alpha=-3., beta=1.,
                 max_iter=5000, random_state=None, n_init=1, n_jobs=1,
                 precompute_distance="auto", bias=None,
                 init=None, verbose=False):
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.random_state = check_random_state(random_state)
        self.n_init = n_init
        self.n_jobs = n_jobs
        self.init = init
        self.verbose = verbose
        self.bias = bias

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
                        use_zero_entries=False,
                        random_state=self.random_state,
                        bias=self.bias,
                        maxiter=self.max_iter)
        return X_
