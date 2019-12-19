import numpy as np
from sklearn.metrics import euclidean_distances
from scipy import sparse


class ConstantDispersion(object):
    def __init__(self, coef=7):
        self.coef = coef

    def fit(self, X, y):
        return self

    def predict(self, X):
        return self.coef * np.ones(X.shape)

    def derivate(self, X):
        return np.zeros(X.shape)


def compute_wish_distances(counts, alpha=-3., beta=1., bias=None):
    """
    Computes wish distances from a counts matrix

    Parameters
    ----------
    counts : ndarray
        Interaction counts matrix

    alpha : float, optional, default: -3
        Coefficient of the power law

    beta : float, optional, default: 1
        Scaling factor

    Returns
    -------
    wish_distances
    """
    if beta == 0:
        raise ValueError("beta cannot be equal to 0.")
    counts = counts.copy()
    if sparse.issparse(counts):
        if not sparse.isspmatrix_coo(counts):
            counts = counts.tocoo()
        if bias is not None:
            bias = bias.flatten()
            counts.data /= bias[counts.row] * bias[counts.col]
        wish_distances = counts / beta
        wish_distances.data[wish_distances.data != 0] **= 1. / alpha
        return wish_distances
    else:
        wish_distances = counts.copy() / beta
        wish_distances[wish_distances != 0] **= 1. / alpha

        return wish_distances


def eval_no_f(x, user_data=None):
    """
    Evaluate the object function (no objective function).
    """
    return 0.


def eval_grad_no_f(X, user_data=None):
    m, n, counts, alpha, beta, d = user_data
    return np.zeros(n * m)


def eval_g_no(x, user_data=None):
    return np.array([0, 0.])


def eval_jac_g_no(x, flag, user_data=None):
    if flag:
        return 0, 0
    return np.array([0., 0.])


def eval_g(x, user_data=None):
    """
    Computes the constraints
    """

    m, n, wish_dist, alpha, beta, d = user_data

    x = x.reshape((m, n))
    dis = euclidean_distances(x)
    dis = dis ** 2
    mask = np.invert(np.tri(m, dtype=np.bool))
    g = np.concatenate([dis[mask].flatten(), ((x - d) ** 2).sum(axis=1)])
    return g


def eval_jac_g(x, flag, user_data=None):
    """
    Computes the jacobian for the constraints mentionned in duan-et-al
    """

    m, n, wish_dist, alpha, beta, d = user_data

    if flag:
        ncon = m * (m - 1) / 2
        row = np.arange(ncon).repeat(2 * n)

        tmp = np.arange(n).repeat(m - 1)
        tmp = tmp.reshape((n, m - 1))
        tmp = tmp.T
        tmp1 = np.arange(n, m * n)
        tmp1 = tmp1.reshape((m - 1, n))
        tmp = np.concatenate((tmp, tmp1), axis=1)
        tmp1 = tmp.copy()
        for it in range(m):
            tmp += n
            tmp = tmp[:-1]
            tmp1 = np.concatenate((tmp1, tmp))

        # The second part of the jacobian is the restrictions on the distances
        # to the origin and/or the distances to the SPB/nucleolus center
        row_2 = np.arange(m).repeat(n)
        col = np.arange(n * m)

        col = np.concatenate([tmp1.flatten(), col])
        row = np.concatenate([row, row_2])
        return row.flatten(), col.flatten()
    else:
        x = x.reshape((m, n))
        tmp = x.repeat(m, axis=0).reshape((m, m, n))
        dif = tmp - tmp.transpose(1, 0, 2)
        mask = np.invert(np.tri(m, dtype=np.bool))
        dif = dif[mask]
        jac = 2 * np.concatenate((dif, - dif), axis=1).flatten()

        # The second part of the jacobian is the restrictions on the distances
        # to the origin and/or the distances to the SPB/nucleolus center
        jac2 = 2 * (x - d).flatten()
        return np.concatenate([jac, jac2]).flatten()


def eval_h(x, lagrange, obj_factor, flag, user_data=None):
    """
    """
    return False
