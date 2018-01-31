import numpy as np
from scipy import sparse
from scipy import optimize
from sklearn.metrics import euclidean_distances
from sklearn.utils import check_random_state

"""Utility method for the pyipopt solver, for the Poisson Model
"""

VERBOSE = False
niter = 0


def poisson_exp(X, counts, alpha, bias=None,
                beta=None, use_empty_entries=True):
    """
    Computes the log likelihood of the poisson exponential model.

    Parameters
    ----------
    X : ndarray
        3D positions

    counts : n * n ndarray
        of interaction frequencies

    alpha : float
        parameter of the expential law

    beta : float, optional, default None
        onstant. If is set to None, it will be computed using the maximum log
        likelihood knowing alpha.

    use_empty_entries : boolean, optional, default False
        whether to use zeroes entries as information or not

    Returns
    -------
    ll : float
        log likelihood
    """
    if VERBOSE:
        print("Poisson power law model : computation of the log likelihood")
    if bias is None:
        bias = np.ones((counts.shape[0], 1))
    if sparse.issparse(counts):
        ll = _poisson_exp_sparse(X, counts, alpha, bias=bias, beta=beta,
                                 use_empty_entries=use_empty_entries)
    else:
        ll = _poisson_exp_dense(X, counts, alpha, bias=bias, beta=beta,
                                use_empty_entries=use_empty_entries)
    return ll


def _poisson_exp_dense(X, counts, alpha, bias,
                       beta=None, use_empty_entries=False):
    m, n = X.shape
    d = euclidean_distances(X)
    if use_empty_entries:
        mask = (np.invert(np.tri(m, dtype=np.bool)))
    else:
        mask = np.invert(np.tri(m, dtype=np.bool)) & (counts != 0) & (d != 0)

    bias = bias.reshape(-1, 1)
    if beta is None:
        beta = counts[mask].sum() / (
            (d[mask] ** alpha) * (bias * bias.T)[mask]).sum()

    g = beta * d ** alpha
    g *= bias * bias.T
    g = g[mask]

    ll = counts[mask] * np.log(beta) + \
        alpha * counts[mask] * np.log(d[mask]) + \
        counts[mask] * np.log(bias * bias.T)[mask]
    ll -= g
    # We are trying to maximise, so we need the opposite of the log likelihood
    if np.isnan(ll.sum()):
        raise ValueError("Objective function is Not a Number")
    return - ll.sum()


def _poisson_exp_sparse(X, counts, alpha, bias,
                        beta=None, use_empty_entries=False):
    m, n = X.shape
    # if X is big this is going to take too much ram.
    # XXX We should create a sparse matrix containing the same entries as the
    # contact count matrix.

    d = np.sqrt(((X[counts.row] - X[counts.col])**2).sum(axis=1))
    if use_empty_entries:
        raise NotImplementedError

    bias = bias.flatten()
    if beta is None:
        beta = counts.sum() / (
            (d ** alpha) * bias[counts.row] * bias[counts.col]).sum()

    g = beta * d ** alpha * \
        bias[counts.row] * bias[counts.col]
    ll = (counts.data * np.log(beta) + alpha * counts.data * np.log(d) +
          counts.data * np.log(bias[counts.row] * bias[counts.col]))
    ll -= g
    if np.isnan(ll.sum()):
        raise ValueError("Objective function is Not a Number")
    return -ll.sum()


def gradient_poisson_exp(X, counts, alpha, bias=None,
                         beta=None, use_empty_entries=True):
    """
    Computes the gradient of the log likelihood of the gradient in alpha and
    beta

    Parameters
    ----------
    X: ndarray
        3D positions

    counts: n * n ndarray
        of interaction frequencies

    alpha: float
        parameter of the expential law

    beta: float
        constant

    use_empty_entries: boolean, optional, default: False
        whether to use the zeros entries as information or not.

    Returns
    -------
    grad_alpha, grad_beta: float, float
        The value of the gradient in alpha and the value of the gradient in
        beta
    """

    if VERBOSE:
        print("Poisson exponential model : computation of the gradient")
    if bias is None:
        bias = np.ones((counts.shape[0], 1))
    if sparse.issparse(counts):
        return _gradient_poisson_exp_sparse(
            X, counts, alpha, bias, beta,
            use_empty_entries=use_empty_entries)
    else:
        return _gradient_poisson_exp_dense(
            X, counts, alpha, bias, beta,
            use_empty_entries=use_empty_entries)


def _gradient_poisson_exp_dense(X, counts, alpha, bias, beta,
                                use_empty_entries=True):
    m, n = X.shape

    d = euclidean_distances(X)

    bias = bias.reshape(-1, 1)

    if use_empty_entries:
        mask = np.invert(np.tri(m, dtype=np.bool))
    else:
        mask = np.invert(np.tri(m, dtype=np.bool)) & (counts != 0)

    beta = counts[mask].sum() / (
        (d[mask] ** alpha) * (bias * bias.T)[mask]).sum()

    grad_alpha = - beta * \
        ((bias * bias.T)[mask] * d[mask] ** alpha * np.log(d[mask])).sum() \
        + (counts[mask] * np.log(d[mask])).sum()
    return - np.array([grad_alpha])


def _gradient_poisson_exp_sparse(X, counts, alpha, bias, beta,
                                 use_empty_entries=True):
    m, n = X.shape
    bias = bias.flatten()

    if use_empty_entries:
        raise NotImplementedError
    d = np.sqrt(((X[counts.row] - X[counts.col])**2).sum(axis=1))

    beta = counts.sum() / (
        (d ** alpha) * bias[counts.row] * bias[counts.col]).sum()

    grad_alpha = - beta * (bias[counts.row] * bias[counts.col] * d ** alpha *
                           np.log(d)).sum() \
        + (counts.data *
           np.log(d)).sum()
    return - np.array([grad_alpha])


def eval_f(x, user_data=None):
    """
    Evaluate the objective function.

    This computes the stress
    """
    if VERBOSE:
        print("Poisson exponential model : eval_f")

    m, n, counts, X, bias, use_empty_entries = user_data
    X = X.reshape((m, n))
    tmp = poisson_exp(X, counts, x[0], bias=bias,
                      use_empty_entries=use_empty_entries)
    return tmp


def eval_f_X(X, user_data=None):
    """
    Evaluate the objective function.

    This computes the stress

    Parameters
    ----------
    X: ndarray, shape m * n
        3D configuration

    user_data: optional, default=None
        m, n, counts, alpha, beta

    Returns
    -------
    loglikelihood of the model.
    """
    if VERBOSE:
        print("Poisson exponential model : computation of the eval_f")

    m, n, counts, alpha, beta, d = user_data
    X = X.reshape((m, n))
    tmp = poisson_exp(X, counts, alpha, beta)
    return tmp


def eval_grad_f(x, user_data=None):
    """
    Evaluate the gradient of the function in alpha
    """
    if VERBOSE:
        print("Poisson exponential model : eval_grad_f (evaluation in alpha)")

    m, n, counts, X, bias, use_empty_entries = user_data
    X = X.reshape((m, n))
    tmp = gradient_poisson_exp(X, counts, x[0], bias=bias, beta=None,
                               use_empty_entries=use_empty_entries)

    return tmp


def eval_grad_f_X(X, user_data=None):
    """
    Evaluate the gradient of the function in X
    """
    global niter
    niter += 1
    if not niter % 10:
        X.dump('%d.sol.npy' % niter)

    if VERBOSE:
        print("Poisson exponential model : eval_grad_f_X (evaluation in f X)")

    m, n, counts, alpha, beta, d = user_data
    X = X.reshape((m, n))
    dis = euclidean_distances(X)

    tmp = X.repeat(m, axis=0).reshape((m, m, n))
    dif = tmp - tmp.transpose(1, 0, 2)
    dis = dis.repeat(n).reshape((m, m, n))
    counts = counts.repeat(n).reshape((m, m, n))

    grad = - alpha * beta * dif / dis * (dis ** (alpha - 1)) + \
        counts * alpha * dif / (dis ** 2)
    grad[np.isnan(grad)] = 0
    return - grad.sum(1)


def eval_stress(X, user_data=None):
    """
    """
    if VERBOSE:
        print("Computing stress: eval_stress")
    m, n, distances, alpha, beta, d = user_data
    X = X.reshape((m, n))
    dis = euclidean_distances(X)
    stress = ((dis - distances) ** 2)[distances != 0].sum()
    return stress


def eval_grad_stress(X, user_data=None):
    """
    Compute the gradient of the stress
    """
    if VERBOSE:
        print('Compute the gradient of the stress: eval_grad_stress')
    m, n, distances, alpha, beta, d = user_data
    X = X.reshape((m, n))
    tmp = X.repeat(m, axis=0).reshape((m, m, n))
    dif = tmp - tmp.transpose(1, 0, 2)
    dis = euclidean_distances(X).repeat(3, axis=1).flatten()
    distances = distances + distances.T
    distances = distances.repeat(3, axis=1).flatten()
    grad = 2 * dif.flatten() * (dis - distances) / dis
    grad[(distances == 0) | np.isnan(grad)] = 0
    return grad.reshape((m, m, n)).sum(axis=1)


def eval_h(x, lagrange, obj_factor, flag, user_data=None):
    """
    """
    return False


def _estimate_beta(counts, X, alpha=-3, bias=None):
    m, n = X.shape
    if bias is None:
        bias = np.ones((counts.shape[0], 1))
    if sparse.issparse(counts):
        counts = counts.tocoo()
        dis = np.sqrt(((X[counts.row] - X[counts.col])**2).sum(axis=1))
        bias = bias.flatten()
        beta = counts.sum() / (
            (dis ** alpha) * bias[counts.row] * bias[counts.col]).sum()

    else:
        dis = euclidean_distances(X)
        mask = np.invert(np.tri(m, dtype=np.bool)) & (counts != 0) & (dis != 0)
        beta = counts[mask].sum() / (
            (dis[mask] ** alpha) * (bias * bias.T)[mask]).sum()

    return beta


def estimate_alpha_beta(counts, X, bias=None, ini=None, verbose=0,
                        use_empty_entries=False, random_state=None):
    """
    Estimate the parameters of g

    Parameters
    ----------
    counts: ndarray

    use_empty_entries: boolean, optional, default: True
        whether to use zeroes entries as information or not

    """

    m, n = X.shape
    bounds = np.array(
        [[-100, 1e-2]])

    random_state = check_random_state(random_state)
    if ini is None:
        ini = - random_state.randint(1, 100, size=(2, )) + \
            random_state.rand(1)
    data = (m, n, counts, X, bias,
            use_empty_entries)

    results = optimize.fmin_l_bfgs_b(
        eval_f,
        ini[0],
        eval_grad_f,
        (data, ),
        bounds=bounds,
        iprint=1,
        maxiter=1000,
        )

    beta = _estimate_beta(counts, X, alpha=results[0], bias=bias)
    return results[0], beta
