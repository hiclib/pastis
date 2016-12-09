import numpy as np
from scipy import special
from scipy import sparse
from scipy import optimize
from sklearn.utils import check_random_state
from .utils import ConstantDispersion


def negative_binomial_obj(X, counts, alpha=-3., beta=1., bias=None,
                          dispersion=None,
                          use_zero_counts=False):
    if bias is None:
        bias = np.ones((counts.shape[0], 1))

    if dispersion is None:
        dispersion = ConstantDispersion()

    if sparse.issparse(counts):
        return _negative_binomial_obj_sparse(
            X, counts, beta=beta, bias=bias,
            alpha=alpha, dispersion=dispersion,
            use_zero_counts=use_zero_counts)
    else:
        raise NotImplementedError
        return _negative_binomial_obj_dense(
            X, counts, beta=beta, bias=bias,
            alpha=alpha, dispersion=dispersion,
            use_zero_counts=use_zero_counts)


def _negative_binomial_obj_dense(X, counts, alpha=-3, beta=1, dispersion=None,
                                 bias=None,
                                 use_zero_counts=False):
    if use_zero_counts:
        raise NotImplementedError


def _negative_binomial_obj_sparse(X, counts, alpha=-3, beta=1., bias=None,
                                  dispersion=None,
                                  use_zero_counts=False):
    if use_zero_counts:
        raise NotImplementedError

    bias = bias.flatten()

    dis = np.sqrt(((X[counts.row] - X[counts.col])**2).sum(axis=1))
    fdis = bias[counts.row] * bias[counts.col] * beta * dis ** alpha

    d = dispersion.predict(fdis)
    if np.any(d < 0):
        print("Negative dispersion")

    obj = - special.gammaln(counts.data + d).sum()
    obj += special.gammaln(d).sum()
    obj -= (counts.data * np.log(fdis)).sum()
    obj -= (d * np.log(d)).sum()
    obj += ((counts.data + d) * np.log(d + fdis)).sum()
    return obj.sum()


def eval_f(x, user_data=None):
    """
    Evaluate the objective function.
    """
    m, n, counts, X, dispersion, bias, use_zero_counts = user_data
    X = X.reshape((m, n))
    return negative_binomial_obj(X, counts, alpha=x[0], beta=x[1],
                                 dispersion=dispersion,
                                 bias=bias,
                                 use_zero_counts=use_zero_counts)


def negative_binomial_gradient(X, counts, alpha=-3, beta=1, bias=None,
                               dispersion=None,
                               use_zero_counts=False):
    if bias is None:
        bias = np.ones((counts.shape[0], 1))

    if dispersion is None:
        dispersion = ConstantDispersion()

    if sparse.issparse(counts):
        return _negative_binomial_gradient_sparse(
            X, counts, beta=beta, bias=bias,
            alpha=alpha, dispersion=dispersion,
            use_zero_counts=use_zero_counts)
    else:
        raise NotImplementedError
        return _negative_binomial_gradient_dense(
            X, counts, beta=beta, bias=bias,
            alpha=alpha, dispersion=dispersion,
            use_zero_counts=use_zero_counts)


def _negative_binomial_gradient_dense(X, counts, alpha=-3, beta=1,
                                      dispersion=None,
                                      bias=None,
                                      use_zero_counts=False):
    if use_zero_counts:
        raise NotImplementedError


def _negative_binomial_gradient_sparse(X, counts, alpha=-3, beta=1.,
                                       dispersion=None,
                                       bias=None,
                                       use_zero_counts=False):
    if use_zero_counts:
        raise NotImplementedError
    bias = bias.flatten()

    dis = np.sqrt(((X[counts.row] - X[counts.col])**2).sum(axis=1))
    fdis = bias[counts.row] * bias[counts.col] * beta * dis ** alpha

    d = dispersion.predict(fdis)

    if np.any(d < 0):
        print("Negative dispersion")

    d_prime = dispersion.derivate(fdis) * fdis * np.log(dis)

    grad = - (special.digamma(counts.data + d) * d_prime).sum()
    grad += (special.digamma(d) * d_prime).sum()
    grad -= (counts.data * np.log(dis)).sum()
    grad -= (d_prime * (np.log(d) + 1)).sum()
    grad += (d_prime * np.log(d + fdis)).sum()
    grad += ((counts.data + d) / (d + fdis) *
             (d_prime + fdis * np.log(dis))).sum()

    d_prime = dispersion.derivate(fdis) * fdis / beta

    grad_beta = - (special.digamma(counts.data + d) * d_prime).sum()
    grad_beta += (special.digamma(d) * d_prime).sum()
    grad_beta -= (d_prime * (np.log(d) + 1)).sum()
    grad_beta -= (counts.data / beta).sum()
    grad_beta += (d_prime * np.log(d + fdis)).sum()
    grad_beta += (
        (counts.data + d) / (d + fdis) * (d_prime + fdis / beta)).sum()

    return np.array([grad, grad_beta])


def eval_grad_f(x, user_data=None):
    """
    Evaluate the gradient of the function in alpha
    """
    m, n, counts, X, dispersion, bias, use_zero_counts = user_data
    X = X.reshape((m, n))
    return negative_binomial_gradient(X, counts, alpha=x[0], beta=x[1],
                                      dispersion=dispersion,
                                      bias=bias,
                                      use_zero_counts=use_zero_counts)


def estimate_alpha_beta(counts, X, ini=None, dispersion=None, bias=None,
                        verbose=0,
                        random_state=None, maxiter=10000):
    """
    Estimate the parameters of g

    Parameters
    ----------
    counts: ndarray

    """
    m, n = X.shape
    bounds = np.array(
        [[-100, 1e-2],
         [1e-2, 10000]])

    random_state = check_random_state(random_state)
    if ini is None:
        ini = 5 * random_state.rand(2)
        ini[0] *= -1.
        ini = ini.astype(float)

    data = (m, n, counts, X, dispersion, bias,
            False)

    results = optimize.fmin_l_bfgs_b(
        eval_f,
        ini.flatten(),
        eval_grad_f,
        (data, ),
        bounds=bounds,
        iprint=1,
        maxiter=maxiter
        )
    return results[0]
