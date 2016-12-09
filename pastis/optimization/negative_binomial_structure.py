import numpy as np
from scipy import sparse
from scipy import special
from scipy import optimize
from sklearn.utils import check_random_state
from .utils import ConstantDispersion

_niter = 0


def negative_binomial_obj(X, counts, alpha=-3., beta=1., bias=None,
                          dispersion=None,
                          use_zero_counts=False, cst=0):
    if bias is None:
        bias = np.ones((counts.shape[0], 1))

    if dispersion is None:
        dispersion = ConstantDispersion()

    if sparse.issparse(counts):
        return _negative_binomial_obj_sparse(
            X, counts, beta=beta, bias=bias,
            alpha=alpha, dispersion=dispersion,
            use_zero_counts=use_zero_counts, cst=cst)
    else:
        return _negative_binomial_obj_dense(
            X, counts, beta=beta, bias=bias,
            alpha=alpha, d=dispersion, use_zero_counts=use_zero_counts)


def _negative_binomial_obj_dense(X, counts, alpha=-3, beta=1, d=7, bias=None,
                                 use_zero_counts=False):
    raise NotImplementedError


def _negative_binomial_obj_sparse(X, counts, alpha=-3, beta=1., bias=None,
                                  dispersion=None,
                                  use_zero_counts=False, cst=0):
    if use_zero_counts:
        raise NotImplementedError

    bias = bias.flatten()

    # XXX This will return nan if the counts data is not hollow !!!
    # We should check for this

    dis = np.sqrt(((X[counts.row] - X[counts.col])**2).sum(axis=1))
    fdis = bias[counts.row] * bias[counts.col] * beta * dis ** alpha

    d = dispersion.predict(fdis)

    obj = - special.gammaln(counts.data + d).sum()
    obj += special.gammaln(d).sum()
    obj -= (counts.data * np.log(fdis)).sum()
    obj -= (d * np.log(d)).sum()
    obj += ((counts.data + d) * np.log(d + fdis)).sum()
    return obj.sum()


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
        return _negative_binomial_gradient_dense(
            X, counts, beta=beta, bias=bias,
            alpha=alpha, d=dispersion,
            use_zero_counts=use_zero_counts)


def _negative_binomial_gradient_dense(X, counts, alpha=-3, beta=1, d=7,
                                      bias=None,
                                      use_zero_counts=False):
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

    diff = X[counts.row] - X[counts.col]

    d = dispersion.predict(fdis)

    d_prime = (dispersion.derivate(fdis) * alpha * beta * bias[counts.row] *
               bias[counts.col] * dis ** (alpha - 2))[:, np.newaxis] * diff

    grad = -(special.digamma(counts.data + d)[:, np.newaxis] * d_prime)
    grad += special.digamma(d)[:, np.newaxis] * d_prime
    grad -= (counts.data * alpha / dis ** 2)[:, np.newaxis] * diff
    grad -= (np.log(d) + 1)[:, np.newaxis] * d_prime
    grad += np.log(d + fdis)[:, np.newaxis] * d_prime
    grad += ((counts.data + d) / (d + fdis))[:, np.newaxis] * (
        (fdis * alpha / dis**2)[:, np.newaxis] * diff + d_prime)

    grad_ = np.zeros(X.shape)

    for i in range(X.shape[0]):
        grad_[i] += grad[counts.row == i].sum(axis=0)
        grad_[i] -= grad[counts.col == i].sum(axis=0)

    return grad_


def eval_f(x, user_data=None):
    n, counts, alpha, beta, dispersion, bias, use_zero_counts, cst = user_data
    x = x.reshape((n, 3))
    obj = negative_binomial_obj(x, counts, alpha=alpha, beta=beta, bias=bias,
                                dispersion=dispersion, cst=cst)
    return obj


def eval_grad_f(x, user_data=None):
    n, counts, alpha, beta, dispersion, bias, use_zero_counts, cst = user_data
    x = x.reshape((n, 3))
    grad = negative_binomial_gradient(x, counts, alpha=alpha,
                                      beta=beta, bias=bias,
                                      dispersion=dispersion)
    x = x.flatten()
    return grad.flatten()


def estimate_X(counts, alpha, beta,
               ini=None, dispersion=None, bias=None,
               verbose=0,
               use_zero_entries=False,
               random_state=None, maxiter=10000):
    """
    Estimate the parameters of g

    Parameters
    ----------
    counts: ndarray

    """
    n = counts.shape[0]

    random_state = check_random_state(random_state)
    if ini is None:
        ini = 1 - 2 * random_state.rand(n * 3)

    data = (n, counts, alpha, beta, dispersion, bias,
            use_zero_entries, 0)

    results = optimize.fmin_l_bfgs_b(
        eval_f,
        ini.flatten(),
        eval_grad_f,
        (data, ),
        iprint=0,
        maxiter=maxiter,
        )
    results = results[0].reshape(-1, 3)
    return results
