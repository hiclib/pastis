import numpy as np
from scipy import sparse
from scipy import special
from scipy import optimize
from scipy import spatial
from sklearn.utils import check_random_state
from .utils import ConstantDispersion

_niter = 0


def negative_binomial_obj(X, counts, alpha=-3., beta=1., bias=None,
                          dispersion=None,
                          use_zero_counts=False, lengths=None):
    """
    Computes the negative binomial objective function.

    Parameters
    ----------

    X : ndarray (n, 3)
        the 3D structure

    counts : ndarray (n, n)
        the contact count matrix

    alpha : float, optional, default: -3
        the count to distance coefficient

    beta : float, optional, default: 1
        a scaling factor

    bias : ndarray (n, 1)
        the bias vector

    """
    if bias is None:
        bias = np.ones((counts.shape[0], 1))
    if len(bias.shape) != 2:
        bias = bias.reshape(-1, 1)
    if bias.shape[0] != counts.shape[0]:
        raise ValueError("Problem in bias shape")

    if dispersion is None:
        dispersion = ConstantDispersion()

    if sparse.issparse(counts):
        return _negative_binomial_obj_sparse(
            X, counts, beta=beta, bias=bias,
            alpha=alpha, dispersion=dispersion,
            use_zero_counts=use_zero_counts, lengths=lengths)
    else:
        return _negative_binomial_obj_dense(
            X, counts, beta=beta, bias=bias,
            alpha=alpha, dispersion=dispersion,
            lengths=lengths,
            use_zero_counts=use_zero_counts)


def _negative_binomial_obj_dense(X, counts, alpha=-3, beta=1, dispersion=None,
                                 bias=None,
                                 lengths=None,
                                 use_zero_counts=False):

    dis = spatial.distance_matrix(X, X)
    if not use_zero_counts:
        mask = np.triu(counts != 0, k=1)
    else:
        mask = (np.triu(np.ones(counts.shape, dtype=np.bool_), k=1) &
                np.invert(np.isnan(counts)))

    bias = bias.reshape(-1, 1)

    fdis = bias.T * bias * beta * dis ** alpha
    # XXX
    d = dispersion.predict(
        dis[mask] ** alpha,
        beta * (bias.T * bias)[mask])

    obj = (- special.gammaln(counts[mask] + d))
    obj += (special.gammaln(d))
    obj -= (counts[mask] * np.log(fdis[mask]))
    obj -= (d * np.log(d))
    obj += ((counts[mask] + d) * np.log(d + fdis[mask]))
    obj = obj.sum()

    if np.any(np.isnan(obj)):
        raise ValueError()

    else:
        return obj


def _negative_binomial_obj_sparse(X, counts, alpha=-3, beta=1., bias=None,
                                  dispersion=None,
                                  use_zero_counts=False, lengths=None):
    if use_zero_counts:
        raise NotImplementedError

    bias = bias.flatten()
    # XXX This will return nan if the counts data is not hollow !!!
    # We should check for this

    dis = np.sqrt(((X[counts.row] - X[counts.col])**2).sum(axis=1))
    fdis = bias[counts.row] * bias[counts.col] * beta * dis ** alpha

    d = dispersion.predict(dis ** alpha,
                           beta * bias[counts.row] * bias[counts.col])

    obj = - (special.gammaln(counts.data + d)).sum()
    obj += (special.gammaln(d)).sum()
    obj -= (counts.data * np.log(fdis)).sum()
    obj -= (d * np.log(d)).sum()
    obj += ((counts.data + d) * np.log(d + fdis)).sum()
    obj = obj.sum()

    return obj


def negative_binomial_gradient(X, counts, alpha=-3, beta=1, bias=None,
                               dispersion=None,
                               lengths=None,
                               use_zero_counts=False):
    if bias is None:
        bias = np.ones((counts.shape[0], 1))

    if dispersion is None:
        dispersion = ConstantDispersion()

    if sparse.issparse(counts):
        return _negative_binomial_gradient_sparse(
            X, counts, beta=beta, bias=bias,
            alpha=alpha, dispersion=dispersion,
            lengths=lengths,
            use_zero_counts=use_zero_counts)
    else:
        return _negative_binomial_gradient_dense(
            X, counts, beta=beta, bias=bias,
            lengths=lengths,
            alpha=alpha, dispersion=dispersion,
            use_zero_counts=use_zero_counts)


def _negative_binomial_gradient_dense(X, counts, alpha=-3, beta=1,
                                      dispersion=None,
                                      lengths=None,
                                      bias=None,
                                      use_zero_counts=False):
    if not use_zero_counts:
        mask = np.triu(counts != 0, k=1)
    else:
        mask = (np.triu(np.ones(counts.shape, dtype=np.bool_), k=1) &
                np.invert(np.isnan(counts)))

    bias = bias.reshape(-1, 1)
    dis = spatial.distance_matrix(X, X)

    fdis = (bias.T * bias) * beta * dis ** alpha
    n = X.shape[0]
    tmp = X.repeat(n, axis=0).reshape((n, n, 3))
    diff = (tmp - tmp.transpose(1, 0, 2))
    del tmp

    d = dispersion.predict(dis ** alpha,
                           beta * (bias.T * bias))

    d_prime = (dispersion.derivate(
        dis ** alpha, beta * (bias.T * bias)) *
        alpha * dis ** (alpha - 2))[:, :, np.newaxis] * diff

    grad = -((special.digamma(counts + d))[:, :, np.newaxis] * d_prime)
    grad += (special.digamma(d))[:, :, np.newaxis] * d_prime
    grad -= (counts * alpha / dis ** 2)[:, :, np.newaxis] * diff
    grad -= ((np.log(d) + 1))[:, :, np.newaxis] * d_prime
    grad += (np.log(d + fdis))[:, :, np.newaxis] * d_prime
    grad += ((counts + d) / (d + fdis))[:, :, np.newaxis] * (
        (fdis * alpha / dis**2)[:, :, np.newaxis] * diff + d_prime)

    grad[np.invert(mask)] = 0
    grad = grad.sum(axis=1) - grad.sum(axis=0)
    if np.any(np.isnan(grad)):
        raise ValueError()
    return grad


def _negative_binomial_gradient_sparse(X, counts, alpha=-3, beta=1.,
                                       dispersion=None,
                                       lengths=None,
                                       bias=None,
                                       use_zero_counts=False):
    if use_zero_counts:
        raise NotImplementedError

    bias = bias.flatten()

    dis = np.sqrt(((X[counts.row] - X[counts.col])**2).sum(axis=1))
    fdis = bias[counts.row] * bias[counts.col] * beta * dis ** alpha

    diff = X[counts.row] - X[counts.col]

    d = dispersion.predict(dis ** alpha,
                           beta * bias[counts.row] * bias[counts.col])

    d_prime = (dispersion.derivate(
        dis ** alpha, beta * bias[counts.row] * bias[counts.col]) *
        alpha * dis ** (alpha - 2))[:, np.newaxis] * diff

    grad = -((special.digamma(counts.data + d))[:, np.newaxis] * d_prime)
    grad += (special.digamma(d))[:, np.newaxis] * d_prime
    grad -= ((counts.data * alpha / dis ** 2))[:, np.newaxis] * diff
    grad -= ((np.log(d) + 1))[:, np.newaxis] * d_prime
    grad += (np.log(d + fdis))[:, np.newaxis] * d_prime
    grad += ((counts.data + d) / (d + fdis))[:, np.newaxis] * (
        (fdis * alpha / dis**2)[:, np.newaxis] * diff + d_prime)

    grad_ = np.zeros(X.shape)

    for i in range(X.shape[0]):
        grad_[i] += grad[counts.row == i].sum(axis=0)
        grad_[i] -= grad[counts.col == i].sum(axis=0)

    return grad_


def eval_f(x, user_data=None):
    (n, counts, alpha, beta, dispersion, bias,
     use_zero_counts, lengths) = user_data
    x = x.reshape((n, 3))
    obj = negative_binomial_obj(x, counts, alpha=alpha, beta=beta, bias=bias,
                                dispersion=dispersion, lengths=lengths,
                                use_zero_counts=use_zero_counts)
    return obj


def eval_grad_f(x, user_data=None):
    (n, counts, alpha, beta, dispersion, bias,
     use_zero_counts, lengths) = user_data
    x = x.reshape((n, 3))
    grad = negative_binomial_gradient(x, counts, alpha=alpha,
                                      beta=beta, bias=bias,
                                      dispersion=dispersion,
                                      lengths=lengths,
                                      use_zero_counts=use_zero_counts)
    x = x.flatten()
    return grad.flatten()


def estimate_X(counts, alpha, beta,
               ini=None, dispersion=None, bias=None,
               lengths=None,
               verbose=0,
               use_zero_entries=False,
               approximate_gradient=False,
               method="L-BFGS-B",
               random_state=None, factr=10,
               maxiter=100000, pgtol=1,
               maxls=20):
    """
    Estimate the parameters of g

    Parameters
    ----------
    counts : ndarray

    """
    n = counts.shape[0]

    random_state = check_random_state(random_state)
    if ini is None:
        ini = 1 - 2 * random_state.rand(n * 3)

    if use_zero_entries and sparse.issparse(counts):
        counts = counts.toarray()
        mask = (counts.sum(axis=0) + counts.sum(axis=1)) == 0
        counts[:, mask] = np.nan
        counts[mask] = np.nan

    data = (n, counts, alpha, beta, dispersion, bias,
            use_zero_entries, lengths)

    options = {"maxiter": maxiter,
               "factr": factr,
               "iprint": 1,
               "approximate_gradient": approximate_gradient,
               "pgtol": pgtol,
               "maxls": maxls}
    if approximate_gradient:
        jac = None
    else:
        jac = eval_grad_f

    results = optimize.minimize(
        eval_f,
        ini.flatten(),
        method=method,
        jac=jac,
        args=(data, ),
        options=options
        )

    results = results["x"].reshape(-1, 3)
    return results
