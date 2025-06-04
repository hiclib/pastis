"""
Estimating the count-to-distance mapping parameters for the Negative Binomial
distribution.
"""

import numpy as np
from scipy import special
from scipy import sparse
from scipy import optimize
from scipy import spatial

from sklearn.utils import check_random_state


from .utils import ConstantDispersion
from .negative_binomial_structure import negative_binomial_obj


def eval_f(x, user_data=None):
    """
    Evaluate the objective function.
    """
    (m, n, counts, X, dispersion, bias,
     use_zero_counts, lengths, infer_beta) = user_data
    X = X.reshape((m, n))
    return negative_binomial_obj(X, counts, alpha=x[0], beta=x[1],
                                 dispersion=dispersion,
                                 bias=bias,
                                 lengths=lengths,
                                 use_zero_counts=use_zero_counts)


def negative_binomial_gradient(X, counts, alpha=-3, beta=1, bias=None,
                               dispersion=None,
                               lengths=None,
                               use_zero_counts=False,
                               infer_beta=False):
    if bias is None:
        bias = np.ones((counts.shape[0], 1))

    if dispersion is None:
        dispersion = ConstantDispersion()

    if sparse.issparse(counts):
        return _negative_binomial_gradient_sparse(
            X, counts, beta=beta, bias=bias,
            alpha=alpha, dispersion=dispersion,
            lengths=lengths,
            use_zero_counts=use_zero_counts,
            infer_beta=infer_beta)
    else:
        return _negative_binomial_gradient_dense(
            X, counts, beta=beta, bias=bias,
            alpha=alpha, dispersion=dispersion,
            lengths=lengths,
            use_zero_counts=use_zero_counts,
            infer_beta=infer_beta)


def _negative_binomial_gradient_dense(X, counts, alpha=-3, beta=1,
                                      dispersion=None,
                                      bias=None,
                                      lengths=None,
                                      use_zero_counts=False,
                                      infer_beta=False):
    if not use_zero_counts:
        mask = np.triu(counts != 0, k=1)
    else:
        mask = (np.triu(np.ones(counts.shape, dtype=np.bool_), k=1) &
                np.invert(np.isnan(counts)))

    if bias is None:
        bias = np.ones((counts.shape[0], 1))

    if infer_beta:
        bias = bias.reshape(-1, 1)
        dis = spatial.distance_matrix(X, X)
        beta_bias = beta * bias.T * bias
        disp = dispersion.predict(dis ** alpha)

        grad_beta = (
            disp * bias.T * bias * (
                - special.digamma(counts + disp * beta_bias) +
                special.digamma(beta_bias * disp) -
                np.log(disp) + np.log(disp + dis ** alpha)))[mask].sum()

    else:
        grad_beta = 0

    grad = _negative_binomial_gradient_alpha_dense(
        X, counts, alpha=alpha, beta=beta,
        bias=bias,
        dispersion=dispersion,
        lengths=lengths,
        use_zero_counts=use_zero_counts)
    grad = np.array([grad, grad_beta])

    return grad


def _negative_binomial_gradient_alpha_dense(X, counts, alpha=-3, beta=1,
                                            dispersion=None, bias=None,
                                            lengths=None,
                                            use_zero_counts=False):
    if not use_zero_counts:
        mask = np.triu(counts != 0, k=1)
    else:
        mask = (np.triu(np.ones(counts.shape, dtype=np.bool_), k=1) &
                np.invert(np.isnan(counts)))

    if bias is not None:
        bias = bias.reshape(-1, 1)

    dis = spatial.distance_matrix(X, X)
    beta_bias = beta * bias.T * bias

    disp = dispersion.predict(dis ** alpha)
    d_prime = (dispersion.derivate(
        dis ** alpha) * dis ** alpha * np.log(dis))

    grad = - (
        special.digamma(counts + disp*beta_bias)
        * beta_bias * d_prime)[mask].sum()

    grad += (
        special.digamma(disp*beta_bias) * beta_bias * d_prime)[mask].sum()
    grad -= (counts * np.log(dis))[mask].sum()
    grad -= (beta_bias * d_prime * (np.log(disp) + 1))[mask].sum()
    grad += (beta_bias * d_prime * np.log(disp + dis**alpha))[mask].sum()
    grad += ((counts + beta_bias * disp) / (disp + dis**alpha) *
             (d_prime + dis**alpha * np.log(dis)))[mask].sum()
    return grad


def _negative_binomial_gradient_sparse(X, counts, alpha=-3, beta=1.,
                                       dispersion=None,
                                       bias=None,
                                       lengths=None,
                                       use_zero_counts=False,
                                       infer_beta=False):
    if use_zero_counts:
        raise NotImplementedError

    if bias is None:
        bias = np.ones((counts.shape[0], 1))

    bias = bias.flatten()

    dis = np.sqrt(((X[counts.row] - X[counts.col])**2).sum(axis=1))
    fdis = bias[counts.row] * bias[counts.col] * beta * dis ** alpha

    disp = dispersion.predict(
        dis ** alpha,
        beta * bias[counts.row] * bias[counts.col])

    if np.any(disp < 0):
        print("Negative dispersion")

    d_prime = dispersion.derivate(
        dis ** alpha,
        beta * bias[counts.row] * bias[counts.col]) * fdis * np.log(dis)

    grad = - (special.digamma(counts.data + disp) * d_prime).sum()
    grad += (special.digamma(disp) * d_prime).sum()
    grad -= (counts.data * np.log(dis)).sum()
    grad -= (d_prime * (np.log(disp) + 1)).sum()
    grad += (d_prime * np.log(disp + fdis)).sum()
    grad += ((counts.data + disp) / (disp + fdis) *
             (d_prime + fdis * np.log(dis))).sum()
    if infer_beta:
        d_prime = dispersion.predict(dis ** alpha,
                                     bias[counts.row]*bias[counts.col])
        p = (
            dispersion.predict(dis ** alpha) /
            (dis ** alpha + dispersion.predict(dis ** alpha)))

        grad_beta = - (d_prime * special.digamma(counts.data + disp)).sum()
        grad_beta += (d_prime * special.digamma(disp)).sum()
        grad_beta -= (d_prime * np.log(p)).sum()
    else:
        grad_beta = 0

    grad = np.array([grad, grad_beta])
    return grad


def eval_grad_f(x, user_data=None):
    """
    Evaluate the gradient of the function in alpha
    """
    (m, n, counts, X, dispersion, bias,
     use_zero_counts, lengths, infer_beta) = user_data
    X = X.reshape((m, n))
    return negative_binomial_gradient(X, counts, alpha=x[0], beta=x[1],
                                      dispersion=dispersion,
                                      bias=bias,
                                      lengths=lengths,
                                      use_zero_counts=use_zero_counts,
                                      infer_beta=infer_beta)


def estimate_alpha_beta(counts, X, ini=None, dispersion=None, bias=None,
                        verbose=0,
                        use_zero_entries=False,
                        lengths=None,
                        infer_beta=True,
                        random_state=None, maxiter=10000):
    """
    Estimate the parameters of g

    Parameters
    ----------
    counts: ndarray

    """
    if lengths is not None:
        if lengths.sum() != X.shape[0]:
            raise ValueError(
                "lengths and X have incompatible sizes.")

    m, n = X.shape
    bounds = np.array(
        [[-10, -1e-8],
            [1e-8, 100]])

    random_state = check_random_state(random_state)
    if ini is None:
        ini = np.array([-3, 1.])

    if use_zero_entries and sparse.issparse(counts):
        counts = counts.toarray()
        mask = (counts.sum(axis=0) + counts.sum(axis=1)) == 0
        counts[:, mask] = np.nan
        counts[mask] = np.nan

    data = (m, n, counts, X, dispersion, bias,
            use_zero_entries, lengths, infer_beta)

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
