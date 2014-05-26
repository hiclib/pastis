import numpy as np
from sklearn.metrics import euclidean_distances
import pyipopt

"""Utility method for the pyipopt solver, for the Poisson Model
"""

VERBOSE = False
niter = 0


def poisson_exp(X, counts, alpha, beta=None, use_empty_entries=True):
    """
    Computes the log likelihood of the poisson exponential model.

    Parameters
    ----------
    X: ndarray
        3D positions

    counts: n * n ndarray
        of interaction frequencies

    alpha: float
        parameter of the expential law

    beta: float, optional, default None
        constant. If is set to None, it will be computed using the maximum log
        likelihood knowing alpha.

    use_empty_entries: boolean, optional, default False
        whether to use zeroes entries as information or not

    Returns
    -------
    ll: float
        log likelihood
    """
    if VERBOSE:
        print "Poisson power law model : computation of the log likelihood"
    m, n = X.shape
    d = euclidean_distances(X)
    if use_empty_entries:
        mask = (np.tri(m, dtype=np.bool) == False)
    else:
        mask = (np.tri(m, dtype=np.bool) == False) & (counts != 0) & (d != 0)

    if beta is None:
        beta = counts[mask].sum() / (d[mask] ** alpha).sum()
    g = beta * d[mask] ** alpha

    ll = counts[mask] * np.log(beta) + alpha * counts[mask] * np.log(d[mask])
    ll -= g
    # We are trying to maximise, so we need the opposite of the log likelihood
    return - ll.sum()


def gradient_poisson_exp(X, counts, alpha, beta, use_empty_entries=True):
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
        print "Poisson exponential model : computation of the gradient"

    m, n = X.shape

    d = euclidean_distances(X)

    if use_empty_entries:
        mask = np.tri(m, dtype=np.bool) == False
    else:
        mask = (np.tri(m, dtype=np.bool) == False) & (counts != 0)

    beta = counts[mask].sum() / (d[mask] ** alpha).sum()

    grad_alpha = - beta * (d[mask] ** alpha * np.log(d[mask])).sum() \
                 + (counts[mask] * np.log(d[mask])).sum()
    return - np.array([grad_alpha])


def eval_no_f(x, user_data=None):
    """
    Evaluate the object function (no objective function).
    """
    if VERBOSE:
        print "No objective function"
    return 0.


def eval_grad_no_f(X, user_data=None):
    if VERBOSE:
        print "Gradient of no objective function"
    m, n, counts, alpha, beta, d = user_data
    X.dump('tmp.npy')
    return np.zeros(n * m)


def eval_f(x, user_data=None):
    """
    Evaluate the objective function.

    This computes the stress
    """
    if VERBOSE:
        print "Poisson exponential model : eval_f"

    m, n, counts, X, use_empty_entries = user_data
    X = X.reshape((m, n))
    tmp = poisson_exp(X, counts, x[0], use_empty_entries=use_empty_entries)
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
        print "Poisson exponential model : computation of the eval_f"

    m, n, counts, alpha, beta, d = user_data
    X = X.reshape((m, n))
    tmp = poisson_exp(X, counts, alpha, beta)
    return tmp


def eval_grad_f(x, user_data=None):
    """
    Evaluate the gradient of the function in alpha
    """
    if VERBOSE:
        print "Poisson exponential model : eval_grad_f (evaluation in alpha)"

    m, n, counts, X, use_empty_entries = user_data
    X = X.reshape((m, n))
    tmp = gradient_poisson_exp(X, counts, x[0], None,
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
        print "Poisson exponential model : eval_grad_f_X (evaluation in f X)"

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
        print "Computing stress: eval_stress"
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
        print 'Compute the gradient of the stress: eval_grad_stress'
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
    if VERBOSE:
        print "Poisson exponential model : eval_g"

    m, n, wish_dist, alpha, beta, d = user_data

    x = x.reshape((m, n))
    dis = euclidean_distances(x)
    dis = dis ** 2
    mask = (np.tri(m, dtype=np.bool) == False)
    g = np.concatenate([dis[mask].flatten(), ((x - d) ** 2).sum(axis=1)])
    return g


def eval_jac_g(x, flag, user_data=None):
    """
    Computes the jacobian for the constraints mentionned in duan-et-al
    """
    if VERBOSE:
        print "Poisson exponential model : eval_jac_g"

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
        mask = (np.tri(m, dtype=np.bool) == False)
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


def estimate_alpha_beta(counts, X, ini=None, verbose=0,
                        use_empty_entries=True):
    """
    Estimate the parameters of g

    Parameters
    ----------
    counts: ndarray

    use_empty_entries: boolean, optional, default: True
        whether to use zeroes entries as information or not

    """
    m, n = X.shape
    nvar = 1
    ncon = 0
    nnzj = 0
    nnzh = 0

    x_L = np.array([- 10000.])
    x_U = np.array([10000000.])

    nlp = pyipopt.create(nvar, x_L, x_U, ncon, x_L,
                        x_U, nnzj, nnzh, eval_f,
                        eval_grad_f, eval_g, eval_jac_g)

    nlp.int_option('max_iter', 100)
    if ini is None:
        if verbose:
            print "Initial values not provided"
        ini = np.random.randint(1, 100, size=(1, )) + \
              np.random.random(size=(1, ))
    results = nlp.solve(ini, (m, n, counts, X, use_empty_entries))
    try:
        x, _, _, _, _ = results
    except ValueError:
        x, _, _, _, _, _ = results


    # Evaluate counts with new estimated model.
    d = euclidean_distances(X)
    mask = (np.tri(m, dtype=np.bool) == False) & (counts != 0) & (d != 0)
    beta = counts[mask].sum() / (d[mask] ** x[0]).sum()
    return x[0], beta


# Let's now work on reconstructing the 3D structure
def estimate_3D_structure(nlp, counts, alpha, beta, d, X=None, m=None,
                          n=None):
    """
    Maximise the loglikelihood in X

    """

    if X is None:
        X = np.random.random(size=(m * n))

    X, _, _, _, _ = nlp.solve(X.flatten(), (m, n, counts, alpha, beta, d))
    X = X.reshape((m, n))
    return X
