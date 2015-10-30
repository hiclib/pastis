import numpy as np
from scipy import sparse
from ._normalization_ import _update_normalization_csr
from .utils import is_symetric_or_tri, is_tri


def ICE_normalization(X, SS=None, max_iter=3000, eps=1e-4, copy=True,
                      norm='l1', verbose=0, output_bias=False):
    """
    ICE normalization

    The imakaev normalization of Hi-C data consists of iteratively estimating
    the bias such that all the rows and columns (ie loci) have equal
    visibility.

    Parameters
    ----------
    X : ndarray or sparse array (n, n)
        raw interaction frequency matrix

    max_iter : integer, optional, default: 3000
        Maximum number of iteration

    eps : float, optional, default: 1e-4
        the relative increment in the results before declaring convergence.

    copy : boolean, optional, default: True
        If copy is True, the original data is not modified.

    norm : string, optional, default: l1
        If set to "l1", will compute the ICE algorithm of the paper. Else, the
        algorithm is adapted to use the l2 norm, as suggested in the SCN
        paper.

    output_bias : boolean, optional, default: False
        whether to output the bias vector.

    Returns
    -------
    X, (bias) : ndarray (n, n)
        Normalized IF matrix and bias of output_bias is True

    Example
    -------
    .. plot:: examples/normalization/plot_ICE_normalization.py
    """
    if copy:
        X = X.copy()

    if sparse.issparse(X):
        if not sparse.isspmatrix_csr(X):
            X = sparse.csr_matrix(X, dtype="float")
        X.sort_indices()
    else:
        X[np.isnan(X)] = 0
    X = X.astype('float')

    mean = X.mean()
    m = X.shape[0]
    is_symetric_or_tri(X)
    old_dbias = None
    bias = np.ones((m, 1))
    for it in np.arange(max_iter):
        if norm == 'l1':
            # Actually, this should be done if the matrix is diag sup or diag
            # inf
            if is_tri(X):
                sum_ds = X.sum(axis=0) + X.sum(axis=1).T - X.diagonal()
            else:
                sum_ds = X.sum(axis=0)
        elif norm == 'l2':
            if is_tri(X):
                sum_ds = ((X**2).sum(axis=0) +
                          (X**2).sum(axis=1).T -
                          (X**2).diagonal())
            else:
                sum_ds = (X**2).sum(axis=0)

        if SS is not None:
            raise NotImplementedError

        dbias = sum_ds.reshape((m, 1))
        # To avoid numerical instabilities
        dbias /= dbias[dbias != 0].mean()

        dbias[dbias == 0] = 1
        bias *= dbias

        if sparse.issparse(X):
            X = _update_normalization_csr(X, np.array(dbias).flatten())
        else:
            X /= dbias * dbias.T

        X *= mean / X.mean()
        if old_dbias is not None and np.abs(old_dbias - dbias).sum() < eps:
            if verbose > 1:
                print("break at iteration %d" % (it,))
            break

        if verbose > 1 and old_dbias is not None:
            print('ICE at iteration %d %s' %
                  (it, np.abs(old_dbias - dbias).sum()))

        # Rescaling X so that the  mean always stays the same.
        # XXX should probably do this properly, ie rescale the bias such that
        # the total scaling of the contact counts don't change.
        old_dbias = dbias.copy()
    if output_bias:
        return X, bias
    else:
        return X


def SCN_normalization(X, max_iter=300, eps=1e-6, copy=True):
    """
    Sequential Component Normalization

    Parameters
    ----------
    X : ndarray (n, n)
        raw interaction frequency matrix

    max_iter : integer, optional, default: 300
        Maximum number of iteration

    eps : float, optional, default: 1e-6
        the relative increment in the results before declaring convergence.

    copy : boolean, optional, default: True
        If copy is True, the original data is not modified.

    Returns
    -------
    X : ndarray,
        Normalized IF
    """
    # X needs to be square, else it's gonna fail

    m, n = X.shape
    if m != n:
        raise ValueError

    if copy:
        X = X.copy()
    X = X.astype(float)

    for it in np.arange(max_iter):
        sum_X = np.sqrt((X ** 2).sum(0))
        sum_X[sum_X == 0] = 1
        X /= sum_X
        X = X.T
        sum_X = np.sqrt((X ** 2).sum(0))
        sum_X[sum_X == 0] = 1
        X /= sum_X
        X = X.T

        if np.abs(X - X.T).sum() < eps:
            print("break at iteration %d" % (it,))
            break

    return X
