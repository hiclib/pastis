import numpy as np
from scipy import linalg


def realign_structures(X, Y, rescale=False, copy=True, verbosity=0):
    """
    Realigns Y and X

    Parameters
    ----------
    X : ndarray (n, 3)
        First 3D structure

    Y : ndarray (n, 3)
        Second 3D structure

    rescale : boolean, optional, default: False
        Whether to rescale Y or not.

    copy : boolean, optional, default: True
        Whether to copy the data or not

    verbosity : integer, optional, default: 0
        The level of verbosity

    Returns
    -------
    Y : ndarray (n, 3)
        Realigned 3D, Xstructure
    """
    if copy:
        Y = Y.copy()
        X = X.copy()

    mask = np.invert(np.isnan(X[:, 0]) | np.isnan(Y[:, 0]))

    if rescale:
        Y, _ = realign_structures(X, Y)
        alpha = (X[mask] * Y[mask]).sum() / (Y[mask]**2).sum()
        Y *= alpha

    Y -= Y[mask].mean(axis=0)
    X -= X[mask].mean(axis=0)

    K = np.dot(X[mask].T, Y[mask])
    U, L, V = linalg.svd(K)
    V = V.T

    # R = np.dot(V, np.dot(t, U.T))
    R = np.dot(V, U.T)
    if linalg.det(R) < 0:
        if verbosity:
            print("Reflexion found")
        V[:, -1] *= -1
        R = np.dot(V, U.T)
    Y_fit = np.dot(Y, R)

    error = ((X[mask] - Y_fit[mask]) ** 2).sum()
    error /= len(X)
    error **= 0.5

    # Check at the mirror
    Y_mirror = Y.copy()
    Y_mirror[:, 0] = - Y[:, 0]

    K = np.dot(X[mask].T, Y_mirror[mask])
    U, L, V = linalg.svd(K)
    V = V.T
    if linalg.det(V) < 0:
        V[:, -1] *= -1

    R = np.dot(V, U.T)
    Y_mirror_fit = np.dot(Y_mirror, R)
    error_mirror = ((X[mask] - Y_mirror_fit[mask]) ** 2).sum()
    error_mirror /= len(X)
    error_mirror **= 0.5
    if error <= error_mirror:
        return Y_fit, error
    else:
        if verbosity:
            print("Reflexion is better")
        return Y_mirror_fit, error_mirror


def find_rotation(X, Y, copy=True):
    """
    Finds the rotation matrice C such that \|x - Q.T Y\| is minimum.

    Parameters
    ----------
    X : ndarray (n, 3)
        First 3D structure

    Y : ndarray (n, 3)
        Second 3D structure

    copy : boolean, optional, default: True
        Whether to copy the data or not

    Returns
    -------
    Y : ndarray (n, 3)
        Realigned 3D structure
    """
    if copy:
        Y = Y.copy()
        X = X.copy()
    mask = np.invert(np.isnan(X[:, 0]) | np.isnan(Y[:, 0]))
    K = np.dot(X[mask].T, Y[mask])
    U, L, V = np.linalg.svd(K)
    V = V.T

    t = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, np.linalg.det(np.dot(V, U.T))]])
    R = np.dot(V, np.dot(t, U.T))
    Y_fit = np.dot(Y, R)
    X_mean = X[mask].mean()
    Y_fit -= Y_fit[mask].mean() - X_mean
    error = ((X[mask] - Y_fit[mask]) ** 2).sum()

    # Check at the mirror
    Y_mirror = Y.copy()
    Y_mirror[:, 0] = - Y[:, 0]

    K = np.dot(X[mask].T, Y_mirror[mask])
    U, L, V = np.linalg.svd(K)
    V = V.T

    t = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, np.linalg.det(np.dot(V, U.T))]])
    R_ = np.dot(V, np.dot(t, U.T))
    Y_mirror_fit = np.dot(Y_mirror, R_)
    Y_mirror_fit -= Y_mirror[mask].mean() - X_mean
    error_mirror = ((X[mask] - Y_mirror_fit[mask]) ** 2).sum()
    return R


def distance_between_structures(X, Y):
    """
    Computes the distances per loci between structures

    Parameters
    ----------
    X : ndarray (n, l)
        First 3D structure

    Y : ndarray (n, l)
        Second 3D structure

    Returns
    -------
    distances : (n, )
        Distances between the 2 structures
    """
    if X.shape != Y.shape:
        raise ValueError("Shapes of the two matrices need to be the same")

    return np.sqrt(((X - Y) ** 2).sum(axis=1))
