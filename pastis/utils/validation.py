

def _check_squared_array(X):
    """
    Check whether arrays are squared

    Parameters
    ----------
    X : ndarray

    Returns
    -------
    X
    """
    if len(X.shape) != 2:
        raise ValueError(
            "The ndarray has %d dimension. 2D array is expected." %
            len(X.shape))

    if X.shape[0] != X.shape[1]:
        raise ValueError(
            "The ndarray is of shape (%d, %d). Squared array is expected." %
            (X.shape[0], X.shape[1]))

    return X
