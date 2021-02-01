import numpy as np
from numpy.testing import assert_raises

from pastis.utils import _check_squared_array


def test_check_squared_array():
    X = np.zeros((10, 9, 0))
    assert_raises(ValueError, _check_squared_array, X)

    X = np.zeros((10, 9))
    assert_raises(ValueError, _check_squared_array, X)
