import numpy as np
from scipy import sparse
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal
from iced.utils import get_genomic_distances

from pastis.dispersion import compute_mean_variance
from pastis.dispersion import _get_indices_genomic_distances
from pastis.dispersion import ExponentialDispersion


def test_compute_mean_variance_sparse_without_zeros():
    n = 100
    counts = np.random.rand(n, n)
    counts = np.triu(counts)
    counts = counts.T + counts
    lengths = np.array([n])
    _, mean_dense, var_dense, _ = compute_mean_variance(
        counts, lengths,
        use_zero_counts=False)
    counts = np.triu(counts)
    _, mean_sparse, var_sparse, _ = compute_mean_variance(
        sparse.coo_matrix(counts),
        lengths,
        use_zero_counts=False)
    assert_array_almost_equal(mean_dense, mean_sparse)
    assert_array_almost_equal(var_dense, var_sparse)


def test_compute_mean_variance_sparse_with_zeros():
    n = 100
    counts = np.random.rand(n, n)
    counts = np.triu(counts)
    counts = counts.T + counts

    lengths = np.array([n])
    _, mean_dense, var_dense, _ = compute_mean_variance(
        counts, lengths,
        use_zero_counts=True)
    counts = np.triu(counts)
    _, mean_sparse, var_sparse, _ = compute_mean_variance(
        sparse.coo_matrix(counts),
        lengths,
        use_zero_counts=True)
    assert_array_almost_equal(mean_dense, mean_sparse)
    assert_array_almost_equal(var_dense, var_sparse)


def test_get_indices_genomic_distances():
    lengths = np.array([25, 50])
    gdis = get_genomic_distances(lengths)
    row, col = _get_indices_genomic_distances(lengths, 10)
    assert_array_equal(10 * np.ones(len(row)), gdis[row, col])


# Ghost tests for dispersion estimation

def test_exponential_dispersion():
    n = 100
    counts = np.random.rand(n, n)
    counts = np.triu(counts)
    counts = counts.T + counts

    lengths = np.array([n])
    _, mean_dense, var_dense, _ = compute_mean_variance(
        counts, lengths,
        use_zero_counts=True)

    for degree in [0, 1, 2]:
        dispersion_ = ExponentialDispersion(degree=degree)
        dispersion_.fit(mean_dense, var_dense)
        disp = dispersion_.predict(mean_dense)
        assert disp.shape == mean_dense.shape

        disp_der = dispersion_.predict(mean_dense)
        assert disp_der.shape == mean_dense.shape
