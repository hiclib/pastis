import numpy as np
from nose.tools import assert_equal
from numpy.testing import assert_array_equal
from scipy import sparse

from iced.utils._genome import get_intra_mask
from iced.utils._genome import get_inter_mask
from iced.utils._genome import get_genomic_distances
from iced.utils._genome import _change_lengths_resolution
from iced.utils._genome import undersample_per_chr
from iced.utils._genome import extract_sub_contact_map
from iced.utils._genome import downsample_resolution


def test_get_intra_mask():
    lengths = np.array([5, 5])
    mask = get_intra_mask(lengths)
    true_mask = np.zeros((10, 10))
    true_mask[:5, :5] = 1
    true_mask[5:, 5:] = 1
    true_mask = true_mask.astype(bool)
    assert_array_equal(mask, true_mask)

    # Now test sparse matrix
    random_state = np.random.RandomState(seed=42)

    m = 15
    rows = random_state.randint(0, 10, size=(m,))
    cols = random_state.randint(0, 10, size=(m,))
    counts = np.zeros((10, 10))
    counts[rows, cols] += 1
    counts = sparse.coo_matrix(np.triu(counts))
    rows = counts.row
    cols = counts.col
    sparse_mask = get_intra_mask(lengths, counts=counts)
    sparse_true_mask = true_mask[rows, cols]
    assert_array_equal(sparse_mask, sparse_true_mask)

    # Providing a matrix that isn't coo
    counts = counts.tocsr()
    sparse_mask = get_intra_mask(lengths, counts=counts)
    sparse_true_mask = true_mask[rows, cols]
    assert_array_equal(sparse_mask, sparse_true_mask)


def test_change_lengths_resolution():
    lengths = np.array([5, 5])
    l = _change_lengths_resolution(lengths, resolution=1)
    assert_array_equal(lengths, l)


def test_get_inter_mask():
    lengths = np.array([5, 5])
    mask = get_inter_mask(lengths)
    true_mask = np.zeros((10, 10))
    true_mask[:5, :5] = 1
    true_mask[5:, 5:] = 1
    assert_array_equal(mask, np.invert(true_mask.astype(bool)))


def test_downsample_resolution():
    random_state = np.random.RandomState(seed=42)

    lengths = np.array([10, 10])
    counts = np.triu(random_state.randint(
        0, 100, (lengths.sum(), lengths.sum())))
    counts = counts + counts.T
    downsampled_counts, downsampled_lengths = downsample_resolution(
        counts, lengths)
    assert_equal(downsampled_lengths.sum(), lengths.sum()/2)

    lengths = np.array([10, 11])
    counts = np.triu(random_state.randint(
        0, 100, (lengths.sum(), lengths.sum())))
    counts = counts + counts.T
    downsampled_counts, downsampled_lengths = downsample_resolution(
        counts, lengths)
    assert_equal(downsampled_lengths.sum(), 11)


def test_undersample_per_chr():
    X = np.array([[1, 1, 0, 0],
                  [1, 1, 0, 0],
                  [0, 0, 0.5, 0.5],
                  [0, 0, 0.5, 0.5]])
    lengths = np.array([2, 2])
    undersampled_X = undersample_per_chr(X, lengths)
    undersampled_X_true = np.array([[1, 0],
                                    [0, 0.5]])
    assert_array_equal(undersampled_X_true, undersampled_X)


def test_return_sample():
    lengths = np.array([50, 75])
    n = lengths.sum()
    X = np.random.randint(0, 50, (n, n))
    X = np.triu(X)
    sub_X, _ = extract_sub_contact_map(X, lengths, [0])
    assert_array_equal(X[:lengths[0], :lengths[0]],
                       sub_X)


def test_get_genomic_distances():
    lengths = np.array([5, 5])
    dense_gdis = get_genomic_distances(lengths)

    # FIXME we should test this!!
    true_gdis = dense_gdis

    # Now test sparse matrix
    random_state = np.random.RandomState(seed=42)

    m = 15
    rows = random_state.randint(0, 10, size=(m,))
    cols = random_state.randint(0, 10, size=(m,))
    counts = np.zeros((10, 10))
    counts[rows, cols] += 1
    counts = sparse.coo_matrix(np.triu(counts))
    rows = counts.row
    cols = counts.col
    sparse_gdis = get_genomic_distances(lengths, counts=counts)
    sparse_true_gdis = true_gdis[rows, cols]
    assert_array_equal(sparse_gdis, sparse_true_gdis)

    # Providing a matrix that isn't coo
    counts = counts.tocsr()
    sparse_gdis = get_genomic_distances(lengths, counts=counts)
    sparse_true_gdis = true_gdis[rows, cols]
    assert_array_equal(sparse_gdis, sparse_true_gdis)
