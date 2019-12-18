import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy import sparse
from iced.normalization import _ca_utils as utils


def test_identify_missing_distances():
    random_state = np.random.RandomState(seed=42)
    counts = random_state.randint(0, 100, (100, 100))
    missing_loci = random_state.rand(100) > 0.95
    lengths = np.array([75, 25])
    counts[missing_loci] = 0
    counts[:, missing_loci] = 0
    counts = np.triu(counts, k=1)
    counts = counts + counts.T

    gdis_dense = utils.get_genomic_distances(
        lengths)

    gdis_dense, num_dense = np.unique(
        gdis_dense[missing_loci],
        return_counts=True)
    num_dense[1] = 0
    gdis, num = utils.identify_missing_distances(
        sparse.coo_matrix(np.triu(counts, 1)), lengths)
    # assert_array_equal(num[:len(num_dense)], num_dense)


def test_get_genomic_distances():
    random_state = np.random.RandomState(seed=42)
    lengths = np.array([25, 75])
    counts = random_state.randint(0, 100, (100, 100))
    missing_loci = random_state.rand(100) > 0.95
    counts[missing_loci] = 0
    counts[:, missing_loci] = 0
    counts = np.triu(counts, k=1).astype(float)
    counts = counts + counts.T

    counts_sparse = sparse.coo_matrix(np.triu(counts), shape=counts.shape)

    gdis_dense = utils.get_genomic_distances(lengths, counts)
    gdis_sparse = utils.get_genomic_distances(lengths, counts_sparse)

    gdis_sparse = sparse.coo_matrix(
        (gdis_sparse, (counts_sparse.row, counts_sparse.col)),
        shape=counts.shape).toarray()
    gdis_dense[gdis_sparse == 0] = 0
    assert_array_equal(gdis_dense, gdis_sparse)


def test_get_mapping():
    random_state = np.random.RandomState(seed=42)
    counts = random_state.randint(0, 100, (100, 100))
    missing_loci = random_state.rand(100) > 0.9
    counts[missing_loci] = 0
    counts[:, missing_loci] = 0
    counts = np.triu(counts, k=1).astype(float)
    counts = counts + counts.T

    counts_sparse = sparse.coo_matrix(np.triu(counts), shape=counts.shape)

    bs = np.ones(counts.shape)
    lengths = np.array([50, 25, 25])
    counts[missing_loci] = np.nan
    counts[:, missing_loci] = np.nan

    counts_dense = counts

    mapping_dense = utils.get_mapping(
        counts_dense, lengths, bs,
        smoothed=False)
    mapping_sparse = utils.get_mapping(
        counts_sparse, lengths, np.ones(counts_sparse.data.shape),
        smoothed=False)
    assert_array_equal(mapping_dense, mapping_sparse)


def test_expected_dense_sparse():
    random_state = np.random.RandomState(seed=42)
    counts = random_state.randint(0, 100, (100, 100))
    missing_loci = random_state.rand(100) > 0.9
    counts[missing_loci] = 0
    counts[:, missing_loci] = 0
    counts = np.triu(counts, k=1).astype(float)
    counts = counts + counts.T

    counts_sparse = sparse.coo_matrix(np.triu(counts), shape=counts.shape)

    bs = np.ones(counts.shape)
    lengths = np.array([75, 25])
    counts[missing_loci] = np.nan
    counts[:, missing_loci] = np.nan

    expected_dense = utils.get_expected(counts, lengths, bs)
    expected_sparse = utils.get_expected(counts_sparse, lengths,
                                         np.ones(counts_sparse.data.shape))
    expected_sparse = sparse.coo_matrix(
        (expected_sparse, (counts_sparse.row, counts_sparse.col)),
        shape=counts.shape).toarray()
    expected_dense[expected_sparse == 0] = 0
    assert_array_almost_equal(expected_sparse,
                              np.triu(expected_dense))


def test_estimate_bias_dense_sparse():
    random_state = np.random.RandomState(seed=42)
    counts = random_state.randint(0, 100, (100, 100))
    missing_loci = random_state.rand(100) > 0.95
    counts[missing_loci] = 0
    counts[:, missing_loci] = 0

    counts = np.triu(counts, k=1).astype(float)
    counts = counts + counts.T
    # Add some sparsity
    counts_sparse = sparse.coo_matrix(np.triu(counts), shape=counts.shape)
    counts[missing_loci] = np.nan
    counts[:, missing_loci] = np.nan

    bs_dense = np.ones(counts.shape)
    bs_sparse = np.ones(counts_sparse.data.shape)

    lengths = np.array([75, 25])
    cnv = 2 * np.ones(lengths.sum())
    cnv[:random_state.randint(0, 100)] += 2
    cnv[random_state.randint(0, 100):random_state.randint(0, 100)] -= 1

    mapping = utils.get_mapping(counts_sparse, lengths, bs_sparse)
    expected_dense = utils.get_expected(counts, lengths, bs_dense,
                                        mapping=mapping)
    expected_sparse = utils.get_expected(counts_sparse, lengths, bs_sparse,
                                         mapping=mapping)

    bias_dense = utils.estimate_bias(counts, cnv, expected_dense, lengths)
    bias_sparse = utils.estimate_bias(
        counts_sparse, cnv, expected_sparse, lengths, mapping)

    bias_sparse = sparse.coo_matrix(
        (bias_sparse, (counts_sparse.row, counts_sparse.col)),
        shape=counts.shape).toarray()
    bias_dense[bias_sparse == 0] = 0
    assert_array_almost_equal(bias_dense, bias_sparse)


def test_num_each_dis():
    random_state = np.random.RandomState(seed=42)
    lengths = random_state.randint(0, 50, 5)
    lengths.sort()
    b = random_state.randint(0, lengths.sum()/2)
    e = random_state.randint(b, lengths.sum())

    # Generate rows and cols
    rows = np.arange(b, e)
    b = random_state.randint(0, lengths.sum()/2)
    e = random_state.randint(b, lengths.sum())
    cols = np.arange(b, e)

    gdis, num = utils._num_each_gdis(rows, cols, lengths)

    gdistances = utils.get_genomic_distances(lengths)
    gdistances = np.triu(gdistances)
    gdis_dense, num_dense = np.unique(
        gdistances[rows][:, cols], return_counts=True)
    m = np.array([i in gdis for i in gdis_dense])
    assert_array_equal(num, num_dense[m])
