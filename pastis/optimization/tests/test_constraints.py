import sys
import pytest
import numpy as np
from sklearn.metrics import euclidean_distances
from scipy import sparse
from numpy.testing import assert_allclose

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 6), reason="Requires python3.6 or higher")

if sys.version_info[0] >= 3:
    from pastis.optimization import constraints
    from pastis.optimization.counts import _format_counts
    from pastis.optimization.multiscale_optimization import decrease_struct_res


def test_bcc_constraint():
    lengths = np.array([20])
    ploidy = 2
    alpha, beta = -3., 1.

    n = lengths.sum()
    X_true = np.concatenate(
        [np.arange(n * ploidy).reshape(-1, 1), np.zeros((n * ploidy, 1)),
            np.zeros((n * ploidy, 1))], axis=1)
    dis = euclidean_distances(X_true)
    dis[dis == 0] = np.inf
    counts = beta * dis ** alpha
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts = np.triu(counts, 1)
    counts = sparse.coo_matrix(counts)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta)

    constraint = constraints.Constraints(
        counts, lengths=lengths, ploidy=ploidy, multiscale_factor=1,
        constraint_lambdas={'bcc': 1},
        constraint_params=None)
    constraint.check()
    obj = constraint.apply(X_true)['obj_bcc']
    assert obj < 1e-6


def test_hsc_constraint():
    lengths = np.array([30])
    ploidy = 2
    seed = 42
    true_interhomo_dis = np.array([10.])
    alpha, beta = -3., 1.

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()

    X_true = np.zeros((n * ploidy, 3), dtype=float)
    for i in range(X_true.shape[0]):
        X_true[i:, random_state.choice([0, 1, 2])] += 1

    X_true[n:] -= X_true[n:].mean(axis=0)
    X_true[:n] -= X_true[:n].mean(axis=0)
    begin = end = 0
    for i in range(len(lengths)):
        end += lengths[i]
        X_true[begin:end, 0] += true_interhomo_dis[i]
        begin = end

    dis = euclidean_distances(X_true)
    dis[dis == 0] = np.inf
    counts = beta * dis ** alpha
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts = np.triu(counts, 1)
    counts = sparse.coo_matrix(counts)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta)

    constraint = constraints.Constraints(
        counts, lengths=lengths, ploidy=ploidy, multiscale_factor=1,
        constraint_lambdas={'hsc': 1},
        constraint_params={'hsc': true_interhomo_dis})
    constraint.check()
    obj = constraint.apply(X_true)['obj_hsc']
    assert obj < 1e-6


def test__mean_interhomolog_counts_unambig():
    lengths = np.array([10, 20])
    ploidy = 2
    seed = 42
    alpha, beta = -3., 1.
    mean_chrom_coords = np.array([[0, 3, 0], [-4, 0, 0], [0, -3, 0], [4, 0, 0]])

    n = lengths.sum()
    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    X_true = random_state.rand(n * ploidy, 3)

    begin = end = 0
    for i in range(len(lengths) * ploidy):
        end += np.tile(lengths, ploidy)[i]
        X_true[begin:end] -= X_true[begin:end].mean(axis=0)
        X_true[begin:end] += mean_chrom_coords[i]
        begin = end

    dis = euclidean_distances(X_true)
    dis[dis == 0] = np.inf
    counts = beta * dis ** alpha
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts = np.triu(counts, 1)

    true_interhomo_dis = constraints._inter_homolog_dis(X_true, lengths=lengths)

    bias = 0.1 + random_state.rand(n)
    counts_biased = counts * np.tile(bias, 2).reshape(-1, 1) * \
        np.tile(bias, 2).reshape(-1, 1).T

    ua_counts = _format_counts(
        counts=sparse.coo_matrix(counts), lengths=lengths, ploidy=ploidy,
        beta=beta)
    ua_counts_biased = _format_counts(
        counts=sparse.coo_matrix(counts_biased), lengths=lengths, ploidy=ploidy,
        beta=beta)

    mhs_k_ua = constraints._mean_interhomolog_counts(ua_counts, lengths=lengths)
    mhs_k_ua_biased = constraints._mean_interhomolog_counts(
        ua_counts_biased, lengths=lengths, bias=bias)

    assert_allclose(mhs_k_ua, mhs_k_ua_biased)
    assert_allclose(mhs_k_ua ** (1 / alpha), true_interhomo_dis, rtol=1e-2)


def test__mean_interhomolog_counts_ambig():
    lengths = np.array([10, 20])
    ploidy = 2
    seed = 42
    alpha, beta = -3., 1.
    mean_chrom_coords = np.array([[0, 3, 0], [-4, 0, 0], [0, -3, 0], [4, 0, 0]])

    n = lengths.sum()
    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    X_true = random_state.rand(n * ploidy, 3)

    begin = end = 0
    for i in range(len(lengths) * ploidy):
        end += np.tile(lengths, ploidy)[i]
        X_true[begin:end] -= X_true[begin:end].mean(axis=0)
        X_true[begin:end] += mean_chrom_coords[i]
        begin = end

    dis = euclidean_distances(X_true)
    dis[dis == 0] = np.inf
    counts = beta * dis ** alpha
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts = np.triu(counts, 1)

    nchrom = lengths.shape[0]
    X_true_lowres = decrease_struct_res(
        X_true, multiscale_factor=lengths.max(), lengths=lengths)
    dis_lowres = euclidean_distances(X_true_lowres)
    dis_lowres[np.tril_indices(dis_lowres.shape[0])] = np.nan
    np.fill_diagonal(dis_lowres[:nchrom, nchrom:], np.nan)
    approx_interhomo_dis_ambig = np.nanmean(dis_lowres)

    ambig_counts_raw = counts[:n, :n] + counts[n:, n:] + counts[:n, n:] + \
        counts[n:, :n]
    ambig_counts = _format_counts(
        counts=sparse.coo_matrix(ambig_counts_raw), lengths=lengths,
        ploidy=ploidy, beta=beta)

    bias = 0.1 + random_state.rand(n)
    ambig_counts_biased = ambig_counts_raw * bias.reshape(-1, 1) * \
        bias.reshape(-1, 1).T
    ambig_counts_biased = _format_counts(
        counts=sparse.coo_matrix(ambig_counts_biased), lengths=lengths,
        ploidy=ploidy, beta=beta)

    mhs_k_ambig = constraints._mean_interhomolog_counts(
        ambig_counts, lengths=lengths)
    mhs_k_ambig_biased = constraints._mean_interhomolog_counts(
        ambig_counts_biased, lengths=lengths, bias=bias)

    assert_allclose(mhs_k_ambig, mhs_k_ambig_biased)
    assert_allclose(
        mhs_k_ambig ** (1 / alpha), approx_interhomo_dis_ambig, rtol=1e-1)


def test_mhs_constraint():
    lengths = np.array([30])
    ploidy = 2
    seed = 42
    true_interhomo_dis = np.array([10.])
    alpha, beta = -3., 1.

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()

    X_true = np.zeros((n * ploidy, 3), dtype=float)
    for i in range(X_true.shape[0]):
        X_true[i:, random_state.choice([0, 1, 2])] += 1

    X_true[n:] -= X_true[n:].mean(axis=0)
    X_true[:n] -= X_true[:n].mean(axis=0)
    begin = end = 0
    for i in range(len(lengths)):
        end += lengths[i]
        X_true[begin:end, 0] += true_interhomo_dis[i]
        begin = end

    dis = euclidean_distances(X_true)
    dis[dis == 0] = np.inf
    counts = beta * dis ** alpha
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts = np.triu(counts, 1)
    counts = sparse.coo_matrix(counts)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta)

    mhs_k = constraints._mean_interhomolog_counts(counts, lengths=lengths)

    constraint = constraints.Constraints(
        counts, lengths=lengths, ploidy=ploidy, multiscale_factor=1,
        constraint_lambdas={'mhs': 1},
        constraint_params={'mhs': mhs_k})
    constraint.check()
    obj = constraint.apply(X_true, alpha=alpha)['obj_mhs']
    assert obj < 1e-3
