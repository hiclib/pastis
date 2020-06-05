import sys
import pytest
import numpy as np
from sklearn.metrics import euclidean_distances
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy import sparse

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 6), reason="Requires python3.6 or higher")

if sys.version_info[0] >= 3:
    from pastis.optimization import multiscale_optimization
    from pastis.optimization import pastis_algorithms
    from pastis.optimization import poisson
    from pastis.optimization.counts import preprocess_counts


def decrease_struct_res_correct(struct, multiscale_factor, lengths, ploidy):
    if multiscale_factor == 1:
        return struct

    struct = struct.copy().reshape(-1, 3)
    lengths = np.array(lengths).astype(int)

    struct_lowres = []
    begin = end = 0
    for length in np.tile(lengths, ploidy):
        end += length
        struct_chrom = struct[begin:end]
        remainder = struct_chrom.shape[0] % multiscale_factor
        struct_chrom_reduced = np.nanmean(
            struct_chrom[:struct_chrom.shape[0] - remainder, :].reshape(
                -1, multiscale_factor, 3), axis=1)
        if remainder == 0:
            struct_lowres.append(struct_chrom_reduced)
        else:
            struct_chrom_overhang = np.nanmean(
                struct_chrom[struct_chrom.shape[0] - remainder:, :],
                axis=0).reshape(-1, 3)
            struct_lowres.extend([struct_chrom_reduced, struct_chrom_overhang])
        begin = end

    return np.concatenate(struct_lowres)


def decrease_counts_res_correct(counts, multiscale_factor, lengths):
    if multiscale_factor == 1:
        return counts

    is_sparse = sparse.issparse(counts)
    if is_sparse:
        counts = counts.toarray()
    lengths = np.array(lengths).astype(int)
    triu = counts.shape[0] == counts.shape[1]
    if triu:
        counts = np.triu(counts, 1)

    lengths_lowres = np.ceil(
        lengths.astype(float) / multiscale_factor).astype(int)
    map_factor_row = int(counts.shape[0] / lengths.sum())
    map_factor_col = int(counts.shape[1] / lengths.sum())

    counts_lowres = np.zeros((
        lengths_lowres.sum() * map_factor_row,
        lengths_lowres.sum() * map_factor_col), dtype=counts.dtype)
    tiled_lengths = np.tile(lengths, max(map_factor_row, map_factor_col))
    tiled_lengths_lowres = np.tile(
        lengths_lowres, max(map_factor_row, map_factor_col))

    for c1 in range(lengths.shape[0] * map_factor_row):
        for i in range(tiled_lengths[c1]):
            row_fullres = tiled_lengths[:c1].sum() + i
            i_lowres = int(np.ceil(float(i + 1) / multiscale_factor) - 1)
            row_lowres = tiled_lengths_lowres[:c1].sum() + i_lowres
            if triu:
                c2_start = c1
            else:
                c2_start = 0
            for c2 in range(c2_start, lengths.shape[0] * map_factor_col):
                if triu and c1 == c2:
                    j_start = i + 1
                else:
                    j_start = 0
                for j in range(j_start, tiled_lengths[c2]):
                    col_fullres = tiled_lengths[:c2].sum() + j
                    j_lowres = int(np.ceil(float(j + 1) / multiscale_factor) - 1)
                    col_lowres = tiled_lengths_lowres[:c2].sum() + j_lowres
                    bin_fullres = counts[row_fullres, col_fullres]
                    if triu:
                        on_diag = c1 == c2 and i_lowres == j_lowres
                    elif c1 >= lengths.shape[0]:
                        on_diag = (c1 - lengths.shape[0]) == c2 and i_lowres == j_lowres
                    elif c2 >= lengths.shape[0]:
                        on_diag = c1 == (c2 - lengths.shape[0]) and i_lowres == j_lowres
                    else:
                        on_diag = c1 == c2 and i_lowres == j_lowres
                    if not np.isnan(bin_fullres) and not on_diag:
                        counts_lowres[row_lowres, col_lowres] += bin_fullres

    if is_sparse:
        counts_lowres = sparse.coo_matrix(counts_lowres)

    return counts_lowres


@pytest.mark.parametrize("multiscale_factor", [1, 2, 3, 4])
def test_decrease_lengths_res(multiscale_factor):
    lengths_lowres_true = np.array([1, 2, 3, 4, 5])
    lengths_fullres = lengths_lowres_true * multiscale_factor
    lengths_lowres = multiscale_optimization.decrease_lengths_res(
        lengths_fullres, multiscale_factor=multiscale_factor)
    assert_array_equal(lengths_lowres_true, lengths_lowres)


def test_increase_struct_res():
    lengths = np.array([10, 21])
    multiscale_factor = 2
    ploidy = 1
    nan_indices_lowres = np.array([0, 4])

    nbeads = lengths.sum() * ploidy
    coord0 = np.arange(nbeads * ploidy, dtype=float).reshape(-1, 1)
    coord1 = coord2 = np.zeros_like(coord0)

    struct_highres_true = np.concatenate(
        [coord0, coord1, coord2], axis=1)

    struct_lowres = decrease_struct_res_correct(
        struct_highres_true, multiscale_factor=multiscale_factor,
        lengths=lengths, ploidy=ploidy)
    struct_lowres[nan_indices_lowres] = np.nan

    struct_highres = multiscale_optimization.increase_struct_res(
        struct_lowres, multiscale_factor=multiscale_factor, lengths=lengths)
    assert_array_almost_equal(
        struct_highres_true, struct_highres)


@pytest.mark.parametrize("multiscale_factor", [1, 2, 3, 4])
def test_decrease_counts_res(multiscale_factor):
    lengths = np.array([10, 21])
    ploidy = 2
    seed = 42
    alpha, beta = -3., 1.
    ratio_ambig, ratio_pa, ratio_ua = [1 / 3] * 3
    nan_indices = np.array([0, 1, 2, 3, 12, 15, 25, 40])

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    X_true = random_state.rand(n * ploidy, 3)
    dis = euclidean_distances(X_true)
    dis[dis == 0] = np.inf
    poisson_intensity = dis ** alpha

    ambig_counts = ratio_ambig * beta * poisson_intensity
    ambig_counts[np.isnan(ambig_counts) | np.isinf(ambig_counts)] = 0
    ambig_counts = ambig_counts[:n, :n] + ambig_counts[n:, n:] + ambig_counts[:n, n:] + ambig_counts[n:, :n]
    ambig_counts = np.triu(ambig_counts, 1)
    ambig_counts[nan_indices[nan_indices < n], :] = np.nan
    ambig_counts[:, nan_indices[nan_indices < n]] = np.nan

    pa_counts = ratio_pa * beta * poisson_intensity
    pa_counts[np.isnan(pa_counts) | np.isinf(pa_counts)] = 0
    pa_counts = pa_counts[:, :n] + pa_counts[:, n:]
    np.fill_diagonal(pa_counts[:n, :], 0)
    np.fill_diagonal(pa_counts[n:, :], 0)
    pa_counts[nan_indices, :] = np.nan
    pa_counts[:, nan_indices[nan_indices < n]] = np.nan

    ua_counts = ratio_ua * beta * poisson_intensity
    ua_counts[np.isnan(ua_counts) | np.isinf(ua_counts)] = 0
    ua_counts = np.triu(ua_counts, 1)
    ua_counts[nan_indices, :] = np.nan
    ua_counts[:, nan_indices] = np.nan

    ambig_counts_lowres_true = decrease_counts_res_correct(
        ambig_counts, multiscale_factor=multiscale_factor, lengths=lengths)
    ambig_counts_lowres = multiscale_optimization.decrease_counts_res(
        ambig_counts, multiscale_factor=multiscale_factor, lengths=lengths,
        ploidy=ploidy)
    ambig_counts_lowres[np.isnan(ambig_counts_lowres)] = 0
    assert_array_equal(ambig_counts_lowres_true, ambig_counts_lowres)

    pa_counts_lowres_true = decrease_counts_res_correct(
        pa_counts, multiscale_factor=multiscale_factor, lengths=lengths)
    pa_counts_lowres = multiscale_optimization.decrease_counts_res(
        pa_counts, multiscale_factor=multiscale_factor, lengths=lengths,
        ploidy=ploidy)
    pa_counts_lowres[np.isnan(pa_counts_lowres)] = 0
    assert_array_equal(pa_counts_lowres_true, pa_counts_lowres)

    ua_counts_lowres_true = decrease_counts_res_correct(
        ua_counts, multiscale_factor=multiscale_factor, lengths=lengths)
    ua_counts_lowres = multiscale_optimization.decrease_counts_res(
        ua_counts, multiscale_factor=multiscale_factor, lengths=lengths,
        ploidy=ploidy)
    ua_counts_lowres[np.isnan(ua_counts_lowres)] = 0
    assert_array_equal(ua_counts_lowres_true, ua_counts_lowres)


@pytest.mark.parametrize("multiscale_factor", [1, 2, 3, 4])
def test_decrease_struct_res(multiscale_factor):
    lengths = np.array([10, 21])
    ploidy = 1
    seed = 42
    nan_indices = np.array([0, 1, 2, 3, 12, 15, 25])

    nbeads = lengths.sum() * ploidy
    random_state = np.random.RandomState(seed=seed)
    struct = random_state.rand(nbeads, 3)
    struct[nan_indices] = np.nan

    struct_lowres_true = decrease_struct_res_correct(
        struct, multiscale_factor=multiscale_factor, lengths=lengths,
        ploidy=ploidy)
    struct_lowres = multiscale_optimization.decrease_struct_res(
        struct, multiscale_factor=multiscale_factor, lengths=lengths)

    assert_array_almost_equal(
        struct_lowres_true, struct_lowres)


@pytest.mark.parametrize("multiscale_factor", [2, 3, 4])
def test_get_multiscale_variances_from_struct(multiscale_factor):
    lengths = np.array([10, 21])
    seed = 42
    nan_indices = np.array([0, 1, 2, 3, 12, 15, 25])

    nbeads = lengths.sum()
    random_state = np.random.RandomState(seed=seed)
    coord0 = random_state.rand(nbeads, 1)
    coord1 = coord2 = np.zeros_like(coord0)
    struct = np.concatenate([coord0, coord1, coord2], axis=1)
    coord0[nan_indices] = np.nan
    struct[nan_indices] = np.nan

    multiscale_variances_true = []
    begin = end = 0
    for l in lengths:
        end += l
        for i in range(begin, end, multiscale_factor):
            slice = coord0[i:min(end, i + multiscale_factor)]
            if np.isnan(slice).sum() == slice.shape[0]:
                multiscale_variances_true.append(np.nan)
            else:
                multiscale_variances_true.append(np.var(slice[~np.isnan(slice)]))
        begin = end
    multiscale_variances_true = np.array(multiscale_variances_true)
    multiscale_variances_true[np.isnan(multiscale_variances_true)] = np.nanmedian(
        multiscale_variances_true)

    multiscale_variances_infer = multiscale_optimization.get_multiscale_variances_from_struct(
        struct, lengths=lengths, multiscale_factor=multiscale_factor)

    assert_array_almost_equal(
        multiscale_variances_true, multiscale_variances_infer)


def test__choose_max_multiscale_factor():
    lengths = np.array([101])

    for min_beads in (5, 10, 11, 100, 101, 200):
        multiscale_rounds = multiscale_optimization._choose_max_multiscale_factor(
            lengths, min_beads)
        assert multiscale_optimization.decrease_lengths_res(
            lengths, 2 ** (multiscale_rounds + 1)).min() <= min_beads


def test_infer_multiscale_variances_ambig():
    lengths = np.array([50])
    ploidy = 2
    seed = 42
    hsc_r = None
    alpha, beta = -3., 1.
    multiscale_rounds = 4

    multiscale_factor = 2 ** multiscale_rounds
    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    X_true = random_state.rand(n * ploidy, 3)
    dis = euclidean_distances(X_true)
    dis[dis == 0] = np.inf
    counts = beta * dis ** alpha
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts = counts[:n, :n] + counts[n:, n:] + counts[:n, n:] + counts[n:, :n]
    counts = np.triu(counts, 1)
    counts = sparse.coo_matrix(counts)

    multiscale_variances_true = multiscale_optimization.get_multiscale_variances_from_struct(
        X_true, lengths=lengths, multiscale_factor=multiscale_factor)

    struct_draft_fullres, _, _, _, draft_converged = pastis_algorithms._infer_draft(
        counts, lengths=lengths, ploidy=ploidy, outdir=None, alpha=alpha,
        seed=seed, normalize=False, filter_threshold=0, beta=beta,
        multiscale_rounds=multiscale_rounds, use_multiscale_variance=True,
        hsc_lambda=0., hsc_r=hsc_r,
        callback_freq={'print': None, 'history': None, 'save': None})

    assert draft_converged

    multiscale_variances_infer = multiscale_optimization.get_multiscale_variances_from_struct(
        struct_draft_fullres, lengths=lengths,
        multiscale_factor=multiscale_factor)

    median_true = np.median(multiscale_variances_true)
    median_infer = np.median(multiscale_variances_infer)
    assert_array_almost_equal(median_true, median_infer, decimal=1)


def test_poisson_objective_multiscale_ambig():
    lengths = np.array([20])
    ploidy = 2
    seed = 42
    alpha, beta = -3., 1.
    multiscale_rounds = 4

    multiscale_factor = 2 ** multiscale_rounds
    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    X_true = random_state.rand(n * ploidy, 3)
    dis = euclidean_distances(X_true)
    dis[dis == 0] = np.inf
    counts = beta * dis ** alpha
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts = counts[:n, :n] + counts[n:, n:] + counts[:n, n:] + counts[n:, :n]
    counts = np.triu(counts, 1)
    counts = sparse.coo_matrix(counts)

    _, _, _, fullres_torm_for_multiscale = preprocess_counts(
        counts_raw=counts, lengths=lengths, ploidy=ploidy, normalize=False,
        filter_threshold=0., multiscale_factor=1, beta=beta)

    counts, _, torm, _ = preprocess_counts(
        counts_raw=counts, lengths=lengths, ploidy=ploidy, normalize=False,
        filter_threshold=0., multiscale_factor=multiscale_factor, beta=beta,
        fullres_torm=fullres_torm_for_multiscale)

    multiscale_variances_true = multiscale_optimization.get_multiscale_variances_from_struct(
        X_true, lengths=lengths, multiscale_factor=multiscale_factor)

    obj = poisson.objective(
        X=X_true, counts=counts, alpha=alpha, lengths=lengths, bias=None,
        multiscale_factor=multiscale_factor,
        multiscale_variances=multiscale_variances_true)

    assert obj < -1e4
