import os

import numpy as np
from scipy import sparse
from warnings import warn

from iced.filter import filter_low_counts
from iced.normalization import ICE_normalization

from .constraints import _constraint_dis_indices
from .utils_poisson import find_beads_to_remove
from .utils_poisson import _intra_counts, _inter_counts

from .multiscale_optimization import decrease_lengths_res
from .multiscale_optimization import decrease_counts_res
from .multiscale_optimization import _count_fullres_per_lowres_bead


def ambiguate_counts(counts, lengths, ploidy, exclude_zeros=False):
    """Convert diploid counts to ambiguous & aggregate counts across matrices.

    If diploid, convert unambiguous and partially ambiguous counts to ambiguous
    and aggregate list of counts into a single counts matrix. If haploid,
    check format of and return the inputted counts matrix.

    Parameters
    ----------
    counts : list of array or coo_matrix or CountsMatrix instances
        Counts data.
    lengths : array_like of int
        Number of beads per homolog of each chromosome.

    Returns
    -------
    coo_matrix or ndarray
        Aggregated and ambiguated contact counts matrix.
    """

    lengths = np.array(lengths)
    n = lengths.sum()

    if not isinstance(counts, list):
        counts = [counts]

    if len(counts) == 1 and counts[0].shape == (n, n):
        return _check_counts_matrix(
            counts[0], lengths=lengths, ploidy=ploidy,
            exclude_zeros=exclude_zeros)

    output = np.zeros((n, n))
    for c in counts:
        if not isinstance(c, ZeroCountsMatrix):
            if not isinstance(c, np.ndarray):
                c = c.toarray()
            c = _check_counts_matrix(
                c, lengths=lengths, ploidy=ploidy, exclude_zeros=True).toarray()
            if c.shape[0] > c.shape[1]:
                c_ambig = np.nansum(
                    [c[:n, :], c[n:, :], c[:n, :].T, c[n:, :].T], axis=0)
            elif c.shape[0] < c.shape[1]:
                c_ambig = np.nansum(
                    [c[:, :n].T, c[:, n:].T, c[:, :n], c[:, n:]], axis=0)
            elif c.shape[0] == n:
                c_ambig = c
            else:
                c_ambig = np.nansum(
                    [c[:n, :n], c[:n, n:], c[:n, n:].T, c[n:, n:]], axis=0)
            output[~np.isnan(c_ambig)] += c_ambig[~np.isnan(c_ambig)]

    output = np.triu(output, 1)
    return _check_counts_matrix(
        output, lengths=lengths, ploidy=ploidy, exclude_zeros=exclude_zeros)


def _create_dummy_counts(counts, lengths, ploidy):
    """Create sparse matrix of 1's with same row and col as input counts.
    """

    rows, cols = _constraint_dis_indices(
        counts, n=lengths.sum(), lengths=lengths, ploidy=ploidy)
    dummy_counts = sparse.coo_matrix((np.ones_like(rows), (rows, cols)), shape=(
        lengths.sum() * ploidy, lengths.sum() * ploidy)).toarray()
    dummy_counts = sparse.coo_matrix(np.triu(dummy_counts + dummy_counts.T, 1))

    return dummy_counts


def _get_chrom_subset_index(ploidy, lengths_full, chrom_full, chrom_subset):
    """Return indices for selected chromosomes only.
    """

    if isinstance(chrom_subset, str):
        chrom_subset = np.array([chrom_subset])
    missing_chrom = [x for x in chrom_subset if x not in chrom_full]
    if len(missing_chrom) > 0:
        raise ValueError("Chromosomes to be subsetted (%s) are not in full"
                         "list of chromosomes (%s)" %
                         (','.join(missing_chrom), ','.join(chrom_full)))

    lengths_subset = lengths_full.copy()
    index = None
    if not np.array_equal(chrom_subset, chrom_full):
        lengths_subset = np.array([lengths_full[i] for i in range(
            len(chrom_full)) if chrom_full[i] in chrom_subset])
        index = []
        for i in range(len(lengths_full)):
            index.append(
                np.full((lengths_full[i],), chrom_full[i] in chrom_subset))
        index = np.concatenate(index)
        if ploidy == 2:
            index = np.tile(index, 2)
    return index, lengths_subset


def subset_chrom(ploidy, lengths_full, chrom_full, chrom_subset=None,
                 counts=None, exclude_zeros=False, struct_true=None):
    """Return data for selected chromosomes only.

    If `chrom_subset` is None, return original data. Otherwise, only return
    data for chromosomes specified by `chrom_subset`.

    Parameters
    ----------
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    lengths_full : array of int
        Number of beads per homolog of each chromosome in the full data.
    chrom_full : array of str
        Label for each chromosome in the full data, or file with chromosome
        labels (one label per line).
    chrom_subset : array of str, optional
        Label for each chromosome to be excised from the full data. If None,
        the full data will be returned.
    counts : list of array or coo_matrix, optional
        Full counts data.

    Returns
    -------
    lengths_subset : array of int
        Number of beads per homolog of each chromosome in the subsetted data
        for the specified chromosomes.
    chrom_subset : array of str
        Label for each chromosome in the subsetted data.
    counts : coo_matrix of int or ndarray or int
        If `counts` is inputted, subsetted counts data containing only the
        specified chromosomes. Otherwise, return None.
    struct_true : None or array of float
        If `struct_true` is inputted, subsetted true structure containing only
        the specified chromosomes. Otherwise, return None.
    """

    if chrom_subset is not None:
        if isinstance(chrom_subset, str):
            chrom_subset = np.array([chrom_subset])
        missing_chrom = [x for x in chrom_subset if x not in chrom_full]
        if len(missing_chrom) > 0:
            raise ValueError("Chromosomes to be subsetted (%s) are not in full"
                             "list of chromosomes (%s)" %
                             (','.join(missing_chrom), ','.join(chrom_full)))
        # Make sure chrom_subset is sorted properly
        chrom_subset = [chrom for chrom in chrom_full if chrom in chrom_subset]

    if chrom_subset is None or np.array_equal(chrom_subset, chrom_full):
        chrom_subset = chrom_full.copy()
        lengths_subset = lengths_full.copy()
        if counts is not None:
            counts = check_counts(
                counts, lengths=lengths_full, ploidy=ploidy,
                exclude_zeros=exclude_zeros)
        return lengths_subset, chrom_subset, counts, struct_true
    else:
        index, lengths_subset = _get_chrom_subset_index(
            ploidy, lengths_full, chrom_full, chrom_subset)

        if struct_true is not None and index is not None:
            struct_true = struct_true[index]

        if counts is not None:
            counts = check_counts(
                counts, lengths=lengths_full, ploidy=ploidy,
                exclude_zeros=exclude_zeros, chrom_subset_index=index)

        return lengths_subset, chrom_subset, counts, struct_true


def _check_counts_matrix(counts, lengths, ploidy, exclude_zeros=False,
                         chrom_subset_index=None):
    """Check counts dimensions, reformat, & excise selected chromosomes.
    """

    if chrom_subset_index is not None and len(chrom_subset_index) / max(counts.shape) not in (1, 2):
        raise ValueError("chrom_subset_index size (%d) does not fit counts"
                         " shape (%d, %d)." %
                         (len(chrom_subset_index), counts.shape[0],
                             counts.shape[1]))
    if len(counts.shape) != 2:
        raise ValueError(
            "Counts matrix must be two-dimensional, current shape = (%s)"
            % ', '.join([str(x) for x in counts.shape]))
    if any([x > lengths.sum() * ploidy for x in counts.shape]):
        raise ValueError("Counts matrix shape (%d, %d) is greater than number"
                         " of beads (%d) in %s genome." %
                         (counts.shape[0], counts.shape[1],
                             lengths.sum() * ploidy,
                             {1: "haploid", 2: "diploid"}[ploidy]))
    if any([x / lengths.sum() not in (1, 2) for x in counts.shape]):
        raise ValueError("Counts matrix shape (%d, %d) does not match lenghts"
                         " (%s)"
                         % (counts.shape[0], counts.shape[1],
                             ",".join(map(str, lengths))))

    empty_val = 0
    torm = np.full((max(counts.shape)), False)
    if not exclude_zeros:
        empty_val = np.nan
        torm = find_beads_to_remove(counts, max(counts.shape))
        counts = counts.astype(float)

    if sparse.issparse(counts) or isinstance(counts, CountsMatrix):
        counts = counts.toarray()
    if not isinstance(counts, np.ndarray):
        counts = np.array(counts)

    if not np.array_equal(counts[~np.isnan(counts)],
                          counts[~np.isnan(counts)].round()):
        warn("Counts matrix must only contain integers or NaN")

    if counts.shape[0] == counts.shape[1]:
        counts[np.tril_indices(counts.shape[0])] = empty_val
        counts[torm, :] = empty_val
        counts[:, torm] = empty_val
        if chrom_subset_index is not None:
            counts = counts[chrom_subset_index[:counts.shape[0]], :][
                :, chrom_subset_index[:counts.shape[1]]]
    elif min(counts.shape) * 2 == max(counts.shape):
        homo1 = counts[:min(counts.shape), :min(counts.shape)]
        homo2 = counts[counts.shape[0] -
                       min(counts.shape):, counts.shape[1] - min(counts.shape):]
        if counts.shape[0] == min(counts.shape):
            homo1 = homo1.T
            homo2 = homo2.T
        np.fill_diagonal(homo1, empty_val)
        np.fill_diagonal(homo2, empty_val)
        homo1[:, torm[:min(counts.shape)] | torm[
            min(counts.shape):]] = empty_val
        homo2[:, torm[:min(counts.shape)] | torm[
            min(counts.shape):]] = empty_val
        # axis=0 is vertical concat
        counts = np.concatenate([homo1, homo2], axis=0)
        counts[torm, :] = empty_val
        if chrom_subset_index is not None:
            counts = counts[chrom_subset_index[:counts.shape[0]], :][
                :, chrom_subset_index[:counts.shape[1]]]
    else:
        raise ValueError("Input counts matrix is - %d by %d. Counts must be"
                         " n-by-n or n-by-2n or 2n-by-2n." %
                         (counts.shape[0], counts.shape[1]))

    if exclude_zeros:
        counts[np.isnan(counts)] = 0
        counts = sparse.coo_matrix(counts)

    return counts


def check_counts(counts, lengths, ploidy, exclude_zeros=False,
                 chrom_subset_index=None):
    """Check counts dimensions and reformat data.

    Check dimensions of each counts matrix, exclude appropriate values,
    and (if applicable) make sure partially ambiguous diploid counts are
    vertically oriented (one matrix above the other).

    Parameters
    ----------
    counts : list of array or coo_matrix
        Counts data.
    lengths : array_like of int
        Number of beads per homolog of each chromosome.

    Returns
    -------
    counts : list of array or coo_matrix
        Checked and reformatted counts data.
    """

    lengths = np.array(lengths)
    if not isinstance(counts, list):
        counts = [counts]
    return [_check_counts_matrix(
        c, lengths=lengths, ploidy=ploidy, exclude_zeros=exclude_zeros,
        chrom_subset_index=chrom_subset_index) for c in counts]


def preprocess_counts(counts_raw, lengths, ploidy, multiscale_factor, normalize,
                      filter_threshold, beta=None, fullres_torm=None,
                      excluded_counts=None, mixture_coefs=None,
                      exclude_zeros=False, input_weight=None, verbose=True):
    """Check counts, reformat, reduce resolution, filter, and compute bias.

    Preprocessing options include reducing resolution, computing bias (if
    `normalize`), and filering. Counts matrices are also checked and reformatted
    for inference. Final matrices are stored as CountsMatrix subclass instances.

    Parameters
    ----------
    counts_raw : list of array or coo_matrix
        Counts data without normalization or filtering.
    lengths : array_like of int
        Number of beads per homolog of each chromosome.
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    multiscale_factor : int, optional
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 indicates full resolution.
    normalize : bool, optional
        Perform ICE normalization on the counts prior to optimization.
        Normalization is reccomended.
    filter_threshold : float, optional
        Ratio of non-zero beads to be filtered out. Filtering is
        reccomended.
    beta : array_like of float, optional
        Scaling parameter that determines the size of the structure, relative to
        each counts matrix. There should be one beta per counts matrix. If None,
        the optimal beta will be estimated.
    fullres_torm : list of array of bool, optional
        For multiscale optimization, this indicates which beads of the full-
        resolution structure do not correspond to any counts data, and should
        therefore be removed. There should be one array per counts matrix.
    excluded_counts : {"inter", "intra"}, optional
        Whether to exclude inter- or intra-chromosomal counts from optimization.

    Returns
    -------
    counts : list of CountsMatrix subclass instances
        Preprocessed counts data.
    bias : array of float
        Biases computed by ICE normalization.
    torm : array of bool of shape (nbeads,)
        Beads that should be removed (set to NaN) in the structure.
    """

    counts_prepped, bias = _prep_counts(
        counts_raw, lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor, normalize=normalize,
        filter_threshold=filter_threshold, exclude_zeros=exclude_zeros,
        verbose=verbose)

    lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)

    if excluded_counts is not None:
        if excluded_counts.lower() == 'intra':
            counts_prepped = [_inter_counts(
                c, lengths=lengths_lowres, ploidy=ploidy,
                exclude_zeros=exclude_zeros) for c in counts_prepped]
        elif excluded_counts.lower() == 'inter':
            counts_prepped = [_intra_counts(
                c, lengths=lengths_lowres, ploidy=ploidy,
                exclude_zeros=exclude_zeros) for c in counts_prepped]
        else:
            raise ValueError(
                "`excluded_counts` must be 'inter', 'intra' or None.")

    counts = _format_counts(
        counts_prepped, beta=beta, input_weight=input_weight,
        lengths=lengths, ploidy=ploidy, exclude_zeros=exclude_zeros,
        multiscale_factor=multiscale_factor, fullres_torm=fullres_torm)

    torm = find_beads_to_remove(counts_prepped,
                                nbeads=lengths_lowres.sum() * ploidy)
    if mixture_coefs is not None:
        torm = np.tile(torm, len(mixture_coefs))

    if multiscale_factor == 1:
        fullres_torm_for_multiscale = [find_beads_to_remove(
            c, nbeads=lengths.sum() * ploidy) for c in counts if c.sum() > 0]
    else:
        fullres_torm_for_multiscale = None

    return counts, bias, torm, fullres_torm_for_multiscale


def _percent_nan_beads(counts):
    """Return percent of beads that would be NaN for current counts matrix.
    """

    return find_beads_to_remove(counts, max(counts.shape)).sum() / max(counts.shape)


def _prep_counts(counts_list, lengths, ploidy=1, multiscale_factor=1,
                 normalize=True, filter_threshold=0.04, exclude_zeros=True,
                 verbose=True):
    """Copy counts, check matrix, reduce resolution, filter, and compute bias.
    """

    if not isinstance(counts_list, list):
        counts_list = [counts_list]

    # Copy counts
    counts_list = [c.copy() for c in counts_list]

    # Check counts
    counts_list = check_counts(
        counts_list, lengths=lengths, ploidy=ploidy, exclude_zeros=True)

    # Determine ambiguity
    nbeads = lengths.sum() * ploidy
    counts_dict = [('haploid' if ploidy == 1 else {1: 'ambig', 1.5: 'pa', 2: 'ua'}[
                    sum(c.shape) / nbeads], c) for c in counts_list]
    if len(counts_dict) != len(dict(counts_dict)):
        raise ValueError("Can't input multiple counts matrices of the same"
                         " type. Inputs (%d) = %s"
                         % (len(counts_dict),
                            ', '.join([x[0] for x in counts_dict])))
    counts_dict = dict(counts_dict)

    # Reduce resolution
    lengths_lowres = lengths
    for counts_type, counts in counts_dict.items():
        if multiscale_factor != 1:
            lengths_lowres = decrease_lengths_res(
                lengths, multiscale_factor=multiscale_factor)
            counts = decrease_counts_res(
                counts, multiscale_factor=multiscale_factor, lengths=lengths,
                ploidy=ploidy)
            counts_dict[counts_type] = counts

    # Optionally filter counts
    if filter_threshold is None:
        filter_threshold = 0
    if filter_threshold and len(counts_list) > 1:
        # If there are multiple counts matrices, filter them together.
        # Counts will be ambiguated for deciding which beads to remove.
        # For diploid, any beads that are filtered out will be removed from both
        # homologs.
        if verbose:
            print("FILTERING LOW COUNTS: manually filtering all counts together"
                  " by %g" % filter_threshold, flush=True)
        all_counts_ambiguated = ambiguate_counts(
            list(counts_dict.values()), lengths=lengths_lowres, ploidy=ploidy,
            exclude_zeros=True)
        initial_zero_beads = find_beads_to_remove(
            all_counts_ambiguated, lengths_lowres.sum()).sum()
        all_counts_filtered = filter_low_counts(
            sparse.coo_matrix(all_counts_ambiguated), sparsity=False,
            percentage=filter_threshold + _percent_nan_beads(all_counts_ambiguated)).tocoo()
        torm = find_beads_to_remove(all_counts_filtered, lengths_lowres.sum())
        if verbose:
            print('                      removing %d beads' %
                  (torm.sum() - initial_zero_beads), flush=True)
        for counts_type, counts in counts_dict.items():
            if sparse.issparse(counts):
                counts = counts.toarray()
            counts[np.tile(torm, int(counts.shape[0] / torm.shape[0])), :] = 0.
            counts[:, np.tile(torm, int(counts.shape[1] / torm.shape[0]))] = 0.
            counts = sparse.coo_matrix(counts)
            counts_dict[counts_type] = counts
    elif filter_threshold:
        # If there is just one counts matrix, filter the full, non-ambiguated
        # counts matrix.
        # For diploid unambiguous or partially ambigous counts, it is possible
        # that a bead will be filtered out on one homolog but not another.
        individual_counts_torms = np.full((lengths_lowres.sum(),), False)
        for counts_type, counts in counts_dict.items():
            if verbose:
                print('FILTERING LOW COUNTS: manually filtering %s counts by %g'
                      % (counts_type.upper(), filter_threshold), flush=True)
            initial_zero_beads = find_beads_to_remove(
                ambiguate_counts(counts, lengths=lengths_lowres, ploidy=ploidy),
                lengths_lowres.sum()).sum()
            if counts_type == 'pa':
                if sparse.issparse(counts):
                    counts = counts.toarray()
                counts_filtered = np.zeros_like(counts)
                homo1_upper = np.triu(counts[:min(counts.shape), :], 1)
                homo1_lower = np.triu(counts[:min(counts.shape), :].T, 1)
                homo2_upper = np.triu(counts[min(counts.shape):, :], 1)
                homo2_lower = np.triu(counts[min(counts.shape):, :].T, 1)
                counts_filtered[:min(counts.shape), :] += filter_low_counts(
                    sparse.coo_matrix(homo1_upper), sparsity=False,
                    percentage=filter_threshold + _percent_nan_beads(homo1_upper)).toarray()
                counts_filtered[:min(counts.shape), :] += filter_low_counts(
                    sparse.coo_matrix(homo1_lower), sparsity=False,
                    percentage=filter_threshold + _percent_nan_beads(homo1_lower)).toarray().T
                counts_filtered[min(counts.shape):, :] += filter_low_counts(
                    sparse.coo_matrix(homo2_upper), sparsity=False,
                    percentage=filter_threshold + _percent_nan_beads(homo2_upper)).toarray()
                counts_filtered[min(counts.shape):, :] += filter_low_counts(
                    sparse.coo_matrix(homo2_lower), sparsity=False,
                    percentage=filter_threshold + _percent_nan_beads(homo2_lower)).toarray().T
                counts = counts_filtered
            else:
                counts = filter_low_counts(
                    sparse.coo_matrix(counts), sparsity=False,
                    percentage=filter_threshold + _percent_nan_beads(counts)).tocoo()
            torm = find_beads_to_remove(
                ambiguate_counts(counts, lengths=lengths_lowres, ploidy=ploidy),
                lengths_lowres.sum())
            if verbose:
                print('                      removing %d beads' %
                      (torm.sum() - initial_zero_beads), flush=True)
            individual_counts_torms = individual_counts_torms | torm
            counts = sparse.coo_matrix(counts)
            counts_dict[counts_type] = counts

    # Optionally normalize counts
    bias = None
    if normalize:
        if verbose:
            print('COMPUTING BIAS: all counts together', flush=True)
        bias = ICE_normalization(
            ambiguate_counts(
                list(counts_dict.values()), lengths=lengths_lowres,
                ploidy=ploidy, exclude_zeros=True),
            max_iter=300, output_bias=True)[1].flatten()
        # In each counts matrix, zero out counts for which bias is NaN
        for counts_type, counts in counts_dict.items():
            initial_zero_beads = find_beads_to_remove(
                ambiguate_counts(counts, lengths=lengths_lowres, ploidy=ploidy),
                lengths_lowres.sum()).sum()
            if sparse.issparse(counts):
                counts = counts.toarray()
            counts[np.tile(np.isnan(bias), int(counts.shape[0] /
                                               bias.shape[0])), :] = 0.
            counts[:, np.tile(np.isnan(bias), int(counts.shape[1] /
                                                  bias.shape[0]))] = 0.
            counts = sparse.coo_matrix(counts)
            counts_dict[counts_type] = counts
            torm = find_beads_to_remove(
                ambiguate_counts(counts, lengths=lengths_lowres, ploidy=ploidy),
                lengths_lowres.sum())
            if verbose and torm.sum() - initial_zero_beads > 0:
                print('                removing %d additional beads from %s' %
                      (torm.sum() - initial_zero_beads, counts_type),
                      flush=True)

    output_counts = check_counts(
        list(counts_dict.values()), lengths=lengths_lowres, ploidy=ploidy,
        exclude_zeros=exclude_zeros)
    return output_counts, bias


def _format_counts(counts, lengths, ploidy, beta=None, input_weight=None,
                   exclude_zeros=False, multiscale_factor=1, fullres_torm=None):
    """Format each counts matrix as a CountsMatrix subclass instance.
    """

    lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)
    n = lengths_lowres.sum()

    # Check input
    counts = check_counts(
        counts, lengths=lengths_lowres, ploidy=ploidy, exclude_zeros=exclude_zeros)

    if beta is not None:
        if not (isinstance(beta, list) or isinstance(beta, np.ndarray)):
            beta = [beta]
        if len(beta) != len(counts):
            raise ValueError("beta needs to contain as many scaling factors"
                             " as there are datasets (%d). It is of length (%d)"
                             % (len(counts), len(beta)))
    else:
        # To estimate compatible betas for each counts matrix, assume a
        # structure with a mean pairwise distance ** alpha of 1
        beta = []
        for counts_maps in counts:
            if exclude_zeros:
                beta_maps = counts_maps.data.mean()

            else:
                beta_maps = np.nanmean(counts_maps)
            if ploidy == 2:
                if counts_maps.shape == (n, n):
                    beta_maps /= 4
                elif counts_maps.shape[0] != counts_maps.shape[1]:
                    beta_maps /= 2
            beta.append(beta_maps)

    if input_weight is not None:
        if not (isinstance(input_weight, list) or isinstance(input_weight, np.ndarray)):
            input_weight = [input_weight]
        if len(input_weight) != len(counts):
            raise ValueError("input_weights needs to contain as many weighting"
                             " factors as there are datasets (%d). It is of"
                             " length (%d)" % (len(counts), len(input_weight)))
        input_weight = np.array(input_weight)
        if input_weight.sum() not in (0, 1):
            input_weight *= len(input_weight) / input_weight.sum()
    else:
        input_weight = [1.] * len(counts)

    if fullres_torm is not None:
        fullres_torm = (fullres_torm if isinstance(
            fullres_torm, list) else [fullres_torm])
        if len(fullres_torm) != len(counts):
            raise ValueError("fullres_torm needs to contain as many scaling"
                             " factors as there are datasets (%d). It is of"
                             " length (%d)" % (len(counts), len(fullres_torm)))
    else:
        fullres_torm = [None] * len(counts)

    # Reformat counts as SparseCountsMatrix or ZeroCountsMatrix instance
    counts_reformatted = []
    for counts_maps, beta_maps, input_weight_maps, fullres_torm_maps in zip(counts, beta, input_weight, fullres_torm):
        counts_reformatted.append(SparseCountsMatrix(
            counts_maps, lengths=lengths, ploidy=ploidy,
            multiscale_factor=multiscale_factor, beta=beta_maps,
            fullres_torm=fullres_torm_maps, weight=input_weight_maps))
        if not exclude_zeros and (counts_maps == 0).sum() > 0:
            counts_reformatted.append(ZeroCountsMatrix(
                counts_maps, lengths=lengths, ploidy=ploidy,
                multiscale_factor=multiscale_factor, beta=beta_maps,
                fullres_torm=fullres_torm_maps, weight=input_weight_maps))

    return counts_reformatted


def _row_and_col(data):
    """Return row and column indices of non-excluded counts data.
    """

    if isinstance(data, np.ndarray):
        return np.where(~np.isnan(data))
    else:
        return data.row, data.col


def _counts_indices_to_3d_indices(counts, n, ploidy):
    """Return distance matrix indices associated with counts matrix data.
    """

    n = int(n)

    row, col = _row_and_col(counts)

    if counts.shape[0] != n * ploidy or counts.shape[1] != n * ploidy:
        nnz = len(row)

        map_factor_rows = int(n * ploidy / counts.shape[0])
        map_factor_cols = int(n * ploidy / counts.shape[1])
        map_factor = map_factor_rows * map_factor_cols

        x, y = np.indices((map_factor_rows, map_factor_cols))
        x = x.flatten()
        y = y.flatten()

        row = np.repeat(
            x, int(nnz * map_factor / x.shape[0])) * min(
                counts.shape) + np.tile(row, map_factor)
        col = np.repeat(
            y, int(nnz * map_factor / y.shape[0])) * min(
                counts.shape) + np.tile(col, map_factor)

    return row, col


def _update_betas_in_counts_matrices(counts, beta):
    """Updates betas in list of CountsMatrix instances with provided values.
    """

    for counts_maps in counts:
        counts_maps.beta = beta[counts_maps.ambiguity]
    return counts


class CountsMatrix(object):
    """Stores counts data, indices, beta, weight, distance matrix indices, etc.

    Counts data and information associated with this counts matrix.

    Parameters
    ----------
    counts : list of CountsMatrix subclass instances
        Preprocessed counts data.
    lengths : array_like of int
        Number of beads per homolog of each chromosome.
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    multiscale_factor : int, optional
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 indicates full resolution.
    beta : float, optional
        Scaling parameter that determines the size of the structure, relative to
        each counts matrix.
    fullres_torm : array of bool, optional
        For multiscale optimization, this indicates which beads of the full-
        resolution structure do not correspond to any counts data, and should
        therefore be removed.

    Attributes
    ----------
    input_sum : int
        Sum of the nonzero counts in the input.
    ambiguity : {"ambig", "pa", "ua"}
        The ambiguity level of the counts data. "ambig" indicates ambiguous,
        "pa" indicates partially ambiguous, and "ua" indicates unambiguous
        or haploid.
    name : {"ambig", "pa", "ua", "ambig0", "pa0", "ua0"}
        For nonzero counts data, this is the same as `ambiguity`. Otherwise,
        it is `amiguity` + "0".
    null : bool
        Whether the counts data should be excluded from the poisson component
        of the objective function. The indices of the counts are still used to
        compute the constraint components of the objective function.
    highres_per_lowres_bead : None or array of int
        For multiscale optimization, this is the number of full-res beads
        corresponding to each low-res bead.
    row3d : array of int
        Distance matrix rows associated with counts matrix rows.
    col3d : array of int
        Distance matrix columns associated with counts matrix columns.
    """

    def __init__(self):
        if self.__class__.__name__ in ('CountsMatrix', 'AtypicalCountsMatrix'):
            raise ValueError("This class is not intended"
                             " to be instantiated directly.")
        self.input_sum = None
        self.ambiguity = None
        self.name = None
        self.beta = None
        self.weight = None
        self.null = None
        self.highres_per_lowres_bead = None
        self.row3d = None
        self.col3d = None

    @property
    def row(self):
        """Row index array of the matrix (COO format).
        """

        pass

    @property
    def col(self):
        """Column index array of the matrix (COO format).
        """

        pass

    @property
    def nnz(self):
        """Number of stored values, including explicit zeros.
        """

        pass

    @property
    def data(self):
        """Data array of the matrix (COO format).
        """

        pass

    @property
    def shape(self):
        """Shape of the matrix.
        """

        pass

    def toarray(self):
        """Convert counts matrix to numpy array format.
        """

        pass

    def tocoo(self):
        """Convert counts matrix to scipy sparse COO format.
        """

        pass

    def copy(self):
        """Copy counts matrix.
        """

        pass

    def sum(self, axis=None, dtype=None, out=None):
        """Sum of current counts matrix.
        """

        pass

    def bias_per_bin(self, bias, ploidy):
        """Determines bias corresponding to each bin of the matrix.

        Returns bias for each bin of the contact counts matrix by multiplying
        the bias for the bin's row and column.

        Parameters
        ----------
        bias : array of float, optional
            Biases computed by ICE normalization.
        ploidy : {1, 2}
            Ploidy, 1 indicates haploid, 2 indicates diploid.

        Returns
        -------
        array of float
            Bias for each bin of the contact counts matrix.
        """

        pass

    def count_fullres_per_lowres_bins(self, multiscale_factor):
        """
        For multiscale: return number of full-res bins per bin at current res.

        Returns the number of full-resolution counts bins corresponding to each
        low-resolution counts bin for each bin of the low-resolution contact
        counts matrix.

        Parameters
        ----------
        multiscale_factor : int, optional
            Factor by which to reduce the resolution. A value of 2 halves the
            resolution. A value of 1 indicates full resolution.

        Returns
        -------
        array of float
            Number of full-resolution counts bins corresponding to each
            low-resolution counts bin for each bin of the low-resolution contact
            counts matrix.
        """

        pass


class SparseCountsMatrix(CountsMatrix):
    """Stores data for non-zero counts bins.
    """

    def __init__(self, counts, lengths, ploidy, multiscale_factor=1,
                 beta=1., fullres_torm=None, weight=1.):
        lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)
        counts = counts.copy()
        if sparse.issparse(counts):
            counts = counts.toarray()
        self.input_sum = np.nansum(counts)
        counts[np.isnan(counts)] = 0
        if np.array_equal(counts[~np.isnan(counts)],
                          counts[~np.isnan(counts)].round()):
            counts = counts.astype(
                sparse.sputils.get_index_dtype(maxval=counts.max()))
        self._counts = sparse.coo_matrix(counts)
        self.ambiguity = {1: 'ambig', 1.5: 'pa', 2: 'ua'}[
            sum(counts.shape) / (lengths_lowres.sum() * ploidy)]
        self.name = self.ambiguity
        self.beta = beta
        self.weight = (1. if weight is None else weight)
        self.null = False

        if multiscale_factor != 1:
            self.highres_per_lowres_bead = _count_fullres_per_lowres_bead(
                multiscale_factor=multiscale_factor, lengths=lengths,
                ploidy=ploidy, fullres_torm=fullres_torm)
        else:
            self.highres_per_lowres_bead = None

        self.row3d, self.col3d = _counts_indices_to_3d_indices(
            self, n=lengths_lowres.sum(), ploidy=ploidy)

    @property
    def row(self):
        return self._counts.row

    @property
    def col(self):
        return self._counts.col

    @property
    def nnz(self):
        return self._counts.nnz

    @property
    def data(self):
        return self._counts.data

    @property
    def shape(self):
        return self._counts.shape

    def toarray(self):
        return self._counts.toarray()

    def tocoo(self):
        return self._counts

    def copy(self):
        self._counts = self._counts.copy()
        self.row3d = self.row3d.copy()
        self.col3d = self.col3d.copy()
        return self

    def sum(self, axis=None, dtype=None, out=None):
        return self._counts.sum(axis=axis, dtype=dtype, out=out)

    def bias_per_bin(self, bias, ploidy):
        if bias is None:
            return np.ones((self.nnz,))
        else:
            bias = bias.flatten()
            bias = np.tile(bias, int(min(self.shape) * ploidy / len(bias)))
            return bias[self.row] * bias[self.col]

    def count_fullres_per_lowres_bins(self, multiscale_factor):
        if multiscale_factor == 1:
            return 1
        else:
            return self.highres_per_lowres_bead[self.row] * self.highres_per_lowres_bead[self.col]


class AtypicalCountsMatrix(CountsMatrix):
    """Stores null counts data or data for zero counts bins.
    """

    def __init__(self):
        CountsMatrix.__init__(self)
        self._row = None
        self._col = None
        self._shape = None

    @property
    def row(self):
        return self._row

    @property
    def col(self):
        return self._col

    @property
    def nnz(self):
        return len(self.row)

    @property
    def data(self):
        return np.zeros_like(self.row)

    @property
    def shape(self):
        return self._shape

    def toarray(self):
        array = np.full(self.shape, np.nan)
        array[self.row, self.col] = 0
        return array

    def tocoo(self):
        return sparse.coo_matrix(self.toarray())

    def copy(self):
        self.row = self.row.copy()
        self.col = self.col.copy()
        self.row3d = self.row3d.copy()
        self.col3d = self.col3d.copy()
        return self

    def sum(self, axis=None, dtype=None, out=None):
        if axis is None or axis == (0, 1) or axis == (1, 0):
            output = 0
        elif axis in (0, 1, -1):
            output = np.zeros((self.shape[int(not axis)]))
        else:
            raise ValueError("Axis %s not understood" % axis)
        if out is not None:
            output = np.array(output).reshape(out.shape)
        if dtype is not None:
            output = dtype(output)
        return output

    def bias_per_bin(self, bias=None, ploidy=None):
        return np.ones((self.nnz,))

    def count_fullres_per_lowres_bins(self, multiscale_factor):
        if multiscale_factor == 1:
            return 1
        else:
            return self.highres_per_lowres_bead[self.row] * self.highres_per_lowres_bead[self.col]


class ZeroCountsMatrix(AtypicalCountsMatrix):
    """Stores data for zero counts bins.
    """

    def __init__(self, counts, lengths, ploidy, multiscale_factor=1,
                 beta=1., fullres_torm=None, weight=1.):
        lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)
        counts = counts.copy()
        if sparse.issparse(counts):
            counts = counts.toarray()
        self.input_sum = np.nansum(counts)
        counts[counts != 0] = np.nan
        dummy_counts = counts.copy() + 1
        dummy_counts[np.isnan(dummy_counts)] = 0
        dummy_counts = sparse.coo_matrix(dummy_counts)
        self._row = dummy_counts.row
        self._col = dummy_counts.col
        self._shape = dummy_counts.shape
        self.ambiguity = {1: 'ambig', 1.5: 'pa', 2: 'ua'}[
            sum(counts.shape) / (lengths_lowres.sum() * ploidy)]
        self.name = '%s0' % self.ambiguity
        self.beta = beta
        self.weight = (1. if weight is None else weight)
        self.null = False

        if multiscale_factor != 1:
            self.highres_per_lowres_bead = _count_fullres_per_lowres_bead(
                multiscale_factor=multiscale_factor, lengths=lengths,
                ploidy=ploidy, fullres_torm=fullres_torm)
        else:
            self.highres_per_lowres_bead = None

        self.row3d, self.col3d = _counts_indices_to_3d_indices(
            self, n=lengths_lowres.sum(), ploidy=ploidy)


class NullCountsMatrix(AtypicalCountsMatrix):
    """Stores null counts data.
    """

    def __init__(self, counts, lengths, ploidy, multiscale_factor=1,
                 beta=1., fullres_torm=None, weight=1.):
        if not isinstance(counts, list):
            counts = [counts]

        # Dummy counts need to be inputted because if a row/col is all 0 it is
        # excluded from calculations.
        # To create dummy counts, ambiguate counts & sum together, then set all
        # non-0 values to 1
        lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)
        dummy_counts = _create_dummy_counts(
            counts=counts, lengths=lengths_lowres, ploidy=ploidy)

        self._row = dummy_counts.row
        self._col = dummy_counts.col
        self._shape = dummy_counts.shape
        self.ambiguity = {1: 'ambig', 1.5: 'pa', 2: 'ua'}[
            sum(dummy_counts.shape) / (lengths_lowres.sum() * ploidy)]
        self.name = '%s0' % self.ambiguity
        self.beta = 0.
        self.weight = 0.
        self.null = True

        self.input_sum = 0.
        for counts_maps in counts:
            if isinstance(counts_maps, np.ndarray):
                self.input_sum += np.nansum(counts_maps)
            else:
                self.input_sum += counts_maps.sum()

        if multiscale_factor != 1:
            self.highres_per_lowres_bead = _count_fullres_per_lowres_bead(
                multiscale_factor=multiscale_factor, lengths=lengths,
                ploidy=ploidy, fullres_torm=fullres_torm)
        else:
            self.highres_per_lowres_bead = None

        self.row3d, self.col3d = _counts_indices_to_3d_indices(
            self, n=lengths_lowres.sum(), ploidy=ploidy)
