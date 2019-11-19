import numpy as np
from scipy import sparse

from ..externals.iced.filter import filter_low_counts
from ..externals.iced.normalization import ICE_normalization

from .utils import find_beads_to_remove
from .multiscale_optimization import decrease_lengths_res, decrease_counts_res, count_fullres_per_lowres_bead


def ambiguate_counts(counts, n, exclude_zeros=None):
    """Convert diploid counts to ambiguous & aggregate counts across matrices.

    :param counts: counts ndarray or sparse matrix, or list of counts
    :param n: sum of lengths
    :return: ambiguated counts
    """

    from scipy import sparse

    if not isinstance(counts, list):
        counts = [counts]

    if len(counts) == 1 and counts[0].shape == (n, n):
        return counts[0].copy()
    output = np.zeros((n, n))
    for c in counts:
        if not isinstance(c, zero_counts_matrix):
            if not isinstance(c, np.ndarray):
                c = c.toarray()
            if c.shape[0] > c.shape[1]:
                c_ambig = np.nansum(
                    [c[:n, :], c[n:, :], c[:n, :].T, c[n:, :].T])
            elif c.shape[0] < c.shape[1]:
                c_ambig = np.nansum(
                    [c[:, :n].T, c[:, n:].T, c[:, :n], c[:, n:]])
            elif c.shape[0] == n:
                c_ambig = c
            else:
                c_ambig = np.nansum(
                    [c[:n, :n], c[n:, :n], c[:n, n:], c[n:, n:]])
            output[~np.isnan(c_ambig)] += c_ambig[~np.isnan(c_ambig)]

    output = np.triu(output, 1)
    if exclude_zeros is None:
        exclude_zeros = all([isinstance(c, sparse_counts_matrix)
                            or sparse.issparse(c) for c in counts])
    return check_counts_matrix(output, exclude_zeros=exclude_zeros)


def create_dummy_counts(counts, lengths, ploidy, multiscale_factor=1):
    """Create sparse matrix of 1's with same row and col as input counts.
    """

    from .utils import constraint_dis_indices
    lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)

    rows, cols = constraint_dis_indices(
        counts, n=lengths_lowres.sum(), lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor, nbeads=lengths.sum() * ploidy)
    dummy_counts = sparse.coo_matrix((np.ones_like(rows), (rows, cols)), shape=(
        lengths.sum() * ploidy, lengths.sum() * ploidy)).toarray()
    dummy_counts = sparse.coo_matrix(np.triu(dummy_counts + dummy_counts.T, 1))

    return dummy_counts


def get_chrom_subset_index(ploidy, lengths_full, chrom_full, chrom_subset):
    """Return indices for selected chromosomes only.
    """

    if not isinstance(chrom_subset, list):
        chrom_subset = [chrom_subset]
    if not all([chrom in chrom_full for chrom in chrom_subset]):
        raise ValueError("Chromosomes to be inferred (%s) are not in genome (%s"
                         ")" % (','.join(chrom_subset), ','.join(chrom_full)))

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
                 counts=None, exclude_zeros=True, struct_true=None):
    """Return data for selected chromosomes only.
    """

    if chrom_subset is None or chrom_subset == chrom_full:
        chrom_subset = chrom_full.copy()
        lengths_subset = lengths_full.copy()
        if counts is not None:
            counts = check_counts(counts, exclude_zeros=exclude_zeros)
        return counts, struct_true, lengths_subset, chrom_subset
    else:
        if isinstance(chrom_subset, str):
            chrom_subset = [chrom_subset]
        if not all([chrom in chrom_full for chrom in chrom_subset]):
            raise ValueError("Chromosomes to be inferred (%s) are not in genome"
                             " (%s)" %
                             (','.join(chrom_subset), ','.join(chrom_full)))
        # Make sure chrom_subset is sorted properly
        chrom_subset = [chrom for chrom in chrom_full if chrom in chrom_subset]

        index, lengths_subset = get_chrom_subset_index(
            ploidy, lengths_full, chrom_full, chrom_subset)

        if struct_true is not None and index is not None:
            struct_true = struct_true[index]

        if counts is not None:
            counts = [check_counts_matrix(
                c, exclude_zeros=exclude_zeros,
                chrom_subset_index=index) for c in counts]

        return counts, struct_true, lengths_subset, chrom_subset


def check_counts_matrix(counts, exclude_zeros=True, chrom_subset_index=None):
    """Check counts dimensions, reformat, & excise selected chromosomes.
    """

    from .utils import find_beads_to_remove

    if chrom_subset_index is not None and len(chrom_subset_index) / max(counts.shape) not in (1, 2):
        raise ValueError("chrom_subset_index size (%d) does not fit counts"
                         " shape (%d, %d)" %
                         (len(chrom_subset_index), counts.shape[0],
                             counts.shape[1]))

    empty_val = 0
    torm = np.full((max(counts.shape)), False)
    if not exclude_zeros:
        empty_val = np.nan
        torm = find_beads_to_remove(counts, max(counts.shape))

    if sparse.isspmatrix_coo(counts) or sparse.issparse(counts):
        counts = counts.toarray()
    if not isinstance(counts, np.ndarray):
        counts = np.array(counts)

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
                         " n-by-n or n-by-2n." %
                         (counts.shape[0], counts.shape[1]))

    if exclude_zeros:
        counts[np.isnan(counts)] = 0
        counts = sparse.coo_matrix(counts)

    return counts


def check_counts(counts, exclude_zeros=True):
    """Check counts dimensions, reformat, & excise selected chromosomes.
    """

    if not isinstance(counts, list):
        counts = [counts]
    return [check_counts_matrix(c, exclude_zeros) for c in counts]


def preprocess_counts(counts, lengths, ploidy, multiscale_factor, normalize,
                      filter_threshold, exclude_zeros, beta, input_weight,
                      verbose=True, fullres_torm=None, output_directory=None):
    """Reduce resolution, filter, reformat, and compute bias on counts.
    """

    from ..externals.iced.io import write_counts
    import os

    counts_prepped, bias = prep_counts(counts, lengths=lengths, ploidy=ploidy,
                                       multiscale_factor=multiscale_factor,
                                       normalize=normalize,
                                       filter_threshold=filter_threshold,
                                       exclude_zeros=exclude_zeros,
                                       verbose=verbose)
    lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)
    counts_formatted = format_counts(counts_prepped, beta=beta,
                                     input_weight=input_weight,
                                     lengths=lengths_lowres, ploidy=ploidy,
                                     exclude_zeros=exclude_zeros,
                                     multiscale_factor=multiscale_factor,
                                     fullres_torm=fullres_torm)

    torm = find_beads_to_remove(counts_prepped,
                                nbeads=lengths_lowres.sum() * ploidy)

    if output_directory is not None:
        try:
            os.makedirs(output_directory)
        except OSError:
            pass

        for c in counts_formatted:
            if not c.null and c.sum() > 0:
                write_counts(os.path.join(output_directory,
                             '%s_filtered.matrix' % c.name), c.tocoo())

    return counts_formatted, bias, torm


def percent_nan_beads(counts):
    """Return percent of beads that would be NaN for current counts matrix.
    """

    return find_beads_to_remove(counts, max(counts.shape)).sum() / max(counts.shape)


def prep_counts(counts_list, lengths, ploidy=1, multiscale_factor=1,
                normalize=True, filter_threshold=0.04, exclude_zeros=True,
                verbose=True):
    """Copy counts, check matrix, reduce resolution, filter, and compute bias.
    """

    nbeads = lengths.sum() * ploidy
    counts_dict = [('haploid' if ploidy == 1 else {1: 'ambig', 1.5: 'pa', 2: 'ua'}[
                    sum(c.shape) / nbeads], c) for c in counts_list]
    if len(counts_dict) != len(dict(counts_dict)):
        raise ValueError("Can't input multiple counts matrices of the same"
                         " type. Inputs = %s"
                         % ', '.join([x[0] for x in counts_dict]))
    counts_dict = dict(counts_dict)

    # Copy counts
    counts_dict = {counts_type: counts.copy()
                   for counts_type, counts in counts_dict.items()}

    # Check counts
    counts_dict = {counts_type: check_counts_matrix(
        counts, exclude_zeros=True) for counts_type, counts in counts_dict.items()}

    # Reduce resolution
    lengths_lowres = lengths
    for counts_type, counts in counts_dict.items():
        if multiscale_factor != 1:
            counts, lengths_lowres = decrease_counts_res(
                counts, multiscale_factor, lengths, ploidy)
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
            list(counts_dict.values()), lengths_lowres.sum())
        initial_zero_beads = find_beads_to_remove(
            all_counts_ambiguated, lengths_lowres.sum()).sum()
        all_counts_filtered = filter_low_counts(
            sparse.coo_matrix(all_counts_ambiguated), sparsity=False,
            percentage=filter_threshold + percent_nan_beads(all_counts_ambiguated)).tocoo()
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
            initial_zero_beads = find_beads_to_remove(ambiguate_counts(
                counts, lengths_lowres.sum()), lengths_lowres.sum()).sum()
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
                    percentage=filter_threshold + percent_nan_beads(homo1_upper)).toarray()
                counts_filtered[:min(counts.shape), :] += filter_low_counts(
                    sparse.coo_matrix(homo1_lower), sparsity=False,
                    percentage=filter_threshold + percent_nan_beads(homo1_lower)).toarray().T
                counts_filtered[min(counts.shape):, :] += filter_low_counts(
                    sparse.coo_matrix(homo2_upper), sparsity=False,
                    percentage=filter_threshold + percent_nan_beads(homo2_upper)).toarray()
                counts_filtered[min(counts.shape):, :] += filter_low_counts(
                    sparse.coo_matrix(homo2_lower), sparsity=False,
                    percentage=filter_threshold + percent_nan_beads(homo2_lower)).toarray().T
                counts = counts_filtered
            else:
                counts = filter_low_counts(
                    sparse.coo_matrix(counts), sparsity=False,
                    percentage=filter_threshold + percent_nan_beads(counts)).tocoo()
            torm = find_beads_to_remove(ambiguate_counts(
                counts, lengths_lowres.sum()), lengths_lowres.sum())
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
        bias = ICE_normalization(ambiguate_counts(list(counts_dict.values(
        )), lengths_lowres.sum()), max_iter=300, output_bias=True)[1].flatten()
        # In each counts matrix, zero out counts for which bias is NaN
        for counts_type, counts in counts_dict.items():
            initial_zero_beads = find_beads_to_remove(ambiguate_counts(
                counts, lengths_lowres.sum()), lengths_lowres.sum()).sum()
            if sparse.issparse(counts):
                counts = counts.toarray()
            counts[np.tile(np.isnan(bias), int(counts.shape[0] /
                                               bias.shape[0])), :] = 0.
            counts[:, np.tile(np.isnan(bias), int(counts.shape[1] /
                                                  bias.shape[0]))] = 0.
            counts = sparse.coo_matrix(counts)
            counts_dict[counts_type] = counts
            torm = find_beads_to_remove(ambiguate_counts(
                counts, lengths_lowres.sum()), lengths_lowres.sum())
            if verbose:
                print('                removing %d beads from %s' %
                      (torm.sum() - initial_zero_beads, counts_type),
                      flush=True)

    return check_counts(list(counts_dict.values()), exclude_zeros), bias


def format_counts(counts, beta, input_weight, lengths, ploidy, exclude_zeros,
                  multiscale_factor, fullres_torm=None):
    """Format each counts matrix as a counts_matrix object.
    """

    # Check input
    counts = check_counts(counts, exclude_zeros)

    if beta is not None:
        beta = (beta if isinstance(beta, list) else [beta])
        if len(beta) != len(counts):
            raise ValueError("beta needs to contain as many scaling factors"
                             " as there are datasets (%d). It is of length (%d)"
                             % (len(counts), len(beta)))
    else:
        beta = [1.] * len(counts)
    if input_weight is not None:
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

    # Reformat counts as sparse_counts_matrix or zero_counts_matrix objects
    counts_reformatted = []
    for counts_maps, beta_maps, input_weight_maps, fullres_torm_maps in zip(counts, beta, input_weight, fullres_torm):
        counts_reformatted.append(sparse_counts_matrix(
            counts_maps, lengths=lengths, ploidy=ploidy, beta=beta_maps,
            weight=input_weight_maps, multiscale_factor=multiscale_factor,
            fullres_torm=fullres_torm_maps))
        if not exclude_zeros and (counts_maps == 0).sum() > 0:
            counts_reformatted.append(zero_counts_matrix(
                counts_maps, lengths=lengths, ploidy=ploidy, beta=beta_maps,
                weight=input_weight_maps, multiscale_factor=multiscale_factor,
                fullres_torm=fullres_torm_maps))

    return counts_reformatted


def row_and_col(data):
    """Return row and column indices of non-excluded counts data.
    """

    if isinstance(data, np.ndarray):
        return np.where(~np.isnan(data))
    else:
        return data.row, data.col


def counts_indices_to_3d_indices(counts, n, ploidy):
    """Return distance matrix indices associated with counts matrix data.

    :param counts: counts ndarray or sparse matrix, or sparse_counts_matrix, zero_counts_matrix, null_counts_matrix object
    :param n: sum of lengths
    :param ploidy: ploidy
    :return:
    """

    n = int(n)

    row, col = row_and_col(counts)

    if counts.shape[0] != n * ploidy or counts.shape[1] != n * ploidy:
        nnz = len(row)

        map_factor = int(n * ploidy / min(counts.shape))
        map_factor_rows = int(n * ploidy / counts.shape[0])
        map_factor_cols = int(n * ploidy / counts.shape[1])

        x, _ = np.indices((map_factor_rows, map_factor_rows))
        _, y = np.indices((map_factor_cols, map_factor_cols))
        x = x.flatten()
        y = y.flatten()
        tile_factor = max(x.shape[0], y.shape[0])

        row = np.repeat(
            x, nnz * int(tile_factor / x.shape[0])) * min(
                counts.shape) + np.tile(row, map_factor ** 2)
        col = np.repeat(
            y, nnz * int(tile_factor / y.shape[0])) * min(
                counts.shape) + np.tile(col, map_factor ** 2)

    return row, col


def update_betas_in_counts_matrices(counts, beta):
    """Updates betas in list of counts_matrix objects with provided values.
    """

    for counts_maps in counts:
        counts_maps.beta = beta[counts_maps.ambiguity]
    return counts


class counts_matrix(object):
    """Stores counts data, indices, beta, weight, distance matrix indices, etc.
    """

    def __init__(self):
        if self.__class__.__name__ in ('counts_matrix', 'atypical_counts_matrix'):
            raise ValueError("This class is not intended"
                             " to be instantiated directly.")
        self.counts = None
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
        """

        pass

    def count_fullres_per_lowres_bins(self, multiscale_factor):
        """
        For multiscale: return number of full-res bins per bin at current res.
        """

        pass


class sparse_counts_matrix(counts_matrix):
    """Stores data for non-zero counts bins.
    """

    def __init__(self, counts, lengths, ploidy, beta=1., weight=1.,
                 multiscale_factor=1, fullres_torm=None):
        counts = counts.copy()
        if sparse.issparse(counts):
            counts = counts.toarray()
        self.input_sum = np.nansum(counts)
        counts[np.isnan(counts)] = 0
        counts = counts.astype(
            sparse.sputils.get_index_dtype(maxval=counts.max()))
        self.counts = sparse.coo_matrix(counts)
        self.ambiguity = {1: 'ambig', 1.5: 'pa', 2: 'ua'}[
            sum(counts.shape) / (lengths.sum() * ploidy)]
        self.name = self.ambiguity
        self.beta = beta
        self.weight = (1. if weight is None else weight)
        self.null = False

        if multiscale_factor != 1:
            self.highres_per_lowres_bead = count_fullres_per_lowres_bead(
                multiscale_factor, lengths, ploidy, fullres_torm)
        else:
            self.highres_per_lowres_bead = None

        self.row3d, self.col3d = counts_indices_to_3d_indices(
            self, n=lengths.sum(), ploidy=ploidy)

    @property
    def row(self):
        return self.counts.row

    @property
    def col(self):
        return self.counts.col

    @property
    def nnz(self):
        return self.counts.nnz

    @property
    def data(self):
        return self.counts.data

    @property
    def shape(self):
        return self.counts.shape

    def toarray(self):
        return self.counts.toarray()

    def tocoo(self):
        return self.counts

    def copy(self):
        self.counts = self.counts.copy()
        self.row3d = self.row3d.copy()
        self.col3d = self.col3d.copy()
        return self

    def sum(self, axis=None, dtype=None, out=None):
        return self.counts.sum(axis=axis, dtype=dtype, out=out)

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


class atypical_counts_matrix(counts_matrix):
    """Stores null counts data or data for zero counts bins.
    """

    def __init__(self):
        counts_matrix.__init__(self)
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


class zero_counts_matrix(atypical_counts_matrix):
    """Stores data for zero counts bins.
    """

    def __init__(self, counts, lengths, ploidy, beta=1., weight=1.,
                 multiscale_factor=1, fullres_torm=None):
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
            sum(counts.shape) / (lengths.sum() * ploidy)]
        self.name = '%s0' % self.ambiguity
        self.beta = beta
        self.weight = (1. if weight is None else weight)
        self.null = False

        if multiscale_factor != 1:
            self.highres_per_lowres_bead = count_fullres_per_lowres_bead(
                multiscale_factor, lengths, ploidy, fullres_torm)
        else:
            self.highres_per_lowres_bead = None

        self.row3d, self.col3d = counts_indices_to_3d_indices(
            self, n=lengths.sum(), ploidy=ploidy)


class null_counts_matrix(atypical_counts_matrix):
    """Stores null counts data.
    """

    def __init__(self, counts, lengths, ploidy, multiscale_factor=1,
                 fullres_torm=None):
        # Dummy counts need to be inputted because if a row/col is all 0 it is
        # excluded from calculations.
        # To create dummy counts, ambiguate counts & sum together, then set all
        # non-0 values to 1
        lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)
        dummy_counts = create_dummy_counts(
            counts=counts, lengths=lengths_lowres, ploidy=ploidy)

        self._row = dummy_counts.row
        self._col = dummy_counts.col
        self._shape = dummy_counts.shape
        self.ambiguity = {1: 'ambig', 1.5: 'pa', 2: 'ua'}[
            sum(counts.shape) / (lengths_lowres.sum() * ploidy)]
        self.name = '%s0' % self.ambiguity
        self.beta = 0.
        self.weight = 0.
        self.input_sum = 0
        self.null = True

        if multiscale_factor != 1:
            self.highres_per_lowres_bead = count_fullres_per_lowres_bead(
                multiscale_factor, lengths, ploidy, fullres_torm)
        else:
            self.highres_per_lowres_bead = None

        self.row3d, self.col3d = counts_indices_to_3d_indices(
            self, n=lengths.sum(), ploidy=ploidy)
