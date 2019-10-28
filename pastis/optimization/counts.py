import numpy as np
from scipy import sparse

from ..externals.iced.filter import filter_low_counts
from ..externals.iced.normalization import ICE_normalization

from .utils import find_beads_to_remove
from .multiscale_optimization import decrease_lengths_res, decrease_counts_res, count_fullres_per_lowres_bead


def ambiguate_counts(counts, n, as_sparse=None):
    """Convert diploid counts to ambiguous & aggregate counts across matrices.

    :param counts: counts ndarray or sparse matrix, or list of counts
    :param n: sum of lengths
    :return: ambiguated counts
    """

    from scipy import sparse
    from .counts import check_counts_matrix, zero_counts_matrix, sparse_counts_matrix

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
    if as_sparse is None:
        as_sparse = all([isinstance(c, sparse_counts_matrix)
                         or sparse.issparse(c) for c in counts])
    return check_counts_matrix(output, as_sparse=as_sparse)


def create_dummy_counts(counts, lengths, ploidy, multiscale_factor=1):
    """Create sparse matrix of 1's with same row and col as input counts.
    """

    from .utils import constraint_dis_indices
    lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)

    rows, cols = constraint_dis_indices(counts, n=lengths_lowres.sum(
    ), lengths=lengths, ploidy=ploidy, multiscale_factor=multiscale_factor, nbeads=lengths.sum() * ploidy)
    dummy_counts = sparse.coo_matrix((np.ones_like(rows), (rows, cols)), shape=(
        lengths.sum() * ploidy, lengths.sum() * ploidy)).toarray()
    dummy_counts = sparse.coo_matrix(np.triu(dummy_counts + dummy_counts.T, 1))

    return dummy_counts


def get_chrom_subset_index(ploidy, genome_lengths, genome_chrom, infer_chrom):
    """Return indices for selected chromosomes only.
    """

    if not isinstance(infer_chrom, list):
        infer_chrom = [infer_chrom]
    if not all([chrom in genome_chrom for chrom in infer_chrom]):
        raise ValueError('Chromosomes to be inferred (%s) are not in genome (%s)' % (
            ','.join(infer_chrom), ','.join(genome_chrom)))

    infer_lengths = genome_lengths.copy()
    index = None
    if not np.array_equal(infer_chrom, genome_chrom):
        infer_lengths = np.array([genome_lengths[i] for i in range(
            len(genome_chrom)) if genome_chrom[i] in infer_chrom])
        index = []
        for i in range(len(genome_lengths)):
            index.append(
                np.full((genome_lengths[i],), genome_chrom[i] in infer_chrom))
        index = np.concatenate(index)
        if ploidy == 2:
            index = np.tile(index, 2)
    return index, infer_lengths


def subset_chrom(ploidy, genome_lengths, genome_chrom, infer_chrom=None, counts=None, as_sparse=True, X_true=None):
    """Return data for selected chromosomes only.
    """

    if infer_chrom is None or infer_chrom == genome_chrom:
        infer_chrom = genome_chrom.copy()
        infer_lengths = genome_lengths.copy()
        if counts is not None:
            counts = check_counts(counts, as_sparse=as_sparse)
        return counts, X_true, infer_lengths, infer_chrom
    else:
        if isinstance(infer_chrom, str):
            infer_chrom = [infer_chrom]
        if not all([chrom in genome_chrom for chrom in infer_chrom]):
            raise ValueError('Chromosomes to be inferred (%s) are not in genome (%s)' % (
                ','.join(infer_chrom), ','.join(genome_chrom)))
        # Make sure infer_chrom is sorted properly
        infer_chrom = [chrom for chrom in genome_chrom if chrom in infer_chrom]

        index, infer_lengths = get_chrom_subset_index(
            ploidy, genome_lengths, genome_chrom, infer_chrom)

        if X_true is not None and index is not None:
            X_true = X_true[index]

        if counts is not None:
            counts = [check_counts_matrix(
                c, as_sparse=as_sparse, chrom_subset_index=index) for c in counts]

        return counts, X_true, infer_lengths, infer_chrom


def check_counts_matrix(counts, as_sparse=True, chrom_subset_index=None):
    """Check counts dimensions, reformat, & excise selected chromosomes.
    """

    from .utils import find_beads_to_remove

    if chrom_subset_index is not None and len(chrom_subset_index) / max(counts.shape) not in (1, 2):
        raise ValueError('chrom_subset_index size (%d) does not fit counts shape (%d, %d)' % (
            len(chrom_subset_index), counts.shape[0], counts.shape[1]))

    empty_val = 0
    torm = np.full((max(counts.shape)), False)
    if not as_sparse:
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
        raise ValueError('Input counts matrix is - %d by %d. Counts must be n-by-n or n-by-2n.' %
                         (counts.shape[0], counts.shape[1]))

    if as_sparse:
        counts[np.isnan(counts)] = 0
        counts = sparse.coo_matrix(counts)

    return counts


def check_counts(counts, as_sparse=True):
    """Check counts dimensions, reformat, & excise selected chromosomes.
    """

    if not isinstance(counts, list):
        counts = [counts]
    return [check_counts_matrix(c, as_sparse) for c in counts]


def percent_nan_beads(counts):
    """Return percent of beads that would be NaN for current counts matrix.
    """

    return find_beads_to_remove(counts, max(counts.shape)).sum() / max(counts.shape)


def prep_counts(counts_list, lengths, ploidy=1, multiscale_factor=1, normalize=False, filter_counts=False, filter_percentage=0.04, as_sparse=True, verbose=True):
    """Copy counts, check matrix, reduce resolution, filter, and compute bias.
    """

    nbeads = lengths.sum() * ploidy
    counts_dict = [('haploid' if ploidy == 1 else {1: 'ambig', 1.5: 'pa', 2: 'ua'}[
                    sum(c.shape) / nbeads], c) for c in counts_list]
    if len(counts_dict) != len(dict(counts_dict)):
        raise ValueError("Can't input multiple counts matrices of the same type. Inputs = %s" % ', '.join(
            [x[0] for x in counts_dict]))
    counts_dict = dict(counts_dict)

    # Copy counts
    counts_dict = {counts_type: counts.copy()
                   for counts_type, counts in counts_dict.items()}

    # Check counts
    counts_dict = {counts_type: check_counts_matrix(
        counts, as_sparse=True) for counts_type, counts in counts_dict.items()}

    # Reduce resolution
    lengths_lowres = lengths
    for counts_type, counts in counts_dict.items():
        if multiscale_factor != 1:
            counts, lengths_lowres = decrease_counts_res(
                counts, multiscale_factor, lengths, ploidy)
            counts_dict[counts_type] = counts

    # Filter all counts together... and if an entire bead has 0 counts from
    # this filtering, set that bead to 0 in all counts matrices
    if filter_counts and len(counts_list) > 1:
        if verbose:
            print('FILTERING LOW COUNTS: manually filtering all counts together by %g' %
                  filter_percentage, flush=True)
        all_counts_ambiguated = ambiguate_counts(
            list(counts_dict.values()), lengths_lowres.sum())
        initial_zero_beads = find_beads_to_remove(
            all_counts_ambiguated, lengths_lowres.sum()).sum()
        all_counts_filtered = filter_low_counts(sparse.coo_matrix(
            all_counts_ambiguated), sparsity=False, percentage=filter_percentage + percent_nan_beads(all_counts_ambiguated)).tocoo()
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

    # Filter counts
    if filter_counts:
        individual_counts_torms = np.full((lengths_lowres.sum(),), False)
        for counts_type, counts in counts_dict.items():
            if verbose:
                print('FILTERING LOW COUNTS: manually filtering %s counts by %g' % (
                    counts_type.upper(), filter_percentage), flush=True)
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
                counts_filtered[:min(counts.shape), :] += filter_low_counts(sparse.coo_matrix(homo1_upper), sparsity=False,
                                                                            percentage=filter_percentage + percent_nan_beads(homo1_upper)).toarray()
                counts_filtered[:min(counts.shape), :] += filter_low_counts(sparse.coo_matrix(homo1_lower), sparsity=False,
                                                                            percentage=filter_percentage + percent_nan_beads(homo1_lower)).toarray().T
                counts_filtered[min(counts.shape):, :] += filter_low_counts(sparse.coo_matrix(homo2_upper), sparsity=False,
                                                                            percentage=filter_percentage + percent_nan_beads(homo2_upper)).toarray()
                counts_filtered[min(counts.shape):, :] += filter_low_counts(sparse.coo_matrix(homo2_lower), sparsity=False,
                                                                            percentage=filter_percentage + percent_nan_beads(homo2_lower)).toarray().T
                counts = counts_filtered
            else:
                counts = filter_low_counts(sparse.coo_matrix(
                    counts), sparsity=False, percentage=filter_percentage + percent_nan_beads(counts)).tocoo()
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
                      (torm.sum() - initial_zero_beads, counts_type), flush=True)

    return check_counts(list(counts_dict.values()), as_sparse), bias


def format_counts(counts, beta, input_weight, lengths, ploidy, as_sparse, multiscale_factor, fullres_torm=None):
    """Format each counts matrix as a counts_matrix object.
    """

    # Check input
    counts = check_counts(counts, as_sparse)

    if beta is not None:
        beta = (beta if isinstance(beta, list) else [beta])
        if len(beta) != len(counts):
            raise ValueError("beta needs to contain as many scaling factors as there "
                             "are datasets (%d). It is of length (%d)" % (len(counts), len(beta)))
    else:
        beta = [None] * len(counts)
    if input_weight is not None:
        if len(input_weight) != len(counts):
            raise ValueError("input_weights needs to contain as many weighting factors as there "
                             "are datasets (%d). It is of length (%d)" % (len(counts), len(input_weight)))
        input_weight = np.array(input_weight)
        if input_weight.sum() not in (0, 1):
            input_weight *= len(input_weight) / input_weight.sum()
    else:
        input_weight = [1.] * len(counts)
    if fullres_torm is not None:
        fullres_torm = (fullres_torm if isinstance(
            fullres_torm, list) else [fullres_torm])
        if len(fullres_torm) != len(counts):
            raise ValueError("fullres_torm needs to contain as many scaling factors as there "
                             "are datasets (%d). It is of length (%d)" % (len(counts), len(fullres_torm)))
    else:
        fullres_torm = [None] * len(counts)

    # Reformat counts as sparse_counts_matrix or zero_counts_matrix objects
    counts_reformatted = []
    for counts_maps, beta_maps, input_weight_maps, fullres_torm_maps in zip(counts, beta, input_weight, fullres_torm):
        counts_reformatted.append(sparse_counts_matrix(
            counts_maps, lengths, ploidy, beta_maps, input_weight_maps, multiscale_factor, fullres_torm_maps))
        if not as_sparse and (counts_maps == 0).sum() > 0:
            counts_reformatted.append(zero_counts_matrix(
                counts_maps, lengths, ploidy, beta_maps, input_weight_maps, multiscale_factor, fullres_torm_maps))

    return counts_reformatted


def row_and_col(data):
    """Return row and column indices of non-excluded counts data.
    """

    if isinstance(data, np.ndarray):
        return np.where(~np.isnan(data))
    else:
        return data.row, data.col


def counts_indices_to_3d_indices(counts, n, ploidy):
    """Return indices distance matrix bins associated with counts matrix bins.

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

        row = np.repeat(x, nnz * int(tile_factor /
                                     x.shape[0])) * min(counts.shape) + np.tile(row, map_factor ** 2)
        col = np.repeat(y, nnz * int(tile_factor /
                                     y.shape[0])) * min(counts.shape) + np.tile(col, map_factor ** 2)

    return row, col


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

    def __init__(self, counts, lengths, ploidy, beta=None, weight=1., multiscale_factor=1, fullres_torm=None):
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

    @property
    def data(self):
        return np.zeros_like(self.row)

    @property
    def nnz(self):
        return len(self.row)

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

    def __init__(self, counts, lengths, ploidy, beta=None, weight=1., multiscale_factor=1, fullres_torm=None):
        counts = counts.copy()
        if sparse.issparse(counts):
            counts = counts.toarray()
        self.input_sum = np.nansum(counts)
        counts[counts != 0] = np.nan
        dummy_counts = counts.copy() + 1
        dummy_counts[np.isnan(dummy_counts)] = 0
        dummy_counts = sparse.coo_matrix(dummy_counts)
        self.row = dummy_counts.row
        self.col = dummy_counts.col
        self.shape = dummy_counts.shape
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

    def __init__(self, counts, lengths, ploidy, multiscale_factor=1, fullres_torm=None):
        # Dummy counts need to be inputted because if a row/col is all 0 it is excluded from calculations
        # To create dummy counts, ambiguate counts & sum together, then set all
        # non-0 values to 1
        lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)
        dummy_counts = create_dummy_counts(
            counts=counts, lengths=lengths_lowres, ploidy=ploidy)

        self.row = dummy_counts.row
        self.col = dummy_counts.col
        self.shape = dummy_counts.shape
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
