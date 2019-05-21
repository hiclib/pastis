import numpy as np
from scipy import sparse

from iced.filter import filter_low_counts
from iced.normalization import ICE_normalization

from topsy.inference.utils import ambiguate_counts
from topsy.inference import multiscale_optimization
from topsy.inference.utils import find_beads_to_remove


def get_chrom_subset_index(ploidy, genome_lengths, genome_chrom, infer_chrom):
    if not isinstance(infer_chrom, list):
        infer_chrom = [infer_chrom]
    if not all([chrom in genome_chrom for chrom in infer_chrom]):
        raise ValueError('Chromosomes to be inferred (%s) are not in genome (%s)' % (','.join(infer_chrom), ','.join(genome_chrom)))

    infer_lengths = genome_lengths.copy()
    index = None  # np.arange(int(genome_lengths.sum() * ploidy))
    if not np.array_equal(infer_chrom, genome_chrom):
        infer_lengths = np.array([genome_lengths[i] for i in range(len(genome_chrom)) if genome_chrom[i] in infer_chrom])
        index = []
        for i in range(len(genome_lengths)):
            #begin = sum(genome_lengths[:i])
            #end = begin + genome_lengths[i]
            index.append(np.full((genome_lengths[i],), genome_chrom[i] in infer_chrom))
            #if genome_chrom[i] in infer_chrom:
            #    index.append(np.arange(begin, end, dtype=int))
        index = np.concatenate(index)
        if ploidy == 2:
            #index = np.concatenate([index, index + genome_lengths.sum()])
            index = np.tile(index, 2)
    return index, infer_lengths


def subset_chrom(ploidy, genome_lengths, genome_chrom, infer_chrom=None, counts=None, as_sparse=True, X_true=None):
    if infer_chrom is None:
        infer_chrom = genome_chrom.copy()
        infer_lengths = genome_lengths.copy()
        if counts is not None:
            counts = check_counts(counts, as_sparse=as_sparse)
        return counts, X_true, infer_lengths, infer_chrom
    else:
        if isinstance(infer_chrom, str):
            infer_chrom = [infer_chrom]
        if not all([chrom in genome_chrom for chrom in infer_chrom]):
            raise ValueError('Chromosomes to be inferred (%s) are not in genome (%s)' % (','.join(infer_chrom), ','.join(genome_chrom)))
        # Make sure infer_chrom is sorted properly
        infer_chrom = [chrom for chrom in genome_chrom if chrom in infer_chrom]

        index, infer_lengths = get_chrom_subset_index(ploidy, genome_lengths, genome_chrom, infer_chrom)

        if X_true is not None and index is not None:
            X_true = X_true[index]

        if counts is not None:
            counts = [check_counts_matrix(c, as_sparse=as_sparse, chrom_subset_index=index) for c in counts]

        return counts, X_true, infer_lengths, infer_chrom


def check_counts_matrix(counts, as_sparse=True, chrom_subset_index=None):
    from .utils import find_beads_to_remove

    if chrom_subset_index is not None and len(chrom_subset_index) / max(counts.shape) not in (1, 2):
        raise ValueError('chrom_subset_index size (%d) does not fit counts shape (%d, %d)' % (len(chrom_subset_index), counts.shape[0], counts.shape[1]))

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
        #counts = np.triu(counts, 1)
        counts[np.tril_indices(counts.shape[0])] = empty_val
        counts[torm, :] = empty_val
        counts[:, torm] = empty_val
        #print('check_counts_matrix %d' % np.isnan(np.diag(counts)).sum())
        #print(np.diag(counts))
        if chrom_subset_index is not None:
            counts = counts[chrom_subset_index[:counts.shape[0]], :][:, chrom_subset_index[:counts.shape[1]]]
    elif min(counts.shape) * 2 == max(counts.shape):
        homo1 = counts[:min(counts.shape), :min(counts.shape)]
        homo2 = counts[counts.shape[0] - min(counts.shape):, counts.shape[1] - min(counts.shape):]
        if counts.shape[0] == min(counts.shape):
            homo1 = homo1.T
            homo2 = homo2.T
        np.fill_diagonal(homo1, empty_val)
        np.fill_diagonal(homo2, empty_val)
        homo1[:, torm[:min(counts.shape)] | torm[min(counts.shape):]] = empty_val
        homo2[:, torm[:min(counts.shape)] | torm[min(counts.shape):]] = empty_val
        counts = np.concatenate([homo1, homo2], axis=0) # axis=0 is vertical concat
        counts[torm, :] = empty_val
        if chrom_subset_index is not None:
            counts = counts[chrom_subset_index[:counts.shape[0]], :][:, chrom_subset_index[:counts.shape[1]]]
    else:
        raise ValueError('Input counts matrix is - %d by %d. Counts must be n-by-n or n-by-2n.' % (counts.shape[0], counts.shape[1]))

    if as_sparse:
        counts[np.isnan(counts)] = 0
        counts = sparse.coo_matrix(counts)

    return counts


def check_counts(counts, as_sparse=True):
    if not isinstance(counts, list):
        counts = [counts]
    return [check_counts_matrix(c, as_sparse) for c in counts]


def percent_zero_beads(counts):
    #axis0sum = np.tile(np.array(counts.sum(axis=0).flatten()).flatten(), int(max(counts.shape) / counts.shape[1]))
    #axis1sum = np.tile(np.array(counts.sum(axis=1).flatten()).flatten(), int(max(counts.shape) / counts.shape[0]))
    #return (axis0sum + axis1sum == 0).astype(float).sum() / max(counts.shape)
    return find_beads_to_remove(counts, max(counts.shape)).sum() / max(counts.shape)


def prep_counts(counts_list, lengths, ploidy=1, multiscale_factor=1, normalize=False, filter_counts=False, filter_percentage=0.04, as_sparse=True, verbose=True):

    nbeads = lengths.sum() * ploidy
    counts_dict = [('haploid' if ploidy == 1 else {1: 'ambig', 1.5: 'pa', 2: 'ua'}[sum(c.shape) / nbeads], c) for c in counts_list]
    if len(counts_dict) != len(dict(counts_dict)):
        raise ValueError("Can't input multiple counts matrices of the same type. Inputs = %s" % ', '.join([x[0] for x in counts_dict]))
    counts_dict = dict(counts_dict)

    # Copy counts
    counts_dict = {counts_type: counts.copy() for counts_type, counts in counts_dict.items()}

    # Check counts
    counts_dict = {counts_type: check_counts_matrix(counts, as_sparse=True) for counts_type, counts in counts_dict.items()}

    # Reduce resolution
    lengths_lowres = lengths
    for counts_type, counts in counts_dict.items():
        if multiscale_factor != 1:
            counts, lengths_lowres = multiscale_optimization.reduce_counts_res(counts, multiscale_factor, lengths, ploidy)
            counts_dict[counts_type] = counts

    # Filter all counts together... and if an entire bead has 0 counts from this filtering, set that bead to 0 in all counts matrices
    if filter_counts and len(counts_list) > 1:
        if verbose:
            print('FILTERING LOW COUNTS: manually filtering all counts together by %g' % filter_percentage, flush=True)
        all_counts_ambiguated = ambiguate_counts(list(counts_dict.values()), lengths_lowres.sum())
        initial_zero_beads = find_beads_to_remove(all_counts_ambiguated, lengths_lowres.sum()).sum()
        all_counts_filtered = filter_low_counts(sparse.coo_matrix(all_counts_ambiguated), sparsity=False, percentage=filter_percentage + percent_zero_beads(all_counts_ambiguated)).tocoo()
        torm = find_beads_to_remove(all_counts_filtered, lengths_lowres.sum())
        if verbose:
            print('                      removing %d beads' % (torm.sum() - initial_zero_beads), flush=True)
        for counts_type, counts in counts_dict.items():
            if sparse.issparse(counts):
                counts = counts.toarray()
            #print(counts_type, counts[np.tile(torm, int(counts.shape[0] / torm.shape[0])), :].sum(), counts[:, np.tile(torm, int(counts.shape[1] / torm.shape[0]))].sum(), flush=True)
            counts[np.tile(torm, int(counts.shape[0] / torm.shape[0])), :] = 0.
            counts[:, np.tile(torm, int(counts.shape[1] / torm.shape[0]))] = 0.
            counts = sparse.coo_matrix(counts)
            counts_dict[counts_type] = counts

    # Filter counts
    if filter_counts:
        individual_counts_torms = np.full((lengths_lowres.sum(),), False)
        for counts_type, counts in counts_dict.items():
            if verbose:
                print('FILTERING LOW COUNTS: manually filtering %s counts by %g' % (counts_type.upper(), filter_percentage), flush=True)
            initial_zero_beads = find_beads_to_remove(ambiguate_counts(counts, lengths_lowres.sum()), lengths_lowres.sum()).sum()
            if counts_type == 'pa':
                if sparse.issparse(counts):
                    counts = counts.toarray()
                counts_filtered = np.zeros_like(counts)
                homo1_upper = np.triu(counts[:min(counts.shape), :], 1)
                homo1_lower = np.triu(counts[:min(counts.shape), :].T, 1)
                homo2_upper = np.triu(counts[min(counts.shape):, :], 1)
                homo2_lower = np.triu(counts[min(counts.shape):, :].T, 1)
                counts_filtered[:min(counts.shape), :] += filter_low_counts(sparse.coo_matrix(homo1_upper), sparsity=False,
                                                                            percentage=filter_percentage + percent_zero_beads(homo1_upper)).toarray()
                counts_filtered[:min(counts.shape), :] += filter_low_counts(sparse.coo_matrix(homo1_lower), sparsity=False,
                                                                            percentage=filter_percentage + percent_zero_beads(homo1_lower)).toarray().T
                counts_filtered[min(counts.shape):, :] += filter_low_counts(sparse.coo_matrix(homo2_upper), sparsity=False,
                                                                            percentage=filter_percentage + percent_zero_beads(homo2_upper)).toarray()
                counts_filtered[min(counts.shape):, :] += filter_low_counts(sparse.coo_matrix(homo2_lower), sparsity=False,
                                                                            percentage=filter_percentage + percent_zero_beads(homo2_lower)).toarray().T
                counts = counts_filtered
            else:
                counts = filter_low_counts(sparse.coo_matrix(counts), sparsity=False, percentage=filter_percentage + percent_zero_beads(counts)).tocoo()
            torm = find_beads_to_remove(ambiguate_counts(counts, lengths_lowres.sum()), lengths_lowres.sum())
            if verbose:
                print('                      removing %d beads' % (torm.sum() - initial_zero_beads), flush=True)
            individual_counts_torms = individual_counts_torms | torm
            counts = sparse.coo_matrix(counts)
            counts_dict[counts_type] = counts
        if verbose:
            print('******* %d' % individual_counts_torms.sum(), flush=True)

    # Optionally normalize counts
    bias = None
    if normalize:
        if verbose:
            print('COMPUTING BIAS: all counts together', flush=True)
        bias = ICE_normalization(ambiguate_counts(list(counts_dict.values()), lengths_lowres.sum()), max_iter=300, output_bias=True)[1].flatten()
        # In each counts matrix, zero out counts for which bias is NaN
        for counts_type, counts in counts_dict.items():
            initial_zero_beads = find_beads_to_remove(ambiguate_counts(counts, lengths_lowres.sum()), lengths_lowres.sum()).sum()
            if sparse.issparse(counts):
                counts = counts.toarray()
            counts[np.tile(np.isnan(bias), int(counts.shape[0] / bias.shape[0])), :] = 0.
            counts[:, np.tile(np.isnan(bias), int(counts.shape[1] / bias.shape[0]))] = 0.
            counts = sparse.coo_matrix(counts)
            counts_dict[counts_type] = counts
            torm = find_beads_to_remove(ambiguate_counts(counts, lengths_lowres.sum()), lengths_lowres.sum())
            if verbose:
                print('                removing %d beads from %s' % (torm.sum() - initial_zero_beads, counts_type), flush=True)

    return check_counts(list(counts_dict.values()), as_sparse), bias


def format_counts(counts, beta, input_weight, lengths, ploidy, as_sparse, lighter0=False, nonUA0=False):
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

    # Reformat counts as sparse_counts_matrix or zero_counts_matrix objects
    counts_reformatted = []
    for counts_maps, beta_maps, input_weight_maps in zip(counts, beta, input_weight):
        counts_reformatted.append(sparse_counts_matrix(counts_maps, lengths, ploidy, beta_maps, input_weight_maps))
        if not as_sparse:
            counts_reformatted.append(zero_counts_matrix(counts_maps, lengths, ploidy, beta_maps, input_weight_maps, lighter0, nonUA0))

    return counts_reformatted


class sparse_counts_matrix(object):
    """
    """
    def __init__(self, counts, lengths, ploidy, beta=None, weight=1.):
        counts = counts.copy()
        if sparse.issparse(counts):
            counts = counts.toarray()
        self.input_sum = np.nansum(counts)
        counts[np.isnan(counts)] = 0
        counts = counts.astype(sparse.sputils.get_index_dtype(maxval=counts.max()))
        self.counts = sparse.coo_matrix(counts)
        self.ambiguity = {1: 'ambig', 1.5: 'pa', 2: 'ua'}[sum(counts.shape) / (lengths.sum() * ploidy)]
        self.name = self.ambiguity
        self.beta = beta
        self.weight = (1. if weight is None else weight)

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
        return self

    def copy(self):
        return self.counts.copy()

    def sum(self, axis=None, dtype=None, out=None):
        return self.counts.sum(axis=axis, dtype=dtype, out=out)

    def bias_per_bin(self, bias, ploidy):
        if bias is None:
            return np.ones((self.nnz,))
        else:
            bias = bias.flatten()
            bias = np.tile(bias, int(min(self.shape) * ploidy / len(bias)))
            return bias[self.row] * bias[self.col]


class zero_counts_matrix(object):
    """
    """
    def __init__(self, counts, lengths, ploidy, beta=None, weight=1., lighter0=False, nonUA0=False):
        from .utils import create_dummy_counts

        counts = counts.copy()
        if sparse.issparse(counts):
            counts = counts.toarray()
        self.input_sum = np.nansum(counts)
        counts[counts != 0] = np.nan
        if not nonUA0:
            dummy_counts = create_dummy_counts(counts, lengths, ploidy)
        else:
            dummy_counts = counts.copy() + 1
            dummy_counts[np.isnan(dummy_counts)] = 0
            dummy_counts = sparse.coo_matrix(dummy_counts)
        self.row = dummy_counts.row
        self.col = dummy_counts.col
        self.shape = dummy_counts.shape
        self.nnz = len(self.row)
        self.ambiguity = {1: 'ambig', 1.5: 'pa', 2: 'ua'}[sum(counts.shape) / (lengths.sum() * ploidy)]
        self.name = '%s0' % self.ambiguity
        self.beta = beta
        self.weight = (1. if weight is None else weight)
        if lighter0:
            self.weight /= {'ambig0': 4, 'pa0': 2, 'ua0': 1}[self.ambiguity]

    def getdata(self):
        return np.zeros_like(self.row)

    @property
    def data(self):
        return self.getdata()

    def toarray(self):
        array = np.full(self.shape, np.nan)
        array[self.row, self.col] = 0
        return array

    def tocoo(self):
        return sparse.coo_matrix(self.toarray())

    def copy(self):
        self.row = self.row.copy()
        self.col = self.col.copy()
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

    '''def set_dis_indices(self, lengths, ploidy, multiscale_factor=1, mask=None):
        from .utils import get_dis_indices
        from .multiscale_optimization import decrease_lengths_res

        self.dis_row = None
        self.dis_col = None

        if multiscale_factor != 1:
            lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)
            dummy_counts = sparse.coo_matrix((np.ones_like(self.row), (self.row, self.col)), shape=self.shape)
            self.dis_row, self.dis_col = get_dis_indices(dummy_counts, lengths_lowres.sum(), lengths, ploidy, multiscale_factor=multiscale_factor, nbeads=lengths.sum() * ploidy, mask=mask)

            idx_dtype = sparse.sputils.get_index_dtype(maxval=lengths.sum() * ploidy)
            self.dis_row = self.dis_row.astype(idx_dtype)
            self.dis_col = self.dis_col.astype(idx_dtype)

    @property
    def dis_row(self):
        if self.dis_row is None:
            return self.row
        else:
            return self.dis_row

    @property
    def dis_col(self):
        if self.dis_col is None:
            return self.col
        else:
            return self.dis_col'''
