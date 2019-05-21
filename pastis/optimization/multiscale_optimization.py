import numpy as np
from .utils import row_and_col


def decrease_lengths_res(lengths, factor):
    return np.ceil(np.array(lengths).astype(float) / factor).astype(int)


def increase_X_res(X, multiscale_factor, lengths, mask=None):
    raise NotImplementedError


def reduce_counts_res(counts, multiscale_factor, lengths, ploidy):
    from topsy.inference.utils import convert_indices_to_full_res
    from scipy import sparse

    input_is_sparse = False
    if sparse.issparse(counts):
        counts = counts.copy().toarray()
        input_is_sparse = True

    lengths_lowres = np.ceil(np.array(lengths).astype(float) / multiscale_factor).astype(int)
    counts_lowres = np.ones(np.array(counts.shape / lengths.sum() * lengths_lowres.sum(), dtype=int))
    np.fill_diagonal(counts_lowres, 0)
    counts_lowres = sparse.coo_matrix(counts_lowres)
    rows_raw, cols_raw = row_and_col(counts_lowres)
    rows, cols = convert_indices_to_full_res(rows_raw, cols_raw, rows_max=counts.shape[0], cols_max=counts.shape[1], multiscale_factor=multiscale_factor, lengths=lengths, n=lengths_lowres.sum(), counts_shape=counts_lowres.shape, ploidy=ploidy)
    data = counts[rows, cols].reshape(multiscale_factor ** 2, -1).sum(axis=0)
    counts_lowres = sparse.coo_matrix((data[data != 0], (rows_raw[data != 0], cols_raw[data != 0])), shape=counts_lowres.shape)

    if not input_is_sparse:
        counts_lowres = counts_lowres.toarray()

    return counts_lowres, lengths_lowres


def get_X_indices(ploidy, multiscale_factor, lengths):
    raise NotImplementedError


def reduce_X_res(X, multiscale_factor, lengths, indices=None, mask=None):
    raise NotImplementedError


def repeat_X_multiscale(structures, lengths, multiscale_factor):
    raise NotImplementedError
