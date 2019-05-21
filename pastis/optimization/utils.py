import numpy as np
from sklearn.metrics import euclidean_distances
from scipy import sparse


def print_code_header(header, sub_header=None, max_length=80, blank_lines=None, verbose=True):
    if verbose:
        print('=' * max_length, flush=True)
        print(('=' * int(np.ceil((max_length - len(header) - 2) / 2))) + ' %s ' % header + ('=' * int(np.floor((max_length - len(header) - 2) / 2))), flush=True)
        if sub_header is not None and len(sub_header) != 0:
            print(('=' * int(np.ceil((max_length - len(sub_header) - 2) / 2))) + ' %s ' % sub_header + ('=' * int(np.floor((max_length - len(sub_header) - 2) / 2))), flush=True)
        print('=' * max_length, flush=True)
        if blank_lines is not None and blank_lines > 0:
            print('\n' * (blank_lines - 1), flush=True)


def row_and_col(data):
    if isinstance(data, np.ndarray):
        return np.where(~np.isnan(data))
    else:
        #if not sparse.isspmatrix_coo(data):
        #    data = sparse.coo_matrix(data)
        return data.row, data.col


def nnz(data):
    if isinstance(data, np.ndarray):
        return (~np.isnan(data)).sum()
    else:
        #if not sparse.isspmatrix_coo(data):
        #    data = sparse.coo_matrix(data)
        return data.nnz


def get_data(data):
    if isinstance(data, np.ndarray):
        return data[~np.isnan(data)].flatten()
    else:
        #if not sparse.isspmatrix_coo(data):
        #    data = sparse.coo_matrix(data)
        return data.data


def is_bin_zero_in_fullres_counts(i, j, fullres_indices):
    return (i, j) in fullres_indices


is_bin_zero_in_fullres_counts_vect = np.vectorize(is_bin_zero_in_fullres_counts, excluded=['fullres_indices'])


def mask_zero_in_fullres_counts(counts, fullres_counts, lengths, ploidy, multiscale_factor):
    from .multiscale_optimization import decrease_lengths_res
    if not isinstance(counts, list):
        counts = [counts]
    if not isinstance(fullres_counts, list):
        fullres_counts = [fullres_counts]
    if len(counts) != len(fullres_counts):
        raise ValueError('Counts list must be the same length as fullres_counts list')
    lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)
    output = []
    for counts_map, fullres_counts_map in zip(counts, fullres_counts):
        rows, cols = get_dis_indices(counts_map, n=lengths_lowres.sum(), lengths=lengths, ploidy=ploidy, multiscale_factor=multiscale_factor, nbeads=lengths.sum() * ploidy)
        fullres_indices = set(zip(row_and_col(fullres_counts_map)))
        output.append(is_bin_zero_in_fullres_counts_vect(i=rows, j=cols, fullres_indices=fullres_indices))
    return output


def is_index_in_fullres_X(i, fullres_index):
    return i in fullres_index


is_index_in_fullres_X_vect = np.vectorize(is_index_in_fullres_X, excluded=['fullres_index'])


def mask_torm_in_fullres_X(fullres_counts, lengths, ploidy, multiscale_factor):
    from .multiscale_optimization import get_X_indices
    torm = find_beads_to_remove(fullres_counts, nbeads=lengths.sum() * ploidy)
    X_indices_fullres = get_X_indices(ploidy=ploidy, multiscale_factor=1, lengths=lengths)[~torm]
    X_indices_lowres = get_X_indices(ploidy=ploidy, multiscale_factor=multiscale_factor, lengths=lengths)
    return is_index_in_fullres_X_vect(i=X_indices_lowres, fullres_index=X_indices_fullres)


def create_dummy_counts(counts, lengths, ploidy, multiscale_factor=1):
    from .multiscale_optimization import decrease_lengths_res
    lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)

    rows, cols = constraint_dis_indices(counts, n=lengths_lowres.sum(), lengths=lengths, ploidy=ploidy, multiscale_factor=multiscale_factor, nbeads=lengths.sum() * ploidy)
    #if multiscale_factor != 1:
    #order = np.lexsort((rows, cols))
    #rows = rows[order]
    #cols = cols[order]
    dummy_counts = sparse.coo_matrix((np.ones_like(rows), (rows, cols)), shape=(lengths.sum() * ploidy, lengths.sum() * ploidy)).toarray()
    dummy_counts = sparse.coo_matrix(np.triu(dummy_counts + dummy_counts.T, 1))

    return dummy_counts


def create_dummy_counts_old(counts, lengths, multiscale_factor=1):
    from .multiscale_optimization import decrease_lengths_res
    lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)
    ambiguated_counts = ambiguate_counts(counts, n=lengths_lowres.sum())

    if sparse.isspmatrix_coo(ambiguated_counts) or multiscale_factor != 1:
        rows, cols = get_dis_indices(ambiguated_counts, n=lengths_lowres.sum(), lengths=lengths, ploidy=1, multiscale_factor=multiscale_factor, nbeads=lengths.sum())
        dummy_counts = sparse.coo_matrix(sparse.coo_matrix((np.ones_like(rows), (rows, cols)), shape=(lengths.sum(), lengths.sum())).toarray())
        dummy_counts = sparse.triu(dummy_counts, 1)
        if not sparse.isspmatrix_coo(ambiguated_counts):
            dummy_counts = dummy_counts.toarray()
    else:
        dummy_counts = np.ones_like(ambiguated_counts)
        dummy_counts[np.isnan(ambiguated_counts)] = np.nan

    return dummy_counts


def find_beads_to_remove(counts, nbeads, threshold=0):
    if not isinstance(counts, list):
        counts = [counts]
    inverse_torm = np.zeros(int(nbeads))
    for c in counts:
        #if not isinstance(c, np.ndarray):
        #    c = c.copy().toarray()
        #axis0sum = np.tile(np.array(np.nansum(c, axis=0).flatten()).flatten(), int(nbeads / c.shape[1]))
        #axis1sum = np.tile(np.array(np.nansum(c, axis=1).flatten()).flatten(), int(nbeads / c.shape[0]))
        if isinstance(c, np.ndarray):
            axis0sum = np.tile(np.array(np.nansum(c, axis=0).flatten()).flatten(), int(nbeads / c.shape[1]))
            axis1sum = np.tile(np.array(np.nansum(c, axis=1).flatten()).flatten(), int(nbeads / c.shape[0]))
        else:
            axis0sum = np.tile(np.array(c.sum(axis=0).flatten()).flatten(), int(nbeads / c.shape[1]))
            axis1sum = np.tile(np.array(c.sum(axis=1).flatten()).flatten(), int(nbeads / c.shape[0]))
        inverse_torm += (axis0sum + axis1sum > threshold).astype(int)
    torm = ~ inverse_torm.astype(bool)
    return torm


def ambiguate_index(rows, cols, n, ambiguate_rows=True, ambiguate_cols=True, triu=0):
    if ambiguate_rows:
        rows[rows >= n] -= n
    if ambiguate_cols:
        cols[cols >= n] -= n
    indices = np.array(sorted(set(zip(rows, cols)), key=lambda x: (x[0], x[1]))).T
    mask = np.where([indices[1], indices[0]] in indices)
    rows = indices[0].flatten()
    cols = indices[1].flatten()
    if triu:
        rows = rows[rows + triu > cols]
        cols = cols[rows + triu > cols]
    return rows, cols


def ambiguate_counts(counts, n, as_sparse=None):
    '''
    :param counts: counts ndarray or sparse matrix, or list of counts
    :param n: sum of lengths
    :return: ambiguated counts
    '''
    from scipy import sparse
    from .prep_counts import check_counts_matrix, zero_counts_matrix, sparse_counts_matrix

    if not isinstance(counts, list):
        counts = [counts]

    if len(counts) == 1 and counts[0].shape == (n, n):
        return counts[0]
    output = np.zeros((n, n))
    for c in counts:
        if not isinstance(c, zero_counts_matrix):
            if not isinstance(c, np.ndarray):
                c = c.toarray()

            if c.shape[0] > c.shape[1]:
                c_ambig = np.nansum([c[:n, :], c[n:, :], c[:n, :].T, c[n:, :].T])
            elif c.shape[0] < c.shape[1]:
                c_ambig = np.nansum([c[:, :n].T, c[:, n:].T, c[:, :n], c[:, n:]])
            elif c.shape[0] == n:
                c_ambig = c
            else:
                c_ambig = np.nansum([c[:n, :n], c[n:, :n], c[:n, n:], c[n:, n:]])
            output[~np.isnan(c_ambig)] += c_ambig[~np.isnan(c_ambig)]

    output = np.triu(output, 1)
    if as_sparse is None:
        as_sparse = all([isinstance(c, sparse_counts_matrix) or sparse.issparse(c) for c in counts])
    return check_counts_matrix(output, as_sparse=as_sparse)


def convert_indices_to_full_res(rows, cols, rows_max, cols_max, multiscale_factor, lengths, n, counts_shape, ploidy):
    if multiscale_factor == 1:
        return rows, cols
    nnz = len(rows)
    x, y = np.indices((multiscale_factor, multiscale_factor))
    rows = np.repeat(x.flatten(), nnz)[:nnz * multiscale_factor ** 2] + np.tile(rows * multiscale_factor, multiscale_factor ** 2)
    cols = np.repeat(y.flatten(), nnz)[:nnz * multiscale_factor ** 2] + np.tile(cols * multiscale_factor, multiscale_factor ** 2)
    rows = rows.reshape(multiscale_factor ** 2, -1)
    cols = cols.reshape(multiscale_factor ** 2, -1)
    # Figure out which rows / cols are out of bounds
    bins_for_rows = np.tile(lengths, int(counts_shape[0] / n)).cumsum()
    bins_for_cols = np.tile(lengths, int(counts_shape[1] / n)).cumsum()
    for i in range(lengths.shape[0] * ploidy):
        rows_binned = np.digitize(rows, bins_for_rows)
        cols_binned = np.digitize(cols, bins_for_cols)
        incorrect_rows = np.invert(np.equal(rows_binned, np.floor(rows_binned.mean(axis=0))))
        incorrect_cols = np.invert(np.equal(cols_binned, np.floor(cols_binned.mean(axis=0))))
        for val in np.unique(rows[:, np.floor(rows_binned.mean(axis=0)) == i][incorrect_rows[:, np.floor(rows_binned.mean(axis=0)) == i]]):
            rows[rows > val] -= 1
        for val in np.unique(cols[:, np.floor(cols_binned.mean(axis=0)) == i][incorrect_cols[:, np.floor(cols_binned.mean(axis=0)) == i]]):
            cols[cols > val] -= 1
    incorrect_indices = incorrect_rows + incorrect_cols + (rows >= rows_max) + (cols >= cols_max)
    # If a bin spills over chromosome / homolog boundaries, set it to (0, 0) so that distances will be 0
    # Must then set dis[dis == 0] = np.inf before raising dis to ** alpha
    #rows = rows.astype(float); cols = cols.astype(float); rows[incorrect_indices] = np.nan; cols[incorrect_indices] = np.nan; print(rows[:, 5]); print(cols[:, 5]); rows[incorrect_indices] = 0; cols[incorrect_indices] = 0; rows = rows.astype(int); cols = cols.astype(int)
    rows[incorrect_indices] = 0
    cols[incorrect_indices] = 0
    rows = rows.flatten()
    cols = cols.flatten()
    return rows, cols


def convert_counts_indices_to_dist(counts, n, ploidy):
    n = int(n)

    rows, cols = row_and_col(counts)

    if counts.shape[0] != n * ploidy or counts.shape[1] != n * ploidy:
        nnz = len(rows)

        map_factor = int(n * ploidy / min(counts.shape))
        map_factor_rows = int(n * ploidy / counts.shape[0])
        map_factor_cols = int(n * ploidy / counts.shape[1])

        x, _ = np.indices((map_factor_rows, map_factor_rows))
        _, y = np.indices((map_factor_cols, map_factor_cols))
        x = x.flatten()
        y = y.flatten()
        tile_factor = max(x.shape[0], y.shape[0])

        rows = np.repeat(x, nnz * int(tile_factor / x.shape[0])) * min(counts.shape) + np.tile(rows, map_factor ** 2)
        cols = np.repeat(y, nnz * int(tile_factor / y.shape[0])) * min(counts.shape) + np.tile(cols, map_factor ** 2)

    return rows, cols


def get_dis_indices(counts, n, lengths, ploidy, multiscale_factor=1, nbeads=None, mask=None):
    n = int(n)
    if nbeads is None:
        if multiscale_factor != 1:
            raise ValueError('Must supply nbeads if multiscale_factor > 1.')
        nbeads = n * ploidy
    nbeads = int(nbeads)

    rows, cols = convert_counts_indices_to_dist(counts, n, ploidy)
    if multiscale_factor != 1:
        rows, cols = convert_indices_to_full_res(rows, cols, rows_max=nbeads, cols_max=nbeads, multiscale_factor=multiscale_factor, lengths=lengths, n=n, counts_shape=(n * ploidy, n * ploidy), ploidy=ploidy)

    if mask is not None:
        rows[~mask] = 0
        cols[~mask] = 0

    return rows, cols


def constraint_dis_indices(counts, n, lengths, ploidy, multiscale_factor=1, nbeads=None, mask=None, adjacent_beads_only=False):
    n = int(n)
    if nbeads is None:
        if multiscale_factor != 1:
            raise ValueError('Must supply nbeads if multiscale_factor > 1.')
        nbeads = n * ploidy
    nbeads = int(nbeads)

    if isinstance(counts, list) and len(counts) == 1:
        counts = counts[0]
    if not isinstance(counts, list):
        rows, cols = convert_counts_indices_to_dist(counts, n, ploidy)
    else:
        rows = []
        cols = []
        for counts_maps in counts:
            rows_maps, cols_maps = convert_counts_indices_to_dist(counts_maps, n, ploidy)
            rows.append(rows_maps)
            cols.append(cols_maps)
        rows, cols = np.split(np.unique(np.concatenate([np.atleast_2d(np.concatenate(rows)), np.atleast_2d(np.concatenate(cols))], axis=0), axis=1), 2, axis=0)
        rows = rows.flatten()
        cols = cols.flatten()
        #order = np.lexsort((rows, cols))
        #rows = rows[order]
        #cols = cols[order]

    if adjacent_beads_only:
        if mask is None:
            # Calculating distances for adjacent beads, which are on the off diagonal line - i & j where j = i + 1
            rows = np.unique(rows)
            rows = rows[np.isin(rows + 1, cols)]
            # Remove if "adjacent" beads are actually on different chromosomes or homologs
            rows = rows[np.digitize(rows, np.tile(lengths, ploidy).cumsum()) == np.digitize(rows + 1, np.tile(lengths, ploidy).cumsum())]
            cols = rows + 1
        else:
            # Calculating distances for adjacent beads, which are on the off diagonal line - i & j where j = i + 1
            rows_adj = np.unique(rows)
            rows_adj = rows_adj[np.isin(rows_adj + 1, cols)]
            # Remove if "adjacent" beads are actually on different chromosomes or homologs
            rows_adj = rows_adj[np.digitize(rows_adj, np.tile(lengths, ploidy).cumsum()) == np.digitize(rows_adj + 1, np.tile(lengths, ploidy).cumsum())]
            cols_adj = rows_adj + 1
            if multiscale_factor != 1:
                rows_adj, cols_adj = convert_indices_to_full_res(rows_adj, cols_adj, rows_max=nbeads, cols_max=nbeads, multiscale_factor=multiscale_factor, lengths=lengths, n=n, counts_shape=(n * ploidy, n * ploidy), ploidy=ploidy)

    if multiscale_factor != 1:
        rows, cols = convert_indices_to_full_res(rows, cols, rows_max=nbeads, cols_max=nbeads, multiscale_factor=multiscale_factor, lengths=lengths, n=n, counts_shape=(n * ploidy, n * ploidy), ploidy=ploidy)

    if mask is not None:
        rows[~mask] = 0
        cols[~mask] = 0
        if adjacent_beads_only:
            include = (np.isin(rows, rows_adj) & np.isin(cols, cols_adj))
            rows = rows[include]
            cols = cols[include]

    return rows, cols


class ConstantDispersion(object):
    def __init__(self, coef=7):
        self.coef = coef

    def fit(self, X, y):
        return self

    def predict(self, X):
        return self.coef * np.ones(X.shape)

    def derivate(self, X):
        return np.zeros(X.shape)


def compute_wish_distances(counts, alpha=-3., beta=1., bias=None):
    """
    Computes wish distances from a counts matrix

    Parameters
    ----------
    counts : ndarray
        Interaction counts matrix

    alpha : float, optional, default: -3
        Coefficient of the power law

    beta : float, optional, default: 1
        Scaling factor

    Returns
    -------
    wish_distances
    """
    if beta == 0:
        raise ValueError("beta cannot be equal to 0.")
    counts = counts.copy()
    if sparse.issparse(counts):
        if not sparse.isspmatrix_coo(counts):
            counts = counts.tocoo()
        if bias is not None:
            bias = bias.flatten()
            counts.data /= bias[counts.row] * bias[counts.col]
        wish_distances = counts / beta
        wish_distances.data[wish_distances.data != 0] **= 1. / alpha
        return wish_distances
    else:
        wish_distances = counts.copy() / beta
        wish_distances[wish_distances != 0] **= 1. / alpha

        return wish_distances


def eval_no_f(x, user_data=None):
    """
    Evaluate the object function (no objective function).
    """
    return 0.


def eval_grad_no_f(X, user_data=None):
    m, n, counts, alpha, beta, d = user_data
    return np.zeros(n * m)


def eval_g_no(x, user_data=None):
    return np.array([0, 0.])


def eval_jac_g_no(x, flag, user_data=None):
    if flag:
        return 0, 0
    return np.array([0., 0.])


def eval_g(x, user_data=None):
    """
    Computes the constraints
    """

    m, n, wish_dist, alpha, beta, d = user_data

    x = x.reshape((m, n))
    dis = euclidean_distances(x)
    dis = dis ** 2
    mask = np.invert(np.tri(m, dtype=np.bool))
    g = np.concatenate([dis[mask].flatten(), ((x - d) ** 2).sum(axis=1)])
    return g


def eval_jac_g(x, flag, user_data=None):
    """
    Computes the jacobian for the constraints mentionned in duan-et-al
    """

    m, n, wish_dist, alpha, beta, d = user_data

    if flag:
        ncon = m * (m - 1) / 2
        row = np.arange(ncon).repeat(2 * n)

        tmp = np.arange(n).repeat(m - 1)
        tmp = tmp.reshape((n, m - 1))
        tmp = tmp.T
        tmp1 = np.arange(n, m * n)
        tmp1 = tmp1.reshape((m - 1, n))
        tmp = np.concatenate((tmp, tmp1), axis=1)
        tmp1 = tmp.copy()
        for it in range(m):
            tmp += n
            tmp = tmp[:-1]
            tmp1 = np.concatenate((tmp1, tmp))

        # The second part of the jacobian is the restrictions on the distances
        # to the origin and/or the distances to the SPB/nucleolus center
        row_2 = np.arange(m).repeat(n)
        col = np.arange(n * m)

        col = np.concatenate([tmp1.flatten(), col])
        row = np.concatenate([row, row_2])
        return row.flatten(), col.flatten()
    else:
        x = x.reshape((m, n))
        tmp = x.repeat(m, axis=0).reshape((m, m, n))
        dif = tmp - tmp.transpose(1, 0, 2)
        mask = np.invert(np.tri(m, dtype=np.bool))
        dif = dif[mask]
        jac = 2 * np.concatenate((dif, - dif), axis=1).flatten()

        # The second part of the jacobian is the restrictions on the distances
        # to the origin and/or the distances to the SPB/nucleolus center
        jac2 = 2 * (x - d).flatten()
        return np.concatenate([jac, jac2]).flatten()


def eval_h(x, lagrange, obj_factor, flag, user_data=None):
    """
    """
    return False
