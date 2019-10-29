import numpy as np
from sklearn.metrics import euclidean_distances
from scipy import sparse


def print_code_header(header, sub_header=None, max_length=80, blank_lines=None,
                      verbose=True):
    """Prints a header, for demarcation of output.
    """

    if verbose:
        print('=' * max_length, flush=True)
        print(('=' * int(np.ceil((max_length - len(header) - 2) / 2))) + ' %s ' %
              header + ('=' * int(np.floor((max_length - len(header) - 2) / 2))), flush=True)
        if sub_header is not None and len(sub_header) != 0:
            print(('=' * int(np.ceil((max_length - len(sub_header) - 2) / 2))) + ' %s ' %
                  sub_header + ('=' * int(np.floor((max_length - len(sub_header) - 2) / 2))), flush=True)
        print('=' * max_length, flush=True)
        if blank_lines is not None and blank_lines > 0:
            print('\n' * (blank_lines - 1), flush=True)


def find_beads_to_remove(counts, nbeads, threshold=0):
    """Determine beads for which no corresponding counts data exists.
    """

    if not isinstance(counts, list):
        counts = [counts]
    inverse_torm = np.zeros(int(nbeads))
    for c in counts:
        if isinstance(c, np.ndarray):
            axis0sum = np.tile(
                np.array(np.nansum(c, axis=0).flatten()).flatten(),
                int(nbeads / c.shape[1]))
            axis1sum = np.tile(
                np.array(np.nansum(c, axis=1).flatten()).flatten(),
                int(nbeads / c.shape[0]))
        else:
            axis0sum = np.tile(
                np.array(c.sum(axis=0).flatten()).flatten(),
                int(nbeads / c.shape[1]))
            axis1sum = np.tile(
                np.array(c.sum(axis=1).flatten()).flatten(),
                int(nbeads / c.shape[0]))
        inverse_torm += (axis0sum + axis1sum > threshold).astype(int)
    torm = ~ inverse_torm.astype(bool)
    return torm


def constraint_dis_indices(counts, n, lengths, ploidy, mask=None,
                           adjacent_beads_only=False):
    """Return distance matrix indices associated with any counts matrix data.
    """

    n = int(n)

    if isinstance(counts, list) and len(counts) == 1:
        counts = counts[0]
    if not isinstance(counts, list):
        rows = counts.row3d
        cols = counts.col3d
    else:
        rows = []
        cols = []
        for counts_maps in counts:
            rows.append(counts_maps.row3d)
            cols.append(counts_maps.col3d)
        rows, cols = np.split(np.unique(np.concatenate([np.atleast_2d(
            np.concatenate(rows)), np.atleast_2d(np.concatenate(cols))],
            axis=0), axis=1), 2, axis=0)
        rows = rows.flatten()
        cols = cols.flatten()

    if adjacent_beads_only:
        if mask is None:
            # Calculating distances for adjacent beads, which are on the off
            # diagonal line - i & j where j = i + 1
            rows = np.unique(rows)
            rows = rows[np.isin(rows + 1, cols)]
            # Remove if "adjacent" beads are actually on different chromosomes
            # or homologs
            rows = rows[np.digitize(rows, np.tile(lengths, ploidy).cumsum()) == np.digitize(
                rows + 1, np.tile(lengths, ploidy).cumsum())]
            cols = rows + 1
        else:
            # Calculating distances for adjacent beads, which are on the off
            # diagonal line - i & j where j = i + 1
            rows_adj = np.unique(rows)
            rows_adj = rows_adj[np.isin(rows_adj + 1, cols)]
            # Remove if "adjacent" beads are actually on different chromosomes
            # or homologs
            rows_adj = rows_adj[np.digitize(rows_adj, np.tile(lengths, ploidy).cumsum(
            )) == np.digitize(rows_adj + 1, np.tile(lengths, ploidy).cumsum())]
            cols_adj = rows_adj + 1

    if mask is not None:
        rows[~mask] = 0
        cols[~mask] = 0
        if adjacent_beads_only:
            include = (np.isin(rows, rows_adj) & np.isin(cols, cols_adj))
            rows = rows[include]
            cols = cols[include]

    return rows, cols


def struct_replace_nan(X, lengths, kind='linear', random_state=None):
    """Replace NaNs in structure via linear interpolation.
    """

    from scipy.interpolate import interp1d
    from warnings import warn

    if random_state is None:
        random_state = np.random.RandomState(seed=0)

    if isinstance(X, str):
        X = np.loadtxt(X)
    else:
        X = X.copy()
    X = X.reshape(-1, 3)
    lengths = np.array(lengths).astype(int)

    ploidy = 1
    if len(X) > lengths.sum():
        ploidy = 2

    if not np.isnan(X).any():
        return(X)
    else:
        nan_chroms = []
        mask = np.invert(np.isnan(X[:, 0]))
        interpolated_X = np.zeros(X.shape)
        begin, end = 0, 0
        for j, length in enumerate(np.tile(lengths, ploidy)):
            end += length
            to_rm = mask[begin:end]
            if to_rm.sum() <= 1:
                interpolated_chr = (
                    1 - 2 * random_state.rand(length * 3)).reshape(-1, 3)
                if ploidy == 1:
                    nan_chroms.append(str(j + 1))
                else:
                    nan_chroms.append(
                        str(j + 1) + '_homo1' if j < lengths.shape[0] else str(j / 2 + 1) + '_homo2')
            else:
                m = np.arange(length)[to_rm]
                beads2interpolate = np.arange(m.min(), m.max() + 1, 1)

                interpolated_chr = np.full_like(X[begin:end, :], np.nan)
                interpolated_chr[beads2interpolate, 0] = interp1d(
                    m, X[begin:end, 0][to_rm], kind=kind)(beads2interpolate)
                interpolated_chr[beads2interpolate, 1] = interp1d(
                    m, X[begin:end, 1][to_rm], kind=kind)(beads2interpolate)
                interpolated_chr[beads2interpolate, 2] = interp1d(
                    m, X[begin:end, 2][to_rm], kind=kind)(beads2interpolate)

                # Fill in beads at start
                diff_beads_at_chr_start = interpolated_chr[beads2interpolate[
                    1], :] - interpolated_chr[beads2interpolate[0], :]
                how_far = 1
                for j in reversed(range(min(beads2interpolate))):
                    interpolated_chr[j, :] = interpolated_chr[
                        beads2interpolate[0], :] - diff_beads_at_chr_start * how_far
                    how_far += 1
                # Fill in beads at end
                diff_beads_at_chr_end = interpolated_chr[
                    beads2interpolate[-2], :] - interpolated_chr[beads2interpolate[-1], :]
                how_far = 1
                for j in range(max(beads2interpolate) + 1, length):
                    interpolated_chr[j, :] = interpolated_chr[
                        beads2interpolate[-1], :] - diff_beads_at_chr_end * how_far
                    how_far += 1

            interpolated_X[begin:end, :] = interpolated_chr
            begin = end

        if len(nan_chroms) != 0:
            warn('The following chromosomes were all NaN: ' + ' '.join(nan_chroms))

        return(interpolated_X)


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
