import numpy as np
from .multiscale_optimization import decrease_lengths_res


def parse_homolog_sep(constraints, homo_init, lengths, counts=None, fullres_counts=None, X_true=None, init_structures=None, alpha=-3., beta=1., multiscale_factor=1, HSC_lowres_beads=5, modifications=None):
    from .poisson_diploid import check_constraints
    _, constraints = check_constraints(lagrange_mult=None, constraints=constraints, verbose=False)

    constraints['homo'] = get_inter_homolog_ratio(constraints['homo'], lengths, true_structure=X_true, init_structure=init_structures, counts=counts, alpha=alpha, beta=beta, multiscale_factor=multiscale_factor, modifications=modifications)
    constraints['homodis'] = get_inter_homolog_dis(constraints['homodis'], lengths, true_structure=X_true, init_structure=init_structures, counts=fullres_counts, alpha=alpha, beta=beta, HSC_lowres_beads=HSC_lowres_beads, modifications=modifications)

    if homo_init is not None and isinstance(homo_init, str):
        if homo_init.lower() in ('homo', 'homodis'):
            homo_init = constraints[homo_init]
        else:
            raise ValueError('Inter-homolog init must be set to inter-homolog ratio ("homo") or inter-homolog distance ("homodis")')

    return constraints, homo_init


# =============================================================================================================================================================================
# =============================================              INTER-HOMOLOG RATIO CONSTRAINT                ====================================================================
# =============================================================================================================================================================================


def get_inter_homolog_ratio(input_ratio, lengths, init_structure=None, true_structure=None, counts=None, alpha=-3., beta=1., multiscale_factor=1, verbose=True, modifications=None):
    if input_ratio is None:
        return None
    try:
        output = float(input_ratio)
    except ValueError:
        lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)
        if input_ratio.lower() == 'true' or input_ratio.lower() == 'x_true' or input_ratio.lower() == 'auto':
            if true_structure is None:
                raise ValueError('Must input X_true to calculate inter-homolog ratio from X_true')
            output = inter_homolog_ratio(true_structure, counts=counts, lengths=lengths_lowres, alpha=alpha, multiscale_factor=1, modifications=modifications)
        elif init_structure is not None:
            output = inter_homolog_ratio(init_structure, counts=counts, lengths=lengths_lowres, alpha=alpha, multiscale_factor=1, modifications=modifications)
        else:
            if counts is None:
                raise ValueError('Must input counts to calculate inter-homolog ratio from counts')
            #ua_index = [i for i in range(len(counts)) if counts[i].shape == (int(lengths_lowres.sum() * 2), int(lengths_lowres.sum() * 2))]
            ua_index = [i for i in range(len(counts)) if counts[i].name == 'ua']
            if input_ratio.lower() == 'ua' or input_ratio.lower() == 'unambig' or (input_ratio.lower() == 'counts' and len(ua_index) > 0):
                output = inter_homolog_ratio_via_ua_counts_OLD(counts[ua_index[0]], lengths_lowres, alpha=alpha, beta=beta, multiscale_factor=1, modifications=modifications)
            elif input_ratio.lower() == 'a' or input_ratio.lower() == 'ambig' or input_ratio.lower() == 'pa':
                output = inter_homolog_ratio_via_ambig_counts(counts, lengths_lowres, alpha=alpha, beta=beta, multiscale_factor=1, modifications=modifications)
            else:
                raise ValueError('Error calculating inter-homolog ratio, input "%s" not recognized' % input_ratio)
    if verbose:
        print('Estimated ratio of inter-homolog counts is %.2g' % output, flush=True)
    return output


def inter_homolog_ratio(X, counts=None, lengths=None, alpha=-3., multiscale_factor=1, modifications=None):
    from .utils import constraint_dis_indices#, get_dis_indices, create_dummy_counts_old
    from sklearn.metrics import euclidean_distances

    X = X.reshape(-1, 3)
    n = int(X.shape[0] / 2)

    if alpha is None:
        alpha = -3.
    if lengths is None:
        lengths = np.array([n])

    if counts is None:
        mask = np.invert(np.isnan(X[:, 0]))
        tmp = np.triu(euclidean_distances(X[mask]), 1)
        if modifications is not None and 'homo_with_alpha' in modifications:
            tmp[np.isclose(tmp, 0)] = np.inf
            tmp = tmp ** alpha
        ratio = (tmp[:n, n:].sum() + tmp[n:, :n].sum()) / tmp.sum()
    else:
        #dummy_counts = create_dummy_counts_old(counts, lengths=lengths, multiscale_factor=multiscale_factor)
        #rows, cols = get_dis_indices(dummy_counts, n=lengths.sum(), lengths=lengths, ploidy=2, multiscale_factor=1, nbeads=X.shape[0])
        #tmp = (((X[rows] - X[cols]) ** 2).sum(axis=1) ** 0.5).reshape(4, -1)
        rows, cols = constraint_dis_indices(counts, n=lengths.sum(), lengths=lengths, ploidy=2, multiscale_factor=multiscale_factor, nbeads=X.shape[0])
        tmp = (((X[rows] - X[cols]) ** 2).sum(axis=1) ** 0.5)
        if modifications is not None and 'homo_with_alpha' in modifications:
            tmp = tmp ** alpha
        inter_homolog_mask = ((rows >= n) & (cols < n)) | ((rows < n) & (cols >= n))
        ratio = (tmp[inter_homolog_mask].sum()) / tmp.sum()
    return ratio


def get_local_addition_per_ij(i, j, quadrant):
    n = quadrant.shape[0]
    mean_of_neighbors = 0.
    prev_max = 1.
    dist = 1
    while mean_of_neighbors == 0 and dist < n / 2:
        mean_of_neighbors = np.nanmean(quadrant[max(i - dist, 0):min(i + dist + 1, n), max(j - dist, 0):min(j + dist + 1, n)])
        mean_of_neighbors = min(mean_of_neighbors, prev_max)
        prev_max = 1. / (dist * 2 + 1) ** 2
        dist += 1
    return mean_of_neighbors


get_local_addition = np.vectorize(get_local_addition_per_ij, excluded=['quadrant'])


def fill_zeros_with_local(quadrant):
    n = quadrant.shape[0]
    rows0, cols0 = np.where(quadrant == 0)
    local_additions = get_local_addition(i=rows0, j=cols0, quadrant=quadrant)
    quadrant[rows0, cols0] += local_additions
    return quadrant


def fill_zeros_in_array(arr, weighted=False):
    n = arr.shape[0]
    additions = np.zeros_like(arr)
    rows0 = np.where(arr == 0)[0]
    for i in rows0:
        mean_of_neighbors = 0.
        dist = 1
        while mean_of_neighbors == 0 and dist < n / 2:
            mean_of_neighbors = np.nanmean(arr[max(i - dist, 0):min(i + dist + 1, n)])
            if weighted:
                mean_of_neighbors = np.mean([0.] * (dist - 1) + [mean_of_neighbors])
            dist += 1
        additions[i] = mean_of_neighbors
    return arr + additions


def fill_zeros_with_diag_mean(quadrant, fill_all_zeros=True, fraction=0.5):
    n = quadrant.shape[0]
    diag_means = np.array([np.nanmean(np.diagonal(quadrant, offset=i).copy()) for i in range(1 - n, n)])
    if fill_all_zeros:
        diag_means = fill_zeros_in_array(diag_means)
    for i in range(1 - n, n):
        rng = np.arange(n - np.abs(i)) + max(0, -i)
        diag_vals = quadrant[rng, rng + i]
        diag_vals[diag_vals == 0] = diag_means[n + i - 1] * fraction #* perc_non0 #[n + i - 1]
        quadrant[rng, rng + i] = diag_vals
    rows0, cols0 = np.where(quadrant == 0)
    return quadrant


def inter_homolog_ratio_via_ua_counts(counts, lengths, alpha=-3., beta=1., multiscale_factor=1, modifications=None):
    from .utils import ambiguate_counts
    if alpha is None:
        alpha = -3.
    if beta is None:
        beta = 1.
    # Find UA counts
    if not isinstance(counts, list):
        counts = [counts]
    #ua_index = [i for i in range(len(counts)) if counts[i].shape == (int(lengths.sum() * 2), int(lengths.sum() * 2))]
    ua_index = [i for i in range(len(counts)) if counts[i].name == 'ua']
    if len(ua_index) > 1:
        raise ValueError("Shouldn't have multiple unambig count matrices here. Pool them, recompute biases, and continue.")
    if len(ua_index) != 1:
        raise ValueError("Can't calculate inter-homolog constraint from unambiguous counts, unambiguous counts not available.")
    ua_counts = counts[ua_index[0]]
    ua_counts = ua_counts.copy()
    if not isinstance(ua_counts, np.ndarray):
        ua_counts = ua_counts.toarray()
    ua_counts = np.triu(ua_counts, 1)
    ua_counts = ua_counts + ua_counts.T
    ua_counts[np.isnan(ua_counts)] = 0
    n = int(ua_counts.shape[0] / 2)
    # Ignore bins for which we have no data, these will not be part of constraint component of objective & should be ignored
    all_counts = ambiguate_counts(counts, n)
    if not isinstance(all_counts, np.ndarray):
        all_counts = all_counts.toarray()
    all_counts = all_counts + all_counts.T
    ua_counts = ua_counts.astype(float)
    ua_counts[:n, n:][all_counts == 0] = np.nan
    ua_counts[n:, :n][all_counts == 0] = np.nan
    ua_counts[:n, :n][all_counts == 0] = np.nan
    ua_counts[n:, n:][all_counts == 0] = np.nan
    if not (modifications is not None and 'homo_with_alpha' in modifications):
        # Divide by beta
        try:
            beta = counts[ua_index[0]].beta
        except:
            if isinstance(beta, list):
                if len(beta) == 1:
                    beta = beta[0]
                else:
                    beta = beta[ua_index[0]]
        if beta is not None:
            ua_counts /= beta
        # Diagonal is zeroed out, should also be ignored
        np.fill_diagonal(ua_counts, np.nan)
        # Fill in other 0 values
        if (ua_counts[:n, n:] == 0).sum() != 0:
            ua_counts[:n, n:] = fill_zeros_with_local(ua_counts[:n, n:])
        if (ua_counts[n:, :n] == 0).sum() != 0:
            ua_counts[n:, :n] = fill_zeros_with_local(ua_counts[n:, :n])
        if (ua_counts[:n, :n] == 0).sum() != 0:
            ua_counts[:n, :n] = fill_zeros_with_local(ua_counts[:n, :n])
        if (ua_counts[n:, n:] == 0).sum() != 0:
            ua_counts[n:, n:] = fill_zeros_with_local(ua_counts[n:, n:])
        # Previously nan values are set to inf so that they become 0 when raised to 1/alpha
        ua_counts[np.isnan(ua_counts)] = np.inf
        ua_counts = ua_counts ** (1 / alpha)
    ratio = (np.nansum(ua_counts[:n, n:]) + np.nansum(ua_counts[n:, :n])) / np.nansum(ua_counts)
    return ratio


def inter_homolog_ratio_via_ua_counts_OLD(counts, lengths, alpha=-3., beta=1., multiscale_factor=1, modifications=None):
    from topsy.inference.utils import find_beads_to_remove
    #from scipy import sparse

    if alpha is None:
        alpha = -3.

    '''tmp = poisson_diploid.format_counts(counts).data
    if not define_constraint_with_alpha:
        tmp = tmp ** (1 / alpha)
    ratio = (tmp[1].sum() + tmp[2].sum()) / tmp.sum()
    return ratio'''

    #if sparse.issparse(counts):
    counts = np.triu(counts.copy().toarray(), 1).astype(float)
    counts = counts + counts.T
    counts[np.isnan(counts)] = 0

    n = int(counts.shape[0] / 2)

    if not (modifications is not None and 'homo_with_alpha' in modifications):
        # If rows/cols are entirely 0, those beads will not be part of objective and should be ignored
        torm = find_beads_to_remove(counts, nbeads=counts.shape[0], threshold=0)
        counts[torm, :] = np.nan
        counts[:, torm] = np.nan

        # Diagonal is zeroed out, should also be ignored
        np.fill_diagonal(counts, np.nan)

        # Fill in other 0 values with the mean of that quadrant
        mean_interhomo = np.nanmean(counts[:n, n:] + counts[n:, :n]) / 2.
        counts[:n, n:][counts[:n, n:] == 0] = mean_interhomo / 10.
        counts[n:, :n][counts[n:, :n] == 0] = mean_interhomo / 10.
        mean_intrahomo = np.nanmean(counts[:n, :n] + counts[n:, n:]) / 2.
        counts[:n, :n][counts[:n, :n] == 0] = mean_intrahomo / 10.
        counts[n:, n:][counts[n:, n:] == 0] = mean_intrahomo / 10.

        # Previously nan values are set to inf so that they become 0 when raised to 1/alpha
        counts[np.isnan(counts)] = np.inf

        counts = counts ** (1 / alpha)
    ratio = (counts[:n, n:].sum() + counts[n:, :n].sum()) / counts.sum()

    return ratio


def inter_homolog_ratio_via_ambig_counts(counts, lengths, alpha=-3., beta=1., multiscale_factor=1, modifications=None):
    if lengths.shape[0] == 1:
        raise ValueError("Can't estimate inter-homolog ratio using inter-chromosome ratio, only 1 chromosome available.")

    raise NotImplementedError('inter_homolog_ratio_via_ambig_counts not implemented')


# =============================================================================================================================================================================
# =============================================              INTER-HOMOLOG DISTANCE CONSTRAINT                =================================================================
# =============================================================================================================================================================================


def get_inter_homolog_dis(input_dis, lengths, init_structure=None, true_structure=None, counts=None, alpha=-3., beta=1., verbose=True, HSC_lowres_beads=5, modifications=None):
    if input_dis is None:
        return None
    try:
        output = float(input_dis)
    except ValueError:
        if input_dis.lower() == 'true' or input_dis.lower() == 'x_true' or input_dis.lower() == 'auto':
            if true_structure is None:
                raise ValueError('Must input X_true to calculate inter-homolog ratio from X_true')
            output = inter_homolog_dis(true_structure, counts=counts, lengths=lengths)
        elif init_structure is not None:
            output = inter_homolog_dis(init_structure, counts=counts, lengths=lengths)
        else:
            if counts is None:
                raise ValueError('Must input counts to calculate inter-homolog ratio from counts')
            #ua_index = [i for i in range(len(counts)) if counts[i].shape == (int(lengths.sum() * 2), int(lengths.sum() * 2))]
            ua_index = [i for i in range(len(counts)) if counts[i].name == 'ua']
            if input_dis.lower() == 'ua' or input_dis.lower() == 'unambig' or (input_dis.lower() == 'counts' and len(ua_index) > 0):
                output = inter_homolog_dis_via_ua_counts_try1(counts[ua_index[0]], lengths, alpha=alpha, beta=beta, HSC_lowres_beads=HSC_lowres_beads, modifications=modifications)
            elif input_dis.lower() == 'a' or input_dis.lower() == 'ambig' or input_dis.lower() == 'pa':
                output = inter_homolog_dis_via_ambig_counts(counts, lengths, alpha=alpha, beta=beta, modifications=modifications)
            else:
                raise ValueError('Error calculating inter-homolog ratio, input "%s" not recognized' % input_dis)
    if verbose:
        print('Estimated distance between homolog barycenters is %.2g' % output, flush=True)
    return output


def inter_homolog_dis(X, lengths=None, counts=None):

    X = X.reshape(-1, 3)
    n = int(X.shape[0] / 2)

    return ((np.nanmean(X[:n, :], axis=0) - np.nanmean(X[n:, :], axis=0)) ** 2).sum() ** 0.5


def inter_homolog_dis_via_ua_counts_try1(counts, lengths, alpha=-3., beta=1, random_state=None, init='mds', as_sparse=False, in_2d=False, input_weights=None, homo_init=None,
                                         HSC_lowres_beads=5, modifications=None, verbose=True):
    from topsy.inference.poisson_diploid import PM1
    from topsy.inference.diploid_algorithms import choose_lowres_genome_factor

    if random_state is None:
        random_state = np.random.RandomState(seed=0)
    if HSC_lowres_beads is None:
        lowres_genome_factor = 1
    else:
        #lowres_genome_factor = int(round(lengths.sum() / HSC_lowres_beads))
        lowres_genome_factor = choose_lowres_genome_factor(lengths, HSC_lowres_beads)
    if alpha is None:
        alpha = -3.
    if beta is None:
        beta = 1.
    if not isinstance(counts, list):
        counts = [counts]
    counts = [c.copy().astype(float) for c in counts]
    # Infer
    converged = False
    i = 0
    while not converged:
        print('**** HSC_lowres_beads %d, lowres_genome_factor %d' % (HSC_lowres_beads, lowres_genome_factor), flush=True)
        if verbose and i != 0:
            print('Finding inter-homolog distance, try %d' % i, flush=True)
        pm1 = PM1(alpha=alpha, beta=beta, max_iter=50000, random_state=random_state, modifications=modifications,
                  init=init, verbose=0, ploidy=2, in_2d=in_2d, X_true=None, as_sparse=as_sparse,
                  input_weight=input_weights, lagrange_mult={'homodis': 0.}, constraints={'homodis': homo_init}, homo_init='homodis', multiscale_factor=lowres_genome_factor)
        pm1.prep_counts(counts, lengths, normalize=False, filter_counts=False)
        pm1.parse_homolog_sep()
        pm1.initialize()
        X = pm1.fit(callback_frequency=1000)
        converged = X is not None and pm1.converged_
        i += 1
    return inter_homolog_dis(X)


def inter_homolog_dis_via_ua_counts_try2(counts, lengths, alpha=-3., beta=1., HSC_lowres_beads=5, modifications=None):
    from topsy.inference.multiscale_optimization import reduce_counts_res
    if HSC_lowres_beads is None:
        multiscale_factor = 1
    else:
        multiscale_factor = int(round(lengths.sum() / HSC_lowres_beads))
    if alpha is None:
        alpha = -3.
    if beta is None:
        beta = 1.
    if not isinstance(counts, list):
        counts = [counts]
    # Reduce resolution, optionally
    counts = [reduce_counts_res(c, multiscale_factor, lengths)[0].toarray() for c in counts]
    lengths = decrease_lengths_res(lengths, multiscale_factor)
    # Get UA counts
    #ua_index = [i for i in range(len(counts)) if counts[i].shape == (int(lengths.sum() * 2), int(lengths.sum() * 2))]
    ua_index = [i for i in range(len(counts)) if counts[i].name == 'ua']
    if len(ua_index) > 1:
        raise ValueError("Shouldn't have multiple unambig count matrices here. Pool them, recompute biases, and continue.")
    if len(ua_index) != 1:
        raise ValueError("Can't calculate inter-homolog constraint from unambiguous counts, unambiguous counts not available.")
    ua_counts = counts[ua_index[0]]
    ua_counts = ua_counts.copy()
    if not isinstance(ua_counts, np.ndarray):
        ua_counts = ua_counts.toarray()
    ua_counts = np.triu(ua_counts, 1)
    ua_counts = ua_counts + ua_counts.T
    ua_counts[np.isnan(ua_counts)] = 0
    n = int(ua_counts.shape[0] / 2)
    # Focus on inter-homolog counts
    inter_homo = ua_counts[:n, n:]
    # Ignore bins for which we have no data, these will not be part of constraint component of objective & should be ignored
    #all_counts = ambiguate_counts(counts, n).toarray()
    #all_counts = all_counts + all_counts.T
    #inter_homo[all_counts == 0] = np.nan
    # Divide by beta
    try:
        beta = counts[ua_index[0]].beta
    except:
        if isinstance(beta, list):
            if len(beta) == 1:
                beta = beta[0]
            else:
                beta = beta[ua_index[0]]
    inter_homo /= (beta * multiscale_factor ** 2)
    # Fill in other 0 values
    if (inter_homo == 0).sum() != 0:
        inter_homo = fill_zeros_with_local(inter_homo)
    # Previously nan values are set to inf so that they become 0 when raised to 1/alpha
    inter_homo[np.isnan(inter_homo)] = np.inf
    inter_homo = inter_homo ** (1 / alpha)
    return inter_homo


def inter_homolog_dis_via_ua_counts(counts, lengths, alpha=-3., beta=1., modifications=None):
    #ua_index = [i for i in range(len(counts)) if counts[i].shape == (int(lengths.sum() * 2), int(lengths.sum() * 2))]
    if not isinstance(counts, list):
        counts = [counts]
    ua_index = [i for i in range(len(counts)) if counts[i].name == 'ua']
    if len(ua_index) > 1:
        raise ValueError("Shouldn't have multiple unambig count matrices here. Pool them, recompute biases, and continue.")
    if len(ua_index) != 1:
        raise ValueError("Can't calculate inter-homolog constraint from unambiguous counts, unambiguous counts not available.")
    ua_counts = counts[ua_index[0]]

    if alpha is None:
        alpha = -3.
    if beta is None:
        beta = 1.

    ua_counts = ua_counts.copy()
    if not isinstance(ua_counts, np.ndarray):
        ua_counts = np.triu(ua_counts.toarray(), 1)
    ua_counts[np.isnan(ua_counts)] = 0

    n = int(ua_counts.shape[0] / 2)

    interhomo = ua_counts[:n, n:].mean()
    interhomo /= beta

    interhomo **= 1 / alpha

    return interhomo


def inter_homolog_dis_via_ambig_counts(counts, lengths, alpha=-3., beta=1., modifications=None):
    if lengths.shape[0] == 1:
        raise ValueError("Can't estimate inter-homolog ratio using inter-chromosome ratio, only 1 chromosome available.")

    raise NotImplementedError("Can't get there from here")
