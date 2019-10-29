import numpy as np


def initialize_struct_mds(counts, lengths, ploidy, alpha, bias, random_state,
                          multiscale_factor=1, verbose=True):
    """Initialize structure via multi-dimensional scaling of unambig counts.
    """

    from .utils import find_beads_to_remove
    from .mds import estimate_X

    if verbose:
        print('INITIALIZATION: multi-dimensional scaling', flush=True)

    ua_index = [i for i in range(len(counts)) if counts[i].name == 'ua']
    if len(ua_index) == 1:
        ua_index = ua_index[0]
    elif len(ua_index) != 0:
        raise ValueError(
            "Multiple unambiguous counts matrices detected."
            "Pool data from unambiguous counts matrices before inference.")

    ua_counts = counts[ua_index]
    ua_beta = ua_counts.beta
    if ua_beta is not None:
        ua_beta *= multiscale_factor ** 2

    X = estimate_X(ua_counts.counts.tocoo().astype(float),
                   alpha=-3. if alpha is None else alpha,
                   beta=ua_beta,
                   verbose=False,
                   use_zero_entries=False,
                   precompute_distances='auto',
                   bias=(np.tile(bias, ploidy) if bias is not None else bias),
                   random_state=random_state, type="MDS2",
                   factr=1e12,
                   maxiter=10000,
                   ini=None)

    X = X.reshape(-1, 3)
    torm = find_beads_to_remove(counts, X.shape[0])
    X[torm] = np.nan

    return X


def initialize_structure(counts, lengths, ploidy, alpha, bias, random_state,
                         init='mds', multiscale_factor=1, verbose=True):
    """Initialize structure, randomly or via MDS of unambig counts.
    """

    from .utils import struct_replace_nan
    from sklearn.utils import check_random_state

    random_state = check_random_state(random_state)

    if not isinstance(counts, list):
        counts = [counts]
    ua_index = [i for i in range(len(counts)) if counts[i].name == 'ua']

    if isinstance(init, np.ndarray):
        if verbose:
            print('INITIALIZATION: 3D structure', flush=True)
        X = init.reshape(-1, 3)
        if X.shape[0] > lengths.sum() * ploidy:
            raise ValueError('Structure used for initialization may not have'
                             'more beads than inferred structure.'
                             'Structure used for initialization is of length'
                             '%d, final inferred structure is of length'
                             '%d' % (X.shape[0], lengths.sum() * ploidy))
    elif (init.lower() in ("mds", 'mds2')) and len(ua_index) != 0:
        X = initialize_struct_mds(counts, lengths, ploidy, alpha, bias,
                                  random_state,
                                  multiscale_factor=multiscale_factor,
                                  verbose=verbose)
    else:
        if verbose:
            print('INITIALIZATION: random points', flush=True)
        X = (1 - 2 * random_state.rand(
            int(lengths.sum() * ploidy * 3))).reshape(-1, 3)

    X = struct_replace_nan(X, lengths, random_state=random_state)

    return X


def initialize(counts, lengths, random_state, init, ploidy, alpha=-3.,
               bias=None, multiscale_factor=1, reorienter=None,
               modifications=None, verbose=False):
    """Initialize optimization, for structure or for rotation and translation.
    """

    from sklearn.utils import check_random_state
    from .multiscale_optimization import increase_X_res

    if reorienter.reorient:
        if isinstance(init, np.ndarray):
            print(
                'INITIALIZATION: inputted translation coordinates'
                'and/or rotation quaternions', flush=True)
            init_reorient = init
        else:
            print('INITIALIZATION: random', flush=True)
            random_state = check_random_state(random_state)
            init_reorient = []
            if reorienter.translate:
                init_reorient.append(1 - 2 * random_state.rand(
                    lengths.shape[0] * 3 * (1 + np.invert(reorienter.fix_homo))))
            if reorienter.rotate:
                init_reorient.append(random_state.rand(
                    lengths.shape[0] * 4 * (1 + np.invert(reorienter.fix_homo))))
            init_reorient = np.concatenate(init_reorient)
        return init_reorient
    else:
        init_struct = initialize_structure(counts, lengths,
                                           ploidy, alpha, bias, random_state,
                                           init=init,
                                           multiscale_factor=multiscale_factor,
                                           verbose=verbose)

        if init_struct.shape[0] < int(lengths.sum() * ploidy):
            if verbose:
                print('INITIALIZATION: increasing resolution of structure by'
                      '%g' % np.ceil(lengths.sum() * ploidy / init_struct.shape[0]),
                      flush=True)
            init_struct = increase_X_res(init_struct, multiscale_factor=np.ceil(
                lengths.sum() * ploidy / init_struct.shape[0]), lengths=lengths)

        return init_struct
