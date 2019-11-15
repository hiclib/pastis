import numpy as np
from sklearn.utils import check_random_state
from .multiscale_optimization import increase_struct_res, decrease_lengths_res, decrease_struct_res


def initialize_struct_mds(counts, lengths, ploidy, alpha, bias, random_state,
                          multiscale_factor=1, verbose=True):
    """Initialize structure via multi-dimensional scaling of unambig counts.
    """

    from .utils import find_beads_to_remove
    from .mds import estimate_X

    if verbose:
        print('INITIALIZATION: multi-dimensional scaling', flush=True)

    random_state = check_random_state(random_state)

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

    struct = estimate_X(
        ua_counts.counts.tocoo().astype(float),
        alpha=-3. if alpha is None else alpha, beta=ua_beta, verbose=False,
        use_zero_entries=False, precompute_distances='auto',
        bias=(np.tile(bias, ploidy) if bias is not None else bias),
        random_state=random_state, type="MDS2", factr=1e12, maxiter=10000,
        ini=None)

    struct = struct.reshape(-1, 3)
    torm = find_beads_to_remove(counts, struct.shape[0])
    struct[torm] = np.nan

    return struct


def initialize_struct(counts, lengths, ploidy, alpha, bias, random_state,
                      init='mds', multiscale_factor=1, mixture_coefs=None,
                      verbose=True):
    """Initialize structure, randomly or via MDS of unambig counts.
    """

    from .utils import struct_replace_nan, format_structures

    random_state = check_random_state(random_state)

    if mixture_coefs is None:
        mixture_coefs = [1.]

    if not isinstance(counts, list):
        counts = [counts]
    ua_index = [i for i in range(len(counts)) if counts[i].name == 'ua']

    if isinstance(init, np.ndarray) or isinstance(init, list):
        if verbose:
            print('INITIALIZATION: 3D structure', flush=True)
        structures = format_structures(init, lengths=lengths, ploidy=ploidy,
                                       mixture_coefs=mixture_coefs)
    elif isinstance(init, str) and (init.lower() in ("mds", 'mds2')) and len(ua_index) != 0:
        struct = initialize_struct_mds(
            counts=counts, lengths=lengths, ploidy=ploidy, alpha=alpha,
            bias=bias, random_state=random_state,
            multiscale_factor=multiscale_factor, verbose=verbose)
        structures = [struct] * len(mixture_coefs)
    elif isinstance(init, str) and (init.lower() in ("random", "rand")):
        if verbose:
            print('INITIALIZATION: random points', flush=True)
        structures = [(1 - 2 * random_state.rand(int(lengths.sum() * ploidy * 3))).reshape(-1, 3) for coef in mixture_coefs]
    else:
        raise ValueError("Initialization method not understood.")

    structures = [struct_replace_nan(struct, lengths, random_state=random_state) for struct in structures]

    for struct in structures:
        if struct.shape[0] < int(lengths.sum() * ploidy):
            if verbose:
                print('INITIALIZATION: increasing resolution of structure by'
                      '%g' % np.ceil(lengths.sum() * ploidy / struct.shape[0]),
                      flush=True)
            struct = increase_struct_res(struct, multiscale_factor=np.ceil(
                lengths.sum() * ploidy / struct.shape[0]), lengths=lengths)
        elif struct.shape[0] > int(lengths.sum() * ploidy):
            if verbose:
                print('INITIALIZATION: decreasing resolution of structure by'
                      '%g' % np.ceil(lengths.sum() * ploidy / struct.shape[0]),
                      flush=True)
            struct = decrease_struct_res(struct, multiscale_factor=np.ceil(
                lengths.sum() * ploidy / struct.shape[0]), lengths=lengths)

    return np.concatenate(structures)


def initialize(counts, lengths, random_state, init, ploidy, alpha=-3.,
               bias=None, multiscale_factor=1, reorienter=None,
               mixture_coefs=None, verbose=False):
    """Initialize optimization, for structure or for rotation and translation.
    """

    from sklearn.utils import check_random_state

    lengths_lowres = decrease_lengths_res(lengths, multiscale_factor=multiscale_factor)

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
                    lengths_lowres.shape[0] * 3 * (1 + np.invert(reorienter.fix_homo))))
            if reorienter.rotate:
                init_reorient.append(random_state.rand(
                    lengths_lowres.shape[0] * 4 * (1 + np.invert(reorienter.fix_homo))))
            init_reorient = np.concatenate(init_reorient)
        return init_reorient
    else:
        struct_init = initialize_struct(
            counts=counts, lengths=lengths_lowres, ploidy=ploidy, alpha=alpha,
            bias=bias, random_state=random_state, init=init,
            multiscale_factor=multiscale_factor, mixture_coefs=mixture_coefs,
            verbose=verbose)
        return struct_init
