import numpy as np
from sklearn.utils import check_random_state
from .multiscale_optimization import increase_struct_res, decrease_lengths_res, decrease_struct_res
import os
from .mds import estimate_X
from .utils_poisson import find_beads_to_remove
from .utils_poisson import _struct_replace_nan, _format_structures


def _initialize_struct_mds(counts, lengths, ploidy, alpha, bias, random_state,
                           multiscale_factor=1, verbose=True):
    """Initialize structure via multi-dimensional scaling of unambig counts.
    """

    if verbose:
        print('INITIALIZATION: multi-dimensional scaling', flush=True)

    random_state = check_random_state(random_state)

    ua_index = [i for i in range(len(counts)) if counts[i].name == 'ua']
    if len(ua_index) == 1:
        ua_index = ua_index[0]
    elif len(ua_index) != 0:
        raise ValueError(
            "Multiple unambiguous counts matrices detected."
            " Pool data from unambiguous counts matrices before inference.")
    else:
        raise ValueError("Unambiguous counts needed to initialize via MDS.")

    ua_counts = counts[ua_index]
    ua_beta = ua_counts.beta
    if ua_beta is not None:
        ua_beta *= multiscale_factor ** 2

    struct = estimate_X(
        ua_counts._counts.astype(float),
        alpha=-3. if alpha is None else alpha, beta=ua_beta, verbose=False,
        use_zero_entries=False, precompute_distances='auto',
        bias=(np.tile(bias, ploidy) if bias is not None else bias),
        random_state=random_state, type="MDS2", factr=1e12, maxiter=10000,
        ini=None)

    struct = struct.reshape(-1, 3)
    torm = find_beads_to_remove(counts, struct.shape[0])
    struct[torm] = np.nan

    return struct


def _initialize_struct(counts, lengths, ploidy, alpha, bias, random_state,
                       init='mds', multiscale_factor=1, mixture_coefs=None,
                       verbose=True):
    """Initialize structure, randomly or via MDS of unambig counts.
    """

    random_state = check_random_state(random_state)

    if mixture_coefs is None:
        mixture_coefs = [1.]

    if not isinstance(counts, list):
        counts = [counts]
    ua_index = [i for i in range(len(counts)) if counts[i].name == 'ua']

    lengths_lowres = decrease_lengths_res(
        lengths, multiscale_factor=multiscale_factor)

    if isinstance(init, np.ndarray) or isinstance(init, list):
        if verbose:
            print('INITIALIZATION: 3D structure', flush=True)
        structures = _format_structures(init, mixture_coefs=mixture_coefs)
    elif isinstance(init, str) and (init.lower() in ("mds", "mds2")) and len(ua_index) != 0:
        struct = _initialize_struct_mds(
            counts=counts, lengths=lengths_lowres, ploidy=ploidy, alpha=alpha,
            bias=bias, random_state=random_state,
            multiscale_factor=multiscale_factor, verbose=verbose)
        structures = [struct] * len(mixture_coefs)
    elif isinstance(init, str) and (init.lower() in ("random", "rand", "mds", "mds2")):
        if verbose:
            print('INITIALIZATION: random points', flush=True)
        structures = [(1 - 2 * random_state.rand(int(
            lengths_lowres.sum() * ploidy * 3))).reshape(-1, 3) for coef in mixture_coefs]
    elif isinstance(init, str) and os.path.exists(init):
        if verbose:
            print('INITIALIZATION: 3D structure, %s' % init, flush=True)
        structures = _format_structures(
            np.loadtxt(init), mixture_coefs=mixture_coefs)
    else:
        raise ValueError("Initialization method not understood.")

    struct_length = set([s.shape[0] for s in structures])
    if len(struct_length) > 1:
        raise ValueError("Initial structures are of different shapes")
    else:
        struct_length = struct_length.pop()
    multiscale_factor_init = int(np.ceil(
        lengths.sum() * ploidy / struct_length))
    lengths_init = decrease_lengths_res(
        lengths, multiscale_factor=multiscale_factor_init)

    structures = _format_structures(
        structures, lengths=lengths_init, ploidy=ploidy,
        mixture_coefs=mixture_coefs)
    structures = [_struct_replace_nan(
        struct, lengths_init,
        random_state=random_state) for struct in structures]

    for i in range(len(structures)):
        if struct_length < int(lengths_lowres.sum() * ploidy):
            resize_factor = int(np.ceil(
                lengths_lowres.sum() * ploidy / struct_length))
            if verbose:
                print('INITIALIZATION: increasing resolution of structure by'
                      ' %d' % resize_factor, flush=True)
            structures[i] = increase_struct_res(
                structures[i], multiscale_factor=resize_factor,
                lengths=lengths_lowres)
        elif struct_length > int(lengths_lowres.sum() * ploidy):
            resize_factor = int(np.ceil(
                struct_length / lengths_lowres.sum() * ploidy))
            if verbose:
                print('INITIALIZATION: decreasing resolution of structure by'
                      ' %d' % resize_factor, flush=True)
            structures[i] = decrease_struct_res(
                structures[i], multiscale_factor=resize_factor,
                lengths=lengths_lowres)

    structures = _format_structures(
        structures, lengths=lengths_lowres, ploidy=ploidy,
        mixture_coefs=mixture_coefs)

    return np.concatenate(structures)


def initialize(counts, lengths, init, ploidy, random_state=None, alpha=-3.,
               bias=None, multiscale_factor=1, reorienter=None,
               mixture_coefs=None, verbose=False):
    """Initialize optimization.

    Create initialization for optimization. Structures can be initialized
    randomly, or via MDS2.

    Parameters
    ----------
    counts : list of pastis.counts.CountsMatrix instances
        Preprocessed counts data.
    lengths : array_like of int
        Number of beads per homolog of each chromosome.
    random_state : int or RandomState instance
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    init : str or array_like of float
        If array of float, this will be used for initialization. Structures
        will be re-sized to the appropriate resolution, and NaN beads will be
        linearly interpolated. If str, indicates the method of initalization:
        random ("random" or "rand") or MDS2 ("mds2" or "mds").
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    alpha : float, optional
        Biophysical parameter of the transfer function used in converting
        counts to wish distances.
    bias : array_like of float, optional
        Biases computed by ICE normalization.
    multiscale_factor : int, optional
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 indicates full resolution.

    Returns
    -------
    array of float
        Initialization for inference.

    """

    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    random_state = check_random_state(random_state)

    lengths_lowres = decrease_lengths_res(
        lengths, multiscale_factor=multiscale_factor)

    if reorienter is not None and reorienter.reorient:
        if isinstance(init, np.ndarray):
            print('INITIALIZATION: inputted translation coordinates'
                  ' and/or rotation quaternions', flush=True)
            init_reorient = init.flatten()
            reorienter.check_X(init_reorient)
        else:
            print('INITIALIZATION: random', flush=True)
            init_reorient = []
            if reorienter.translate:
                init_reorient.append(1 - 2 * random_state.rand(
                    lengths_lowres.shape[0] * 3 * (
                        1 + np.invert(reorienter.fix_homo))))
            if reorienter.rotate:
                init_reorient.append(random_state.rand(
                    lengths_lowres.shape[0] * 4 * (
                        1 + np.invert(reorienter.fix_homo))))
            init_reorient = np.concatenate(init_reorient)
        return init_reorient
    else:
        struct_init = _initialize_struct(
            counts=counts, lengths=lengths, ploidy=ploidy, alpha=alpha,
            bias=bias, random_state=random_state, init=init,
            multiscale_factor=multiscale_factor, mixture_coefs=mixture_coefs,
            verbose=verbose)
        return struct_init
