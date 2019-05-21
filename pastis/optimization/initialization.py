import numpy as np
from sklearn.utils import check_random_state
from topsy.inference import mds
from .multiscale_optimization import increase_X_res, decrease_lengths_res
from ..datasets.samples_generator import make_3d
from scipy import sparse
from sklearn.metrics import euclidean_distances
from .utils import create_dummy_counts


def initialize_struct_randomly(lengths, ploidy, init='randpts', random_state=None, in_2d=False, verbose=1):
    nbeads = int(lengths.sum() * ploidy)
    if init == 'randwalk':
        if verbose:
            print('INITIALIZATION: random walk with clash avoidance', flush=True)
        X = make_3d(lengths, ploidy, random_state, in_2d=in_2d)
    else:
        if verbose:
            print('INITIALIZATION: random points', flush=True)
        X = 1 - 2 * random_state.rand(nbeads * 3)
        if in_2d:
            X = X.reshape(-1, 3)
            X[:, 2] = 0.
            X = X.flatten()
    return X


def sep_homo_in_rand_init_struct(lengths, ploidy, init='randpts', random_state=None, homo_init=None, homo_init_style=None, counts=None, alpha=None, multiscale_factor=1, in_2d=False, verbose=1, modifications=None):
    from .poisson_diploid import estimate_null, translate_and_rotate
    from .separating_homologs import inter_homolog_ratio

    # Initialize randomly & separate the homologs
    if homo_init is not None and ploidy > 1:
        if homo_init_style is None:
            homo_init_style = 'homo'
        homo_init_style = homo_init_style.lower()

        if homo_init_style == 'homo':
            if verbose:
                print('INITIALIZATION: separating homologs by ratio of %.4g' % homo_init, flush=True)
            dummy_counts = create_dummy_counts(counts, lengths, multiscale_factor)
            iteration = 0
            success = False
            while not success and iteration < 5:
                X_rand = initialize_struct_randomly(lengths, ploidy, init=init, random_state=random_state, in_2d=in_2d, verbose=verbose)
                best_translations, converged = estimate_null(dummy_counts, lengths, init_X=np.zeros(lengths.shape[0] * 3 * ploidy), alpha=alpha, max_iter=10000000000,
                                                             verbose=False, ploidy=ploidy, in_2d=in_2d, lagrange_mult={homo_init_style: 1.}, constraints={homo_init_style: homo_init}, multiscale_factor=1,
                                                             init_structures=X_rand, translate=True, rotate=False, fix_homo=False, modifications=modifications)
                X = translate_and_rotate(best_translations, lengths, init_structures=X_rand, translate=True, rotate=False, fix_homo=False)[0]
                ratio = inter_homolog_ratio(X, counts=dummy_counts, alpha=alpha, modifications=modifications)
                if converged and np.allclose(homo_init, ratio, atol=0.001, rtol=.1):
                    success = True
                iteration += 1
            if not success:
                if not converged:
                    print('Error in separating homologs of initialized structure: does not converge', flush=True)
                else:
                    print('Error in separating homologs of initialized structure by ratio of %.4g:  separated homologs by %.4g  (translation distance: %.4g)' % (homo_init, ratio, euclidean_distances(best_translations.reshape(-1, 3))[0, 1]), flush=True)
                X = None
            else:
                X = X.flatten()
                if verbose:
                    print('INITIALIZATION: succesfully separated homologs by ratio of %.4g (separated homologs by %.4g, translation distance: %.4g)' % (homo_init, ratio, euclidean_distances(best_translations.reshape(-1, 3))[0, 1]), flush=True)
        elif homo_init_style == 'homodis':
            if verbose:
                print('INITIALIZATION: separating homologs by distance of %.4g' % homo_init, flush=True)
            X_rand = initialize_struct_randomly(lengths, ploidy, init=init, random_state=random_state, in_2d=in_2d, verbose=verbose)
            translations = np.concatenate([np.repeat(np.array([[0., 0., 0.]]), repeats=lengths.shape[0], axis=0), np.repeat(np.array([[homo_init, 0., 0.]]), repeats=lengths.shape[0], axis=0)])
            X = translate_and_rotate(translations, lengths, init_structures=X_rand, translate=True, rotate=False, fix_homo=False)[0].flatten()
        else:
            raise ValueError('Homo init style (%s) not recognized' % homo_init_style)

    else:
        # Initialize randomly & do NOT separate homologs
        X = initialize_struct_randomly(lengths, ploidy, init=init, random_state=random_state, in_2d=in_2d, verbose=verbose).flatten()
    return X


def initialize_struct(lengths, ploidy, init='mds', random_state=None, homo_init=None, homo_init_style=None, counts=None, bias=None, alpha=None, betas=None, multiscale_factor=1, mask_fullresX=None, in_2d=False, verbose=1, modifications=None):
    from .separating_homologs import inter_homolog_ratio, inter_homolog_dis
    from ..metrics.utils import X_replace_nan, mask_X

    random_state = check_random_state(random_state)

    # Make stuff a list if it isn't already
    if not isinstance(counts, list):
        counts = [counts]
    #if not isinstance(betas, list):
    #    betas = [betas]

    # Define nbeads, number of beads
    if lengths is None:
        raise ValueError('Must input lengths to intitialize_X()')
    nbeads = int(lengths.sum() * ploidy)

    if isinstance(init, np.ndarray):
        if verbose:
            print('INITIALIZATION: 3D structure', flush=True)
        X = init.reshape(-1, 3)
        if len(X) > nbeads:
            raise ValueError('Structure used for initialization may not have more beads than inferred structure')
        elif len(X) < nbeads:
            if verbose:
                print('INITIALIZATION: increasing resolution of 3D structure by %g' % np.ceil(nbeads / len(X)), flush=True)
            X = increase_X_res(X, multiscale_factor=np.ceil(nbeads / len(X)), lengths=lengths, mask=mask_fullresX)

    else:
        init = init.lower()

        # For MDS init, see which of the counts are unambiguous
        lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)
        #nbeads_lowres = lengths_lowres.sum() * ploidy
        #ua_index = [i for i in range(len(counts)) if counts[i].shape == (nbeads_lowres, nbeads_lowres)]
        ua_index = [i for i in range(len(counts)) if counts[i].name == 'ua']

        # Initialize
        if (init == "mds" or init == 'mds2') and len(ua_index) != 0:
            if verbose:
                print('INITIALIZATION: multi-dimensional scaling', flush=True)
            if len(ua_index) == 1:
                ua_index = ua_index[0]
            elif len(ua_index) != 0:
                raise ValueError("Multiple unambiguous counts matrices detected. Pool data from unambiguous counts matrices before inference.")
            ua_counts = counts[ua_index]
            #if isinstance(ua_counts, np.ndarray):
            #    ua_counts = ua_counts.copy()
            #    ua_counts[np.isnan(ua_counts)] = 0
            #    ua_counts = sparse.coo_matrix(ua_counts)
            ua_beta = ua_counts.beta
            if ua_beta is not None:
                ua_beta *= multiscale_factor ** 2
            ua_alpha = alpha
            if ua_alpha is None:
                ua_alpha = -3.

            #MDS_init = sep_homo_in_rand_init_struct(lengths, ploidy, random_state=random_state, homo_init=homo_init, homo_init_style=homo_init_style, counts=counts, alpha=alpha, in_2d=False, verbose=False, modifications=modifications)
            MDS_init = None

            #test = ua_counts.counts.copy().tocoo()
            #print(test.shape, test.row.)
            X = mds.estimate_X(ua_counts.counts.copy().tocoo().astype(float), alpha=ua_alpha, beta=ua_beta,
                               verbose=False,
                               use_zero_entries=False,
                               precompute_distances='auto',
                               bias=(np.tile(bias, ploidy) if bias is not None else bias),
                               random_state=random_state, type="MDS2",
                               factr=1e12,
                               maxiter=10000,
                               ini=MDS_init)

            # Remove beads for which there is no ua_counts data, then interpolate NaNs
            X = X_replace_nan(mask_X(X, ua_counts), lengths_lowres)

            if len(X) < nbeads:
                if homo_init_style is None or homo_init_style == 'homo': print(inter_homolog_ratio(X, counts=counts, lengths=lengths_lowres, alpha=alpha, multiscale_factor=1, modifications=modifications))
                if verbose:
                    print('INITIALIZATION: increasing resolution of low-res MDS-initialized structure by %.4g' % round(nbeads / len(X)), flush=True)
                X = increase_X_res(X, multiscale_factor=round(nbeads / len(X)), lengths=lengths)

            if in_2d:
                X[2] = 0
            if ploidy > 1 and (homo_init_style is None or homo_init_style == 'homo'):
                MDS_ratio = inter_homolog_ratio(X, counts=counts, lengths=lengths, alpha=alpha, multiscale_factor=multiscale_factor, modifications=modifications)
                if verbose:
                    print('INITIALIZATION: homologs in MDS structure are separated by ratio of %.4g' % MDS_ratio)
                    #print(inter_homolog_ratio(X, counts=create_dummy_counts(counts, lengths, multiscale_factor), lengths=lengths, alpha=alpha, multiscale_factor=1, modifications=modifications))
            if ploidy > 1 and homo_init_style == "homodis" and verbose:
                print('INITIALIZATION: homologs in MDS structure are separated by distance of %.4g' % inter_homolog_dis(X))
        else:
            # Initialize randomly & separate the homologs
            X = sep_homo_in_rand_init_struct(lengths, ploidy, random_state=random_state, homo_init=homo_init, homo_init_style=homo_init_style, counts=counts, alpha=alpha, multiscale_factor=multiscale_factor, in_2d=in_2d, verbose=verbose, modifications=modifications)
    return X


def initialize_X(counts, lengths, random_state, init, ploidy, alpha=-3., beta=1., bias=None, in_2d=False, homo_init=None, homo_init_style=None, multiscale_factor=1, mask_fullresX=None, init_structures=None, translate=False, rotate=False, fix_homo=True, modifications=None, verbose=False):
    if init_structures is not None or translate or rotate:
        if isinstance(init, np.ndarray):
            print('NITIALIZATION: inputted translation coordinates and/or rotation quaternions', flush=True)
            init_X = init
        else:
            print('NITIALIZATION: random', flush=True)
            init_X = []
            if translate:
                init_X.append(1 - 2 * random_state.rand(lengths.shape[0] * 3 * (1 + np.invert(fix_homo))))
            if rotate:
                init_X.append(random_state.rand(lengths.shape[0] * 4 * (1 + np.invert(fix_homo))))
            init_X = np.concatenate(init_X)
    else:
        init_X = initialize_struct(lengths, ploidy=ploidy, init=init, random_state=random_state, homo_init=homo_init, homo_init_style=homo_init_style, counts=counts, bias=bias,
                                   alpha=alpha, betas=beta, multiscale_factor=multiscale_factor, mask_fullresX=mask_fullresX, in_2d=in_2d, verbose=verbose, modifications=modifications)

        if init_X is not None:
            init_X = init_X.reshape(-1, 3)

    return init_X
