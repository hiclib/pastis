import numpy as np
from scipy import optimize
from sklearn.utils import check_random_state
import warnings
from topsy.inference.quaternion import quat_to_rotation_matrix

import autograd.numpy as ag_np
from autograd.builtins import SequenceBox
from autograd import grad
#import jax.numpy as ag_np
#from jax import grad

from .multiscale_optimization import reduce_X_res, decrease_lengths_res, repeat_X_multiscale
from ..metrics.generate_metrics import simulated_vs_inferred
from ..metrics.utils import mask_X
from .utils import get_dis_indices, constraint_dis_indices, create_dummy_counts, get_data
from .prep_counts import check_counts, format_counts


n_iter = iter_obj_full = iter_details = iter_grad = None
# Set up for callback function
def callback_setup():
    global n_iter
    global iter_obj_full
    global iter_details
    global iter_grad
    iter_grad = np.zeros((1))
    n_iter = 0
    iter_obj_full = {k: np.nan for k in ('obj', 'obj_main', 'obj_adj', 'obj_homo', 'obj_homodis', 'obj_X', 'obj_ambig', 'obj_ua', 'obj_pa', 'obj_ambig0', 'obj_ua0', 'obj_pa0')}
    iter_details = {'iter': [], 'obj': [], 'obj_main': [], 'obj_ambig': [], 'obj_ua': [], 'obj_pa': [],
                    'unscaled_rmsd_interchr': [], 'unscaled_disterror_interchr': [], 'unscaled_rmsd_intra': [], 'unscaled_disterror_intra': [], 'unscaled_rmsd_interhomo': [], 'unscaled_disterror_interhomo': [],
                    'rmsd_interchr': [], 'disterror_interchr': [], 'rmsd_intra': [], 'disterror_intra': [], 'rmsd_interhomo': [], 'disterror_interhomo': [],
                    'unscaled_gdt_ts': [], 'gdt_ts': [], 'unscaled_rmsd_btwn_homo': [], 'unscaled_disterror_btwn_homo': [], 'rmsd_btwn_homo': [], 'disterror_btwn_homo': [],
                    'unscaled_rmsd_reorient': [], 'unscaled_disterror_reorient': [], 'rmsd_reorient': [], 'disterror_reorient': [],
                    'inter_homo_ratio_chr': [], 'neighbor_dist_var': [], 'neighbor_dist_mean': [],
                    'obj_adj': [], 'obj_homo': [], 'obj_homodis': [], 'obj_X': [], 'alpha': []}


def get_constraints(structures, counts, alpha=-3., lengths=None, lagrange_mult=None, constraints=None, multiscale_factor=1, mask_fullres0=None, mixture_coefs=None, modifications=None):
    if lagrange_mult is None or len(lagrange_mult) == 0 or sum(lagrange_mult.values()) == 0:
        return 0.
    else:
        if mixture_coefs is not None and len(structures) != len(mixture_coefs):
            raise ValueError(
                "The number of structures (%d) and of mixture coefficents (%d) "
                "should be identical." % (len(structures), len(mixture_coefs)))
        elif mixture_coefs is None:
            mixture_coefs = [1]

        constraints_dict = {k: ag_np.float64(0.) for k, v in lagrange_mult.items() if v != 0}

        lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)
        mask = (None if mask_fullres0 == [None] * len(counts) else sum(mask_fullres0).astype(bool))

        if not (modifications is not None and 'fullres_constraints' in modifications):
            if not (modifications is not None and 'multiscale3' in modifications):
                structures = [reduce_X_res(X, multiscale_factor, lengths, mask=mask).reshape(-1, 3) for X in structures]
            lengths = lengths_lowres
            multiscale_factor = 1
        nbeads_per_homo = lengths.sum()
        ploidy = int(structures[0].shape[0] / lengths.sum())
        rows, cols = constraint_dis_indices(counts, n=lengths_lowres.sum(), lengths=lengths, ploidy=ploidy, multiscale_factor=multiscale_factor, nbeads=structures[0].shape[0], mask=mask)

        for X, gamma in zip(structures, mixture_coefs):
            if lagrange_mult['adj']:
                if constraints['adj'] == 'var':
                    # Calculating distances for neighbors, which are on the off diagonal line - i & j where j = i + 1
                    rows_adj = ag_np.unique(rows)
                    rows_adj = rows_adj[ag_np.isin(rows_adj + 1, cols)]
                    # Remove if "neighbor" beads are actually on different chromosomes or homologs
                    rows_adj = rows_adj[ag_np.digitize(rows_adj, np.tile(lengths, ploidy).cumsum()) == ag_np.digitize(rows_adj + 1, np.tile(lengths, ploidy).cumsum())]
                    cols_adj = rows_adj + 1

                    neighbor_dis = ((X[rows_adj] - X[cols_adj]) ** 2).sum(axis=1) ** 0.5
                    n_edges = neighbor_dis.shape[0]
                    constraints_dict['adj'] += gamma * lagrange_mult['adj'] * (n_edges * (neighbor_dis ** 2).sum() / (neighbor_dis.sum() ** 2) - 1.)
            if lagrange_mult['homo']:
                tmp = (((X[rows] - X[cols]) ** 2).sum(axis=1) ** 0.5)
                inter_homolog_mask = ((rows >= nbeads_per_homo) & (cols < nbeads_per_homo)) | ((rows < nbeads_per_homo) & (cols >= nbeads_per_homo))
                if modifications is not None and 'homo_with_alpha' in modifications:
                    tmp = tmp ** alpha
                #constraints_dict['homo'] += gamma * lagrange_mult['homo'] * ((tmp[1].sum() + tmp[2].sum()) / tmp.sum() - float(constraints['homo'])) ** 2
                constraints_dict['homo'] += gamma * lagrange_mult['homo'] * (tmp[inter_homolog_mask].sum() / tmp.sum() - float(constraints['homo'])) ** 2
            if lagrange_mult['homodis']:
                homo_dis = ((X[:nbeads_per_homo, :].mean(axis=0) - X[nbeads_per_homo:, :].mean(axis=0)) ** 2).sum() ** 0.5
                if not (modifications is not None and 'homodis_gte' in modifications) or homo_dis < float(constraints['homodis']):
                    constraints_dict['homodis'] += gamma * lagrange_mult['homodis'] * (homo_dis - float(constraints['homodis'])) ** 2
            if lagrange_mult['X']:
                X_myres = reduce_X_res(X, factor=X.shape[0] / constraints['X'].shape[0], lengths_prev=lengths, mask=mask_fullres0).reshape(-1, 3)
                constraints_dict['X'] += gamma * lagrange_mult['X'] * np.sqrt(np.nanmean((constraints['X'] - X_myres) ** 2)) / (constraints['X'].max() - constraints['X'].min())

        # If saving details of each iteration, save constraints component
        if len(constraints_dict) > 0 and isinstance(list(constraints_dict.values())[0], np.float64):
            global iter_obj_full
            if iter_obj_full is not None:
                for k, v in constraints_dict.items():
                    iter_obj_full['obj_' + k] = v

        # Sum constraints
        obj_constraints = 0.
        for k, v in constraints_dict.items():
            if ag_np.isnan(v):
                raise ValueError("Constraint %s is nan" % k)
            elif ag_np.isinf(v):
                raise ValueError("Constraint %s is infinite" % k)
            else:
                obj_constraints += v

        return obj_constraints


ag_np_log_vect = ag_np.vectorize(ag_np.log)


def _poisson_obj_single(structures, counts, lengths, alpha=-3., beta=None, bias=None, multiscale_factor=1, mask_fullres0=None, mixture_coefs=None, modifications=None):
    """
        Objective function of our model
    """

    if bias is not None and bias.sum() == 0:
        obj = 0.
    else:
        # Process structures for multiscale 3.0
        if modifications is not None and 'multiscale3' in modifications and multiscale_factor != 1:
            #print('*******\nX: %dx%d     counts: %dx%d' % (structures[0].shape[0], structures[0].shape[1], counts.shape[0], counts.shape[1])); print(structures[0])
            structures = repeat_X_multiscale(structures, lengths, multiscale_factor)

        if mixture_coefs is not None and len(structures) != len(mixture_coefs):
            raise ValueError("The number of structures (%d) and of mixture coefficents (%d) "
                             "should be identical." % (len(structures), len(mixture_coefs)))
        elif mixture_coefs is None:
            mixture_coefs = [1.]

        ploidy = int(structures[0].shape[0] / lengths.sum())
        lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)
        rows, cols = get_dis_indices(counts, n=lengths_lowres.sum(), lengths=lengths, ploidy=ploidy, multiscale_factor=multiscale_factor, nbeads=structures[0].shape[0], mask=mask_fullres0)

        lambda_intensity = ag_np.zeros(counts.nnz)
        for X, gamma in zip(structures, mixture_coefs):
            #print(multiscale_factor); print(X); print(multiscale_factor ** 2)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message='invalid value encountered in sqrt', category=RuntimeWarning)
                dis = ag_np.where(rows == cols, np.inf, ((X[rows] - X[cols]) ** 2).sum(axis=1)) ** 0.5
            tmp = (dis ** alpha).reshape(-1, int(len(rows) / multiscale_factor ** 2)).sum(axis=0)
            tmp = tmp.reshape(-1, counts.nnz).sum(axis=0)
            lambda_intensity = lambda_intensity + gamma * counts.bias_per_bin(bias, ploidy) * counts.beta * tmp

        # Sum main objective function
        obj = lambda_intensity.sum() - (get_data(counts) * ag_np.log(lambda_intensity)).sum()

        if ag_np.isnan(obj):
            print('\n\n%s, %s: X.shape=%s, counts.shape=%s, nan(X)=%d, nan(lambda)=%d, nan(counts)=%d, nan(log(lambda))=%d, nan(dis)=%d, zero(dis)=%d, nan(tmp)=%d, nan(bias)=%d\n\n\n' % (counts.name, ('obj' if type(obj) == np.float64 else 'grad'), 'x'.join(map(str, list(X.shape))), 'x'.join(map(str, list(counts.shape))), np.isnan(X[:, 0]).sum(), np.isnan(lambda_intensity).sum(), np.isnan(get_data(counts)).sum(), np.isnan(ag_np.log(lambda_intensity)).sum(), np.isnan(dis).sum(), (dis == 0).sum(), np.isnan(tmp).sum(), np.isnan(counts.bias_per_bin(bias, ploidy)).sum()))
            if np.isnan(dis).sum() > 0:
                print(counts.name); print(rows[np.isnan(dis)][:5]); print(cols[np.isnan(dis)][:5]); print(((X[rows] - X[cols]) ** 2).sum(axis=1)[np.isnan(dis)][:5]); print(X[rows][np.isnan(dis)][:5]); print(X[cols][np.isnan(dis)][:5])
            if (dis == 0).sum() > 0:
                print(counts.name); print(rows[dis == 0][:5]); print(cols[dis == 0][:5]); print(((X[rows] - X[cols]) ** 2).sum(axis=1)[dis == 0][:5]); print(X[rows][dis == 0][:5]); print(X[cols][dis == 0][:5])
            raise ValueError("Poisson component of objective function is nan")
        elif ag_np.isinf(obj):
            print('\n\n%s, %s: X.shape=%s, counts.shape=%s, nan(X)=%d, nan(lambda)=%d, nan(counts)=%d, nan(log(lambda))=%d, nan(dis)=%d, zero(dis)=%d, nan(tmp)=%d, nan(bias)=%d\n\n\n' % (counts.name, ('obj' if type(obj) == np.float64 else 'grad'), 'x'.join(map(str, list(X.shape))), 'x'.join(map(str, list(counts.shape))), np.isnan(X[:, 0]).sum(), np.isnan(lambda_intensity).sum(), np.isnan(get_data(counts)).sum(), np.isnan(ag_np.log(lambda_intensity)).sum(), np.isnan(dis).sum(), (dis == 0).sum(), np.isnan(tmp).sum(), np.isnan(counts.bias_per_bin(bias, ploidy)).sum()))
            raise ValueError("Poisson component of objective function is infinite")

        # If evaluating obj (not gradient), save objective
        if isinstance(obj, np.float64):
            global iter_obj_full
            if iter_obj_full is not None:
                iter_obj_full['obj_' + counts.name] = np.float64(obj)

    return obj


def translate_and_rotate(X, lengths, init_structures, translate, rotate, fix_homo=True):
    if not isinstance(init_structures, list):
        init_structures = [init_structures]
    init_structures = [init_structure.reshape(-1, 3) for init_structure in init_structures]

    if not (translate or rotate):
        return init_structures
    else:
        nchrom = lengths.shape[0]
        if not fix_homo:
            nchrom *= 2

        if translate and rotate:
            translations = X[:nchrom * 3].reshape(-1, 3)
            rotations = X[nchrom * 3:].reshape(-1, 4)
        elif translate:
            translations = X.reshape(-1, 3)
            rotations = ag_np.zeros((nchrom, 4))
        elif rotate:
            rotations = X.reshape(-1, 4)
            translations = ag_np.zeros((nchrom, 3))
        else:
            raise ValueError('Must select translate=True and/or rotate=True when finding ideal rotation and/or translation')

        ploidy = int(init_structures[0].shape[0] / lengths.sum())
        lengths = np.tile(lengths, ploidy)
        if fix_homo:
            translations = ag_np.tile(translations, (ploidy, 1))
            rotations = ag_np.tile(rotations, (ploidy, 1))

        new_structures = []
        for init_structure in init_structures:
            new_structure = []
            begin = end = 0
            for i in range(lengths.shape[0]):
                length = lengths[i]
                end += length
                if rotate:
                    new_structure.append(ag_np.dot(init_structure[begin:end, :] + translations[i, :], quat_to_rotation_matrix(rotations[i, :])))
                else:
                    new_structure.append(init_structure[begin:end, :] + translations[i, :])
                begin = end

            new_structure = ag_np.concatenate(new_structure)
            new_structures.append(new_structure)

        return new_structures


def poisson_obj(structures, counts, alpha=-3, beta=None, bias=None, input_weight=None, lengths=None, lagrange_mult=None, constraints=None,
                in_2d=False, multiscale_factor=1, mask_fullres0=None, mixture_coefs=None, modifications=None):
    """
    Computes the poisson objective function.

    Parameters
    ----------
    structures : list of ndarray or ndarray of shape (n, 3)

    counts : list of contact maps of shape (n, n) or (m, m)

    alpha : float, optional, default: -3
        counts-to-distance mapping parameter

    beta : float or array of floats, optional, default: None
        scaling factor of the structures. If counts is a list of contact maps,
        beta should be a list of scaling factors of the same length.
        if None, the optimal beta will be estimated.

    mixture_coefs : list, optional, default: None
        If inferring a mixture model, the mixture coefficients

    Returns
    -------
    obj : the value of the negative likelihood of the poisson model
    """

    #print(set([type(c) for c in counts]))

    # Check format of input
    counts = (counts if isinstance(counts, list) else [counts])
    lengths = (np.array([min([min(counts_maps.shape) for counts_maps in counts])]) if lengths is None else lengths)
    bias = (np.ones((min([min(counts_maps.shape) for counts_maps in counts]),)) if bias is None else bias)
    if not (isinstance(structures, list) or isinstance(structures, SequenceBox)):
        structures = [structures]
    if len(structures) > 1:
        raise NotImplementedError('Can only handle 1 structure right now')
    if mixture_coefs is None:
        mixture_coefs = [1.] * len(structures)
    if mask_fullres0 is None:
        mask_fullres0 = [None] * len(counts)
    if in_2d:
        structures = [X[:, :2] for X in structures]

    # Get poisson components of objective
    obj_poisson = np.float64(0.)
    for beta_maps, counts_maps, mask_fullres0_maps in zip(beta, counts, mask_fullres0):
        obj_poisson += counts_maps.weight * _poisson_obj_single(structures, counts_maps, lengths=lengths, alpha=alpha, beta=beta_maps, bias=bias, multiscale_factor=multiscale_factor, mask_fullres0=mask_fullres0_maps, mixture_coefs=mixture_coefs, modifications=modifications)

    # Get constraint components of objective
    obj_constraints = get_constraints(structures, counts, alpha=alpha, lengths=lengths, lagrange_mult=lagrange_mult, constraints=constraints, multiscale_factor=multiscale_factor, mask_fullres0=mask_fullres0, mixture_coefs=mixture_coefs, modifications=modifications)
    obj = obj_poisson + obj_constraints

    if isinstance(obj, np.float64):
        global iter_obj_full
        if iter_obj_full is not None:
            iter_obj_full['obj_main'] = np.float64(obj_poisson)
            iter_obj_full['obj'] = np.float64(obj)

    return obj


def obj_orient(X, counts, alpha=-3, beta=None, bias=None, input_weight=None, lengths=None, lagrange_mult=None, constraints=None,
               init_structures=None, translate=False, rotate=False, fix_homo=True, in_2d=False, multiscale_factor=1, mask_fullres0=None, mixture_coefs=None, modifications=None):
    # Optionally translate & rotate structures
    if translate or rotate:
        structures = translate_and_rotate(X, lengths, init_structures=init_structures, translate=translate, rotate=rotate, fix_homo=fix_homo)
    else:
        structures = X

    new_obj = poisson_obj(structures, counts, alpha=alpha, beta=beta, bias=bias, input_weight=input_weight, lengths=lengths, lagrange_mult=lagrange_mult, constraints=constraints,
                          in_2d=in_2d, multiscale_factor=multiscale_factor, mask_fullres0=mask_fullres0, mixture_coefs=mixture_coefs, modifications=modifications)
    return new_obj


def format_Xstructure(X, mixture_coefs=None):
    try:
        X = X.reshape((-1, 3))
    except ValueError:
        raise ValueError("X should contain k 3D structures")

    if mixture_coefs is None:
        mixture_coefs = [1]
    k = len(mixture_coefs)
    n = int(X.shape[0] / k)
    structures = [X[i * n:(i + 1) * n] for i in range(k)]

    return structures, mixture_coefs


def format_Xorient(X, init_structures, lengths, translate, rotate, fix_homo=True, mixture_coefs=None):
    if init_structures is None:
        raise ValueError('Must supply initial structures (init_structures=) when finding ideal rotation and/or translation')
    if lengths is None:
        raise ValueError('Must supply chromosome lengths (lengths=) when finding ideal rotation and/or translation')
    if not translate and not rotate:
        raise ValueError('Must select translate=True and/or rotate=True when finding ideal rotation and/or translation')

    nchrom = lengths.shape[0]
    if not fix_homo:
        nchrom *= 2

    if X.shape[0] != nchrom * (translate * 3 + rotate * 4):
        raise ValueError("X should contain rotation quaternions (length=4) and/or translation coordinates (length=3) for each of %d chromosomes. It is of length %d" % (nchrom, X.shape[0]))

    if mixture_coefs is None:
        mixture_coefs = [1]
    if not isinstance(init_structures, list):
        init_structures = [init_structures]
    init_structures = [structure.reshape(-1, 3) for structure in init_structures]

    ploidy = list(set([structure.shape[0] / lengths.sum() for structure in init_structures]))
    if len(ploidy) > 1:
        raise ValueError('Initial structures must all be the same shape')
    if ploidy[0] != 1 and ploidy[0] != 2:
        raise ValueError('Length of initial structures (%d) must be 1x or 2x the sum of chromosome lengths (%d), for haploid or diploid, respectively' % (init_structures[0].shape[0], lengths.sum()))

    return X, init_structures, mixture_coefs


def objective_wrapper(X, counts, alpha=-3., beta=None, bias=None, input_weight=None, lengths=None, lagrange_mult=None, constraints=None,
                      init_structures=None, translate=False, rotate=False, fix_homo=True, in_2d=False, multiscale_factor=1, mask_fullres0=None, mixture_coefs=None, modifications=None):
    """
    Objective function wrapper to match scipy.optimize's interface
    """

    if init_structures is not None or translate or rotate:
        X, init_structures, mixture_coefs = format_Xorient(X, init_structures, lengths, translate, rotate, fix_homo, mixture_coefs)
    else:
        X, mixture_coefs = format_Xstructure(X, mixture_coefs)

    new_obj = obj_orient(X, counts, alpha=alpha, beta=beta, bias=bias, input_weight=input_weight, lengths=lengths, lagrange_mult=lagrange_mult, constraints=constraints,
                         init_structures=init_structures, translate=translate, rotate=rotate, fix_homo=fix_homo, in_2d=in_2d, multiscale_factor=multiscale_factor, mask_fullres0=mask_fullres0, mixture_coefs=mixture_coefs,
                         modifications=modifications)

    return new_obj


gradient_orient = grad(obj_orient)


def fprime_wrapper(X, counts, alpha=-3, beta=None, bias=None, input_weight=None, lengths=None, lagrange_mult=None, constraints=None,
                   init_structures=None, translate=False, rotate=False, fix_homo=True, in_2d=False, multiscale_factor=1, mask_fullres0=None, mixture_coefs=None, modifications=None):

    if init_structures is not None or translate or rotate:
        X, init_structures, mixture_coefs = format_Xorient(X, init_structures, lengths, translate, rotate, fix_homo, mixture_coefs)
    else:
        X, mixture_coefs = format_Xstructure(X, mixture_coefs)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message='Using a non-tuple sequence for multidimensional indexing is deprecated', category=FutureWarning)
        new_grad = np.array(gradient_orient(X, counts, alpha=alpha, beta=beta, bias=bias, input_weight=input_weight, lengths=lengths, lagrange_mult=lagrange_mult, constraints=constraints,
                                            init_structures=init_structures, translate=translate, rotate=rotate, fix_homo=fix_homo, in_2d=in_2d, multiscale_factor=multiscale_factor, mask_fullres0=mask_fullres0, mixture_coefs=mixture_coefs,
                                            modifications=modifications)).flatten()

    #global iter_grad
    #iter_grad = new_grad

    return new_grad


def _estimate_beta_single(structures, counts, lengths, alpha=-3, bias=None, multiscale_factor=1, mask_fullres0=None, mixture_coefs=None):

    n, m = counts.shape

    if mixture_coefs is not None and len(structures) != len(mixture_coefs):
        raise ValueError("The number of structures (%d) and of mixture coefficents (%d) "
                         "should be identical." % (len(structures), len(mixture_coefs)))
    elif mixture_coefs is None:
        mixture_coefs = [1.]

    ploidy = int(structures[0].shape[0] / lengths.sum())
    lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)
    rows, cols = get_dis_indices(counts, n=lengths_lowres.sum(), lengths=lengths, ploidy=ploidy, multiscale_factor=multiscale_factor, nbeads=structures[0].shape[0], mask=mask_fullres0)
    if mask_fullres0 is not None:
        rows[~mask_fullres0] = 0
        cols[~mask_fullres0] = 0

    K = 0
    for X, gamma in zip(structures, mixture_coefs):
        #dis = np.sqrt(((X[rows] - X[cols]) ** 2).sum(axis=1))  # row sum
        dis = ag_np.where(rows == cols, np.float64(np.inf), ((X[rows] - X[cols]) ** 2).sum(axis=1)) ** 0.5
        tmp = dis ** alpha
        tmp = tmp.reshape(-1, int(len(rows) / multiscale_factor ** 2)).sum(axis=0)
        tmp = tmp.reshape(-1, counts.nnz).sum(axis=0)
        K += (gamma * counts.bias_per_bin(bias, ploidy) * tmp).sum()
    #beta = counts.input_sum / K

    return K


def _estimate_beta(X, counts, alpha=-3, bias=None, lengths=None, init_structures=None, translate=False, rotate=False, fix_homo=True, in_2d=False, multiscale_factor=1, mask_fullres0=None, mixture_coefs=None):

    if init_structures is not None or translate or rotate:
        X, init_structures, mixture_coefs = format_Xorient(X, init_structures, lengths, translate, rotate, fix_homo, mixture_coefs)
        structures = translate_and_rotate(X, lengths, init_structures=init_structures, translate=translate, rotate=rotate, fix_homo=fix_homo)
    else:
        structures, mixture_coefs = format_Xstructure(X, mixture_coefs)

    # Check format of input
    counts = (counts if isinstance(counts, list) else [counts])
    lengths = (np.array([min([min(counts_maps.shape) for counts_maps in counts])]) if lengths is None else lengths)
    bias = (np.ones((min([min(counts_maps.shape) for counts_maps in counts]),)) if bias is None else bias)
    if not (isinstance(structures, list) or isinstance(structures, SequenceBox)):
        structures = [structures]
    if len(structures) > 1:
        raise NotImplementedError('Can only handle 1 structure right now')
    if mixture_coefs is None:
        mixture_coefs = [1.] * len(structures)
    if mask_fullres0 is None:
        mask_fullres0 = [None] * len(counts)
    if in_2d:
        structures = [X[:, :2] for X in structures]

    # Estimate beta for each structure
    K = {counts_maps.ambiguity: 0. for counts_maps in counts}
    for counts_maps, mask_fullres0_maps in zip(counts, mask_fullres0):
        K[counts_maps.ambiguity] += _estimate_beta_single(structures, counts_maps, lengths, alpha=alpha, bias=bias, multiscale_factor=multiscale_factor, mask_fullres0=mask_fullres0_maps, mixture_coefs=mixture_coefs)
    beta = []
    for counts_maps in counts:
        beta_maps = counts_maps.input_sum / K[counts_maps.ambiguity]
        beta.append(beta_maps)
        if np.isnan(beta_maps):
            raise ValueError("Inferred beta is not a number.")
        elif np.isinf(beta_maps):
            raise ValueError("Inferred beta is infinite.")
        elif beta_maps == 0:
            raise ValueError("Inferred beta is zero.")

    #beta = []
    #for counts_maps, mask_fullres0_maps in zip(counts, mask_fullres0):
    #    beta.append(_estimate_beta_single(structures, counts_maps, lengths, alpha=alpha, bias=bias, multiscale_factor=multiscale_factor, mask_fullres0=mask_fullres0_maps, mixture_coefs=mixture_coefs))

    print('INFERRED BETA: %s' % ', '.join('%s=%.2g' % (counts[i].name, beta[i]) for i in range(len(beta))), flush=True)
    return beta


def check_constraints(lagrange_mult, constraints, verbose=1):
    # Set defaults
    lagrange_mult_defaults = {'adj': 0., 'noshade': 0., 'homo': 0., 'inter': 0., 'intra': 0., 'X': 0., 'homodis': 0.}
    lagrange_mult_all = lagrange_mult_defaults
    if lagrange_mult is not None:
        for k, v in lagrange_mult.items():
            if k not in lagrange_mult_all:
                raise ValueError('lagrange_mult key not recognized - %s' % k)
            elif v is not None:
                lagrange_mult_all[k] = float(v)
    lagrange_mult = lagrange_mult_all

    constraints_defaults = {'adj': 'var', 'noshade': 'poisson', 'homo': None, 'inter': 0.5, 'intra': 0.5, 'X': None, 'homodis': None}
    constraints_all = constraints_defaults
    if constraints is not None:
        for k, v in constraints.items():
            if k not in constraints_all:
                raise ValueError('constraints key not recognized - %s' % k)
            elif v is not None:
                if isinstance(v, int):
                    v = float(v)
                constraints_all[k] = v
    constraints = constraints_all

    # check constraints
    for k, v in lagrange_mult.items():
        if v != lagrange_mult_defaults[k]:
            if v < 0:
                raise ValueError("Lagrange multipliers must be >= 0. lagrange_mult[%s] is %g" % (k, v))
            if constraints[k] is None:
                raise ValueError("Lagrange multiplier for %s is supplied, but constraint is not" % k)
        elif constraints[k] != constraints_defaults[k]:
            raise ValueError("Constraint for %s is supplied, but lagrange multiplier is 0" % k)

    if verbose and sum(lagrange_mult.values()) != 0:
        lagrange_mult_to_print = {k: 'lambda = %.2g' % v for k, v in lagrange_mult.items() if v != 0}
        if 'adj' in lagrange_mult_to_print and constraints['adj'] == 'var':
            print('CONSTRAINTS: bead chain connectivity %s' % lagrange_mult_to_print.pop('adj'), flush=True)
        if 'homodis' in lagrange_mult_to_print:
            print('CONSTRAINTS: homolog-separating %s,    r = %s    (barycenter distance)' % (lagrange_mult_to_print.pop('homodis'), ('%.3g' % constraints['homodis']) if isinstance(constraints['homodis'], float) else ('inferred' if constraints['homodis'] is None else constraints['homodis'])), flush=True)
        if 'homo' in lagrange_mult_to_print:
            print('CONSTRAINTS: homolog-separating %s,    r = %s    (inter-homolog counts ratio)' % (lagrange_mult_to_print.pop('homo'), ('%.3g' % constraints['homo']) if isinstance(constraints['homo'], float) else ('inferred' if constraints['homo'] is None else constraints['homo'])), flush=True)
        if len(lagrange_mult_to_print) > 0:
            print('CONSTRAINTS:  %s' % (',  '.join([str(k) + ' ' + str(constraints[k]) + ': %.1g' % v for k, v in lagrange_mult_to_print.items() if v != 0])), flush=True)
    return lagrange_mult, constraints


def estimate_X(counts, init_X, alpha=-3., beta=None, bias=None, max_iter=10000000000, verbose=0,
               lengths=None, input_weight=None, lagrange_mult=None, constraints=None, in_2d=False,
               callback_frequency=2, X_true=None, multiscale_factor=1, mask_fullres0=None, mask_fullresX=None,
               as_sparse=True, mixture_coefs=None, high_accuracy=False,
               init_structures=None, translate=False, rotate=False, fix_homo=True,
               modifications=None, return_nonconverged=False, ploidy=1):
    """
    Estimates a 3D model

    Parameters
    ----------
    counts : list of contact maps of shape (n, n) or (m, m)

    alpha : float, optional, default: -3
        counts-to-distance mapping parameter

    beta : float or array of floats, optional, default: None
        scaling factor of the structures. If counts is a list of contact maps,
        beta should be a list of scaling factors of the same length.
        if None, the optimal beta will be estimated.

    mixture_coefs : list, optional, default: None
        If inferring a mixture model, the mixture coefficients

    use_zero_counts : boolean, optional, default: False
        Whether to use zero contact counts
    """

    # TODO interface multi dataset with the rest.

    # Check format of input
    counts = (counts if isinstance(counts, list) else [counts])
    if lengths is None:
        if ploidy != 1:
            raise ValueError("Must supply lengths for diploid inference")
        lengths = [min([min(counts_maps.shape) for counts_maps in counts])]
    lengths = np.array(lengths)
    bias = (np.ones((min([min(counts_maps.shape) for counts_maps in counts]),)) if bias is None else bias)
    if init_structures is not None or translate or rotate:
        if init_structures is None:
            raise ValueError('Must supply initial structures (init_structures=) when finding ideal rotation and/or translation')
        if not translate and not rotate:
            raise ValueError('Must select translate=True and/or rotate=True when finding ideal rotation and/or translation')

    # Set lengths_lowres (same resolution as counts)
    lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)

    # Estimate beta if necessary
    if beta is None:
        beta = _estimate_beta(init_X.flatten(), counts, alpha=alpha, bias=bias, lengths=lengths, init_structures=init_structures,
                              translate=translate, rotate=rotate, fix_homo=fix_homo, in_2d=in_2d, multiscale_factor=multiscale_factor, mask_fullres0=mask_fullres0, X_indices=None, mixture_coefs=mixture_coefs)
        for counts_maps, beta_maps in zip(counts, beta):
            counts_maps.beta = beta_maps

    # Setup callback function
    if callback_frequency is None:
        callback_fxn = None
    else:
        from timeit import default_timer as timer
        from datetime import timedelta
        time_start = timer()
        def callback_fxn(Xi, final_results=False):
            global n_iter
            global iter_obj_full
            if verbose == 0 and (n_iter <= 10 or n_iter % 100 == 0):
                #global iter_grad
                #global iter_obj_full
                #proj_g = ag_np.linalg.norm(iter_grad)
                time_current = str(timedelta(seconds=timer() - time_start)).split('.')
                info_dict = {'At iterate': ' ' * (5 - len(str(n_iter))) + str(n_iter), 'f= ': '%.6g' % iter_obj_full['obj'], 'time= ': time_current[0] + '.' + time_current[1][:2]}  # , '|proj g|= ': '%.6g' % proj_g}
                print('\t\t'.join(['%s%s' % (k, v) for k, v in info_dict.items()]), flush=True)
                if n_iter == 10:
                    print('. . .', flush=True)
                print('', flush=True)
            if (callback_frequency and n_iter % callback_frequency == 0) or n_iter in (0, 1) or final_results:
                current_X = Xi.copy()
                if init_structures is not None or translate or rotate:
                    current_X = translate_and_rotate(current_X, lengths, init_structures, translate, rotate, fix_homo)[0]
                if not (modifications is not None and 'multiscale3' in modifications):
                    current_X = reduce_X_res(current_X, multiscale_factor, lengths, mask=mask_fullresX)
                Xi_masked = mask_X(current_X.reshape(-1, 3), counts)
                #Xi_masked = mask_X(Xi.copy().reshape(-1, 3), create_dummy_counts(counts, lengths, multiscale_factor=multiscale_factor))
                global iter_details
                #global iter_obj_full
                for k, v in iter_obj_full.items():
                    if k in iter_details:
                        iter_details[k].append(v)
                    else:
                        iter_details[k] = [v]
                iter_details['iter'].append(n_iter)
                if X_true is not None:
                    metrics = simulated_vs_inferred(X_true, Xi_masked, lengths_lowres, verbose=0, modifications=modifications)
                    for k, v in metrics.items():
                        if k in iter_details:
                            iter_details[k].append(v)
                        else:
                            iter_details[k] = [v]
            n_iter += 1

        # Initialize global variables for callback function
        callback_setup()

        # Generate metrics for init
        objective_wrapper(init_X.flatten(), counts, alpha, beta, bias, input_weight, lengths, lagrange_mult, constraints,
                          init_structures, translate, rotate, fix_homo, in_2d, multiscale_factor, mask_fullres0, mixture_coefs, modifications=modifications)

    if verbose == 0:
        print('=' * 30, flush=True)
        print('\nRUNNING THE L-BFGS-B CODE\n\n           * * *\n\nMachine precision = %.4g\n' % np.finfo(np.float).eps, flush=True)
        if callback_frequency is not None:
            callback_fxn(init_X)

    if high_accuracy:
        pgtol = 1e-10; factr = 10.0  # a fun choice if you're okay with nonconvergence issues
    else:
        pgtol = 1e-05; factr = 10000000.0  # fmin_l_bfgs_b defaults

    results = optimize.fmin_l_bfgs_b(
        objective_wrapper,
        x0=init_X.flatten(),
        fprime=fprime_wrapper,
        iprint=verbose,
        maxiter=max_iter,
        callback=callback_fxn,
        pgtol=pgtol,
        factr=factr,
        args=(counts, alpha, beta, bias, input_weight, lengths, lagrange_mult, constraints,
              init_structures, translate, rotate, fix_homo, in_2d, multiscale_factor, mask_fullres0,
              mixture_coefs, modifications))

    if callback_frequency is not None:
        global n_iter
        if callback_frequency == 0 or n_iter % callback_frequency != 0:
            callback_fxn(results[0], final_results=True)
        global iter_details
        iter_details = {k: v for k, v in iter_details.items() if len(v) != 0 and set(v) != {np.nan}}
    else:
        iter_details = {}

    # Get final objective value - FYI THIS ISN'T NECESSARY, RESULTS[1] IS FINAL_OBJ
    #final_obj = np.float64(objective_wrapper(results[0].flatten(), counts, alpha, beta, bias, input_weight, lengths, lagrange_mult, constraints,
    #                                         init_structures, translate, rotate, fix_homo, in_2d, multiscale_factor, mask_fullres0, mixture_coefs, modifications=modifications))
    final_obj = results[1]

    # Reduce resolution of X
    results = list(results)
    if not (modifications is not None and 'multiscale3' in modifications):
        results[0] = reduce_X_res(results[0], multiscale_factor, lengths, mask=mask_fullresX)

    return results, final_obj, iter_details


def estimate_null(counts, lengths, init_X, alpha=-3., max_iter=10000000000, ploidy=1, verbose=False, in_2d=False, lagrange_mult=None, constraints=None, X_true=None, callback_frequency=None,
                  init_structures=None, translate=False, rotate=False, fix_homo=True, multiscale_factor=1, mask_fullres0=None, mask_fullresX=None, as_sparse=True, mixture_coefs=None, modifications=None):
    """
    Estimate a "null" structure

    Estimates a "null" structure that fullfills the constraint but is not
    fitted to the data

    Parameters
    ----------

    FIXME
    """
    # Dummy counts need to be inputted because if a row/col is all 0 it is excluded from calculations
    # To create dummy counts, ambiguate counts & sum together, then set all non-0 values to 1
    lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)
    dummy_counts = [create_dummy_counts(counts=counts, lengths=lengths_lowres, ploidy=ploidy).astype(float)]

    # Set beta & biases & input_weight to 0 to ensure that poisson is not applied
    dummy_beta = [0.] * len(dummy_counts)
    dummy_biases = np.zeros((dummy_counts[0].shape[0] * ploidy, ))
    dummy_input_weight = [0.] * len(dummy_counts)

    # Reformat counts as sparse_counts_matrix or zero_counts_matrix objects
    dummy_counts = format_counts(dummy_counts, dummy_beta, dummy_input_weight, lengths=lengths_lowres, ploidy=ploidy, as_sparse=as_sparse)

    lagrange_mult, constraints = check_constraints(lagrange_mult, constraints, verbose=True)

    if init_X is None:
        return None, False
    else:
        # Infer structure
        optimized, obj, iter_details = estimate_X(
            counts=dummy_counts,
            init_X=init_X.flatten(),
            beta=dummy_beta,
            alpha=alpha,
            bias=dummy_biases,
            ploidy=ploidy,
            max_iter=max_iter,
            verbose=verbose,
            X_true=X_true,
            lengths=lengths,
            input_weight=dummy_input_weight,
            lagrange_mult=lagrange_mult,
            constraints=constraints,
            init_structures=init_structures,
            translate=translate,
            rotate=rotate,
            fix_homo=fix_homo,
            callback_frequency=callback_frequency,
            in_2d=in_2d,
            multiscale_factor=multiscale_factor,
            mask_fullres0=mask_fullres0,
            mask_fullresX=mask_fullresX,
            as_sparse=as_sparse,
            mixture_coefs=mixture_coefs,
            modifications=modifications)

        null, func, d = optimized
        converged = d['warnflag'] == 0
        return null, converged, obj


def get_obj_details(X, counts, alpha=-3., beta=None, bias=None, input_weight=None, lengths=None, lagrange_mult=None, constraints=None,
                    init_structures=None, translate=False, rotate=False, fix_homo=True, in_2d=False, multiscale_factor=1, mask_fullres0=None, mixture_coefs=None, modifications=None):
    callback_setup()

    obj = objective_wrapper(X, counts, alpha=alpha, beta=beta, bias=bias,
                            input_weight=input_weight, lengths=lengths, lagrange_mult=lagrange_mult, constraints=constraints,
                            init_structures=init_structures, translate=translate, rotate=rotate, fix_homo=fix_homo, in_2d=in_2d,
                            multiscale_factor=multiscale_factor, mask_fullres0=mask_fullres0, mixture_coefs=mixture_coefs, modifications=modifications)

    global iter_obj_full
    obj_details = {k: v for k, v in iter_obj_full.items() if not np.isnan(v)}

    return obj_details


class PM1(object):
    """
    """
    def __init__(self, alpha=-3., beta=1., max_iter=10000000000, random_state=None,
                 init=None, verbose=False, ploidy=1, in_2d=False, X_true=None,
                 input_weight=None, lagrange_mult=None, constraints=None, homo_init=None, as_sparse=True,
                 multiscale_factor=1, mixture_coefs=None, HSC_lowres_beads=5, modifications=None,
                 init_structures=None, translate=False, rotate=False, fix_homo=True):

        print('%s\n%s 3D STRUCTURAL INFERENCE' % ('=' * 30, {2: 'DIPLOID', 1: 'HAPLOID'}[ploidy]), flush=True)

        if isinstance(X_true, list) and len(X_true) == 1:
            X_true = X_true[0]

        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = (beta if isinstance(beta, list) or beta is None else [beta])
        self.random_state = check_random_state(random_state)
        self.init = init
        self.verbose = verbose
        self.ploidy = (1 if ploidy is None else ploidy)
        self.in_2d = in_2d
        self.X_true = X_true
        self.input_weight = input_weight
        self.lagrange_mult, self.constraints = check_constraints(lagrange_mult, constraints, verbose=True)
        self.homo_init = homo_init
        self.multiscale_factor = multiscale_factor
        self.as_sparse = as_sparse
        self.mixture_coefs = mixture_coefs
        self.HSC_lowres_beads = HSC_lowres_beads

        self.counts = None
        self.fullres_counts = None
        self.lengths = None
        self.homo_init_style = None
        self.mask_fullres0 = None
        self.mask_fullresX = None

        if init_structures is not None or translate or rotate:
            if init_structures is None:
                raise ValueError('Must supply initial structures (init_structures=) when finding ideal rotation and/or translation')
            if not translate and not rotate:
                raise ValueError('Must select translate=True and/or rotate=True when finding ideal rotation and/or translation')
        self.init_structures = init_structures
        self.translate = translate
        self.rotate = rotate
        self.fix_homo = fix_homo

        if modifications is not None and len(modifications) > 0:
            print('MODIFICATIONS: %s' % ', '.join(modifications))
        self.modifications = modifications

    def prep_counts(self, counts, lengths, normalize=False, filter_counts=False, filter_percentage=0.04, mask_fullres0=False, mask_fullresX=False, lighter0=False, nonUA0=False):
        from .prep_counts import prep_counts
        from .utils import mask_zero_in_fullres_counts, mask_torm_in_fullres_X, find_beads_to_remove

        if lengths is None:
            if self.ploidy != 1:
                raise ValueError("Must supply lengths for diploid inference")
            lengths = [min([min(counts_maps.shape) for counts_maps in counts])]
        self.lengths = np.array(lengths)

        # Check counts, reduce resolution, filter, compute bias
        counts = check_counts(counts, self.as_sparse)
        counts_prepped, self.bias = prep_counts(counts, lengths, ploidy=self.ploidy, multiscale_factor=self.multiscale_factor, normalize=normalize, filter_counts=filter_counts, filter_percentage=filter_percentage, as_sparse=self.as_sparse, verbose=True)
        fullres_counts_prepped, self.full_res_bias = prep_counts(counts, lengths, ploidy=self.ploidy, multiscale_factor=1, normalize=normalize, filter_counts=filter_counts, filter_percentage=filter_percentage, as_sparse=self.as_sparse, verbose=False)

        # Reformat counts as sparse_counts_matrix or zero_counts_matrix objects
        lengths_lowres = decrease_lengths_res(self.lengths, self.multiscale_factor)
        self.counts = format_counts(counts_prepped, self.beta, self.input_weight, lengths=lengths_lowres, ploidy=self.ploidy, as_sparse=self.as_sparse, lighter0=lighter0, nonUA0=nonUA0)
        self.fullres_counts = format_counts(fullres_counts_prepped, self.beta, self.input_weight, lengths=lengths, ploidy=self.ploidy, as_sparse=self.as_sparse, lighter0=lighter0, nonUA0=nonUA0)

        if (mask_fullres0 or mask_fullresX) and self.multiscale_factor != 1:
            if mask_fullres0:
                self.mask_fullres0 = mask_zero_in_fullres_counts(self.counts, self.fullres_counts, self.lengths, self.ploidy, self.multiscale_factor)
            self.mask_fullresX = mask_torm_in_fullres_X(self.fullres_counts, self.lengths, self.ploidy, self.multiscale_factor)

        # Get vector of beads to remove (torm)
        self.torm = find_beads_to_remove(self.counts, decrease_lengths_res(lengths, self.multiscale_factor).sum() * self.ploidy)

        return [('counts' if self.ploidy == 1 else {1: 'ambig_counts', 1.5: 'pa_counts', 2: 'ua_counts'}[sum(c.shape) / (lengths.sum() * self.ploidy)], c) for c in fullres_counts_prepped]

    def parse_homolog_sep(self):
        from .separating_homologs import parse_homolog_sep

        self.homo_init_style = (self.homo_init.lower() if isinstance(self.homo_init, str) else self.homo_init)
        self.constraints, self.homo_init = parse_homolog_sep(self.constraints, self.homo_init, self.lengths, counts=self.counts, fullres_counts=self.fullres_counts, X_true=self.X_true, init_structures=self.init_structures, alpha=self.alpha, beta=self.beta, multiscale_factor=self.multiscale_factor, HSC_lowres_beads=self.HSC_lowres_beads, modifications=self.modifications)

        return self.constraints

    def initialize(self):
        from .initialization import initialize_X

        if isinstance(self.init, str) and self.init.lower() in ('true', 'xtrue', 'x_true'):
            if self.X_true is None:
                raise ValueError('Attempting to initialize with X_true but X_true is None')
            self.init = self.X_true

        if self.modifications is not None and 'fullres_mds' in self.modifications:
            counts = self.fullres_counts
            multiscale_factor = 1
            lengths = self.lengths
        elif self.modifications is not None and 'multiscale3' in self.modifications:
            counts = self.counts
            multiscale_factor = 1
            lengths = decrease_lengths_res(self.lengths, self.multiscale_factor)
        else:
            counts = self.counts
            multiscale_factor = self.multiscale_factor
            lengths = self.lengths
        init_X = initialize_X(counts, lengths, self.random_state, init=self.init, ploidy=self.ploidy, alpha=self.alpha, beta=self.beta, bias=self.full_res_bias, in_2d=self.in_2d,
                              homo_init=self.homo_init, homo_init_style=self.homo_init_style, multiscale_factor=multiscale_factor, mask_fullresX=self.mask_fullresX,
                              init_structures=self.init_structures, translate=self.translate, rotate=self.rotate, fix_homo=self.fix_homo,
                              modifications=self.modifications, verbose=True)
        self.init_X_ = init_X

    def obj_for_X_true(self):
        X_true_obj = get_obj_details(self.X_true, self.counts, alpha=self.alpha, beta=self.beta, bias=self.bias, input_weight=self.input_weight, lengths=decrease_lengths_res(self.lengths, self.multiscale_factor),
                                     lagrange_mult=self.lagrange_mult, constraints=self.constraints, multiscale_factor=1, mask_fullres0=None,
                                     init_structures=self.init_structures, translate=self.translate, rotate=self.rotate, fix_homo=self.fix_homo, in_2d=self.in_2d, modifications=self.modifications)
        return X_true_obj

    def null(self, callback_frequency=0):
        """
        """

        # Dummy counts need to be inputted because if a row/col is all 0 it is excluded from calculations
        null, converged, obj = estimate_null(self.counts, self.lengths, init_X=self.init_X_.flatten(), alpha=self.alpha, max_iter=self.max_iter, verbose=self.verbose, ploidy=self.ploidy,
                                             in_2d=self.in_2d, lagrange_mult=self.lagrange_mult, constraints=self.constraints, X_true=self.X_true, callback_frequency=callback_frequency,
                                             multiscale_factor=self.multiscale_factor, mask_fullres0=self.mask_fullres0, mask_fullresX=self.mask_fullresX, as_sparse=self.as_sparse, mixture_coefs=self.mixture_coefs,
                                             init_structures=self.init_structures, translate=self.translate, rotate=self.rotate, fix_homo=self.fix_homo, modifications=self.modifications)
        null = null.reshape(-1, 3)

        # Save final objective
        self.obj_ = obj

        # Since beta isn't inferred with null
        self.beta_ = self.beta

        # Since alpha isn't inferred
        self.alpha_ = self.alpha

        self.converged_ = converged
        if callback_frequency is not None:
            self.iter_details_ = iter_details
        return null

    def infer_beta(self):
        structure = self.init_X_.flatten()
        if self.init_structures is not None or self.translate or self.rotate:
            structure = self.init_structures
        inferred_beta = _estimate_beta(structure, self.counts, alpha=self.alpha, bias=self.bias, lengths=self.lengths, init_structures=self.init_structures,
                                       multiscale_factor=self.multiscale_factor, mixture_coefs=self.mixture_coefs, mask_fullres0=self.mask_fullres0,
                                       translate=self.translate, rotate=self.rotate, fix_homo=self.fix_homo, in_2d=self.in_2d)
        self.beta_ = []
        for i in range(len(self.counts)):
            self.counts[i].beta = inferred_beta[i]
            if self.counts[i].sum() != 0:
                self.beta_.append(inferred_beta[i])
        return inferred_beta

    def fit(self, callback_frequency=2):
        """
        """

        if self.init_X_ is None:
            self.converged_ = False
            return None
        else:
            # Estimate betas
            inferred_beta = self.beta
            self.beta_ = self.beta
            if self.beta is None:
                inferred_beta = self.infer_beta()

            # Infer structure
            optimized, obj, iter_details = estimate_X(
                self.counts,
                init_X=self.init_X_.flatten(),
                beta=inferred_beta,
                alpha=self.alpha,
                bias=self.bias,
                ploidy=self.ploidy,
                max_iter=self.max_iter,
                verbose=self.verbose,
                X_true=self.X_true,
                lengths=self.lengths,
                input_weight=self.input_weight,
                lagrange_mult=self.lagrange_mult,
                constraints=self.constraints,
                init_structures=self.init_structures,
                translate=self.translate,
                rotate=self.rotate,
                fix_homo=self.fix_homo,
                callback_frequency=callback_frequency,
                in_2d=self.in_2d,
                multiscale_factor=self.multiscale_factor,
                mask_fullres0=self.mask_fullres0,
                mask_fullresX=self.mask_fullresX,
                as_sparse=self.as_sparse,
                mixture_coefs=self.mixture_coefs,
                modifications=self.modifications)

            X, func, d = optimized
            if not (self.translate or self.rotate):
                X = X.reshape(-1, 3)
                X[self.torm] = np.nan

            # Save final objective
            self.obj_ = obj

            # Since alpha isn't inferred
            self.alpha_ = self.alpha

            self.converged_ = d['warnflag'] == 0
            if callback_frequency is not None:
                self.iter_details_ = iter_details
            return X
