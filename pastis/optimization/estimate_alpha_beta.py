import numpy as np
from scipy import optimize
import warnings
from autograd import grad
from autograd.builtins import SequenceBox
from sklearn.utils import check_random_state
from .poisson import _format_X, objective
from .counts import _update_betas_in_counts_matrices


def _estimate_beta_single(structures, counts, alpha, lengths, bias=None,
                          mixture_coefs=None):
    """Facilitates estimation of beta for a single counts object.

    Computes the sum of distances (K) corresponding to a given counts matrix.
    """

    n, m = counts.shape

    if mixture_coefs is not None and len(structures) != len(mixture_coefs):
        raise ValueError("The number of structures (%d) and of mixture"
                         " coefficents (%d) should be identical." %
                         (len(structures), len(mixture_coefs)))
    elif mixture_coefs is None:
        mixture_coefs = [1.]

    ploidy = int(structures[0].shape[0] / lengths.sum())

    K = 0
    for struct, gamma in zip(structures, mixture_coefs):
        dis = ((struct[counts.row3d] - struct[counts.col3d]) ** 2).sum(
            axis=1) ** 0.5
        tmp = dis ** alpha
        tmp = tmp.reshape(-1, counts.nnz).sum(axis=0)
        K += (gamma * counts.bias_per_bin(bias, ploidy) * tmp).sum()

    return K


def _estimate_beta(X, counts, alpha, lengths, bias=None, reorienter=None,
                   mixture_coefs=None, verbose=True, simple_diploid=False):
    """Estimates beta for all counts matrices.
    """

    structures, mixture_coefs = _format_X(X, reorienter, mixture_coefs)
    if reorienter is not None and reorienter.reorient:
        structures = reorienter.translate_and_rotate(X)

    # Check format of input
    counts = (counts if isinstance(counts, list) else [counts])
    if lengths is None:
        lengths = np.array([min([min(counts_maps.shape) for counts_maps in counts])])
    lengths = np.array(lengths)
    if bias is None:
        bias = np.ones((min([min(counts_maps.shape) for counts_maps in counts]),))
    if not (isinstance(structures, list) or isinstance(structures, SequenceBox)):
        structures = [structures]
    if mixture_coefs is None:
        mixture_coefs = [1.] * len(structures)

    # Estimate beta for each type of counts (ambig, pa, ua)
    counts_sum = {counts_maps.ambiguity: counts_maps.input_sum for counts_maps in counts}
    K = {counts_maps.ambiguity: 0. for counts_maps in counts}

    if simple_diploid:
        structures_homo1 = [s[:lengths.sum()] for s in structures]
        structures_homo2 = [s[lengths.sum():] for s in structures]
        for structures_homo in (structures_homo1, structures_homo2):
            for counts_maps in counts:
                K[counts_maps.ambiguity] += _estimate_beta_single(
                    structures_homo, counts_maps, alpha=alpha, lengths=lengths,
                    bias=bias, mixture_coefs=mixture_coefs)
        K = {k / 2 for k in K}
    else:
        for counts_maps in counts:
            K[counts_maps.ambiguity] += _estimate_beta_single(
                structures, counts_maps, alpha=alpha, lengths=lengths,
                bias=bias, mixture_coefs=mixture_coefs)

    beta = {k: counts_sum[k] / K[k] for k in counts_sum.keys()}
    for ambiguity, beta_maps in beta.items():
        if np.isnan(beta_maps):
            raise ValueError("Beta inferred for %s counts is not a number."
                             % ambiguity)
        elif np.isinf(beta_maps):
            raise ValueError("Beta inferred for %s counts is infinite."
                             % ambiguity)
        elif beta_maps == 0:
            raise ValueError("Beta inferred for %s counts is zero."
                             % ambiguity)

    counts = _update_betas_in_counts_matrices(counts=counts, beta=beta)

    if verbose:
        print('INFERRED BETA: %s' % ', '.join(['%s=%.2g' %
              (counts_maps.name, counts_maps.beta) for counts_maps in counts]),
              flush=True)

    return counts


def objective_alpha(alpha, counts, X, lengths, bias=None, constraints=None,
                    reorienter=None, multiscale_factor=1,
                    multiscale_variances=None, mixture_coefs=None,
                    return_components=False):
    """Computes the objective function.

    Computes the negative log likelihood of the poisson model and constraints.

    Parameters
    ----------
    alpha : float, optional
        Biophysical parameter of the transfer function used in converting
        counts to wish distances.
    counts : list of CountsMatrix subclass instances
        Preprocessed counts data.
    X : array of float
        Structure being inferred.
    lengths : array of int
        Number of beads per homolog of each chromosome.
    bias : array of float, optional
        Biases computed by ICE normalization.
    constraints : Constraints instance, optional
        Object to compute constraints at each iteration.
    multiscale_factor : int, optional
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 indicates full resolution.
    multiscale_variances : float or array of float, optional
        For multiscale optimization at low resolution, the variances of each
        group of full-resolution beads corresponding to a single low-resolution
        bead.

    Returns
    -------
    obj : float
        The total negative log likelihood of the poisson model and constraints.
    """

    return objective(X, counts, alpha=alpha, lengths=lengths, bias=bias,
                     constraints=constraints, reorienter=reorienter,
                     multiscale_factor=multiscale_factor,
                     multiscale_variances=multiscale_variances,
                     mixture_coefs=mixture_coefs,
                     return_components=return_components)


gradient_alpha = grad(objective_alpha)


def objective_wrapper_alpha(alpha, counts, X, lengths, bias=None,
                            constraints=None, reorienter=None,
                            multiscale_factor=1, multiscale_variances=None,
                            mixture_coefs=None, callback=None):
    """Objective function wrapper to match scipy.optimize's interface.
    """

    counts = _estimate_beta(X, counts, alpha=alpha, lengths=lengths, bias=bias,
                            reorienter=reorienter, mixture_coefs=mixture_coefs)

    X, mixture_coefs = _format_X(X, reorienter, mixture_coefs)

    new_obj, obj_logs, structures, alpha = objective_alpha(
        alpha, counts=counts, X=X, lengths=lengths, bias=bias, constraints=constraints,
        reorienter=reorienter, multiscale_factor=multiscale_factor,
        multiscale_variances=multiscale_variances, mixture_coefs=mixture_coefs,
        return_components=True)

    if callback is not None:
        callback.on_epoch_end(obj_logs, structures, alpha, X)

    return new_obj


def fprime_wrapper_alpha(alpha, counts, X, lengths, bias=None, constraints=None,
                         reorienter=None, multiscale_factor=1,
                         multiscale_variances=None, mixture_coefs=None,
                         callback=None):
    """Gradient function wrapper to match scipy.optimize's interface.
    """

    counts = _estimate_beta(X, counts, alpha=alpha, lengths=lengths, bias=bias,
                            reorienter=reorienter, mixture_coefs=mixture_coefs)

    X, mixture_coefs = _format_X(X, reorienter, mixture_coefs)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message='Using a non-tuple sequence for multidimensional'
                              ' indexing is deprecated', category=FutureWarning)
        new_grad = np.array(gradient_alpha(
            alpha, counts=counts, X=X, lengths=lengths, bias=bias,
            constraints=constraints, reorienter=reorienter,
            multiscale_factor=multiscale_factor,
            multiscale_variances=multiscale_variances,
            mixture_coefs=mixture_coefs)).flatten()

    return new_grad


def estimate_alpha(counts, X, alpha_init, lengths, bias=None,
                   constraints=None, multiscale_factor=1,
                   multiscale_variances=None, random_state=None,
                   max_iter=10000000000, factr=10000000., pgtol=1e-05,
                   callback=None, alpha_loop=None, reorienter=None,
                   mixture_coefs=None, verbose=True):
    """Estimates alpha, given current structure.

    Parameters
    ----------
    Given a chromatin structure, infer alpha from Hi-C contact counts data for
    haploid or diploid organisms at a given resolution.

    Parameters
    ----------
    counts : list of CountsMatrix subclass instances
        Preprocessed counts data.
    X : array_like of float
        3D chromatin structure.
    alpha_init : float
        Initialization of alpha, the biophysical parameter of the transfer
        function used in converting counts to wish distances.
    lengths : array_like of int
        Number of beads per homolog of each chromosome.
    bias : array_like of float, optional
        Biases computed by ICE normalization.
    constraints : Constraints instance, optional
        Object to compute constraints at each iteration.
    multiscale_factor : int, optional
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 indicates full resolution.
    multiscale_variances : float or array_like of float, optional
        For multiscale optimization at low resolution, the variances of each
        group of full-resolution beads corresponding to a single low-resolution
        bead.
    max_iter : int, optional
        Maximum number of iterations per optimization.
    factr : float, optional
        factr for scipy's L-BFGS-B, alters convergence criteria.
    pgtol : float, optional
        pgtol for scipy's L-BFGS-B, alters convergence criteria.
    callback : pastis.callbacks.Callback object, optional
        Object to perform callback at each iteration and before and after
        optimization.
    alpha_loop : int, optional
        Current iteration of alpha/structure optimization.

    Returns
    -------
    alpha : float
        Output of the optimization, the biophysical parameter of the transfer
        function used in converting counts to wish distances.
    obj : float
        Final objective value.
    converged : bool
        Whether the optimization successfully converged.
    callback.history : list of dict
        History generated by the callback, containing information about the
        objective function during optimization.
    """

    # Check format of input
    counts = (counts if isinstance(counts, list) else [counts])
    lengths = np.array(lengths)
    if bias is None:
        bias = np.ones((min([min(counts_maps.shape) for counts_maps in counts]),))

    # Initialize alpha if necessary
    if random_state is None:
        random_state = np.random.RandomState(seed=0)
    random_state = check_random_state(random_state)
    if alpha_init is None:
        alpha_init = (- random_state.randint(1, 100) + random_state.rand(1))[0]

    if verbose:
        print('=' * 30, flush=True)
        print('\nRUNNING THE L-BFGS-B CODE\n\n           * * *\n\n Machine'
              ' precision = %.4g\n' % np.finfo(np.float).eps, flush=True)

    if callback is not None:
        if reorienter is not None and reorienter.reorient:
            opt_type = 'alpha.chrom_reorient'
        else:
            opt_type = 'alpha'
        callback.on_training_begin(opt_type=opt_type, alpha_loop=alpha_loop)
        objective_wrapper_alpha(
            alpha=alpha_init, counts=counts, X=X.flatten(), lengths=lengths,
            bias=bias, constraints=constraints, reorienter=reorienter,
            multiscale_factor=multiscale_factor,
            multiscale_variances=multiscale_variances,
            mixture_coefs=mixture_coefs, callback=callback)

    results = optimize.fmin_l_bfgs_b(
        objective_wrapper_alpha,
        x0=np.float64(alpha_init),
        fprime=fprime_wrapper_alpha,
        iprint=0,
        maxiter=max_iter,
        pgtol=pgtol,
        factr=factr,
        bounds=np.array([[-100, 1e-2]]),
        args=(counts, X.flatten(), lengths, bias, constraints,
              reorienter, multiscale_factor, multiscale_variances,
              mixture_coefs, callback))

    if callback is not None:
        callback.on_training_end()

    alpha, obj, d = results
    converged = d['warnflag'] == 0

    if verbose:
        if converged:
            print('CONVERGED\n\n', flush=True)
        else:
            print('OPTIMIZATION DID NOT CONVERGE', flush=True)
            print(d['task'].decode('utf8') + '\n\n', flush=True)
        print('INIT ALPHA: %.3g, FINAL ALPHA: %.3g' %
              (alpha_init, alpha), flush=True)

    return float(alpha), obj, converged, callback.history
