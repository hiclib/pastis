import numpy as np
from scipy import optimize
import warnings
from autograd import grad
from autograd.builtins import SequenceBox
from .poisson_diploid import format_X, objective


def _estimate_beta_single(structures, counts, lengths, alpha=-3, bias=None,
                          multiscale_factor=1, mixture_coefs=None):
    """Facilitates estimation of beta for a single counts object.

    Computes the sum of distances (K) corresponding to a given counts matrix.
    """

    n, m = counts.shape

    if mixture_coefs is not None and len(structures) != len(mixture_coefs):
        raise ValueError("The number of structures (%d) and of mixture "
                         "coefficents (%d) should be identical." %
                         (len(structures), len(mixture_coefs)))
    elif mixture_coefs is None:
        mixture_coefs = [1.]

    ploidy = int(structures[0].shape[0] / lengths.sum())

    K = 0
    for X, gamma in zip(structures, mixture_coefs):
        dis = ((X[counts.row3d] - X[counts.col3d]) ** 2).sum(axis=1) ** 0.5
        tmp = dis ** alpha
        tmp = tmp.reshape(-1, counts.nnz).sum(axis=0)
        K += (gamma * counts.bias_per_bin(bias, ploidy) * tmp).sum()

    return K


def _estimate_beta(X, counts, alpha=-3, bias=None, lengths=None,
                   reorienter=None, multiscale_factor=1, mixture_coefs=None,
                   verbose=False):
    """Estimates beta for all counts matrices.
    """

    structures, mixture_coefs = format_X(X, reorienter, mixture_coefs)
    if reorienter is not None and reorienter.reorient:
        structures = reorienter.translate_and_rotate(X)

    # Check format of input
    counts = (counts if isinstance(counts, list) else [counts])
    lengths = (np.array([min([min(counts_maps.shape)
                              for counts_maps in counts])]) if lengths is None else lengths)
    bias = (np.ones((min([min(counts_maps.shape)
                          for counts_maps in counts]),)) if bias is None else bias)
    if not (isinstance(structures, list) or isinstance(structures, SequenceBox)):
        structures = [structures]
    if len(structures) > 1:
        raise NotImplementedError('Can only handle 1 structure right now')
    if mixture_coefs is None:
        mixture_coefs = [1.] * len(structures)

    # Estimate beta for each structure
    K = {counts_maps.ambiguity: 0. for counts_maps in counts}
    for counts_maps in counts:
        K[counts_maps.ambiguity] += _estimate_beta_single(structures,
                                                          counts_maps, lengths,
                                                          alpha=alpha,
                                                          bias=bias,
                                                          multiscale_factor=multiscale_factor,
                                                          mixture_coefs=mixture_coefs)
    for i in range(len(counts)):
        beta_maps = counts[i].input_sum / K[counts[i].ambiguity]
        if np.isnan(beta_maps):
            raise ValueError("Inferred beta is not a number.")
        elif np.isinf(beta_maps):
            raise ValueError("Inferred beta is infinite.")
        elif beta_maps == 0:
            raise ValueError("Inferred beta is zero.")
        # Assign new beta to counts object
        counts[i].beta = beta_maps

    if verbose:
        print('INFERRED BETA: %s' % ', '.join(['%s=%.2g' % (
            counts_maps.name, counts_maps.beta) for counts_maps in counts]), flush=True)

    return counts


def objective_alpha(alpha, counts, X, bias=None, lengths=None, constraints=None,
                    reorienter=None, multiscale_factor=1, mixture_coefs=None,
                    modifications=None, return_components=False):
    """Compute the objective.
    """

    return objective(X, counts, alpha=alpha, bias=bias, lengths=lengths,
                     constraints=constraints, reorienter=reorienter,
                     multiscale_factor=multiscale_factor,
                     mixture_coefs=mixture_coefs, modifications=modifications,
                     return_components=return_components)


gradient_alpha = grad(objective_alpha)


def objective_wrapper_alpha(alpha, counts, X, bias=None, lengths=None,
                            constraints=None, reorienter=None,
                            multiscale_factor=1, mixture_coefs=None,
                            modifications=None, callback=None):
    """
    Objective function wrapper to match scipy.optimize's interface.
    """

    counts = _estimate_beta(X, counts, alpha=alpha, bias=bias, lengths=lengths,
                            reorienter=reorienter,
                            multiscale_factor=multiscale_factor,
                            mixture_coefs=mixture_coefs)

    X, mixture_coefs = format_X(X, reorienter, mixture_coefs)

    new_obj, obj_logs, structures, alpha = objective_alpha(alpha, counts, X=X,
                                                           bias=bias,
                                                           lengths=lengths,
                                                           constraints=constraints,
                                                           reorienter=reorienter,
                                                           multiscale_factor=multiscale_factor,
                                                           mixture_coefs=mixture_coefs,
                                                           modifications=modifications,
                                                           return_components=True)

    if callback is not None:
        callback.on_epoch_end(obj_logs, structures, alpha, X)

    return new_obj


def fprime_wrapper_alpha(alpha, counts, X, bias=None, lengths=None,
                         constraints=None, reorienter=None, multiscale_factor=1,
                         mixture_coefs=None, modifications=None, callback=None):
    """
    Gradient function wrapper to match scipy.optimize's interface.
    """

    counts = _estimate_beta(X, counts, alpha=alpha, bias=bias, lengths=lengths,
                            reorienter=reorienter,
                            multiscale_factor=multiscale_factor,
                            mixture_coefs=mixture_coefs)

    X, mixture_coefs = format_X(X, reorienter, mixture_coefs)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message='Using a non-tuple sequence for multidimensional'
                              'indexing is deprecated', category=FutureWarning)
        new_grad = np.array(gradient_alpha(alpha, counts, X=X, bias=bias,
                                           lengths=lengths,
                                           constraints=constraints,
                                           reorienter=reorienter,
                                           multiscale_factor=multiscale_factor,
                                           mixture_coefs=mixture_coefs,
                                           modifications=modifications)).flatten()

    return new_grad


def estimate_alpha(counts, X, alpha_init=-3., ploidy=1, bias=None, verbose=0,
                   lengths=None, constraints=None, callback=None, X_true=None,
                   multiscale_factor=1, multiscale_variances=None,
                   mixture_coefs=None, random_state=None, reorienter=None,
                   alpha_true=None, alpha_loop=None, modifications=None,
                   max_iter=10000000000, pgtol=1e-05, factr=10000000.0):
    """
    Estimates alpha.

    Parameters
    ----------
    """

    # Check format of input
    counts = (counts if isinstance(counts, list) else [counts])
    if lengths is None:
        if ploidy != 1:
            raise ValueError("Must supply lengths for diploid inference")
        lengths = [min([min(counts_maps.shape) for counts_maps in counts])]
    lengths = np.array(lengths)
    bias = (np.ones((min([min(counts_maps.shape)
                          for counts_maps in counts]),)) if bias is None else bias)

    # Initialize alpha if necessary
    if alpha_init is None:
        alpha_init = (- random_state.randint(1, 100) + random_state.rand(1))[0]

    if verbose:
        print('=' * 30, flush=True)
        print('\nRUNNING THE L-BFGS-B CODE\n\n           * * *\n\n'
              'Machine precision = %.4g\n' % np.finfo(np.float).eps, flush=True)

    if callback is not None:
        if reorienter is not None and reorienter.reorient:
            opt_type = 'alpha.chrom_reorient'
        else:
            opt_type = 'alpha'
        callback.on_training_begin(opt_type=opt_type, alpha_loop=alpha_loop)
        objective_wrapper_alpha(alpha_init, counts, X.flatten(), bias, lengths,
                                constraints, reorienter, multiscale_factor,
                                multiscale_variances, mixture_coefs,
                                modifications, callback)

    results = optimize.fmin_l_bfgs_b(
        objective_wrapper_alpha,
        x0=np.float64(alpha_init),
        fprime=fprime_wrapper_alpha,
        iprint=0,
        maxiter=max_iter,
        pgtol=pgtol,
        factr=factr,
        bounds=np.array([[-100, 1e-2]]),
        args=(counts, X.flatten(), bias, lengths, constraints,
              reorienter, multiscale_factor, multiscale_variances,
              mixture_coefs, modifications, callback))

    if callback is not None:
        history = callback.on_training_end()

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

    return float(alpha), obj, converged, history
