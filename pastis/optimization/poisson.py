import numpy as np
from scipy import optimize
import warnings
import autograd.numpy as ag_np
from autograd.builtins import SequenceBox
from autograd import grad
from .multiscale_optimization import decrease_lengths_res


def _poisson_obj_single(structures, counts, alpha, lengths, bias=None,
                        multiscale_factor=1, multiscale_variances=None,
                        mixture_coefs=None):
    """Computes the poisson objective function for each counts matrix.
    """

    if (bias is not None and bias.sum() == 0) or counts.nnz == 0:
        return 0.

    if mixture_coefs is not None and len(structures) != len(mixture_coefs):
        raise ValueError("The number of structures (%d) and of mixture"
                         " coefficents (%d) should be identical." %
                         (len(structures), len(mixture_coefs)))
    elif mixture_coefs is None:
        mixture_coefs = [1.]

    lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)
    ploidy = int(structures[0].shape[0] / lengths_lowres.sum())

    if multiscale_variances is not None:
        if isinstance(multiscale_variances, np.ndarray):
            var_per_dis = multiscale_variances[
                counts.row3d] + multiscale_variances[counts.col3d]
        else:
            var_per_dis = multiscale_variances * 2
    else:
        var_per_dis = 0
    num_highres_per_lowres_bins = counts.count_fullres_per_lowres_bins(
        multiscale_factor)

    lambda_intensity = ag_np.zeros(counts.nnz)
    for struct, gamma in zip(structures, mixture_coefs):
        dis = ag_np.sqrt((ag_np.square(
            struct[counts.row3d] - struct[counts.col3d])).sum(axis=1))
        if multiscale_variances is None:
            tmp1 = ag_np.power(dis, alpha)
        else:
            tmp1 = ag_np.power(ag_np.square(dis) + var_per_dis, alpha / 2)
        tmp = tmp1.reshape(-1, counts.nnz).sum(axis=0)
        lambda_intensity = lambda_intensity + gamma * counts.bias_per_bin(
            bias, ploidy) * counts.beta * num_highres_per_lowres_bins * tmp

    # Sum main objective function
    obj = lambda_intensity.sum() - (counts.data * ag_np.log(
        lambda_intensity)).sum()

    if ag_np.isnan(obj):
        raise ValueError("Poisson component of objective function is nan")
    elif ag_np.isinf(obj):
        raise ValueError("Poisson component of objective function is infinite")

    return counts.weight * obj


def objective(X, counts, alpha, lengths, bias=None, constraints=None,
              reorienter=None, multiscale_factor=1, multiscale_variances=None,
              mixture_coefs=None, return_extras=False):
    """
    Computes the objective function.

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

    # Optionally translate & rotate structures
    if reorienter is not None and reorienter.reorient:
        structures = reorienter.translate_and_rotate(X)
    else:
        structures = X

    # Check format of input
    counts = (counts if isinstance(counts, list) else [counts])
    if lengths is None:
        lengths = np.array([min([min(counts_maps.shape)
                                 for counts_maps in counts])])
    lengths = np.array(lengths)
    if bias is None:
        bias = np.ones((min([min(counts_maps.shape)
                             for counts_maps in counts]),))
    if not (isinstance(structures, list) or isinstance(structures, SequenceBox)):
        structures = [structures]
    if mixture_coefs is None:
        mixture_coefs = [1.] * len(structures)

    if constraints is None:
        obj_constraints = {}
    else:
        obj_constraints = constraints.apply(structures, mixture_coefs)
    obj_poisson = {}
    for counts_maps in counts:
        obj_poisson['obj_' + counts_maps.name] = _poisson_obj_single(
            structures, counts_maps, alpha=alpha, lengths=lengths, bias=bias,
            multiscale_factor=multiscale_factor,
            multiscale_variances=multiscale_variances,
            mixture_coefs=mixture_coefs)
    obj_poisson_sum = sum(obj_poisson.values())
    obj = obj_poisson_sum + sum(obj_constraints.values())

    if return_extras:
        obj_logs = {**obj_poisson, **obj_constraints, **{'obj': obj, 'obj_poisson': obj_poisson_sum}}
        return obj, obj_logs, structures, alpha
    else:
        return obj


def _format_X(X, reorienter=None, mixture_coefs=None):
    """Reformat and check X.
    """

    if mixture_coefs is None:
        mixture_coefs = [1]

    if reorienter is not None and reorienter.reorient:
        reorienter.check_format(X, mixture_coefs)
    else:
        try:
            X = X.reshape(-1, 3)
        except ValueError:
            raise ValueError("X should contain k 3D structures")
        k = len(mixture_coefs)
        n = int(X.shape[0] / k)
        X = [X[i * n:(i + 1) * n] for i in range(k)]

    return X, mixture_coefs


def objective_wrapper(X, counts, alpha, lengths, bias=None, constraints=None,
                      reorienter=None, multiscale_factor=1,
                      multiscale_variances=None, mixture_coefs=None,
                      callback=None):
    """Objective function wrapper to match scipy.optimize's interface
    """

    X, mixture_coefs = _format_X(X, reorienter, mixture_coefs)

    new_obj, obj_logs, structures, alpha = objective(
        X, counts=counts, alpha=alpha, lengths=lengths, bias=bias,
        constraints=constraints, reorienter=reorienter,
        multiscale_factor=multiscale_factor,
        multiscale_variances=multiscale_variances, mixture_coefs=mixture_coefs,
        return_extras=True)

    if callback is not None:
        callback.on_epoch_end(obj_logs, structures, alpha, X)

    return new_obj


gradient = grad(objective)


def fprime_wrapper(X, counts, alpha, lengths, bias=None, constraints=None,
                   reorienter=None, multiscale_factor=1,
                   multiscale_variances=None, mixture_coefs=None,
                   callback=None):
    """Gradient function wrapper to match scipy.optimize's interface.
    """

    X, mixture_coefs = _format_X(X, reorienter, mixture_coefs)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Using a non-tuple sequence for multidimensional"
            " indexing is deprecated", category=FutureWarning)
        new_grad = np.array(gradient(
            X, counts=counts, alpha=alpha, lengths=lengths, bias=bias,
            constraints=constraints, reorienter=reorienter,
            multiscale_factor=multiscale_factor,
            multiscale_variances=multiscale_variances,
            mixture_coefs=mixture_coefs)).flatten()

    return new_grad


def estimate_X(counts, init_X, alpha, lengths, ploidy, bias=None,
               constraints=None, multiscale_factor=1, multiscale_variances=None,
               max_iter=10000000000, factr=10000000.0, pgtol=1e-05,
               callback=None, alpha_loop=None, reorienter=None,
               mixture_coefs=None, verbose=True):
    """Estimates a 3D structure, given current alpha.

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

    # Check format of input
    counts = (counts if isinstance(counts, list) else [counts])
    if lengths is None:
        if ploidy != 1:
            raise ValueError("Must supply lengths for diploid inference")
        lengths = [min([min(counts_maps.shape) for counts_maps in counts])]
    lengths = np.array(lengths)
    if bias is None:
        bias = np.ones((min([min(counts_maps.shape)
                             for counts_maps in counts]),))

    if verbose:
        print('=' * 30, flush=True)
        print("\nRUNNING THE L-BFGS-B CODE\n\n           * * *\n\nMachine"
              " precision = %.4g\n" % np.finfo(np.float).eps, flush=True)

    if callback is not None:
        if reorienter is not None and reorienter.reorient:
            opt_type = 'chrom_reorient'
        else:
            opt_type = 'structure'
        callback.on_training_begin(opt_type=opt_type, alpha_loop=alpha_loop)
        objective_wrapper(
            init_X.flatten(), counts=counts, alpha=alpha, lengths=lengths,
            bias=bias, constraints=constraints, reorienter=reorienter,
            multiscale_factor=multiscale_factor,
            multiscale_variances=multiscale_variances,
            mixture_coefs=mixture_coefs, callback=callback)

    results = optimize.fmin_l_bfgs_b(
        objective_wrapper,
        x0=init_X.flatten(),
        fprime=fprime_wrapper,
        iprint=0,
        maxiter=max_iter,
        pgtol=pgtol,
        factr=factr,
        args=(counts, alpha, lengths, bias, constraints,
              reorienter, multiscale_factor, multiscale_variances,
              mixture_coefs, callback))

    if callback is not None:
        callback.on_training_end()

    X, obj, d = results
    converged = d['warnflag'] == 0

    if verbose:
        if converged:
            print('CONVERGED\n\n', flush=True)
        else:
            print('OPTIMIZATION DID NOT CONVERGE', flush=True)
            print(d['task'].decode('utf8') + '\n\n', flush=True)

    return X, obj, converged, callback.history


def _convergence_criteria(f_k, f_kplus1, factr=10000000.0):
    """Convergence criteria for joint inference of alpha & structure.
    """
    if f_k is None:
        return False
    else:
        return np.abs(f_k - f_kplus1) / max(np.abs(f_k), np.abs(
            f_kplus1), 1) <= factr * np.finfo(float).eps


class PastisPM(object):
    """Infer 3D structures with PASTIS.

    null - Estimates a "null" structure that fullfills the constraints but is not fitted to the data
    """

    def __init__(self, counts, lengths, ploidy, alpha, beta=1., init=None,
                 bias=None, constraints=None, callback=None,
                 multiscale_factor=1, multiscale_variances=None, alpha_init=-3.,
                 max_alpha_loop=20, max_iter=1e15, factr=10000000.0, pgtol=1e-05,
                 alpha_factr=1000000000000., reorienter=None, null=False,
                 mixture_coefs=None, verbose=True):
        from .constraints import Constraints
        from .callbacks import Callback
        from .stepwise_whole_genome import ChromReorienter

        print('%s\n%s 3D STRUCTURAL INFERENCE' %
              ('=' * 30, {2: 'DIPLOID', 1: 'HAPLOID'}[ploidy]), flush=True)

        if constraints is None:
            constraints = Constraints(
                counts=counts, lengths=lengths, ploidy=ploidy,
                multiscale_factor=multiscale_factor)
        if callback is None:
            callback = Callback(
                lengths=lengths, ploidy=ploidy, counts=counts,
                multiscale_factor=multiscale_factor,
                frequency={'print': 100, 'history': 100, 'save': None})
        if reorienter is None:
            reorienter = ChromReorienter(lengths=lengths, ploidy=ploidy)

        self.counts = counts
        self.lengths = lengths
        self.ploidy = ploidy
        self.alpha = alpha
        self.beta = beta
        self.init_X = init
        self.bias = bias
        self.constraints = constraints
        self.callback = callback
        self.multiscale_factor = multiscale_factor
        self.multiscale_variances = multiscale_variances
        self.alpha_init = alpha_init
        self.max_alpha_loop = max_alpha_loop
        self.max_iter = max_iter
        self.factr = factr
        self.pgtol = pgtol
        self.alpha_factr = alpha_factr
        self.reorienter = reorienter
        self.null = null
        self.mixture_coefs = mixture_coefs
        self.verbose = verbose

        self.X_ = None
        self.alpha_ = None
        self.beta_ = None
        self.obj_ = None
        self.converged_ = None
        self.history_ = None
        self.struct_ = None
        self.orientation_ = None

    def _infer_beta(self, verbose=True):
        """Estimate beta, given current structure and alpha.
        """

        from .estimate_alpha_beta import _estimate_beta

        self.counts = _estimate_beta(
            self.X_.flatten(), self.counts, alpha=self.alpha_, bias=self.bias,
            lengths=self.lengths, mixture_coefs=self.mixture_coefs,
            reorienter=self.reorienter, verbose=verbose)
        return [c.beta for c in self.counts if c.sum() != 0]

    def _fit_structure(self, alpha_loop=None):
        """Fit structure to counts data, given current alpha.
        """

        self.X_, self.obj_, self.converged_, history_ = estimate_X(
            counts=self.counts,
            init_X=self.X_.flatten(),
            alpha=self.alpha_,
            lengths=self.lengths,
            ploidy=self.ploidy,
            bias=self.bias,
            constraints=self.constraints,
            multiscale_factor=self.multiscale_factor,
            multiscale_variances=self.multiscale_variances,
            max_iter=self.max_iter,
            factr=self.factr,
            pgtol=self.pgtol,
            callback=self.callback,
            alpha_loop=alpha_loop,
            reorienter=self.reorienter,
            mixture_coefs=self.mixture_coefs,
            verbose=self.verbose)

        self.history_.extend(history_)

    def _fit_alpha(self, alpha_loop=None):
        """Fit alpha to counts data, given current structure.
        """

        from .estimate_alpha_beta import estimate_alpha

        self.alpha_, self.alpha_obj_, self.converged_, history_ = estimate_alpha(
            counts=self.counts,
            X=self.X_.flatten(),
            alpha_init=self.alpha_,
            lengths=self.lengths,
            ploidy=self.ploidy,
            bias=self.bias,
            constraints=self.constraints,
            multiscale_factor=self.multiscale_factor,
            multiscale_variances=self.multiscale_variances,
            random_state=None,
            max_iter=self.max_iter,
            factr=self.factr,
            pgtol=self.pgtol,
            callback=self.callback,
            alpha_loop=alpha_loop,
            reorienter=self.reorienter,
            mixture_coefs=self.mixture_coefs,
            verbose=self.verbose)

        self.history_.extend(history_)

    def fit(self):
        """ Fit structure to counts data, optionally estimate alpha
        """

        from timeit import default_timer as timer
        from datetime import timedelta
        from .counts import NullCountsMatrix

        if self.null:
            print('GENERATING NULL STRUCTURE', flush=True)
            # Dummy counts need to be inputted because we need to know which
            # row/col to include in calculations of constraints
            self.counts = [NullCountsMatrix(
                counts=self.counts, lengths=self.lengths, ploidy=self.ploidy,
                multiscale_factor=self.multiscale_factor)]

        self.X_ = self.init_X
        if self.alpha is not None:
            self.alpha_ = self.alpha
        else:
            self.alpha_ = self.alpha_init
        if self.beta is None:
            self.beta_ = self._infer_beta()
        else:
            self.beta_ = self.beta

        # Infer structure
        self.history_ = []
        time_start = timer()
        if self.alpha is not None or self.multiscale_factor != 1:
            self._fit_structure()
        else:
            print("JOINTLY INFERRING STRUCTURE + ALPHA: inferring structure,"
                  " with initial guess of alpha=%.3g"
                  % self.alpha_init, flush=True)
            self._fit_structure(alpha_loop=0)
            prev_alpha_obj = None
            if self.converged:
                for alpha_loop in range(1, self.max_alpha_loop + 1):
                    time_current = str(
                        timedelta(seconds=round(timer() - time_start)))
                    print("JOINTLY INFERRING STRUCTURE + ALPHA (#%d):"
                          " inferring alpha, total elapsed time=%s" %
                          (alpha_loop, time_current), flush=True)
                    self._fit_alpha(alpha_loop=alpha_loop)
                    self.beta_ = self._infer_beta()
                    if not self.converged:
                        break
                    time_current = str(
                        timedelta(seconds=round(timer() - time_start)))
                    print("JOINTLY INFERRING STRUCTURE + ALPHA (#%d): inferring"
                          " structure, total elapsed time=%s" %
                          (alpha_loop, time_current), flush=True)
                    self._fit_structure(alpha_loop=alpha_loop)
                    if not self.converged:
                        break
                    if _convergence_criteria(
                            f_k=prev_alpha_obj, f_kplus1=self.alpha_obj_,
                            factr=self.alpha_factr):
                        break
                    prev_alpha_obj = self.alpha_obj_
        time_current = str(timedelta(seconds=round(timer() - time_start)))
        print("OPTIMIZATION AT %dX RESOLUTION COMPLETE, TOTAL ELAPSED TIME=%s" %
              (self.multiscale_factor, time_current), flush=True)

        if self.callback is None or self.callback.frequency['history'] is None or self.callback.frequency['history'] == 0:
            self.history_ = None

        if self.reorienter.reorient:
            self.orientation_ = self.X_
            self.struct_ = self.reorienter.translate_and_rotate(self.X_)[
                0].reshape(-1, 3)
        else:
            self.struct_ = self.X_.reshape(-1, 3)
