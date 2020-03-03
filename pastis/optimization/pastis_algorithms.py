from __future__ import print_function

import sys

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

import os
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from .utils_poisson import _print_code_header, _load_infer_var
from .utils_poisson import _output_subdir
from .counts import preprocess_counts, ambiguate_counts
from .counts import _update_betas_in_counts_matrices, check_counts
from .initialization import initialize
from .callbacks import Callback
from .constraints import Constraints, distance_between_homologs
from .poisson import PastisPM
from .estimate_alpha_beta import _estimate_beta
from .multiscale_optimization import get_multiscale_variances_from_struct
from .multiscale_optimization import _choose_max_multiscale_factor
from .multiscale_optimization import decrease_lengths_res, decrease_struct_res
from ..io.read import load_data
from .poisson import objective


def _infer_draft(counts_raw, lengths, ploidy, simple_diploid_init=None,
                 outdir=None, alpha=None, seed=0, normalize=True,
                 filter_threshold=0.04, alpha_init=-3., max_alpha_loop=20,
                 beta=None, multiscale_rounds=1, use_multiscale_variance=True,
                 init='mds', max_iter=10000000000,
                 factr=10000000., pgtol=1e-05, alpha_factr=1000000000000.,
                 bcc_lambda=0., hsc_lambda=0., hsc_r=None, hsc_min_beads=5,
                 fullres_torm=None, struct_draft_fullres=None,
                 callback_freq=None, callback_function=None, reorienter=None,
                 alpha_true=None, struct_true=None, input_weight=None,
                 exclude_zeros=False, null=False, mixture_coefs=None,
                 verbose=True):
    """Infer draft 3D structures with PASTIS via Poisson model.
    """

    alpha_ = alpha
    beta_ = beta
    if struct_draft_fullres is None and ((
            multiscale_rounds > 1 and use_multiscale_variance) or alpha is None):
        if verbose:
            _print_code_header(
                "Inferring full-res draft structure",
                max_length=50, blank_lines=1)
        if outdir is None:
            fullres_outdir = None
        else:
            fullres_outdir = os.path.join(outdir, 'struct_draft_fullres')
        struct_draft_fullres, infer_var_fullres = infer(
            counts_raw=counts_raw, outdir=fullres_outdir, lengths=lengths,
            ploidy=ploidy, alpha=alpha, seed=seed, normalize=normalize,
            filter_threshold=filter_threshold, alpha_init=alpha_init,
            max_alpha_loop=max_alpha_loop, beta=beta, init=init,
            max_iter=max_iter, factr=factr, pgtol=pgtol,
            alpha_factr=alpha_factr, draft=True, simple_diploid=(ploidy == 2),
            simple_diploid_init=simple_diploid_init,
            callback_function=callback_function, callback_freq=callback_freq,
            reorienter=reorienter, alpha_true=alpha_true,
            struct_true=struct_true, input_weight=input_weight,
            exclude_zeros=exclude_zeros, null=null, mixture_coefs=mixture_coefs,
            verbose=verbose)
        if not infer_var_fullres['converged']:
            return struct_draft_fullres, alpha_, beta_, hsc_r, False
        alpha_ = infer_var_fullres['alpha']
        beta_ = infer_var_fullres['beta']

    if hsc_lambda > 0 and hsc_r is None:
        if verbose:
            _print_code_header(
                "Inferring low-res draft structure for `hsc_r` estimation",
                max_length=50, blank_lines=1)
        if ploidy == 1:
            raise ValueError("Can not apply homolog-separating constraint"
                             " to haploid data.")
        if alpha_ is None:
            raise ValueError("Alpha must be set prior to inferring r from"
                             " counts data")
        if outdir is None:
            lowres_outdir = None
        else:
            lowres_outdir = os.path.join(outdir, 'struct_draft_lowres')
        ua_index = [i for i in range(len(
            counts_raw)) if counts_raw[i].shape == (lengths.sum() * ploidy,
                                                    lengths.sum() * ploidy)]
        if len(ua_index) == 1:
            counts_for_lowres = counts_raw[ua_index[0]]
            simple_diploid_for_lowres = False
            fullres_torm = fullres_torm[ua_index[0]]
        elif len(ua_index) > 1:
            raise ValueError("Only input one matrix of unambiguous counts."
                             " Please pool unambiguos counts before"
                             " inputting.")
        else:
            if lengths.shape[0] == 1:
                raise ValueError("Please input more than one chromosome to"
                                 " estimate hsc_r from ambiguous data.")
            counts_for_lowres = counts_raw
            simple_diploid_for_lowres = True
        multiscale_factor_for_lowres = _choose_max_multiscale_factor(
            lengths=lengths, min_beads=hsc_min_beads)
        struct_draft_lowres, infer_var_lowres = infer(
            counts_raw=counts_for_lowres, outdir=lowres_outdir,
            lengths=lengths, ploidy=ploidy, alpha=alpha_,
            seed=seed, normalize=normalize,
            filter_threshold=filter_threshold, beta=beta_,
            multiscale_factor=multiscale_factor_for_lowres,
            use_multiscale_variance=use_multiscale_variance,
            init=init, max_iter=max_iter, factr=factr, pgtol=pgtol,
            fullres_torm=fullres_torm,
            struct_draft_fullres=struct_draft_fullres, draft=True,
            simple_diploid=simple_diploid_for_lowres,
            simple_diploid_init=simple_diploid_init,
            callback_function=callback_function,
            callback_freq=callback_freq,
            reorienter=reorienter, alpha_true=alpha_true,
            struct_true=struct_true, input_weight=input_weight,
            exclude_zeros=exclude_zeros, null=null,
            mixture_coefs=mixture_coefs, verbose=verbose)
        if not infer_var_lowres['converged']:
            return struct_draft_fullres, alpha_, beta_, hsc_r, False
        hsc_r = distance_between_homologs(
            structures=struct_draft_lowres,
            lengths=decrease_lengths_res(
                lengths=lengths,
                multiscale_factor=multiscale_factor_for_lowres),
            ploidy=ploidy, mixture_coefs=mixture_coefs,
            simple_diploid=simple_diploid_for_lowres)
        if verbose:
            print("Estimated distance between homolog barycenters for each"
                  " chromosome: %s" % ' '.join(map(str, hsc_r.round(2))),
                  flush=True)

    return struct_draft_fullres, alpha_, beta_, hsc_r, True


def infer(counts_raw, lengths, ploidy, outdir='', alpha=None, seed=0,
          normalize=True, filter_threshold=0.04, alpha_init=-3.,
          max_alpha_loop=20, beta=None, multiscale_factor=1,
          multiscale_rounds=1, use_multiscale_variance=True,
          final_multiscale_round=False, init='mds', max_iter=10000000000,
          factr=10000000., pgtol=1e-05, alpha_factr=1000000000000.,
          bcc_lambda=0., hsc_lambda=0., hsc_r=None, hsc_min_beads=5,
          fullres_torm=None, struct_draft_fullres=None, draft=False,
          simple_diploid=False, simple_diploid_init=None,
          callback_freq=None, callback_function=None, reorienter=None,
          alpha_true=None, struct_true=None, input_weight=None,
          exclude_zeros=False, null=False, mixture_coefs=None, verbose=True):
    """Infer 3D structures with PASTIS via Poisson model.

    Optimize 3D structure from Hi-C contact counts data for diploid
    organisms. Optionally perform multiscale optimization during inference.

    Parameters
    ----------
    counts_raw : list of array or coo_matrix
        Counts data without normalization or filtering.
    lengths : array_like of int
        Number of beads per homolog of each chromosome.
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    outdir : str, optional
        Directory in which to save results.
    alpha : float, optional
        Biophysical parameter of the transfer function used in converting
        counts to wish distances. If alpha is not specified, it will be
        inferred.
    seed : int, optional
        Random seed used when generating the starting point in the
        optimization.
    normalize : bool, optional
        Perform ICE normalization on the counts prior to optimization.
        Normalization is reccomended.
    filter_threshold : float, optional
        Ratio of non-zero beads to be filtered out. Filtering is
        reccomended.
    alpha_init : float, optional
        For PM2, the initial value of alpha to use.
    max_alpha_loop : int, optional
        For PM2, Number of times alpha and structure are inferred.
    beta : array_like of float, optional
        Scaling parameter that determines the size of the structure, relative to
        each counts matrix. There should be one beta per counts matrix. If None,
        the optimal beta will be estimated.
    multiscale_factor : int, optional
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 indicates full resolution.
    multiscale_rounds : int, optional
        The number of resolutions at which a structure should be inferred
        during multiscale optimization. Values of 1 or 0 disable multiscale
        optimization.
    final_multiscale_round : bool, optional
        Whether this is the final (full-resolution) round of multiscale
        optimization.
    init : optional
        Method by which to initialize the structure.
    max_iter : int, optional
        Maximum number of iterations per optimization.
    factr : float, optional
        factr for scipy's L-BFGS-B, alters convergence criteria.
    pgtol : float, optional
        pgtol for scipy's L-BFGS-B, alters convergence criteria.
    alpha_factr : float, optional
        factr for convergence criteria of joint alpha/structure inference.
    bcc_lambda : float, optional
        Lambda of the bead chain connectivity constraint.
    hsc_lambda : float, optional
        For diploid organisms: lambda of the homolog-separating
        constraint.
    hsc_r : list of float, optional
        For diploid organisms: hyperparameter of the homolog-separating
        constraint specificying the expected distance between homolog
        centers of mass for each chromosome. If not supplied, `hsc_r` will
        be inferred from the counts data.
    hsc_min_beads : int, optional
        For diploid organisms: number of beads in the low-resolution
        structure from which `hsc_r` is estimated.
    fullres_torm : list of array of bool, optional
        For multiscale optimization, this indicates which beads of the full-
        resolution structure do not correspond to any counts data, and should
        therefore be removed. There should be one array per counts matrix.
    struct_draft_fullres : np.ndarray, optional
        The full-resolution draft structure from whihc to derive multiscale
        variances.
    draft: bool, optional
        Whether this optimization is inferring a draft structure.
    simple_diploid: bool, optional
        For diploid organisms: whether this optimization is inferring a "simple
        diploid" structure in which homologs are assumed to be identical and
        completely overlapping with one another.
    simple_diploid_init : np.ndarray, optional
        For diploid organisms: structure to be used for beta estimation when
        `simple_diploid` is True.

    Returns
    -------
    struct_ : array_like of float of shape (lengths.sum() * ploidy, 3)
        3D structure resulting from the optimization.
    infer_var : dict
        A few of the variables used in inference or generated by inference.
        Keys: 'alpha', 'beta', 'hsc_r', 'obj', and 'seed'.
    """

    if outdir is not None:
        try:
            os.makedirs(outdir)
        except OSError:
            pass
        if seed is None:
            seed_str = ''
        else:
            seed_str = '.%03d' % seed
        out_file = os.path.join(outdir, 'struct_inferred%s.coords' % seed_str)
        orient_file = os.path.join(
            outdir, 'orient_inferred%s.coords' % seed_str)
        history_file = os.path.join(outdir, 'history%s' % seed_str)
        infer_var_file = os.path.join(
            outdir, 'inference_variables%s' % seed_str)
        out_fail = os.path.join(
            outdir, 'struct_nonconverged%s.coords' % seed_str)

        if os.path.exists(out_file) or os.path.exists(out_fail):
            if os.path.exists(out_file):
                print('CONVERGED', flush=True)
            elif os.path.exists(out_fail):
                print('OPTIMIZATION DID NOT CONVERGE', flush=True)
            infer_var = _load_infer_var(infer_var_file)
            struct_ = np.loadtxt(out_file)
            return struct_, infer_var

    random_state = np.random.RandomState(seed)
    random_state = check_random_state(random_state)

    # PREPARE COUNTS OBJECTS
    counts_raw = check_counts(
        counts_raw, lengths=lengths, ploidy=ploidy, exclude_zeros=exclude_zeros)
    if simple_diploid:
        counts_raw = [ambiguate_counts(
            counts=counts_raw, lengths=lengths, ploidy=ploidy,
            exclude_zeros=exclude_zeros)]
        ploidy = 1
    counts, bias, torm, fullres_torm_for_multiscale = preprocess_counts(
        counts_raw=counts_raw, lengths=lengths, ploidy=ploidy, normalize=normalize,
        filter_threshold=filter_threshold, multiscale_factor=multiscale_factor,
        exclude_zeros=exclude_zeros, beta=beta, input_weight=input_weight,
        verbose=verbose, fullres_torm=fullres_torm, mixture_coefs=mixture_coefs)
    if simple_diploid:
        if simple_diploid_init is None:
            raise ValueError("Must provide simple_diploid_init.")
        simple_diploid_init = decrease_struct_res(
            simple_diploid_init, multiscale_factor=multiscale_factor,
            lengths=lengths)
        counts = _estimate_beta(
            simple_diploid_init, counts=counts, alpha=alpha, bias=bias,
            lengths=lengths, reorienter=reorienter, mixture_coefs=mixture_coefs,
            verbose=verbose, simple_diploid=True)

    # SIMPlE DIPLOID
    if simple_diploid and struct_true is not None:
        struct_true = np.mean(
            [struct_true[:lengths.sum()], struct_true[lengths.sum():]], axis=0)

    # INITIALIZATION
    if isinstance(init, str) and init.lower() == 'true':
        if struct_true is None:
            raise ValueError("Attempting to initialize with struct_true but"
                             " struct_true is None")
        print('INITIALIZATION: initializing with true structure', flush=True)
        init = struct_true
    struct_init = initialize(
        counts=counts, lengths=lengths, init=init, ploidy=ploidy,
        random_state=random_state, alpha=alpha, bias=bias,
        multiscale_factor=multiscale_factor, reorienter=reorienter,
        mixture_coefs=mixture_coefs, verbose=verbose)

    # MULTISCALE VARIANCES
    if multiscale_factor != 1 and use_multiscale_variance and struct_draft_fullres is not None:
        multiscale_variances = np.median(get_multiscale_variances_from_struct(
            struct_draft_fullres, lengths=lengths,
            multiscale_factor=multiscale_factor, ploidy=ploidy,
            mixture_coefs=mixture_coefs))
    else:
        multiscale_variances = None

    # HOMOLOG-SEPARATING CONSTRAINT
    if hsc_lambda > 0:
        if ploidy == 1:
            raise ValueError("Can not apply homolog-separating constraint to"
                             " haploid data.")
        if hsc_r is not None:
            hsc_r = np.array(hsc_r, dtype=float).reshape(-1, )
            if hsc_r.shape[0] == 1 and lengths.shape[0] != 1:
                hsc_r = np.tile(hsc_r, lengths.shape[0])
        if hsc_r is None and reorienter is not None and reorienter.reorient:
            hsc_r = distance_between_homologs(
                structures=reorienter.struct_init, lengths=lengths,
                ploidy=ploidy, mixture_coefs=mixture_coefs)

    # INFER DRAFT STRUCTURES (for estimation of multiscale_variance & hsc_r)
    alpha_ = alpha
    beta_ = beta
    if multiscale_factor == 1 and not (draft or simple_diploid):
        struct_draft_fullres, alpha_, beta_, hsc_r, draft_converged = _infer_draft(
            counts_raw, lengths=lengths, ploidy=ploidy,
            simple_diploid_init=struct_init, outdir=outdir, alpha=alpha,
            seed=seed, normalize=normalize, filter_threshold=filter_threshold,
            alpha_init=alpha_init, max_alpha_loop=max_alpha_loop, beta=beta,
            multiscale_rounds=multiscale_rounds,
            use_multiscale_variance=use_multiscale_variance, init=init,
            max_iter=max_iter, factr=factr, pgtol=pgtol,
            alpha_factr=alpha_factr, bcc_lambda=bcc_lambda,
            hsc_lambda=hsc_lambda, hsc_r=hsc_r, hsc_min_beads=hsc_min_beads,
            fullres_torm=fullres_torm_for_multiscale,
            struct_draft_fullres=struct_draft_fullres,
            callback_freq=callback_freq, callback_function=callback_function,
            reorienter=reorienter, alpha_true=alpha_true,
            struct_true=struct_true, input_weight=input_weight,
            exclude_zeros=exclude_zeros, null=null, mixture_coefs=mixture_coefs,
            verbose=verbose)
        if not draft_converged:
            return None, {'alpha': alpha_, 'beta': beta_, 'seed': seed,
                          'converged': draft_converged}
        if beta_ != beta:
            counts = _update_betas_in_counts_matrices(
                counts=counts,
                beta={counts[i].ambiguity: np.repeat(
                    beta_, 2)[i] for i in range(len(counts))})

    if multiscale_rounds <= 1 or multiscale_factor > 1 or final_multiscale_round:
        # INFER STRUCTURE
        constraints = Constraints(counts=counts, lengths=lengths, ploidy=ploidy,
                                  multiscale_factor=multiscale_factor,
                                  constraint_lambdas={'bcc': bcc_lambda,
                                                      'hsc': hsc_lambda},
                                  constraint_params={'hsc': hsc_r})

        if outdir is not None and struct_true is not None and not null and (
                reorienter is None or not reorienter.reorient):
            _, obj_true, _, _ = objective(
                struct_true, counts=counts, alpha=alpha_, lengths=lengths,
                bias=bias, constraints=constraints,
                multiscale_factor=multiscale_factor,
                multiscale_variances=multiscale_variances,
                mixture_coefs=mixture_coefs, return_extras=True)
            pd.Series(obj_true).to_csv(
                os.path.join(outdir, 'struct_true_obj'), sep='\t', header=False)

        if callback_freq is None:
            callback_freq = {'print': 100, 'history': 100, 'save': None}
        callback = Callback(lengths, ploidy, counts=counts,
                            multiscale_factor=multiscale_factor,
                            analysis_function=callback_function,
                            frequency=callback_freq, directory=outdir,
                            struct_true=struct_true, alpha_true=alpha_true)

        pm = PastisPM(counts=counts, lengths=lengths, ploidy=ploidy,
                      alpha=alpha_, init=struct_init, bias=bias,
                      constraints=constraints, callback=callback,
                      multiscale_factor=multiscale_factor,
                      multiscale_variances=multiscale_variances,
                      alpha_init=alpha_init, max_alpha_loop=max_alpha_loop,
                      max_iter=max_iter, factr=factr, pgtol=pgtol,
                      alpha_factr=alpha_factr, reorienter=reorienter, null=null,
                      mixture_coefs=mixture_coefs, verbose=verbose)
        pm.fit()
        struct_ = pm.struct_.reshape(-1, 3)
        struct_[torm] = np.nan

        infer_var = {'alpha': pm.alpha_, 'beta': pm.beta_, 'obj': pm.obj_,
                     'seed': seed, 'converged': pm.converged_}
        if hsc_lambda > 0:
            infer_var['hsc_r'] = hsc_r

        if outdir is not None:
            with open(infer_var_file, 'w') as f:
                for k, v in infer_var.items():
                    if isinstance(v, np.ndarray) or isinstance(v, list):
                        f.write(
                            '%s\t%s\n' % (k, ' '.join(['%g' % x for x in v])))
                    else:
                        f.write('%s\t%g\n' % (k, v))
            if reorienter is not None and reorienter.reorient:
                np.savetxt(orient_file, pm.orientation_)
            if pm.converged_:
                np.savetxt(out_file, struct_)
                if pm.history_ is not None:
                    pd.DataFrame(
                        pm.history_).to_csv(history_file, sep='\t', index=False)
            else:
                np.savetxt(out_fail, struct_)

        if pm.converged_:
            return struct_, infer_var
        else:
            return None, infer_var

    else:
        # BEGIN MULTISCALE OPTIMIZATION
        all_multiscale_factors = 2 ** np.flip(
            np.arange(multiscale_rounds), axis=0)
        struct_ = init

        for i in all_multiscale_factors:
            if verbose:
                _print_code_header(
                    'MULTISCALE FACTOR %d' % i, max_length=50, blank_lines=1)
            if i == 1:
                multiscale_outdir = outdir
                final_multiscale_round = True
                fullres_torm_for_multiscale = None
            else:
                multiscale_outdir = os.path.join(outdir, 'multiscale_x%d' % i)
                final_multiscale_round = False
            struct_, infer_var = infer(
                counts_raw=counts_raw, outdir=multiscale_outdir,
                lengths=lengths, ploidy=ploidy, alpha=alpha_, seed=seed,
                normalize=normalize, filter_threshold=filter_threshold,
                alpha_init=alpha_init, max_alpha_loop=max_alpha_loop,
                beta=beta_, multiscale_factor=i,
                multiscale_rounds=multiscale_rounds,
                use_multiscale_variance=use_multiscale_variance,
                final_multiscale_round=final_multiscale_round, init=struct_,
                max_iter=max_iter, factr=factr, pgtol=pgtol,
                alpha_factr=alpha_factr, bcc_lambda=bcc_lambda,
                hsc_lambda=hsc_lambda, hsc_r=hsc_r, hsc_min_beads=hsc_min_beads,
                fullres_torm=fullres_torm_for_multiscale,
                struct_draft_fullres=struct_draft_fullres,
                callback_function=callback_function,
                callback_freq=callback_freq, reorienter=reorienter,
                alpha_true=alpha_true, struct_true=struct_true,
                input_weight=input_weight, exclude_zeros=exclude_zeros,
                null=null, mixture_coefs=mixture_coefs, verbose=verbose)
            if not infer_var['converged']:
                return struct_, infer_var
        return struct_, infer_var


def pastis_poisson(counts, lengths, ploidy, outdir='', chromosomes=None,
                   chrom_subset=None, alpha=None, seed=0, normalize=True,
                   filter_threshold=0.04, alpha_init=-3., max_alpha_loop=20,
                   beta=None, multiscale_rounds=1, use_multiscale_variance=True,
                   max_iter=10000000000, factr=10000000., pgtol=1e-05,
                   alpha_factr=1000000000000., bcc_lambda=0., hsc_lambda=0.,
                   hsc_r=None, hsc_min_beads=5, callback_function=None,
                   print_freq=100, history_freq=100, save_freq=None,
                   piecewise=False, piecewise_step=None, piecewise_chrom=None,
                   piecewise_min_beads=5, piecewise_fix_homo=False,
                   piecewise_opt_orient=True, alpha_true=None, struct_true=None,
                   init='mds', input_weight=None, exclude_zeros=False,
                   null=False, mixture_coefs=None, verbose=True):
    """Infer 3D structures with PASTIS via Poisson model.

    Infer 3D structure from Hi-C contact counts data for haploid or diploid
    organisms.

    Parameters
    ----------
    counts : list of str
        Counts data files in the hiclib format or as numpy ndarrays.
    lengths : str or list
        Number of beads per homolog of each chromosome, or hiclib .bed file with
        lengths data.
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    outdir : str, optional
        Directory in which to save results.
    chromosomes : list of str, optional
        Label for each chromosome in the data, or file with chromosome labels
        (one label per line).
    chrom_subset : list of str, optional
        Labels of chromosomes for which inference should be performed.
    alpha : float, optional
        Biophysical parameter of the transfer function used in converting
        counts to wish distances. If alpha is not specified, it will be
        inferred.
    seed : int, optional
        Random seed used when generating the starting point in the
        optimization.
    normalize : bool, optional
        Perfrom ICE normalization on the counts prior to optimization.
        Normalization is reccomended.
    filter_threshold : float, optional
        Ratio of non-zero beads to be filtered out. Filtering is
        reccomended.
    alpha_init : float, optional
        For PM2, the initial value of alpha to use.
    max_alpha_loop : int, optional
        For PM2, Number of times alpha and structure are inferred.
    beta : array_like of float, optional
        Scaling parameter that determines the size of the structure, relative to
        each counts matrix. There should be one beta per counts matrix. If None,
        the optimal beta will be estimated.
    multiscale_rounds : int, optional
        The number of resolutions at which a structure should be inferred
        during multiscale optimization. Values of 1 or 0 disable multiscale
        optimization.
    max_iter : int, optional
        Maximum number of iterations per optimization.
    factr : float, optional
        factr for scipy's L-BFGS-B, alters convergence criteria.
    pgtol : float, optional
        pgtol for scipy's L-BFGS-B, alters convergence criteria.
    alpha_factr : float, optional
        factr for convergence criteria of joint alpha/structure inference.
    bcc_lambda : float, optional
        Lambda of the bead chain connectivity constraint.
    hsc_lambda : float, optional
        For diploid organisms: lambda of the homolog-separating
        constraint.
    hsc_r : list of float, optional
        For diploid organisms: hyperparameter of the homolog-separating
        constraint specificying the expected distance between homolog
        centers of mass for each chromosome. If not supplied, `hsc_r` will
        be inferred from the counts data.
    hsc_min_beads : int, optional
        For diploid organisms: number of beads in the low-resolution
        structure from which `hsc_r` is estimated.

    Returns
    -------
    struct_ : array_like of float of shape (lengths.sum() * ploidy, 3)
        3D structure resulting from the optimization.
    infer_var : dict
        A few of the variables used in inference or generated by inference.
        Keys: 'alpha', 'beta', 'hsc_r', 'obj', and 'seed'.
    """

    lengths_full = lengths
    chrom_full = chromosomes
    callback_freq = {'print': print_freq, 'history': history_freq,
                     'save': save_freq}

    counts, lengths_subset, chrom_subset, lengths_full, chrom_full, struct_true = load_data(
        counts=counts, lengths_full=lengths_full, ploidy=ploidy,
        chrom_full=chrom_full, chrom_subset=chrom_subset,
        exclude_zeros=exclude_zeros, struct_true=struct_true)

    outdir = _output_subdir(
        outdir=outdir, chrom_full=chrom_full, chrom_subset=chrom_subset,
        null=null)

    if (not piecewise) or len(chrom_subset) == 1:
        struct_, infer_var = infer(
            counts_raw=counts, outdir=outdir, lengths=lengths_subset,
            ploidy=ploidy, alpha=alpha, seed=seed, normalize=normalize,
            filter_threshold=filter_threshold, alpha_init=alpha_init,
            max_alpha_loop=max_alpha_loop, beta=beta,
            multiscale_rounds=multiscale_rounds,
            use_multiscale_variance=use_multiscale_variance, init=init,
            max_iter=max_iter, factr=factr, pgtol=pgtol,
            alpha_factr=alpha_factr, bcc_lambda=bcc_lambda,
            hsc_lambda=hsc_lambda, hsc_r=hsc_r, hsc_min_beads=hsc_min_beads,
            callback_function=callback_function, callback_freq=callback_freq,
            alpha_true=alpha_true, struct_true=struct_true,
            input_weight=input_weight, exclude_zeros=exclude_zeros,
            null=null, mixture_coefs=mixture_coefs, verbose=verbose)
    else:
        from .piecewise_whole_genome import infer_piecewise

        struct_, infer_var = infer_piecewise(
            counts_raw=counts, outdir=outdir, lengths=lengths_subset,
            ploidy=ploidy, chromosomes=chrom_subset, alpha=alpha, seed=seed,
            normalize=normalize, filter_threshold=filter_threshold,
            alpha_init=alpha_init, max_alpha_loop=max_alpha_loop, beta=beta,
            multiscale_rounds=multiscale_rounds,
            use_multiscale_variance=use_multiscale_variance, max_iter=max_iter,
            factr=factr, pgtol=pgtol, alpha_factr=alpha_factr,
            bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda, hsc_r=hsc_r,
            hsc_min_beads=hsc_min_beads, callback_function=callback_function,
            callback_freq=callback_freq,
            piecewise_step=piecewise_step,
            piecewise_chrom=piecewise_chrom,
            piecewise_min_beads=piecewise_min_beads,
            piecewise_fix_homo=piecewise_fix_homo,
            piecewise_opt_orient=piecewise_opt_orient,
            alpha_true=alpha_true, struct_true=struct_true, init=init,
            input_weight=input_weight, exclude_zeros=exclude_zeros, null=null,
            mixture_coefs=mixture_coefs, verbose=verbose)

    return struct_, infer_var
