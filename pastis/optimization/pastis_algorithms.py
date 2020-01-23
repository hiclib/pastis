import sys

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

import os
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from .utils_poisson import _print_code_header
from distutils.util import strtobool


def _test_objective(struct, counts, lengths, ploidy, alpha, bias,
                    multiscale_factor, multiscale_variances, constraints,
                    reorienter=None, mixture_coefs=None, output_file=None):
    """Computes all components of the objective for a given structure.
    """

    from .poisson import objective_wrapper
    from .callbacks import Callback

    callback = Callback(lengths, ploidy, multiscale_factor=multiscale_factor,
                        frequency={'print': None, 'history': 1, 'save': None})
    if reorienter is not None and reorienter.reorient:
        opt_type = 'chrom_reorient'
    else:
        opt_type = 'structure'
    callback.on_training_begin(opt_type=opt_type)
    objective_wrapper(struct.flatten(), counts, alpha=alpha, bias=bias,
                      lengths=lengths, constraints=constraints,
                      reorienter=reorienter,
                      multiscale_factor=multiscale_factor,
                      multiscale_variances=multiscale_variances,
                      mixture_coefs=mixture_coefs, callback=callback)
    if output_file is not None:
        pd.Series(callback.obj).to_csv(output_file, sep='\t', header=False)

    return callback.obj


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
        For diploid organisms: initialization to be used for inference when
        `simple_diploid` is True.

    Returns
    -------
    struct_ : array_like of float of shape (lengths.sum() * ploidy, 3)
        3D structure resulting from the optimization.
    infer_var : dict
        A few of the variables used in inference or generated by inference.
        Keys: 'alpha', 'beta', 'hsc_r', 'obj', and 'seed'.
    """

    from .counts import preprocess_counts, ambiguate_counts, _update_betas_in_counts_matrices
    from .initialization import initialize
    from .callbacks import Callback
    from .constraints import Constraints, distance_between_homologs
    from .poisson import PastisPM
    from .estimate_alpha_beta import _estimate_beta
    from .multiscale_optimization import get_multiscale_variances_from_struct, _choose_max_multiscale_factor, decrease_lengths_res
    from .utils_poisson import find_beads_to_remove

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
            infer_var = dict(pd.read_csv(
                infer_var_file, sep='\t', header=None, squeeze=True,
                index_col=0))
            infer_var['beta'] = [float(b) for b in infer_var['beta'].split()]
            infer_var['hsc_r'] = [float(r) for r in infer_var['hsc_r'].split()]
            infer_var['alpha'] = float(infer_var['alpha'])
            infer_var['converged'] = strtobool(infer_var['converged'])
            struct_ = np.loadtxt(out_file)
            return struct_, infer_var

    random_state = np.random.RandomState(seed)
    random_state = check_random_state(random_state)

    # PREPARE COUNTS OBJECTS
    if simple_diploid:
        counts_raw = ambiguate_counts(
            counts=counts_raw, lengths=lengths, ploidy=ploidy)
        ploidy = 1
    counts, bias, torm = preprocess_counts(
        counts_raw=counts_raw, lengths=lengths, ploidy=ploidy, normalize=normalize,
        filter_threshold=filter_threshold, multiscale_factor=multiscale_factor,
        exclude_zeros=exclude_zeros, beta=beta, input_weight=input_weight,
        verbose=verbose, fullres_torm=fullres_torm, output_directory=None)
    if mixture_coefs is not None:
        torm = np.tile(torm, len(mixture_coefs))
    if simple_diploid:
        if simple_diploid_init is None:
            raise ValueError("Must provide simple_diploid_init.")
        counts = _estimate_beta(
            simple_diploid_init, counts=counts, alpha=alpha, bias=bias,
            lengths=lengths, reorienter=reorienter, mixture_coefs=mixture_coefs,
            verbose=verbose, simple_diploid=True)

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
                structures=reorienter.init_structures, lengths=lengths, ploidy=ploidy,
                mixture_coefs=mixture_coefs)

    # INFER DRAFT STRUCTURES (for estimation of multiscale_variance & hsc_r)
    alpha_ = alpha
    beta_ = beta
    if multiscale_factor == 1 and not draft:
        if struct_draft_fullres is None and ((
                multiscale_rounds > 1 and use_multiscale_variance) or alpha is None):
            struct_draft_fullres, infer_var_fullres = infer(
                counts_raw=counts_raw,
                outdir=os.path.join(outdir, 'struct_draft_fullres'),
                lengths=lengths, ploidy=ploidy, alpha=alpha,
                seed=seed, normalize=normalize,
                filter_threshold=filter_threshold, alpha_init=alpha_init,
                max_alpha_loop=max_alpha_loop, init=init, max_iter=max_iter,
                factr=factr, pgtol=pgtol, alpha_factr=alpha_factr, draft=True,
                simple_diploid=(ploidy == 2), simple_diploid_init=struct_init,
                callback_function=callback_function,
                callback_freq=callback_freq, reorienter=reorienter,
                alpha_true=alpha_true, struct_true=struct_true,
                input_weight=input_weight, exclude_zeros=exclude_zeros,
                null=null, mixture_coefs=mixture_coefs, verbose=verbose)
            if not infer_var_fullres['converged']:
                return struct_draft_fullres, infer_var_fullres
            alpha_ = infer_var_fullres['alpha']
            beta_ = infer_var_fullres['beta']
            counts = _update_betas_in_counts_matrices(
                counts=counts,
                beta={counts[i].ambiguity: np.repeat(
                    beta_, 2)[i] for i in range(len(counts))})

        if hsc_lambda > 0 and hsc_r is None:
            if ploidy == 1:
                raise ValueError("Can not apply homolog-separating constraint"
                                 " to haploid data.")
            if alpha_ is None:
                raise ValueError("Alpha must be set prior to inferring r from"
                                 " counts data")
            fullres_torm_for_lowres = [find_beads_to_remove(
                c, nbeads=lengths.sum() * ploidy) for c in counts]
            ua_index = [i for i in range(len(counts)) if counts[
                i].name == 'ua']
            if len(ua_index) == 1:
                counts_for_lowres = counts_raw[ua_index[0]]
                simple_diploid_for_lowres = False
                fullres_torm_for_lowres = fullres_torm_for_lowres[ua_index[0]]
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
                counts_raw=counts_for_lowres,
                outdir=os.path.join(outdir, 'struct_draft_lowres'),
                lengths=lengths, ploidy=ploidy, alpha=alpha_,
                seed=seed, normalize=normalize,
                filter_threshold=filter_threshold, beta=beta_,
                multiscale_factor=multiscale_factor_for_lowres,
                use_multiscale_variance=use_multiscale_variance,
                init=init, max_iter=max_iter, factr=factr, pgtol=pgtol,
                bcc_lambda=bcc_lambda, fullres_torm=fullres_torm_for_lowres,
                struct_draft_fullres=struct_draft_fullres, draft=True,
                simple_diploid=simple_diploid_for_lowres,
                simple_diploid_init=struct_init,
                callback_function=callback_function,
                callback_freq=callback_freq,
                reorienter=reorienter, alpha_true=alpha_true,
                struct_true=struct_true, input_weight=input_weight,
                exclude_zeros=exclude_zeros, null=null,
                mixture_coefs=mixture_coefs, verbose=verbose)
            if not infer_var_lowres['converged']:
                return struct_draft_lowres, infer_var_lowres
            hsc_r = distance_between_homologs(
                structures=struct_draft_lowres,
                lengths=decrease_lengths_res(
                    lengths=lengths, factor=multiscale_factor_for_lowres),
                ploidy=ploidy, mixture_coefs=mixture_coefs,
                simple_diploid=simple_diploid_for_lowres)
            if verbose:
                print("Estimated distance between homolog barycenters for each"
                      " chromosome: %s" % ' '.join(map(str, hsc_r.round(2))), flush=True)

    if multiscale_rounds <= 1 or multiscale_factor > 1 or final_multiscale_round:
        # INFER STRUCTURE
        constraints = Constraints(counts=counts, lengths=lengths, ploidy=ploidy,
                                  multiscale_factor=multiscale_factor,
                                  constraint_lambdas={'bcc': bcc_lambda,
                                                      'hsc': hsc_lambda},
                                  constraint_params={'hsc': hsc_r})

        if struct_true is not None and not null and (reorienter is None or not reorienter.reorient):
            _test_objective(
                struct=struct_true, counts=counts, lengths=lengths,
                ploidy=ploidy, alpha=alpha_, bias=bias,
                multiscale_factor=multiscale_factor,
                multiscale_variances=multiscale_variances,
                constraints=constraints, reorienter=reorienter,
                mixture_coefs=mixture_coefs,
                output_file=os.path.join(outdir, 'struct_true_obj'))

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

        infer_var = {'alpha': pm.alpha_, 'beta': pm.beta_, 'hsc_r': hsc_r,
                     'obj': pm.obj_, 'seed': seed, 'converged': pm.converged_}

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
        fullres_torm_for_multiscale = [find_beads_to_remove(
            c, nbeads=lengths.sum() * ploidy) for c in counts]

        for i in all_multiscale_factors:
            if verbose:
                _print_code_header(
                    'MULTISCALE FACTOR %d' % i, max_length=50, blank_lines=1)
            if multiscale_factor == 1:
                multiscale_outdir = outdir
                final_multiscale_round = True
                fullres_torm_for_multiscale = None
            else:
                multiscale_outdir = os.path.join(
                    outdir, 'multiscale_x%d' % multiscale_factor)
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
                   multiscale_rounds=1, use_multiscale_variance=True,
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

    from .load_data import load_data
    from .piecewise_whole_genome import piecewise_inference

    lengths_full = lengths
    chrom_full = chromosomes
    callback_freq = {'print': print_freq, 'history': history_freq,
                     'save': save_freq}

    counts, lengths_subset, chrom_subset, lengths_full, chrom_full, struct_true = load_data(
        counts=counts, lengths_full=lengths_full, ploidy=ploidy,
        chrom_full=chrom_full, chrom_subset=chrom_subset,
        exclude_zeros=exclude_zeros, struct_true=struct_true)

    if len(chrom_subset) == 1:
        piecewise = False

    if (not piecewise) or len(chrom_subset) == 1:
        outdir = _output_subdir(
            outdir=outdir, chrom_full=chrom_full, chrom_subset=chrom_subset,
            null=null)

        struct_, infer_var = infer(
            counts_raw=counts, outdir=outdir, lengths=lengths_subset,
            ploidy=ploidy, alpha=alpha, seed=seed, normalize=normalize,
            filter_threshold=filter_threshold, alpha_init=alpha_init,
            max_alpha_loop=max_alpha_loop, multiscale_rounds=multiscale_rounds,
            use_multiscale_variance=use_multiscale_variance, init=init,
            max_iter=max_iter, factr=factr, pgtol=pgtol,
            alpha_factr=alpha_factr, bcc_lambda=bcc_lambda,
            hsc_lambda=hsc_lambda, hsc_r=hsc_r, hsc_min_beads=hsc_min_beads,
            callback_function=callback_function, callback_freq=callback_freq,
            alpha_true=alpha_true, struct_true=struct_true,
            input_weight=input_weight, exclude_zeros=exclude_zeros,
            null=null, mixture_coefs=mixture_coefs, verbose=verbose)
    else:
        struct_, infer_var = piecewise_inference(
            counts=counts, outdir=outdir, lengths=lengths_subset, ploidy=ploidy,
            chromosomes=chrom_subset, alpha=alpha, seed=seed, normalize=normalize,
            filter_threshold=filter_threshold, alpha_init=alpha_init,
            max_alpha_loop=max_alpha_loop, multiscale_rounds=multiscale_rounds,
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


def _output_subdir(outdir, chrom_full, chrom_subset=None, null=False,
                   piecewise=False, piecewise_step=None,
                   piecewise_chrom=None):
    """Returns subdirectory for inference output files.
    """

    if null:
        outdir = os.path.join(outdir, 'null')

    if (not piecewise) or (piecewise_step is not None and piecewise_step != 2):
        if chrom_subset is not None and len(chrom_subset) != len(chrom_full):
            outdir = os.path.join(outdir, '.'.join(chrom_subset))

    if piecewise:
        if piecewise_step is None:
            raise ValueError("piecewise_step may not be None")
        if piecewise_step == 1:
            outdir = os.path.join(outdir, 'step1_lowres')
        elif piecewise_step == 2:
            if piecewise_chrom is None:
                raise ValueError("piecewise_chrom may not be None")
            outdir = os.path.join(outdir, piecewise_chrom)
        elif piecewise_step == 3:
            outdir = os.path.join(outdir, 'step3_assembled')

    return outdir
