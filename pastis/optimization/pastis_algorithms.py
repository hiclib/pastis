import os
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from .utils import print_code_header


def test_objective(struct, counts, lengths, ploidy, alpha, bias,
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


def infer(counts_raw, outdir, lengths, ploidy, alpha, seed=0, normalize=True,
          filter_threshold=0.04, alpha_init=-3., max_alpha_loop=20, beta=None,
          multiscale_factor=1, multiscale_rounds=0, use_multiscale_variance=True,
          final_multiscale_round=False, init='msd', max_iter=1e40,
          factr=10000000.0, pgtol=1e-05, alpha_factr=1000000000000.,
          bcc_lambda=0., hsc_lambda=0., hsc_r=None, hsc_min_beads=5,
          fullres_torm=None, struct_draft_fullres=None, draft=False,
          simple_diploid=False, simple_diploid_init=None,
          callback_function=None, callback_freq=None, reorienter=None,
          alpha_true=None, struct_true=None, input_weight=None,
          exclude_zeros=False, null=False, mixture_coefs=None, verbose=True):
    """Infer 3D structures with PASTIS.
    """

    from .counts import preprocess_counts, ambiguate_counts, update_betas_in_counts_matrices
    from .initialization import initialize
    from .callbacks import Callback
    from .constraints import Constraints, distance_between_homologs
    from .poisson import PastisPM
    from .estimate_alpha_beta import _estimate_beta
    from .multiscale_optimization import get_multiscale_variances_from_struct, choose_max_multiscale_factor, decrease_lengths_res
    from .utils import find_beads_to_remove

    try:
        os.makedirs(outdir)
    except OSError:
        pass
    if seed is None:
        seed_str = ''
    else:
        seed_str = '.%03d' % seed
    out_file = os.path.join(outdir, 'struct_inferred%s.coords' % seed_str)
    orient_file = os.path.join(outdir, 'orient_inferred%s.coords' % seed_str)
    history_file = os.path.join(outdir, 'history%s' % seed_str)
    infer_var_file = os.path.join(
        outdir, 'inference_variables%s' % seed_str)
    out_fail = os.path.join(outdir, 'struct_nonconverged%s.coords' % seed_str)

    if os.path.exists(out_file):
        print('CONVERGED', flush=True)
        infer_var = dict(pd.read_csv(
            infer_var_file, sep='\t', header=None, squeeze=True, index_col=0))
        infer_var['beta'] = [float(b) for b in infer_var['beta'].split()]
        infer_var['alpha'] = float(infer_var['alpha'])
        struct_ = np.loadtxt(out_file)
        return struct_, infer_var
    elif os.path.exists(out_fail):
        print('OPTIMIZATION DID NOT CONVERGE', flush=True)
        exit(1)

    random_state = np.random.RandomState(seed)
    random_state = check_random_state(random_state)

    # PREPARE COUNTS OBJECTS
    if simple_diploid:
        counts_raw = ambiguate_counts(counts=counts_raw, n=lengths.sum())
        ploidy = 1
    counts, bias, torm = preprocess_counts(
        counts=counts_raw, lengths=lengths, ploidy=ploidy, normalize=normalize,
        filter_threshold=filter_threshold, multiscale_factor=multiscale_factor,
        exclude_zeros=exclude_zeros, beta=beta, input_weight=input_weight,
        verbose=verbose, fullres_torm=fullres_torm, output_directory=None)
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
        counts=counts, lengths=lengths, random_state=random_state, init=init,
        ploidy=ploidy, alpha=alpha, bias=bias,
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
                multiscale_rounds > 0 and use_multiscale_variance) or alpha is None):
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
            alpha_ = infer_var_fullres['alpha']
            beta_ = infer_var_fullres['beta']
            counts = update_betas_in_counts_matrices(
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
            multiscale_factor_for_lowres = choose_max_multiscale_factor(
                lengths=lengths, min_beads=hsc_min_beads)
            struct_draft_lowres, _ = infer(
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
            hsc_r = distance_between_homologs(
                structures=struct_draft_lowres,
                lengths=decrease_lengths_res(
                    lengths=lengths, factor=multiscale_factor_for_lowres),
                ploidy=ploidy, mixture_coefs=mixture_coefs,
                simple_diploid=simple_diploid_for_lowres)
            if verbose:
                print("Estimated distance between homolog barycenters for each"
                      " chromosome: %s" % ' '.join(map(str, hsc_r.round(2))), flush=True)

    if multiscale_rounds == 0 or multiscale_factor > 1 or final_multiscale_round:
        # INFER STRUCTURE
        constraints = Constraints(counts=counts, lengths=lengths, ploidy=ploidy,
                                  multiscale_factor=multiscale_factor,
                                  constraint_lambdas={'bcc': bcc_lambda,
                                                      'hsc': hsc_lambda},
                                  constraint_params={'hsc': hsc_r})

        if struct_true is not None and not null and (reorienter is None or not reorienter.reorient):
            test_objective(struct=struct_true, counts=counts,
                           lengths=lengths, ploidy=ploidy, alpha=alpha_,
                           bias=bias, multiscale_factor=multiscale_factor,
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
                      alpha=alpha_, beta=beta_, init=struct_init, bias=bias,
                      constraints=constraints, callback=callback,
                      multiscale_factor=multiscale_factor,
                      multiscale_variances=multiscale_variances,
                      alpha_init=alpha_init, max_alpha_loop=max_alpha_loop,
                      max_iter=max_iter, factr=factr, pgtol=pgtol,
                      alpha_factr=alpha_factr, reorienter=reorienter, null=null,
                      mixture_coefs=mixture_coefs, verbose=verbose)
        pm.fit()
        struct_ = pm.struct_.reshape(-1, 3)
        struct_[np.tile(torm, len(mixture_coefs))] = np.nan

        infer_var = {'alpha': pm.alpha_, 'beta': pm.beta_, 'hsc_r': hsc_r,
                     'obj': pm.obj_, 'seed': seed}

        if reorienter is not None and reorienter.reorient:
            np.savetxt(orient_file, pm.orientation_)
        if pm.converged_:
            np.savetxt(out_file, struct_)
            pd.Series(infer_var).to_csv(infer_var_file, sep='\t', header=False)
            if pm.history_ is not None:
                pd.DataFrame(
                    pm.history_).to_csv(history_file, sep='\t', index=False)
            return struct_, infer_var
        else:
            np.savetxt(out_fail, struct_)
            exit(1)

    else:
        # BEGIN MULTISCALE OPTIMIZATION
        all_multiscale_factors = 2 ** np.flip(
            np.arange(multiscale_rounds), axis=0)
        struct_ = init
        fullres_torm_for_multiscale = [find_beads_to_remove(
            c, nbeads=lengths.sum() * ploidy) for c in counts]

        for i in all_multiscale_factors:
            if verbose:
                print_code_header(
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
        return struct_, infer_var


def pastis(counts, outdir, lengths_full, ploidy, chrom_full, chrom_subset,
           alpha, seed=0, normalize=True, filter_threshold=0.04, alpha_init=-3.,
           max_alpha_loop=20, multiscale_rounds=0, use_multiscale_variance=True,
           max_iter=1e40, factr=10000000.0, pgtol=1e-05,
           alpha_factr=1000000000000., bcc_lambda=0., hsc_lambda=0., hsc_r=None,
           hsc_min_beads=5, callback_function=None, callback_freq=None,
           stepwise_genome=False, stepwise_genome__step=None,
           stepwise_genome__chrom=None, stepwise_genome__min_beads=5,
           stepwise_genome__fix_homo=False,
           stepwise_genome__optimize_orient=True, alpha_true=None,
           struct_true=None, init='msd', input_weight=None, exclude_zeros=False,
           null=False, mixture_coefs=None, verbose=True):
    """Infer 3D structures with PASTIS.
    """

    from .load_data import load_data
    from .stepwise_whole_genome import stepwise_inference

    counts, struct_true, lengths_subset, chrom_subset, lengths_full, chrom_full = load_data(
        counts=counts, ploidy=ploidy, lengths_full=lengths_full,
        chrom_full=chrom_full, chrom_subset=chrom_subset,
        exclude_zeros=exclude_zeros, struct_true=struct_true)

    if len(chrom_subset) == 1:
        stepwise_genome = False

    if (not stepwise_genome) or len(chrom_subset) == 1:
        outdir = output_subdir(
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
        stepwise_inference(
            counts=counts, outdir=outdir, lengths=lengths_subset, ploidy=ploidy,
            chromosomes=chrom_subset, alpha=alpha, seed=seed, normalize=normalize,
            filter_threshold=filter_threshold, alpha_init=alpha_init,
            max_alpha_loop=max_alpha_loop, multiscale_rounds=multiscale_rounds,
            use_multiscale_variance=use_multiscale_variance, max_iter=max_iter,
            factr=factr, pgtol=pgtol, alpha_factr=alpha_factr,
            bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda, hsc_r=hsc_r,
            hsc_min_beads=hsc_min_beads, callback_function=callback_function,
            callback_freq=callback_freq,
            stepwise_genome__step=stepwise_genome__step,
            stepwise_genome__chrom=stepwise_genome__chrom,
            stepwise_genome__min_beads=stepwise_genome__min_beads,
            stepwise_genome__fix_homo=stepwise_genome__fix_homo,
            stepwise_genome__optimize_orient=stepwise_genome__optimize_orient,
            alpha_true=alpha_true, struct_true=struct_true, init=init,
            input_weight=input_weight, exclude_zeros=exclude_zeros, null=null,
            mixture_coefs=mixture_coefs, verbose=verbose)


def output_subdir(outdir, chrom_full, chrom_subset=None, null=False,
                  stepwise_genome=False, stepwise_genome__step=None,
                  stepwise_genome__chrom=None):
    """Returns subdirectory for inference output files.
    """

    if null:
        outdir = os.path.join(outdir, 'null')

    if (not stepwise_genome) or (stepwise_genome__step is not None and stepwise_genome__step != 2):
        if chrom_subset is not None and len(chrom_subset) != len(chrom_full):
            outdir = os.path.join(outdir, '.'.join(chrom_subset))

    if stepwise_genome:
        if stepwise_genome__step is None:
            raise ValueError("stepwise_genome__step may not be None")
        if stepwise_genome__step == 1:
            outdir = os.path.join(outdir, 'step1_lowres')
        elif stepwise_genome__step == 2:
            if stepwise_genome__chrom is None:
                raise ValueError("stepwise_genome__chrom may not be None")
            outdir = os.path.join(outdir, stepwise_genome__chrom)
        elif stepwise_genome__step == 3:
            outdir = os.path.join(outdir, 'step3_assembled')

    return outdir
