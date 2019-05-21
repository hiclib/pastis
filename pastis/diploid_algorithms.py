import os
import numpy as np
import pandas as pd
from topsy.externals.iced.io import write_counts, write_lengths
from topsy.inference.poisson_diploid import PM1, translate_and_rotate, get_obj_details
from topsy.inference import multiscale_optimization
from topsy.inference.prep_counts import subset_chrom, prep_counts, format_counts
from topsy.inference.utils import print_code_header
from topsy.metrics import generate_metrics
try:
    from topsy.plot.plot_distances import plot_dist
    from topsy.plot.plot_counts import plot_counts
except:
    pass
from scipy import sparse


def load_data(counts_files, ploidy, genome_lengths, genome_chrom=None, infer_chrom=None, as_sparse=True, X_true=None):
    from topsy.externals.iced.io import load_counts, load_lengths

    # Load lengths
    if isinstance(genome_lengths, str) and os.path.exists(genome_lengths):
        genome_lengths = load_lengths(genome_lengths)
    elif genome_lengths is not None and (isinstance(genome_lengths, list) or isinstance(genome_lengths, np.ndarray)):
        if len(genome_lengths) == 1 and isinstance(genome_lengths[0], str) and os.path.exists(genome_lengths[0]):
            genome_lengths = load_lengths(genome_lengths[0])
    genome_lengths = np.array(genome_lengths).astype(int)

    # Load chromosomes
    if isinstance(genome_chrom, str) and os.path.exists(genome_chrom):
        genome_chrom = np.array(np.genfromtxt(genome_chrom, dtype='str')).reshape(-1)
    elif genome_chrom is not None and (isinstance(genome_chrom, list) or isinstance(genome_chrom, np.ndarray)):
        if len(genome_chrom) == 1 and isinstance(genome_chrom[0], str) and os.path.exists(genome_chrom[0]):
            genome_chrom = np.array(np.genfromtxt(genome_chrom[0], dtype='str')).reshape(-1)
        genome_chrom = np.array(genome_chrom)
    else:
        genome_chrom = np.array(['num%d' % i for i in range(1, len(genome_lengths) + 1)])

    counts = []
    for f in counts_files:
        if f.endswith("npy"):
            counts_maps = np.load(f)
            counts_maps[np.isnan(counts_maps)] = 0
        elif f.endswith(".matrix"):
            counts_maps = load_counts(f, lengths=genome_lengths)
        else:
            raise ValueError('Counts file must end with .npy (for numpy array) or .matrix (for hiclib / iced format)')
        counts.append(counts_maps)

    if X_true is not None and isinstance(X_true, str):
        X_true = np.loadtxt(X_true)

    counts, X_true, infer_lengths, infer_chrom = subset_chrom(counts=counts, ploidy=ploidy, genome_lengths=genome_lengths, genome_chrom=genome_chrom, infer_chrom=infer_chrom, as_sparse=as_sparse, X_true=X_true)

    return counts, X_true, infer_lengths, infer_chrom, genome_lengths, genome_chrom


def analyze_inferred_X(counts, X, out_file, init_file, X_true, lengths, ploidy, init, outdir, multiscale_factor, seed, alpha=-3, modifications=None):
    from topsy.datasets.samples_generator import haploid_structure2counts, diploid_structure2counts

    print('Analyzing %s' % out_file)

    if modifications is None:
        modifications = []
    if alpha is None:
        alpha = -3.

    lengths_lowres = multiscale_optimization.decrease_lengths_res(lengths, multiscale_factor)

    ambiguity = [{1: 'ambig', 1.5: 'pa', 2: 'ua'}[sum(c.shape) / (lengths.sum() * ploidy)] for c in counts]
    nreads = [np.nansum((c.toarray() if sparse.isspmatrix(c) else c)) for c in counts]
    perc_ua = perc_pa = 0.
    if 'ua' in ambiguity:
        perc_ua = sum([nreads[i] for i in range(len(counts)) if ambiguity[i] == 'ua']) / sum(nreads)
    if 'pa' in ambiguity:
        perc_pa = sum([nreads[i] for i in range(len(counts)) if ambiguity[i] == 'pa']) / sum(nreads)

    # Plot euclidian distances
    try:
        plot_dist(out_file, lengths_lowres, 'Inferred distances')
    except:
        print('WARNING: Plotting distances failed', flush=True)

    # Plot "counts" simulated from inferred X
    print('Plotting "counts" that are simulated from inferred X', flush=True)
    fake_counts_outdir = os.path.join(outdir, 'fake_counts_from_X', '' if seed is None else '%03d' % int(seed))
    try:
        os.makedirs(fake_counts_outdir)
    except OSError:
        pass
    if ploidy == 1:
        fake_counts, _ = haploid_structure2counts(X, alpha, sum(nreads), random_state=np.random.RandomState(seed=0))
        write_counts(os.path.join(fake_counts_outdir, "counts.matrix"), np.triu(fake_counts, 1))
    else:
        fake_all_counts, fake_ambig_counts, fake_ua_counts, fake_pa_counts, _ = \
            diploid_structure2counts(X, alpha=alpha, nreads=sum(nreads), random_state=np.random.RandomState(seed=0),
                                     return_intensity_not_counts=False, lengths=lengths_lowres, perc_ua=perc_ua, perc_pa=perc_pa)
        write_counts(os.path.join(fake_counts_outdir, "counts.matrix"), np.triu(fake_all_counts, 1))
        write_counts(os.path.join(fake_counts_outdir, "ambig_counts.matrix"), np.triu(fake_ambig_counts, 1))
        write_counts(os.path.join(fake_counts_outdir, "ua_counts.matrix"), np.triu(fake_ua_counts, 1))
        write_counts(os.path.join(fake_counts_outdir, "pa_counts.matrix"), fake_pa_counts)
    write_lengths(os.path.join(fake_counts_outdir, "counts.bed"), lengths_lowres)
    try:
        plot_counts(fake_counts_outdir)
    except:
        print('WARNING: Plotting counts-derived-from-inferred-struct failed', flush=True)

    # Generate metrics
    generate_metrics.error_scores(X_true=X_true, X_inferred=out_file, lengths=lengths_lowres, homolog_order_defined=('ua' in ambiguity or 'pa' in ambiguity), alpha=alpha, modifications=modifications, ploidy=ploidy)
    if os.path.exists(init_file):
        generate_metrics.error_scores(X_true=X_true, X_inferred=init_file, lengths=lengths_lowres, homolog_order_defined=('ua' in ambiguity or 'pa' in ambiguity), alpha=alpha, modifications=modifications, ploidy=ploidy)


def infer(counts, lengths, alpha, beta, ploidy, init, outdir, seed, lagrange_mult, constraints, homo_init, input_weight, multiscale_factor, as_sparse, norm, filter_counts,
          HSC_lowres_beads=5, init_structures=None, translate=False, rotate=False, stepwise_genome__fix_homo=True, modifications=None, null=False, filter_percentage=0.04, X_true=None, max_iter=1e40, in_2d=False, redo_analysis=False):

    if modifications is None:
        modifications = []

    try:
        os.makedirs(outdir)
    except OSError:
        pass

    # Define output files
    seed_str = '' if seed is None else '.%03d' % int(seed)
    X_true_obj_file = os.path.join(outdir, 'X_true_obj.txt')
    init_file = os.path.join(outdir, 'X_init%s.txt' % seed_str)
    out_file = os.path.join(outdir, 'X_inferred%s.txt' % seed_str)
    orient_file = os.path.join(outdir, 'orient_inferred%s.txt' % seed_str)
    orient_init_file = os.path.join(outdir, 'orient_init%s.txt' % seed_str)
    out_fail = os.path.join(outdir, 'X_inferred_nonconverged%s.txt' % seed_str)
    info_file = os.path.join(outdir, 'per_iteration%s.txt' % seed_str)
    infer_var_file = os.path.join(outdir, 'inference_variables%s.txt' % seed_str)
    bias_file = os.path.join(outdir, 'bias.txt')

    ambiguity = [{1: 'ambig', 1.5: 'pa', 2: 'ua'}[sum(c.shape) / (lengths.sum() * ploidy)] for c in counts]

    if os.path.exists(out_file):
        print('CONVERGED', flush=True)
        if os.path.exists(infer_var_file):
            infer_var = dict(pd.read_csv(infer_var_file, sep='\t', header=None, names=('label', 'value')).set_index('label').value)
            infer_var['beta'] = [float(infer_var['beta.%s' % x]) for x in ambiguity]
            infer_var['alpha'] = float(infer_var['alpha'])
        else:
            print('%s not found' % infer_var_file, flush=True)
            infer_var = {'alpha': None, 'beta': None}
        X = np.loadtxt(out_file)
        if redo_analysis and not null:
            alpha_ = alpha if alpha is not None else float(infer_var['alpha'])
            analyze_inferred_X(counts=counts, X=X, out_file=out_file, init_file=init_file, X_true=X_true, lengths=lengths,
                               ploidy=ploidy, init=init, outdir=outdir, multiscale_factor=multiscale_factor, seed=seed, alpha=alpha_, modifications=modifications)
        return True, X, infer_var
    elif os.path.exists(out_fail):
        if np.loadtxt(out_fail).shape == (1,):
            print('FAILURE INITIALIZING X', flush=True)
        else:
            print('X DID NOT CONVERGE', flush=True)
        return False, None, None
    else:
        # Create the random state object
        random_state = np.random.RandomState(seed=0 if seed is None else seed)

        # Define callback frequency
        if max([c.shape[0] for c in counts]) <= 100:
            callback_frequency = 2
        else:
            callback_frequency = 100

        # Setup
        pm1 = PM1(alpha=alpha, beta=beta, max_iter=max_iter, random_state=random_state,
                  init=init, verbose=0, ploidy=ploidy, in_2d=in_2d, X_true=X_true,
                  input_weight=input_weight, lagrange_mult=lagrange_mult, constraints=constraints, homo_init=homo_init, multiscale_factor=multiscale_factor,
                  HSC_lowres_beads=HSC_lowres_beads, as_sparse=as_sparse,
                  modifications=modifications, init_structures=init_structures, translate=translate, rotate=rotate, fix_homo=stepwise_genome__fix_homo)
        fullres_counts_prepped = pm1.prep_counts(counts, lengths, normalize=norm, filter_counts=norm and filter_counts, filter_percentage=filter_percentage,
                                                 mask_fullres0=('mask_fullres0' in modifications), mask_fullresX=('mask_fullresx' in modifications),
                                                 lighter0=('lighter0' in modifications), nonUA0=('nonua0' in modifications))
        if norm and filter_counts:
            try:
                os.makedirs(os.path.join(outdir, 'filtered_counts'))
            except OSError:
                pass
            write_lengths(os.path.join(outdir, 'filtered_counts', 'counts.bed'), lengths)
            for counts_type, c in fullres_counts_prepped:
                c = c.copy()
                if not isinstance(c, np.ndarray):
                    c = c.toarray()
                c[np.isnan(c)] = 0
                write_counts(os.path.join(outdir, 'filtered_counts', '%s.matrix' % counts_type), sparse.coo_matrix(c))
            try:
                plot_counts(os.path.join(outdir, 'filtered_counts'))
            except:
                print('WARNING: Plotting filtered counts failed', flush=True)
        constraints_parsed = pm1.parse_homolog_sep()

        # Save computed biases
        if norm:
            np.savetxt(bias_file, pm1.bias)

        # Save obj for true structure
        if X_true is not None and not null and not (init_structures is not None or translate or rotate):
            obj_for_X_true = pm1.obj_for_X_true()
            pd.Series(obj_for_X_true).to_csv(X_true_obj_file, sep='\t', header=False)

        # Estimate X
        pm1.initialize()
        if not null:
            X = pm1.fit(callback_frequency=callback_frequency)
        else:
            print('GENERATING NULL STRUCTURE', flush=True)
            X = pm1.null()

        X_init = pm1.init_X_
        if translate or rotate:
            if X is not None:
                orient = X.copy()
                np.savetxt(orient_file, orient)
                X = translate_and_rotate(X, lengths=lengths, init_structures=init_structures, translate=translate, rotate=rotate, fix_homo=stepwise_genome__fix_homo)[0].reshape(-1, 3)
                X[pm1.torm] = np.nan
                np.savetxt(orient_init_file, pm1.init_X_)
                X_init = translate_and_rotate(init, lengths=lengths, init_structures=init_structures, translate=translate, rotate=rotate, fix_homo=stepwise_genome__fix_homo)[0].reshape(-1, 3)
                X_init[pm1.torm] = np.nan

        infer_var = {'alpha': pm1.alpha_, 'obj': pm1.obj_, 'seed': seed, 'beta': pm1.beta_}
        for i in range(len(pm1.beta_)):
            infer_var['beta.%s' % ambiguity[i]] = pm1.beta_[i]
        try:
            for homo_sep_constraint in ('homo', 'homodis'):
                if homo_sep_constraint in constraints:
                    infer_var[homo_sep_constraint] = constraints_parsed[homo_sep_constraint]
        except:
            print('oh dear', flush=True)

        if X is None:
            print('FAILURE INITIALIZING X', flush=True)
            X = np.zeros((1))
            np.savetxt(out_fail, X)
            return False, None, None

        else:
            # Save structures
            if not pm1.converged_:
                print('X DID NOT CONVERGE', flush=True)
                np.savetxt(out_fail, X)
            else:
                print('CONVERGED', flush=True)
                np.savetxt(out_file, X)
                np.savetxt(infer_var_file, np.array(list(infer_var.items())), fmt="%s", delimiter="\t")

                if (translate or rotate) or not null:
                    if not (modifications is not None and 'multiscale3' in modifications):
                        X_init = multiscale_optimization.reduce_X_res(X_init, multiscale_factor, lengths)
                    np.savetxt(init_file, X_init)

                if len(pm1.iter_details_) > 0:
                    pd.DataFrame(pm1.iter_details_).to_csv(info_file, sep='\t', index=False)

                # Generate metrics & figures
                if not null:
                    analyze_inferred_X(counts=counts, X=X, out_file=out_file, init_file=init_file, X_true=X_true, lengths=lengths,
                                       ploidy=ploidy, init=init, outdir=outdir, multiscale_factor=multiscale_factor, seed=seed, alpha=pm1.alpha_, modifications=modifications)

            return pm1.converged_, X, infer_var


def choose_best_seed(outdir):
    import glob

    infer_var_files = glob.glob('%s*.txt' % os.path.join(outdir, 'inference_variables'))
    if len(infer_var_files) == 0:
        raise ValueError('No inferred structures found in %s' % outdir)

    var_per_seed = [dict(pd.read_csv(f, sep='\t', header=None, names=('label', 'value')).set_index('label').value) for f in infer_var_files]
    try:
        best_seed_var = [x for x in var_per_seed if x['obj'] == pd.DataFrame(var_per_seed).obj.min()][0]
    except KeyError as e:
        print(e, flush=True)
        print(infer_var_files, flush=True)
        print(pd.DataFrame(var_per_seed), flush=True)
        exit(0)
    return best_seed_var


def choose_X_inferred_file(outdir, seed=None, verbose=True):
    # If multiple inference seeds were used per chromosome, only load results from seed that yielded the best final objective value

    if seed is None:
        best_seed_var = choose_best_seed(outdir)
        if 'seed' in best_seed_var:
            seed = best_seed_var['seed']

    if seed is None or seed == 'None':
        if verbose:
            print('Loading %s' % os.path.basename(outdir))
        return os.path.join(outdir, 'X_inferred.txt')
    else:
        if verbose:
            print('Loading %s from seed %03d' % (os.path.basename(outdir), int(seed)))
        return os.path.join(outdir, 'X_inferred.%03d.txt' % int(seed))


def load_inferred_X(outdir, seed=None, verbose=True):
    return np.loadtxt(choose_X_inferred_file(outdir, seed=seed, verbose=verbose))


def infer_multiscale(counts, lengths, alpha, beta, ploidy, init, outdir, lagrange_mult, constraints, homo_init, input_weight, as_sparse, norm, filter_counts,
                     init_structures=None, translate=False, rotate=False, stepwise_genome__fix_homo=True, lowres_genome_factor=None, initial_seed=0, num_infer=1, num_attempts=1, multiscale_rounds=None, HSC_lowres_beads=5, modifications=None, null=False, filter_percentage=0.04, X_true=None, max_iter=1e40, in_2d=False, redo_analysis=False):

    if modifications is None:
        modifications = []

    if multiscale_rounds is not None:
        if lowres_genome_factor is not None:
            raise ValueError("Must set either multiscale_rounds or lowres_genome_factor, not both")
        all_multiscale_factors = 2 ** np.flip(np.arange(multiscale_rounds), axis=0)
    elif lowres_genome_factor is not None:
        all_multiscale_factors = [lowres_genome_factor]
    else:
        raise ValueError("Must set multiscale_rounds or lowres_genome_factor, both are currently None")

    var_per_seed = []

    i = 0
    for j in range(num_infer):
        converged_at_final_res = False
        prev_attempts = 0
        while prev_attempts < num_attempts and not converged_at_final_res:
            seed = None
            if initial_seed is not None:
                seed = initial_seed + i
                print_code_header('(%d) INFERRING WITH SEED %03d' % (j, seed), max_length=60, blank_lines=1)

            X_init = init
            alpha_ = alpha
            beta_ = beta

            for multiscale_factor in all_multiscale_factors:
                if lowres_genome_factor is not None or (multiscale_rounds is not None and multiscale_rounds > 1):
                    print_code_header('MULTISCALE FACTOR %d' % multiscale_factor, max_length=50, blank_lines=1)

                # Reduce resolution
                lengths_lowres = multiscale_optimization.decrease_lengths_res(lengths, multiscale_factor)
                X_true_lowres = X_true
                if X_true is not None:
                    X_true_lowres = multiscale_optimization.reduce_X_res(X_true, multiscale_factor, lengths)

                # Define output directory
                multiscale_outdir = outdir
                if multiscale_factor != 1 and lowres_genome_factor is None:
                    multiscale_outdir = os.path.join(outdir, 'res-x-%d' % multiscale_factor)

                # Infer!
                if 'old_multiscale' not in modifications:
                    converged, X, infer_var = infer(counts, lengths=lengths, alpha=alpha_, beta=beta_,
                                                    ploidy=ploidy, init=X_init, outdir=multiscale_outdir, seed=seed, homo_init=homo_init, as_sparse=as_sparse,
                                                    norm=norm, filter_counts=filter_counts, filter_percentage=filter_percentage, null=null, modifications=modifications,
                                                    lagrange_mult=lagrange_mult, constraints=constraints, input_weight=input_weight, multiscale_factor=multiscale_factor, HSC_lowres_beads=HSC_lowres_beads,
                                                    init_structures=init_structures, translate=translate, rotate=rotate, stepwise_genome__fix_homo=stepwise_genome__fix_homo,
                                                    max_iter=max_iter, in_2d=in_2d, X_true=X_true_lowres, redo_analysis=redo_analysis)
                else:
                    converged, X, infer_var = infer(prep_counts(counts, lengths, ploidy=ploidy, multiscale_factor=multiscale_factor, verbose=False)[0], lengths=lengths_lowres,
                                                    alpha=alpha_, beta=(None if beta_ is None else list(beta_ * multiscale_factor ** 2)),
                                                    ploidy=ploidy, init=X_init, outdir=multiscale_outdir, seed=seed, homo_init=homo_init, as_sparse=as_sparse,
                                                    norm=norm, filter_counts=filter_counts, filter_percentage=filter_percentage, null=null, modifications=modifications,
                                                    lagrange_mult=lagrange_mult, constraints=constraints, input_weight=input_weight, multiscale_factor=1, HSC_lowres_beads=HSC_lowres_beads,
                                                    init_structures=init_structures, translate=translate, rotate=rotate, stepwise_genome__fix_homo=stepwise_genome__fix_homo,
                                                    max_iter=max_iter, in_2d=in_2d, X_true=X_true_lowres, redo_analysis=redo_analysis)
                alpha_ = infer_var['alpha']
                beta_ = infer_var['beta']

                if not converged:
                    break
                elif multiscale_factor == all_multiscale_factors[-1]:
                    converged_at_final_res = True
                    var_per_seed.append(infer_var)

                X_init = X

            i += 1
            prev_attempts += 1


def save_info_for_X_true(outdir, X_true, infer_lengths, infer_chrom, genome_chrom, multiscale_rounds=None, lowres_genome_factor=None, alpha=-3, modifications=None):
    from topsy.inference.separating_homologs import inter_homolog_ratio, inter_homolog_dis
    from topsy.metrics.utils import neighbor_var, neighbor_distance

    if modifications is None:
        modifications = []

    if len(infer_chrom) != len(genome_chrom):
        outdir = os.path.join(outdir, '.'.join(infer_chrom))

    if multiscale_rounds is not None:
        if lowres_genome_factor is not None:
            raise ValueError("Must set either multiscale_rounds or lowres_genome_factor, not both")
        all_multiscale_factors = 2 ** np.flip(np.arange(multiscale_rounds), axis=0)
    elif lowres_genome_factor is not None:
        all_multiscale_factors = [lowres_genome_factor, 1]
    else:
        raise ValueError("Must set multiscale_rounds or lowres_genome_factor, both are currently None")

    for multiscale_factor in all_multiscale_factors:
        infer_lengths_lowres = multiscale_optimization.decrease_lengths_res(infer_lengths, multiscale_factor)
        X_true_lowres = multiscale_optimization.reduce_X_res(X_true, multiscale_factor, infer_lengths)

        try:
            os.makedirs(outdir)
        except OSError:
            pass

        inter_chr = inter_homolog_ratio(X_true_lowres, lengths=infer_lengths_lowres, counts=None, alpha=(-3 if alpha is None else alpha), modifications=modifications)
        info = pd.Series({'inter_homo_ratio_chr': inter_chr,
                          'inter_homo_dis': inter_homolog_dis(X_true_lowres, infer_lengths_lowres),
                          'neighbor_dist_var': neighbor_var(X_true_lowres, infer_lengths_lowres),
                          'neighbor_dist_mean': np.mean(neighbor_distance(X_true_lowres, infer_lengths_lowres)),
                          'lengths': ",".join(map(str, infer_lengths_lowres)),
                          'chrom': ",".join(map(str, infer_chrom)),
                          'multiscale_factor': multiscale_factor})
        info.to_csv(os.path.join(outdir, 'res-x-%d.txt' % multiscale_factor), sep='\t', header=False)


def choose_lowres_genome_factor(infer_lengths, stepwise_genome__lowres_min_beads):
    multiscale_i = 0
    infer_lengths_lowres = infer_lengths
    lowres_genome_factor = 0
    while infer_lengths_lowres.min() >= stepwise_genome__lowres_min_beads:
        multiscale_i += 1
        lowres_genome_factor = 2 ** multiscale_i
        infer_lengths_lowres = multiscale_optimization.decrease_lengths_res(infer_lengths, lowres_genome_factor)
    return lowres_genome_factor


def inference_output_subdirectory(outdir, genome_chrom, infer_chrom, stepwise_genome__lowres_min_beads, stepwise_genome__fix_homo, stepwise_genome__optimize_orient, stepwise_genome__step, stepwise_genome__chrom=None):
    infer_outdir = outdir
    if isinstance(infer_chrom, str):
        infer_chrom = [infer_chrom]
    if stepwise_genome__step is not None and not isinstance(stepwise_genome__step, list):
        stepwise_genome__step = [stepwise_genome__step]

    if stepwise_genome__step is not None and max(stepwise_genome__step) == 2 and len(infer_chrom) != 1 and stepwise_genome__chrom is not None:
        infer_chrom = [stepwise_genome__chrom]
    if infer_chrom is not None and len(infer_chrom) != len(genome_chrom):
        infer_outdir = os.path.join(infer_outdir, '.'.join(infer_chrom))

    if stepwise_genome__step is not None:
        if max(stepwise_genome__step) == 1:
            infer_outdir = os.path.join(infer_outdir, 'lowres_genome.min%dbeads' % stepwise_genome__lowres_min_beads)
        elif max(stepwise_genome__step) >= 3:
            infer_outdir = os.path.join(infer_outdir, 'stepwise_genome.min%dbeads' % stepwise_genome__lowres_min_beads + ('.fix_homo' if stepwise_genome__fix_homo else '') + ('.no_orient_opt' if not stepwise_genome__optimize_orient else ''))
            if max(stepwise_genome__step) == 3:
                infer_outdir = os.path.join(infer_outdir, 'oriented_chrom')
    return infer_outdir


def infer_3d(counts_files, genome_lengths, alpha, beta, ploidy, init, outdir, lagrange_mult, constraints, homo_init, input_weight, as_sparse, norm, filter_counts,
             genome_chrom=None, infer_chrom=None, HSC_lowres_beads=5,
             stepwise_genome=False, stepwise_genome__step=None, stepwise_genome__chrom=None, stepwise_genome__lowres_min_beads=5, stepwise_genome__fix_homo=True, stepwise_genome__optimize_orient=True,
             initial_seed=0, num_infer=1, num_attempts=1, multiscale_rounds=1, modifications=None, null=False, filter_percentage=0.04, X_true_file=None, max_iter=1e40, in_2d=False, redo_analysis=False):
    from topsy.inference.config import save_params

    counts, X_true, infer_lengths, infer_chrom, genome_lengths, genome_chrom = load_data(counts_files, ploidy, genome_lengths, genome_chrom=genome_chrom, infer_chrom=infer_chrom, as_sparse=as_sparse, X_true=X_true_file)

    if null:
        outdir = os.path.join(outdir, 'null')

    X_true_info_dir = None
    if X_true_file is not None:
        X_true_info_dir = os.path.join(os.path.dirname(X_true_file), 'X_true_info')

    if (not stepwise_genome) or len(infer_chrom) == 1:
        infer_outdir = inference_output_subdirectory(outdir, genome_chrom, infer_chrom, stepwise_genome__lowres_min_beads, stepwise_genome__fix_homo, stepwise_genome__optimize_orient, stepwise_genome__step)

        if X_true_file is not None:
            save_info_for_X_true(X_true_info_dir, X_true, infer_lengths, infer_chrom, genome_chrom, multiscale_rounds=multiscale_rounds, alpha=alpha, modifications=modifications)

        save_params(counts=counts, infer_lengths=infer_lengths, alpha=alpha, beta=beta, ploidy=ploidy, init=init, outdir=infer_outdir, lagrange_mult=lagrange_mult, constraints=constraints, homo_init=homo_init, input_weight=input_weight, as_sparse=as_sparse, norm=norm, filter_counts=filter_counts,
                    infer_chrom=infer_chrom, HSC_lowres_beads=HSC_lowres_beads,
                    lowres_genome_factor=None, multiscale_rounds=multiscale_rounds, modifications=modifications, null=null, filter_percentage=filter_percentage, max_iter=max_iter, in_2d=in_2d)

        infer_multiscale(counts=counts, lengths=infer_lengths, alpha=alpha, beta=beta, ploidy=ploidy, init=init, outdir=infer_outdir, lagrange_mult=lagrange_mult, constraints=constraints, homo_init=homo_init, input_weight=input_weight, as_sparse=as_sparse, norm=norm, filter_counts=filter_counts,
                         initial_seed=initial_seed, num_infer=num_infer, num_attempts=num_attempts, multiscale_rounds=multiscale_rounds, HSC_lowres_beads=HSC_lowres_beads, modifications=modifications, null=null, filter_percentage=filter_percentage, X_true=X_true, max_iter=max_iter, in_2d=in_2d, redo_analysis=redo_analysis)
    else:
        raise NotImplementedError
