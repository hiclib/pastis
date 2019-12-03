import os
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from .optimization import MDS, NMDS
from .load_data import load_data
from .counts import preprocess_counts
from .initialization import initialize
from .callbacks import Callback
from .pastis_algorithms import _output_subdir


def mds(counts, nonmetric, lengths=None, outdir='', chromosomes=None,
        chrom_subset=None, alpha=None, seed=0, normalize=True,
        filter_threshold=0.04, init='random', max_iter=10000000000,
        factr=10000000., pgtol=1e-05, callback_function=None, print_freq=100,
        history_freq=100, save_freq=None, struct_true=None, verbose=True):
    """Infer 3D structures with metric or nonmetric multidimensional scaling.

    Infer 3D structure from Hi-C contact counts data for haploid organisms via
    metric or nonmetric multi-dimensional scaling.

    Parameters
    ----------
    counts : list of str
        Counts data files in the hiclib format or as numpy ndarrays.
    nonmetric : bool
        If True, infer a structure with nonmetric MDS. Otherwise, infer a
        structure with metric MDS.
    lengths : str or list
        Number of beads per homolog of each chromosome, or hiclib .bed file with
        lengths data.
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
    max_iter : int, optional
        Maximum number of iterations per optimization.
    factr : float, optional
        factr for scipy's L-BFGS-B, alters convergence criteria.
    pgtol : float, optional
        pgtol for scipy's L-BFGS-B, alters convergence criteria.
    print_freq : int, optional
        Frequency of iterations at which to print during optimization.
    history_freq : int, optional
        Frequency of iterations at which to log objective value and other
        information during optimization.
    save_freq : int, optional
        Frequency of iterations at which to save the 3D structure during
        optimization.
    """

    lengths_full = lengths
    chrom_full = chromosomes
    callback_freq = {'print': print_freq, 'history': history_freq,
                     'save': save_freq}

    counts_raw, lengths_subset, chrom_subset, lengths_full, chrom_full, struct_true = load_data(
        counts=counts, lengths_full=lengths_full, ploidy=1,
        chrom_full=chrom_full, chrom_subset=chrom_subset, exclude_zeros=True,
        struct_true=struct_true)

    outdir = _output_subdir(
        outdir=outdir, chrom_full=chrom_full, chrom_subset=chrom_subset)

    try:
        os.makedirs(outdir)
    except OSError:
        pass
    if seed is None:
        seed_str = ''
    else:
        seed_str = '.%03d' % seed
    if nonmetric:
        algorithm = 'nmds'
    else:
        algorithm = 'mds'
    out_file = os.path.join(
        outdir, '%s.struct_inferred%s.coords' % (algorithm, seed_str))
    history_file = os.path.join(
        outdir, '%s.history%s' % (algorithm, seed_str))
    infer_var_file = os.path.join(
        outdir, '%s.inference_variables%s' % (algorithm, seed_str))
    out_fail = os.path.join(
        outdir, '%s.struct_nonconverged%s.coords' % (algorithm, seed_str))

    if os.path.exists(out_file):
        print('CONVERGED', flush=True)
        exit(0)
    elif os.path.exists(out_fail):
        print('OPTIMIZATION DID NOT CONVERGE', flush=True)
        exit(0)

    random_state = np.random.RandomState(seed)
    random_state = check_random_state(random_state)

    # PREPARE COUNTS OBJECTS
    counts, bias, torm = preprocess_counts(
        counts_raw=counts_raw, lengths=lengths, ploidy=1, normalize=normalize,
        filter_threshold=filter_threshold, exclude_zeros=True, verbose=verbose,
        output_directory=None)

    # INITIALIZATION
    if isinstance(init, str) and init.lower() == 'true':
        if struct_true is None:
            raise ValueError("Attempting to initialize with struct_true but"
                             " struct_true is None")
        print('INITIALIZATION: initializing with true structure', flush=True)
        init = struct_true
    struct_init = initialize(
        counts=counts, lengths=lengths, init=init, ploidy=1,
        random_state=random_state, alpha=alpha, bias=bias, verbose=verbose)

    # INFER STRUCTURE
    callback = Callback(lengths, ploidy=1, counts=counts,
                        analysis_function=callback_function,
                        frequency=callback_freq, directory=outdir,
                        struct_true=struct_true)

    counts_sparse = [c.tocoo() for c in counts][0]
    if nonmetric:
        mds = NMDS(
            counts=counts_sparse, lengths=lengths, alpha=alpha, beta=1.,
            init=struct_init, bias=bias, callback=callback, max_iter=max_iter,
            factr=factr, pgtol=pgtol, verbose=verbose)
    else:
        mds = MDS(
            counts=counts_sparse, lengths=lengths, alpha=alpha, beta=1.,
            init=struct_init, bias=bias, callback=callback, max_iter=max_iter,
            factr=factr, pgtol=pgtol, verbose=verbose)
    mds.fit()
    struct_ = mds.struct_.reshape(-1, 3)
    struct_[torm] = np.nan

    infer_var = {'obj': mds.obj_, 'seed': seed}

    if mds.converged_:
        np.savetxt(out_file, struct_)
        pd.Series(infer_var).to_csv(infer_var_file, sep='\t', header=False)
        if mds.history_ is not None:
            pd.DataFrame(
                mds.history_).to_csv(history_file, sep='\t', index=False)
    else:
        np.savetxt(out_fail, struct_)
