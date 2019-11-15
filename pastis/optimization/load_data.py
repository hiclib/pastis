import numpy as np
import os
from ..externals.iced.io import load_counts, load_lengths


def get_lengths(lengths):
    """Load chromosome lengths from file, or reformat lengths object.
    """

    if isinstance(lengths, str) and os.path.exists(lengths):
        lengths = load_lengths(lengths)
    elif lengths is not None and (isinstance(lengths, list) or isinstance(lengths, np.ndarray)):
        if len(lengths) == 1 and isinstance(lengths[0], str) and os.path.exists(lengths[0]):
            lengths = load_lengths(lengths[0])
    lengths = np.array(lengths).astype(int)
    return lengths


def get_chrom(chrom, lengths):
    """Load chromosome names from file, or reformat chromosome names object.
    """

    lengths = get_lengths(lengths)
    if isinstance(chrom, str) and os.path.exists(chrom):
        chrom = np.array(np.genfromtxt(chrom, dtype='str')).reshape(-1)
    elif chrom is not None and (isinstance(chrom, list) or isinstance(chrom, np.ndarray)):
        if len(chrom) == 1 and isinstance(chrom[0], str) and os.path.exists(chrom[0]):
            chrom = np.array(np.genfromtxt(chrom[0], dtype='str')).reshape(-1)
        chrom = np.array(chrom)
    else:
        chrom = np.array(['num%d' % i for i in range(1, len(lengths) + 1)])
    return chrom


def get_counts(counts, lengths):
    """Load counts from file, or reformat counts object.
    """

    if not isinstance(counts, list):
        counts = [counts]
    lengths = get_lengths(lengths)
    output = []
    for f in counts:
        if isinstance(f, np.ndarray):
            counts_maps = f
        if f.endswith(".npy"):
            counts_maps = np.load(f)
            counts_maps[np.isnan(counts_maps)] = 0
        elif f.endswith(".matrix"):
            counts_maps = load_counts(f, lengths=lengths)
        else:
            raise ValueError("Counts file must end with .npy (for numpy array)"
                             " or .matrix (for hiclib / iced format)")
        output.append(counts_maps)
    return output


def load_data(counts, ploidy, lengths_full, chrom_full=None,
              chrom_subset=None, as_sparse=True, struct_true=None):
    """Load all input data from file, or reformat data objects.
    """

    from .counts import subset_chrom

    lengths_full = get_lengths(lengths_full)
    chrom_full = get_chrom(chrom_full, lengths_full)
    counts = get_counts(counts, lengths_full)

    if struct_true is not None and isinstance(struct_true, str):
        struct_true = np.loadtxt(struct_true)

    counts, struct_true, lengths_subset, chrom_subset = subset_chrom(
        counts=counts, ploidy=ploidy, lengths_full=lengths_full,
        chrom_full=chrom_full, chrom_subset=chrom_subset, as_sparse=as_sparse,
        struct_true=struct_true)

    return counts, struct_true, lengths_subset, chrom_subset, lengths_full, chrom_full


def choose_best_seed(outdir):
    """Choose seed with the lowest final objective value.
    """

    import pandas as pd
    import glob

    infer_var_files = glob.glob(
        '%s*.txt' % os.path.join(outdir, 'inference_variables'))
    if len(infer_var_files) == 0:
        raise ValueError('No inferred structures found in %s' % outdir)

    var_per_seed = [dict(pd.read_csv(f, sep='\t', header=None, names=(
        'label', 'value')).set_index('label').value) for f in infer_var_files]
    try:
        best_seed_var = [x for x in var_per_seed if x['obj']
                         == pd.DataFrame(var_per_seed).obj.min()][0]
    except KeyError as e:
        print(e, flush=True)
        print(infer_var_files, flush=True)
        print(pd.DataFrame(var_per_seed), flush=True)
        exit(0)
    return best_seed_var


def choose_struct_inferred_file(outdir, seed=None, verbose=True):
    """Choose inferred structure with the lowest final objective value.
    """

    if seed is None:
        best_seed_var = choose_best_seed(outdir)
        if 'seed' in best_seed_var:
            seed = best_seed_var['seed']

    if seed is None or seed == 'None':
        if verbose:
            print('Loading %s' % os.path.basename(outdir))
        return os.path.join(outdir, 'struct_inferred.txt')
    else:
        if verbose:
            print('Loading %s from seed %03d' %
                  (os.path.basename(outdir), int(seed)))
        return os.path.join(outdir, 'struct_inferred.%03d.txt' % int(seed))


def load_inferred_struct(outdir, seed=None, verbose=True):
    """Load inferred structure with the lowest final objective value.
    """

    return np.loadtxt(
        choose_struct_inferred_file(outdir, seed=seed, verbose=verbose))
