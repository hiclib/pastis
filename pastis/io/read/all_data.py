import numpy as np
import os
from scipy import sparse
from iced.io import load_lengths
from .hiclib import load_hiclib_counts
from ...optimization.counts import subset_chrom


def _get_lengths(lengths):
    """Load chromosome lengths from file, or reformat lengths object.
    """

    if isinstance(lengths, str) and os.path.exists(lengths):
        lengths = load_lengths(lengths)
    elif lengths is not None and (isinstance(lengths, list) or isinstance(lengths, np.ndarray)):
        if len(lengths) == 1 and isinstance(lengths[0], str) and os.path.exists(lengths[0]):
            lengths = load_lengths(lengths[0])
    lengths = np.array(lengths).astype(int)
    return lengths

    if lengths is not None:
        if (isinstance(lengths, list) or isinstance(lengths, np.ndarray)) \
          and len(lengths) == 1:
            lengths = lengths[0]
        if isinstance(lengths, str):
            if os.path.exists(lengths):
                lengths = load_lengths(lengths)
            else:
                raise ValueError("Path to lengths does not exist.")
    lengths = np.array(lengths).astype(int)
    return lengths


def _get_chrom(chrom, lengths):
    """Load chromosome names from file, or reformat chromosome names object.
    """

    lengths = _get_lengths(lengths)
    if chrom is not None:
        if (isinstance(chrom, list) or isinstance(chrom, np.ndarray)) \
          and len(chrom) == 1:
            chrom = chrom[0]
        if isinstance(chrom, str):
            if os.path.exists(chrom):
                chrom = np.genfromtxt(chrom, dtype='str')
            else:
                raise ValueError("Path to chrom does not exist.")
        chrom = np.array(chrom).reshape(-1)
    else:
        chrom = np.array(['num%d' % i for i in range(1, len(lengths) + 1)])
    return chrom


def _get_counts(counts, lengths):
    """Load counts from file, or reformat counts object.
    """

    if not isinstance(counts, list):
        counts = [counts]
    lengths = _get_lengths(lengths)
    output = []
    for f in counts:
        if isinstance(f, np.ndarray) or sparse.issparse(f):
            counts_maps = f
        elif f.endswith(".npy"):
            counts_maps = np.load(f)
        elif f.endswith(".matrix"):
            counts_maps = load_hiclib_counts(f, lengths=lengths)
        else:
            raise ValueError("Counts file must end with .npy (for numpy array)"
                             " or .matrix (for hiclib / iced format)")
        if sparse.issparse(counts_maps):
            counts_maps = counts_maps.toarray()
        counts_maps[np.isnan(counts_maps)] = 0
        output.append(sparse.coo_matrix(counts_maps))
    return output


def _get_bias(bias):
    """Load bias from file, or reformat bias object.
    """

    if bias is not None:
        if (isinstance(bias, list) or isinstance(bias, np.ndarray)) \
          and len(bias) == 1:
            bias = bias[0]
        if isinstance(bias, str):
            if os.path.exists(bias):
                if bias.endswith(".npy"):
                    bias = np.load(bias)
                else:
                    bias = np.loadtxt(bias)
            else:
                raise ValueError("Path to bias vector does not exist.")
        bias = np.array(bias).astype(float)
    return bias


def load_data(counts, lengths_full, ploidy, chrom_full=None,
              chrom_subset=None, exclude_zeros=False, struct_true=None,
              bias=None):
    """Load all input data from files, and/or reformat data objects.

    If files are provided, load data from files. Also reformats data objects.

    Parameters
    ----------
    counts : list of str or list of array or list of coo_matrix
        Counts data files in the hiclib format or as numpy ndarrays.
    lengths_full : str or list
        Number of beads per homolog of each chromosome in the inputted data, or
        hiclib .bed file with lengths data.
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    chrom_full : str or list of str, optional
        Label for each chromosome in the in the inputted data, or file with
        chromosome labels (one label per line).
    chrom_subset : list of str, optional
        Label for each chromosome to be excised from the full data; labels of
        chromosomes for which inference should be performed.
    bias : str, optional
        The path to the bias vector to normalize the counts with.

    Returns
    -------
    counts : coo_matrix of int or ndarray or int
        Counts data. If `chrom_subset` is not None, only counts data for the
        specified chromosomes are returned.
    lengths_subset : array of int
        Number of beads per homolog of each chromosome in the returned data. If
        `chrom_subset` is not None, only chromosome lengths for the specified
        chromosomes are returned.
    chrom_subset : array of str
        Label for each chromosome in the returned data; labels of chromosomes
        for which inference should be performed.
    lengths_full : array of int
        Number of beads per homolog of each chromosome in the inputted data.
    chrom_full : array of str
        Label for each chromosome in the inputted data.
    struct_true : None or array of float
        The true structure. If `chrom_subset` is not None, only beads for the
        specified chromosomes are returned.
    """

    lengths_full = _get_lengths(lengths_full)
    chrom_full = _get_chrom(chrom_full, lengths_full)
    counts = _get_counts(counts, lengths_full)
    bias = _get_bias(bias)

    if struct_true is not None and isinstance(struct_true, str):
        struct_true = np.loadtxt(struct_true)

    lengths_subset, chrom_subset, counts, struct_true = subset_chrom(
        counts=counts, ploidy=ploidy, lengths_full=lengths_full,
        chrom_full=chrom_full, chrom_subset=chrom_subset,
        exclude_zeros=exclude_zeros, struct_true=struct_true)

    return counts, lengths_subset, chrom_subset, lengths_full, chrom_full, struct_true, bias
