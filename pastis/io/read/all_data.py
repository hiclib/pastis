import numpy as np
import os
from scipy import sparse
from iced.io import load_lengths
from .hiclib import load_hiclib_counts
from .cooler import load_cooler_counts
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


def _get_chrom(chrom, lengths):
    """Load chromosome names from file, or reformat chromosome names object.
    """

    lengths = _get_lengths(lengths)
    if isinstance(chrom, str) and os.path.exists(chrom):
        chrom = np.array(np.genfromtxt(chrom, dtype='str')).reshape(-1)
    elif chrom is not None and (isinstance(chrom, list) or isinstance(chrom, np.ndarray)):
        if len(chrom) == 1 and isinstance(chrom[0], str) and os.path.exists(chrom[0]):
            chrom = np.array(np.genfromtxt(chrom[0], dtype='str')).reshape(-1)
        chrom = np.array(chrom)
    else:
        chrom = np.array(['num%d' % i for i in range(1, len(lengths) + 1)])
    return chrom


def _get_counts(lengths, maternal_maternal=None, paternal_paternal=None,
                maternal_paternal=None, maternal_unknown=None,
                paternal_unknown=None, unknown_unknown=None, haploid=None,
                counts=None):
    """Load counts from file(s), or reformat counts object.
    """

    if (counts != None) & (not isinstance(counts, list)):
        counts = [counts]
    lengths = _get_lengths(lengths)
    output = []
    if (counts != None):  # not cooler 
        for f in counts:
            if isinstance(f, np.ndarray) or sparse.issparse(f):
                counts_maps = f
            elif f.endswith(".npy"):
                 counts_maps = np.load(f)
            elif f.endswith(".matrix"):
                 counts_maps = load_hiclib_counts(f, lengths=lengths)
            else:
                  raise ValueError("Counts file must end with .npy (for numpy array)"
                                   " or .matrix (for hiclib / iced format) or .cool"
                                   " (for cooler files)")
            if sparse.issparse(counts_maps):
                 counts_maps = counts_maps.toarray()
                 counts_maps[np.isnan(counts_maps)] = 0
            output.append(sparse.coo_matrix(counts_maps))
        return output
    else: # cooler files
        if (((maternal_maternal != None) and
            (not maternal_maternal.endswith(".cool"))) or
           ((paternal_paternal != None) and
            (not paternal_paternal.endswith(".cool"))) or
           ((maternal_paternal != None) and
            (not maternal_paternal.endswith(".cool"))) or
           ((maternal_unknown != None) and
            (not maternal_unknown.endswith(".cool"))) or
           ((paternal_unknown != None) and
            (not paternal_unknown.endswith(".cool"))) or
           ((unknown_unknown != None) and
            (not unknown_unknown.endswith(".cool"))) or
           ((haploid != None) and
            (not haploid.endswith(".cool")))):
           raise ValueError("Counts file must end with .npy (for numpy array)"
                                   " or .matrix (for hiclib / iced format) or .cool"
                                   " (for cooler files)")
        cooler_counts = load_cooler_counts(lengths=lengths,
           maternal_maternal=maternal_maternal,
           paternal_paternal=paternal_paternal,
           maternal_paternal=paternal_paternal,
           maternal_unknown=maternal_unknown,
           paternal_unknown=paternal_unknown,
           unknown_unknown=unknown, haploid=haploid)
        if sparse.issparse(cooler_counts):
             cooler_counts = cooler_counts.toarray()
             cooler_counts[np.isnan(cooler_counts)] = 0
        return cooler_counts

def load_data(lengths_full, ploidy, maternal_maternal=None,
              paternal_paternal=None, maternal_paternal=None,
              maternal_unknown=None, paternal_unknown=None,
              unknown_unknown=None, haploid=None, counts=None, chrom_full=None,
              chrom_subset=None, exclude_zeros=False, struct_true=None):
    """Load all input data from files, and/or reformat data objects.

    If files are provided, load data from files. Also reformats data objects.

    Parameters
    ----------
    lengths_full : str or list
        Number of beads per homolog of each chromosome in the inputted data, or
        hiclib .bed file with lengths data.
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    maternal_maternal: str
        Maternal maternal (for diploid unambiguous) counts data file in the
        cooler format
    paternal_paternal: str
        Paternal paternal (for diploid unambiguous) counts data file in the
        cooler format
    maternal_paternal: str
        Maternal paternal (for diploid unambiguous) counts data file in the
        cooler format
    maternal_unknown: str
        Maternal unknown (for diploid partially ambiguous) counts data file in
        the cooler format
    paternal_unknown: str
        Paternal unknown (for diploid partially ambiguous) counts data file in
        the cooler format
    unknown_unknown: str
        Unknown unknown (for diploid ambiguous) counts data file in the cooler
        format
    haploid: str
        Haploid counts data file in the cooler format
    counts : list of str or list of array or list of coo_matrix
        Counts data files in the hiclib format or as numpy ndarrays.
    chrom_full : str or list of str, optional
        Label for each chromosome in the in the inputted data, or file with
        chromosome labels (one label per line).
    chrom_subset : list of str, optional
        Label for each chromosome to be excised from the full data; labels of
        chromosomes for which inference should be performed.

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
    counts = _get_counts(lengths=lengths_full,
                maternal_maternal=maternal_maternal,
                paternal_paternal=paternal_paternal,
                maternal_paternal=paternal_paternal,
                maternal_unknown=maternal_unknown,
                paternal_unknown=paternal_unknown,
                unknown_unknown=unknown, haploid=haploid, counts=counts)

    if struct_true is not None and isinstance(struct_true, str):
        struct_true = np.loadtxt(struct_true)

    lengths_subset, chrom_subset, counts, struct_true = subset_chrom(
        counts=counts, ploidy=ploidy, lengths_full=lengths_full,
        chrom_full=chrom_full, chrom_subset=chrom_subset,
        exclude_zeros=exclude_zeros, struct_true=struct_true)

    return counts, lengths_subset, chrom_subset, lengths_full, chrom_full, struct_true
