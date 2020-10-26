import numpy as np
import pandas as pd
from scipy import sparse
from warnings import warn
import os
import cooler

def load_cooler_counts(lengths, maternal_maternal=None, paternal_paternal=None,
                maternal_paternal=None, maternal_unknown=None,
                paternal_unknown=None, unknown_unknown=None, haploid=None):
    """
    Load all input data from the given cooler files, and/or reformat data
    objects.

    Parameters
    ----------
    lengths : str or list
        Number of beads per homolog of each chromosome in the inputted data, or
        hiclib .bed file with lengths data.
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

    Returns
    -------
    counts : coo_matrix of int or ndarray or int
    """

    if (haploid == None):  # diploid
        return assemble_diploid(lengths, maternal_maternal=maternal_maternal,
                     paternal_paternal=paternal_paternal,
                     maternal_paternal=maternal_paternal,
                     maternal_unknown=maternal_unknown,
                     paternal_unknown=paternal_unknown,
                     unknown_unknown=unknown)
    else:  # haploid
        return assemble_haploid(haploid=haploid)

def read_cooler(filepath, symmetric=True):
    
    """
    Reads the cooler file given by filepath and returns the counts.
    """
        
    # load cooler file and details
    c = cooler.Cooler(filepath)
    chroms = c.chroms()[:]
    resolution = c.info['bin-size']
    lengths = (chroms.length / resolution + 1).astype(int)
    n = int(lengths.sum())
    
    # get pixels and build counts matrix
    pixels = c.pixels()[:]
    counts = sparse.coo_matrix(
        (pixels.count, (pixels.bin1_id, pixels.bin2_id)),
        shape=(n, n)).toarray()
    np.fill_diagonal(counts, 0)

    if symmetric and c.storage_mode == "square":
        if np.array_equal(counts, counts.T):
            counts = np.triu(counts, 1)
        elif np.tril(counts).sum() != 0:
            warn("Symmetrizing asymmetric counts matrix, pooling counts into"
                 " upper triangular.")
            counts = counts + counts.T

    if not symmetric and c.storage_mode == "symmetric-upper":
        raise ValueError('Cooler storage mode is "symmetric-upper", expected'
                         ' "square".')

    return counts

def assemble_diploid(lengths, maternal_maternal=None, paternal_paternal=None,
                     maternal_paternal=None, maternal_unknown=None,
                     paternal_unknown=None, unknown_unknown=None):
    """
    Assembles the diploid using the given filenames and lengths, and returns
    the diploid.
    """
    # process lengths
    lengths = np.array(lengths).astype(int)
    n = int(lengths.sum())
    
    # diploid unambiguous (fully phased)
    if maternal_maternal is not None and paternal_paternal is not None:
        unambig = np.zeros((n * 2, n * 2))
        unambig[:n, :n] = read_cooler(maternal_maternal, symmetric=True)
        unambig[n:, n:] = read_cooler(paternal_paternal, symmetric=True)
        if maternal_paternal is not None:
            unambig[:n, n:] = read_cooler(maternal_paternal, symmetric=False)
        unambig = sparse.coo_matrix(unambig)
        return unambig
    
    # diploid partially ambiguous
    if maternal_unknown is not None and paternal_unknown is not None:
        partially_ambig = np.zeros((n, n * 2))
        partially_ambig[:, :n] = read_cooler(maternal_unknown, symmetric=False)
        partially_ambig[:, n:] = read_cooler(paternal_unknown, symmetric=False)
        partially_ambig = sparse.coo_matrix(partially_ambig)
        return partially_ambig    

    # diploid ambiguous
    if unknown_unknown is not None:
        ambig = sparse.coo_matrix(read_cooler(unknown_unknown, symmetric=True))
        return ambig

    raise ValueError("Specify diploid files for unambiguous / partially ambig "
                     "/ ambiguous cases")

def assemble_haploid(haploid):
    """
    Assembles and returns the haploid.
    """
    haploid = sparse.coo_matrix(read_cooler(haploid, symmetric=True))
    return haploid
