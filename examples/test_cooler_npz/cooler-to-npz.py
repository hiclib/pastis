#! /usr/bin/env python

import numpy as np
import pandas as pd
from scipy import sparse
from warnings import warn
import os

def save_counts_npz(counts, filename):
    """
    Converts the given counts file to the hiclib format, and stores the hiclib
    counts in the file specified by filename. Counts must be a scipy.coo.matrix
    . If the file already exists, overwrites it.
    """
     
    sparse.save_npz(filename, counts)

def read_cooler(filepath, symmetric=True):
    """
    Reads the cooler file given by filepath and returns the counts.
    """

    import cooler
    # load cooler file and details
    c = cooler.Cooler(filepath)
    chroms = c.chroms()[:]
    resolution = c.info['bin-size']
    lengths = (chroms.length / resolution + 1).astype(int)
    n = int(lengths.sum())

    # get pixels and build counts matrix
    pixels = c.pixels()[:]
    counts = sparse.coo_matrix((pixels['count'],
                               (pixels['bin1_id'], pixels['bin2_id'])),
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
    from iced.io import load_lengths
    lengths = load_lengths(lengths);
    lengths = np.array(lengths).astype(int)
    n = int(lengths.sum())
    
    # diploid unambiguous (fully phased)
    if maternal_maternal is not None and paternal_paternal is not None:
        unambig = np.zeros((n * 2, n * 2))
        unambig[:n, :n] = read_cooler(maternal_maternal[0], symmetric=True)
        unambig[n:, n:] = read_cooler(paternal_paternal[0], symmetric=True)
        if maternal_paternal is not None:
            unambig[:n, n:] = read_cooler(maternal_paternal[0], symmetric=False)
        unambig = sparse.coo_matrix(unambig)
        return unambig

    # diploid partially ambiguous
    if maternal_unknown is not None and paternal_unknown is not None:
        partially_ambig = np.zeros((n, n * 2))
        partially_ambig[:, :n] = read_cooler(maternal_unknown[0], symmetric=False)
        partially_ambig[:, n:] = read_cooler(paternal_unknown[0], symmetric=False)
        partially_ambig = sparse.coo_matrix(partially_ambig)
        return partially_ambig

    # diploid ambiguous
    if unknown_unknown is not None:
        ambig = sparse.coo_matrix(read_cooler(unknown_unknown[0], symmetric=True))
        return ambig

    raise ValueError("Specify diploid files for unambiguous / partially ambig "
                     "/ ambiguous cases")

def assemble_haploid(haploid):
    """
    Assembles and returns the haploid.
    """
    haploid = sparse.coo_matrix(read_cooler(haploid, symmetric=True))
    return haploid

def load_cooler_counts(lengths, maternal_maternal=None, paternal_paternal=None,
                maternal_paternal=None, maternal_unknown=None,
                paternal_unknown=None, unknown_unknown=None, haploid=None):
    """
    Load all input data from the given cooler files, and/or reformat data
    objects.
    """

    if (haploid == None):  # diploid
        return assemble_diploid(lengths[0], maternal_maternal=maternal_maternal,
                     paternal_paternal=paternal_paternal,
                     maternal_paternal=maternal_paternal,
                     maternal_unknown=maternal_unknown,
                     paternal_unknown=paternal_unknown,
                     unknown_unknown=unknown_unknown)
    else:  # haploid
        return assemble_haploid(haploid=haploid[0])

def main():
    """
    Load all input data from the given cooler files, and/or reformat data
    objects. Then, saves the counts file in a .npz file. This .npz filed
    can be used as a counts file when running PASTIS.

    Parameters
    ----------
    lengths : str or list
        Number of beads per homolog of each chromosome in the inputted data, or
        hiclib .bed file with lengths data.
    output_file : str
        Name of file to store the hiclib counts. File name must end in .npz. If
        the file already exists, it will be overwritten.
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
    """
    
    # parse arguments
    import argparse
    parser = argparse.ArgumentParser(
             description="Convert cooler counts files to the hiclib format."
                         " Required to pass in a lengths file as well as all"
                         " the counts files for the specific case (diploid "
                         " unambiguous, diploid partially ambiguous, diploid "
                         " ambiguous, or haploid)")
    parser.add_argument("--lengths", nargs="+", type=str, required=True,
                    help="Number of beads per homolog of each chromosome.")
    parser.add_argument("--output_file", nargs="+", type=str, required=True,
                    help="Name of file to store the hiclib counts. File name"
                    " must end in .npz. If the file already exists, it will"
                    " be overwritten.")
    parser.add_argument("--maternal_maternal", nargs="+", type=str,
                    required=False, default=None, help="Maternal maternal"
                    " (diploid unambiguous) counts data file in cooler"
                    " format")
    parser.add_argument("--paternal_paternal", nargs="+", type=str,
                    required=False, default=None, help="Paternal paternal"
                    " (diploid unambiguous) counts data file in cooler"
                    " format")
    parser.add_argument("--maternal_paternal", nargs="+", type=str,
                    required=False, default=None, help="Maternal paternal"
                    " (diploid unambiguous) counts data file in cooler"
                    " format. Not strictly required for diploid unambiguous"
                    " case.")
    parser.add_argument("--maternal_unknown", nargs="+", type=str,
                    required=False, default=None, help="Maternal unknown"
                    " (diploid partially ambiguous) counts data file in cooler"
                    " format")
    parser.add_argument("--paternal_unknown", nargs="+", type=str,
                    required=False, default=None, help="Paternal unknown"
                    " (diploid partially ambiguous) counts data file in cooler"
                    " format")
    parser.add_argument("--unknown_unknown", nargs="+", type=str,
                    required=False, default=None, help="Unknown unknown"
                    " (diploid ambiguous) counts data file in cooler format")
    parser.add_argument("--haploid", nargs="+", type=str, required=False,
                    default=None, help="Haploid counts data file in cooler"
                    " format")
    args = parser.parse_args()

    # get counts
    counts = load_cooler_counts(args.lengths, args.maternal_maternal,
                                args.paternal_paternal, args.maternal_paternal,
                                args.maternal_unknown, args.paternal_unknown,
                                args.unknown_unknown, args.haploid)

    # save counts in a .npz file
    save_counts_npz(counts, args.output_file[0])

if __name__ == "__main__":
    main()
