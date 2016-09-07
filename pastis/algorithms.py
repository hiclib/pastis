###############################################################################
#
# 160906 Giancarlo Bonora
# Output 3D co-ordinates in PDB format a la coords.cpp
#
###############################################################################

import os
import numpy as np
from scipy import sparse

from .config import parse
from .optimization import MDS, PM1, PM2, NMDS
from . import fastio
from .externals import iced

max_iter = 5

###############################################################################
# Python implementation of 'print_pdb_atom' function in coords.cpp

def fprintf(fp, fmt, *args):
    fp.write(fmt % args)

def print_pdb_atom (outfile,
                    #chrom_index,
                    #copy_index,
                    atom_index,
                    is_node, # Is this a node or an edge atom
                    atom_name, # eg "N", "O", or "C"
                    scale_factor,
                    my_coords):
    # prev_chrom_index = -1
    # atom_index
    # if chrom_index != prev_chrom_index:
    #     atom_index = 1
    #     prev_chrom_index = chrom_index

    # http://www.biochem.ucl.ac.uk/~roman/procheck/manual/manappb.html
    fprintf(outfile, "ATOM  ")              #  1- 6: Record ID
    fprintf(outfile, "%5d", atom_index)     #  7-11: Atom serial number
    fprintf(outfile, " ")                   #    12: Blank
    # if (is_node) {                        # 13-16: Atom name
    #   fprintf(outfile, "N   ")
    # } else {
    #   fprintf(outfile, "O   ")
    # }
    fprintf(outfile,"%s   ",atom_name)
    fprintf(outfile, " ")                   # 17-17: Alternative location code
    if is_node:                             # 18-20: 3-letter amino acid code
        fprintf(outfile, "NOD")
    else:
        fprintf(outfile, "EDG")
    fprintf(outfile, " ")                   #    21: Blank
    fprintf(outfile, "%c",                  #    22: Chain identifier code
            # get_chrom_id(chrom_index, copy_index))
            'A')
    fprintf(outfile, "    ")                # 23-26: Residue sequence number
    fprintf(outfile, " ")                   #    27: Insertion code
    fprintf(outfile, "   ")                 # 28-30: Blank
    fprintf(outfile, "%8.3f%8.3f%8.3f",     # 31-54: Atom coordinates
            # (my_coords->x + 1.0) * SCALE_FACTOR,
            # (my_coords->y + 1.0) * SCALE_FACTOR,
            # (my_coords->z + 1.0) * SCALE_FACTOR)
            (my_coords[0] + 1.0) * scale_factor,
            (my_coords[1] + 1.0) * scale_factor,
            (my_coords[2] + 1.0) * scale_factor)
    fprintf(outfile, "%6.2f", 1.0)          # 55-60: Occupancy value
    if is_node:
        fprintf(outfile, "%6.2f", 50.0)     # 61-66: B-value (thermal factor)
    else:
        fprintf(outfile, "%6.2f", 75.0)     # 61-66: B-value (thermal factor)
    fprintf(outfile, " ")                   #    67: Blank
    fprintf(outfile, "   ")                 # 68-70: Blank
    fprintf(outfile, "\n")

def writePDB(Xpdb, pdbfilename):
    print('PDB file creation!')
    pdboutfile = open(pdbfilename,'w')
    atom_name = 'O'
    scale_factor = 100 # 100 for a sphere with radius 1, 1000 for a sphere with radius 10
    for coordIdx in range(0, np.shape(Xpdb)[0]):
        print coordIdx
        if coordIdx == 0 or coordIdx == np.shape(Xpdb)[0]:
            is_node = True
        else:
            is_node = False
        my_coords = Xpdb[coordIdx,:]
        print_pdb_atom(pdboutfile,
                       #chrom_index,
                       #copy_index,
                       coordIdx,
                       is_node, # Is this a node or an edge atom
                       atom_name, # eg "N", "O", or "C",
                       scale_factor,
                       my_coords)
    pdboutfile.close()


###############################################################################

def run_mds(directory):
    if os.path.exists(os.path.join(directory,
                                   "config.ini")):
        config_file = os.path.join(directory, "config.ini")
    else:
        config_file = None

    options = parse(config_file)

    random_state = np.random.RandomState(seed=options["seed"])

    # First, compute MDS
    if options["lengths"].endswith(".bed"):
        lengths = fastio.load_lengths(
            os.path.join(directory,
                         options["lengths"]))
    else:
        lengths = None

    if options["counts"].endswith("npy"):
        counts = np.load(os.path.join(directory, options["counts"]))
    elif options["counts"].endswith(".matrix"):
        counts = fastio.load_counts(
            os.path.join(directory,
                         options["counts"]),
            lengths=lengths)

    if options["normalize"]:
        counts = iced.filter.filter_low_counts(counts, sparsity=False,
                                               percentage=0.04)
        counts = iced.normalization.ICE_normalization(
            counts,
            max_iter=300)

    if not sparse.issparse(counts):
        counts = sparse.coo_matrix(counts)
    else:
        counts = counts.tocsr()
        counts.eliminate_zeros()
        counts = counts.tocoo()

    mds = MDS(alpha=options["alpha"],
              beta=options["beta"],
              random_state=random_state,
              max_iter=options["max_iter"],
              verbose=options["verbose"])
    X = mds.fit(counts)
    torm = np.array((counts.sum(axis=0) == 0)).flatten()
    X[torm] = np.nan
    np.savetxt(
        os.path.join(
            directory,
            "MDS." + options["output_name"]),
        X)

    # PDB file
    pdbfilename = os.path.join(
        directory,
        "MDS." + options["output_name"] + ".pdb")
    # pdbfilename = "test.pdb"
    writePDB(X, pdbfilename)

    return True


###############################################################################

def run_nmds(directory):
    if os.path.exists(os.path.join(directory,
                                   "config.ini")):
        config_file = os.path.join(directory, "config.ini")
    else:
        config_file = None

    options = parse(config_file)

    random_state = np.random.RandomState(seed=options["seed"])

    # First, compute MDS
    if options["lengths"].endswith(".bed"):
        lengths = fastio.load_lengths(
            os.path.join(directory,
                         options["lengths"]))
    else:
        lengths = None

    if options["counts"].endswith("npy"):
        counts = np.load(os.path.join(directory, options["counts"]))
    elif options["counts"].endswith(".matrix"):
        counts = fastio.load_counts(
            os.path.join(directory,
                         options["counts"]),
            lengths=lengths)

    if options["normalize"]:
        counts = iced.filter.filter_low_counts(counts, sparsity=False,
                                               percentage=0.04)
        counts = iced.normalization.ICE_normalization(
            counts,
            max_iter=300)

    if not sparse.issparse(counts):
        counts = sparse.coo_matrix(counts)
    else:
        counts = counts.tocsr()
        counts.eliminate_zeros()
        counts = counts.tocoo()

    torm = np.array((counts.sum(axis=0) == 0)).flatten()
    nmds = NMDS(alpha=options["alpha"],
                beta=options["beta"],
                random_state=random_state,
                max_iter=options["max_iter"],
                verbose=options["verbose"])
    X = nmds.fit(counts)

    X[torm] = np.nan
    np.savetxt(
        os.path.join(
            directory,
            "NMDS." + options["output_name"]),
        X)

    # PDB file
    pdbfilename = os.path.join(
        directory,
        "MDS." + options["output_name"] + ".pdb")
    # pdbfilename = "test.pdb"
    writePDB(X, pdbfilename)

    return True


###############################################################################

def run_pm1(directory):
    if os.path.exists(os.path.join(directory,
                                   "config.ini")):
        config_file = os.path.join(directory, "config.ini")
    else:
        config_file = None

    options = parse(config_file)

    random_state = np.random.RandomState(seed=options["seed"])

    options = parse(config_file)

    if options["lengths"].endswith(".bed"):
        lengths = fastio.load_lengths(
            os.path.join(directory,
                         options["lengths"]))
    else:
        lengths = None

    if options["counts"].endswith("npy"):
        counts = np.load(os.path.join(directory, options["counts"]))
    elif options["counts"].endswith(".matrix"):
        counts = fastio.load_counts(
            os.path.join(directory,
                         options["counts"]),
            lengths=lengths)

    if options["normalize"]:
        counts = iced.filter.filter_low_counts(counts, sparsity=False,
                                               percentage=0.04)

        _, bias = iced.normalization.ICE_normalization(
            counts,
            max_iter=300,
            output_bias=True)
    else:
        bias = None

    if not sparse.issparse(counts):
        counts = sparse.coo_matrix(counts)
    else:
        counts = counts.tocsr()
        counts.eliminate_zeros()
        counts = counts.tocoo()

    pm1 = PM1(alpha=options["alpha"],
              beta=options["beta"],
              random_state=random_state,
              max_iter=options["max_iter"],
              bias=bias,
              verbose=options["verbose"])
    X = pm1.fit(counts)
    torm = np.array((counts.sum(axis=0) == 0)).flatten()

    X[torm] = np.nan

    np.savetxt(
        os.path.join(
            directory,
            "PM1." + options["output_name"]),
        X)

    # PDB file
    pdbfilename = os.path.join(
        directory,
        "MDS." + options["output_name"] + ".pdb")
    # pdbfilename = "test.pdb"
    writePDB(X, pdbfilename)

    return True


###############################################################################

def run_pm2(directory):
    if os.path.exists(os.path.join(directory,
                                   "config.ini")):
        config_file = os.path.join(directory, "config.ini")
    else:
        config_file = None

    options = parse(config_file)

    random_state = np.random.RandomState(seed=options["seed"])

    options = parse(config_file)

    if options["lengths"].endswith(".bed"):
        lengths = fastio.load_lengths(
            os.path.join(directory,
                         options["lengths"]))
    else:
        lengths = None

    if options["counts"].endswith("npy"):
        counts = np.load(os.path.join(directory, options["counts"]))
        counts[np.arange(len(counts)), np.arange(len(counts))] = 0
        counts = sparse.coo_matrix(np.triu(counts))
    elif options["counts"].endswith(".matrix"):
        counts = fastio.load_counts(
            os.path.join(directory, options["counts"]),
            lengths=lengths)

    if options["normalize"]:
        counts = iced.filter.filter_low_counts(counts, sparsity=False,
                                               percentage=0.04)

        _, bias = iced.normalization.ICE_normalization(
            counts,
            max_iter=300,
            output_bias=True)
    else:
        bias = None

    if not sparse.issparse(counts):
        counts = sparse.coo_matrix(counts)
    else:
        counts = counts.tocsr()
        counts.eliminate_zeros()
        counts = counts.tocoo()

    pm2 = PM2(alpha=options["alpha"],
              beta=options["beta"],
              random_state=random_state,
              max_iter=options["max_iter"],
              bias=bias,
              verbose=options["verbose"])
    X = pm2.fit(counts)

    torm = np.array((counts.sum(axis=0) == 0)).flatten()

    X[torm] = np.nan

    np.savetxt(
        os.path.join(
            directory,
            "PM2." + options["output_name"]),
        X)

    # PDB file
    pdbfilename = os.path.join(
        directory,
        "MDS." + options["output_name"] + ".pdb")
    # pdbfilename = "test.pdb"
    writePDB(X, pdbfilename)

    return True
