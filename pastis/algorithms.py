###############################################################################
#
# 160906 Giancarlo Bonora
# Output 3D co-ordinates in PDB format a la coords.cpp
#
###############################################################################

import os
import numpy as np
from scipy import sparse
import iced

from .config import parse
from .optimization import MDS, PM1, PM2, NMDS
from iced.io import load_counts, load_lengths
from .io import writePDB


max_iter = 5


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
        lengths = load_lengths(
            os.path.join(directory,
                         options["lengths"]))
    else:
        lengths = None

    if options["counts"].endswith("npy"):
        counts = np.load(os.path.join(directory, options["counts"]))
    elif options["counts"].endswith(".matrix"):
        counts = load_counts(
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
        lengths = load_lengths(
            os.path.join(directory,
                         options["lengths"]))
    else:
        lengths = None

    if options["counts"].endswith("npy"):
        counts = np.load(os.path.join(directory, options["counts"]))
    elif options["counts"].endswith(".matrix"):
        counts = load_counts(
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
        "NMDS." + options["output_name"] + ".pdb")
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
        lengths = load_lengths(
            os.path.join(directory,
                         options["lengths"]))
    else:
        lengths = None

    if options["counts"].endswith("npy"):
        counts = np.load(os.path.join(directory, options["counts"]))
        counts[np.isnan(counts)] = 0
    elif options["counts"].endswith(".matrix"):
        counts = load_counts(
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
        counts[np.isnan(counts)] = 0
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
        "PM1." + options["output_name"] + ".pdb")
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
        lengths = load_lengths(
            os.path.join(directory,
                         options["lengths"]))
    else:
        lengths = None

    if options["counts"].endswith("npy"):
        counts = np.load(os.path.join(directory, options["counts"]))
        counts[np.arange(len(counts)), np.arange(len(counts))] = 0
    elif options["counts"].endswith(".matrix"):
        counts = load_counts(
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
        counts[np.isnan(counts)] = 0
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

    torm = np.array(((counts + counts.transpose()).sum(axis=0) == 0)).flatten()

    X[torm] = np.nan

    np.savetxt(
        os.path.join(
            directory,
            "PM2." + options["output_name"]),
        X) 
    # PDB file
    pdbfilename = os.path.join(
        directory,
        "PM2." + options["output_name"] + ".pdb")
    # pdbfilename = "test.pdb"
    writePDB(X, pdbfilename)

    return True
