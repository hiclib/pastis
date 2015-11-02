import os
import shutil
import stat
import subprocess
import numpy as np

from sklearn.metrics import euclidean_distances
from sklearn.isotonic import IsotonicRegression

from .config import parse
from .optimization import MDS, PM1, PM2
from . import fastio
from .externals import iced

max_iter = 5


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

    return True


def run_nmds(directory):
    raise NotImplementedError
    if os.path.exists(os.path.join(directory,
                                   "config.ini")):
        config_file = os.path.join(directory, "config.ini")
    else:
        config_file = None

    options = parse(config_file)
    run_mds(directory)

    for i in range(0, max_iter):
        if i == 0:
            try:
                X = np.loadtxt(
                    os.path.join(directory,
                                 "MDS." + options["output_name"] + ".txt"))
            except IOError:
                return
        else:
            X = np.loadtxt(
                os.path.join(directory,
                             '%d.NMDS.' % (i) + options["output_name"] +
                             ".txt"))

        X = X.reshape((len(X) / 3, 3))

        dis = euclidean_distances(X) * 1000
        counts = np.load(
            os.path.join(directory, options["counts"]))
        counts[np.isnan(counts)] = 0

        wish_distances = np.zeros(counts.shape)

        print "Fitting isotonic regression..."
        ir = IsotonicRegression()
        wish_distances[counts != 0] = ir.fit_transform(
            1. / counts[counts != 0],
            dis[counts != 0])
        print "writing wish distances"

        lengths = np.loadtxt(
            os.path.join(directory, options["organism_structure"]))

        try:
            len(lengths)
        except TypeError:
            lengths = np.array([lengths])

        write(wish_distances,
              os.path.join(directory,
                           '%d.NMDS.wish_distances.txt' % i),
              lengths=lengths, resolution=options["resolution"])

        if i == 0:
            shutil.copy(
                os.path.join(directory,
                             "MDS." + options["output_name"] + ".txt"),
                os.path.join(directory,
                             '%d.NMDS.' % (i + 1) + options["output_name"] +
                             ".temp.txt"))
        else:
            shutil.copy(
                os.path.join(directory,
                             '%d.NMDS.' % i + options["output_name"] + ".txt"),
                os.path.join(directory,
                             '%d.NMDS.' % (i + 1) + options["output_name"] +
                             ".temp.txt"))

        cmd = CMD_MDS % (options["binary_mds"],
                         os.path.join(directory,
                                      "%d.NMDS." % (i + 1) +
                                      options["output_name"]),
                         options["resolution"],
                         os.path.join(directory,
                                      options["organism_structure"]),
                         os.path.join(directory,
                                      "%d.NMDS.wish_distances.txt" % (i)),
                         options["adjacent_beads"],
                         options["chromosomes"],
                         os.path.join(directory,
                                      str(i + 1) + '.NMDS.log'))

        filename = os.path.join(directory, str(i + 1) + '.NMDS.sh')
        fileptr = open(filename, 'wb')
        fileptr.write(cmd)
        fileptr.close()
        st = os.stat(filename)
        os.chmod(filename, st.st_mode | stat.S_IXUSR)
        subprocess.call(filename.split(), shell='True')


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

    return True


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
    return True
