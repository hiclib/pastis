import numpy as np
from scipy import sparse
import pandas as pd


def load_counts(filename, lengths=None):
    """
    Fast loading of a raw interaction counts file

    Parameters
    ----------
    filename : str
        path to the file to load. The file should be of the following format:
        i, j, counts

    Returns
    --------
    X : the interaction counts file
    """
    n = None
    if lengths is not None:
        n = lengths.sum()
        shape = (n, n)
    else:
        shape = None
    # This is the interaction count files
    dataframe = pd.read_csv(filename, sep="\t", header=None)
    row, col, data = dataframe.as_matrix().T
    # XXX We need to deal with the fact that we should not duplicate entries
    # for the diagonal.
    # XXX what if n doesn't exist?
    if (col.min() >= 1 and row.min() >= 1) and \
       ((n is None) or (col.max() == n)):
        # This is a hack to deal with the fact that sometimes, the files are
        # indexed at 1 and not 0
        col -= 1
        row -= 1

    data = data.astype(float)
    counts = sparse.coo_matrix((data, (row, col)), shape=shape)
    return counts


def load_lengths(filename):
    """
    Fast loading of the bed files

    Parameters
    ----------
    filename : str,
        path to the file to load. The file should be a bed file

    Returns
    -------
    lengths : the lengths of each chromosomes
    """
    data = pd.read_csv(filename, sep="\t", header=None)
    data = data.as_matrix()
    lengths = [(data[:, 0] == i).sum() for i in np.unique(data[:, 0])]
    return np.array(lengths)


def write_counts(filename, counts):
    """
    Write counts

    Parameters
    ----------

    filename : str

    counts: array-like
    """
    if not sparse.isspmatrix_coo(counts):
        if sparse.issparse(counts):
            counts = counts.tocoo()
        else:
            counts = sparse.coo_matrix(counts)
    # XXX this is slow and memory intensive
    data = np.concatenate([counts.row[:, np.newaxis],
                           counts.col[:, np.newaxis],
                           counts.data[:, np.newaxis]], axis=1)
    np.savetxt(filename, data, fmt="%d\t%d\t%f")


def write_lengths(filename, lengths, resolution=1):
    """
    Write lengths as bed file
    """
    chromosomes = ["Chr%02d" % (i + 1) for i in range(len(lengths))]
    j = 0
    with open(filename, "w") as bed_file:
        for chrid, l in enumerate(lengths):
            for i in range(l):
                bed_file.write(
                    "%s\t%d\t%d\t%d\n" % (chromosomes[chrid],
                                          i * resolution + 1,
                                          (i + 1) * resolution,
                                          j))
                j += 1
