from .write_struct.pdb import writePDB
import numpy as np


def write(X, filename, lengths=None, resolution=50000, copy=True):
    """
    Write the similarity matrix to a file

    Parameters
    ----------
    X: ndarray,
    """
    if copy:
        X = X.copy()
    X = X.astype(float)

    output_file = open(filename, 'w')

    tr = X.sum(axis=0) == 0
    if tr.sum() != 0:
        X[tr, :] = np.nan
        X[:, tr] = np.nan

    for i in range(len(X)):
        chr1, loc1 = _get_index(i, lengths)
        if loc1 > lengths[chr1 - 1]:
            raise ValueError("Problem !")
        loc1 *= resolution
        loc1 += 1
        for j in range(len(X)):
            if i > j:
                continue
            chr2, loc2 = _get_index(j, lengths)
            if loc2 > lengths[chr2 - 1]:
                raise ValueError("Problem !")
            loc2 *= resolution
            loc2 += 1

            if chr2 == chr1:
                dis = loc2 - loc1
            else:
                dis = 0

            if not X[i, j] or np.isnan(X[i, j]):
                continue

            output_file.write('%d\t%d\t%d\t%d\t%d\t%f\t0\t0\n' %
                              (chr1, loc1, chr2, loc2, dis, X[i, j]))

    output_file.close()


def _get_index(i, lengths):
    """
    From an index, return the chromosome number and the loci number

    Parameters
    ----------
    i : integer
        index in the full matrix

    lengths : ndarray (n, )

    Returns
    -------
    tuple : (c, l)
        chromosome number, locus
    """
    length_cum = lengths.cumsum()
    c = (length_cum > i).argmax() + 1

    if c > 1:
        length = i - length_cum[c - 2]
    else:
        length = i
    if len(lengths) > 1 and length > lengths[c - 1]:
        raise ValueError("there's a problem")

    if c > len(lengths):
        raise ValueError("There's a problem")
    return c, length
