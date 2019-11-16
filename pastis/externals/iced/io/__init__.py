from .fastio_ import loadtxt, savetxt
try:
    from ._io_pandas import load_counts, load_lengths
    from ._io_pandas import write_counts
except ImportError:
    from ._io_else import load_lengths, load_counts
    from ._io_else import write_counts


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


def get_counts_shape(row_max, col_max, lengths=None):
    """
    Return shape of counts matrix.
    """

    import numpy as np

    if lengths is None:
        if round((row_max + 1) / (col_max + 1)) == 2:
            n = max(row_max + 1, (col_max + 1) * 2)
        elif round((col_max + 1) / (row_max + 1)) == 2:
            n = max((row_max + 1) * 2, col_max + 1)
        else:
            n = max(row_max, col_max) + 1
    else:
        lengths = np.array(lengths)
        n = lengths.sum()

    nrows = row_max + 1
    ncols = col_max + 1
    # Round up to nearest (n/2)
    nrows = int((n / 2) * np.ceil(float(nrows) / (n / 2)))
    ncols = int((n / 2) * np.ceil(float(ncols) / (n / 2)))
    return (nrows, ncols)
