from __future__ import print_function

import numpy as np
from scipy import sparse


def get_genomic_distances(lengths, counts=None):
    """
    Get genomic distances
    """
    if sparse.issparse(counts):
        if not sparse.isspmatrix_coo(counts):
            counts = counts.tocoo()
        return _get_genomic_distances_sparse(lengths, counts)
    else:
        from iced import utils
        return utils.get_genomic_distances(lengths)


def _get_genomic_distances_sparse(lengths, counts):
    """
    """
    chr_id = np.array([i for i, l in enumerate(lengths) for _ in range(l)])
    gdis = counts.col - counts.row
    gdis[chr_id[counts.col] != chr_id[counts.row]] = -1
    return gdis
