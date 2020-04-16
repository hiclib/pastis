import numpy as np
from scipy import sparse


def _case_resampling(counts, random_state=None):
    """

    Parameters

    counts : coo sparse matrix
    (For A + UA this must be triu or tril to make sure output is symmetric)
    """

    counts = counts.copy()
    if sparse.isspmatrix_coo(counts):
        counts = counts.toarray()
    counts[np.isnan(counts)] = 0
    counts = sparse.coo_matrix(counts).astype(int)

    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    # Create a matrix of indices where each entry corresponds to an
    # interacting pair of loci, and where interacting pairs appear the number
    # of time they interact
    ind = np.repeat(np.arange(len(counts.data)), counts.data, axis=0)

    # Shuffle the indices and select f*nreads number of interaction
    boot_ind = random_state.choice(
        ind, size=int(counts.sum()), replace=True)

    # Recreate the interaction counts matrix.
    c = sparse.coo_matrix(
        (np.ones(len(boot_ind)), (counts.row[boot_ind],
                                  counts.col[boot_ind])),
        shape=counts.shape, dtype=float)
    return c


def bootstrap_counts(counts, random_state=None):
    """
    """

    print('BOOTSTRAPPING COUNTS', flush=True)

    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    if not isinstance(counts, list):
        return _case_resampling(counts, random_state=random_state)
    else:
        return [_case_resampling(c, random_state=random_state) for c in counts]