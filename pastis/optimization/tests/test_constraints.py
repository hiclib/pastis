import sys
import pytest
import numpy as np
from sklearn.metrics import euclidean_distances
from scipy import sparse

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 6), reason="Requires python3.6 or higher")

if sys.version_info[0] >= 3:
    from pastis.optimization import constraints
    from pastis.optimization.counts import _format_counts


def test_bcc_constraint():
    lengths = np.array([20])
    ploidy = 2
    alpha, beta = -3., 1.

    n = lengths.sum()
    X_true = np.concatenate(
        [np.arange(n * ploidy).reshape(-1, 1), np.zeros((n * ploidy, 1)),
            np.zeros((n * ploidy, 1))], axis=1)
    dis = euclidean_distances(X_true)
    dis[dis == 0] = np.inf
    counts = beta * dis ** alpha
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts = np.triu(counts, 1)
    counts = sparse.coo_matrix(counts)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta)

    constraint = constraints.Constraints(
        counts, lengths=lengths, ploidy=ploidy, multiscale_factor=1,
        constraint_lambdas={'bcc': 1},
        constraint_params=None)
    constraint.check()
    obj = constraint.apply(X_true)['obj_bcc']
    assert obj < 1e-6


def test_hsc_constraint():
    lengths = np.array([30])
    ploidy = 2
    hsc_r = 1.
    alpha, beta = -3., 1.

    n = lengths.sum()
    X_true = np.zeros((n * ploidy, 3))
    X_true[n:, 0] += hsc_r
    dis = euclidean_distances(X_true)
    dis[dis == 0] = np.inf
    counts = beta * dis ** alpha
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts = np.triu(counts, 1)
    counts = sparse.coo_matrix(counts)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta)

    constraint = constraints.Constraints(
        counts, lengths=lengths, ploidy=ploidy, multiscale_factor=1,
        constraint_lambdas={'hsc': 1},
        constraint_params={'hsc': hsc_r})
    constraint.check()
    obj = constraint.apply(X_true)['obj_hsc']
    assert obj < 1e-6
