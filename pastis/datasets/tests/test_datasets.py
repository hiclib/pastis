import numpy as np
from sklearn.metrics import euclidean_distances
from pastis.datasets import generate_dataset_from_distances
from numpy.testing import assert_array_equal


def test_generate_datasets_from_distances():
    random_state = np.random.RandomState(seed=42)
    lengths = np.array([10, 10])
    n = lengths.sum()
    X = random_state.randn(n, 3)
    dis = euclidean_distances(X, X)

    contact_counts = generate_dataset_from_distances(dis, random_state=42)
    contact_counts_inter = generate_dataset_from_distances(
        dis, alpha_inter=-3, random_state=42, lengths=lengths)

    assert_array_equal(contact_counts, contact_counts_inter)
