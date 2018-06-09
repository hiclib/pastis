"""
====================
Simulating Hi-C data
====================

An example illustrating how to generate data, from a 3D structure.
"""


import numpy as np
from sklearn.metrics import euclidean_distances
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm


X = np.loadtxt(
    'data/yeast.pdb.txt')
X = X.reshape((len(X) / 3, 3))
n, _ = X.shape
dis = euclidean_distances(X)

alpha = -3.
beta = 0.6

poisson_intensity = beta * dis ** alpha
poisson_intensity[np.isinf(poisson_intensity)] = 0

random_state = np.random.RandomState(seed=0)
counts = random_state.poisson(poisson_intensity)

# The counts matrix should be symmetric
counts[np.tri(n, n, -1).astype(bool)] = 0
counts = (counts + counts.T).astype(float)

# Only plot the first five chromosomes
lengths = np.loadtxt("data/yeast_lengths.txt")

fig, ax = plt.subplots()
ax.set_xlim((0), lengths[:5].sum())
ax.set_ylim((0), lengths[:5].sum())

m = ax.matshow(counts,
               cmap="pink_r", norm=SymLogNorm(1), origin="bottom")
ax.set_title("Simulated Hi-C data: Budding Yeast")
ax.set_xlabel("Loci")
ax.set_ylabel("Loci")


for length in lengths.cumsum():
    ax.axhline(length, linestyle="--", linewidth=1)
    ax.axvline(length, linestyle="--", linewidth=1)


plt.show()
