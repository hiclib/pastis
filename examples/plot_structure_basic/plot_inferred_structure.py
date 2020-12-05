"""
Plot an inferred 3-d structure with matplotlib
==============================================

This example showcases how to plot an inferred 3-d structure output by
PASTIS using matplotlib.
"""

import numpy as np
from mpl_toolkits.mplot3d import axes3d  # noqa: F401 unused import
import matplotlib
import matplotlib.pyplot as plt
from scipy import interpolate


##############################################################################
# Define the interpolation function.
# ----------------------------------
def interpolate_chromosomes(X, lengths, eps=1e-1, smooth=0):

    """ Returns a smoothed interpolation of the chromosomes.

        Parameters
        ----------
        X : ndarray (n, 3)
        The 3D structure

        lengths : ndarray (L, )
        The lengths of the chromosomes. Note that the sum of the lengths
        should correspond to the length of the ndarray of the 3D structure.

        Returns
        -------
        smoothed_X : the smoothed 3D structure, with a set of coordinates for
        each chromosome (ie, the first chromosome's points will be
        smoothed_X[0]). smoothed_X[0][i] will be a list with an x, y, and
        z coordinate.
    """

    smoothed_X = [[], []]

    mask = np.invert(np.isnan(X[:, 0]))

    begin, end = 0, 0

    # Loops over each chromosome
    enumerated_lengths = enumerate(lengths)
    for i, length in enumerated_lengths:
        end += length

        # Get the current chromosome's coordinates
        x = X[begin:end, 0]
        y = X[begin:end, 1]
        z = X[begin:end, 2]
        indx = mask[begin:end]

        if not len(x):
            break

        # Encode missing values with nan
        if not indx[0]:
            x[0] = x[indx][0]
            x[indx][0] = np.nan
            y[0] = y[indx][0]
            y[indx][0] = np.nan

            z[0] = z[indx][0]
            z[indx][0] = np.nan

        # Encode missing values with nan
        if not indx[-1]:
            x[-1] = x[indx][-1]
            x[indx][-1] = np.nan
            y[-1] = y[indx][-1]
            z[indx][-1] = np.nan
            z[-1] = z[indx][-1]
            z[indx][-1] = np.nan

        indx = np.invert(np.isnan(x))

        m = np.arange(len(x))[indx]

        # Get interpolation functions for the x, y, and z coordinates
        f_x = interpolate.Rbf(m, x[indx], smooth=smooth)
        f_y = interpolate.Rbf(m, y[indx], smooth=smooth)
        f_z = interpolate.Rbf(m, z[indx], smooth=smooth)

        # Use interpolation functions to create curve segments between
        # adjacent (adjacent in the parameter X, for the current chromosome)
        # coordinate beads
        for j in range(length - 1):
            if (j < length - 2):
                m = np.arange(j, j + 1 + 0.1, 0.1)
            else:
                m = np.arange(j, j + 1, 0.1)
            smoothed_X[i].append([f_x(np.arange(m.min(), m.max(), 0.1)),
                                  f_y(np.arange(m.min(), m.max(), 0.1)),
                                  f_z(np.arange(m.min(), m.max(), 0.1))])

        begin = end

    return smoothed_X


##############################################################################
# Define the function to scatter the beads.
# -----------------------------------------
def scatter_beads(x, y, z, bead_size, name, curr_cmap, indices, ax):
    last_idx = len(x) - 1
    ax.scatter(x[0], y[0], z[0], s=bead_size, color=curr_cmap(indices[0]),
               label=name + ' start')
    ax.scatter(x[1:last_idx], y[1:last_idx], z[1:last_idx], s=bead_size,
               cmap=curr_cmap, c=indices[1:last_idx])
    ax.scatter(x[last_idx], y[last_idx], z[last_idx], s=bead_size,
               color=curr_cmap(indices[last_idx]), label=name + ' end')


##############################################################################
# Define the function to lighten or darken our color map.
# -------------------------------------------------------
def cmap_map(function, cmap):
    # Function taken from:
    # https://scipy-cookbook.readthedocs.io/items/Matplotlib_ColormapTransformations.html

    """ Applies function (which should operate on vectors of shape 3:
        [r, g, b]), on colormap cmap. This routine will break any discontinuous
        points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # First get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step: np.array(cmap(step)[0: 3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red', 'green', 'blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j, i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap', cdict, 1024)


##############################################################################
# Load the coordinates for the two chromosomes.
# ---------------------------------------------
#
# Note: both of the chromosomes have length 60 (the array is length 120).
coordinates = np.loadtxt('data/struct_inferred.000.coords', delimiter=" ")
theLengths = np.array([60, 60])

# Get the first chromosome's coordinates
start, end = 0, theLengths[0]
x1 = coordinates[:, 0][start: end]
y1 = coordinates[:, 1][start: end]
z1 = coordinates[:, 2][start: end]

# Get the second chromosome's coordinates
start, end = end, end + theLengths[1]
x2 = coordinates[:, 0][start: end]
y2 = coordinates[:, 1][start: end]
z2 = coordinates[:, 2][start: end]


##############################################################################
# Interpolate the coordinates of the chromosomes.
# -----------------------------------------------
chrom1, chrom2 = interpolate_chromosomes(coordinates, theLengths)


##############################################################################
# Plot the chromosomes.
# ---------------------
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')

# Get our colors set up. color_beads determines which color in our color map
# each bead will have.
color_beads = np.linspace(0, 1, len(x1))

# We will use a spectral color cmap. Since we are plotting two chromosomes,
# let's make chromosome one a bit darker (hence x * 0.85) and chromosome two
# a bit lighter (hence x * 1.5). Feel free to twiddle with these configruations
# or change color maps entirely.
cmap1 = cmap_map(lambda x: x*0.85, matplotlib.cm.get_cmap('Spectral'))
cmap2 = cmap_map(lambda x: x*1.5, matplotlib.cm.get_cmap('Spectral'))

# Scatter the beads for each chromosome.
scatter_beads(x1, y1, z1, 50, 'chromosome 1', cmap1, color_beads, ax)
scatter_beads(x2, y2, z2, 50, 'chromosome 2', cmap2, color_beads, ax)

# Plot the points we interpolated that go in between each chromosome as
# a curve.
for i in range(59):
    ax.plot(chrom1[i][0], chrom1[i][1], chrom1[i][2],
            color=cmap1(color_beads[i]))
    ax.plot(chrom2[i][0], chrom2[i][1], chrom2[i][2],
            color=cmap2(color_beads[i]))

# Set the labels, title, legend, and view angle and position. Show the plot.
ax.set_xlabel('x', fontsize=20)
ax.set_ylabel('y', fontsize=20)
ax.set_zlabel('z', fontsize=20)
plt.legend(loc='best')
plt.title('Inferred 3d Structures', fontsize=25)
ax.view_init(15, 300)
plt.show()
