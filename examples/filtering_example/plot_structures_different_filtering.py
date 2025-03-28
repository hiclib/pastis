"""
Plot 3-d structures inferred with different levels of filtering with matplotlib
===============================================================================

This example plots two 3-d structures (with matplotlib) inferred with different
levels of filtering to showcase how increasing filtering when running PASTIS
can result in a better-looking inferred structure.
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import interpolate


##############################################################################
# Define the interpolation function.
# ----------------------------------
def interpolate_chromosomes(struct, lengths, eps=1e-1, smooth=0):

    """ Returns a smoothed interpolation of the chromosomes.

        Parameters
        ----------
        struct : ndarray (n, 3)
        The 3D structure

        lengths : ndarray (L, )
        The lengths of the chromosomes. Note that the sum of the lengths
        should correspond to the length of the ndarray of the 3D structure.

        Returns
        -------
        struct_smoothed : the smoothed 3D structure, with a set of coordinates
        for each chromosome (ie, the first chromosome's points will be
        struct_smoothed[0]). struct_smoothed[0][i] will be a list with an x, y,
        and z coordinate.
    """

    lengths = np.array(lengths, copy=False, ndmin=1, dtype=int).ravel()
    mask = np.invert(np.isnan(struct[:, 0]))

    # Loops over each chromosome
    struct_smoothed = [[], []]
    begin, end = 0, 0
    for i, length in enumerate(lengths):
        end += length

        # Get the current chromosome's coordinates
        x = struct[begin:end, 0]
        y = struct[begin:end, 1]
        z = struct[begin:end, 2]
        indx = mask[begin:end]

        if not x.shape[0]:
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

        m = np.arange(x.shape[0])[indx]

        # Get interpolation functions for the x, y, and z coordinates
        f_x = interpolate.Rbf(m, x[indx], smooth=smooth)
        f_y = interpolate.Rbf(m, y[indx], smooth=smooth)
        f_z = interpolate.Rbf(m, z[indx], smooth=smooth)

        # Use interpolation functions to create curve segments between
        # adjacent (adjacent in the parameter 'struct', for the current chrom)
        # coordinate beads
        for j in range(length - 1):
            if (j < length - 2):
                m = np.arange(j, j + 1 + 0.1, 0.1)
            else:
                m = np.arange(j, j + 1, 0.1)
            struct_smoothed[i].append([
                f_x(np.arange(m.min(), m.max(), 0.1)),
                f_y(np.arange(m.min(), m.max(), 0.1)),
                f_z(np.arange(m.min(), m.max(), 0.1))])

        begin = end

    return struct_smoothed


##############################################################################
# Define the function to scatter the beads.
# -----------------------------------------
def scatter_beads(x, y, z, bead_size, curr_cmap, indices, ax):
    last_idx = x.shape[0] - 1
    ax.scatter(x[0], y[0], z[0], s=bead_size, color=curr_cmap(indices[0]),
               label='start')
    ax.scatter(x[1:last_idx], y[1:last_idx], z[1:last_idx], s=bead_size,
               cmap=curr_cmap, c=indices[1:last_idx])
    ax.scatter(x[last_idx], y[last_idx], z[last_idx], s=bead_size,
               color=curr_cmap(indices[last_idx]), label='end')


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
    def reduced_cmap(step):
        return np.array(cmap(step)[0:3])
    cmap_stepped = []
    for curr_step in step_list:
        cmap_stepped.append(reduced_cmap(curr_step))
    old_LUT = np.array(cmap_stepped)
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
# Define the function to plot the coordinates
# -------------------------------------------
def plot_structure(struct_file, lengths=None, title=None, output_dir=None):
    # Setup output files
    output_file = os.path.basename(struct_file).replace(".txt", "").replace(
        ".coords", "")
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, output_file)

    # Load in the structure from its file
    struct = np.loadtxt(struct_file, delimiter=" ")

    # Get the number of beads in each molecule of the structure
    if lengths is None:  # Assume structure contains 1 haploid chromosome
        lengths = np.array([struct.shape[0]])
    else:
        lengths = np.array(lengths, copy=False, ndmin=1, dtype=int).ravel()

    # Using a spectral color cmap.
    the_cmap = cmap_map(
        lambda c: c * 0.85, cmap=matplotlib.colormaps['Spectral'])

    # Interpolate the coordinates of the structure
    struct_smoothed = interpolate_chromosomes(struct, lengths=lengths)

    begin, end = 0, 0
    for i, length in enumerate(lengths):
        end += length

        # Get the current chromosome's coordinates
        x = struct[begin:end, 0]
        y = struct[begin:end, 1]
        z = struct[begin:end, 2]

        # Get the plot ready
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')

        # Get our colors set up. color_beads determines which color in our color
        # map each bead will have.
        color_indices = np.linspace(0, 1, x.shape[0])

        # Scatter the beads for the chromosome.
        scatter_beads(
            x=x, y=y, z=z, bead_size=50, curr_cmap=the_cmap,
            indices=color_indices, ax=ax)

        # Plot the points we interpolated that go in between the beads of the
        # chromosome.
        chrom_smoothed = struct_smoothed[i]
        for j in range(length - 1):
            ax.plot(
                chrom_smoothed[j][0], chrom_smoothed[j][1],
                chrom_smoothed[j][2], color=the_cmap(color_indices[j]))

        # Set the labels, title, legend, and view angle and position.
        ax.set_xlabel('x', fontsize=20)
        ax.set_ylabel('y', fontsize=20)
        ax.set_zlabel('z', fontsize=20)
        plt.legend(loc='best')
        plt.title(title, fontsize=25)
        ax.view_init(15, 300)

        # Save figure
        fig.savefig(
            f"{output_file}.molecule{i:03d}.png", bbox_inches='tight', dpi=500)
        plt.close()

        begin = end


##############################################################################
# Plot the structure with less filtering. As we can see, this structure looks
# quite odd, with large loops extending out to single beads that are far away
# from the rest of the structure.
plot_structure(
    'data/struct_inferred_less.000.coords',
    title='Inferred 3-d structure (Less Filtering)')


##############################################################################
# Plot the structure with more filtering. As we can see, this structure looks
# significantly better, as bins with low coverage in the contact counts matrix
# were filtered more heavily before PASTIS inferred the structure.
plot_structure(
    'data/struct_inferred_more.000.coords',
    title='Inferred 3-d structure (More Filtering)')
