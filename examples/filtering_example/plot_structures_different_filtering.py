"""
Plot 3D structures inferred with different levels of filtering with matplotlib
===============================================================================

This example plots two 3D structures (with matplotlib) inferred with different
levels of filtering to showcase how increasing filtering when running PASTIS
can result in a better-looking inferred structure.
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy import interpolate
from iced.io import load_lengths


def modify_cmap(function, cmap):
    """Lighten or darken a given colormap.

    Applies function (which should operate on vectors of shape 3:
    [r, g, b]), on colormap cmap. This routine will break any discontinuous
    points in a colormap.

    Adapted from:
    https://scipy-cookbook.readthedocs.io/items/Matplotlib_ColormapTransformations.html
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))

    # Compute the LUT, and apply the function to the LUT
    def reduce_cmap(step):
        return np.array(cmap(step)[0:3])
    old_LUT = np.array([reduce_cmap(x) for x in step_list])
    new_LUT = np.array([function(x) for x in old_LUT])

    # Try to make a minimal segment definition of the new LUT
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


def interpolate_chromosome(chrom, eps=0.1):
    """Interpolate a smooth line between the beads of a given chromosome.

    Should only be run on a single chromosome molecule.

    Parameters
    ----------
    chrom : array of float
    The 3D structure of a given chromosome (nbeads, 3).

    eps : float, optional
    Frequency of points to be interpolated. A value of '1' simply replaces any
    NaN beads with interpolated values, but does not form a line connecting
    between adjacent beads.

    Returns
    -------
    chrom_interp : array of float
    The smoothed 3D structure for the given chromosome.
    """

    # Select non-NaN beads
    mask = np.invert(np.isnan(chrom[:, 0]))
    if mask.sum() < 2:
        return None  # Skip chromosomes with < 2 non-NaN beads

    x = chrom[mask, 0]
    y = chrom[mask, 1]
    z = chrom[mask, 2]

    # Get interpolation functions for the x, y, and z coordinates
    bead_idx = np.arange(chrom.shape[0])[mask]
    f_x = interpolate.Rbf(bead_idx, x, smooth=0)
    f_y = interpolate.Rbf(bead_idx, y, smooth=0)
    f_z = interpolate.Rbf(bead_idx, z, smooth=0)

    # Use interpolation functions to create line segments between adjacent
    # coordinate beads
    line_idx = np.arange(bead_idx.min(), bead_idx.max() + eps, eps)
    chrom_interp = np.array([f_x(line_idx), f_y(line_idx), f_z(line_idx)]).T

    return chrom_interp


def plot_chromosome(chrom, ax=None, cmap=None, name=None, bead_size=50):
    """Plot a single chromosome molecule on the supplied axis.

    Beads are represented as translucent circles, and a line is interpolated
    between adjacent beads. Should only be run on a single chromosome molecule.

    Parameters
    ----------
    chrom : array of float
    The coordinates of the 3D structure of a given chromosome,
    shape=(nbeads_chrom, 3).

    ax : matplotlib.axes.Axes object, optional
    The matplotlib axis on which to plot the chromosome. If absent, a new
    figure will be created.

    cmap : matplotlib.colors.Colormap object, optional
    The matplotlib colormap with which to color the beads and lines.

    name : str, optional
    The name for the given chromosome, to be displayed in the plot legend.

    bead_size : int, optional
    Size for each bead in the plotted 3D structure.

    Returns
    -------
    ax : matplotlib.axes.Axes object
    The matplotlib axis on which the chromosome has been plotted.
    """

    if cmap is None:
        cmap = matplotlib.colormaps['Spectral']
    if ax is None:
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('x', fontsize=20)
        ax.set_ylabel('y', fontsize=20)
        ax.set_zlabel('z', fontsize=20)
        ax.view_init(15, 300)
    if name is None:
        name = ""
    elif name != "":
        name = f"{name} "

    # Select non-NaN beads
    mask = np.invert(np.isnan(chrom[:, 0]))
    if mask.sum() == 0:
        return ax  # Chromosome beads are all NaN, nothing to plot

    # Determine which color in our colormap each bead will have
    bead_color_idx = np.linspace(0, 1, chrom.shape[0])[mask]

    # Plot the beads for the current chromosome
    x = chrom[mask, 0]
    y = chrom[mask, 1]
    z = chrom[mask, 2]
    ax.scatter(x[0], y[0], z[0], s=bead_size, color=cmap(bead_color_idx[0]),
               label=f"{name}start")
    if x.shape[0] > 2:
        ax.scatter(x[1:-1], y[1:-1], z[1:-1], s=bead_size,
                   cmap=cmap, c=bead_color_idx[1:-1])
    ax.scatter(x[-1], y[-1], z[-1], s=bead_size,
               color=cmap(bead_color_idx[-1]), label=f"{name}end")

    # Interpolate the structure's beads to form a continuous line between beads
    chrom_interp = interpolate_chromosome(chrom)

    if chrom_interp is not None:
        # Determine which color in our colormap each line segment will have
        line_color_idx = np.linspace(
            bead_color_idx.min(), bead_color_idx.max(), chrom_interp.shape[0])

        # Plot this chromosome's interpolated coordinates as a continuous line
        # that connects between adjacent beads
        points = chrom_interp.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = Line3DCollection(segments, cmap=cmap)
        lc.set_array(line_color_idx)  # Set the values used for colormapping
        lc.set_linewidth(2)
        ax.add_collection(lc)

    return ax


def plot_structure(struct_file, lengths=None, ploidy=1, title=None,
                   output_dir=None):
    """Plot each chromosome molecule of structure individually in 3D.

    Plot every chromosome molecule of the 3D genome structure, each on a
    separate plot. Beads are represented as translucent circles, and a line is
    interpolated between adjacent beads on each chromosome.

    Parameters
    ----------
    struct_file : str
    File containing whitespace-delimited coordinates of the 3D structure,
    shape=(nbeads, 3).

    lengths : str or array_like of int, optional
    Number of beads in each chromosome of the structure. For haploid organisms,
    the sum of the lengths should equal the total number of beads in the 3D
    structure. For diploid organisms, the sum of the lengths is half of the
    total number of beads in the 3D structure. If inputted as string, is assumed
    to be the path to a bed file containing chromosome lengths.

    ploidy : int, optional
    Ploidy of the organism: 1 indicates haploid, 2 indicates diploid.

    title : str, optional
    The matplotlib axis on which to plot the chromosome. If absent, a new
    figure will be created.

    output_dir : str, optional
    Directory in which to save the figures. If absent, figures will be saved
    in current working directory.
    """

    # Setup output files
    output_prefix = os.path.basename(struct_file).replace(".txt", "").replace(
        ".coords", "")
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_prefix = os.path.join(output_dir, output_prefix)

    # Load in the structure from its file
    struct = np.loadtxt(struct_file)

    # Get the number of beads in each chromosome of the structure
    # The sum of the lengths should equal the total number of beads in the 3D
    # structure.
    if lengths is None:  # Assume structure contains 1 haploid chromosome
        lengths = np.array([struct.shape[0]])
    elif isinstance(str, lengths) and os.path.isfile(lengths):
        lengths = load_lengths(lengths)
    else:
        lengths = np.array(lengths, copy=False, ndmin=1, dtype=int).ravel()
    lengths = np.tile(lengths, ploidy)  # In case data is diploid

    # Using a modified spectral colormap
    cmap = modify_cmap(
        lambda x: x * 0.85, cmap=matplotlib.colormaps['Spectral'])

    begin, end = 0, 0
    for i, length in enumerate(lengths):
        end += length

        if np.isnan(struct[begin:end, 0].sum()) == length:
            continue  # Chromosome beads are all NaN, skipping

        # Get the plot ready for this chromosome
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')

        # Plot current chromosome
        ax = plot_chromosome(struct[begin:end], ax=ax, cmap=cmap)

        # Set the labels, title, legend, and view angle and position.
        ax.set_xlabel('x', fontsize=20)
        ax.set_ylabel('y', fontsize=20)
        ax.set_zlabel('z', fontsize=20)
        plt.legend(loc='best')
        plt.title(title, fontsize=25)
        ax.view_init(15, 300)

        # Save figure for this chromosome
        if lengths.size == 1:
            output_file = f"{output_prefix}.png"
        else:
            output_file = f"{output_prefix}.molecule{i + 1}of{lengths.size}.png"
        fig.savefig(output_file, bbox_inches='tight', dpi=500)
        plt.close()

        begin = end


##############################################################################

# PASTIS can optionally filter the inputted counts matrix prior to inference.
# This filtering removes loci with inadequate coverage (eg unmappable regions).

# This first inferred structure was generated with FEWER loci filtered out.
# Observe that this structure looks quite odd, with large loops extending out
# to single beads that are far away from the rest of the structure.
plot_structure(
    'data/struct_inferred_less.000.coords',
    title='Inferred 3D structure (Less filtering)')

# This second inferred structure was generated with MORE loci filtered out.
# Observe that this structure look significantly better than the first.
plot_structure(
    'data/struct_inferred_more.000.coords',
    title='Inferred 3D structure (More filtering)')
