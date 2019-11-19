import numpy as np


def decrease_lengths_res(lengths, multiscale_factor):
    """Reduce resolution of chromosome lengths.
    """

    return np.ceil(
        np.array(lengths).astype(float) / multiscale_factor).astype(int)


def increase_struct_res(struct, multiscale_factor, lengths, mask=None):
    """Linearly interpolate structure to increase resolution.
    """

    from scipy.interpolate import interp1d
    from ..externals.iced.io import load_lengths

    if int(multiscale_factor) != multiscale_factor:
        raise ValueError('The multiscale_factor must be an integer')
    multiscale_factor = int(multiscale_factor)
    if multiscale_factor == 1:
        return struct
    if isinstance(struct, str):
        struct = np.loadtxt(struct)
    struct = struct.reshape(-1, 3)
    if isinstance(lengths, str):
        lengths = load_lengths(lengths)
    lengths = np.array(lengths).astype(int)
    lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)
    ploidy = struct.shape[0] / lengths_lowres.sum()
    if ploidy != 1 and ploidy != 2:
        raise ValueError("Not consistent with haploid or diploid... struct is"
                         " %d beads (and 3 cols), sum of lengths is %d" %
                         (struct.reshape(-1, 3).shape[0], lengths_lowres.sum()))
    ploidy = int(ploidy)

    indices = _get_struct_indices(ploidy, multiscale_factor,
                                  lengths).reshape(multiscale_factor, -1)
    if mask is not None:
        indices[~mask.reshape(multiscale_factor, -1)] = np.nan

    struct_highres = np.full((lengths.sum() * ploidy, 3), np.nan)
    begin_lowres, end_lowres = 0, 0
    for i in range(lengths.shape[0] * ploidy):
        end_lowres += np.tile(lengths_lowres, ploidy)[i]

        # Beads of struct that are NaN
        struct_nan = np.isnan(struct[begin_lowres:end_lowres, 0])

        # Get indices for this chrom at low & high res
        chrom_indices = indices[:, begin_lowres:end_lowres]
        chrom_indices[:, struct_nan] = np.nan
        chrom_indices_lowres = np.nanmean(chrom_indices, axis=0)
        chrom_indices_highres = chrom_indices.T.flatten()

        # Note which beads are unknown
        highres_mask = ~np.isnan(chrom_indices_highres)
        highres_mask[highres_mask] = (chrom_indices_highres[highres_mask] >=
                                      np.nanmin(chrom_indices_lowres)) & (chrom_indices_highres[highres_mask] <= np.nanmax(chrom_indices_lowres))
        unknown_beads = np.where(~highres_mask)[
            0] + np.tile(lengths, ploidy)[:i].sum()
        unknown_beads = unknown_beads[unknown_beads < np.tile(lengths, ploidy)[
            :i + 1].sum()]
        unknown_beads_at_begin = [unknown_beads[k] for k in range(len(unknown_beads)) if unknown_beads[
            k] == unknown_beads.min() or all([unknown_beads[k] - j == unknown_beads[k - j] for j in range(k + 1)])]
        if len(unknown_beads) - len(unknown_beads_at_begin) > 0:
            unknown_beads_at_end = [unknown_beads[k] for k in range(len(unknown_beads)) if unknown_beads[k] == unknown_beads.max(
            ) or all([unknown_beads[k] + j == unknown_beads[k + j] for j in range(len(unknown_beads) - k)])]
            chrom_indices_highres = np.arange(
                max(unknown_beads_at_begin) + 1, min(unknown_beads_at_end))
        else:
            unknown_beads_at_end = []
            chrom_indices_highres = np.arange(
                max(unknown_beads_at_begin) + 1, int(np.nanmax(chrom_indices_highres)) + 1)

        struct_highres[chrom_indices_highres, 0] = interp1d(
            chrom_indices_lowres[~struct_nan],
            struct[begin_lowres:end_lowres, 0][~struct_nan],
            kind="linear")(chrom_indices_highres)
        struct_highres[chrom_indices_highres, 1] = interp1d(
            chrom_indices_lowres[~struct_nan],
            struct[begin_lowres:end_lowres, 1][~struct_nan],
            kind="linear")(chrom_indices_highres)
        struct_highres[chrom_indices_highres, 2] = interp1d(
            chrom_indices_lowres[~struct_nan],
            struct[begin_lowres:end_lowres, 2][~struct_nan],
            kind="linear")(chrom_indices_highres)

        # Fill in beads at start
        diff_beads_at_chr_start = struct_highres[chrom_indices_highres[
            1], :] - struct_highres[chrom_indices_highres[0], :]
        how_far = 1
        for j in reversed(unknown_beads_at_begin):
            struct_highres[j, :] = struct_highres[chrom_indices_highres[
                0], :] - diff_beads_at_chr_start * how_far
            how_far += 1
        # Fill in beads at end
        diff_beads_at_chr_end = struct_highres[
            chrom_indices_highres[-2], :] - struct_highres[chrom_indices_highres[-1], :]
        how_far = 1
        for j in unknown_beads_at_end:
            struct_highres[j, :] = struct_highres[
                chrom_indices_highres[-1], :] - diff_beads_at_chr_end * how_far
            how_far += 1

        begin_lowres = end_lowres

    return struct_highres


def _convert_indices_to_full_res(rows, cols, rows_max, cols_max,
                                 multiscale_factor, lengths, n, counts_shape,
                                 ploidy):
    """Return full-res counts indices grouped by the corresponding low-res bin.
    """

    if multiscale_factor == 1:
        return rows, cols
    # Move all indices to upper triangular (necessary for proper full-res
    # indexing!)
    rows_new = np.minimum(rows, cols)
    cols_new = np.maximum(rows, cols)
    rows = rows_new
    cols = cols_new
    # Convert low-res indices to full-res
    nnz = len(rows)
    x, y = np.indices((multiscale_factor, multiscale_factor))
    rows = np.repeat(x.flatten(), nnz)[:nnz * multiscale_factor ** 2] + \
        np.tile(rows * multiscale_factor, multiscale_factor ** 2)
    cols = np.repeat(y.flatten(), nnz)[:nnz * multiscale_factor ** 2] + \
        np.tile(cols * multiscale_factor, multiscale_factor ** 2)
    rows = rows.reshape(multiscale_factor ** 2, -1)
    cols = cols.reshape(multiscale_factor ** 2, -1)
    # Figure out which rows / cols are out of bounds
    bins_for_rows = np.tile(lengths, int(counts_shape[0] / n)).cumsum()
    bins_for_cols = np.tile(lengths, int(counts_shape[1] / n)).cumsum()
    for i in range(lengths.shape[0] * ploidy):
        rows_binned = np.digitize(rows, bins_for_rows)
        cols_binned = np.digitize(cols, bins_for_cols)
        incorrect_rows = np.invert(
            np.equal(rows_binned, np.floor(rows_binned.mean(axis=0))))
        incorrect_cols = np.invert(
            np.equal(cols_binned, np.floor(cols_binned.mean(axis=0))))
        for val in np.flip(np.unique(rows[:, np.floor(rows_binned.mean(axis=0)) == i]
                                         [incorrect_rows[:,
                                                         np.floor(rows_binned.mean(axis=0)) == i]])):
            rows[rows > val] -= 1
        for val in np.flip(np.unique(cols[:, np.floor(cols_binned.mean(axis=0)) == i]
                                         [incorrect_cols[:,
                                                         np.floor(cols_binned.mean(axis=0)) == i]])):
            cols[cols > val] -= 1
        # Because if the last low-res bin in this homolog of this chromosome is
        # all zero, that could mess up indices for subsequent
        # homologs/chromosomes
        rows_binned = np.digitize(rows, bins_for_rows)
        row_mask = np.floor(rows_binned.mean(axis=0)) == i
        current_rows = rows[:, row_mask][
            np.invert(incorrect_rows)[:, row_mask]]
        if current_rows.shape[0] > 0:
            max_row = current_rows.max()
            if max_row < bins_for_rows[i] - 1:
                rows[rows > max_row] -= multiscale_factor - \
                    (bins_for_rows[i] - max_row - 1)
        cols_binned = np.digitize(cols, bins_for_cols)
        col_mask = np.floor(cols_binned.mean(axis=0)) == i
        current_cols = cols[:, col_mask][
            np.invert(incorrect_cols)[:, col_mask]]
        if current_cols.shape[0] > 0:
            max_col = current_cols.max()
            if max_col < bins_for_cols[i] - 1:
                cols[cols > max_col] -= multiscale_factor - \
                    (bins_for_cols[i] - max_col - 1)

    incorrect_indices = incorrect_rows + incorrect_cols + \
        (rows >= rows_max) + (cols >= cols_max)
    rows[incorrect_indices] = 0
    cols[incorrect_indices] = 0
    rows = rows.flatten()
    cols = cols.flatten()
    return rows, cols


def decrease_counts_res(counts, multiscale_factor, lengths, ploidy):
    """Decrease resolution of counts matrix by summing adjacent bins.
    """

    from .counts import _row_and_col
    from scipy import sparse

    input_is_sparse = False
    if sparse.issparse(counts):
        counts = counts.copy().toarray()
        input_is_sparse = True

    lengths_lowres = np.ceil(np.array(lengths).astype(
        float) / multiscale_factor).astype(int)
    counts_lowres = np.ones(
        np.array(counts.shape / lengths.sum() * lengths_lowres.sum(), dtype=int))
    np.fill_diagonal(counts_lowres, 0)
    counts_lowres = sparse.coo_matrix(counts_lowres)
    rows_raw, cols_raw = _row_and_col(counts_lowres)
    rows, cols = _convert_indices_to_full_res(
        rows_raw, cols_raw, rows_max=counts.shape[0], cols_max=counts.shape[1],
        multiscale_factor=multiscale_factor, lengths=lengths,
        n=lengths_lowres.sum(), counts_shape=counts_lowres.shape, ploidy=ploidy)
    data = counts[rows, cols].reshape(multiscale_factor ** 2, -1).sum(axis=0)
    counts_lowres = sparse.coo_matrix((data[data != 0], (rows_raw[data != 0],
                                                         cols_raw[data != 0])),
                                      shape=counts_lowres.shape)

    if not input_is_sparse:
        counts_lowres = counts_lowres.toarray()

    return counts_lowres, lengths_lowres


def _get_struct_indices(ploidy, multiscale_factor, lengths):
    """Return full-res struct indices grouped by the corresponding low-res bead.
    """

    lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)

    indices = np.arange(lengths_lowres.sum() * ploidy).astype(float)
    indices = np.repeat(np.indices([multiscale_factor]), indices.shape[
                        0]) + np.tile(indices * multiscale_factor, multiscale_factor)
    indices = indices.reshape(multiscale_factor, -1)

    # Figure out which rows / cols are out of bounds
    bins = np.tile(lengths, ploidy).cumsum()
    for i in range(lengths.shape[0] * ploidy):
        indices_binned = np.digitize(indices, bins)
        incorrect_indices = np.invert(
            np.equal(indices_binned, indices_binned.min(axis=0)))
        for val in np.flip(np.unique(indices[:, indices_binned.min(axis=0) == i]
                                     [incorrect_indices[:, indices_binned.min(axis=0) == i]])):
            indices[indices > val] -= 1
    incorrect_indices += indices >= lengths.sum() * ploidy

    incorrect_indices = incorrect_indices.flatten()
    indices = indices.flatten()

    # If a bin spills over chromosome / homolog boundaries, set it to NaN - it
    # will get ignored later
    indices[incorrect_indices] = np.nan

    return indices


def _group_highres_struct(struct, multiscale_factor, lengths, indices=None, mask=None):
    """Group beads of full-res struct by the low-res bead they correspond to.

    Axes of final array:
        0: all highres beads corresponding to each lowres bead, size = multiscale factor
        1: beads, size = struct[0]
        2: coordinates, size = struct[1] = 3
    """

    lengths = np.array(lengths).astype(int)

    ploidy = struct.reshape(-1, 3).shape[0] / lengths.sum()
    if ploidy != 1 and ploidy != 2:
        raise ValueError("Not consistent with haploid or diploid... struct is"
                         " %d beads (and 3 cols), sum of lengths is %d" % (
                             struct.reshape(-1, 3).shape[0], lengths.sum()))
    ploidy = int(ploidy)

    if indices is None:
        indices = _get_struct_indices(ploidy, multiscale_factor, lengths)
    else:
        indices = indices.copy()
    incorrect_indices = np.isnan(indices)

    # If a bin spills over chromosome / homolog boundaries, set it to whatever
    # - it will get ignored later
    indices[incorrect_indices] = 0
    indices = indices.astype(int)

    # Apply mask
    if mask is not None and mask != [None]:
        indices[~mask] = 0
        incorrect_indices = (incorrect_indices + np.invert(
            mask)).astype(bool).astype(int)

    # Apply to struct, and set incorrect indices to np.nan
    return np.where(np.repeat(incorrect_indices.reshape(-1, 1), 3, axis=1), np.nan,
                    struct.reshape(-1, 3)[indices, :]).reshape(multiscale_factor, -1, 3)


def decrease_struct_res(struct, multiscale_factor, lengths, indices=None, mask=None):
    """Decrease resolution of structure by averaging adjacent beads.
    """

    if int(multiscale_factor) != multiscale_factor:
        raise ValueError(
            "The multiscale_factor to reduce size by must be an integer.")
    if multiscale_factor == 1:
        return struct

    grouped_struct = _group_highres_struct(
        struct, multiscale_factor, lengths, indices=indices, mask=mask)

    return np.nanmean(grouped_struct, axis=0)


def _count_fullres_per_lowres_bead(multiscale_factor, lengths, ploidy,
                                   fullres_torm=None):
    """Count the number of full-res beads corresponding to each low-res bead.
    """

    if multiscale_factor == 1:
        return None

    fullres_indices = _get_struct_indices(
        ploidy=ploidy, multiscale_factor=multiscale_factor,
        lengths=lengths).reshape(multiscale_factor, -1)

    if fullres_torm is not None and fullres_torm.sum() != 0:
        fullres_indices[fullres_indices == np.where(fullres_torm)[0]] = np.nan()

    return (~ np.isnan(fullres_indices)).sum(axis=0)


def get_multiscale_variances_from_struct(structures, lengths, multiscale_factor,
                                         ploidy, mixture_coefs=None):
    """Compute multiscale variances from full-res structure.
    """

    from .utils import _format_structures

    if multiscale_factor == 1:
        return None

    structures = _format_structures(structures, lengths=lengths, ploidy=ploidy,
                                    mixture_coefs=mixture_coefs)

    multiscale_variances = []
    for struct in structures:
        struct_grouped = _group_highres_struct(
            struct, multiscale_factor, lengths)
        multiscale_variances.append(_var3d(struct_grouped))

    return np.mean(multiscale_variances, axis=0)


def _var3d(struct_grouped):
    """Compute variance of beads in 3D.
    """

    multiscale_variances = np.full(struct_grouped.shape[1], np.nan)
    for i in range(struct_grouped.shape[1]):
        struct_group = struct_grouped[:, i, :]
        mean_coords = np.nanmean(struct_group, axis=0)
        # Euclidian distance formula = ((A - B) ** 2).sum(axis=1) ** 0.5
        var = (1 / np.invert(np.isnan(struct_group)).sum()) * \
            np.nansum((struct_group - mean_coords) ** 2)
        multiscale_variances[i] = var
    return multiscale_variances


def _choose_max_multiscale_rounds(lengths, min_beads):
    multiscale_rounds = 0
    while decrease_lengths_res(lengths, 2 ** (multiscale_rounds + 1)).min() >= min_beads:
        multiscale_rounds += 1
    return multiscale_rounds


def _choose_max_multiscale_factor(lengths, min_beads):
    return 2 ** _choose_max_multiscale_rounds(
        lengths=lengths, min_beads=min_beads)
