import autograd.numpy as ag_np
import numpy as np
from sklearn.metrics import euclidean_distances


class ChromReorienter(object):
    """
    """

    def __init__(self, lengths, ploidy, init_structures=None, translate=False, rotate=False, fix_homo=True):
        self.lengths = lengths
        self.ploidy = ploidy
        self.reorient = init_structures is not None or translate or rotate
        self.translate = translate
        self.rotate = rotate
        self.fix_homo = fix_homo
        self.nchrom = lengths.shape[0]
        if not fix_homo:
            self.nchrom *= 2
        if self.reorient:
            if lengths is None:
                raise ValueError('Must supply chromosome lengths (lengths=) when finding ideal rotation and/or translation')
            if init_structures is None:
                raise ValueError('Must supply initial structures (init_structures=) when finding ideal rotation and/or translation')
            if not translate and not rotate:
                raise ValueError('Must select translate=True and/or rotate=True when finding ideal rotation and/or translation')
            if not isinstance(init_structures, list):
                init_structures = [init_structures]
            self.init_structures = [init_structure.reshape(-1, 3) for init_structure in init_structures]
            if len(set(structure.shape[0] for structure in init_structures)) > 1:
                print('Structures must all be of the same shape')
        else:
            self.init_structures = None

    def check_format(self, X, mixture_coefs=None):
        if X.shape[0] != self.nchrom * (self.translate * 3 + self.rotate * 4):
            raise ValueError("X should contain rotation quaternions (length=4) and/or translation coordinates (length=3)"
                             " for each of %d chromosomes. It is of length %d" % (self.nchrom, X.shape[0]))
        if mixture_coefs is None:
            mixture_coefs = [1]
        if len(mixture_coefs) != len(self.init_structures):
            raise ValueError("Must input the same number of mixture coefficients as initial structures.")

    def translate_and_rotate(self, X):
        if not self.reorient:
            return X
        else:
            nchrom = self.lengths.shape[0]
            if not self.fix_homo:
                nchrom *= 2

            if self.translate and self.rotate:
                translations = X[:nchrom * 3].reshape(-1, 3)
                rotations = X[nchrom * 3:].reshape(-1, 4)
            elif self.translate:
                translations = X.reshape(-1, 3)
                rotations = ag_np.zeros((nchrom, 4))
            elif self.rotate:
                rotations = X.reshape(-1, 4)
                translations = ag_np.zeros((nchrom, 3))
            else:
                raise ValueError('Must select translate=True and/or rotate=True when finding ideal rotation and/or translation')

            lengths = np.tile(self.lengths, self.ploidy)
            if self.fix_homo:
                translations = ag_np.tile(translations, (self.ploidy, 1))
                rotations = ag_np.tile(rotations, (self.ploidy, 1))

            new_structures = []
            for init_structure in self.init_structures:
                new_structure = []
                begin = end = 0
                for i in range(lengths.shape[0]):
                    length = lengths[i]
                    end += length
                    if self.rotate:
                        new_structure.append(ag_np.dot(init_structure[begin:end, :] + translations[i, :], _quat_to_rotation_matrix(rotations[i, :])))
                    else:
                        new_structure.append(init_structure[begin:end, :] + translations[i, :])
                    begin = end

                new_structure = ag_np.concatenate(new_structure)
                new_structures.append(new_structure)

            return new_structures


def _norm(x):
    return ag_np.sqrt(sum(i**2 for i in x))


def _quat_to_rotation_matrix(q):
    q = q.flatten()
    if q.shape[0] != 4:
        raise ValueError('Quaternion must be of length 4')

    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    n = _norm(q)
    if n == 0.0:
        raise ZeroDivisionError("Input to `as_rotation_matrix({0})` has zero norm".format(q))
    elif abs(n - 1.0) < np.finfo(np.float).eps:  # Input q is basically normalized
        return ag_np.array([
            [1 - 2 * (y ** 2 + z ** 2),   2 * (x * y - z * w),         2 * (x * z + y * w)],
            [2 * (x * y + z * w),         1 - 2 * (x ** 2 + z ** 2),   2 * (y * z - x * w)],
            [2 * (x * z - y * w),         2 * (y * z + x * w),         1 - 2 * (x ** 2 + y ** 2)]])
    else:  # Input q is not normalized
        return ag_np.array([
            [1 - 2 * (y ** 2 + z ** 2) / n,   2 * (x * y - z * w) / n,         2 * (x * z + y * w) / n],
            [2 * (x * y + z * w) / n,         1 - 2 * (x ** 2 + z ** 2) / n,   2 * (y * z - x * w) / n],
            [2 * (x * z - y * w) / n,         2 * (y * z + x * w) / n,         1 - 2 * (x ** 2 + y ** 2) / n]])


def _realignment_error(X, Y, error_type):
    if error_type.lower() == 'rmsd':
        return np.sqrt(((X - Y) ** 2.).sum() / len(X))
    elif error_type.lower() == 'distanceerror':
        return np.sqrt(((euclidean_distances(X) - euclidean_distances(Y)) ** 2.).sum())
    else:
        raise ValueError('Error error_type must be rmsd or distanceerror')


def _realign_structures(X, Y, rescale=False, copy=True, verbose=False, error_type='rmsd'):
    """
    Realigns Y and X

    Parameters
    ----------
    X : ndarray (n, 3)
        First 3D structure

    Y : ndarray (n, 3)
        Second 3D structure

    rescale : boolean, optional, default: False
        Whether to rescale Y or not.

    copy : boolean, optional, default: True
        Whether to copy the data or not

    verbose : boolean, optional, default: False
        The level of verbosity

    Returns
    -------
    Y : ndarray (n, 3)
        Realigned 3D, Xstructure
    """

    from scipy import linalg

    if copy:
        Y = Y.copy()
        X = X.copy()

    mask = np.invert(np.isnan(X[:, 0]) | np.isnan(Y[:, 0]))

    if rescale:
        Y, _, _, _ = _realign_structures(X, Y)
        if error_type.lower() == 'rmsd':
            alpha = (X[mask] * Y[mask]).sum() / (Y[mask] ** 2).sum()  # np.sqrt(((X - Y) ** 2.).sum() / len(X))
        elif error_type.lower() == 'distanceerror':
            alpha = (euclidean_distances(X[mask]) * euclidean_distances(Y[mask])).sum() / (euclidean_distances(Y[mask]) ** 2).sum()  # np.sqrt(((euclidean_distances(X) - euclidean_distances(Y)) ** 2.).sum())

        Y *= alpha

    X -= np.nanmean(X, axis=0)
    Y -= np.nanmean(Y, axis=0)

    K = np.dot(X[mask].T, Y[mask])
    U, L, V = linalg.svd(K)
    V = V.T

    R = np.dot(V, U.T)
    if linalg.det(R) < 0:
        if verbose:
            print("Reflexion found")
        V[:, -1] *= -1
        R = np.dot(V, U.T)
    Y_fit = np.dot(Y, R)

    error = _realignment_error(X[mask], Y_fit[mask], error_type)

    # Check at the mirror
    Y_mirror = Y.copy()
    Y_mirror[:, 0] = - Y[:, 0]

    K = np.dot(X[mask].T, Y_mirror[mask])
    U, L, V = linalg.svd(K)
    V = V.T
    if linalg.det(V) < 0:
        V[:, -1] *= -1

    R_mirror = np.dot(V, U.T)
    Y_mirror_fit = np.dot(Y_mirror, R_mirror)
    error_mirror = _realignment_error(X[mask], Y_mirror_fit[mask], error_type)

    if error <= error_mirror:
        best_Y_fit = Y_fit
        best_error = error
        mirror = False
        best_R = R
    else:
        if verbose:
            print("Reflexion is better")
        best_Y_fit = Y_mirror_fit
        best_error = error_mirror
        mirror = True
        best_R = R_mirror

    return best_Y_fit, best_error, mirror, best_R


def _lowres_genome_vs_highres_chrom(X_genome_lowres, X_chrom, ploidy, lengths, chromosomes, chrom, lowres_genome_factor, piecewise_fix_homo):
    from .counts import _get_chrom_subset_index
    from .multiscale_optimization import decrease_lengths_res, decrease_struct_res
    import quaternion

    # Extract the chromosome from low-res whole-genome structure
    index, chrom_lengths_lowres = _get_chrom_subset_index(ploidy, lengths_full=decrease_lengths_res(lengths, lowres_genome_factor), chrom_full=chromosomes, chrom_subset=[chrom])
    X_lowres_2chrom = X_genome_lowres = X_genome_lowres[index]

    # Reduce the resolution of high-res single-chromosome structure
    chrom_lengths = np.array([int(X_chrom.shape[0] / ploidy)])
    X_chrom_2lowres = decrease_struct_res(X=X_chrom, multiscale_factor=lowres_genome_factor, lengths=chrom_lengths)

    # Compute for each chromosome (all homologs together) or each homolog
    X_lowres_2chrom_list = [X_lowres_2chrom]
    X_chrom_2lowres_list = [X_chrom_2lowres]
    X_chrom_list = [X_chrom]
    if ploidy == 2 and not piecewise_fix_homo:
        X_lowres_2chrom_list = [X_lowres_2chrom[:chrom_lengths_lowres.sum()], X_lowres_2chrom[chrom_lengths_lowres.sum():]]
        X_chrom_2lowres_list = [X_chrom_2lowres[:chrom_lengths_lowres.sum()], X_chrom_2lowres[chrom_lengths_lowres.sum():]]
        X_chrom_list = [X_chrom[:chrom_lengths.sum()], X_chrom[chrom_lengths.sum():]]

    homolog_translations = []
    homolog_rotations = []
    highres_homolog_mirrored = []
    for each_X_lowres_2chrom, each_X_chrom_2lowres, each_X_chrom in zip(X_lowres_2chrom_list, X_chrom_2lowres_list, X_chrom_list):
        # Find ideal translation, rotation & mirroring; Convert rotation matrix to quaternions
        homolog_translations.append(np.nanmean(each_X_lowres_2chrom, axis=0) - np.nanmean(each_X_chrom_2lowres, axis=0))
        Y_fit, rmsd, mirror, rotation_matrix = _realign_structures(X=each_X_lowres_2chrom, Y=each_X_chrom_2lowres, rescale=False, type='RMSD', return_realignment_details=True)
        homolog_rotations.append(quaternion.as_float_array(quaternion.from_rotation_matrix(rotation_matrix)))

        # Apply mirror to highres chrom
        if mirror:
            each_X_chrom[:, 0] = - each_X_chrom[:, 0]
        highres_homolog_mirrored.append(each_X_chrom)

    return homolog_translations, homolog_rotations, highres_homolog_mirrored


def _assemble_highres_chrom_via_lowres_genome(outdir, outdir_lowres, outdir_orient, chromosomes, lengths, alpha, ploidy, lowres_genome_factor, piecewise_fix_homo=True, msv_type=None, struct_true=None, modifications=None):
    import os
    from ..io.read import _load_inferred_struct

    # Load inferred lowres genome
    X_genome_lowres = _load_inferred_struct(outdir_lowres)

    # Load each inferred chromosome
    # Use lowres genome to decide when to mirror inferred chromosomes & initialize chromosome positons
    X_all_chrom = []
    all_rotations = []
    all_translations = []
    for chrom in chromosomes:
        X_chrom = _load_inferred_struct(os.path.join(outdir, chrom))
        homolog_translations, homolog_rotations, highres_homolog_mirrored = _lowres_genome_vs_highres_chrom(X_genome_lowres, X_chrom, ploidy, lengths, chromosomes, chrom, lowres_genome_factor, piecewise_fix_homo)
        X_all_chrom.extend(highres_homolog_mirrored)
        all_rotations.extend(homolog_rotations)
        all_translations.extend(homolog_translations)
    X_all_chrom = np.concatenate(X_all_chrom)
    all_rotations = np.array(all_rotations)
    all_translations = np.array(all_translations)
    trans_rot_init = np.concatenate([all_translations.flatten(), all_rotations.flatten()])
    reorienter = ChromReorienter(lengths, ploidy, init_structures=X_all_chrom, translate=True, rotate=True, fix_homo=piecewise_fix_homo)
    reorienter.translate_and_rotate(trans_rot_init)[0].reshape(-1, 3)
    try:
        os.makedirs(outdir_orient)
    except OSError:
        pass
    np.savetxt(os.path.join(outdir_orient, 'orient_via_lowres.txt'), trans_rot_init)
    np.savetxt(os.path.join(outdir_orient, 'orient_via_lowres.X.txt'), X_all_chrom)

    return X_all_chrom, trans_rot_init


def stepwise_inference(counts, outdir, lengths, ploidy, chromosomes, alpha, seed=0, normalize=True,
                       filter_threshold=0.04, alpha_init=-3., max_alpha_loop=20, multiscale_rounds=1,
                       use_multiscale_variance=True,
                       max_iter=1e40, factr=10000000.0, pgtol=1e-05, alpha_factr=1000000000000.,
                       bcc_lambda=0., hsc_lambda=0., hsc_r=None, hsc_min_beads=5,
                       callback_function=None, callback_freq=None,
                       piecewise_step=None, piecewise_chrom=None,
                       piecewise_min_beads=5, piecewise_fix_homo=False, piecewise_opt_orient=True,
                       alpha_true=None, struct_true=None, init='msd', input_weight=None,
                       exclude_zeros=False, null=False, mixture_coefs=None, verbose=True):
    """
    """

    import os
    from ..io.read import _load_inferred_struct, _choose_best_seed
    from .counts import subset_chrom
    from .pastis_algorithms import infer, _output_subdir
    from .utils import _choose_max_multiscale_factor, _print_code_header

    if piecewise_step is None:
        piecewise_step = [1, 2, 3, 4]
    piecewise_step = [{'lowres': 1, 'chrom': 2, 'orient': 3, 'final': 4, 1: 1, 2: 2, 3: 3, 4: 4}[x.lower() if isinstance(x, str) else x] for x in piecewise_step]

    lowres_genome_factor = _choose_max_multiscale_factor(lengths=lengths, min_beads=piecewise_min_beads)

    # Set directories
    outdir_lowres = _output_subdir(outdir=outdir, chrom_full=chromosomes, null=null, piecewise=True, piecewise_step=1)
    outdir_orient = _output_subdir(outdir=outdir, chrom_full=chromosomes, null=null, piecewise=True, piecewise_step=3)
    outdir_final = _output_subdir(outdir=outdir, chrom_full=chromosomes, null=null, piecewise=True, piecewise_step=4)

    # Infer entire genome at low res; optionally use to infer alpha????? probably, but WOULD have to test that
    # Also used to initialize individual chrom, right? account for seed tho.... ugh but this would really increase run time
    # It WILL at least initialize the positions of chrom when assembling them all together
    if 1 in piecewise_step:
        _print_code_header('STEWISE GENOME ASSEMBLY: STEP 1', sub_header='Inferring low-res whole-genome structure', max_length=80, blank_lines=2)

        # Infer all chromosomes at this resolution only
        infer(
            counts=counts, lengths=lengths, alpha=alpha, ploidy=ploidy, init=init, outdir=outdir_lowres, bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda, hsc_r=hsc_r, input_weight=input_weight, exclude_zeros=exclude_zeros, normalize=normalize,
            alpha_init=alpha_init, alpha_true=alpha_true, max_alpha_loop=max_alpha_loop, alpha_factr=alpha_factr,
            lowres_genome_factor=lowres_genome_factor, hsc_min_beads=hsc_min_beads, null=null, filter_threshold=filter_threshold, struct_true=struct_true, max_iter=max_iter, factr=factr, pgtol=pgtol)

    # Load lowres genome inferred variables
    lowres_var = None
    if (alpha is None) and (2 in piecewise_step or 3 in piecewise_step or 4 in piecewise_step):
        lowres_var = _choose_best_seed(outdir_lowres)
        alpha = float(lowres_var['alpha'])
        beta = [float(x) for x in lowres_var['beta'].strip('[]').split(' ')]

    # Infer each chromosome individually
    if 2 in piecewise_step:
        _print_code_header('STEWISE GENOME ASSEMBLY: STEP 2', sub_header='Inferring high-res structure per chromosome', max_length=80, blank_lines=2)
        if piecewise_chrom is None:
            piecewise_chrom = chromosomes
        for chrom in piecewise_chrom:
            _print_code_header('CHROMOSOME %s' % chrom, max_length=70, blank_lines=1)
            chrom_lengths, _, chrom_counts, chrom_struct_true = subset_chrom(counts=counts, ploidy=ploidy, lengths_full=lengths, chrom_full=chromosomes, chrom_subset=chrom, exclude_zeros=exclude_zeros, struct_true=struct_true)

            infer(counts=chrom_counts, lengths=chrom_lengths, alpha=alpha, ploidy=ploidy, init=init, outdir=os.path.join(outdir, chrom), bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda, hsc_r=hsc_r, input_weight=input_weight, exclude_zeros=exclude_zeros, normalize=normalize,
                  alpha_init=alpha_init, alpha_true=alpha_true, max_alpha_loop=max_alpha_loop, alpha_factr=alpha_factr,
                  multiscale_rounds=multiscale_rounds, hsc_min_beads=hsc_min_beads, null=null, filter_threshold=filter_threshold, struct_true=chrom_struct_true, max_iter=max_iter, factr=factr, pgtol=pgtol)

    # Assemble 3D structure for entire genome
    if 3 in piecewise_step:
        _print_code_header('STEWISE GENOME ASSEMBLY: STEP 3', sub_header='Orienting high-res chromosomes', max_length=80, blank_lines=2)

        X_all_chrom, trans_rot_init = _assemble_highres_chrom_via_lowres_genome(outdir=outdir, outdir_lowres=outdir_lowres, outdir_orient=outdir_orient, chromosomes=chromosomes, lengths=lengths, alpha=alpha, ploidy=ploidy, lowres_genome_factor=lowres_genome_factor, piecewise_fix_homo=piecewise_fix_homo, struct_true=struct_true)

        # OPTIONAL - Assemble all chromosomes together - rotate & translate previously inferred chromosomes
        if piecewise_opt_orient:
            infer(
                counts=counts, lengths=lengths, alpha=alpha, ploidy=ploidy, init=trans_rot_init, outdir=outdir_orient, bcc_lambda=0., hsc_lambda=hsc_lambda, hsc_r=hsc_r, input_weight=input_weight, exclude_zeros=exclude_zeros, normalize=normalize,
                alpha_init=alpha_init, alpha_true=alpha_true, max_alpha_loop=max_alpha_loop, alpha_factr=alpha_factr,
                initial_seed=None, num_infer=1, multiscale_rounds=multiscale_rounds, hsc_min_beads=hsc_min_beads,
                init_structures=X_all_chrom, translate=True, rotate=True, piecewise_fix_homo=piecewise_fix_homo,
                null=null, filter_threshold=filter_threshold, struct_true=struct_true, max_iter=max_iter, factr=factr, pgtol=pgtol)

    # Infer once more, letting all bead positions vary
    if 4 in piecewise_step:
        _print_code_header('STEWISE GENOME ASSEMBLY: STEP 4', sub_header='Final whole-genome inference, allowing all beads to vary', max_length=80, blank_lines=2)

        if piecewise_opt_orient:
            X_all_chrom_oriented = _load_inferred_struct(outdir_orient)
        else:
            X_all_chrom_oriented = np.loadtxt(os.path.join(outdir_orient, 'X.orient_via_lowres.txt'))

        infer(
            counts=counts, lengths=lengths, alpha=alpha, ploidy=ploidy, init=X_all_chrom_oriented, outdir=outdir_final, bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda, hsc_r=hsc_r, input_weight=input_weight, exclude_zeros=exclude_zeros, normalize=normalize,
            alpha_init=alpha_init, alpha_true=alpha_true, max_alpha_loop=max_alpha_loop, alpha_factr=alpha_factr,
            initial_seed=None, num_infer=1, multiscale_rounds=multiscale_rounds, hsc_min_beads=hsc_min_beads, null=null, filter_threshold=filter_threshold, struct_true=struct_true, max_iter=max_iter, factr=factr, pgtol=pgtol)
