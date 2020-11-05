import autograd.numpy as ag_np
import numpy as np
from sklearn.metrics import euclidean_distances
import os
from scipy import linalg
from .counts import subset_chrom, _get_chrom_subset_index, preprocess_counts
from .pastis_algorithms import infer, _infer_draft
from .utils_poisson import _print_code_header, _load_infer_var
from .utils_poisson import _format_structures, _output_subdir
from .multiscale_optimization import _choose_max_multiscale_factor
from .multiscale_optimization import decrease_lengths_res, decrease_struct_res


class ChromReorienter(object):
    """
    """

    def __init__(self, lengths, ploidy, struct_init=None, translate=False,
                 rotate=False, fix_homo=True, mixture_coefs=None):
        self.lengths_fullres = lengths
        self.ploidy = ploidy
        self.reorient = translate or rotate
        self.translate = translate
        self.rotate = rotate
        self.fix_homo = fix_homo
        self.nchrom = lengths.shape[0]
        if not fix_homo:
            self.nchrom *= self.ploidy
        if self.reorient:
            if lengths is None:
                raise ValueError(
                    "Must supply lengths when rotating or translating.")
            if struct_init is None:
                raise ValueError(
                    "Must supply struct_init when rotating or translating.")
            if not isinstance(struct_init, list):
                struct_init = [struct_init]
            self.struct_init_fullres = _format_structures(
                struct_init, lengths=lengths, ploidy=ploidy,
                mixture_coefs=mixture_coefs)
        else:
            self.struct_init_fullres = None
        self.lengths = self.lengths_fullres
        self.struct_init = self.struct_init_fullres

    def set_multiscale_factor(self, multiscale_factor=1):
        self.lengths = decrease_lengths_res(
            self.lengths_fullres, multiscale_factor=multiscale_factor)
        if self.reorient:
            self.struct_init = [decrease_struct_res(
                s, multiscale_factor=multiscale_factor,
                lengths=self.lengths_fullres) for s in self.struct_init_fullres]

    def check_X(self, X):
        """
        """

        if self.reorient and len(X.shape) > 1:
            raise ValueError(
                "X must be 1-dimensional. it is of shape (%s)"
                % (','.join([str(x) for x in X.shape])))
        expected_X_len = self.nchrom * (self.translate * 3 + self.rotate * 4)
        if X.shape[0] != expected_X_len and self.translate and self.rotate:
            raise ValueError(
                "X should contain rotation quaternions (4,) and translation"
                " coordinates (3,) for each of %d chromosomes. Expected shape"
                " is (%d,). It is of shape (%d,)"
                % (self.nchrom, expected_X_len, X.shape[0]))
        elif X.shape[0] != expected_X_len and self.translate:
            raise ValueError(
                "X should contain translation coordinates (3,) for each of %d"
                " chromosomes. Expected shape is (%d,). It is of shape (%d,)"
                % (self.nchrom, expected_X_len, X.shape[0]))
        elif X.shape[0] != expected_X_len and self.rotate:
            raise ValueError(
                "X should contain rotation quaternions (4,) for each of %d "
                "chromosomes. Expected shape is (%d,). It is of shape (%d,)"
                % (self.nchrom, expected_X_len, X.shape[0]))

    def translate_and_rotate(self, X):
        """
        """

        if not self.reorient:
            return X
        else:
            if self.translate and self.rotate:
                translations = X[:self.nchrom * 3].reshape(-1, 3)
                rotations = X[self.nchrom * 3:].reshape(-1, 4)
            elif self.translate:
                translations = X.reshape(-1, 3)
                rotations = ag_np.zeros((self.nchrom, 4))
            elif self.rotate:
                rotations = X.reshape(-1, 4)
                translations = ag_np.zeros((self.nchrom, 3))
            else:
                raise ValueError("Must translate or rotate.")

            lengths_tiled = np.tile(self.lengths, self.ploidy)
            if self.fix_homo:
                translations = ag_np.tile(translations, (self.ploidy, 1))
                rotations = ag_np.tile(rotations, (self.ploidy, 1))

            new_structures = []
            for init_structure in self.struct_init:
                new_structure = []
                begin = end = 0
                for i in range(lengths_tiled.shape[0]):
                    length = lengths_tiled[i]
                    end += length
                    if self.rotate:
                        new_structure.append(ag_np.dot(
                            init_structure[begin:end, :] + translations[i, :],
                            _quat_to_rotation_matrix(rotations[i, :])))
                    else:
                        new_structure.append(
                            init_structure[begin:end, :] + translations[i, :])
                    begin = end

                new_structure = ag_np.concatenate(new_structure)
                new_structures.append(new_structure)

            return new_structures


def _norm(x):
    """Vector norm.
    """

    return ag_np.sqrt(ag_np.sum(x ** 2))


def _quat_to_rotation_matrix(q):
    """Convert quartnerion to rotation matrix.
    """

    q = q.flatten()
    if q.shape[0] != 4:
        raise ValueError('Quaternion must be of length 4')

    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    n = ag_np.sum(q ** 2)
    if n == 0.0:
        raise ZeroDivisionError(
            "Input to `_quat_to_rotation_matrix({0})` has zero norm".format(q))
    elif abs(n - 1.0) < np.finfo(np.float).eps:
        # Input q is basically normalized
        return ag_np.array([
            [1 - 2 * (y ** 2 + z ** 2),   2 * (x * y - z * w),         2 * (x * z + y * w)],
            [2 * (x * y + z * w),         1 - 2 * (x ** 2 + z ** 2),   2 * (y * z - x * w)],
            [2 * (x * z - y * w),         2 * (y * z + x * w),         1 - 2 * (x ** 2 + y ** 2)]])
    else:
        # Input q is not normalized
        return ag_np.array([
            [1 - 2 * (y ** 2 + z ** 2) / n,   2 * (x * y - z * w) / n,         2 * (x * z + y * w) / n],
            [2 * (x * y + z * w) / n,         1 - 2 * (x ** 2 + z ** 2) / n,   2 * (y * z - x * w) / n],
            [2 * (x * z - y * w) / n,         2 * (y * z + x * w) / n,         1 - 2 * (x ** 2 + y ** 2) / n]])


def _realignment_error(X, Y, use_disterror=False):
    """Error score to be used for realignment, RMSD or distance error.
    """

    mask = np.invert(np.isnan(X[:, 0]) | np.isnan(Y[:, 0]))

    if use_disterror:
        dis_X = euclidean_distances(X[mask])
        dis_Y = euclidean_distances(Y[mask])
        return np.sqrt(((dis_X - dis_Y) ** 2.).sum())
    else:
        return np.sqrt(((X[mask] - Y[mask]) ** 2.).sum() / len(X[mask]))


def _realign_structures(X, Y, rescale=False, copy=True, verbose=False,
                        use_disterror=False):
    """Realigns Y and X.

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

    if copy:
        Y = Y.copy()
        X = X.copy()

    mask = np.invert(np.isnan(X[:, 0]) | np.isnan(Y[:, 0]))

    X -= np.nanmean(X, axis=0)
    Y -= np.nanmean(Y, axis=0)

    if rescale:
        Y, _, _, _ = _realign_structures(X, Y)
        if use_disterror:
            dis_X = euclidean_distances(X[mask])
            dis_Y = euclidean_distances(Y[mask])
            scale_factor = (dis_X * dis_Y).sum() / (dis_X ** 2).sum()
        else:
            scale_factor = (X[mask] * Y[mask]).sum() / (Y[mask] ** 2).sum()

        Y *= scale_factor

    K = np.dot(X[mask].T, Y[mask])
    U, L, V = linalg.svd(K)
    V = V.T

    R = np.dot(V, U.T)
    if linalg.det(R) < 0:
        if verbose:
            print("Reflexion found.", flush=True)
        V[:, -1] *= -1
        R = np.dot(V, U.T)
    Y_fit = np.dot(Y, R)

    error = _realignment_error(
        X[mask], Y_fit[mask], use_disterror=use_disterror)

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
    error_mirror = _realignment_error(
        X[mask], Y_mirror_fit[mask], use_disterror=use_disterror)

    if error < error_mirror:
        best_Y_fit = Y_fit
        best_error = error
        mirror = False
        best_R = R
    else:
        if verbose:
            print("Reflexion is better.", flush=True)
        best_Y_fit = Y_mirror_fit
        best_error = error_mirror
        mirror = True
        best_R = R_mirror

    return best_Y_fit, best_error, mirror, best_R


def _orient_single_fullres_chrom(struct_genome_lowres, struct_chrom_fullres,
                                 ploidy, lengths, chromosomes, chrom,
                                 piecewise_factor, piecewise_fix_homo):
    """
    """

    import quaternion

    # Extract the chromosome from low-res whole-genome structure
    index, chrom_lengths_lowres = _get_chrom_subset_index(
        ploidy=ploidy,
        lengths_full=decrease_lengths_res(lengths, piecewise_factor),
        chrom_full=chromosomes, chrom_subset=[chrom])
    struct_lowres_genome2chrom = struct_genome_lowres[index]

    # Reduce the resolution of full-res single-chromosome structure
    chrom_lengths = np.array([int(struct_chrom_fullres.shape[0] / ploidy)])
    struct_chrom_fullres2lowres = decrease_struct_res(
        struct=struct_chrom_fullres, multiscale_factor=piecewise_factor,
        lengths=chrom_lengths)

    # For diploid, optionally find separate orientations per homolog
    if ploidy == 2 and not piecewise_fix_homo:
        n_lowres = chrom_lengths_lowres.sum()
        struct_lowres_genome2chrom = [struct_lowres_genome2chrom[:n_lowres],
                                      struct_lowres_genome2chrom[n_lowres:]]
        struct_chrom_fullres2lowres = [struct_chrom_fullres2lowres[:n_lowres],
                                       struct_chrom_fullres2lowres[n_lowres:]]
        struct_chrom_fullres = [struct_chrom_fullres[:chrom_lengths.sum()],
                                struct_chrom_fullres[chrom_lengths.sum():]]
    else:
        struct_lowres_genome2chrom = [struct_lowres_genome2chrom]
        struct_chrom_fullres2lowres = [struct_chrom_fullres2lowres]
        struct_chrom_fullres = [struct_chrom_fullres]

    # Find ideal translation, rotation & mirroring
    homo_struct = zip(
        struct_lowres_genome2chrom, struct_chrom_fullres2lowres,
        struct_chrom_fullres)
    trans = []
    rot = []
    fullres_chrom_mirrored = []
    for lowres_genome2chrom, chrom_fullres2lowres, chrom_fullres in homo_struct:
        trans.append(np.nanmean(lowres_genome2chrom, axis=0) - np.nanmean(
            chrom_fullres2lowres, axis=0))
        _, _, mirror, rotation_matrix = _realign_structures(
            X=lowres_genome2chrom, Y=chrom_fullres2lowres, rescale=False,
            use_disterror=False)

        # Convert rotation matrix to quaternions
        rot.append(quaternion.as_float_array(
            quaternion.from_rotation_matrix(rotation_matrix)))

        # Apply mirror to full-res chrom
        if mirror:
            chrom_fullres[:, 0] = - chrom_fullres[:, 0]
        fullres_chrom_mirrored.append(chrom_fullres)

    return trans, rot, fullres_chrom_mirrored


def _orient_fullres_chroms_via_lowres_genome(outdir, seed, chromosomes,
                                             lengths, alpha, ploidy,
                                             piecewise_factor,
                                             piecewise_fix_homo=True,
                                             mixture_coefs=None):
    """
    """

    outdir_lowres = _output_subdir(outdir=outdir, piecewise_step=1)
    outdir_chrom = _output_subdir(outdir=outdir, piecewise_step=2)
    outdir_orient = _output_subdir(outdir=outdir, piecewise_step=3)
    outdir_orient_init = os.path.join(outdir_orient, 'via_step1')

    struct_genome_lowres = np.loadtxt(os.path.join(
        outdir_lowres, 'struct_inferred.%03d.coords' % seed))

    struct_fullres_genome_init = []
    rotations_homo1 = []
    translations_homo1 = []
    rotations_homo2 = []
    translations_homo2 = []
    for chrom in chromosomes:
        struct_chrom_fullres = np.loadtxt(os.path.join(
            outdir_chrom, chrom, 'struct_inferred.%03d.coords' % seed))
        trans, rot, fullres_chrom_mirrored = _orient_single_fullres_chrom(
            struct_genome_lowres=struct_genome_lowres,
            struct_chrom_fullres=struct_chrom_fullres, ploidy=ploidy,
            lengths=lengths, chromosomes=chromosomes, chrom=chrom,
            piecewise_factor=piecewise_factor,
            piecewise_fix_homo=piecewise_fix_homo)
        struct_fullres_genome_init.extend(fullres_chrom_mirrored)
        translations_homo1.extend(trans[0])
        rotations_homo1.extend(rot[0])
        if len(trans) > 1:
            translations_homo2.extend(trans[1])
            rotations_homo2.extend(rot[1])
    translations = translations_homo1 + translations_homo2
    rotations = rotations_homo1 + rotations_homo2

    struct_fullres_genome_init = np.concatenate(struct_fullres_genome_init)
    reorient_init = np.concatenate(
        [np.array(translations).flatten(),
         np.array(rotations).flatten()])

    reorienter = ChromReorienter(
        lengths=lengths, ploidy=ploidy, struct_init=struct_fullres_genome_init,
        translate=True, rotate=True, fix_homo=piecewise_fix_homo,
        mixture_coefs=mixture_coefs)
    struct_fullres_genome_reoriented = reorienter.translate_and_rotate(
        reorient_init)[0].reshape(-1, 3)

    try:
        os.makedirs(outdir_orient_init)
    except OSError:
        pass
    np.savetxt(
        os.path.join(outdir_orient_init,
                     'orient_via_step1.%03d.trans_rot' % seed),
        reorient_init)
    np.savetxt(
        os.path.join(outdir_orient_init,
                     'struct_orient_via_step1.%03d.coords' % seed),
        struct_fullres_genome_reoriented)

    return reorient_init, struct_fullres_genome_init, struct_fullres_genome_reoriented


def infer_piecewise(counts_raw, outdir, lengths, ploidy, chromosomes, alpha,
                    seed=0, normalize=True, filter_threshold=0.04,
                    alpha_init=-3., max_alpha_loop=20, beta=None,
                    multiscale_rounds=1, use_multiscale_variance=True,
                    max_iter=1e40, factr=10000000., pgtol=1e-05,
                    alpha_factr=1000000000000., bcc_lambda=0., hsc_lambda=0.,
                    hsc_r=None, hsc_min_beads=5, mhs_lambda=0., mhs_k=None,
                    callback_function=None, callback_freq=None,
                    piecewise_step=None, piecewise_chrom=None,
                    piecewise_min_beads=5, piecewise_fix_homo=False,
                    piecewise_opt_orient=True, piecewise_step3_multiscale=False,
                    piecewise_step1_accuracy=1,
                    alpha_true=None, struct_true=None, init='msd',
                    input_weight=None, exclude_zeros=False, null=False,
                    mixture_coefs=None, verbose=True):
    """Infer whole genome 3D structures piecewise, first inferring chromosomes.
    """

    if piecewise_step is None:
        piecewise_step = [1, 2, 3, 4]
    elif not (isinstance(piecewise_step, list) or isinstance(
            piecewise_step, np.ndarray)):
        piecewise_step = [piecewise_step]
    for step in piecewise_step:
        if step not in (1, 2, 3, 4):
            raise ValueError(
                "`piecewise_step` of %d not understood. Options = (1, 2, 3, 4)"
                % step)

    piecewise_factor = _choose_max_multiscale_factor(
        lengths=lengths, min_beads=piecewise_min_beads)

    # Set directories
    outdir_lowres = _output_subdir(outdir=outdir, piecewise_step=1)
    outdir_chrom = _output_subdir(outdir=outdir, piecewise_step=2)
    outdir_orient = _output_subdir(outdir=outdir, piecewise_step=3)

    # Infer draft structure
    step1_multiscale = 1 in piecewise_step and piecewise_factor > 1
    step2_multiscale = multiscale_rounds > 1 and 2 in piecewise_step
    step3_multiscale = multiscale_rounds > 1 and 3 in piecewise_step and \
        piecewise_step3_multiscale
    infer_draft_lowres = hsc_lambda > 0 and hsc_r is None
    need_multiscale_var = use_multiscale_variance and (
        step1_multiscale or step2_multiscale or step3_multiscale or
        infer_draft_lowres)
    infer_draft_fullres = need_multiscale_var or alpha is None

    if infer_draft_fullres or infer_draft_lowres:
        struct_draft_fullres, alpha_, beta_, hsc_r, draft_converged = _infer_draft(
            counts_raw, lengths=lengths, ploidy=ploidy, outdir=outdir,
            alpha=alpha, seed=seed, normalize=normalize,
            filter_threshold=filter_threshold, alpha_init=alpha_init,
            max_alpha_loop=max_alpha_loop, beta=beta, multiscale_rounds=2,
            use_multiscale_variance=use_multiscale_variance, init=init,
            max_iter=max_iter, factr=factr, pgtol=pgtol,
            alpha_factr=alpha_factr, hsc_lambda=hsc_lambda, hsc_r=hsc_r,
            hsc_min_beads=hsc_min_beads, callback_freq=callback_freq,
            callback_function=callback_function, alpha_true=alpha_true,
            struct_true=struct_true, input_weight=input_weight,
            exclude_zeros=exclude_zeros, null=null, mixture_coefs=mixture_coefs,
            verbose=verbose)
        if not draft_converged:
            return None, {'alpha': alpha_, 'beta': beta_, 'seed': seed,
                          'converged': draft_converged}
    else:
        alpha_ = alpha
        beta_ = beta
        struct_draft_fullres = None

    # Prepare multiscale optimization
    if step1_multiscale or step2_multiscale or step3_multiscale:
        _, _, _, fullres_torm_for_multiscale = preprocess_counts(
            counts_raw=counts_raw, lengths=lengths, ploidy=ploidy,
            normalize=normalize, filter_threshold=filter_threshold,
            multiscale_factor=1, exclude_zeros=exclude_zeros, beta=beta,
            input_weight=input_weight, verbose=verbose,
            mixture_coefs=mixture_coefs)
    else:
        fullres_torm_for_multiscale = None

    # Infer entire genome at low res
    if 1 in piecewise_step:

        note1 = []
        note2 = []
        if piecewise_step1_accuracy != 1:
            factr_lowres = 10 ** (np.log10(factr) * piecewise_step1_accuracy)
            note1.append('low-accuracy')
            note2.append(('factr=%.3g' % factr_lowres).replace(
                'e+0', 'e').replace('e+', 'e'))
        else:
            factr_lowres = factr
        if piecewise_factor > 1:
            note1.append('low-res')
            note2.append('%dx' % piecewise_factor)
        note1 = ', '.join(note1)
        note2 = ', '.join(note2)
        if len(note2) > 0:
            note1 = ' %s' % note1
            note2 = ' (%s)' % note2

        _print_code_header(
            ['PIECEWISE WHOLE GENOME: STEP 1',
                'Inferring%s whole-genome structure%s' % (note1, note2)],
            max_length=80, blank_lines=2)

        struct_, infer_var = infer(
            counts_raw=counts_raw, outdir=outdir_lowres,
            lengths=lengths, ploidy=ploidy, alpha=alpha_, seed=seed,
            normalize=normalize, filter_threshold=filter_threshold,
            alpha_init=alpha_init, max_alpha_loop=max_alpha_loop,
            beta=beta_, multiscale_factor=piecewise_factor,
            use_multiscale_variance=use_multiscale_variance,
            init=init, max_iter=max_iter, factr=factr_lowres, pgtol=pgtol,
            alpha_factr=alpha_factr, bcc_lambda=bcc_lambda,
            hsc_lambda=hsc_lambda, hsc_r=hsc_r, mhs_lambda=mhs_lambda,
            mhs_k=mhs_k, fullres_torm=fullres_torm_for_multiscale,
            struct_draft_fullres=struct_draft_fullres,
            callback_function=callback_function,
            callback_freq=callback_freq,
            alpha_true=alpha_true, struct_true=struct_true,
            input_weight=input_weight, exclude_zeros=exclude_zeros,
            null=null, mixture_coefs=mixture_coefs, verbose=verbose)
        if not infer_var['converged']:
            return struct_, infer_var

    # Load variables for subsequent steps
    infer_var_file = os.path.join(
        outdir_lowres, 'inference_variables.%03d' % seed)
    infer_var = _load_infer_var(infer_var_file)
    if not infer_var['converged']:
        return None, infer_var
    alpha_ = infer_var['alpha']
    beta_ = infer_var['beta']
    if 'hsc_r' in infer_var:
        hsc_r = infer_var['hsc_r']
    if 'mhs_k' in infer_var:
        mhs_k = infer_var['mhs_k']

    # Infer each chromosome individually
    if 2 in piecewise_step:
        _print_code_header(
            ['PIECEWISE WHOLE GENOME: STEP 2',
                'Inferring full-res structure per chromosome'],
            max_length=80, blank_lines=2)

        if piecewise_chrom is None:
            piecewise_chrom = chromosomes

        for i in range(len(piecewise_chrom)):
            chrom = piecewise_chrom[i]
            if hsc_r is not None:
                hsc_r_chrom = hsc_r[i]
            else:
                hsc_r_chrom = None
            if mhs_k is not None and isinstance(mhs_k, np.ndarray):
                mhs_k_chrom = mhs_k[i]
            else:
                mhs_k_chrom = mhs_k

            _print_code_header(
                'CHROMOSOME %s' % chrom, max_length=70, blank_lines=1)

            chrom_lengths, _, chrom_counts, chrom_struct_true = subset_chrom(
                counts=counts_raw, ploidy=ploidy, lengths_full=lengths,
                chrom_full=chromosomes, chrom_subset=chrom,
                exclude_zeros=exclude_zeros, struct_true=struct_true)
            index, _ = _get_chrom_subset_index(
                ploidy=ploidy, lengths_full=lengths, chrom_full=chromosomes,
                chrom_subset=chrom)
            if ploidy == 2:
                draft_index = index[:lengths.sum()]
            else:
                draft_index = index
            if struct_draft_fullres is None:
                struct_draft_fullres_chrom = None
            else:
                struct_draft_fullres_chrom = struct_draft_fullres[draft_index]
            if fullres_torm_for_multiscale is None:
                fullres_torm_chrom = None
            else:
                fullres_torm_chrom = [
                    x[index] for x in fullres_torm_for_multiscale]

            struct_, infer_var = infer(
                counts_raw=chrom_counts,
                outdir=os.path.join(outdir_chrom, chrom),
                lengths=chrom_lengths, ploidy=ploidy, alpha=alpha_, seed=seed,
                normalize=normalize, filter_threshold=filter_threshold,
                alpha_init=alpha_init, max_alpha_loop=max_alpha_loop,
                beta=beta_, multiscale_rounds=multiscale_rounds,
                use_multiscale_variance=use_multiscale_variance,
                init=init, max_iter=max_iter, factr=factr, pgtol=pgtol,
                alpha_factr=alpha_factr, bcc_lambda=bcc_lambda,
                hsc_lambda=hsc_lambda, hsc_r=hsc_r_chrom,
                mhs_lambda=mhs_lambda, mhs_k=mhs_k_chrom,
                fullres_torm=fullres_torm_chrom,
                struct_draft_fullres=struct_draft_fullres_chrom,
                callback_function=callback_function,
                callback_freq=callback_freq, alpha_true=alpha_true,
                struct_true=chrom_struct_true, input_weight=input_weight,
                exclude_zeros=exclude_zeros, null=null,
                mixture_coefs=mixture_coefs, verbose=verbose)
            if not infer_var['converged']:
                return struct_, infer_var

    # Assemble 3D structure for entire genome
    if 3 in piecewise_step:
        _print_code_header(
            ['PIECEWISE WHOLE GENOME: STEP 3', 'Orienting full-res chromosomes'],
            max_length=80, blank_lines=2)

        reorient_init, struct_fullres_genome_init, struct_fullres_genome_reoriented = _orient_fullres_chroms_via_lowres_genome(
            outdir=outdir, seed=seed, chromosomes=chromosomes,
            lengths=lengths, alpha=alpha, ploidy=ploidy,
            piecewise_factor=piecewise_factor,
            piecewise_fix_homo=piecewise_fix_homo, mixture_coefs=mixture_coefs)
        outdir_orient_init = os.path.join(outdir_orient, 'via_step1')
        infer(
            counts_raw=counts_raw, outdir=outdir_orient_init,
            lengths=lengths, ploidy=ploidy, alpha=alpha_, seed=seed,
            normalize=normalize, filter_threshold=filter_threshold,
            alpha_init=alpha_init, max_alpha_loop=max_alpha_loop,
            beta=beta_, multiscale_rounds=1,
            use_multiscale_variance=use_multiscale_variance,
            init=struct_fullres_genome_reoriented, max_iter=0, factr=factr,
            pgtol=pgtol, alpha_factr=alpha_factr, bcc_lambda=bcc_lambda,
            hsc_lambda=hsc_lambda, hsc_r=hsc_r, mhs_lambda=mhs_lambda,
            mhs_k=mhs_k, callback_function=callback_function,
            callback_freq={'print': 0, 'history': callback_freq['history'],
                           'save': 0},
            alpha_true=alpha_true, struct_true=struct_true,
            input_weight=input_weight, exclude_zeros=exclude_zeros,
            null=null, mixture_coefs=mixture_coefs, verbose=False)

        # Optionally rotate & translate previously inferred chromosomes
        if piecewise_opt_orient:
            reorienter = ChromReorienter(
                lengths=lengths, ploidy=ploidy,
                struct_init=struct_fullres_genome_init, translate=True,
                rotate=True, fix_homo=piecewise_fix_homo,
                mixture_coefs=mixture_coefs)

            if piecewise_fix_homo:
                hsc_lambda_3 = 0.
                mhs_lambda_3 = 0.
            else:
                hsc_lambda_3 = hsc_lambda
                mhs_lambda_3 = mhs_lambda

            if piecewise_step3_multiscale:
                multiscale_rounds_3 = multiscale_rounds
            else:
                multiscale_rounds_3 = 1

            struct_, infer_var = infer(
                counts_raw=counts_raw, outdir=outdir_orient,
                lengths=lengths, ploidy=ploidy, alpha=alpha_, seed=seed,
                normalize=normalize, filter_threshold=filter_threshold,
                alpha_init=alpha_init, max_alpha_loop=max_alpha_loop,
                beta=beta_, multiscale_rounds=multiscale_rounds_3,
                use_multiscale_variance=use_multiscale_variance,
                init=reorient_init, max_iter=max_iter, factr=factr,
                pgtol=pgtol, alpha_factr=alpha_factr, bcc_lambda=0.,
                hsc_lambda=hsc_lambda_3, hsc_r=hsc_r, mhs_lambda=mhs_lambda_3,
                mhs_k=mhs_k, excluded_counts='intra',
                fullres_torm=fullres_torm_for_multiscale,
                struct_draft_fullres=struct_draft_fullres,
                callback_function=callback_function,
                callback_freq=callback_freq, reorienter=reorienter,
                alpha_true=alpha_true, struct_true=struct_true,
                input_weight=input_weight, exclude_zeros=exclude_zeros,
                null=null, mixture_coefs=mixture_coefs, verbose=verbose)
            if not infer_var['converged']:
                return struct_, infer_var

    # Infer once more, letting all bead positions vary
    if 4 in piecewise_step:
        _print_code_header(
            ['PIECEWISE WHOLE GENOME: STEP 4',
                'Final whole-genome inference, allow all beads to vary'],
            max_length=80, blank_lines=2)

        if piecewise_opt_orient:
            X_all_chrom_oriented = np.loadtxt(os.path.join(
                outdir_orient, 'struct_inferred.%03d.coords' % seed))
        else:
            X_all_chrom_oriented = np.loadtxt(os.path.join(
                outdir_orient, 'struct_orient_via_lowres.%03d.coords' % seed))

        struct_, infer_var = infer(
            counts_raw=counts_raw, outdir=outdir, lengths=lengths,
            ploidy=ploidy, alpha=alpha_, seed=seed, normalize=normalize,
            filter_threshold=filter_threshold, alpha_init=alpha_init,
            max_alpha_loop=max_alpha_loop, beta=beta_,
            init=X_all_chrom_oriented, max_iter=max_iter, factr=factr,
            pgtol=pgtol, alpha_factr=alpha_factr, bcc_lambda=bcc_lambda,
            hsc_lambda=hsc_lambda, hsc_r=hsc_r,
            mhs_lambda=mhs_lambda, mhs_k=mhs_k,
            callback_function=callback_function, callback_freq=callback_freq,
            alpha_true=alpha_true, struct_true=struct_true,
            input_weight=input_weight, exclude_zeros=exclude_zeros,
            null=null, mixture_coefs=mixture_coefs, verbose=verbose)

    return struct_, infer_var
