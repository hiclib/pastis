import sys

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

import numpy as np
import autograd.numpy as ag_np
from autograd.builtins import SequenceBox
from .multiscale_optimization import decrease_lengths_res
from .multiscale_optimization import _count_fullres_per_lowres_bead
from .utils_poisson import find_beads_to_remove, _inter_counts


class Constraints(object):
    """Compute objective constraints.

    Prepares constraints and computes the negative log likelhood of each
    constraint.

    Parameters
    ----------
    counts : list of CountsMatrix subclass instances
        Preprocessed counts data.
    lengths : array_like of int
        Number of beads per homolog of each chromosome.
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    multiscale_factor : int, optional
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 indicates full resolution.
    constraint_lambdas : dict, optional
        Lambdas for each constraint. Keys should match `constraint_params`
        when applicable.
    constraint_params : dict, optional
        Any parameters used for the calculation of each constraint. Keys should
        be in `constraint_lambdas`.

    Attributes
    ----------
    lambdas : dict
        Lambdas for each constraint.
    params : dict
        Any parameters used for the calculation of each constraint.
    row : array of int
        Rows of the distance matrix to be used in calculation of constraints.
    col : array of int
        Columns of the distance matrix to be used in calculation of constraints.
    row_adj : array of int
        Rows of the distance matrix indicating adjacent beads, to be used in
        calculation of the bead-chain-connectivity constraint.
    col_adj : array of int
        Columns of the distance matrix indicating adjacent beads, to be used in
        calculation of the bead-chain-connectivity constraint.
    lengths : array of int
        Number of beads per homolog of each chromosome in the full-resolution
        data.
    lengths_lowres : array of int
        Number of beads per homolog of each chromosome in the data at the
        current resolution (defined by `multiscale_factor`).
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    multiscale_factor : int
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 indicates full resolution.

    """

    def __init__(self, counts, lengths, ploidy, multiscale_factor=1,
                 constraint_lambdas=None, constraint_params=None, verbose=True):

        self.lengths = np.array(lengths)
        self.lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)
        self.ploidy = ploidy
        self.multiscale_factor = multiscale_factor
        if constraint_lambdas is None:
            self.lambdas = {}
        elif isinstance(constraint_lambdas, dict):
            self.lambdas = constraint_lambdas
        else:
            raise ValueError("Constraint lambdas must be inputted as dict.")
        if constraint_params is None:
            self.params = {}
        elif isinstance(constraint_params, dict):
            for hsc in ("hsc", "mhs"):
                if hsc in constraint_params and constraint_params[hsc] is not None:
                    constraint_params[hsc] = np.array(
                        constraint_params[hsc]).flatten().astype(float)
            self.params = constraint_params
        else:
            raise ValueError("Constraint params must be inputted as dict.")

        self.check(verbose=verbose)

        self.bead_weights = None
        if self.lambdas["hsc"] or self.lambdas["mhs"]:
            n = self.lengths_lowres.sum()
            torm = find_beads_to_remove(
                counts=counts, nbeads=self.lengths_lowres.sum() * ploidy)
            if multiscale_factor != 1:
                highres_per_lowres_bead = np.max(
                    [c.highres_per_lowres_bead for c in counts], axis=0)
                bead_weights = highres_per_lowres_bead / multiscale_factor
            else:
                bead_weights = np.ones((self.lengths_lowres.sum() * ploidy,))
            bead_weights[torm] = 0.
            begin = end = 0
            for i in range(len(self.lengths_lowres)):
                end = end + self.lengths_lowres[i]
                bead_weights[:n][begin:end] /= np.sum(
                    bead_weights[:n][begin:end])
                bead_weights[n:][begin:end] /= np.sum(
                    bead_weights[n:][begin:end])
                begin = end
            self.bead_weights = np.repeat(
                bead_weights.reshape(-1, 1), 3, axis=1)

        self.subtracted = None
        if self.lambdas["mhs"]:
            lambda_intensity = np.ones((self.lengths.shape[0],))
            self.subtracted = (lambda_intensity.sum() - (
                1 * np.log(lambda_intensity)).sum())

        self.row = self.col = None
        self.row_adj = self.col_adj = None
        if self.lambdas["bcc"]:
            self.row, self.col = _constraint_dis_indices(
                counts=counts, n=self.lengths_lowres.sum(),
                lengths=self.lengths_lowres, ploidy=ploidy)
            # Calculating distances for neighbors, which are on the off diagonal
            # line - i & j where j = i + 1
            row_adj = np.unique(self.row).astype(int)
            row_adj = row_adj[np.isin(row_adj + 1, self.col)]
            # Remove if "neighbor" beads are actually on different chromosomes or
            # homologs
            self.row_adj = row_adj[np.digitize(row_adj, np.tile(
                lengths, ploidy).cumsum()) == np.digitize(
                row_adj + 1, np.tile(lengths, ploidy).cumsum())]
            self.col_adj = self.row_adj + 1

    def check(self, verbose=True):
        """Check constraints object.

        Check that lambdas are greater than zero, and that necessary parameters
        are supplied. Optionally print summary of constraints.

        Parameters
        ----------
        verbose : bool
            Verbosity.
        """

        # Set defaults
        lambda_defaults = {"bcc": 0., "hsc": 0., "mhs": 0.}
        lambda_all = lambda_defaults
        if self.lambdas is not None:
            for k, v in self.lambdas.items():
                if k not in lambda_all:
                    raise ValueError(
                        "constraint_lambdas key not recognized - %s" % k)
                elif v is not None:
                    lambda_all[k] = float(v)
        self.lambdas = lambda_all

        params_defaults = {"hsc": None, "mhs": None}
        params_all = params_defaults
        if self.params is not None:
            for k, v in self.params.items():
                if k not in params_all:
                    raise ValueError('params key not recognized - %s' % k)
                elif v is not None:
                    if isinstance(v, int):
                        v = float(v)
                    params_all[k] = v
        self.params = params_all

        # Check constraints
        for k, v in self.lambdas.items():
            if v != lambda_defaults[k]:
                if v < 0:
                    raise ValueError("Lambdas must be >= 0. Lambda for"
                                     " %s is %g" % (k, v))
                if k in self.params and self.params[k] is None:
                    raise ValueError("Lambda for %s is supplied,"
                                     " but constraint is not" % k)
            elif k in self.params and not np.array_equal(self.params[k],
                                                         params_defaults[k]):
                print(self.params[k], type(self.params[k]))
                raise ValueError("Constraint for %s is supplied, but lambda is"
                                 " 0" % k)

        if (self.lambdas["hsc"] or self.lambdas["mhs"]) and self.ploidy == 1:
            raise ValueError("Homolog-separating constraint can not be"
                             " applied to haploid genome.")

        # Print constraints
        constraint_names = {"bcc": "bead chain connectivity",
                            "hsc": "homolog-separating",
                            "mhs": "multiscale homolog-separating"}
        lambda_to_print = {k: v for k, v in self.lambdas.items() if v != 0}
        if verbose and len(lambda_to_print) > 0:
            for constraint, lambda_val in lambda_to_print.items():
                print("CONSTRAINT: %s lambda = %.2g" % (
                    constraint_names[constraint], lambda_val), flush=True)
                if constraint in self.params and constraint in ("hsc", "mhs"):
                    if self.params[constraint] is None:
                        print("            param = inferred", flush=True)
                    elif isinstance(self.params[constraint], np.ndarray):
                        label = "            param = "
                        print(label + np.array2string(
                            self.params[constraint],
                            formatter={'float_kind': lambda x: "%.3g" % x},
                            prefix=" " * len(label), separator=", "))
                    elif isinstance(self.params[constraint], float):
                        print("            param = %.3g" % self.params[constraint],
                              flush=True)
                    else:
                        print("            %s" % self.params[constraint],
                              flush=True)

    def apply(self, structures, alpha=None, inferring_alpha=False,
              mixture_coefs=None):
        """Apply constraints using given structure(s).

        Compute negative log likelhood for each constraint using the given
        structure.

        Parameters
        ----------
        structures : array or autograd SequenceBox or list of structures
            3D chromatin structure(s) for which to compute the constraint.
        alpha : float, optional
            Biophysical parameter of the transfer function used in converting
            counts to wish distances. If alpha is not specified, it will be
            inferred.
        inferring_alpha : bool, optional
            A value of "True" indicates that the current optimization aims to
            infer alpha, rather than the structure.

        Returns
        -------
        dict
            Dictionary of constraint names and negative log likelihoods.

        """

        if len(self.lambdas) == 0 or sum(self.lambdas.values()) == 0:
            return {}

        if mixture_coefs is None:
            mixture_coefs = [1.]
        if not (isinstance(structures, list) or isinstance(structures, SequenceBox)):
            structures = [structures]
        if len(structures) != len(mixture_coefs):
            raise ValueError(
                "The number of structures (%d) and of mixture coefficents (%d)"
                " should be identical." % (len(structures), len(mixture_coefs)))

        obj = {k: 0. for k, v in self.lambdas.items() if v != 0}

        if self.lambdas["bcc"] and not inferring_alpha:
            for struct, gamma in zip(structures, mixture_coefs):
                neighbor_dis = ag_np.sqrt((ag_np.square(
                    struct[self.row_adj] - struct[self.col_adj])).sum(axis=1))
                n_edges = neighbor_dis.shape[0]
                obj['bcc'] = obj['bcc'] + gamma * self.lambdas['bcc'] * \
                    (n_edges * ag_np.square(neighbor_dis).sum() / ag_np.square(
                        neighbor_dis.sum()) - 1.)
        if self.lambdas["hsc"] and not inferring_alpha:
            for struct, gamma in zip(structures, mixture_coefs):
                homo_sep = self._homolog_separation(struct)
                hsc_diff = 0.
                for i in range(len(self.lengths_lowres)):
                    hsc_diff = hsc_diff + ag_np.square(
                        ag_np.max([self.params["hsc"][i] - homo_sep[i], 0]))
                obj["hsc"] = obj["hsc"] + gamma * self.lambdas["hsc"] * hsc_diff
        if self.lambdas["mhs"]:
            if alpha is None:
                raise ValueError("Must input alpha for multiscale-based homolog"
                                 " separating constraint.")
            lambda_intensity = ag_np.zeros(self.lengths_lowres.shape[0])
            for struct, gamma in zip(structures, mixture_coefs):
                homo_sep = self._homolog_separation(struct)
                lambda_intensity = lambda_intensity + gamma * homo_sep
            lambda_intensity = lambda_intensity / (self.params["mhs"] ** (1 / alpha))
            poisson_mhs = lambda_intensity.sum() - \
                ag_np.log(lambda_intensity).sum()
            obj["mhs"] = self.lambdas["mhs"] * (poisson_mhs - self.subtracted)

        # Check constraints objective
        for k, v in obj.items():
            if ag_np.isnan(v):
                raise ValueError("Constraint %s is nan" % k)
            elif ag_np.isinf(v):
                raise ValueError("Constraint %s is infinite" % k)

        return {'obj_' + k: v for k, v in obj.items()}

    def _homolog_separation(self, struct):
        """Compute distance between homolog centers of mass per chromosome.
        """

        struct_bw = struct * self.bead_weights
        n = self.lengths_lowres.sum()

        homo_sep = []
        begin = end = 0
        for l in self.lengths_lowres:
            end = end + l
            chrom1_mean = ag_np.sum(struct_bw[begin:end], axis=0)
            chrom2_mean = ag_np.sum(struct_bw[(n + begin):(n + end)], axis=0)
            homo_sep.append(ag_np.sqrt(ag_np.sum(ag_np.square(
                chrom1_mean - chrom2_mean))))
            begin = end

        return ag_np.array(homo_sep)


def _mean_interhomolog_counts(counts, lengths, bias=None):
    """Determine or estimate the mean interhomolog counts, divided by beta.
    """

    from .counts import ambiguate_counts

    n = lengths.sum()
    torm = find_beads_to_remove(counts=counts, nbeads=n * 2)
    beads_per_homolog = _count_fullres_per_lowres_bead(
        multiscale_factor=lengths.max(), lengths=lengths, ploidy=2,
        fullres_torm=torm)
    bins_per_interhomolog = beads_per_homolog[:lengths.shape[0]] * \
        beads_per_homolog[lengths.shape[0]:]

    counts_non0 = [c for c in counts if c.sum() != 0]
    ua_index = [i for i in range(len(
        counts_non0)) if counts_non0[i].name == "ua"]
    if len(ua_index) != 0:
        if len(ua_index) > 1:
            raise ValueError(
                "Only input one matrix of unambiguous counts. Please pool "
                "unambiguos counts before inputting.")
        ua_counts = counts_non0[ua_index[0]]
        mhs_beta = ua_counts.beta
        mhs_counts = ua_counts.toarray().astype(float)
        if bias is not None:
            ua_bias = np.tile(bias, 2).reshape(-1, 1)
            mhs_counts /= (ua_bias * ua_bias.T)
        mhs_counts_interhomo = mhs_counts[:n, n:]
        mhs_counts_interhomo[torm[:n], :] = np.nan
        mhs_counts_interhomo[:, torm[n:]] = np.nan
        mean_interhomo_counts = []
        begin = end = 0
        #print(bins_per_interhomolog)
        for l in lengths:
            end += l
            mean_interhomo_counts.append(np.nanmean(
                mhs_counts_interhomo[begin:end, begin:end]))
            #print((~np.isnan(mhs_counts_inter[begin:end, begin:end])).sum())
            begin = end
    else:
        if lengths.shape[0] == 1:
            raise ValueError(
                "Estimating mean inter-homolog counts from ambiguous"
                " inter-chromosomal counts requires data for  more than one"
                " chromosome.")
        mhs_beta = sum([c.beta for c in counts_non0])
        mhs_counts = ambiguate_counts(
            counts=counts_non0, lengths=lengths, ploidy=2,
            exclude_zeros=True).toarray().astype(float)
        if bias is not None:
            ambig_bias = bias.reshape(-1, 1)
            mhs_counts /= (ambig_bias * ambig_bias.T)
        mhs_counts_inter = _inter_counts(
            mhs_counts, lengths=lengths, ploidy=2, exclude_zeros=False)
        mean_interhomo_counts = [np.nanmean(mhs_counts_inter) / 4]

    mean_interhomo_counts = np.array(mean_interhomo_counts)
    return mean_interhomo_counts / mhs_beta


def _constraint_dis_indices(counts, n, lengths, ploidy, mask=None,
                            adjacent_beads_only=False):
    """Return distance matrix indices associated with any counts matrix data.
    """

    n = int(n)

    if isinstance(counts, list) and len(counts) == 1:
        counts = counts[0]
    if not isinstance(counts, list):
        rows = counts.row3d
        cols = counts.col3d
    else:
        rows = []
        cols = []
        for counts_maps in counts:
            rows.append(counts_maps.row3d)
            cols.append(counts_maps.col3d)
        rows, cols = np.split(np.unique(np.concatenate([np.atleast_2d(
            np.concatenate(rows)), np.atleast_2d(np.concatenate(cols))],
            axis=0), axis=1), 2, axis=0)
        rows = rows.flatten()
        cols = cols.flatten()

    if adjacent_beads_only:
        if mask is None:
            # Calculating distances for adjacent beads, which are on the off
            # diagonal line - i & j where j = i + 1
            rows = np.unique(rows)
            rows = rows[np.isin(rows + 1, cols)]
            # Remove if "adjacent" beads are actually on different chromosomes
            # or homologs
            rows = rows[np.digitize(rows, np.tile(
                lengths, ploidy).cumsum()) == np.digitize(
                rows + 1, np.tile(lengths, ploidy).cumsum())]
            cols = rows + 1
        else:
            # Calculating distances for adjacent beads, which are on the off
            # diagonal line - i & j where j = i + 1
            row_adj = np.unique(rows)
            row_adj = row_adj[np.isin(row_adj + 1, cols)]
            # Remove if "adjacent" beads are actually on different chromosomes
            # or homologs
            row_adj = row_adj[np.digitize(row_adj, np.tile(
                lengths, ploidy).cumsum()) == np.digitize(
                row_adj + 1, np.tile(lengths, ploidy).cumsum())]
            col_adj = row_adj + 1

    if mask is not None:
        rows[~mask] = 0
        cols[~mask] = 0
        if adjacent_beads_only:
            include = (np.isin(rows, row_adj) & np.isin(cols, col_adj))
            rows = rows[include]
            cols = cols[include]

    return rows, cols


def _inter_homolog_dis(struct, lengths):
    """Computes distance between homologs for a normal diploid structure.
    """

    struct = struct.copy().reshape(-1, 3)

    n = int(struct.shape[0] / 2)
    homo1 = struct[:n, :]
    homo2 = struct[n:, :]

    homo_dis = []
    begin = end = 0
    for l in lengths:
        end += l
        if np.isnan(homo1[begin:end, 0]).sum() == l or np.isnan(
                homo2[begin:end, 0]).sum() == l:
            homo_dis.append(np.nan)
        else:
            homo_dis.append(((np.nanmean(homo1[
                begin:end, :], axis=0) - np.nanmean(
                homo2[begin:end, :], axis=0)) ** 2).sum() ** 0.5)
        begin = end

    homo_dis = np.array(homo_dis)
    homo_dis[np.isnan(homo_dis)] = np.nanmean(homo_dis)

    return homo_dis


def _inter_homolog_dis_via_simple_diploid(struct, lengths):
    """Computes distance between chromosomes for a faux-haploid structure.
    """

    from sklearn.metrics import euclidean_distances

    struct = struct.copy().reshape(-1, 3)

    chrom_barycenters = []
    begin = end = 0
    for l in lengths:
        end += l
        if np.isnan(struct[begin:end, 0]).sum() < l:
            chrom_barycenters.append(
                np.nanmean(struct[begin:end, :], axis=0).reshape(1, 3))
        begin = end

    chrom_barycenters = np.concatenate(chrom_barycenters)

    homo_dis = euclidean_distances(chrom_barycenters)
    homo_dis[np.tril_indices(homo_dis.shape[0])] = np.nan

    return np.full(lengths.shape, np.nanmean(homo_dis))


def distance_between_homologs(structures, lengths, mixture_coefs=None,
                              simple_diploid=False):
    """Computes distances between homologs for a given structure.

    For diploid organisms, this computes the distance between homolog centers
    of mass for each chromosome.

    Parameters
    ----------
    structures : array of float or list of array of float
        3D chromatin structure(s) for which to assess inter-homolog distances.
    lengths : array_like of int
        Number of beads per homolog of each chromosome.
    simple_diploid: bool, optional
        For diploid organisms: whether the structure is an inferred "simple
        diploid" structure in which homologs are assumed to be identical and
        completely overlapping with one another.

    Returns
    -------
    array of float
        Distance between homologs per chromosome.

    """

    from .utils_poisson import _format_structures

    structures = _format_structures(
        structures=structures, lengths=lengths,
        ploidy=(1 if simple_diploid else 2),
        mixture_coefs=mixture_coefs)

    homo_dis = []
    for struct in structures:
        if simple_diploid:
            homo_dis.append(_inter_homolog_dis_via_simple_diploid(
                struct=struct, lengths=lengths))
        else:
            homo_dis.append(_inter_homolog_dis(struct=struct, lengths=lengths))

    return np.mean(homo_dis, axis=0)
