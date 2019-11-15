import numpy as np
import autograd.numpy as ag_np
from .multiscale_optimization import decrease_struct_res, decrease_lengths_res
from .utils import find_beads_to_remove


class Constraints(object):
    """Compute objective constraints.
    """

    def __init__(self, counts, lengths, ploidy, multiscale_factor=1,
                 constraint_lambdas=None, constraint_params=None):

        self.lengths = lengths
        self.lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)
        self.ploidy = ploidy
        self.multiscale_factor = multiscale_factor
        if constraint_lambdas is None:
            self.lambdas = {}
        else:
            self.lambdas = constraint_lambdas
        if constraint_params is None:
            self.params = {}
        else:
            self.params = constraint_params
        torm = find_beads_to_remove(
            counts=counts, nbeads=self.lengths_lowres.sum() * ploidy)
        self.torm_3d = np.repeat(torm.reshape(-1, 1), 3, axis=1)

        self.row, self.col = constraint_dis_indices(
            counts=counts, n=self.lengths_lowres.sum(),
            lengths=self.lengths_lowres, ploidy=ploidy)
        # Calculating distances for neighbors, which are on the off diagonal
        # line - i & j where j = i + 1
        rows_adj = ag_np.unique(self.row)
        rows_adj = rows_adj[ag_np.isin(rows_adj + 1, self.col)]
        # Remove if "neighbor" beads are actually on different chromosomes or
        # homologs
        self.rows_adj = rows_adj[ag_np.digitize(rows_adj, np.tile(
            lengths, ploidy).cumsum()) == ag_np.digitize(
            rows_adj + 1, np.tile(lengths, ploidy).cumsum())]
        self.cols_adj = self.rows_adj + 1
        self.check()

    def check(self, verbose=True):
        """Check constraints object.
        """

        # Set defaults
        lagrange_mult_defaults = {'bcc': 0., 'struct': 0., 'hsc': 0.}
        lagrange_mult_all = lagrange_mult_defaults
        if self.lambdas is not None:
            for k, v in self.lambdas.items():
                if k not in lagrange_mult_all:
                    raise ValueError(
                        "constraint_lambdas key not recognized - %s" % k)
                elif v is not None:
                    lagrange_mult_all[k] = float(v)
        self.lambdas = lagrange_mult_all

        params_defaults = {'struct': None, 'hsc': None}
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
            if v != lagrange_mult_defaults[k]:
                if v < 0:
                    raise ValueError("Lagrange multipliers must be >= 0."
                                     " constraint_lambdas[%s] is %g" % (k, v))
                if self.params[k] is None:
                    raise ValueError("Lagrange multiplier for %s is supplied,"
                                     " but constraint is not" % k)
            elif k in self.params and self.params[k] != params_defaults[k]:
                raise ValueError("Constraint for %s is supplied, but lagrange"
                                 " multiplier is 0" % k)

        if 'hsc' in self.lambdas and self.lambdas['hsc'] and self.ploidy == 1:
            raise ValueError("Constraint to separate homologs can not be"
                             " applied to haploid genome.")

        if verbose and sum(self.lambdas.values()) != 0:
            lagrange_mult_to_print = {k: 'lambda = %.2g' %
                                      v for k, v in self.lambdas.items() if v != 0}
            if 'bcc' in lagrange_mult_to_print:
                print("CONSTRAINTS: bead chain connectivity %s" %
                      lagrange_mult_to_print.pop('bcc'), flush=True)
            if 'hsc' in lagrange_mult_to_print:
                to_print = self.params['hsc']
                if self.params['hsc'] is None:
                    to_print = 'inferred'
                elif isinstance(self.params['hsc'], np.ndarray):
                    to_print = ' '.join(map(str, self.params['hsc'].round(3)))
                else:
                    to_print = self.params['hsc'].round(3)
                print("CONSTRAINTS: homolog-separating %s,    r = %s" %
                      (lagrange_mult_to_print.pop('hsc'), to_print), flush=True)
            if len(lagrange_mult_to_print) > 0:
                to_print = [str(k) + ' ' + str(
                    self.params[k]) + ': %.1g'
                    % v for k, v in lagrange_mult_to_print.items() if v != 0]
                print("CONSTRAINTS:  %s" % (',  '.join(to_print)), flush=True)

    def apply(self, structures, mixture_coefs=None):
        """Apply constraints using via given structure.
        """

        if len(self.lambdas) == 0 or sum(self.lambdas.values()) == 0:
            return {}

        if mixture_coefs is None:
            mixture_coefs = [1.]
        if not isinstance(structures, list):
            structures = [structures]
        if len(structures) != len(mixture_coefs):
            raise ValueError(
                "The number of structures (%d) and of mixture coefficents (%d)"
                " should be identical." % (len(structures), len(mixture_coefs)))

        obj = {k: ag_np.float64(0.) for k, v in self.lambdas.items() if v != 0}

        for struct, gamma in zip(structures, mixture_coefs):
            if 'bcc' in self.lambdas and self.lambdas['bcc']:
                neighbor_dis = ag_np.sqrt((ag_np.square(
                    struct[self.rows_adj] - struct[self.cols_adj])).sum(axis=1))
                n_edges = neighbor_dis.shape[0]
                obj['bcc'] = obj['bcc'] + gamma * self.lambdas['bcc'] * \
                    (n_edges * ag_np.square(neighbor_dis).sum() / ag_np.square(
                        neighbor_dis.sum()) - 1.)
            if 'hsc' in self.lambdas and self.lambdas['hsc']:
                nbeads_per_homo = self.lengths_lowres.sum()
                struct_masked = ag_np.where(self.torm_3d, np.nan, struct)
                homo1 = struct_masked[:nbeads_per_homo, :]
                homo2 = struct_masked[nbeads_per_homo:, :]
                begin = end = 0
                for i in range(len(self.lengths_lowres)):
                    end = end + self.lengths_lowres[i]
                    homo_dis = ag_np.sqrt(ag_np.square(np.nanmean(
                        homo1[begin:end, :], axis=0) - np.nanmean(
                        homo2[begin:end, :], axis=0)).sum())
                    obj['hsc'] = obj['hsc'] + gamma * self.lambdas['hsc'] * \
                        ag_np.square(ag_np.array([float(
                            self.params['hsc'][i]) - homo_dis, 0]).max())
                    begin = end
            if 'struct' in self.lambdas and self.lambdas['struct']:
                struct_myres = decrease_struct_res(
                    struct,
                    multiscale_factor=struct.shape[
                        0] / self.params['struct'].shape[0],
                    lengths_prev=self.lengths).reshape(-1, 3)
                obj['struct'] = obj['struct'] + gamma * \
                    self.lambdas['struct'] * ag_np.sqrt(np.nanmean(ag_np.square(
                        self.params['struct'] - struct_myres))) / (
                        self.params['struct'].max() - self.params['struct'].min())

        # Check constraints objective
        for k, v in obj.items():
            if ag_np.isnan(v):
                raise ValueError("Constraint %s is nan" % k)
            elif ag_np.isinf(v):
                raise ValueError("Constraint %s is infinite" % k)

        return {'obj_' + k: v for k, v in obj.items()}


def constraint_dis_indices(counts, n, lengths, ploidy, mask=None,
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
            rows_adj = np.unique(rows)
            rows_adj = rows_adj[np.isin(rows_adj + 1, cols)]
            # Remove if "adjacent" beads are actually on different chromosomes
            # or homologs
            rows_adj = rows_adj[np.digitize(rows_adj, np.tile(
                lengths, ploidy).cumsum()) == np.digitize(
                rows_adj + 1, np.tile(lengths, ploidy).cumsum())]
            cols_adj = rows_adj + 1

    if mask is not None:
        rows[~mask] = 0
        cols[~mask] = 0
        if adjacent_beads_only:
            include = (np.isin(rows, rows_adj) & np.isin(cols, cols_adj))
            rows = rows[include]
            cols = cols[include]

    return rows, cols


def inter_homolog_dis(struct, lengths):
    """Computes distance between homologs for a normal diploid structure.
    """

    struct = struct.copy().reshape(-1, 3)

    nbeads_per_homo = int(struct.shape[0] / 2)
    homo1 = struct[:nbeads_per_homo, :]
    homo2 = struct[nbeads_per_homo:, :]

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


def inter_homolog_dis_via_simple_diploid(struct, lengths):
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

    return np.full_like(lengths, np.mean(homo_dis))


def distance_between_homologs(structures, lengths, ploidy, mixture_coefs=None,
                              simple_diploid=False):
    """Computes distance between homologs for a given structure.
    """

    from .utils import format_structures

    structures = format_structures(
        structures=structures, lengths=lengths, ploidy=ploidy,
        mixture_coefs=mixture_coefs)

    homo_dis = []
    for struct in structures:
        if simple_diploid:
            homo_dis.append(inter_homolog_dis_via_simple_diploid(
                struct=struct, lengths=lengths))
        else:
            homo_dis.append(inter_homolog_dis(struct=struct, lengths=lengths))

    return np.mean(homo_dis, axis=0)
