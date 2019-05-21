try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser
import os
from distutils.util import strtobool

import pandas as pd
import numpy as np


def save_params(counts, infer_lengths, alpha, beta, ploidy, init, outdir, lagrange_mult, constraints, homo_init, input_weight, as_sparse, norm, filter_counts,
                infer_chrom, stepwise_genome=False, stepwise_genome__lowres_min_beads=5, stepwise_genome__fix_homo=True,
                lowres_genome_factor=None, multiscale_rounds=None, HSC_lowres_beads=5, modifications=None, null=False, filter_percentage=0.04, max_iter=1e40, in_2d=False):
    from topsy.inference.poisson_diploid import check_constraints

    params = pd.Series({'counts': ','.join(sorted([{1: 'ambig', 1.5: 'pa', 2: 'ua'}[sum(c.shape) / (infer_lengths.sum() * ploidy)] for c in counts])),
                        'ploidy': ploidy,
                        'normalize': norm,
                        'filter': (str(int(filter_percentage * 100)) + '%' if norm and filter_counts else None),
                        'init': init,
                        'alpha': alpha,
                        'beta': (None if beta is None else ','.join([str(b) for b in beta])),
                        'max_iter': max_iter,
                        'homo_init': homo_init,
                        'multiscale_rounds': multiscale_rounds,
                        'lengths': ",".join(map(str, infer_lengths)),
                        'chrom': ",".join(infer_chrom),
                        'input_weight': (None if input_weight is None else ','.join([str(x) for x in input_weight])),
                        'sparsity': ('sparse' if as_sparse else 'dense'),
                        'null_inference': ('- Poisson' if null else '+ Poisson')})

    if len(infer_lengths) > 1:
        params['stepwise_genome'] = stepwise_genome
        if stepwise_genome:
            params['stepwise_genome.lowres_min_beads'] = stepwise_genome__lowres_min_beads
            params['stepwise_genome.fix_homo'] = stepwise_genome__fix_homo

    lagrange_mult, constraints = check_constraints(lagrange_mult, constraints, verbose=False)
    for k, v in lagrange_mult.items():
        if v is not None and v != 0:
            params['constraint.%s' % k] = constraints[k]
            params['lambda.%s' % k] = v
            if k.lower() == 'homodis':
                params['homodis.lowres_beads'] = HSC_lowres_beads

    if 'in_2d':
        params['in_2d'] = True
    if modifications is not None and len(modifications) != 0:
        params['modifications'] = '.'.join(modifications)

    if multiscale_rounds is not None:
        if lowres_genome_factor is not None:
            raise ValueError("Must set either multiscale_rounds or lowres_genome_factor, not both")
        all_multiscale_factors = 2 ** np.flip(np.arange(multiscale_rounds), axis=0)
    elif lowres_genome_factor is not None:
        all_multiscale_factors = [lowres_genome_factor]
    else:
        raise ValueError("Must set multiscale_rounds or lowres_genome_factor, both are currently None")

    for multiscale_factor in all_multiscale_factors:
        current_outdir = outdir
        if multiscale_factor != 1:
            current_outdir = os.path.join(current_outdir, 'res-x-%d' % multiscale_factor)
        try:
            os.makedirs(current_outdir)
        except OSError:
            pass
        params['multiscale_factor'] = multiscale_factor
        params.to_csv(os.path.join(current_outdir, 'inference_params.txt'), sep='\t', header=False)
        params['multiscale_factor'] = None


def get_default_options():
    """
    Returns default options

    Here are all the options:

    output_name : structure.pdb, str
        The name of the PDB file to write. The name of the algorithm used will
        be appended to this filename. The coordinates of each beads will also
        be outputted in a text file `structure.pdb.txt`.
        For example, running the algorithm MDS with an output_name
        `structure.pdb` will yield the files `MDS.structure.pdb` and
        `MDS.structure.pdb.txt`

    counts : data/counts.npy, str
        The numpy ndarray of counts

    organism_structure : files/budding_yeast_structure
        The path to the filename containing the organism structure. This file
        should contain the lengths of the chromosomes of the organism.

    resolution : 10000, integer
        Resolution to which perform the optimization.

    chromosomes : "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16", str
        coma separated list of chromosomes onto which the optimization should
        be performed.

    binary_mds : "MDS_all"
        Path to the binary MDS_all

    binary_pm : "PM_all"
        Path to the binary PM_all

    alpha : -3., float
        Coefficient of the power law used in converting interaction counts to
        wish distances

    beta : 1., float
        Scaling factor of the structure.

    seed : 0, integer
        Random seed used when generating the starting point in the
        optimization.
    """
    options = {"output_name": "structure.pdb",
               "resolution": 10000,
               "input_name": "wish_distances.txt",
               "counts": "data/counts.npy",
               "chromosomes": "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16",
               "organism_structure": "files/budding_yeast_structure",
               "alpha": -3.,
               "beta": 1.,
               "logging_file": "MDS.log",
               "binary_mds": "MDS_all",
               "binary_pm": "PM_all",
               "nucleus_size": None,
               "adjacent_beads": None,
               "lengths": "",
               "verbose": 1,
               "normalize": False,
               "max_iter": 10000,
               "seed": 0,
               }

    return options


def parse(filename=None):
    """
    Parses a configuration file.

    Parameters
    ----------
    filename : str, optional, default: None
        If a filename is provided, reads the configuration file, and returns
        the options. If None, returns the default options.

    Returns
    -------
    options : dict
    """
    options = get_default_options()
    if filename is None:
        return options

    if not os.path.exists(filename):
        raise IOError("File %s doesn't existe" % filename)

    config = ConfigParser.ConfigParser()
    config.readfp(open(filename))
    for key in options.keys():
        try:
            if type(options[key]) == bool:
                options[key] = bool(strtobool(
                    config.get("all", key)))
            else:
                options[key] = type(options[key])(
                    config.get("all", key))
        except ConfigParser.NoOptionError:
            pass

    # Now process the options so that distances that are set to None are
    # negative
    if options["adjacent_beads"] is None:
        options["adjacent_beads"] = -1
    if options["nucleus_size"] is None:
        options["nucleus_size"] = -1

    return options
