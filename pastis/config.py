import ConfigParser
import os


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

    nucleus_size : float, optional, default: None
        The size of the nucleus. If None, no constraints will be applied.
        New in 0.1

    adjacent_beads : float, optional, default: None
        The distances between adjacent beads. If None, no constraints will be
        applied.
        New in 0.1

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
    for key in options.iterkeys():
        try:
            options[key] = type(options[key])(
                config.get("all", key))
        except ConfigParser.NoOptionError:
            pass
    return options
