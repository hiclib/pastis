import numpy as np
import argparse
import os


def _save_params(outfile, **kwargs):
    """Save all parameters to file.
    """

    output = []
    for k, v in kwargs.items():
        if isinstance(v, float):
            output.append('%s\t%g' % (k, v))
        elif isinstance(v, list) or isinstance(v, np.ndarray):
            output.append('%s\t%s' % (k, ','.join(map(str, v))))
        elif isinstance(v, type(None)):
            output.append('%s\t' % k)
        else:
            output.append('%s\t%s' % (k, v))

    with open(outfile, 'w') as f:
        f.write('\n'.join(output) + '\n')


def _make_parser_class(parser):
    """Return argparse parser instance.
    """

    store_action_dict = vars(parser)['_option_string_actions']
    flags = {}
    const = {}
    for flag, object in store_action_dict.items():
        if not isinstance(object, argparse._HelpAction):
            flags[object.dest] = flag
            const[object.dest] = object.const

    class PastisArgParser(argparse.ArgumentParser):
        def convert_arg_line_to_args(self, arg_line):
            bool_states = {'1': True, 'yes': True, 'true': True, 'on': True,
                           '0': False, 'no': False, 'false': False,
                           'off': False}
            if len(arg_line.split('\t')) != 2:
                raise ValueError("Error reading arguments from file.")
            dest, val = arg_line.split('\t')
            if dest not in const:
                raise ValueError("Error reading arguments from file, argument"
                                 " '%s' not recognized" % dest)
            if const[dest] is None and val != '':
                return [flags[dest]] + val.split(',')
            elif isinstance(const[dest], bool) and bool_states[val.lower()] == const[dest]:
                return [flags[dest]]
            elif val == const[dest]:
                return [flags[dest]]

    return PastisArgParser


def _make_parser(parser_class=None, dev_options=None):
    """Make command-line argument parser.
    """

    if parser_class is None:
        parser_class = argparse.ArgumentParser
    parser = parser_class(description="Run PASTIS-PM1 or PASTIS-PM2.",
                          fromfile_prefix_chars='@', prog="PASTIS")
    parser.add_argument("--counts", nargs="+", type=str, required=True,
                        help="Counts data files in the hiclib format or as"
                             " numpy ndarrays.")
    parser.add_argument("--lengths", nargs="+", type=str, required=True,
                        help="Number of beads per homolog of each chromosome,"
                             " or hiclib .bed file with lengths data")
    parser.add_argument('--ploidy', type=int, required=True, choices=[1, 2],
                        help="Ploidy, 1 indicates haploid, 2 indicates diploid.")
    parser.add_argument("--outdir", type=str, default="",
                        help="Directory in which to save results.")
    parser.add_argument("--chromosomes", nargs="+", default=None, type=str,
                        help="Label for each chromosome in the data, or file"
                             " with chromosome labels (one label per line).")
    parser.add_argument("--chrom_subset", nargs="+", default=None, type=str,
                        help="Labels of chromosomes for which inference should"
                             " be performed.")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Biophysical parameter of the transfer function used"
                             " in converting counts to wish distances. If alpha is"
                             " not specified, it will be inferred.")
    parser.add_argument("--seed", default=0, type=int,
                        help="Random seed used when generating the starting point"
                             " in the optimization.")
    parser.add_argument('--dont-normalize', dest="normalize", default=True,
                        action='store_false',
                        help="Prevents ICE normalization on the counts prior to"
                             " optimization. Normalization is reccomended.")
    parser.add_argument('--filter_threshold', default=0.04, type=float,
                        help="Ratio of non-zero beads to be filtered out. Filtering"
                             " is reccomended.")
    parser.add_argument("--alpha_init", type=float, default=-3.,
                        help="For PM2, the initial value of alpha to use.")
    parser.add_argument("--max_alpha_loop", type=int, default=20,
                        help="For PM2, Number of times alpha and structure are"
                             " inferred.")
    parser.add_argument("--multiscale_rounds", default=1, type=int,
                        help="The number of resolutions at which a structure"
                             " should be inferred during multiscale optimization."
                             " Values of 1 or 0 disable multiscale"
                             " optimization.")

    # Optimization convergence
    parser.add_argument("--max_iter", default=10000000000, type=float,
                        help="Maximum number of iterations per optimization.")
    parser.add_argument("--factr", default=10000000., type=float,
                        help="factr for scipy's L-BFGS-B, alters convergence"
                             " criteria.")
    parser.add_argument("--pgtol", default=1e-05, type=float,
                        help="pgtol for scipy's L-BFGS-B, alters convergence"
                             " criteria.")
    parser.add_argument("--alpha_factr", default=1000000000000., type=float,
                        help="factr for convergence criteria of joint"
                             " alpha/structure inference.")

    # Constraints
    parser.add_argument("--bcc_lambda", type=float, default=0.,
                        help="Lambda of the bead chain connectivity constraint.")
    parser.add_argument("--hsc_lambda", type=float, default=0.,
                        help="For diploid organisms: lambda of the homolog-"
                             "separating constraint.")
    parser.add_argument("--hsc_r", default=None, type=float, nargs="+",
                        help="For diploid organisms: hyperparameter of the"
                             " homolog-separating constraint specificying the"
                             " expected distance between homolog centers of mass"
                             " for each chromosome. If not supplied, `hsc_r` will"
                             " be inferred from the counts data.")
    parser.add_argument("--hsc_min_beads", type=int, default=5,
                        help="For diploid organisms: number of beads in the"
                             " low-resolution structure from which `hsc_r` is"
                             " estimated.")

    # Callback
    parser.add_argument("--print_freq", type=int, default=100,
                        help="Frequency of iterations at which to print during"
                             " optimization.")
    parser.add_argument("--history_freq", type=int, default=100,
                        help="Frequency of iterations at which to log objective"
                             " value and other information during"
                             " optimization.")
    parser.add_argument("--save_freq", type=int, default=None,
                        help="Frequency of iterations at which to save the 3D"
                             " structure during optimization.")

    if dev_options is not None:
        parser = dev_options(parser)
    return parser


def _parse_pastis_args(dev_options=None):
    """Parse commpand line arguments, save parameters, generate structure.
    """

    parser = _make_parser(
        parser_class=_make_parser_class(_make_parser(dev_options=dev_options)),
        dev_options=dev_options)

    args = vars(parser.parse_args())

    _save_params(os.path.join(args['outdir'], 'params'), **args)

    return args
