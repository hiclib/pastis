#! /usr/bin/env python

from pastis.optimization.pastis_algorithms import pastis_poisson


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Run PASTIS-PM1 or PASTIS-PM2 on diploid organisms.",
        fromfile_prefix_chars='@', prog="PASTIS")
    parser.add_argument(
        "--counts", nargs="+", type=str, required=True,
        help="Counts data files in the hiclib format or as numpy ndarrays.")
    parser.add_argument(
        "--lengths", nargs="+", type=str, required=True,
        help="Number of beads per homolog of each chromosome.")
    parser.add_argument(
        '--ploidy', type=int, required=True, choices=[1, 2],
        help="Ploidy, 1 indicates haploid, 2 indicates diploid.")
    parser.add_argument(
        "--outdir", type=str, default="",
        help="Directory in which to save results")
    parser.add_argument(
        "--chromosomes", nargs="+", default=None, type=str,
        help="Label for each chromosome in the data.")
    parser.add_argument(
        "--chrom_subset", nargs="+", default=None, type=str,
        help="Chromosomes for which inference should be performed.")
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="Biophysical parameter of the transfer function used"
        " in converting counts to wish distances. If alpha is"
        " not specified, it will be inferred.")
    parser.add_argument(
        "--seed", default=0, type=int,
        help="Random seed used when generating the starting point in the"
        " optimization.")
    parser.add_argument(
        '--dont-normalize', dest="normalize", default=True, action='store_false',
        help="Prevents ICE normalization on the counts prior to"
        " optimization. Normalization is reccomended.")
    parser.add_argument(
        '--filter_threshold', default=0.04, type=float,
        help=("Ratio of non-zero beads to be filtered out. Filtering"
              " is recommended."))
    parser.add_argument(
        "--alpha_init", type=float, default=-3.,
        help="For PM2, the initial value of alpha to use.")
    parser.add_argument(
        "--max_alpha_loop", type=int, default=20,
        help="For PM2, Number of times alpha and structure are inferred.")
    parser.add_argument(
        "--beta", nargs="+", default=None, type=float,
        help="Scaling parameter that determines the size of the structure,"
        " relative to each counts matrix. There should be one beta per counts"
        " matrix. If None, the optimal beta will be estimated.")
    parser.add_argument(
        "--multiscale_rounds", default=1, type=int,
        help="The number of resolutions at which a structure should be inferred"
        " during multiscale optimization. Values of 1 or 0 disable multiscale"
        " optimization.")

    # Optimization convergence
    parser.add_argument(
        "--max_iter", default=30000, type=int,
        help="Maximum number of iterations per optimization.")
    parser.add_argument(
        "--factr", default=10000000.0, type=float,
        help="factr for scipy's L-BFGS-B, alters convergence criteria")
    parser.add_argument(
        "--pgtol", default=1e-05, type=float,
        help="pgtol for scipy's L-BFGS-B, alters convergence criteria")
    parser.add_argument(
        "--alpha_factr", default=1000000000000., type=float,
        help="factr for convergence criteria of joint alpha/structure"
        " inference")

    # Constraints
    parser.add_argument(
        "--bcc_lambda", type=float, default=0.,
        help="Lambda of the bead chain connectivity constraint")
    parser.add_argument(
        "--hsc_lambda", type=float, default=0.,
        help="For diploid organisms, lambda of the homolog-separating"
        "  constraint.")
    parser.add_argument(
        "--hsc_r", default=None, type=float, nargs="+",
        help="For diploid organisms, hyperparameter of the homolog-separating"
        " constraint specificying the expected distance between homolog centers"
        " of mass for each chromosome. If not supplied, `hsc_r` will be"
        " inferred from the counts data.")
    parser.add_argument(
        "--hsc_min_beads", type=int, default=5,
        help="For diploid organisms, number of beads in the low-resolution"
        " structure from which `hsc_r` is estimated.")
    parser.add_argument(
        "--mhs_lambda", type=float, default=0.,
        help="For diploid organisms: lambda of the multiscale-based"
        " homolog-separating constraint.")
    parser.add_argument(
        "--mhs_k", default=None, type=float, nargs="+",
        help="For diploid organisms: hyperparameter of the multiscale-based"
        " homolog-separating constraint specificying the expected mean"
        " inter-homolog count for each chromosome, scaled by beta and biases."
        " If not supplied, `mhs_k` will be estimated from the counts data.")

    # Hidden options
    parser.add_argument(
        '--piecewise', default=False, action='store_true',
        help=argparse.SUPPRESS)
    parser.add_argument(
        "--piecewise_step", nargs="+", default=None, type=int,
        help=argparse.SUPPRESS)
    parser.add_argument(
        "--piecewise_chrom", nargs="+", default=None, type=str,
        help=argparse.SUPPRESS)
    parser.add_argument(
        "--piecewise_min_beads", default=5, type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        '--piecewise_fix_homo', default=False, action='store_true',
        help=argparse.SUPPRESS)
    parser.add_argument(
        '--piecewise_dont_optimize_orientation', dest='piecewise_opt_orient',
        default=True, action='store_false', help=argparse.SUPPRESS)
    parser.add_argument(
        "--alpha_true", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--struct_true", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--init", default="mds", type=str, help=argparse.SUPPRESS)
    parser.add_argument(
        "--input_weight", default=None, nargs="+", type=float,
        help=argparse.SUPPRESS)
    parser.add_argument(
        '--no_multiscale_variance', dest='use_multiscale_variance',
        default=True, action='store_false', help=argparse.SUPPRESS)
    parser.add_argument(
        '--exclude_zeros', default=False, action='store_true',
        help=argparse.SUPPRESS)
    parser.add_argument(
        '--null', dest='null', default=False, action='store_true',
        help=argparse.SUPPRESS)

    pastis_poisson(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
