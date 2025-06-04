#! /usr/bin/env python

from pastis.algorithms import run_mds


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run MDS.')
    parser.add_argument('directory', type=str,
                        help='directory', default=None)
    args = parser.parse_args()

    if args.directory is not None and not "":
        run_mds(args.directory)


if __name__ == "__main__":
    main()
