#! /usr/bin/env python

from pastis.algorithms import run_pm2


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run PM2.')
    parser.add_argument('directory', metavar='N', type=str,
                        help='directory')
    args = parser.parse_args()
    run_pm2(args.directory)


if __name__ == "__main__":
    main()
