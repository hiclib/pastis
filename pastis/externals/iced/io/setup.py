# License: BSD Style.
import os
from os.path import join

import numpy


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config = Configuration('io', parent_package, top_path)

    config.add_extension(
        'fastio_',
        libraries=libraries,
        sources=['fastio_.c'],
        include_dirs=[join('..', 'src', 'cblas'),
                      numpy.get_include()])
    config.add_extension(
        'read',
        libraries=libraries,
        sources=['read.c'],
        include_dirs=[join('..', 'src', 'cblas'),
                      numpy.get_include()])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
