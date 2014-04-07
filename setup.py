import os
from numpy.distutils.core import setup

DISTNAME = 'pastis'
DESCRIPTION = ''
MAINTAINER = 'Nelle Varoquaux'
MAINTAINER_EMAIL = 'nelle.varoquaux@ensmp.fr'
VERSION = '0.0'


def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)

    config.add_subpackage('pastis')

    return config


if __name__ == "__main__":
    setup(configuration=configuration,
          name=DISTNAME,
          maintainer=MAINTAINER,
          include_package_data=True,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          version=VERSION,
          zip_safe=False,  # the package can run out of an .egg file
          scripts=["pastis/script/pastis-mds",
                   'pastis/script/pastis-nmds', "pastis/script/pastis-pm1",
                   "pastis/script/pastis-pm2"],
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: C',
              'Programming Language :: C++',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS'
             ]
    )
