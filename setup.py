import os
import shutil
from numpy.distutils.core import setup
from distutils.command.clean import clean as Clean


DISTNAME = 'pastis'
DESCRIPTION = 'A set of algorithms for the 3D inference of the genome'
MAINTAINER = 'Nelle Varoquaux'
MAINTAINER_EMAIL = 'nelle.varoquaux@ensmp.fr'
VERSION = '0.2.0'
LICENSE = "New BSD"
URL = 'http://cbio.ensmp.fr/pastis'
DOWNLOAD_URL = 'https://github.com/hiclib/pastis/releases'


class CleanCommand(Clean):
    description = ("Remove build directories, and compiled file in the "
                   "source tree")

    def run(self):
        Clean.run(self)
        if os.path.exists('build'):
            shutil.rmtree('build')
        for dirpath, dirnames, filenames in os.walk('pastis'):
            for filename in filenames:
                if (filename.endswith('.so') or filename.endswith('.pyd')
                   or filename.endswith('.dll')
                   or filename.endswith('.pyc')):
                    os.unlink(os.path.join(dirpath, filename))


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
          url=URL,
          download_url=DOWNLOAD_URL,
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
              'Operating System :: MacOS'],
          install_requires=["numpy", "scipy", "scikit-learn", "argparse"],
          cmdclass={'clean': CleanCommand})
