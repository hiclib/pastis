from setuptools import setup, find_packages


DISTNAME = 'pastis'
DESCRIPTION = 'A set of algorithms for the 3D inference of the genome'
MAINTAINER = 'Nelle Varoquaux'
MAINTAINER_EMAIL = 'nelle.varoquaux@ensmp.fr'
VERSION = '0.5.0'
LICENSE = "New BSD"
URL = 'http://cbio.ensmp.fr/pastis'
DOWNLOAD_URL = 'https://github.com/hiclib/pastis/releases'

SCIPY_MIN_VERSION = '0.19.0'
NUMPY_MIN_VERSION = '1.16.0'

setup(
    name=DISTNAME,
    version=VERSION,
    author=MAINTAINER,
    author_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS'],
    scripts=["pastis/script/pastis-mds",
             "pastis/script/pastis-nmds",
             "pastis/script/pastis-pm1",
             "pastis/script/pastis-pm2",
             "pastis/script/pastis-poisson"],
    packages=find_packages(where="."),
    include_package_data=True,
    zip_safe=False,  # the package can run out of an .egg file
    extras_require={
        'alldeps': (
            'numpy >= {0}'.format(NUMPY_MIN_VERSION),
            'scipy >= {0}'.format(SCIPY_MIN_VERSION),
        ),
    },
)
