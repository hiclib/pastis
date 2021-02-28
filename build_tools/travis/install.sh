#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# License: 3-clause BSD

# Travis clone hiclib/iced repository in to a local repository.
# We use a cached directory with three iced repositories (one for each
# matrix entry) from which we pull from local Travis repository. This allows
# us to keep build artefact for gcc + cython, and gain time

set -e
set -v
pip install --upgrade pip pytest pytest-cov

if [[ $NUMPY_VERSION != "*" ]]; then
    pip install --upgrade \
        numpy==$NUMPY_VERSION
else
    pip install numpy --upgrade
fi

if [[ $SCIPY_VERSION != "*" ]]; then
    pip install --upgrade scipy==$SCIPY_VERSION
else
    pip install --upgrade scipy
fi


if [[ $PANDAS_VERSION != "*" ]]; then
    pip install --upgrade pandas==$PANDAS_VERSION
else
    pip install --upgrade pandas
fi

if [[ $SKLEARN_VERSION != "*" ]]; then
    pip install --upgrade scikit-learn==$SKLEARN_VERSION
else
    pip install --upgrade scikit-learn
fi


if [[ $AUTOGRAD_VERSION != "*" ]]; then
    pip install --upgrade autograd==$AUTOGRAD_VERSION
else
    pip install --upgrade autograd
fi

pip install rmsd
pip install cython


if [[ "$COVERAGE" == "true" ]]; then
    pip install pytest coverage==4.5.4 coveralls
fi

if [ ! -d "$CACHED_BUILD_DIR" ]; then
    mkdir -p $CACHED_BUILD_DIR
fi

rsync -av --exclude '.git/' --exclude='testvenv/' \
      $TRAVIS_BUILD_DIR $CACHED_BUILD_DIR

cd $CACHED_BUILD_DIR/pastis

# Build pastis in the install.sh script to collapse the verbose
# build output in the travis output when it succeeds.
python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"

make cython
python setup.py build_src
python setup.py develop

if [[ "$RUN_FLAKE8" == "true" ]]; then
    # flake8 version is temporarily set to 2.5.1 because the next
    # version available on conda (3.3.0) has a bug that checks non
    # python files and cause non meaningful flake8 errors
    pip install flake8==2.5.1
fi


if [[ "$BUILD_DOC" == "true" ]]; then
    # Install dependencies for building documentation
    pip install sphinx sphinx_gallery numpydoc pillow matplotlib
fi


pip install iced
