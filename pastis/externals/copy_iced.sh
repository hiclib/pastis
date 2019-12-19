#!/bin/sh
# Script to do a local install of joblib
rm -rf tmp 
PYTHON_VERSION=$(python -c 'import sys; print("{0[0]}.{0[1]}".format(sys.version_info))')
SITE_PACKAGES="$PWD/tmp/lib/python$PYTHON_VERSION/site-packages"

mkdir -p $SITE_PACKAGES
mkdir -p tmp/bin
export PYTHONPATH="$SITE_PACKAGES"
easy_install -Zeab tmp iced

cd tmp/joblib/
python setup.py install --prefix $OLDPWD/tmp
cd $OLDPWD
cp -r $SITE_PACKAGES/iced-*.egg/iced .
rm -rf tmp
# Needed to rewrite the doctests
# Note: BSD sed -i needs an argument unders OSX
# so first renaming to .bak and then deleting backup files
find iced -name "*.py" | xargs sed -i.bak "s/from iced/from pastis.externals.iced/"
find iced -name "*.bak" | xargs rm

# Remove the tests folders to speed-up test time for scikit-learn.
# joblib is already tested on its own CI infrastructure upstream.
#rm -r joblib/test

chmod -x iced/*.py
