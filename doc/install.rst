================================================================================
Installation
================================================================================

Dependencies
============

- python (>= 2.7)
- setuptools
- numpy (>= 1.3)
- scipy (>= 0.7)
- scikit-learn (>= 0.13)
- iced

Additional dependencies for new features (diploid inference
multiscale optimization, etc):
- python (>= 3.6)
- autograd (>= 1.3)

Most of these dependencies can be installed at once using conda:
`http://conda.pydata.org/miniconda.html <http://conda.pydata.org/miniconda.html>`_

Once conda is installed, just type the following::

  conda install numpy scipy scikit-learn pandas

Or, to include the new features::

    conda install numpy scipy scikit-learn pandas autograd

`iced` can be installed using pip::

  pip install --user iced

Install PASTIS
==============

This package uses distutils, which is the default way of installing
python modules.

To install in your home directory, use::

    python setup.py install --user

or using pip::

    pip install --user pastis

To install for all users on Unix/Linux::

    python setup.py build
    sudo python setup.py install

or using pip::

  pip install pastis

This will install a python package ``pastis``, and five programs:
``pastis-mds``, ``pastis-nmds``, ``pastis-pm1``, ``pastis-pm2``, and
``pastis-poisson``. Calling any of those five programs will display the help.

