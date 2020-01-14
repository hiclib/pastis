================================================================================
Installation
================================================================================

pastis
=======

This package uses distutils, which is the default way of installing
python modules.

The dependencies are:

- python (>= 2.7)
- setuptools
- numpy (>= 1.3)
- scipy (>= 0.7)
- scikit-learn (>= 0.13)
- iced

For the diploid version, additional dependencies are required:
- python (>= 3.6)
- autograd (>= 1.3)

Most of these dependencies can be installed at once using `Anaconda
<http://docs.continuum.io/anaconda/install.html>`_

`iced` can be installed using pip::

  pip install --user iced


To install in your home directory, use::

    python setup.py install --user

or using pip::

    pip install --user pastis

To install for all users on Unix/Linux::

    python setup.py build
    sudo python setup.py install

or using pip::

  pip install pastis

This will install a python package ``pastis``, and four programs ``pastis-mds``,
``pastis-nmds``, ``pastis-pm1`` and ``pastis-pm2``. Calling any of those four
programs will display the help.

