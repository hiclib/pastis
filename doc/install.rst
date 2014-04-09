================================================================================
Installation
================================================================================

This software is composed of two parts: a python package ``pastis``, and two
C++ programs, ``MDS_all``, ``PM_all``, based on the work of `Duan and others
<http://noble.gs.washington.edu/proj/yeast-architecture/>`_.
The installation is in two steps: first the python module ``pastis``, then the
two programs ``MDS_all`` and ``PM_all``.

pastis
=====

This package uses distutils, which is the default way of installing
python modules.

The dependencies are:

- python (>= 2.6)
- setuptools
- numpy (>= 1.3)
- scipy (>= 0.7)
- scikit-learn (>= 0.13)
- argparse if python < 2.7

All of these dependencies can be installed at once using `Anaconda
<http://docs.continuum.io/anaconda/install.html>`_

In addition, `pyipopt <https://github.com/xuy/pyipopt>`_ is needed for pm2.

To install in your home directory, use::

    python setup.py install --user

or using pip::

    pip install --user pastis

.. note::

   You may have to create the folder .local/bin manually in your home
   directory, in order for the four programs ``pastis-mds``, ``pastis-nmds``,
   ``pastis-pm1``, ``pastis-pm2`` to be installed properly.

To install for all users on Unix/Linux::

    python setup.py build
    sudo python setup.py install

or using pip::

  pip install pastis

This will install a python package ``pastis``, and four programs ``pastis-mds``,
``pastis-nmds``, ``pastis-pm1`` and ``pastis-pm2``. Calling any of those four
programs will display the help.


MDS_all/PM_all
==============

You also need to compile the two softwares in src/MDS and src/PM, and place
them in the directory of your choice.

- First install IPOPT: https://projects.coin-or.org/Ipopt IPOPT can be
  installed anywhere.
- You have to edit the file Makefile in both the ``src/MDS`` and ``src/PM``,
  and set the following 3 variables at the top: IPOPTPATH, IPOPTINCDIR,
  IPOPTLIBDIR.
- Type make.
