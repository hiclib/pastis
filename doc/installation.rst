================================================================================
Installation
================================================================================

Dependencies
============

- python (>= 3.7)
- numpy (>= 1.16)
- scipy (>= 1.2)
- scikit-learn (>= 0.13)
- pandas (>= 0.24)
- autograd (>= 1.2)

All of these dependencies can be installed easily using conda:
`http://conda.pydata.org/miniconda.html <http://conda.pydata.org/miniconda.html>`_

Once conda is installed, just type the following::

    conda install numpy scipy scikit-learn pandas autograd

Installation
============

The easiest way to install PASTIS is through pip.

To install for all users, use the command::

    pip install pastis

To only install in your home directory, use::

    pip install --user pastis


Alternatively, you can install the most up-to-date code from GitHub using the
following commands.

To install for all users::

    git clone https://github.com/hiclib/pastis
    cd pastis
    python setup.py install

To only install in your home directory::

    git clone https://github.com/hiclib/pastis
    cd pastis
    python setup.py install --user


These commands will install a python package ``pastis``, and three programs:
``pastis-pm``, ``pastis-mds``, and ``pastis-nmds``. Calling any of those three
programs will display the help.
