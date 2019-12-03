PASTIS: Poisson-based Algorithm for STable Inference of DNA Structure
=====================================================================

TODO write blurb about PASTIS & diploid PASTIS

Dependencies
------------

- python (>= 3.7)
- numpy (>= 1.16)
- scipy (>= 1.2)
- scikit-learn (>= 0.13)
- pandas (>= 0.24)
- autograd (>= 1.2)

All of these dependencies can be installed easily using conda python3.7:
`http://conda.pydata.org/miniconda.html <http://conda.pydata.org/miniconda.html>`_

Once conda is installed, just type the following::

    conda install numpy scipy scikit-learn pandas autograd


Install PASTIS
--------------

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

This will install a python package ``pastis``, and three programs ``pastis-pm``,
``pastis-mds``, and ``pastis-nmds``. Calling any of those three programs will
display the help.

Usage
-----

PASTIS
******

TODO

MDS
***

TODO