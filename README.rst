PASTIS: Poisson-based algorithm for stable inference of DNA Structure
=====================================================================


Dependencies
------------

For Pastis:

- python 2.7
- numpy
- scipy
- scikit-learn
- pandas

All of these dependencies can be installed easily using conda python2.7:
`http://conda.pydata.org/miniconda.html <http://conda.pydata.org/miniconda.html>`_

Once conda is installed, just type the following::

  conda install numpy scipy scikit-learn pandas

and then install pastis with::

  python setup.py install

Install
-------

Pastis
*******
This package uses distutils, which is the default way of installing
python modules. To install in your home directory, use::

  python setup.py install --user

To install for all users on Unix/Linux::

    python setup.py build
    sudo python setup.py install


