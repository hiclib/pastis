PASTIS: Poisson-based Algorithm for STable Inference of DNA Structure
=====================================================================


Dependencies
------------

For Pastis:

- python (>= 2.7)
- numpy
- scipy
- scikit-learn
- pandas
- iced

Additional dependencies for the diploid version:
- python (>= 3.6)
- autograd (>= 1.3)

All of these dependencies can be installed easily using conda:
`http://conda.pydata.org/miniconda.html <http://conda.pydata.org/miniconda.html>`_

Once conda is installed, just type the following::

  conda install numpy scipy scikit-learn pandas


`iced` can be installed via::

  pip install iced

Install PASTIS
--------------

This package uses distutils, which is the default way of installing
python modules. To install in your home directory, use::

  python setup.py install --user

To install for all users on Unix/Linux::

  python setup.py build
  sudo python setup.py install


or using pip::

    pip install --user pastis

