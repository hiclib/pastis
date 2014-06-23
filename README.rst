PASTIS: Poisson-based algorithm for stable inference of DNA Structure
=====================================================================

This package is divided into two parts: a python module and two C++ programs.
To function properly, you need both the python module and the C++ programs.

Dependencies
------------

For Pastis:

- numpy
- scipy
- scikit-learn
- pyipopt

For MDS_all and PM_all:
- ipopt

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


MDS_all/PM_all
**************

You also need to compile the two softwares in src/MDS and src/PM, and place
them in the directory of your choice.

1) First install IPOPT: https://projects.coin-or.org/Ipopt
IPOPT can be installed anywhere, you will have to tell "3dmodel" where IPOPT
is.
2) The "3dmodel" directory can be anywhere you like. 
3) You have to edit the file Makefile, and set the following 3 variables at
the
top: IPOPTPATH, IPOPTINCDIR, IPOPTLIBDIR.
4) Type make. 

More information can be found on the project website:
http://cbio.ensmp.fr/pastis
