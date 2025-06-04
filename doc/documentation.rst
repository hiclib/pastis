================================================================================
Documentation
================================================================================

.. toctree::
   :maxdepth: 2

   auto_examples/index
   modules/classes.rst


Pastis
======

In this package, we provide four algorithms to infer the 3D structure of the
genome: MDS, NMDS, PM1, PM2. For all methods, chromosomes are represented as
beads on a string, at a given resolution. All are installed as a standalone
program: pastis-mds, pastis-nmds, pastis-pm1, pastis-pm2.

Running the example
===================

We provide a sample interaction count matrix of the first 5 chromosomes and
configuration file ``config.ini``. To run the example, edit the two options
``binary_mds`` (respectively ``binary_pm``) to the path where you have
installed ``MDS_all`` (respectively ``PM_all``).

.. code-block:: guess

   [all]
   resolution: 10000
   output_name: structure.pdb
   counts: data/counts.matrix


To run the code, simply call the program of your choice, and the repository
containing the configuration file as argument. From the root of the
repository, to run the MDS::

  pastis-mds example

A bunch of files, necessary for the optimization are written in the same
folder as the optimization, including the results of the optimization:
``MDS.structure`` and ``MDS.structure.pdb``. The ``txt`` file contains the
array of coordinates while the ``pdb`` contains a smoothed interpolation of
the structure for visualization purposes.

Running the algorithms on your own structure
============================================

Running the algorithms on your own structure, and your own organism will
require a bit more work.

The **interaction count matrix** needs to be saved as a `numpy array
<http://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html>`_, in
the folder of your choice. Numpy arrays are memory-efficient container that
provides fast numerical operations: the ideal structure to work efficiently
with matrices in Python. For efficient and accurate

The **organism structure** (chromosomes lengths of the organism) have to be
specified in a text file, one chromosomes length per base pair. See
``example/files/budding_yeast_structure`` for an example.

Then, create the configuration file, specifying the resolution on which to run
the algorithm: the resolution of the interaction counts matrix, the
chromosomes lengths and the **resolution** should be coherent.

