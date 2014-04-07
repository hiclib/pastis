================================================================================
Tutorial
================================================================================

MDS, NMDS, PM1, PM2
===================

In this package, we provide four algorithms to infer the 3D structure of the
genome: MDS, NMDS, PM1, PM2. For all methods, chromosomes are represented as
beads on a string, at a given resolution.

Running the example
===================

We provide a sample interaction count matrix of the first 5 chromosomes and
configuration file ``config.ini``. To run the example, edit the two options
``binary_mds`` (respectively ``binary_pm``) to the path where you have
installed ``MDS_all`` (respectively ``PM_all``).

.. code-block:: guess

   [all]
   binary_mds: /bioinfo/users/nvaroqua/.local/bin/MDS_all
   binary_pm: /bioinfo/users/nvaroqua/.local/bin/PM_all
   resolution: 10000
   output_name: structure.pdb
   chromosomes: 1,2,3,4,5
   organism_structure: files/budding_yeast_structure
   counts: data/counts.npy

