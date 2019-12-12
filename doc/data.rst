==========
Input Data
==========

Interaction counts matrix
=========================

PASTIS accepts two formats for interaction counts data files.

1. `Numpy array
<http://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html>`_: TODO

2. Hiclib format: TODO


For diploid
-----------

If a portion of your diploid data is segregated by allele, input three matrices:

1. **"Unambiguous" counts** contain data where both ends of each contact count are segregated by allele.

2. **"Partially ambiguous" counts** contain data where exactly one end of each contact count is segregated by allele.

3. **"Ambiguous" counts** contain data where neither end of the contact count is segregated by allele.

For unambiguous and partially ambiguous counts, the matrices should be arranged
such that all chromosomes from the first homolog precede all chromosomes from
the second homolog.

TODO put figure illustrating this.


Chromosome lengths
==================

The chromosome lengths indicate how many beads are in each chromosome in the
data. The order of the chromosomes  in the lengths data should match the order
of the chromosomes in the interaction counts matrix.

Lengths can be inputted via two options:

1. A list of chromosome lengths can be inputted directly via the command-line API.

2. Chromosome lengths can be loaded from a file in the Hiclib "organism structure" format (see ``example/files/budding_yeast_structure`` for an example.)

For diploid
-----------

Dach homologous pair is represeneted by only one entry in the chromosome lengths
data. For example, if chr1 is represented by 300 beads and chr2 is represented
by 400 beads, the lengths should be "300 400" (rather than "300 400 300 400").

Optional chromosome names
=========================

TODO

