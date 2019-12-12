.. PASTIS documentation master file, created by
   sphinx-quickstart on Tue Dec 10 14:23:02 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. image:: _static/logo/logo_1.png
    :width: 300px

|

.. image:: https://travis-ci.org/hiclib/pastis.svg?branch=master
   :target: https://travis-ci.org/hiclib/pastis

.. image:: https://readthedocs.org/projects/pastis/badge/?version=latest
   :target: http://pastis.readthedocs.io/en/latest/?badge=latest

|

About
=====

PASTIS infers a single "consensus" 3D chromatin structure from population
Hi-C data - **for haploid or diploid genomes**.

PASTIS models chromosomes as beads on a chain. It operates by modeling contact
counts between two loci via a Poisson distribution that is proportional to the
distances between corresponding beads. This statistical model allows PASTIS to
handle noise better than its metric-based predecessors, such as those that
rely on multidimensional scaling.

The relationship between contact counts and distances is handled by an
empirically-based biophysical model. Because this relationship can vary based
on various biological and technical factors, a parameter of the biophysical
model is jointly inferred alongside the structure.

Is your data diploid?
---------------------

Diploid organisms present additional challenges for any structural inference
approach because Hi-C data does not inherently discriminate between the
homologs (`see here for details
<https://pastis.readthedocs.io/en/latest/diploidy.html>`_).
Nearly all structural inference methods are designed exclusively for
haploid organisms. Applying a haploid method to a diploid genome has
significant downsides: it makes the strong assumption that the strucures
of both homologs are identical, and it prevents modeling of more than one
individual chromosome at a time.

PASTIS has the unique ability to infer chromatin structures for diploid
organisms regardless of whether the Hi-C data is segregatated by allele.
It takes in any and all Hi-C data

PASTIS has the unique ability to infer chromatin structures for diploid
organisms, even if the Hi-C data is not segregated by allele. And if you do
have allelically segregated Hi-C data, you can input that alongside the
unsegregated data to build an even better model.

References
==========

N. Varoquaux, F. Ay, W. S. Noble, and J.-P. Vert. `A statistical approach for
inferring the 3D structure of the genome.
<http://bioinformatics.oxfordjournals.org/content/30/12/i26.short>`_
Bioinformatics, 30(12):i26–i33, 2014.

A. G. Cauer, G. Yardimci, J.-P. Vert, N. Varoquaux, and W. S. Noble. `Inferring
diploid 3D chromatin structures from Hi-C data.
<http://drops.dagstuhl.de/opus/volltexte/2019/11041/>`_ In 19th International
Workshop on Algorithms in Bioinformatics (WABI 2019), volume 143 of Leibniz
International Proceedings in Informatics (LIPIcs), pages 11:1–11:13, Dagstuhl,
Germany, 2019. Schloss Dagstuhl–Leibniz-Zentrum fuer Informatik.

Contacts
========

If you have any questions or suggestions, please email gesine at uw dot edu and
nelle dot varoquaux at ensmp dot fr, or open a ticket on `Github
<https://github.com/hiclib/pastis/issues>`_.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   self
   diploidy.rst
   faq.rst
   whats_new.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Usage

   installation.rst
   data.rst
   tutorial.rst
   api.rst
   advanced.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Features

   constraints.rst
