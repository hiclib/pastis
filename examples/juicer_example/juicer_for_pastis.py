"""
Using a .hic file (output of Juicer) with PASTIS
================================================

This example showcases how to convert a .hic file output by Juicer into a numpy
array, output a .bed file for the lengths, and then run PASTIS.
"""


import numpy as np
import strawC
import os
from scipy import sparse


##############################################################################
# Returns a numpy array with counts data from the given .hic file.
#     val_type : "observed" for raw values; "oe" for observed/expected values
#     norm : the normalization to apply ("NONE" for no normalization)
#     hic_file : the path to the .hic file we want to get counts data for
#     width : the size of the returned counts matrix
#     chrom1 : the first chromosome
#     chrom2 : the second chromosome
#     resolution : the BP resolution
# --------------------------------------------------------------------------
def grabRegion(val_type, norm, hic_file, width, chrom1, chrom2, resolution):
    chrom_range1 = str(chrom1)
    chrom_range2 = str(chrom2)
    result = strawC.strawC(val_type, norm, hic_file, chrom_range1,
                           chrom_range2, 'BP', resolution)
    row_indices, col_indices, data = list(), list(), list()
    for record in result:
        row_indices.append(record.binX)
        col_indices.append(record.binY)
        data.append(record.counts)
        if chrom1 == chrom2 and record.binX != record.binY:
            row_indices.append(record.binY)
            col_indices.append(record.binX)
            data.append(record.counts)
    row_indices = (np.asarray(row_indices)) / resolution
    col_indices = (np.asarray(col_indices)) / resolution
    matrix = sparse.coo_matrix((data, (row_indices.astype(int),
                               col_indices.astype(int))),
                               shape=(width, width)).toarray()
    matrix[np.isnan(matrix)] = 0
    matrix[np.isinf(matrix)] = 0
    return matrix


##############################################################################
# Define the path to our .hic file.
# ---------------------------------
FILE_PATH = './data/Pfalciparum_trophozoite_Ay2014.hic'


##############################################################################
# Use the grabRegion method to get counts data for the first chromosome
# in the .hic file at a resolution of 25000. We won't apply any normalization.
# ----------------------------------------------------------------------------
counts = grabRegion('observed', 'NONE', FILE_PATH, 26, 1, 1, 25000)


##############################################################################
# Output a .bed file of lengths. Let's call the file "counts.bed".
# ----------------------------------------------------------------
with open("./data/counts.bed", "w") as f:
    the_length = len(counts)
    for i in range(the_length):
        curr_line = 'Chr01\t' + str(i + 1) + '\t' + str(i + 1) + '\t'
        curr_line += str(i)
        print(curr_line, file=f)


##############################################################################
# Save the counts as a .npy file. Let's call the file "counts.npy".
# -----------------------------------------------------------------
np.save("./data/counts.npy", counts)


##############################################################################
# Make the following call using os to run PASTIS with the counts data.
# The call could be made in command line as well. Note that there are other
# settings we could run PASTIS with; this is simply one set of those settings.
# ----------------------------------------------------------------------------
settings = "pastis-poisson --seed 0 --counts ./data/counts.npy --outdir"
settings += " ./results --lengths ./data/counts.bed --ploidy 1"
settings += " --filter_threshold 0.04 --multiscale_rounds 1 --max_iter 50000"
settings += " --dont-normalize --alpha 3"
os.system(settings)
