import numpy as np

def bezier(X, lengths):
    smoothed_X = []
    for l in lengths:
        smoothed_chrom = []
        for i_locus in range(l):
            if i_locus == 0:
                x = X[i_locus]


        smoothed_X.append(smoothed_chrom)
    return smoothed_X
