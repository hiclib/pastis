import numpy as np
from scipy.ndimage import gaussian_filter


def smooth(counts, lengths,  sigma=1):
    begin_i, end_i = 0, 0
    for i, l_i in enumerate(lengths):
        end_i = l_i
        begin_j, end_j = 0, 0
        for j, l_j in enumerate(lengths):
            end_j = l_j
            sc = counts[begin_i:end_i, begin_j:end_j]
            counts[begin_i:end_i, begin_j:end_j] = gaussian_filter(sc, sigma)
            begin_j = l_j
        begin_i = end_i
    return counts


def average(counts, lengths):
    begin_i, end_i = 0, 0
    for i, l_i in enumerate(lengths):
        end_i = l_i
        begin_j, end_j = 0, 0
        for j, l_j in enumerate(lengths):
            end_j = l_j
            sc = counts[begin_i:end_i, begin_j:end_j]
            counts[begin_i:end_i, begin_j:end_j] = np.nanmean(sc)
            begin_j = l_j
        begin_i = end_i
    return counts
