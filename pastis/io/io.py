import numpy as np


def write(X, filename, lengths=None,
          resolution=50000,
          copy=True):
    """
    Write the similarity matrix to a file

    Parameters
    ----------
    X: ndarray,
    """
    if copy:
        X = X.copy()
    X = X.astype(float)

    output_file = open(filename, 'w')

    tr = X.sum(axis=0) == 0
    if tr.sum() != 0:
        X[tr, :] = np.nan
        X[:, tr] = np.nan

    for i in range(len(X)):
        chr1, loc1 = _get_index(i, lengths)
        if loc1 > lengths[chr1 - 1]:
            raise ValueError("Problem !")
        loc1 *= resolution
        loc1 += 1
        for j in range(len(X)):
            if i > j:
                continue
            chr2, loc2 = _get_index(j, lengths)
            if loc2 > lengths[chr2 - 1]:
                raise ValueError("Problem !")
            loc2 *= resolution
            loc2 += 1

            if chr2 == chr1:
                dis = loc2 - loc1
            else:
                dis = 0

            if not X[i, j] or np.isnan(X[i, j]):
                continue

            output_file.write('%d\t%d\t%d\t%d\t%d\t%f\t0\t0\n' %
                              (chr1, loc1, chr2, loc2, dis, X[i, j]))

    output_file.close()


def _get_index(i, lengths):
    """
    From an index, return the chromosome number and the loci number

    Parameters
    ----------
    i : integer
        index in the full matrix

    lengths : ndarray (n, )

    Returns
    -------
    tuple : (c, l)
        chromosome number, locus
    """
    length_cum = lengths.cumsum()
    c = (length_cum > i).argmax() + 1

    if c > 1:
        length = i - length_cum[c - 2]
    else:
        length = i
    if len(lengths) > 1 and length > lengths[c - 1]:
        raise ValueError("there's a problem")

    if c > len(lengths):
        raise ValueError("There's a problem")
    return c, length


###############################################################################
# 160907 GBonora
# Python implementation of 'print_pdb_atom' function in coords.cpp

def fprintf(fp, fmt, *args):
    """
    Simply an alias for fprintf to allow Python implementation to remain similar to original C version.

    Parameters
    ----------
    fp : output file

    fmt : C-style format string

    args: arguments for format string

    """
    fp.write(fmt % args)


def print_pdb_atom(outfile,
                   # chrom_index,
                   # copy_index,
                   atom_index,
                   is_node,
                   atom_name,
                   scale_factor,
                   my_coords):
    """
    Prints one line to PDB file.

    Parameters
    ----------
    outfile : output file

    atom_index : integer
        'atom' ID

    is_node : boolean
        Is this a node or an edge atom?

    atom_name : string
        eg "N", "O", or "C"

    scale_factor : integer
        100 for a sphere with radius 1, 1000 for a sphere with radius 10

    my_coords :
        3D co-ordinates to ouput


    """
    # prev_chrom_index = -1
    # atom_index
    # if chrom_index != prev_chrom_index:
    #     atom_index = 1
    #     prev_chrom_index = chrom_index

    # http://www.biochem.ucl.ac.uk/~roman/procheck/manual/manappb.html
    fprintf(outfile, "ATOM  ")  # 1- 6: Record ID
    fprintf(outfile, "%5d", atom_index)  # 7-11: Atom serial number
    fprintf(outfile, " ")  # 12: Blank
    # if (is_node) {                        # 13-16: Atom name
    #   fprintf(outfile, "N   ")
    # } else {
    #   fprintf(outfile, "O   ")
    # }
    fprintf(outfile, "%s   ", atom_name)
    fprintf(outfile, " ")  # 17-17: Alternative location code
    if is_node:  # 18-20: 3-letter amino acid code
        fprintf(outfile, "NOD")
    else:
        fprintf(outfile, "EDG")
    fprintf(outfile, " ")  # 21: Blank
    fprintf(outfile, "%c",  # 22: Chain identifier code
            # get_chrom_id(chrom_index, copy_index))
            'A')
    fprintf(outfile, "    ")  # 23-26: Residue sequence number
    fprintf(outfile, " ")  # 27: Insertion code
    fprintf(outfile, "   ")  # 28-30: Blank
    fprintf(outfile, "%8.3f%8.3f%8.3f",  # 31-54: Atom coordinates
            # (my_coords->x + 1.0) * SCALE_FACTOR,
            # (my_coords->y + 1.0) * SCALE_FACTOR,
            # (my_coords->z + 1.0) * SCALE_FACTOR)
            (my_coords[0] + 1.0) * scale_factor,
            (my_coords[1] + 1.0) * scale_factor,
            (my_coords[2] + 1.0) * scale_factor)
    fprintf(outfile, "%6.2f", 1.0)  # 55-60: Occupancy value
    if is_node:
        fprintf(outfile, "%6.2f", 50.0)  # 61-66: B-value (thermal factor)
    else:
        fprintf(outfile, "%6.2f", 75.0)  # 61-66: B-value (thermal factor)
    fprintf(outfile, " ")  # 67: Blank
    fprintf(outfile, "   ")  # 68-70: Blank
    fprintf(outfile, "\n")


def writePDB(Xpdb, pdbfilename):
    """
    Write 3D ouput from pastis to PDB file.

    Parameters
    ----------
    Xpdb : nparray
        the 3D co-ordinates

    pdbfilename : File to write PDB file to.

    """
    # print('PDB file creation!')
    atom_name = 'O'
    scale_factor = 100  # 100 for a sphere with radius 1, 1000 for a sphere with radius 10
    # pdboutfile = open(pdbfilename,'w')
    with open(pdbfilename, "w") as pdboutfile:
        for coordIdx in range(0, np.shape(Xpdb)[0]):
            # print coordIdx
            if coordIdx == 0 or coordIdx == np.shape(Xpdb)[0]:
                is_node = True
            else:
                is_node = False
            my_coords = Xpdb[coordIdx, :]
            print_pdb_atom(pdboutfile,
                           # chrom_index,
                           # copy_index,
                           coordIdx,
                           is_node,  # Is this a node or an edge atom
                           atom_name,  # eg "N", "O", or "C",
                           scale_factor,
                           my_coords)
    # pdboutfile.close()
