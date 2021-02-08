from __future__ import print_function
from scipy import linalg
from sklearn.metrics import euclidean_distances
import numpy as np

# 160907 GBonora
# Python implementation of 'print_pdb_atom' function in coords.cpp


def fprintf(fp, fmt, *args):
    """
    Simply an alias for fprintf to allow Python implementation to remain
    similar to original C version.
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

def writePDB(Xpdb, bead_truths, pdbfilename):
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
            if (bead_truths != None):
              if not bead_truths[coordIdx]:
                  atom_name = 'H'
              else:
                  atom_name = 'O'
            print_pdb_atom(pdboutfile,
                           # chrom_index,
                           # copy_index,
                           coordIdx,
                           is_node,  # Is this a node or an edge atom
                           atom_name,  # eg "N", "O", "H", or "C",
                           scale_factor,
                           my_coords)
    # pdboutfile.close()

def realignment_error(X, Y, error_type):
    """
    If an error occurs during realignment, processes it.
    """
    if error_type.lower() == 'rmsd':
        return np.sqrt(((X - Y) ** 2.).sum() / len(X))
    elif error_type.lower() == 'distanceerror':
        return np.sqrt((
            (euclidean_distances(X) - euclidean_distances(Y)) ** 2.).sum())
    else:
        raise ValueError('Error error_type must be rmsd or distanceerror')


def realign_structures(X, Y, rescale=False, copy=True, verbose=False,
                       error_type='rmsd'):
    """
    Realigns Y and X

    Parameters
    ----------
    X : ndarray (n, 3)
        First 3D structure

    Y : ndarray (n, 3)
        Second 3D structure

    rescale : boolean, optional, default: False
        Whether to rescale Y or not.

    copy : boolean, optional, default: True
        Whether to copy the data or not

    verbose : boolean, optional, default: False
        The level of verbosity

    Returns
    -------
    Y : ndarray (n, 3)
        Realigned 3D, Xstructure
    """
    if copy:
        Y = Y.copy()
        X = X.copy()

    mask = np.invert(np.isnan(X[:, 0]) | np.isnan(Y[:, 0]))

    if rescale:
        Y, _, _, _ = realign_structures(X, Y)
        if error_type.lower() == 'rmsd':
            alpha = (X[mask] * Y[mask]).sum() / (Y[mask] ** 2).sum()
        elif error_type.lower() == 'distanceerror':
            dis_X = euclidean_distances(X[mask])
            dis_Y = euclidean_distances(Y[mask])
            alpha = (dis_X * dis_Y).sum() / (dis_Y ** 2).sum()

        Y *= alpha

    X -= np.nanmean(X, axis=0)
    Y -= np.nanmean(Y, axis=0)

    K = np.dot(X[mask].T, Y[mask])
    U, L, V = linalg.svd(K)
    V = V.T

    R = np.dot(V, U.T)
    if linalg.det(R) < 0:
        if verbose:
            print("Reflexion found")
        V[:, -1] *= -1
        R = np.dot(V, U.T)
    Y_fit = np.dot(Y, R)

    error = realignment_error(X[mask], Y_fit[mask], error_type)

    # Check at the mirror
    Y_mirror = Y.copy()
    Y_mirror[:, 0] = - Y[:, 0]

    K = np.dot(X[mask].T, Y_mirror[mask])
    U, L, V = linalg.svd(K)
    V = V.T
    if linalg.det(V) < 0:
        V[:, -1] *= -1

    R_mirror = np.dot(V, U.T)
    Y_mirror_fit = np.dot(Y_mirror, R_mirror)
    error_mirror = realignment_error(X[mask], Y_mirror_fit[mask], error_type)

    if error <= error_mirror:
        best_Y_fit = Y_fit
        best_error = error
        mirror = False
        best_R = R
    else:
        if verbose:
            print("Reflexion is better")
        best_Y_fit = Y_mirror_fit
        best_error = error_mirror
        mirror = True
        best_R = R_mirror

    return best_Y_fit, best_error, mirror, best_R


def find_rotation(X, Y, copy=True):
    """
    Finds the rotation matrice C such that \|x - Q.T Y\| is minimum.

    Parameters
    ----------
    X : ndarray (n, 3)
        First 3D structure

    Y : ndarray (n, 3)
        Second 3D structure

    copy : boolean, optional, default: True
        Whether to copy the data or not

    Returns
    -------
    Y : ndarray (n, 3)
        Realigned 3D structure
    """
    if copy:
        Y = Y.copy()
        X = X.copy()
    mask = np.invert(np.isnan(X[:, 0]) | np.isnan(Y[:, 0]))
    K = np.dot(X[mask].T, Y[mask])
    U, L, V = np.linalg.svd(K)
    V = V.T

    t = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, np.linalg.det(np.dot(V, U.T))]])
    R = np.dot(V, np.dot(t, U.T))
    Y_fit = np.dot(Y, R)
    X_mean = X[mask].mean()
    Y_fit -= Y_fit[mask].mean() - X_mean

    # Check at the mirror
    Y_mirror = Y.copy()
    Y_mirror[:, 0] = - Y[:, 0]

    K = np.dot(X[mask].T, Y_mirror[mask])
    U, L, V = np.linalg.svd(K)
    V = V.T

    t = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, np.linalg.det(np.dot(V, U.T))]])
    R_ = np.dot(V, np.dot(t, U.T))
    Y_mirror_fit = np.dot(Y_mirror, R_)
    Y_mirror_fit -= Y_mirror[mask].mean() - X_mean
    error_mirror = ((X[mask] - Y_mirror_fit[mask]) ** 2).sum()
    return R


def distance_between_structures(X, Y):
    """
    Computes the distances per loci between structures

    Parameters
    ----------
    X : ndarray (n, l)
        First 3D structure

    Y : ndarray (n, l)
        Second 3D structure

    Returns
    -------
    distances : (n, )
        Distances between the 2 structures
    """
    if X.shape != Y.shape:
        raise ValueError("Shapes of the two matrics need to be the same")

    return np.sqrt(((X - Y) ** 2).sum(axis=1))


def _round_struct_for_pdb(struct, max_num_char=8):
    """
    Rounds the structure for PDB
    """
    struct = np.asarray(struct)
    struct_abs = np.where(
        np.isfinite(struct) & (struct != 0),
        np.abs(struct), 10 ** (max_num_char - 1))
    round_factor = 10 ** (max_num_char - 1 - np.floor(np.log10(struct_abs)))
    round_factor = np.where(
        np.floor(np.log10(struct_abs)) < max_num_char,
        round_factor / 10, round_factor)
    return np.round(struct * round_factor) / round_factor


def realign_struct_for_plotting(struct_to_plot, struct_to_match, rescale=False):
    """
    Realigns the structure for plotting
    """

    from rmsd import kabsch
    struct_to_plot = struct_to_plot.copy()
    struct_to_match = struct_to_match.copy()
    struct_to_plot -= np.nanmean(struct_to_plot, axis=0)
    struct_to_match -= np.nanmean(struct_to_match, axis=0)
    mask = np.invert(
        np.isnan(struct_to_plot[:, 0]) | np.isnan(struct_to_match[:, 0]))
    struct_to_plot, _, _, _ = realign_structures(
        struct_to_match, struct_to_plot, rescale=rescale, error_type='rmsd')
    struct_to_plot[mask] = np.dot(
        struct_to_plot[mask], kabsch(
            struct_to_plot[mask], struct_to_match[mask]))
    return struct_to_plot

def interpolate_chromosomes(X, lengths, eps=1e-1, smooth=0):
    from scipy import interpolate
    from scipy.sparse import issparse
    """
    Return a smoothed interpolation of the chromosomes

    Parameters
    ----------
    X : ndarray (n, 3)
        The 3D structure

    lengths : ndarray (L, )
        The lengths of the chromosomes. Note that the sum of the lengths
        should correspond to the length of the ndarray of the 3D structure.

    Returns
    -------
    smoothed_X : the smoothed 3D structure, with a set of coordinates for
        each chromosome (ie, the first chromosome's points will be 
        smoothed_X[0]). smoothed_X[0][i] will be a list with an x, y, and
        z coordinate.

    """
    
    smoothed_X = [[],[]]
    smoothed_beginning = []
    smoothed_end = []
    
    mask = np.invert(np.isnan(X[:, 0]))
    
    begin, end = 0, 0
    
    enumerated = enumerate(lengths)
    
    for i, length in enumerated:        
        end += length
        x = X[begin:end, 0]
        y = X[begin:end, 1]
        z = X[begin:end, 2]
        indx = mask[begin:end]
        
        if not len(x):
            break
        
        if not indx[0]:
            x[0] = x[indx][0]
            x[indx][0] = np.nan
            y[0] = y[indx][0]
            y[indx][0] = np.nan

            z[0] = z[indx][0]
            z[indx][0] = np.nan

        if not indx[-1]:
            x[-1] = x[indx][-1]
            x[indx][-1] = np.nan
            y[-1] = y[indx][-1]
            z[indx][-1] = np.nan
            z[-1] = z[indx][-1]
            z[indx][-1] = np.nan

        indx = np.invert(np.isnan(x))
        
        m = np.arange(len(x))[indx]

        f_x = interpolate.Rbf(m, x[indx], smooth=smooth)
        f_y = interpolate.Rbf(m, y[indx], smooth=smooth)
        f_z = interpolate.Rbf(m, z[indx], smooth=smooth)
        
        for j in range(length - 1):
            if (j < length - 2):
                m = np.arange(j, j + 1 + 0.1, 0.1)
            else:
                m = np.arange(j, j + 1, 0.1)
            smoothed_X[i].append([f_x(np.arange(m.min(), m.max(), 0.1)),
                                  f_y(np.arange(m.min(), m.max(), 0.1)),
                                  f_z(np.arange(m.min(), m.max(), 0.1))])
        
        begin = end
        
    return smoothed_X

def reshape_interpolate(interpolated_x):
    new_structure = []
    for i in range(len(interpolated_x)):
        curr_row = interpolated_x[i]
        xs, ys, zs = curr_row[0], curr_row[1], curr_row[2]
        for j in range(len(xs)):
            new_structure.append([xs[j], ys[j], zs[j]])
    return np.array(new_structure)

def combine_structs(X_beads, X_interp):
    result = []
    result_truth = []
    for i in range(len(X_beads) - 1):
        bead = X_beads[i]
        curr_interp = X_interp[i]
        xs, ys, zs = curr_interp[0], curr_interp[1], curr_interp[2]
        result.append(bead)
        result_truth.append(True)
        for j in range(len(xs)):
            result.append([xs[j], ys[j], zs[j]])
            result_truth.append(False)
    result.append(X_beads[-1])
    result_truth.append(True)
    return result, result_truth

def main():
    # Parse the arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True,
                        help="input file name, should be the structure output by running PASTIS")
    parser.add_argument('--output_file', type=str, required=True,
                        help='output file name, ie <ouput_file_name.pdb>')
    parser.add_argument('--interpolate_coords', type=bool, default=True, required=False,
                        help='if true, interpolates the points between the coordinate beads as hydrogen')
    args = parser.parse_args()
  
    # Load the structure
    struct_coords = np.loadtxt(args.input_file, delimiter=" ")
   
   # Get the length of the structure
   struct_length = len(struct_coords)
   
   struct_interpolated = None
   if (args.interpolate_coords):
     struct_interpolated = interpolate_chromosomes(struct_coords, [struct_length])[0]
     struct_coords, struct_truths = combine_structs(struct_coords, struct_interpolated)

    # Check the length
    if (len(struct_coords) > 10000):
      raise ValueError('structure can be at most 10000 beads')
  
    # Realign the structure
    struct_realigned = realign_struct_for_plotting(struct_to_plot=struct_coords,
                                                   struct_to_match=struct_coords)
  
    # Round the structure
    struct_round = _round_struct_for_pdb(struct_realigned)

    # Write the structure to a PDB file
    writePDB(struct_round, struct_truths, args.output_file)

    # Load the PDB file again
    pdb_file = open(args.output_file, 'r').read().split('\n')

    # Fine which lines are nan
    bad_lines = []
    for i in range(len(pdb_file)):
        tokens = pdb_file[i].split()
        true_line = []
        for j in range(len(tokens)):
            if ((j >= 5) & (j <= 7)):
                token = tokens[j]
                if (str(token) != 'nan'):
                    true_line.append(token)
            else:
                true_line.append(tokens[j])
        if len(true_line) != len(tokens):
            bad_lines.append(i)
  
    # Print out the not nan lines
    with open(args.output_file, 'w') as pdb_output_file:
        for i in range(len(pdb_file)):
            if i not in bad_lines:
              print(pdb_file[i], file=pdb_output_file) 

if __name__ == "__main__":
      main()
