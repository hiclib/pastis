import numpy as np


def convert_back(pdb_file):
    """
    Loads the PDB file and returns the coordinates in a numpy array.
    Parameters
    ----------
    pdb_file: the full file name of the PDB file to load

    Returns
    -------
    result: numpy array
            an array of the coordinates from the input PDB file
    """
    lines = open(pdb_file).read().splitlines()
    converted = []
    for line in lines:
        line = line.split()
        if len(line) != 0:
            converted.append((float(line[5]), float(line[6]), float(line[7])))
    return np.array(converted)


def remove_nans(curr_struct):
    """
    Removes any coordinate triples that have a NAN coordinate. Returns the
    coordinates.
    Parameters
    ----------
    curr_struct: the coordinates to remove NANs from; should be a numpy array.

    Returns
    -------

    result: numpy array
            an array of the coordinates frmo curr_struct, with any coordinate
            triples containing a NAN coordinate removed.e
    """
    result = []
    for coords in curr_struct:
        skip = False
        for coord in coords:
            skip = np.isnan(coord)
        if (skip):
            continue
        else:
            result.append(coords)
    result = np.array(result)
    return result


def main():
    """
    Loads the PASTIS coordinates and the PDB version of those pastis
    coordinates. Centers the PDB coordinates around the origin and scales them
    back down. Asserts that the PASTIS coordinates and the modified PDB
    coordinates are equal to each other to one decimal place.

    Parameters
    ----------
    pastis_coords_file : str
        Name of the pastis coordinates file; it should be the coordinates of
        3d structure output by PASTIS.
    pdb_file: str
        Name of the PDB file (the pdb version of the pastis file).
    """

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pastis_coords_file", type=str, required=True,
                        help="coordinates of the pastis structure")
    parser.add_argument("--pdb_file", type=str, required=True,
                        help="pdb file version of pastis_coords_file")
    args = parser.parse_args()

    # Load the two files
    struct_original = np.loadtxt(args.pastis_coords_file)
    struct_pdb = convert_back(args.pdb_file)

    # Recenter and scale the PDB coordinates (scale factor is 10)
    struct_pdb_recentered = (struct_pdb - np.mean(struct_pdb, axis=0)) / 10
    struct_original = remove_nans(struct_original)

    # Assert they are equal to one decimal place
    np.testing.assert_array_almost_equal(struct_original,
                                         struct_pdb_recentered,
                                         decimal=1)


if __name__ == "__main__":
    main()
