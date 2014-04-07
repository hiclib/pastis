/**************************************************************************
 * FILE: coords.h
 * AUTHOR: William Stafford Noble
 * CREATE DATE: 24 October 2008
 * PROJECT: BLAU
 * DESCRIPTION: 3D coordinates in a unit sphere
 **************************************************************************/
#ifndef COORDS_H
#define COORDS_H

typedef struct coords {
  double x;
  double y;
  double z;
} COORDS_T;

//typedef struct coords COORDS_T;

/**************************************************************************
 * Allocate space for one set of coordinates.
 **************************************************************************/
COORDS_T* new_coords ();

/**************************************************************************
 * Copy the contents of one coordinate into another.
 *
 * This function does not allocate memory.
 **************************************************************************/
void copy_coords
  (COORDS_T* source_coords,
   COORDS_T* destination_coords);

/**************************************************************************
 * Copy the contents of one coordinate into another.
 *
 * This function does not allocate memory.
 **************************************************************************/
void copy_coords_xyz
  (double x, double y, double z,
   COORDS_T* destination_coords);

/**************************************************************************
 * Choose random coordinates within the unit sphere.
 **************************************************************************/
void randomize_coords
  (COORDS_T* my_coords);

/**************************************************************************
 * Compute the Euclidean distance between two points.
 **************************************************************************/
double compute_distance
  (COORDS_T* first,
   COORDS_T* second);

/**************************************************************************
 * Pick a point that is less than or equal to a specified distance
 * away from a given point in a randomly selected direction.
 **************************************************************************/
void pick_nearby_coords
  (double     distance,
   COORDS_T* start_coords,
   COORDS_T* new_coords);

/**************************************************************************
 * Move a point at random within a small sphere.
 **************************************************************************/
void jiggle_coords
  (double     distance,
   COORDS_T* my_coords);

/*****************************************************************************
 * Move a given point a specified distance toward a second point.
 *****************************************************************************/
//static
void move_toward
  (COORDS_T* coords1,
   COORDS_T* coords2,
   double     distance_to_move);

void move_toward_1D
  (int & from,
   int to,
   COORDS_T* coords1,
   COORDS_T* coords2,
   double     distance_to_move);

/*****************************************************************************
 * Move a given point a portion of the toward a second point.
 *****************************************************************************/
void move_toward_portion
  (COORDS_T* coords1,
   COORDS_T* coords2,
   double     portion_to_move);  // fraction between 0 and 1

/*****************************************************************************
 * Print a series of points corresponding to an edge between two nodes.
 *****************************************************************************/
void print_pdb_edge
  (FILE*     outfile,
   int       chrom_index,
   int       copy_index,
   COORDS_T* coords1,
   COORDS_T* coords2);

/*****************************************************************************
 * Print a series of points corresponding to an edge between two nodes.
 *****************************************************************************/
void print_1D_3D_edge
  (FILE*     outfile,
   int       chrom_index,
   int       copy_index,
   COORDS_T* coords1,
   COORDS_T* coords2,
   int       from_1D,
   int       to_1D);

/*****************************************************************************
 * Print a series of points along a quadratic bezier curve with control points p0,p1,p2
 *****************************************************************************/
void print_pdb_quadratic (FILE* outfile, int chrom_index, int copy_index,
  COORDS_T* p0, COORDS_T* p1, COORDS_T* p2, int has_centromere);

/*****************************************************************************
 * Print 1D coordinates and a series of points along a quadratic bezier curve with control points p0,p1,p2
 *****************************************************************************/
void print_1D_3D_quadratic (FILE* outfile, int chrom_index, int copy_index,
  COORDS_T* p0, COORDS_T* p1, COORDS_T* p2, int from_1D, int to_1D);

/*****************************************************************************
 * Print the coordinates of one atom in PDB format.
 *****************************************************************************/
void print_pdb_coords
  (FILE*     outfile,
   int       chrom_index,
   int       copy_index,
   COORDS_T* my_coords);

void print_pdb_atom
  (FILE*     outfile,
   int       chrom_index,
   int       copy_index,
   BOOLEAN_T is_node,    // Is this a node or an edge atom?
   char*    atom_name,   // eg "N", "O", or "C"
   COORDS_T* my_coords);

void print_1D_3D_atom
  (FILE*     outfile,
   int       chrom_index,
   int       copy_index,
   COORDS_T* my_coords,
   int       genomic_position);


#endif

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 2
 * End:
 */
