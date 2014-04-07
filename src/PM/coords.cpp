/**************************************************************************
 * FILE: coords.c
 * AUTHOR: William Stafford Noble
 * CREATE DATE: 24 October 2008
 * PROJECT: BLAU
 * DESCRIPTION: 3D coordinates in a unit sphere
 **************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include "utils.h"
#include "coords.h"
#include <time.h>


/**************************************************************************
 * Allocate space for one set of coordinates.
 **************************************************************************/
COORDS_T* new_coords () {

  // initialize random seed:
  srand ( time(NULL) );

  COORDS_T* return_value;

  return_value = (COORDS_T*)mymalloc(sizeof(COORDS_T));
  return_value->x = rand();
  return_value->y = rand();
  return_value->z = rand();
  return(return_value);
}

/**************************************************************************
 * Copy the contents of one coordinate into another.
 *
 * This function does not allocate memory.
 **************************************************************************/
void copy_coords
  (COORDS_T* source_coords,
   COORDS_T* destination_coords)
{
  destination_coords->x = source_coords->x;
  destination_coords->y = source_coords->y;
  destination_coords->z = source_coords->z;
}

/**************************************************************************
 * Copy the contents of one coordinate into another.
 *
 * This function does not allocate memory.
 **************************************************************************/
void copy_coords_xyz
  (double x, double y, double z,
   COORDS_T* destination_coords)
{
  destination_coords->x = x;
  destination_coords->y = y;
  destination_coords->z = z;
}

/**************************************************************************
 * Compute the Euclidean distance between two points.
 **************************************************************************/
double compute_distance
  (COORDS_T* first,
   COORDS_T* second)
{
  double x_diff = first->x - second->x;
  double y_diff = first->y - second->y;
  double z_diff = first->z - second->z;

  return(sqrt((x_diff * x_diff) + (y_diff * y_diff) + (z_diff * z_diff)));
}

/**************************************************************************
 * Compute the distance from the given point to the origin.
 **************************************************************************/
static double compute_distance_to_origin
  (COORDS_T* my_coords)
{
  return(sqrt((my_coords->x * my_coords->x) +
	      (my_coords->y * my_coords->y) +
	      (my_coords->z * my_coords->z)));
}

/**************************************************************************
 * Is a given point within the unit sphere?
 **************************************************************************/
static BOOLEAN_T valid_coords
  (COORDS_T* my_coords)
{
  if (compute_distance_to_origin(my_coords) < 1.0) {
    return(TRUE);
  }
  return(FALSE);
}

/**************************************************************************
 * Choose random coordinates within the unit sphere.
 **************************************************************************/
void randomize_coords
  (COORDS_T* my_coords)
{
  do {
    my_coords->x = (2.0 * drand48()) - 1;
    my_coords->y = (2.0 * drand48()) - 1;
    my_coords->z = (2.0 * drand48()) - 1;
  } while (!valid_coords(my_coords));
}

/**************************************************************************
 * Pick a random point on the unit sphere.
 **************************************************************************/
static void choose_point_on_unit_sphere
  (COORDS_T* my_coords)
{
  // Choose a random point inside the unit sphere.
  randomize_coords(my_coords);

  // Get its distance from the origin.
  double length = compute_distance_to_origin(my_coords);

  // Scale it by the specified distance.
  my_coords->x /= length;
  my_coords->y /= length;
  my_coords->z /= length;

}

/**************************************************************************
 * Pick a point that is less than or equal to a specified distance
 * away from a given point in a randomly selected direction.
 **************************************************************************/
void pick_nearby_coords
  (double     distance,
   COORDS_T* start_coords,
   COORDS_T* new_coords)
{
  static COORDS_T random_coords;

  do {

    // Choose a random point in the unit sphere.
    randomize_coords(&random_coords);

    // Scale it by the specified distance.
    random_coords.x *= distance;
    random_coords.y *= distance;
    random_coords.z *= distance;

    // Translate our given point.
    new_coords->x = start_coords->x + random_coords.x;
    new_coords->y = start_coords->y + random_coords.y;
    new_coords->z = start_coords->z + random_coords.z;

  } while (!valid_coords(new_coords));
}

/**************************************************************************
 * Move a point at random within a small sphere.
 **************************************************************************/
void jiggle_coords
  (double     distance,
   COORDS_T* my_coords)
{
  COORDS_T new_coords;
  pick_nearby_coords(distance, my_coords, &new_coords);
  my_coords->x = new_coords.x;
  my_coords->y = new_coords.y;
  my_coords->z = new_coords.z;
}

/*****************************************************************************
 * Print the coordinates of one atom in PDB format.
 *****************************************************************************/
static char get_chrom_id
  (int chrom_index,
   int       copy_index)
{
  char return_value = '\0';
  // looks like the chain name should be capital letter, not lowercase, that way rasmol can color different chains by a specified color
  if (chrom_index == 0) return_value = 'A';
  else if (chrom_index == 1) return_value = 'B';
  else if (chrom_index == 2) return_value = 'C';
  else if (chrom_index == 3) return_value = 'D';
  else if (chrom_index == 4) return_value = 'E';
  else if (chrom_index == 5) return_value = 'F';
  else if (chrom_index == 6) return_value = 'G';
  else if (chrom_index == 7) return_value = 'H';
  else if (chrom_index == 8) return_value = 'I';
  else if (chrom_index == 9) return_value = 'J';
  else if (chrom_index == 10) return_value = 'K';
  else if (chrom_index == 11) return_value = 'L';
  else if (chrom_index == 12) return_value = 'M';
  else if (chrom_index == 13) return_value = 'N';
  else if (chrom_index == 14) return_value = 'O';
  else if (chrom_index == 15) return_value = 'P';
  else if (chrom_index == 16) return_value = 'Q';
  else if (chrom_index == 17) return_value = 'R';
  else if (chrom_index == 18) return_value = 'S';
  else if (chrom_index == 19) return_value = 'T';
  else if (chrom_index == 20) return_value = 'U';
  else if (chrom_index == 21) return_value = 'V';
  else if (chrom_index == 22) return_value = 'W';
  else if (chrom_index == 23) return_value = 'X';
  else if (chrom_index == 24) return_value = 'Y';
  else if (chrom_index == 25) return_value = 'Z';
  else if (chrom_index == 26) return_value = 'P';




  if (return_value == '\0') {
    die("Invalid chromosome index (%d).\n", chrom_index);
  }

  // Use uppercase for the second copy of each chromosome.
  if (copy_index == 1) {
    return_value = (char)toupper(return_value);
  }
  return(return_value);
}

/*****************************************************************************
 * Print the coordinates of one atom in PDB format.
 *****************************************************************************/
//#define SCALE_FACTOR 1000
#define SCALE_FACTOR 100
// 100 for a sphere with radius 1, 1000 for a sphere with radius 10
// TODO: add it as a parameter
//static
void print_pdb_atom
  (FILE*     outfile,
   int       chrom_index,
   int       copy_index,
   BOOLEAN_T is_node,    // Is this a node or an edge atom?
   char*    atom_name,   // eg "N", "O", or "C"
   COORDS_T* my_coords)
{

  static int prev_chrom_index = -1;
  static int atom_index;
  if (chrom_index != prev_chrom_index) {
    atom_index = 1;
    prev_chrom_index = chrom_index;
  }

  // http://www.biochem.ucl.ac.uk/~roman/procheck/manual/manappb.html
  fprintf(outfile, "ATOM  ");              //  1- 6: Record ID
  fprintf(outfile, "%5d", atom_index);     //  7-11: Atom serial number
  fprintf(outfile, " ");                   //    12: Blank
  // if (is_node) {                           // 13-16: Atom name
  //   fprintf(outfile, "N   ");
  // } else {
  //   fprintf(outfile, "O   ");
  // }
  fprintf(outfile,"%s   ",atom_name);
  fprintf(outfile, " ");                   // 17-17: Alternative location code
  if (is_node) {                           // 18-20: 3-letter amino acid code
    fprintf(outfile, "NOD");
  } else {
    fprintf(outfile, "EDG");
  }
  fprintf(outfile, " ");                   //    21: Blank
  fprintf(outfile, "%c",                   //    22: Chain identifier code
	  get_chrom_id(chrom_index, copy_index));
  fprintf(outfile, "    ");                // 23-26: Residue sequence number
  fprintf(outfile, " ");                   //    27: Insertion code
  fprintf(outfile, "   ");                 // 28-30: Blank
  fprintf(outfile, "%8.3f%8.3f%8.3f",      // 31-54: Atom coordinates
	  (my_coords->x + 1.0) * SCALE_FACTOR,
	  (my_coords->y + 1.0) * SCALE_FACTOR,
	  (my_coords->z + 1.0) * SCALE_FACTOR);
  fprintf(outfile, "%6.2f", 1.0);          // 55-60: Occupancy value
  if (is_node) {
    fprintf(outfile, "%6.2f", 50.0);       // 61-66: B-value (thermal factor)
  } else {
    fprintf(outfile, "%6.2f", 75.0);       // 61-66: B-value (thermal factor)
  }
  fprintf(outfile, " ");                   //    67: Blank
  fprintf(outfile, "   ");                 // 68-70: Blank
  fprintf(outfile, "\n");

  atom_index++;
}



void print_1D_3D_atom
  (FILE*     outfile,
   int       chrom_index,
   int       copy_index,
   COORDS_T* my_coords,
   int       genomic_position)
{

  static int prev_chrom_index = -1;
  static int atom_index;
  if (chrom_index != prev_chrom_index) {
    atom_index = 1;
    prev_chrom_index = chrom_index;
  }

    fprintf (outfile, "%d\t%d\t", chrom_index+1, genomic_position);

    // NOTE: For some reason, in the submitted PDB, x and z are reversed!!!
    // I'm reversing them here too, just for consistency (in fact it doesn't matter)

    fprintf(outfile, "%.3f\t%.3f\t%.3f\n",      // 31-54: Atom coordinates
	  (my_coords->z + 1.0) * SCALE_FACTOR,
	  (my_coords->y + 1.0) * SCALE_FACTOR,
	  (my_coords->x + 1.0) * SCALE_FACTOR);

    atom_index++;
}




/*****************************************************************************
 * Print the coordinates of one locus in PDB format.
 *****************************************************************************/
void print_pdb_coords
  (FILE*     outfile,
   int       chrom_index,
   int       copy_index,
   COORDS_T* my_coords)
{
  print_pdb_atom(outfile, chrom_index, copy_index, TRUE, "N", my_coords);
}


/*****************************************************************************
 * Move a given point a specified distance toward a second point.
 *****************************************************************************/
//static
void move_toward
  (COORDS_T* coords1,
   COORDS_T* coords2,
   double     distance_to_move)
{
  double distance_between = compute_distance(coords1, coords2);
  coords1->x
    += (coords2->x - coords1->x) * (distance_to_move / distance_between);
  coords1->y
    += (coords2->y - coords1->y) * (distance_to_move / distance_between);
  coords1->z
    += (coords2->z - coords1->z) * (distance_to_move / distance_between);
}


void move_toward_1D
  (int &     from,
   int       to,
   COORDS_T* coords1,
   COORDS_T* coords2,
   double    distance_to_move)
{
  double distance_between = compute_distance(coords1, coords2);
  from = from + (to-from)*distance_to_move/distance_between;
}

/*****************************************************************************
 * Move a given point a portion of the way toward a second point.
 *****************************************************************************/
//static
void move_toward_portion
  (COORDS_T* coords1,
   COORDS_T* coords2,
   double     portion_to_move)  // fraction between 0 and 1
{
  coords1->x = portion_to_move * coords2->x + (1-portion_to_move)*coords1->x;
  coords1->y = portion_to_move * coords2->y + (1-portion_to_move)*coords1->y;
  coords1->z = portion_to_move * coords2->z + (1-portion_to_move)*coords1->z;
}

/*****************************************************************************
 * Print a series of points corresponding to an edge between two nodes.
 *****************************************************************************/
#define EDGE_LENGTH 0.01 // Distance between adjacent points on an edge.
//#define EDGE_LENGTH 0.0015 // Distance between adjacent points on an edge.
void print_pdb_edge
  (FILE*     outfile,
   int       chrom_index,
   int       copy_index,
   COORDS_T* coords1,
   COORDS_T* coords2)
{

  // Compute the distance between the two given points.
  double distance = compute_distance(coords1, coords2);

  // Make a copy of the first point.
  COORDS_T intermediate;
  intermediate.x = coords1->x;
  intermediate.y = coords1->y;
  intermediate.z = coords1->z;

  // while (distance > 0.0) {
  for (double t=0; t<=distance; t+=EDGE_LENGTH) {

    // Push the first point toward the second point.
    intermediate.x = coords1->x;
    intermediate.y = coords1->y;
    intermediate.z = coords1->z;
    move_toward(&intermediate, coords2, t);

    // Print it.
    print_pdb_atom(outfile, chrom_index, copy_index, FALSE, "O", &intermediate);

    //distance -= EDGE_LENGTH;
  }
}

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
   int       to_1D)
{

  // Compute the distance between the two given points.
  double distance = compute_distance(coords1, coords2);

  // Make a copy of the first point.
  COORDS_T intermediate;
  intermediate.x = coords1->x;
  intermediate.y = coords1->y;
  intermediate.z = coords1->z;

  // while (distance > 0.0) {
  for (double t=0; t<=distance; t+=EDGE_LENGTH) {

    // Push the first point toward the second point.
    intermediate.x = coords1->x;
    intermediate.y = coords1->y;
    intermediate.z = coords1->z;

    int genomic_position = from_1D;
    move_toward_1D (genomic_position, to_1D, &intermediate, coords2, t);
    move_toward(&intermediate, coords2, t);

    // Print it.
    //print_pdb_atom(outfile, chrom_index, copy_index, FALSE, "O", &intermediate);
    print_1D_3D_atom(outfile, chrom_index, copy_index, &intermediate, genomic_position);

    //distance -= EDGE_LENGTH;
  }
}

/*****************************************************************************
 * Print 1D coordinates and a series of points along a quadratic bezier curve with control points p0,p1,p2
 *****************************************************************************/
void print_1D_3D_quadratic (FILE* outfile, int chrom_index, int copy_index,
  COORDS_T* p0, COORDS_T* p1, COORDS_T* p2, int from_1D, int to_1D)
{
  COORDS_T q0,q1;

  for(double t=0; t<=1; t+=0.05) {
    copy_coords(p0,&q0);
    copy_coords(p1,&q1);
    move_toward_portion(&q0,p1,t);
    move_toward_portion(&q1,p2,t);
    move_toward_portion(&q0,&q1,t);

    int genomic_position = from_1D + (to_1D-from_1D)*t;
    print_1D_3D_atom(outfile, chrom_index, copy_index, &q0, genomic_position);
  }
}


/*****************************************************************************
 * Print a series of points along a quadratic bezier curve with control points p0,p1,p2
 *****************************************************************************/
void print_pdb_quadratic (FILE* outfile, int chrom_index, int copy_index,
  COORDS_T* p0, COORDS_T* p1, COORDS_T* p2, int has_centromere)
{
  COORDS_T q0,q1;

  for(double t = 0; t <= 1; t += 0.05) {
    copy_coords(p0, &q0);
    copy_coords(p1, &q1);
    move_toward_portion(&q0, p1, t);
    move_toward_portion(&q1, p2, t);
    move_toward_portion(&q0, &q1, t);

    if(fabs(t - 0.5) <= 0.001 && has_centromere)
    {
      print_pdb_atom(outfile, chrom_index, copy_index, FALSE, "C", &q0);
    }
    else
      print_pdb_atom(outfile, chrom_index, copy_index, FALSE, "O", &q0);
  }
}


/*****************************************************************************
 * MAIN
 *****************************************************************************/
#ifdef MAIN
#include <time.h>
VERBOSE_T verbosity = NORMAL_VERBOSE;

int main(int argc, char *argv[])
{
  char usage[100] = "USAGE: coords in|on txt|pdb <int>\n";
  if (argc != 4) {
    die(usage);
  }

  // Find out whether to pick points on or in the unit sphere.
  BOOLEAN_T in_sphere;
  if (strcmp(argv[1], "in") == 0) {
    in_sphere = TRUE;
  } else if (strcmp(argv[1], "on") == 0) {
    in_sphere = FALSE;
  } else {
    die("Invalid first option (%s).\n", argv[1]);
  }

  // Determine output format.
  BOOLEAN_T pdb_format;
  if (strcmp(argv[2], "txt") == 0) {
    pdb_format = FALSE;
  } else if (strcmp(argv[2], "pdb") == 0) {
    pdb_format = TRUE;
  } else {
    die("Invalid second option (%s).\n", argv[2]);
  }

  // How many points to pick.
  int num_points = atoi(argv[3]);

  // Initialize the random number generator.
  srand48(time(0));

  // Randomly select and print points.
  COORDS_T my_coords;
  int i_point;
  for (i_point = 0; i_point < num_points; i_point ++) {
    if (in_sphere) {
      randomize_coords(&my_coords);
    } else {
      choose_point_on_unit_sphere(&my_coords);
    }
    if (pdb_format) {
      print_pdb_coords(stdout, 0, 0, &my_coords);
    } else {
      printf("%6.3f %6.3f %6.3f\n", my_coords.x, my_coords.y, my_coords.z);
    }
  }
  return (0);
}/* Main coords */
#endif

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 2
 * End:
 */
