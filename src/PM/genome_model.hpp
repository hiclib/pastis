#ifndef __GENOME_MODEL_HPP__
#define __GENOME_MODEL_HPP__

// libraries used to convert the chromosome names in yeast
#include <map>
#include <string>
#include <iostream>
using namespace std;

#include "frequencies.hpp"
#include "utils.h"
#include "coords.h"


/*****************************************************************************
 * One or two copies of a given chromosome.
 *****************************************************************************/
typedef struct chrom {
  char        id[3];
  int         num_copies;
  int         num_bases;
  int         num_loci;
  COORDS_T*** loci;
} CHROM_T;


/*****************************************************************************
 * A linked list of interactions among pairs of loci.  Each
 * interaction is stored so that chrom1_index <= chrom2_index.  If the
 * two chromosome indices are the same, then locus1_index <
 * locus2_index.  It is an error to specify an interaction between
 * identical loci.
 *****************************************************************************/
typedef struct interaction {
  unsigned int chrom1_index;
  unsigned int locus1_index;
  unsigned int chrom2_index;
  unsigned int locus2_index;
  double freq;
  unsigned int count;       // the actual number of interactions connecting these 2 variables
  double optimal_dist; // the euclidean distance between these two loci,
                       // after the optimization step
  double  wish_dist;  // a wish distance
  unsigned int length; // Surface of the bin 
  struct interaction* next;
} INTERACTION_T;


class GENOME
{
    public:
        // Attributes
        int chrlen[NUM_CHROMS + 1];
        int centromeres[NUM_CHROMS];
        int organism;
        double n_total_interactions;
        int n_total_intra_interactions;
        int use_weights;


        GENOME();
        GENOME(GENOME * genome);
        ~GENOME();
        GENOME(int bp_per_locus, double min_dist, double max_dist,
               char *interactions_filename, int *chromosome,
               int add_75_rDNA, char *rDNA_interactions_file,
               int use_weights, char* unseen_interactions_filename,
               char * frequencies_distances_filename,
               char * structure_filename);
        GENOME(int bp_per_locus, double min_dist, double max_dist,
               char *interactions_filename, char* chromosome, int add_75_rDNA,
               char *rDNA_interactions_file, int use_weights,
               char* unseen_interactions_filename,
               char * frequencies_distances_filename);

        void read_chrlen(char * filename);
        void initialize_chrlen();
        void print_pdb_genome(FILE* outfile);
        void print_pdb_genome(char *pdb_filename);
        void print_1D_3D_genome(char *pdb_filename);
        int get_num_loci();
        int get_chromosome(int locus);
        int get_num_adjacent_loci();
        double get_min_dist() { return min_locus_distance; }
        double get_max_dist() { return max_locus_distance; }
        int get_num_chroms() { return num_chroms; }
        int get_num_loci(int c){ return chroms[c]->num_loci; }
        int get_num_interactions(){ return num_interactions; }
        int get_num_rDNA_interactions()  { return num_rDNA_interactions; }
        int get_num_unseen_interactions()  { return num_unseen_interactions; }
        COORDS_T * get_coords(int index);
        void set_coords(double *x);
        void set_coords(char * char_x);
        INTERACTION_T* get_first_interaction() { return interactions; }
        INTERACTION_T* get_first_rDNA_interaction() { return rDNA_interactions; }
        INTERACTION_T* get_first_unseen_interaction() { return unseen_interactions; }

        int get_x_index(int chrom_index, int locus_index);
        void save_interaction_adjacency_matrix(char *output_filename);

        void save_adjacency_matrix_for_diffusion(char *output_filename,
                                                 int operation_type);
        // if operation_type is 0, print binary
        // if operation_type is 1, print frequencies (sum of frequencies)
        // if operation_type is 2, print counts

        void save_optimal_distances(char *output_filename);
        void save_interaction_achievement(char *output_filename);
        void read_cplex_output(char *cplex_filename);
        void read_txt_input(char *txt_filename);
        void save_txt (char *txt_filename);
        int get_chrom_index (int chrom);
        // chrom is 1-16
        // return the index in our data structures. This depends on which chromosomes 
        // are wanted (chromosome)

    private:
        int bp_per_locus;
        unsigned short max_num_chrom;
        double min_locus_distance;
        double max_locus_distance;
        char *interactions_filename;
        char *rDNA_interactions_filename;
        int num_chroms;
        CHROM_T** chroms;
        int num_interactions;
        int num_rDNA_interactions;
        int num_unseen_interactions;
        INTERACTION_T* interactions;
        INTERACTION_T* rDNA_interactions;
        INTERACTION_T* unseen_interactions;
        int * wanted_chrom;
        Mapping mapping;


        //functions

        void new_haploid_yeast_genome();
        void new_haploid_yeast_genome(int *chromosome);
        void new_haploid_yeast_genome(int chromosome);
        void new_haploid_yeast_genome_chr3();
        void new_haploid_yeast_genome_chr4();
        void new_haploid_yeast_genome_chr34();
        int add_interactions(char *interactions_filename,
                             INTERACTION_T* &interactions,
                             int *chromosome);
        int get_matrix_index (int chrom, int locus);
        void compute_wish_distances(INTERACTION_T* &interactions,
                                    int use_weights);
        void set_wish_distances(INTERACTION_T* &interactions,
                                char * distances_filename);
        CHROM_T* new_chrom
            (char* id,
            int num_copies,
            int num_bases);

        INTERACTION_T* get_next_interaction
            (int       line_number,
            FILE*     interactions_file,
            int *chromosome);

        int chrom_id_to_index (string chrom_id);
        COORDS_T* get_locus_coords(int i_copy,
                                   int i_locus,
                                   CHROM_T* my_chrom);

        CHROM_T* get_nth_chrom (int chrom_index);
        INTERACTION_T* has_interaction(INTERACTION_T* query,
                                       INTERACTION_T* list);
        int add_rDNA;

};

#endif


