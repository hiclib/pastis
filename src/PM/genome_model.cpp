
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <float.h>

// libraries used to convert the chromosome names in yeast
#include <map>
#include <string>
#include <iostream>
using namespace std;

#include "genome_model.hpp"
#include "utils.h"
#include "coords.h"
#define MAX_LINE 50
#define BUDDINGYEAST 0
#define MALARIA 1
#define TMC 2
#define OTHER 3



GENOME::GENOME(int bp_per_locus, double min_dist, double max_dist,
               char *interactions_filename, int *chromosome,
               int add_rDNA, char *rDNA_interactions_filename,
               int use_weights, char* unseen_interactions_filename,
               char * frequencies_distances_filename,
               char * structure_filename)
{
    if(frequencies_distances_filename[0] == '\0'){
      this->mapping = Mapping();
    }else{
      this->mapping = Mapping(frequencies_distances_filename);
    }
    this->bp_per_locus = bp_per_locus;
    this->min_locus_distance = min_dist;
    this->max_locus_distance = max_dist;
    this->interactions_filename = interactions_filename;
    this->rDNA_interactions_filename = rDNA_interactions_filename;
    this->wanted_chrom = chromosome;
    this->n_total_interactions = 0;
    this->n_total_intra_interactions = 0;
    this->use_weights = use_weights;

    // Budding yeast: 0
    // Plasmodium: 1
    this->organism = OTHER;

    this->add_rDNA = 0;
    if(this->organism == BUDDINGYEAST){ 
      this->add_rDNA = 1;
    }

    // to add as argument given by the user
    long int seed = (long int)time(0);

    // Initialize the random number generator.
    my_srand(seed);
    num_chroms = 0;
    this->num_interactions = 0;
    num_rDNA_interactions = 0;
    interactions = NULL;
    rDNA_interactions = NULL;

    if(this->organism == OTHER){
      this->read_chrlen(structure_filename);

    } else {
      this->initialize_chrlen();
    }
    cout << this->max_num_chrom << endl;

    new_haploid_yeast_genome(chromosome);   // (chromosome);
    printf ("Adding interactions\n");
    this->num_interactions = add_interactions(interactions_filename,
                                        interactions,
                                        chromosome);
    printf ("Finished adding interactions\n");
    printf("Using weights %d \n", use_weights);
    if(use_weights == 5 || use_weights == 6 || use_weights == 7 ||
       use_weights == 8){
        compute_wish_distances(interactions, use_weights);
    }

    num_rDNA_interactions = 0;
    rDNA_interactions = NULL;
    if (strcmp(rDNA_interactions_filename,"") != 0)
    {
      num_rDNA_interactions = add_interactions(
                                rDNA_interactions_filename,
                                rDNA_interactions, chromosome);
    }

    num_unseen_interactions = 0;
    unseen_interactions = NULL;
    if (strcmp(unseen_interactions_filename,"") != 0)
    {
      num_unseen_interactions = add_interactions(
                                      unseen_interactions_filename,
                                      unseen_interactions, chromosome);
    }

}

GENOME::GENOME(GENOME * genome)
{
    this->bp_per_locus = genome->bp_per_locus;
    this->min_locus_distance = genome->min_locus_distance;
    this->max_locus_distance = genome->max_locus_distance;
    this->interactions_filename = genome->interactions_filename;
    this->rDNA_interactions_filename = genome->rDNA_interactions_filename;
    this->wanted_chrom = genome->wanted_chrom;
    this->add_rDNA = genome->add_rDNA;

    this->num_chroms = genome->num_chroms;
    this->num_interactions = genome->num_interactions;
    this->num_rDNA_interactions = genome->num_rDNA_interactions;
    this->interactions = genome->interactions;
    this->rDNA_interactions = genome->rDNA_interactions;
    this->unseen_interactions = genome->unseen_interactions;
    this->chroms = genome->chroms;

    this->initialize_chrlen();
}

GENOME::GENOME(int bp_per_locus, double min_dist, double max_dist,
               char *interactions_filename, char* chromosome, int add_rDNA,
               char *rDNA_interactions_filename, int use_weights,
               char* unseen_interactions_filename, char * frequencies_distances_filename)
{
    this->bp_per_locus = bp_per_locus;
    this->min_locus_distance = min_dist;
    this->max_locus_distance = max_dist;
    this->interactions_filename = interactions_filename;
    this->rDNA_interactions_filename = rDNA_interactions_filename;
    this->wanted_chrom = (int *) chromosome;
    this->add_rDNA = add_rDNA;
    this->n_total_interactions = 0;
    this->n_total_intra_interactions = 0;
    this->use_weights = use_weights;

    if(frequencies_distances_filename[0] == '\0'){
      this->mapping = Mapping();
    }else{
      this->mapping = Mapping(frequencies_distances_filename);
    }


    // to add as argument given by the user
    long int seed = (long int)time(0);

    // Initialize the random number generator.
    my_srand(seed);
    num_chroms = 0;
    this->num_interactions = 0;
    num_rDNA_interactions = 0;
    interactions = NULL;
    rDNA_interactions = NULL;

    this->initialize_chrlen();

    new_haploid_yeast_genome(this->wanted_chrom);   // (chromosome);
    printf ("Adding interactions\n");
    this->num_interactions = add_interactions(interactions_filename,
                          interactions, this->wanted_chrom);
    printf ("Finished adding interactions\n");
    if (use_weights == 5 || use_weights == 6 || use_weights == 7 || use_weights == 8)
        compute_wish_distances(interactions, use_weights);

    num_rDNA_interactions = 0;
    rDNA_interactions = NULL;
    if (strcmp(rDNA_interactions_filename,"") != 0)
    {
        num_rDNA_interactions = add_interactions(rDNA_interactions_filename,
                                                 rDNA_interactions, this->wanted_chrom);
    }

    num_unseen_interactions = 0;
    unseen_interactions = NULL;
    if (strcmp(unseen_interactions_filename,"") != 0)
    {
        num_unseen_interactions = add_interactions(unseen_interactions_filename,
                                                   unseen_interactions,
                                                   this->wanted_chrom);
    }

}


void GENOME::read_chrlen(char * filename){
  FILE * structure_file;
  if(open_file(filename, "r", TRUE, "chrlen", "chrlen", &structure_file) == 0){
      exit(1);
  }

  char one_line[MAX_LINE];
  unsigned int length;
  while(1){
    if(fgets(one_line, MAX_LINE, structure_file) == NULL){
      break;
    }
    int num_scanned = sscanf(one_line, "%d", &length);
    this->max_num_chrom += 1;
    this->chrlen[this->max_num_chrom] = length;
  }
}

void GENOME::initialize_chrlen(){
    if(this->organism == BUDDINGYEAST){
        this->max_num_chrom = 16;
        // this is budding yest
        // fill up the array of chromosome lengths
        this->chrlen[1] = 230208;
        this->chrlen[2] = 813178;
        this->chrlen[3] = 316617;
        this->chrlen[4] = 1531918;
        this->chrlen[5] = 576869;
        this->chrlen[6] = 270148;
        this->chrlen[7] = 1090947;
        this->chrlen[8] = 562643;
        this->chrlen[9] = 439885;
        this->chrlen[10] = 745745;
        this->chrlen[11] = 666454;
        if (add_rDNA){
            this->chrlen[12] = 1078175 + SIZE_RDNA * NUM_RDNA_TO_ADD;
        }else{
            this->chrlen[12] = 1078175;
        }
        this->chrlen[13] = 924429;
        this->chrlen[14] = 784333;
        this->chrlen[15] = 1091289;
        this->chrlen[16] = 948062;

        // initialize the array of centromere locations
        this->centromeres[0] =  151584;
        this->centromeres[1] =  238325;
        this->centromeres[2] =  114499;
        this->centromeres[3] =  449819;
        this->centromeres[4] =  152103;
        this->centromeres[5] =  148622;
        this->centromeres[6] =  497042;
        this->centromeres[7] =  105698;
        this->centromeres[8] =  355742;
        this->centromeres[9] =  436418;
        this->centromeres[10] =  439889;
        this->centromeres[11] = 150946;
        this->centromeres[12] =  268149;
        this->centromeres[13] =  628877;
        this->centromeres[14] =  326703;
        this->centromeres[15] =  556070;
    }else if(this->organism == MALARIA){
        this->max_num_chrom = 14;
        // this is Plasmodium
        this->chrlen[1] = 640851;
        this->chrlen[2] = 947102;
        this->chrlen[3] = 1067971;
        this->chrlen[4] = 1200490;
        this->chrlen[5] = 1343557;
        this->chrlen[6] = 1418242;
        this->chrlen[7] = 1445207;
        this->chrlen[8] = 1472805;
        this->chrlen[9] = 1541735;
        this->chrlen[10] = 1687656;
        this->chrlen[11] = 2038340;
        this->chrlen[12] = 2271494;
        this->chrlen[13] = 2925236;
        this->chrlen[14] = 3291936;

        this->centromeres[0] = 460341;
        this->centromeres[1] = 448856;
        this->centromeres[2] = 595821;
        this->centromeres[3] = 650161;
        this->centromeres[4] = 456666;
        this->centromeres[5] = 479881;
        this->centromeres[6] = 865996;
        this->centromeres[7] = 301321;
        this->centromeres[8] = 1243256;
        this->centromeres[9] = 936751;
        this->centromeres[10] = 833106;
        this->centromeres[11] = 1283721;
        this->centromeres[12] = 1169596;
        this->centromeres[13] = 1073081;

    }else{
        this->max_num_chrom = 26;

        this->chrlen[1] = 249250621;
        this->chrlen[2] = 243199373;
        this->chrlen[3] = 198022430;
        this->chrlen[4] = 191154276;
        this->chrlen[5] = 180915260;
        this->chrlen[6] = 171115067;
        this->chrlen[7] = 159138663;
        this->chrlen[8] = 146364022;
        this->chrlen[9] = 141213431;
        this->chrlen[10] = 135534747;
        this->chrlen[11] = 135006516;
        this->chrlen[12] = 133851895;
        this->chrlen[13] = 115169878;
        this->chrlen[14] = 107349540;
        this->chrlen[15] = 102531392;
        this->chrlen[16] = 90354753;
        this->chrlen[17] = 81195210;
        this->chrlen[18] = 78077248;
        this->chrlen[19] = 59128983;
        this->chrlen[20] = 63025520;
        this->chrlen[21] = 48129895;
        this->chrlen[22] = 51304566;
        this->chrlen[23] = 155270560;
        this->chrlen[24] = 59373566;
        this->chrlen[25] = 146364022;
        this->chrlen[25] = 146364022;
        this->chrlen[26] = 90000000 - 61100000;

        this->centromeres[0] = 460341;
        this->centromeres[1] = 448856;
        this->centromeres[2] = 595821;
        this->centromeres[3] = 650161;
        this->centromeres[4] = 456666;
        this->centromeres[5] = 479881;
        this->centromeres[6] = 865996;
        this->centromeres[7] = 301321;
        this->centromeres[8] = 1243256;
        this->centromeres[9] = 936751;
        this->centromeres[10] = 833106;
        this->centromeres[11] = 1283721;
        this->centromeres[12] = 1169596;
        this->centromeres[13] = 1073081;
        this->centromeres[14] = 460341;
        this->centromeres[15] = 448856;
        this->centromeres[16] = 595821;
        this->centromeres[17] = 650161;
        this->centromeres[18] = 456666;
        this->centromeres[19] = 479881;
        this->centromeres[20] = 865996;
        this->centromeres[21] = 301321;
        this->centromeres[22] = 1243256;
        this->centromeres[23] = 936751;
        this->centromeres[24] = 833106;
        this->centromeres[24] = 833106;
        this->centromeres[25] = 0;
    }

}

/*****************************************************************************
 * Allocate a new yeast genome, haploid
 *****************************************************************************/
void GENOME::new_haploid_yeast_genome(int *chromosome)
{
    int i;
    this->num_chroms = 0;

    for(i = 0; i < this->max_num_chrom; i++)
    {
        this->num_chroms += chromosome[i];
  }

  this->chroms = (CHROM_T**)mymalloc(sizeof(CHROM_T*) * this->num_chroms);
  cout << this->num_chroms << endl;
  int index = 0;
  for (i=0; i < this->max_num_chrom; i++)
  {
      if (chromosome[i])
      {
          char id[10];
          sprintf(id, "%d", i+1);
          this->chroms[index] = new_chrom (id, 1, chrlen[i+1]);
          index++;
      }
    }
    this->num_interactions = 0;
    this->interactions = NULL;
}


/*****************************************************************************
 * Allocate a new yeast genome, haploid
 *****************************************************************************/
void GENOME::new_haploid_yeast_genome()
{

  this->num_chroms = 16;
  this->chroms = (CHROM_T**)mymalloc(sizeof(CHROM_T*) * this->max_num_chrom);
  this->chroms[0] = new_chrom ("1", 1, chrlen[1]);
  this->chroms[1] = new_chrom ("2", 1, chrlen[2]);
  this->chroms[2] = new_chrom ("3", 1, chrlen[3]);
  this->chroms[3] = new_chrom ("4", 1, chrlen[4]);
  this->chroms[4] = new_chrom ("5", 1, chrlen[5]);
  this->chroms[5] = new_chrom ("6", 1, chrlen[6]);
  this->chroms[6] = new_chrom ("7", 1, chrlen[7]);
  this->chroms[7] = new_chrom ("8", 1, chrlen[8]);
  this->chroms[8] = new_chrom ("9", 1, chrlen[9]);
  this->chroms[9] = new_chrom ("10", 1, chrlen[10]);
  this->chroms[10] = new_chrom ("11", 1, chrlen[11]);
  this->chroms[11] = new_chrom ("12", 1, chrlen[12]);
  this->chroms[12] = new_chrom ("13", 1, chrlen[13]);
  this->chroms[13] = new_chrom ("14", 1, chrlen[14]);
  this->chroms[14] = new_chrom ("15", 1, chrlen[15]);
  this->chroms[15] = new_chrom ("16", 1, chrlen[16]);

  this->num_interactions = 0;
  this->interactions = NULL;

}

/*****************************************************************************
 * Allocate a new yeast genome, haploid
 *****************************************************************************/
void GENOME::new_haploid_yeast_genome (int chromosome)
{
  this->num_chroms = 1;
  this->chroms = (CHROM_T**)mymalloc(sizeof(CHROM_T*) * this->max_num_chrom);
  char chrnum[5];
  sprintf (chrnum, "%d", chromosome);
  this->chroms[0] = new_chrom (chrnum, 1, chrlen[chromosome]);
  this->num_interactions = 0;
  this->interactions = NULL;

}

/****************************************************************************
 *
 * Compute the wish distances
 * To get the exact numbers, I used the file 2009-10-09/convert_frequencies_into_distances.m
 *
 *****************************************************************************/
void GENOME::compute_wish_distances(INTERACTION_T* &interactions,
                                    int use_weights)
{
  INTERACTION_T *interaction = interactions;
  INTERACTION_T * previous_interaction = NULL;

  while (interaction != NULL)
  {
    double freq;
    if(use_weights == 5){
      // the sum of frequencies
      freq = interaction->freq;
    }else if (use_weights == 6){
      freq = ((1.0 * interaction->freq) / interaction->count);
    }else if(use_weights == 7){
      double div = ((double) this->n_total_interactions * (double) interaction->length);
      freq = (1e18 * (double) interaction->freq) / div;
    }else if(use_weights == 8){
      interaction->wish_dist = interaction->freq / 1000. ;
    }else{
        printf("Use_weights should be 5, 6 or 7 and is %d, ABORT!\n",
               use_weights);
        exit(1);
    }

    if(use_weights == 8){
      interaction = interaction->next;
    }else{
      if(this->mapping.get_wish_dist(freq) == NULL){
        // If the frequency is too small to have any meaning, discard the entry.

        // If we are at the beginning of the list
        if(previous_interaction == NULL){
          this->interactions = interaction->next;
          myfree(interaction);
          interaction = this->interactions;
        }else{
          previous_interaction->next = interaction->next;
          myfree(interaction);
          interaction = previous_interaction->next;
        }
        this->num_interactions--;
      }else{
        // those were nano-meters, so make sure I make them micro-meters
        interaction->wish_dist = (double) this->mapping.get_wish_dist(freq) / 1000.0;

        previous_interaction = interaction;
        interaction = interaction->next;
      }
    }
  }

}


/*****************************************************************************
 * Allocate a new chromosome.
 *****************************************************************************/
int GENOME::add_interactions(char *interactions_filename,
                             INTERACTION_T* &interactions,
                             int *chromosome)
{
    // Punt if no file was provided.
    if (interactions_filename == NULL)
    {
        return 0;
    }

    // Open the file for reading.
    FILE* interactions_file;
    if (open_file(interactions_filename, "r", TRUE, "interactions",
                    "interactions", &interactions_file) == 0)
    {
        exit(1);
    }

    // Build the linked list of interactions.
    INTERACTION_T* this_interaction;
    INTERACTION_T* found_interaction;
    int line_number = 0;
    while ((this_interaction = get_next_interaction(line_number + 1,
                                                    interactions_file, chromosome))
            != NULL)
            // if it's the same locus, get_next_interaction won't add it
    {
        // found_interaction = has_interaction(this_interaction, interactions);
        found_interaction = NULL;
        if(found_interaction != NULL){
            if(this->use_weights == 8){
              printf("There are two interactions for the same bead");
              exit(1);
            }
            found_interaction->freq += this_interaction->freq;
            found_interaction->count += this_interaction->count;
            myfree(this_interaction);
        }else{
            INTERACTION_T* next_interaction = this->interactions;
            this->interactions = this_interaction;
            this_interaction->next = next_interaction;
            this->num_interactions++;
        }
        line_number++;
    }
    fprintf(stdout, "Read %d interactions from %d lines in %s.\n",
            this->num_interactions, line_number, interactions_filename);

    fclose(interactions_file);
    return this->num_interactions;
}

/*****************************************************************************
 * Check whether a given interaction is in a linked list.
 * Mirela: Oct 17, 2009. If I find it, increase the frequency of the interaction that is already in the list
 * Mirela: Oct 22, 2009. Return a pointer to the interaction that has it, or nULL
 * Mirela: Oct 22, 2009. Don't increase the frequency here
 *****************************************************************************/
INTERACTION_T* GENOME::has_interaction(INTERACTION_T* query,
                                       INTERACTION_T* list){

    if (list == NULL or list == 0x0) {
        return(NULL);
    }

    if ((query->chrom1_index == list->chrom1_index) &&
        (query->locus1_index == list->locus1_index) &&
        (query->chrom2_index == list->chrom2_index) &&
        (query->locus2_index == list->locus2_index))
    {
        return(list);
    }

    if ((query->chrom1_index == list->chrom2_index) &&
        (query->locus1_index == list->locus2_index) &&
        (query->chrom2_index == list->chrom1_index) &&
        (query->locus2_index == list->locus1_index))
    {
        return(list);
    }

    if (list->next == NULL or list->next == 0x0) {
        return(NULL);
    }

    return(has_interaction(query, list->next));
}


/*****************************************************************************
 * Read one interaction from a file.  Return the new interaction object.
 *****************************************************************************/
#define MAX_LINE 1000 // Maximum number of characters in one line.
INTERACTION_T* GENOME::get_next_interaction
  (int       line_number,
   FILE*     interactions_file,
   int *chromosome)
{

    int chrom1;
    int chrom2;
    int locus1;
    int locus2;
    int dist = 0;   // we are not interested in this, but it is in the file
    float freq = 0;   // we are interested in frequency, and we'll use it as distance
    // TODO: the precision of float is not great, is up to e-38 to e38
    float pvalue;
    float qvalue;
    int chrom1_index;
    int chrom2_index;
    int locus1_index;
    int locus2_index;
    int length = 0;

    char one_line[MAX_LINE];

    while (1)
    {
        if (fgets(one_line, MAX_LINE, interactions_file) == NULL)
        {
            return(NULL);
        }

        int num_scanned = sscanf(one_line, "%d %d %d %d %d %f %e %e",
                                 &chrom1, &locus1, &chrom2, &locus2,
                                 &dist, &freq, &pvalue, &qvalue);

        if (num_scanned != 8) {
            die("Format error on line %d: <%s>\n", line_number, one_line);
        }

        this->n_total_interactions += freq;
        if(chrom1 == chrom2){
          this->n_total_intra_interactions += freq;
        }

        // chrom1 and chrom2 start from 1
        // if chrom1 is wanted, chromosome[chrom1-1] should be 1
        if (chromosome[chrom1 - 1] && chromosome[chrom2 - 1])
        {
            //printf ("Found the right chrom, break the loop\n");
            break;
        }

    }

    // do the rDNA adjustment
    if (add_rDNA)
    {
        if (chrom1 == 12)   // rDNA is on chrom 12
        {
            if (locus1 >= RDNA_POSITION)
                locus1 += SIZE_RDNA*NUM_RDNA_TO_ADD;
        }

        if (chrom2 == 12)   // rDNA is on chrom 12
        {
            if (locus2 >= RDNA_POSITION)
                locus2 += SIZE_RDNA*NUM_RDNA_TO_ADD;
        }
    }

    // Convert the chromosome ID to an index.
    chrom1_index = get_chrom_index(chrom1);   //chrom_id_to_index(chrom1_id);
    chrom2_index = get_chrom_index(chrom2);   //chrom_id_to_index(chrom2_id);

    // Reduce precision of locus coordinates.
    locus1_index = locus1 / bp_per_locus;
    locus2_index = locus2 / bp_per_locus;
    // Compute length of the bin. Most of the time, it will be equal to the
    // resolution. Only at the end of the chromosomes will it be different.
    length = bp_per_locus * bp_per_locus;
    if (chrom1_index == chrom2_index && locus1_index == locus2_index)
    {
        // same locus, ignore
        return get_next_interaction(line_number + 1,
                                    interactions_file,
                                    chromosome);
    }

    INTERACTION_T* return_value
        = (INTERACTION_T*) mymalloc(sizeof(INTERACTION_T));

    return_value->chrom1_index = chrom1_index;
    return_value->chrom2_index = chrom2_index;
    return_value->locus1_index = locus1_index;
    return_value->locus2_index = locus2_index;
    return_value->freq = freq;
    return_value->length = length;
    return_value->next = NULL;

    if (pvalue < 1e-10){
      pvalue = 1e-10;
    }

    if(qvalue < 1e-10){
      qvalue = 1e-10;
    }

    return_value->count = 1;

    return(return_value);
}

/**
 * get chromosome index.
 *
 * Returns the index in the data structures, or -1 if not in the
 * data structure.
 **/
int GENOME::get_chrom_index(int chrom)
{
    int index = 0;
    for(int i=1; i < chrom; i++)
    {
      if(this->wanted_chrom[i - 1])
        index++;
    }

    if(this->wanted_chrom[chrom - 1]){
      return index;
    }else{
      return -1;
    }
}

/*****************************************************************************
 * Retrieve a pointer to a specified locus, with bounds checking.
 *
 * Print an error and return NULL if illegal.
 *****************************************************************************/
COORDS_T* GENOME::get_locus_coords
  (int i_copy,
   int i_locus,
   CHROM_T* my_chrom)
{
  if ((i_copy < 0) || (i_copy >= my_chrom->num_copies)) {
    fprintf(stderr,
            "Invalid copy index (%d) for chromosome %s with %d copies.\n",
            i_copy, my_chrom->id, my_chrom->num_copies);
    return(NULL);
  }
  if ((i_locus < 0) || (i_locus >= my_chrom->num_loci)) {
    fprintf(stderr,
            "Invalid locus index (%d) for chromosome %s with %d loci.\n",
            i_locus, my_chrom->id, my_chrom->num_loci);
    return(NULL);
  }
  return(my_chrom->loci[i_copy][i_locus]);
}

/*****************************************************************************
 * Retrieve a pointer to the nth chromosome.  Do bounds checking.
 *
 * Print an error and return NULL if illegal.
 *****************************************************************************/
CHROM_T* GENOME::get_nth_chrom
  (int chrom_index)
{
  if ((chrom_index < 0) || (chrom_index >= this->num_chroms)) {
    fprintf(stderr, "Invalid chromosome index (%d, %d).\n",
            chrom_index, this->num_chroms);
    return(NULL);
  }
  return(chroms[chrom_index]);
}

/*****************************************************************************
 * Read one interaction from a file.  Return the new interaction object.
 *****************************************************************************/
int GENOME::chrom_id_to_index
  (string chrom_id)
{
  int return_value = atoi(chrom_id.c_str());
printf ("Chrom id: %s %d\n", chrom_id.c_str(), return_value);
  if (return_value == 0) {
    if (chrom_id=="X") {
      return_value = 23;
    } else if (chrom_id == "Y") {
      return_value = 24;
    }
  }
  return_value -= 1;  // Indices are indexed from zero.
  if ((return_value < 0) || (return_value >= 24)) {
    die("Invalid chromosome ID (%s).\n", chrom_id.c_str());
  }
  return(return_value);
}

/*****************************************************************************
 * Allocate a new chromosome.
 *****************************************************************************/
CHROM_T* GENOME::new_chrom
  (char*  id,
   int    num_copies,
   int    num_bases)
{
  CHROM_T* return_value;

  //  fprintf(stderr, "Creating %d copies of chromosome %s.\n", num_copies, id);

  // Allocate the chromosome.
  return_value = (CHROM_T*)mymalloc(sizeof(CHROM_T));

  // Copy the chromosome ID.
  strncpy(return_value->id, id, 3);

  // Store the number of copies.
  return_value->num_copies = num_copies;

  // Compute and store the resolution, number of bases and number of loci.
  return_value->num_bases = num_bases;

  return_value->num_loci = (num_bases / bp_per_locus) + 1;
  return_value->loci = (COORDS_T***)mymalloc(sizeof(COORDS_T**) * num_copies);
  int i_copy = 0;
  for (i_copy = 0; i_copy < num_copies; i_copy++) {
    return_value->loci[i_copy]
      = (COORDS_T**)mymalloc(sizeof(COORDS_T*) * return_value->num_loci);

    // Put the initial locus at a random location.
    return_value->loci[i_copy][0] = new_coords();
    randomize_coords(return_value->loci[i_copy][0]);

    // Place subsequent loci near the previous one.
    int i_locus;
    for (i_locus = 1; i_locus < return_value->num_loci; i_locus++) {
      return_value->loci[i_copy][i_locus] = new_coords();
      pick_nearby_coords(max_locus_distance,
                         return_value->loci[i_copy][i_locus-1],
                         return_value->loci[i_copy][i_locus]);
    }
  }

  return(return_value);
}

/*****************************************************************************
 * Print a genome in PDB format, to a specified output file
 *****************************************************************************/
void GENOME::print_pdb_genome (char *pdb_filename)
{
    FILE* pdb_file;
    if (open_file(pdb_filename, "w", TRUE, "PDB", "PDB", &pdb_file) == 0) {
        exit(1);
    }
    print_pdb_genome (pdb_file);
    fclose (pdb_file);
}

/*****************************************************************************
 * Print a genome in PDB format, to a file already opened, or to stdout.
 *****************************************************************************/
#define PRINT_BEZIER 1
//#define PRINT_LINEAR 1

#ifdef PRINT_LINEAR
void GENOME::print_pdb_genome (FILE* outfile)
{
  int i_atom = 0;

  int i_chrom;
  for (i_chrom = 0; i_chrom < num_chroms; i_chrom++) {
    CHROM_T* this_chrom = chroms[i_chrom];

    int i_copy;
    for (i_copy = 0; i_copy < this_chrom->num_copies; i_copy++) {

      int i_locus;
      for (i_locus = 0; i_locus < this_chrom->num_loci; i_locus++) {
        if (i_locus != 0) {

          print_pdb_edge(outfile, i_chrom, i_copy,
                         this_chrom->loci[i_copy][i_locus-1],
                         this_chrom->loci[i_copy][i_locus]);
          i_atom++;
        }

        print_pdb_coords(outfile, i_chrom, i_copy,
                         this_chrom->loci[i_copy][i_locus]);
        i_atom++;
      }
    }
  }
}


void GENOME::print_1D_3D_genome (char *filename)
{
    FILE* outfile;
    if (open_file(filename, "w", TRUE, "1Dto3D", "1Dto3D", &outfile) == 0) {
        exit(1);
    }
    fprintf (outfile, "chrom\tlocus\t3D_x\t3D_y\t3D_z\n");


  int i_atom = 0;

  int i_chrom;
  for (i_chrom = 0; i_chrom < num_chroms; i_chrom++) {
    CHROM_T* this_chrom = chroms[i_chrom];

    int i_copy;
    for (i_copy = 0; i_copy < this_chrom->num_copies; i_copy++) {

      int i_locus;
      for (i_locus = 0; i_locus < this_chrom->num_loci; i_locus++) {
		int from_1D = (i_locus-1)*bp_per_locus + bp_per_locus/2;
		int to_1D = (i_locus)*bp_per_locus + bp_per_locus/2;
        if (i_locus != 0) {
          print_1D_3D_edge(outfile, i_chrom, i_copy,
                         this_chrom->loci[i_copy][i_locus-1],
                         this_chrom->loci[i_copy][i_locus], from_1D, to_1D);
          i_atom++;
        }

        print_1D_3D_atom(outfile, i_chrom, i_copy,
                         this_chrom->loci[i_copy][i_locus], i_locus*bp_per_locus+1);
        i_atom++;
      }
    }
  }
  fclose (outfile);
}
#endif

#ifdef PRINT_ELBOWS
#define ELBOW_SIZE 0.1
void GENOME::print_pdb_genome (FILE* outfile)
{
  int i_atom = 0;

  int i_chrom;
  for (i_chrom = 0; i_chrom < num_chroms; i_chrom++) {
    CHROM_T* this_chrom = chroms[i_chrom];

    int i_copy;
    for (i_copy = 0; i_copy < this_chrom->num_copies; i_copy++) {

      int i_locus;
      for (i_locus = 0; i_locus < this_chrom->num_loci; i_locus++) {
        if (i_locus != 0) {
          COORDS_T p0,p1,p2;
          copy_coords(this_chrom->loci[i_copy][i_locus-1],&p1);
          copy_coords(this_chrom->loci[i_copy][i_locus],&p2);
          if(i_locus!=1) {
            copy_coords(this_chrom->loci[i_copy][i_locus-1],&p0);
            move_toward(&p1, this_chrom->loci[i_copy][i_locus], ELBOW_SIZE);
            move_toward(&p0,this_chrom->loci[i_copy][i_locus-2], ELBOW_SIZE);
            print_pdb_edge(outfile,i_chrom,i_copy,&p0,&p1);
          }
          if(i_locus!=this_chrom->num_loci-1)
            move_toward(&p2, this_chrom->loci[i_copy][i_locus-1], ELBOW_SIZE);
          print_pdb_edge(outfile, i_chrom, i_copy,&p1,&p2);
                       //  this_chrom->loci[i_copy][i_locus-1],
                       //  this_chrom->loci[i_copy][i_locus]);
          i_atom++;
        }

        // print_pdb_coords(outfile, i_chrom, i_copy,
        //                 this_chrom->loci[i_copy][i_locus]);
        i_atom++;
      }
    }
  }
}
#endif

#ifdef PRINT_BEZIER
void GENOME::print_pdb_genome (FILE* outfile)
{
  for (int i_chrom = 0; i_chrom < num_chroms; i_chrom++) {

    CHROM_T* this_chrom = chroms[i_chrom];

    for (int i_copy = 0; i_copy < this_chrom->num_copies; i_copy++) {
      for (int i_locus = 0; i_locus < this_chrom->num_loci; i_locus++) {

        COORDS_T p0,p1,p2;
        if (i_locus == 0) {   // if it's the first locus
            copy_coords(this_chrom->loci[i_copy][0], &p0);
            copy_coords(this_chrom->loci[i_copy][1], &p1);
            move_toward_portion(&p1, &p0, 0.5);  // move p1 to the middle between p0 and p1
            move_toward(&p0, &p1, 0.01);  // move p0 a little bit towards the new (middle) p1
            print_pdb_edge(outfile, i_chrom, i_copy, &p0, &p1);
        }
        else if(i_locus == this_chrom->num_loci - 1) {      // if it's the last locus
            copy_coords(this_chrom->loci[i_copy][this_chrom->num_loci - 2], &p0);
            copy_coords(this_chrom->loci[i_copy][this_chrom->num_loci - 1], &p1);
            move_toward_portion(&p0, &p1, 0.5);
            move_toward(&p1, &p0, 0.01);
            print_pdb_edge(outfile, i_chrom, i_copy, &p0, &p1);
        }
        else {      // a locus other than the first or last
          // compute Bezier control points
          copy_coords(this_chrom->loci[i_copy][i_locus], &p1);
          copy_coords(this_chrom->loci[i_copy][i_locus - 1], &p0);
          move_toward_portion(&p0, &p1, 0.5);
          copy_coords(this_chrom->loci[i_copy][i_locus + 1], &p2);
          move_toward_portion(&p2, &p1, 0.5);

          if (i_locus == this->centromeres[i_chrom] / this->bp_per_locus)
          {
            //printf ("Found centromere for chr %d\n", i_chrom+1);
            print_pdb_quadratic(outfile, i_chrom, i_copy, &p0, &p1, &p2, 1);
          }
          else
            print_pdb_quadratic(outfile, i_chrom, i_copy, &p0, &p1, &p2, 0);
        }

        if (i_locus == 0 || i_locus==this_chrom->num_loci - 1) {
            print_pdb_coords(outfile, i_chrom, i_copy,
                         this_chrom->loci[i_copy][i_locus]);
        }

      }
    }
  }
}


void GENOME::print_1D_3D_genome (char *filename)
// this performs exactly the same steps as print_pdb_genome, except it doesn't print it in pdb format,
//  but in a tab-delimited format which has the 1D cordinates too
{
    FILE* outfile;
    if (open_file(filename, "w", TRUE, "1Dto3D", "1Dto3D", &outfile) == 0) {
        exit(1);
    }
    fprintf (outfile, "chrom\tlocus\t3D_x\t3D_y\t3D_z\n");

    for (int i_chrom = 0; i_chrom < num_chroms; i_chrom++) {
        CHROM_T* this_chrom = chroms[i_chrom];
        for (int i_copy = 0; i_copy < this_chrom->num_copies; i_copy++) {
          for (int i_locus = 0; i_locus < this_chrom->num_loci; i_locus++) {
            COORDS_T p0,p1,p2;
            printf ("Chrom %d, locus %d, genpos %d - %d\n",
                i_chrom+1, i_locus+1, i_locus*bp_per_locus+1, (i_locus+1)*bp_per_locus);

            if (i_locus==0) {   // if it's the first locus
                // NOTE: this one should be here, not at the end of the if, to be printed before the first edge
                print_1D_3D_atom(outfile, i_chrom, i_copy,
                         this_chrom->loci[i_copy][i_locus], i_locus*bp_per_locus+1);

                copy_coords(this_chrom->loci[i_copy][0],&p0);
                copy_coords(this_chrom->loci[i_copy][1],&p1);
                move_toward_portion(&p1,&p0,0.5);  // move p1 to the middle between p0 and p1

                // do this before modifying p0
                int from_1D = i_locus*bp_per_locus+1;
                int to_1D = i_locus*bp_per_locus + bp_per_locus/2;
                move_toward_1D (from_1D, to_1D, &p0, &p1, 0.01);

                move_toward(&p0,&p1,0.01);  // move p0 a little bit towards the new (middle) p1
                // calculate the new genomic positions for p0 and p1
                print_1D_3D_edge(outfile,i_chrom,i_copy,&p0,&p1, from_1D, to_1D);
            }
            else if(i_locus==this_chrom->num_loci-1) {      // if it's the last locus
                copy_coords(this_chrom->loci[i_copy][this_chrom->num_loci-2],&p0);
                copy_coords(this_chrom->loci[i_copy][this_chrom->num_loci-1],&p1);
                move_toward_portion(&p0,&p1,0.5);

                int from_1D = (i_locus-1)*bp_per_locus + bp_per_locus/2;
                int to_1D = i_locus*bp_per_locus;
                move_toward_1D (from_1D, to_1D, &p1,&p0,0.01);
                move_toward(&p1,&p0,0.01);
                print_1D_3D_edge(outfile,i_chrom,i_copy,&p0,&p1,from_1D, to_1D);

                // print the last point (used to be after the if)

                print_1D_3D_atom(outfile, i_chrom, i_copy,
                         this_chrom->loci[i_copy][i_locus], i_locus*bp_per_locus);
            }
            else {      // a locus other than the first or last
              // compute Bezier control points
              copy_coords(this_chrom->loci[i_copy][i_locus],&p1);
              copy_coords(this_chrom->loci[i_copy][i_locus-1],&p0);
              int from_1D = (i_locus-1)*bp_per_locus + bp_per_locus/2;
              int to_1D = (i_locus)*bp_per_locus + bp_per_locus/2;

              move_toward_portion(&p0,&p1,0.5);
              copy_coords(this_chrom->loci[i_copy][i_locus+1],&p2);
              move_toward_portion(&p2,&p1,0.5);

              print_1D_3D_quadratic(outfile,i_chrom,i_copy,&p0,&p1,&p2, from_1D, to_1D);

            }

          }
        }
    }

    fclose (outfile);
}


#endif

void GENOME::set_coords (double *x)
{
    int i = 0;
    for (int i_chrom = 0; i_chrom < num_chroms; i_chrom++)
    {
        CHROM_T* this_chrom = chroms[i_chrom];
        for(int i_copy = 0; i_copy < this_chrom->num_copies; i_copy++)
        {
            for(int i_locus = 0; i_locus < this_chrom->num_loci; i_locus++)
            {
                copy_coords_xyz ( x[i++], x[i++], x[i++],
                    this_chrom->loci[i_copy][i_locus]);
            }
        }
    }
}

/**
 * From an array of coordinates (double, passed as char for the python
 * bindings), sets the coordinates of the loci.
 *
 * The array is of the form: x_1, y_1, z_1, x_2, y_2, z_2...
 *
 **/
void GENOME::set_coords(char * char_x)
{
    double *x = (double *) char_x;
    int i = 0;
    int i_chrom;
    for (i_chrom = 0; i_chrom < num_chroms; i_chrom++)
    {
        CHROM_T* this_chrom = chroms[i_chrom];
        int i_copy;
        for (i_copy = 0; i_copy < this_chrom->num_copies; i_copy++)
        {
            int i_locus;
            for (i_locus = 0; i_locus < this_chrom->num_loci; i_locus++)
            {
                copy_coords_xyz(x[i++], x[i++], x[i++],
                                this_chrom->loci[i_copy][i_locus]);
            }
        }
    }
}



int GENOME::get_x_index (int chrom_index, int locus_index)
{
    // TODO: provide the chromosome copy too!
    //printf ("Chrom index: %d\n", chrom_index);
    CHROM_T* this_chrom = chroms[chrom_index];
    if (this_chrom->num_copies > 1)
    {
        printf ("More than 1 chromosome copies, not implemented!\n");
        exit(1);
    }

    int index = 0;
    for (int i_chrom = 0; i_chrom < chrom_index; i_chrom++)
    {
        CHROM_T* this_chrom = chroms[i_chrom];
        index += this_chrom->num_copies * this_chrom->num_loci;
    }
    index += locus_index;

    // multiply by 3 because there are 3 coordinates x, y and z
    index *= 3;

    return index;
}


/*****************************************************************************
 * Get the x, y and z coordinates
 * TODO: THis is stupidely slow, to redo, maybe use static
 *****************************************************************************/
COORDS_T * GENOME::get_coords (int index)
{
    int index_sofar = 0;
    for (int i_chrom = 0; i_chrom < num_chroms; i_chrom++)
    {
        CHROM_T* this_chrom = chroms[i_chrom];
        for (int i_copy = 0; i_copy < this_chrom->num_copies; i_copy++)
        {
            for (int i_locus = 0; i_locus < this_chrom->num_loci; i_locus++)
            {
                if (index == index_sofar)   // this is my locus
                {
                    return this_chrom->loci[i_copy][i_locus];
                }
                index_sofar++;
            }
        }
    }
    return NULL;
}


/*****************************************************************************
 *
 *****************************************************************************/
int GENOME::get_num_loci()
{
    int num_loci = 0;
    for (int i_chrom = 0; i_chrom < num_chroms; i_chrom++)
    {
        CHROM_T* this_chrom = chroms[i_chrom];
        num_loci += this_chrom->num_loci * this_chrom->num_copies;
    }
    return num_loci;
}

/*****************************************************************************
 *
 *****************************************************************************/
int GENOME::get_chromosome(int locus)
{
    int num_loci = 0;
    for (int i_chrom = 0; i_chrom < num_chroms; i_chrom++)
    {
        CHROM_T* this_chrom = chroms[i_chrom];
        num_loci += this_chrom->num_loci * this_chrom->num_copies;
        if (locus <= num_loci)  return i_chrom;
    }
    return -1;
}

/*****************************************************************************
 *
 *****************************************************************************/
int GENOME::get_num_adjacent_loci()
{
    int num_loci = 0;
    for (int i_chrom = 0; i_chrom < num_chroms; i_chrom++)
    {
        CHROM_T* this_chrom = chroms[i_chrom];
        num_loci += (this_chrom->num_loci-1) * this_chrom->num_copies;
    }
    return num_loci;
}

/*****************************************************************************
 * Return the index that will be used in the binary matrix.
 * The index will have loci for chrom 1, loci for chrom 2 etc
 *****************************************************************************/
int GENOME::get_matrix_index (int chrom, int locus)
{
    int index = 0;
    for (int j_chrom = 0; j_chrom < chrom; j_chrom++)
        index += chroms[j_chrom]->num_loci;
    index += locus;
    return index;
}

/*****************************************************************************
 *
 *****************************************************************************/
void GENOME::save_adjacency_matrix_for_diffusion (char *output_filename, int operation_type)
// if operation_type is 0, print binary
// if operation_type is 1, print frequencies (sum of frequencies)
// if operation_type is 2, print counts
{
    FILE* output_file;
    if (open_file(output_filename, "w", TRUE,
                  "Interaction adjacency", "Interaction adjacency",
                  &output_file) == 0) {
        exit(1);
    }
    // TODO: what happens if there are several copies of each chromosome?

    // firt print a row with headers
    fprintf (output_file, "Labels");
    for (int j_chrom = 0; j_chrom < num_chroms; j_chrom++)
    {
        for (int j_locus = 0; j_locus < chroms[j_chrom]->num_loci; j_locus++)
        {
            fprintf (output_file, "\tchr%d_locus%d", j_chrom+1, j_locus+1);
        }
//         if (j_chrom < num_chroms-1)
//             fprintf (output_file, "\t\t\t");
    }
    fprintf (output_file, "\n");

    int matrix_size = 0;
    for (int i_chrom = 0; i_chrom < num_chroms; i_chrom++)
    {
        matrix_size += chroms[i_chrom]->num_loci;
    }

    // I'll store 1 and 0 in this matrix, so that I don't have to traverse the interactions linked list every time, but I just traverse it once
    int **matrix;
    matrix = new int*[matrix_size];
    for (int i=0; i < matrix_size; i++)
    {
        matrix[i] = new int[matrix_size];
        for (int j=0; j < matrix_size; j++)
            matrix[i][j] = 0;
    }
    INTERACTION_T *myint = interactions;
    while (myint != NULL)
    {
        int index1 = get_matrix_index (myint->chrom1_index, myint->locus1_index);
        int index2 = get_matrix_index (myint->chrom2_index, myint->locus2_index);
        //printf ("Chrom %d, locus %d, index %d\n", myint->chrom1_index, myint->locus1_index, index1);
        //printf ("Chrom %d, locus %d, index %d\n", myint->chrom2_index, myint->locus2_index, index2);
        if (operation_type == BINARY)    // binary
        {
            matrix[index1][index2] = 1;
            matrix[index2][index1] = 1;
        }
        else if (operation_type == FREQ)
        {
            matrix[index1][index2] = myint->freq;
            matrix[index2][index1] = myint->freq;
        }
        else if (operation_type == COUNT)
        {
            matrix[index1][index2] = myint->count;
            matrix[index2][index1] = myint->count;
        }
        myint = myint->next;
    }

    //printf ("Num chroms: %d\n", num_chroms);
    for (int i_chrom = 0; i_chrom < num_chroms; i_chrom++)
    {
        //printf ("Chrom %d, num loci %d\n", i_chrom, chroms[i_chrom]->num_loci);
        for (int i_locus = 0; i_locus < chroms[i_chrom]->num_loci; i_locus++)
        {
            fprintf (output_file, "chr%d_locus%d", i_chrom+1, i_locus+1);
            for (int j_chrom = 0; j_chrom < num_chroms; j_chrom++)
            {
                for (int j_locus = 0; j_locus < chroms[j_chrom]->num_loci; j_locus++)
                {
                    int index1 = get_matrix_index (i_chrom, i_locus);
                    int index2 = get_matrix_index (j_chrom, j_locus);

                    // make sure the adjacent loci along the same chromosome have value 1
                    // not sure I should do this, for now I'll only do it for binary
                    if (operation_type == BINARY)
                    {
                        if (i_chrom == j_chrom && (i_locus+1 == j_locus || i_locus == j_locus+1))
                            matrix[index1][index2] = 1;
                    }
                    else if (operation_type == COUNT)
                    {
                        if (i_chrom == j_chrom && (i_locus+1 == j_locus || i_locus == j_locus+1) &&
                            matrix[index1][index2] == 0)
                            matrix[index1][index2] = 1;
                    }

                    fprintf (output_file, "\t%d", matrix[index1][index2]);
                }
/*                // finished this chromosome, add empty entries to show grey lines in the picture
                if (j_chrom < num_chroms-1)
                    fprintf (output_file, "\t\t\t");  */
            }
            fprintf (output_file, "\n");
        }
/*        // insert 3 empty lines
        for (int k=1; k <= 3; k++)
        {
            if (i_chrom < num_chroms-1)
            {
                //fprintf (output_file, "empty");
                for (int j_chrom = 0; j_chrom < num_chroms; j_chrom++)
                {
                    for (int j_locus = 0; j_locus < chroms[j_chrom]->num_loci; j_locus++)
                    {
                        fprintf (output_file, "\t");
                    }
                    // finished this chromosome, add empty entries to show grey lines in the picture
                    if (j_chrom < num_chroms-1)
                        fprintf (output_file, "\t\t\t");
                }
                fprintf (output_file, "\n");
            }
        } */
    }
    fclose (output_file);
    for (int i=0; i < matrix_size; i++)
        delete [] matrix[i];
    delete [] matrix;
}


/*****************************************************************************
 *
 *****************************************************************************/
void GENOME::save_interaction_adjacency_matrix (char *output_filename)
{
    FILE* output_file;
    if (open_file(output_filename, "w", TRUE, "Interaction adjacency", "Interaction adjacency", &output_file) == 0) {
        exit(1);
    }
    // TODO: what happens if there are several copies of each chromosome?

    for (int i_chrom = 0; i_chrom < num_chroms; i_chrom++)
    {
        for (int i_locus = 0; i_locus < chroms[i_chrom]->num_loci; i_locus++)
        {
            for (int j_chrom = 0; j_chrom < num_chroms; j_chrom++)
            {
                for (int j_locus = 0; j_locus < chroms[j_chrom]->num_loci; j_locus++)
                {
                    // get the interaction score (or just 0/1 values)
                    INTERACTION_T myint;
                    myint.chrom1_index = i_chrom;
                    myint.locus1_index = i_locus;
                    myint.chrom2_index = j_chrom;
                    myint.locus2_index = j_locus;
                    if (has_interaction (&myint, interactions))
                        fprintf (output_file, "1 ");
                    else
                        fprintf (output_file, "0 ");
                }
                if (j_chrom != num_chroms-1)    fprintf (output_file, "0.3 ");
            }
            fprintf (output_file, "\n");
        }
        // add a row of 0.5 between chromosomes, for visualization purposes
        if (i_chrom != num_chroms-1)
        {
            for (int j_chrom = 0; j_chrom < num_chroms; j_chrom++)
                for (int j_locus = 0; j_locus < chroms[j_chrom]->num_loci; j_locus++)
                    fprintf (output_file, "0.3 ");
            fprintf (output_file, "0.3\n");
        }
    }
    fclose (output_file);
}

/*****************************************************************************
 *
 *****************************************************************************/

void GENOME::save_interaction_achievement (char *output_filename)
{
    FILE* output_file;
    if (open_file(output_filename, "w", TRUE, "Interaction achievement", "Interaction achievement", &output_file) == 0) {
        exit(1);
    }
    // TODO: what happens if there are several copies of each chromosome?

    for (int i_chrom = 0; i_chrom < num_chroms; i_chrom++)
    {
        for (int i_locus = 0; i_locus < chroms[i_chrom]->num_loci; i_locus++)
        {
            for (int j_chrom = 0; j_chrom < num_chroms; j_chrom++)
            {
                for (int j_locus = 0; j_locus < chroms[j_chrom]->num_loci; j_locus++)
                {
                    // get the interaction score (or just 0/1 values)
                    INTERACTION_T myint;
                    myint.chrom1_index = i_chrom;
                    myint.locus1_index = i_locus;
                    myint.chrom2_index = j_chrom;
                    myint.locus2_index = j_locus;

                    COORDS_T *coord1 = chroms[i_chrom]->loci[0][i_locus];
                    COORDS_T *coord2 = chroms[j_chrom]->loci[0][j_locus];
                    double dist = compute_distance (coord1, coord2);

                    if (has_interaction (&myint, interactions))
                    {
                        fprintf (output_file, "%.4lf ", dist);
                    }
                    else
                        fprintf (output_file, "0.01 ");
                }
                if (j_chrom != num_chroms-1)    fprintf (output_file, "0.012 ");
            }
            fprintf (output_file, "\n");
        }
        // add a row of 0.5 between chromosomes, for visualization purposes
        if (i_chrom != num_chroms-1)
        {
            for (int j_chrom = 0; j_chrom < num_chroms; j_chrom++)
                for (int j_locus = 0; j_locus < chroms[j_chrom]->num_loci; j_locus++)
                    fprintf (output_file, "0.012 ");
            fprintf (output_file, "0.012\n");
        }
    }
    fclose (output_file);
}



/*****************************************************************************
 *
 *****************************************************************************/
void GENOME::save_optimal_distances (char *output_filename)
{
    FILE* output_file;
    if (open_file(output_filename, "w", TRUE, "Optimal distances", "Optimal distances", &output_file) == 0) {
        exit(1);
    }
    // TODO: what happens if there are several copies of each chromosome?

    for (int i_chrom = 0; i_chrom < num_chroms; i_chrom++)
    {
        for (int i_locus = 0; i_locus < chroms[i_chrom]->num_loci; i_locus++)
        {
            for (int j_chrom = 0; j_chrom < num_chroms; j_chrom++)
            {
                for (int j_locus = 0; j_locus < chroms[j_chrom]->num_loci; j_locus++)
                {
                    COORDS_T *coord1 = chroms[i_chrom]->loci[0][i_locus];
                    COORDS_T *coord2 = chroms[j_chrom]->loci[0][j_locus];

                    double dist = compute_distance (coord1, coord2);
                    // only display the closest interactions
                    if (1-dist >= 0.9)  fprintf (output_file, "%.4lf ", 1-dist);
                    else                fprintf (output_file, "0.8 ");
                }
                if (j_chrom != num_chroms-1)    fprintf (output_file, "0.87 ");
            }
            fprintf (output_file, "\n");
        }

        // add a row of 0.5 between chromosomes, for visualization purposes
        if (i_chrom != num_chroms-1)
        {
            for (int j_chrom = 0; j_chrom < num_chroms; j_chrom++)
                for (int j_locus = 0; j_locus < chroms[j_chrom]->num_loci; j_locus++)
                    fprintf (output_file, "0.87 ");
            fprintf (output_file, "0.87\n");
        }
    }
    fclose (output_file);
}

void GENOME::read_txt_input(char *txt_filename)
{
    FILE* file = fopen(txt_filename, "r");
    if(file == 0)
      die ("Unable to open text input file. \n");

    int num_variables = get_num_loci() * 3;
    double *x = new double[num_variables];

    for(int i=0; i < num_variables; i++)
    {
      fscanf(file, "%le\n", &x[i]);
    }
    fclose(file);
    set_coords((double*) x);
    delete [] x;
}

void GENOME::save_txt (char *txt_filename)
{
    FILE* file=fopen(txt_filename,"w");

    for (int i=0; i < get_num_loci(); i++)
    {
        COORDS_T *coord = get_coords(i);
        if (coord == NULL)
            die ("Something wrong in GENOME::get_coords, returned NULL\n");
        fprintf(file,"%e\n",coord->x);
        fprintf(file,"%e\n",coord->y);
        fprintf(file,"%e\n",coord->z);
    }
    fclose(file);
}

#define MAX_LINE 1000 // Maximum number of characters in one line.
void GENOME::read_cplex_output (char *cplex_filename)
{
    FILE* file;
    // first traverse the file and get the maximum index. That will be num_variables
    if (open_file(cplex_filename, "r", TRUE, "cplex", "cplex", &file) == 0)
    {
        exit(1);
    }
    char one_line[MAX_LINE];
    int xindex;
    double value;
    int num_variables = 0;
    while (1)
    {
        if (fgets(one_line, MAX_LINE, file) == NULL)
        {
            break;
        }
        int num_scanned = sscanf(one_line, "%d", &xindex);

        //printf ("%s, num_scanned = %d\n", one_line, num_scanned);
        if (num_scanned == 1)
        {
            if (xindex > num_variables)
                num_variables = xindex;
        }
    }
    fclose(file);
    printf ("Num variables: %d\n", num_variables);

    // Open the file for reading.
    double *x = new double[num_variables];
    if (open_file(cplex_filename, "r", TRUE, "cplex", "cplex", &file) == 0)
    {
        exit(1);
    }
    while (1)
    {
        if (fgets(one_line, MAX_LINE, file) == NULL)
        {
            break;
        }
        int num_scanned = sscanf(one_line, "%d %le",
                                &xindex, &value);
        printf ("%s, num_scanned = %d\n", one_line, num_scanned);
        if (num_scanned == 2)
        {
            printf ("x%d has value %e\n", xindex, value);
            x[xindex]=value;
        }
    }
    set_coords((double*)x);
    delete [] x;
    fclose(file);
}

