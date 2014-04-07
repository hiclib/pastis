// Copyright (C) 2005, 2007 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Common Public License.
//
// $Id: genome_ipopt_main.cpp,v 1.6 2010/12/15 08:09:54 andrones Exp $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2005-08-10
#include <stdio.h>
#include <string.h>
#include "IpIpoptApplication.hpp"
#include "genome_ipopt_nlp.hpp"
#include "metric.hpp"
#include "getopt.h"

#define HAVE_STDIO_H 0

// for printf
#ifdef HAVE_CSTDIO
# include <cstdio>
#else
# ifdef HAVE_STDIO_H
#  include <stdio.h>
# else
#  error "don't have header file for stdio"
# endif
#endif

using namespace Ipopt;

char interactions_filename[1000]="../2009-06-15/sets-U5-C1/interactions_at_fdr_0.01_U_intra_atleast20000_chr3.txt";
char logging_filename[1000] = "ipopt.out";
int bp_per_locus = 1000;
double sphere_radius = -10.;
double max_dist = -0.0091;   // 0.0091 microm max length of a (packed around nucleosomes) 1000bp
double min_dist = -0.0033;   // 0.0066 microm min length of a (packed around nucleosomes) 1000bp
double min_clash_dist = -0.03;   // 30 nm distance
double min_clash_dist_inter = 0.075;
int add_rDNA = 1;
char output_pdb[1000] = "output.pdb";
char input_txt[1000] = "";
char output_binary_diffusion[1000] = "";
int diffusion_operation_type = 0;
double alpha = -3.;
double beta = 1.;

char output_cplex_input[1000] = "";     // output for us, input for cplex
char input_cplex_output[1000] = "";     // input for us, output for cplex

int num_chroms = 30;
int chromosome[30]; // 16 chromosomes in yeast

// this array will have 1 for the chromosomes I'm interested in, and 0 for the ones I'm not interested in

char chrom_string[100] = "1";
int use_weights = 8;
float weight_of_inter = 1.;
char rDNA_interactions_filename[1000] = "";
char structure_filename[1000] = "";
char rDNA_frequency_normalizer = 1;

char unseen_interactions_filename[1000] = "";
char frequencies_distances_filename[1000] = "";
double weight_unseen = 1;

void usage(const char *prog)
{
    printf("\nUsage: %s -i <interactions_file> [options]\n", prog);
    printf("\n");
    printf("Options:\n");
    printf("========\n");
    printf("  -c <chromosomes>, optional, default: 1\n");
    printf("     The list of chromosomes to infer.\n");
    printf("     The list should be coma separated, eg: '1,2,3'\n");
    printf("\n");
    printf("  -r <resolution>, optional, default %d\n", bp_per_locus);
    printf("     The resolution of the inference. The structure of the\n");
    printf("     genome and the resolution of the inference must be\n");
    printf("     coherent with the wish distance/interaction counts\n");
    printf("     datafile.\n");
    printf("\n");

    printf("  -s <sphere_radius>, optional, default %g\n", sphere_radius);
    printf("     The radius of the nucleus. If negative, no constraints\n");
    printf("     will be applied on the beads.\n");
    printf("\n");

    printf("  -d <distance_perkb>, optional, default %g\n", max_dist);
    printf("     Maximum value between adjacent beads per kb. If negative,\n");
    printf("     no constraints will be applied on adjacent beads.\n");
    printf("\n");
    printf("  -o <output_pdb>, optional, default: 'output.pdb'\n"); 
    printf("\n");
    printf("  -k <structure_file>\n");
    printf("\n");
    printf("  -l <logging_file>\n");
    printf("\n");
    printf("  -a <alpha>, optional, default: -3\n");
    printf("\n");
    printf("  -b <beta>, optional, default: 1\n");
    printf("\n");
    printf("  -h Print this help message\n\n");

    exit(0);
}


void get_arguments (int argc, char *argv[])
{
    int c;
    extern char *optarg;
    extern int optind;
    int errflag = 0;

    while ((c = getopt (argc, argv, "a:b:c:d:e:f:g:h:i:j:k:l:n:r:s:w:y:o:p:t:x:z:?")) != -1)
    {
        switch (c)
        {
            case 'c':
                strcpy (chrom_string, optarg);
                break;
            case 'n':
                strcpy (rDNA_interactions_filename, optarg);
                break;
            case 'k':
                strcpy (structure_filename, optarg);
                break;
            case 'e':
                rDNA_frequency_normalizer = atoi (optarg);
                break;
            case 'f':
                strcpy (unseen_interactions_filename, optarg);
                break;
            case 'g':
                weight_unseen = atof (optarg);
                break;
            case 'i':
                strcpy (interactions_filename, optarg);
                break;
            case 'r':
                bp_per_locus = atoi (optarg);
                break;
            case 's':
                sphere_radius = atof (optarg);
                break;
            case 'j':
                min_clash_dist_inter = atof (optarg);
                break;
            case 'd':
                max_dist = atof (optarg);
                break;
            case 'o':
                strcpy (output_pdb, optarg);
                break;
            case 'w':
                use_weights = atoi (optarg);
                break;
            case 'y':
                weight_of_inter = atof(optarg);
                break;
            case 'b':
                beta = atof(optarg);
                break;
            case 'a':
                alpha = atof(optarg);
                break;
            case 'p':
                strcpy (output_cplex_input, optarg);
                break;
            case 't':
                strcpy (input_cplex_output, optarg);
                break;
            case 'x':
                strcpy (input_txt, optarg);
                break;
            case 'z':
                strcpy (frequencies_distances_filename, optarg);
                break;
            case 'l':
                strcpy(logging_filename, optarg);
                break;
            case 'h':
            case '?':
            default:
                errflag = 1;
        }
    }
    if (errflag || optind != argc || argc < 2)
        usage (argv[0]);

}


int main(int argc, char* argv[])
{

    get_arguments (argc, argv);

    // figure out which chromosomes I want
    for (int i=0; i < num_chroms; i++)
    {
        chromosome[i] = 0;
    }
    char *token;
    token = strtok(chrom_string,",");
    while (token != NULL)
    {
        int index = atoi(token) - 1;
        chromosome[index] = 1;
        token = strtok(NULL,",");
    }

    // the max_dist is per kb, so adjust it so that it depends on the resolution
    max_dist = max_dist * bp_per_locus / 1000.0;
    // same for min_dist
    min_dist = min_dist * bp_per_locus / 1000.0;

    // Create a new instance of your nlp
    GENOME *mygenome = new GENOME(bp_per_locus, min_dist, max_dist,
                                  interactions_filename, chromosome, add_rDNA,
                                  rDNA_interactions_filename, use_weights,
                                  unseen_interactions_filename,
                                  frequencies_distances_filename,
                                  structure_filename);

    // If a temporary pdb file exists, upload it and initialize the problem
    // there

    char temp_filename[1000];
    sprintf(temp_filename, "%s.temp.txt", output_pdb);
    FILE* file = fopen(temp_filename, "r");
    if(file == 0){
      printf("No temporary file - starting optimization from scratch\n");
    }else{
      fclose(file);
      mygenome->read_txt_input(temp_filename);
    }

    if (strcmp (input_txt, "") != 0)
    {
        printf ("Reading the txt input\n");
        mygenome->read_txt_input(input_txt);
        mygenome->print_pdb_genome(output_pdb);
        return 1;
    }
    else if (strcmp (output_binary_diffusion, "") != 0)
    {
        mygenome->save_adjacency_matrix_for_diffusion(
                output_binary_diffusion,
                diffusion_operation_type);
        return 1;
    }
    else if (strcmp (output_cplex_input, "") != 0)
    // just write the .lp file for cplex
    {
        SmartPtr<genome_ipopt_nlp> mynlp = new genome_ipopt_nlp(
                                                  mygenome, min_clash_dist,
                                                  min_clash_dist_inter,
                                                  output_pdb, sphere_radius,
                                                  use_weights, bp_per_locus,
                                                  rDNA_frequency_normalizer,
                                                  weight_of_inter,
                                                  weight_unseen, true,
                                                  alpha, beta);
        mynlp->write_cplex_input (output_cplex_input);
        return 1;
    }
    else if (strcmp (input_cplex_output, "") != 0)
    // read the file in .sol format (cplex output) and write a pdb file
    {
        mygenome->read_cplex_output(input_cplex_output);
        mygenome->print_pdb_genome(output_pdb);
        return 1;
    }
  //mygenome->save_interaction_adjacency_matrix (initial_interaction_matrix);
  //mygenome->print_pdb_genome (initial_pdb);

  // FIXME we haven't interfaced many of the options.
  SmartPtr<TNLP> mynlp = new genome_ipopt_nlp(
                              mygenome, min_clash_dist, min_clash_dist_inter,
                              output_pdb, sphere_radius, use_weights,
                              bp_per_locus, rDNA_frequency_normalizer,
                              weight_of_inter, weight_unseen, true, alpha, beta
                              );

  // Create a new instance of IpoptApplication
  //  (use a SmartPtr, not raw)
  SmartPtr<IpoptApplication> app = new IpoptApplication();

  // Change some options
  // Note: The following choices are only examples, they might not be
  //       suitable for your optimization problem.
  app->Options()->SetNumericValue("tol", 1e-1);
  app->Options()->SetNumericValue("acceptable_tol", 1e-1);
  // I'm going to set the constraint violation to very very high.
  app->Options()->SetNumericValue("constr_viol_tol", 1e10);
  app->Options()->SetNumericValue("mu_init", 0.0001);
  // This is super high. It should not impact the results
  app->Options()->SetNumericValue("dual_inf_tol", 10000);
  app->Options()->SetNumericValue("compl_inf_tol", 1000000);
  // This should be zero.
  app->Options()->SetIntegerValue("acceptable_iter", 0);

  app->Options()->SetIntegerValue("max_iter", 100000);

  //app->Options()->SetStringValue("mu_strategy", "monotone");
  app->Options()->SetStringValue("mu_strategy", "adaptive");
  app->Options()->SetStringValue("output_file", logging_filename);
  app->Options()->SetStringValue("hessian_approximation", "limited-memory");

  // The following overwrites the default name (ipopt.opt) of the
  // options file
  // app->Options()->SetStringValue("option_file_name", "hs071.opt");

  // Intialize the IpoptApplication and process the options
  ApplicationReturnStatus status;
  status = app->Initialize();
  if (status != Solve_Succeeded) {
    printf("\n\n*** Error during initialization!\n");
    return (int) status;
  }

  // Ask Ipopt to solve the problem
  status = app->OptimizeTNLP(mynlp);

  if (status == Solve_Succeeded || status == Solved_To_Acceptable_Level) {
    printf("\n\n*** The problem solved!\n");
  }
  else {
    printf("\n\n*** The problem FAILED!\n");
    exit(1);
  }

  // As the SmartPtrs go out of scope, the reference count
  // will be decremented and the objects will automatically
  // be deleted.

  return 1;     //(int) status;

}


