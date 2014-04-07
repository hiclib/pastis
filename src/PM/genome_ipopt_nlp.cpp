// Copyright (C) 2005, 2006 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Common Public License.
//
// $Id: genome_ipopt_nlp.cpp,v 1.5 2010/05/12 03:05:11 andrones Exp $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2005-08-16

#include <iostream>
#include <iomanip>
#include <assert.h>
#include <fstream>
#define HAVE_STDIO_H 0

using namespace std;

#include "genome_ipopt_nlp.hpp"
#include "genome_model.hpp"
#include "coords.h"

#include <cmath>
#include <math.h>
#define BUDDINGYEAST 0
#define MALARIA 1
#define WITH_CONSTRAINTS 1

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

typedef struct coords COORDS_T;

using namespace Ipopt;

// constructor
genome_ipopt_nlp::genome_ipopt_nlp()
{}

// my additional constructor
genome_ipopt_nlp::genome_ipopt_nlp(GENOME *mygenome, double min_clash_dist,
                                    double min_clash_dist_inter,
                                    char* output_pdb, double sphere_radius,
                                    int use_weights, int bp_per_locus,
                                    int rDNA_frequency_normalizer,
                                    float weight_of_inter, double weight_unseen,
                                    bool poisson_model, double alpha,
                                    double beta)
{
    this->mygenome = mygenome;
    this->output_pdb = output_pdb;
    this->min_clash_dist = min_clash_dist;
    this->min_clash_dist_inter = min_clash_dist_inter;
    this->sphere_radius = sphere_radius;
    this->use_weights = use_weights;
    this->weight_of_inter = weight_of_inter;
    this->bp_per_locus = bp_per_locus;
    this->rDNA_frequency_normalizer = rDNA_frequency_normalizer;
    this->weight_unseen = weight_unseen;
    this->poisson_model = true;
    this->weighted_objective = 1;
    // weighted objective:
    //  - 0: no weights
    //  - 1: inverse of wish distance

    if(this->poisson_model){
      cout << "RUNNING POISSON MODEL OPTIMIZATION" << endl;
    }else{
      cout << "RUNNING MDS OPTIMIZATION PROBLEM"  << endl;
    }

    this->alpha = alpha;
    cout << alpha << endl;
    cout << beta << endl;
    this->beta = beta;

    this->iteration = 0;

    // The number of variables: 3 coordinates for each locus
    this->num_loci = this->mygenome->get_num_loci();
    this->num_interactions = this->mygenome->get_num_interactions();

    this->num_variables = this->num_loci * 3;

    printf("num variables = %d\n", this->num_variables);

    // Number of constraints
    // first add the number of loci, for the constraint to be in a sphere
    // NOTE: if I don't add any sphere constraints, it will be in a cube
    if(this->sphere_radius > 0){
      this->num_sphere_constraints = this->num_loci;
    }else{
      this->num_sphere_constraints = 0;
    }

    // add the number of adjacent loci, for the break constraints
    if(this->max_dist > 0){
      this->num_break_constraints = this->mygenome->get_num_adjacent_loci();
    }else{
      this->num_break_constraints = 0;
    }

    // add the number of non-adjacent loci, for the clash constraints
    // NOTE: If I remove the clash constraints, the figure looks just terrible
    //this->num_clash_constraints = this->num_loci * (this->num_loci - 1) / 2;
    this->num_clash_constraints = 0;
    // I could remove the num_break_constraints, but it's simpler to leave
    // them in

    // number of distance constraints = number of interactions
    if(this->use_weights == 2){
        this->num_dist_constraints = this->num_interactions;
    }else{
        this->num_dist_constraints = 0;
    }

    // Constraints for the centromeric area and for the nucleolus
    // 16 for the centromeres of all chromosomes
    // and a bunch for the rDNA regions
    int chr12 = mygenome->get_chrom_index(12);

    if(this->mygenome->organism == BUDDINGYEAST and chr12 != -1){
      this->num_small_sphere_constraints = 1 +
        NUM_RDNA_TO_ADD * SIZE_RDNA / this->bp_per_locus;
    }else{
      this->num_small_sphere_constraints = 0;
    }


    this->num_constraints = num_sphere_constraints + num_break_constraints +
                      num_clash_constraints + num_dist_constraints +
                      num_small_sphere_constraints;

    printf("num constraints = %d = %d + %d + %d + %d + %d\n",
        num_constraints, num_sphere_constraints, num_break_constraints,
        num_clash_constraints, num_dist_constraints,
        num_small_sphere_constraints);

    this->min_dist = this->mygenome->get_min_dist();
    this->max_dist = this->mygenome->get_max_dist();

}

//destructor
genome_ipopt_nlp::~genome_ipopt_nlp()
{}


// returns the size of the problem
bool genome_ipopt_nlp::get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
                             Index& nnz_h_lag, IndexStyleEnum& index_style)
{
    // The number of variables
    n = this->num_variables;

    // Number of constraints
    m = this->num_constraints;

    // The number of non-zero elements in the Jacobian
    nnz_jac_g = this->num_sphere_constraints * 3 +
                this->num_break_constraints * 6 +
                this->num_clash_constraints * 6 +
                this->num_dist_constraints * 6 +
                this->num_small_sphere_constraints * 3;

    // number of non-zero elements in the Hessian
    // don't need to set nnz_h_lag if I use quasi-newton

    // use the C style indexing (0-based)
    index_style = TNLP::C_STYLE;

    return true;
}

// returns the variable bounds
bool genome_ipopt_nlp::get_bounds_info(Index n, Number* x_l, Number* x_u,
                                Index m, Number* g_l, Number* g_u)
{
    // here, the n and m we gave IPOPT in get_nlp_info are passed back to us.
    // If desired, we could assert to make sure they are what we think they are.
    assert(n == this->num_variables);
    assert(m == this->num_constraints);

    // the bounds on the variables

    // first the xyz locations
    // we are in a sphere with radius sphere_radius
    for(Index i = 0; i < this->num_variables; i++)
    {
        x_l[i] = - abs(this->sphere_radius);
        x_u[i] = abs(this->sphere_radius);
    }

    // Let's start by the constraints on the sphere: loci should lie in the
    // nucleus, which approximately a sphere of size  `sphere_radius`.
    for(Index i = 0; i < this->num_sphere_constraints; i++)
    {
        g_l[i] = -2e19; 
        g_u[i] = this->sphere_radius * this->sphere_radius;
    }

    int msofar = this->num_sphere_constraints;

    // Break constraints !
    for(Index i=msofar; i < msofar + this->num_break_constraints; i++)
    {
        // FIXME doesn't use break constraints anymore. Here we set it to
        // minus the min_dist, but in fact, we really want to get rid of this.
        g_l[i] = -10 * min_dist * min_dist;
        g_u[i] = max_dist * max_dist;
    }

    msofar += this->num_break_constraints;

    // FIXME More break constraints we would like to get rid of.
    Index i = msofar;
    if(this->num_clash_constraints != 0){
      for(int i1 = 0; i1 < this->num_loci; i1++){
          int chr1 = mygenome->get_chromosome(i1);
          for(int i2 = i1 + 1; i2 < num_loci; i2++)
          {
              int chr2 = mygenome->get_chromosome(i2);
              if (chr1 == chr2){
                  g_l[i] = -10 * min_clash_dist * min_clash_dist;
              }else{
                  g_l[i] = - 10 * min_clash_dist_inter * min_clash_dist_inter;
              }
              g_u[i] = 2e19;
              i++;
          }
      }
    }

    msofar += this->num_clash_constraints;

    // the distance constraints
    i = msofar;
    if(num_dist_constraints > 0)
    {
        INTERACTION_T *interaction = mygenome->get_first_interaction();
        while (interaction != NULL)
        {
            // FIXME what is this ?
            g_l[i] = - 1.0 / interaction->freq * 1.0 / interaction->freq * min_dist * min_dist;
            g_u[i] = this->sphere_radius * 2 * this->sphere_radius * 2;
            i++;
            interaction = interaction->next;
        }
    }
    msofar += this->num_dist_constraints;

    // If the organism is budding yeast, we want to constrain the rDNA inside
    // the nucleolus. If it is malaria, don't do anything.
    int chr12 = mygenome->get_chrom_index(12);
    if(this->mygenome->organism == BUDDINGYEAST and chr12 != -1 and this->num_small_sphere_constraints > 0){
      g_l[msofar] = -2e19;
      g_u[msofar] = CENTRO_RADIUS * CENTRO_RADIUS;
      for(Index i=msofar + 1; i < msofar + num_small_sphere_constraints; i++)
      {
          g_l[i] = -2e19;
          g_u[i] = NUCLEO_RADIUS * NUCLEO_RADIUS;
      }
      msofar += num_small_sphere_constraints;
    }
    cout << m << " " << msofar << endl;
    assert (m == msofar);

    return true;
}


// returns the initial point for the problem
bool genome_ipopt_nlp::get_starting_point(Index n, bool init_x, Number* x,
                                   bool init_z, Number* z_L, Number* z_U,
                                   Index m, bool init_lambda,
                                   Number* lambda)
{
    // Here, we assume we only have starting values for x, if you code
    // your own NLP, you can provide starting values for the dual variables
    // if you wish

    assert(init_x == true);
    assert(init_z == false);
    assert(init_lambda == false);

    assert(n == num_variables);

    // initialize to the given starting point
    for(Index i=0; i < num_loci * 3; i += 3)
    {
        COORDS_T *coord = mygenome->get_coords(i / 3);
        if (coord == NULL){
            die ("Something wrong in GENOME::get_coords, returned NULL\n");
        }
        // NOTE: I tried to set these coordinates to 0.0 or 0.9. The IPOPT problem fails (it says Restoration failed)
        x[i] = coord->x;  // x coordinate
        x[i + 1] = coord->y;  // y coordinate
        x[i + 2] = coord->z;  // z coordinate
    }

    return true;
}

int genome_ipopt_nlp::is_inter_arms(INTERACTION_T *interaction)
// return 1 if this interaction is between 2 arms (of the same or different chrom)
// return 0 is this is intra within the same arm
{
    if (interaction->chrom1_index != interaction->chrom2_index)
        return 1;

    // same chromosomes
    int centro_locus = (int)(this->mygenome->centromeres[interaction->chrom1_index]/bp_per_locus);
    if (( interaction->locus1_index < centro_locus && interaction->locus2_index > centro_locus) ||
        ( interaction->locus1_index > centro_locus && interaction->locus2_index < centro_locus)
        )
       return 1;

    return 0;
}

int genome_ipopt_nlp::is_inter_centro_telo (INTERACTION_T *interaction)
// return 1 if this interaction is between the centromere area and the telomere area of the same chromosome
// return 0 if it's not
{
    if (interaction->chrom1_index != interaction->chrom2_index)
        return 0;

    // same chromosomes
    int centro_locus = (int)(this->mygenome->centromeres[interaction->chrom1_index]/bp_per_locus);

    // NOTE: I'm not using the telomere information here. TODO???
    int telo_locus = (int)(this->mygenome->chrlen[interaction->chrom1_index]/bp_per_locus);

    int centro1 = centro_locus - WINDOW/bp_per_locus;
    int centro2 = centro_locus + WINDOW/bp_per_locus;
    int telo1 = 2*WINDOW/bp_per_locus;
    int telo2 = telo_locus - 2*WINDOW/bp_per_locus;

    printf ("Is inter centro telo: %d %d %d %d -- %d - %d %d %d - %d\n",
            interaction->chrom1_index, interaction->locus1_index,
            interaction->chrom2_index, interaction->locus2_index,
            telo1, centro1, centro_locus, centro2, telo2);


    // left arm
    if (interaction->locus1_index >= centro1 &&
        interaction->locus1_index <= centro_locus &&
        interaction->locus2_index <= telo1)
        return 1;

    // reverse of left arm
    if (interaction->locus2_index >= centro1 &&
        interaction->locus2_index <= centro_locus &&
        interaction->locus1_index <= telo1)
        return 1;

    // right arm
    if (interaction->locus1_index >= centro_locus &&
        interaction->locus1_index <= centro2 &&
        interaction->locus2_index >= telo2)
        return 1;

    //reverse of the right arm
    if (interaction->locus2_index >= centro_locus &&
        interaction->locus2_index <= centro2 &&
        interaction->locus1_index >= telo2)
        return 1;

    return 0;
}


bool genome_ipopt_nlp::likelihood(Index n, const Number * x, bool new_x,
                                  Number & obj_value){ 
  assert(n == num_variables);

  if(this->num_interactions > 0){
    INTERACTION_T * interaction = this->mygenome->get_first_interaction();
    while(interaction != NULL){
      int l = mygenome->get_x_index(interaction->chrom1_index,
                                    interaction->locus1_index);
      int r = mygenome->get_x_index(interaction->chrom2_index,
                                    interaction->locus2_index);

      double distance = sqrt((x[l] - x[r]) * (x[l]-x[r]) +
                             (x[l + 1] - x[r + 1]) * (x[l + 1] - x[r + 1]) +
                             (x[l + 2] - x[r + 2]) * (x[l + 2] - x[r + 2]));
      double d = interaction->freq * log(this->beta) -
                 interaction->freq * distance * this->alpha -
                 this->beta * exp(- this->alpha * distance);
      obj_value -= d;
      interaction = interaction->next;
    }
  }
  return true;
}


bool genome_ipopt_nlp::likelihood_power_law(Index n, const Number * x,
                                            bool new_x, Number & obj_value){ 
  assert(n == this->num_variables);

  if(this->num_interactions > 0){
    INTERACTION_T * interaction = this->mygenome->get_first_interaction();
    while(interaction != NULL){
      int l = mygenome->get_x_index(interaction->chrom1_index,
                                    interaction->locus1_index);
      int r = mygenome->get_x_index(interaction->chrom2_index,
                                    interaction->locus2_index);

      double distance = sqrt((x[l] - x[r]) * (x[l] - x[r]) +
                             (x[l + 1] - x[r + 1]) * (x[l + 1] - x[r + 1]) +
                             (x[l + 2] - x[r + 2]) * (x[l + 2] - x[r + 2]));
      double d = interaction->freq * log(this->beta) +
                 interaction->freq * this->alpha * log(distance) - 
                 this->beta * pow(distance, this->alpha) ;

      obj_value -= d;
      interaction = interaction->next;
    }
  }
  return true;
}


bool genome_ipopt_nlp::eval_f(Index n,
                              const Number* x, bool new_x, Number& obj_value)
{
  if(this->poisson_model){
    this->likelihood_power_law(n, x, new_x, obj_value);
  }else{
    this->mds(n, x, new_x, obj_value);
  }

  // Save the file every 500 iterations, so that we can check what it's doin'
  iteration++;
  if(iteration % 500 == 0)
  {
      mygenome->set_coords((double*) x);
      char temp_filename[1000];
      sprintf(temp_filename, "%s.temp.pdb", output_pdb);
      mygenome->print_pdb_genome(temp_filename);
      sprintf(temp_filename, "%s.temp.txt", output_pdb);
      mygenome->save_txt(temp_filename);
  }
  return true;

}


// returns the value of the objective function
bool genome_ipopt_nlp::mds(Index n,
                           const Number* x,
                           bool new_x,
                           Number& obj_value)
{
    // printf ("Trying eval f\n");
    assert(n == this->num_variables);

    obj_value = 0;

    if (num_interactions > 0)
    {
        INTERACTION_T *interaction = mygenome->get_first_interaction();
        while (interaction != NULL)
        {
            int l = mygenome->get_x_index(interaction->chrom1_index,
                                          interaction->locus1_index);
            int r = mygenome->get_x_index(interaction->chrom2_index,
                                          interaction->locus2_index);
            if(use_weights == 5 || use_weights == 6 || use_weights == 7 || use_weights == 8)   // use wish_dist
            {
                // Unfortunately, this function won't be quadratic any more
                double w = 1;
                double d = sqrt((x[l] - x[r]) * (x[l] - x[r]) +
                                (x[l + 1] - x[r + 1]) * (x[l + 1] - x[r + 1]) +
                                (x[l + 2] - x[r + 2]) * (x[l + 2] - x[r + 2])) -
                             interaction->wish_dist;

                // add the weight_of_intra, if applicable
                if (is_inter_arms(interaction))
                {
                    d = d * weight_of_inter;
                }

                if(this->weighted_objective == 1){
                  w =  1. / (interaction->wish_dist * interaction->wish_dist);
                }

                obj_value += w * d * d;
            }
            else    // use weights in the objective function
            {
                double w;
                switch (use_weights)
                {
                    case 0: w = 1; break;
                    case 1: w = interaction->freq; break;
                    case 2: w = 1; break;
                    case 3:
                      exit(0);
                      break;
                    case 4:
                        exit(0);
                        break;
                    default: w = 1;
                }

                // NOTE: f can be negative (for example for centromere of 12 vs rDNA), so I have to keep the sign
                if (w > 0)
                {
                    obj_value +=
                        w*w*(x[l]-x[r])*(x[l]-x[r]) +
                        w*w*(x[l+1]-x[r+1])*(x[l+1]-x[r+1]) +
                        w*w*(x[l+2]-x[r+2])*(x[l+2]-x[r+2]);
                }
                else
                {
                    obj_value = obj_value -
                        w*w*(x[l]-x[r])*(x[l]-x[r]) -
                        w*w*(x[l+1]-x[r+1])*(x[l+1]-x[r+1]) -
                        w*w*(x[l+2]-x[r+2])*(x[l+2]-x[r+2]);
                }
            }
            interaction = interaction->next;
        }
    }

}

bool genome_ipopt_nlp::grad_likelihood(Index n, const Number * x, bool new_x,
                                       Number * grad_f){
  assert(n == num_variables);

  for(Index i=0; i < this->num_loci * 3; i++)
  {
    grad_f[i] = 0;
  }

  if(num_interactions > 0){
    INTERACTION_T * interaction = this->mygenome->get_first_interaction();
    while (interaction != NULL)
    {
      int l = this->mygenome->get_x_index(interaction->chrom1_index,
                                          interaction->locus1_index);
      int r = this->mygenome->get_x_index(interaction->chrom2_index,
                                          interaction->locus2_index);

      double dis = sqrt((x[l] - x[r]) * (x[l] - x[r]) +
                        (x[l + 1] - x[r + 1]) * (x[l + 1] - x[r + 1]) +
                        (x[l + 2] - x[r + 2]) * (x[l + 2] - x[r + 2]));

      grad_f[l] -= (x[l] - x[r]) / dis * 
                   (- interaction->freq * this->alpha +
                    this->beta * this->alpha * exp(- this->alpha * dis));
      grad_f[l + 1] -= (x[l + 1] - x[r + 1]) / dis * 
                   (- interaction->freq * this->alpha +
                    this->beta * this->alpha * exp(- this->alpha * dis));
      grad_f[l + 2] -= (x[l + 2] - x[r + 2]) / dis * 
                   (- interaction->freq * this->alpha +
                    this->beta * this->alpha * exp(- this->alpha * dis));
      grad_f[r] += (x[l] - x[r]) / dis * 
                   (- interaction->freq * this->alpha +
                    this->beta * this->alpha * exp(- this->alpha * dis));
      grad_f[r + 1] += (x[l + 1] - x[r + 1]) / dis * 
                   (- interaction->freq * this->alpha +
                    this->beta * this->alpha * exp(- this->alpha * dis));
      grad_f[r + 2] += (x[l + 2] - x[r + 2]) / dis * 
                   (- interaction->freq * this->alpha +
                    this->beta * this->alpha * exp(- this->alpha * dis));
      interaction = interaction->next;
    }
  }
  return true;
}


bool genome_ipopt_nlp::grad_likelihood_power_law(Index n, const Number * x,
                                       bool new_x,
                                       Number * grad_f){
  assert(n == num_variables);

  for(Index i=0; i < this->num_loci * 3; i++)
  {
    grad_f[i] = 0;
  }

  if(num_interactions > 0){
    INTERACTION_T * interaction = this->mygenome->get_first_interaction();
    while (interaction != NULL)
    {
      int l = this->mygenome->get_x_index(interaction->chrom1_index,
                                          interaction->locus1_index);
      int r = this->mygenome->get_x_index(interaction->chrom2_index,
                                          interaction->locus2_index);

      double dis = sqrt((x[l] - x[r]) * (x[l] - x[r]) +
                        (x[l + 1] - x[r + 1]) * (x[l + 1] - x[r + 1]) +
                        (x[l + 2] - x[r + 2]) * (x[l + 2] - x[r + 2]));
      if(dis < 1e-14){
        cout << "Distance is 0" << endl;
        interaction = interaction->next;
        continue;
      }

      grad_f[l] -=  this->alpha * (x[l] - x[r]) / (dis * dis) *
                    (interaction->freq - this->beta * pow(dis, this->alpha));
        
      grad_f[l + 1] -=  this->alpha * (x[l + 1] - x[r + 1]) / (dis * dis) *
                    (interaction->freq - this->beta * pow(dis, this->alpha));

      grad_f[l + 2] -=  this->alpha * (x[l + 2] - x[r + 2]) / (dis * dis) *
                    (interaction->freq - this->beta * pow(dis, this->alpha));

      grad_f[r] +=  this->alpha * (x[l] - x[r]) / (dis * dis) *
                    (interaction->freq - this->beta * pow(dis, this->alpha));
        
      grad_f[r + 1] +=  this->alpha * (x[l + 1] - x[r + 1]) / (dis * dis) *
                    (interaction->freq - this->beta * pow(dis, this->alpha));

      grad_f[r + 2] +=  this->alpha * (x[l + 2] - x[r + 2]) / (dis * dis) *
                    (interaction->freq - this->beta * pow(dis, this->alpha));

      interaction = interaction->next;
    }
  }
  return true;
}



// return the gradient of the objective function grad_{x} f(x)
bool genome_ipopt_nlp::grad_mds(Index n, const Number* x, bool new_x,
                                Number* grad_f)
{
    assert(n == num_variables);

    for (Index i=0; i<num_loci*3; i++)
    {
        grad_f[i] = 0;
    }


    if (num_interactions > 0)
    {
        INTERACTION_T *interaction = mygenome->get_first_interaction();
        while (interaction != NULL)
        {
            //printf ("%d, %d\t", interaction->chrom1_index, interaction->locus1_index);
            //printf ("%d, %d\n", interaction->chrom2_index, interaction->locus2_index);
            int l = mygenome->get_x_index(interaction->chrom1_index,
                                          interaction->locus1_index);
            int r = mygenome->get_x_index(interaction->chrom2_index,
                                          interaction->locus2_index);

            if(use_weights == 5 || use_weights == 6 || use_weights == 7 || use_weights == 8)   // use wish_dist
            {
                // Unfortunately, this function won't be quadratic any more
                double sq = sqrt((x[l] - x[r]) * (x[l] - x[r]) +
                                 (x[l + 1] - x[r + 1]) * (x[l + 1] - x[r + 1]) +
                                 (x[l + 2] - x[r + 2]) * (x[l + 2] - x[r + 2]));
                double di = sq - interaction->wish_dist;
                double w = 1.;

                // add the weight_of_inter, if applicable

                if (is_inter_arms(interaction))
                {
                    di = di*weight_of_inter;
                }

                if(this->weighted_objective == 1){
                  w = 1. / (interaction->wish_dist * interaction->wish_dist);
                }


                // NOT SURE why this is this way, but it looks wrong to me (Dec 2, 2009)
				        // No, this is right, because there's a square
                if(sq != 0){
                  grad_f[l] += w * di / sq * 2 * (x[l] - x[r]);
                  grad_f[l + 1] += w * di / sq * 2 * (x[l + 1] - x[r + 1]);
                  grad_f[l + 2] += w * di / sq * 2 * (x[l + 2] - x[r + 2]);
                  grad_f[r] -= w * di / sq * 2 * (x[l] - x[r]);
                  grad_f[r + 1] -= w * di / sq * 2 * (x[l + 1] - x[r + 1]);
                  grad_f[r + 2] -= w * di / sq * 2 * (x[l + 2] - x[r + 2]);
                }
            }
            else
            {
                double w;
                switch (use_weights)
                {
                    case 0: w = 1; break;
                    case 1: w = interaction->freq; break;
                    case 2: w = 1; break;
                    case 3:
                        exit(0);
                        break;
                    case 4:
                        exit(0);
                        break;
                    default: w = 1;
                }

                if (w > 0)
                {
                    grad_f[l] += 2*w*w*(x[l]-x[r]);
                    grad_f[l+1] += 2*w*w*(x[l+1]-x[r+1]);
                    grad_f[l+2] += 2*w*w*(x[l+2]-x[r+2]);
                    grad_f[r] += -2*w*w*(x[l]-x[r]);
                    grad_f[r+1] += -2*w*w*(x[l+1]-x[r+1]);
                    grad_f[r+2] += -2*w*w*(x[l+2]-x[r+2]);
                }
                else
                {
                    grad_f[l] -= 2*w*w*(x[l]-x[r]);
                    grad_f[l+1] -= 2*w*w*(x[l+1]-x[r+1]);
                    grad_f[l+2] -= 2*w*w*(x[l+2]-x[r+2]);
                    grad_f[r] -= -2*w*w*(x[l]-x[r]);
                    grad_f[r+1] -= -2*w*w*(x[l+1]-x[r+1]);
                    grad_f[r+2] -= -2*w*w*(x[l+2]-x[r+2]);
                }
            }
            interaction = interaction->next;
        }
    }

    return true;
}

// return the value of the constraints: g(x)
bool genome_ipopt_nlp::eval_g(Index n, const Number* x, bool new_x, Index m, Number* g)
{
    //printf ("Trying eval g\n");
    assert(n == this->num_variables);
    assert(m == num_constraints);

    int xvari = num_loci * 3;

    // CAREFUL: I have to make sure I have the right order of constraints

    // first the sphere constraints
    // 0 <= d^2(p_i,0) <= 1, for all loci p_i
    for (Index i=0; i<num_sphere_constraints*3; i+=3)
    {
        g[i / 3] = x[i] * x[i] + x[i + 1] * x[i + 1] + x[i + 2] * x[i + 2];
    }

    // the break constraints
    Index consi = num_sphere_constraints;
    int i=0;
    if (num_break_constraints > 0)
    {
        for (int c = 0; c < mygenome->get_num_chroms(); c++)
        {
            for(int l = 0; l < mygenome->get_num_loci(c) - 1; l++)
            {
                g[consi++] =
                    (x[i] - x[i + 3]) * (x[i] - x[i + 3]) +
                    (x[i + 1] - x[i + 4]) * (x[i + 1] - x[i + 4]) +
                    (x[i + 2] - x[i + 5]) * (x[i + 2] - x[i + 5]);
                i += 3;
            }
            i += 3;
        }
    }

    //vari += num_break_constraints;
    // the clash constraints
    assert (consi == num_sphere_constraints + num_break_constraints);
    if (num_clash_constraints > 0)
    {
        int j;
        for (i=0; i < (num_loci-1)*3; i+=3)
        {
            for (j=i+3; j < num_loci*3; j+=3)
            {
                g[consi++] = (x[i]-x[j])*(x[i]-x[j]) + (x[i+1]-x[j+1])*(x[i+1]-x[j+1]) + (x[i+2]-x[j+2])*(x[i+2]-x[j+2]);
            }
        }
    }

    assert (consi == num_sphere_constraints + num_break_constraints + num_clash_constraints);

    // the distance constraints
    if (num_dist_constraints > 0)
    {
        INTERACTION_T *interaction = mygenome->get_first_interaction();
        while (interaction != NULL)
        {
            int l = mygenome->get_x_index(interaction->chrom1_index, interaction->locus1_index);
            int r = mygenome->get_x_index(interaction->chrom2_index, interaction->locus2_index);;
           // printf ("%d %d = %d ; %d %d = %d\n",
           //         interaction->chrom1_index, interaction->locus1_index, l,
           //         interaction->chrom2_index, interaction->locus2_index, r);

            g[consi] =
                (x[l]-x[r])*(x[l]-x[r]) +
                (x[l+1]-x[r+1])*(x[l+1]-x[r+1]) +
                (x[l+2]-x[r+2])*(x[l+2]-x[r+2]);
            consi++;
            interaction = interaction->next;
        }
    }

    if(this->mygenome->organism == BUDDINGYEAST){
      int chr = mygenome->get_chrom_index(12);
      if(chr != -1){
          int centro_x = mygenome->get_x_index(chr,
                                              this->mygenome->centromeres[chr] / bp_per_locus);
          g[consi] =  (x[centro_x]-CENTRO_CENTER)*(x[centro_x]-CENTRO_CENTER) +
                      x[centro_x+1]*x[centro_x+1] +
                      x[centro_x+2]*x[centro_x+2];
      }
    }
    consi++;


    // The rest is done only for the budding yeast genome
    if(this->mygenome->organism == BUDDINGYEAST){
      int chr12 = mygenome->get_chrom_index(12);
      if(chr12 != -1){
          for(Index i = 0; i < num_small_sphere_constraints - 1; i++)
          {
              int centro_x = mygenome->get_x_index(
                                            chr12,
                                            RDNA_POSITION / bp_per_locus + i);
              g[consi] = (
                  x[centro_x] - NUCLEO_CENTER) * (x[centro_x] - NUCLEO_CENTER) +
                  x[centro_x + 1] * x[centro_x + 1] +
                  x[centro_x + 2] * x[centro_x + 2];
              consi++;
          }
      }

      // FIXME this should be checked even if chromosome 12 was not loaded
      if(mygenome->get_chrom_index(12) != -1){
          assert (m == consi);
      }
      assert (n == xvari);
    }

    //printf ("Eval g OK\n");
    return true;
}

bool genome_ipopt_nlp::eval_grad_f(Index n, const Number* x, bool new_x,
                                   Number* grad_f)
{
  if(this->poisson_model){
    return this->grad_likelihood_power_law(n, x, new_x, grad_f);
  }else{
    return this->grad_mds(n, x, new_x, grad_f);
  }
}

// return the structure or values of the jacobian
bool genome_ipopt_nlp::eval_jac_g(Index n, const Number* x, bool new_x,
                           Index m, Index nele_jac, Index* iRow, Index *jCol,
                           Number* values)
{
    int ijac = 0;
    int xvari = num_loci * 3;
    if (values == NULL) {
        // return the structure of the jacobian

        // CAREFUL: I have to make sure I have the right order of constraints

        // first the sphere constraints
        // 0 <= d^2(p_i,0) <= 1, for all loci p_i
        for (Index i = 0; i < num_sphere_constraints * 3; i += 3)
        {
            iRow[ijac] = i / 3;
            jCol[ijac++] = i;

            iRow[ijac] = i / 3;
            jCol[ijac++] = i + 1;

            iRow[ijac] = i / 3;
            jCol[ijac++] = i + 2;
        }

        Index consi = num_sphere_constraints;

        // the break constraints
        if (num_break_constraints > 0)
        {
            int i=0;
            for (int c=0; c < mygenome->get_num_chroms(); c++)
            {
                for (int l=0; l < mygenome->get_num_loci(c)-1; l++)
                {
                    iRow[ijac] = consi;
                    jCol[ijac++] = i;

                    iRow[ijac] = consi;
                    jCol[ijac++] = i + 1;

                    iRow[ijac] = consi;
                    jCol[ijac++] = i + 2;

                    iRow[ijac] = consi;
                    jCol[ijac++] = i + 3;

                    iRow[ijac] = consi;
                    jCol[ijac++] = i + 4;

                    iRow[ijac] = consi;
                    jCol[ijac++] = i + 5;

                    consi++;
                    i+=3;
                    //g[consi++] =
                    //    (x[i]-x[i+3])*(x[i]-x[i+3]) + (x[i+1]-x[i+4])*(x[i+1]-x[i+4]) + (x[i+2]-x[i+5])*(x[i+2]-x[i+5])
                }
                i += 3;
            }
        }
        assert (consi == num_sphere_constraints + num_break_constraints);

        if (num_clash_constraints > 0)
        {
            // the clash constraints
            int j;
            //vari += num_break_constraints;
            for (int i=0; i < (num_loci - 1) * 3; i += 3)
            {
                for(j = i + 3; j < num_loci * 3; j += 3)
                {
                    iRow[ijac] = consi;
                    jCol[ijac++] = i;

                    iRow[ijac] = consi;
                    jCol[ijac++] = i + 1;

                    iRow[ijac] = consi;
                    jCol[ijac++] = i + 2;

                    iRow[ijac] = consi;
                    jCol[ijac++] = j;

                    iRow[ijac] = consi;
                    jCol[ijac++] = j + 1;

                    iRow[ijac] = consi;
                    jCol[ijac++] = j + 2;

                    consi++;
                    //g[consi++] = (x[i]-x[j])*(x[i]-x[j]) + (x[i+1]-x[j+1])*(x[i+1]-x[j+1]) + (x[i+2]-x[j+2])*(x[i+2]-x[j+2]);
                }
            }
        }
        assert (consi == num_sphere_constraints + num_break_constraints + num_clash_constraints);

        // the distance constraints
        if (num_dist_constraints > 0)
        {
            INTERACTION_T *interaction = mygenome->get_first_interaction();
            while (interaction != NULL)
            {
                int l = mygenome->get_x_index(interaction->chrom1_index,
                                              interaction->locus1_index);
                int r = mygenome->get_x_index(interaction->chrom2_index,
                                              interaction->locus2_index);

                iRow[ijac] = consi;
                jCol[ijac++] = l;

                iRow[ijac] = consi;
                jCol[ijac++] = l + 1;

                iRow[ijac] = consi;
                jCol[ijac++] = l + 2;

                iRow[ijac] = consi;
                jCol[ijac++] = r;

                iRow[ijac] = consi;
                jCol[ijac++] = r + 1;

                iRow[ijac] = consi;
                jCol[ijac++] = r + 2;

                interaction = interaction->next;
                consi++;
            }
        }

        if(this->mygenome->organism == BUDDINGYEAST)
        {
            for (Index i=consi; i < this->num_small_sphere_constraints * 3; i += 3)
            {
                iRow[ijac] = i / 3;
                jCol[ijac++] = i;

                iRow[ijac] = i / 3;
                jCol[ijac++] = i + 1;

                iRow[ijac] = i / 3;
                jCol[ijac++] = i + 2;
            }

            int i=12;
            int chr = mygenome->get_chrom_index(i);
            if(chr != -1){
                int centro_x = mygenome->get_x_index(chr, this->mygenome->centromeres[chr]/bp_per_locus);

                iRow[ijac] = consi;
                jCol[ijac++] = centro_x;

                iRow[ijac] = consi;
                jCol[ijac++] = centro_x+1;

                iRow[ijac] = consi;
                jCol[ijac++] = centro_x+2;
                consi++;
            }
        }

        // FIXME fails if chromosome 12 is not loaded.
        if(this->mygenome->organism == BUDDINGYEAST){

          int chr12 = mygenome->get_chrom_index(12);
          if(chr12 != -1){
              for (Index i=0; i < num_small_sphere_constraints - 1; i++)
              {
                  int centro_x = mygenome->get_x_index(chr12, RDNA_POSITION/bp_per_locus + i);

                  iRow[ijac] = consi;
                  jCol[ijac++] = centro_x;

                  iRow[ijac] = consi;
                  jCol[ijac++] = centro_x+1;

                  iRow[ijac] = consi;
                  jCol[ijac++] = centro_x+2;

                  consi++;
              }
          }

          // FIXME we should check the number of constraints even if chromosome 12 is
          // not loaded.
          if(chr12 != -1){
              assert (consi == m);
          }
        }
    }
    else {
        // return the values of the jacobian of the constraints

        for (Index i=0; i<num_sphere_constraints*3; i+=3)
        {
            values[ijac++] = 2*x[i];    // i/3, i
            values[ijac++] = 2*x[i+1];  // i/3, i+1
            values[ijac++] = 2*x[i+2];  // i/3, i+2
        }

        Index consi = num_sphere_constraints;
        if (num_break_constraints > 0)
        {
            // the break constraints
            int i=0;
            for (int c=0; c < mygenome->get_num_chroms(); c++)
            {
                for (int l=0; l < mygenome->get_num_loci(c)-1; l++)
                {
                    values[ijac++] = 2*(x[i]-x[i+3]);   // consi, i
                    values[ijac++] = 2*(x[i+1]-x[i+4]);   // consi, i+1
                    values[ijac++] = 2*(x[i+2]-x[i+5]);   // consi, i+2
                    values[ijac++] = -2*(x[i]-x[i+3]);   // consi, i+3
                    values[ijac++] = -2*(x[i+1]-x[i+4]);   // consi, i+4
                    values[ijac++] = -2*(x[i+2]-x[i+5]);   // consi, i+5
                    consi++;
                    i+=3;
                    //g[consi++] =
                    //    (x[i]-x[i+3])*(x[i]-x[i+3]) + (x[i+1]-x[i+4])*(x[i+1]-x[i+4]) + (x[i+2]-x[i+5])*(x[i+2]-x[i+5])
                }
                i+=3;
            }
        }

        assert (consi == num_sphere_constraints + num_break_constraints);
        if (num_clash_constraints > 0)
        {
            // the clash constraints
            for (Index i=0; i < (num_loci-1)*3; i+=3)
            {
                for (Index j=i+3; j < num_loci*3; j+=3)
                {
                    values[ijac++] = 2*(x[i]-x[j]);     // consi, i
                    values[ijac++] = 2*(x[i+1]-x[j+1]); // consi, i+1
                    values[ijac++] = 2*(x[i+2]-x[j+2]); // consi, i+2

                    values[ijac++] = -2*(x[i]-x[j]);    // consi, j
                    values[ijac++] = -2*(x[i+1]-x[j+1]);    // consi, j+1
                    values[ijac++] = -2*(x[i+2]-x[j+2]);    // consi, j+2

                    consi++;
                    //g[consi++] = (x[i]-x[j])*(x[i]-x[j]) + (x[i+1]-x[j+1])*(x[i+1]-x[j+1]) + (x[i+2]-x[j+2])*(x[i+2]-x[j+2]);
                }
            }
        }

        assert (consi == num_sphere_constraints + num_break_constraints + num_clash_constraints);

        if (num_dist_constraints > 0)
        {
            INTERACTION_T *interaction = mygenome->get_first_interaction();
            while (interaction != NULL)
            {

                //printf ("Begin int values: ijac = %d, consi = %d\n", ijac, consi);
                int l = mygenome->get_x_index(interaction->chrom1_index,
                                              interaction->locus1_index);
                int r = mygenome->get_x_index(interaction->chrom2_index,
                                              interaction->locus2_index);
                values[ijac++] = 2*(x[l]-x[r]);
                values[ijac++] = 2*(x[l+1]-x[r+1]);
                values[ijac++] = 2*(x[l+2]-x[r+2]);
                values[ijac++] = -2*(x[l]-x[r]);
                values[ijac++] = -2*(x[l+1]-x[r+1]);
                values[ijac++] = -2*(x[l+2]-x[r+2]);


                //g[consi] = (x[l]-x[r])*(x[l]-x[r]) +
                //          (x[l+1]-x[r+1])*(x[l+1]-x[r+1]) +
                //          (x[l+2]-x[r+2])*(x[l+2]-x[r+2]) - x[xvari]*x[xvari];

                interaction = interaction->next;
                consi++;
            }
        }

        if(this->mygenome->organism == BUDDINGYEAST){
            int i=12;
            int chr = mygenome->get_chrom_index(i);
            if(chr != -1){
                int centro_x = mygenome->get_x_index(chr, this->mygenome->centromeres[chr]/bp_per_locus);

                values[ijac++] = 2*(x[centro_x]-CENTRO_CENTER);
                values[ijac++] = 2*x[centro_x+1];
                values[ijac++] = 2*x[centro_x+2];
                consi++;
            }

          int chr12 = mygenome->get_chrom_index(12);
          if(chr12 != -1){
              for (Index i=0; i < num_small_sphere_constraints - 1; i++)
              {
                  int centro_x = mygenome->get_x_index(chr12, RDNA_POSITION/bp_per_locus + i);
                  values[ijac++] = 2*(x[centro_x]-NUCLEO_CENTER);
                  values[ijac++] = 2*x[centro_x+1];
                  values[ijac++] = 2*x[centro_x+2];
                  consi++;
              }
          }

          // FIXME should check num constraint even if it is chromsome 12
          if(chr12 != -1){
              assert(consi == m);
          }

        }
    }

    if(this->mygenome->organism == BUDDINGYEAST){
        //printf ("ijac=%d, nele_jac=%d\n", ijac, nele_jac);
        if(mygenome->get_chrom_index(12) != -1){
            assert (ijac == nele_jac);
        }
    }

    assert (n == xvari);

    return true;
}


//return the structure or values of the hessian
bool genome_ipopt_nlp::eval_h(Index n, const Number* x, bool new_x,
                       Number obj_factor, Index m, const Number* lambda,
                       bool new_lambda, Index nele_hess, Index* iRow,
                       Index* jCol, Number* values)
{
    printf ("Eval H OK\n");
    return false;   // if it's Quasi-Newton, just return false
}

void genome_ipopt_nlp::finalize_solution(SolverReturn status,
                                  Index n, const Number* x, const Number* z_L,
                                  const Number* z_U,
                                  Index m, const Number* g,
                                  const Number* lambda,
                                  Number obj_value,
				  const IpoptData* ip_data,
				  IpoptCalculatedQuantities* ip_cq)
{
    // here is where we would store the solution to variables, or write to a file, etc
    // so we could use the solution.

    printf("\n\nObjective value\n");
    printf("f(x*) = %e\n", obj_value);

    // now pass the info to mygenome and print pdb file
    if( status == Solve_Succeeded || status == Solved_To_Acceptable_Level){
      mygenome->set_coords((double*)x);
      char output_filename[1000];
      sprintf (output_filename, "%s", output_pdb); // output_pdb already has .pdb
      mygenome->print_pdb_genome(output_filename);
      sprintf (output_filename, "%s.txt", output_pdb);
      mygenome->save_txt(output_filename);
    }
}


bool genome_ipopt_nlp::eval_h_likelihood_power_law(Index n, const Number* x, bool new_x,
                       Number obj_factor, Index m, const Number* lambda,
                       bool new_lambda, Index nele_hess, Index* iRow,
                       Index* jCol, Number* values)
{
    // The hessian is dense
    for(Index i=0; i < this->num_loci * 3; i++){
        values[i] = 0;
    }

    if(num_interactions > 0){
        INTERACTION_T * interaction = this->mygenome->get_first_interaction();
        while (interaction != NULL)
        {
            int l = this->mygenome->get_x_index(interaction->chrom1_index,
                                                interaction->locus1_index);
            int r = this->mygenome->get_x_index(interaction->chrom2_index,
                                                interaction->locus2_index);

            double dis = sqrt((x[l] - x[r]) * (x[l] - x[r]) +
                              (x[l + 1] - x[r + 1]) * (x[l + 1] - x[r + 1]) +
                              (x[l + 2] - x[r + 2]) * (x[l + 2] - x[r + 2]));

            // values[l] 
        }
    }

}




bool genome_ipopt_nlp::write_cplex_input (char *cplex_input_filename)
{
    FILE* output;
    if (open_file(cplex_input_filename, "w", TRUE, "CPLEX", "CPLEX", &output) == 0) {
        exit(1);
    }
    fprintf (output, "MINIMIZE\n");

    // first compute the quadratic matrix (qmatrix) of the objective function

    int **qmatrix = new int* [num_variables];
    for (int i=0; i < num_variables; i++)
    {
        qmatrix[i] = new int[num_variables];
        for (int j=i; j < num_variables; j++)
        {
            qmatrix[i][j] = 0;
        }
    }


    // First, I have to figure out what's the coefficient of every x_i * x_j pair of variables
    // Cplex doesn't like as input x1^2 + x1^2, is has to be 2 x1^2.
    if (num_interactions > 0)
    {
        INTERACTION_T *interaction = mygenome->get_first_interaction();
        while (interaction != NULL)
        {
            int l = mygenome->get_x_index(interaction->chrom1_index, interaction->locus1_index);
            int r = mygenome->get_x_index(interaction->chrom2_index, interaction->locus2_index);
            int f;
            if (use_weights == 1)
            {
                f = interaction->freq;
                printf ("NOT IMPLEMENTED YET\n");
                exit(1);
            }
            else
            {
                // make sure l < r;
                if (l > r)
                {
                    int temp = l;
                    l = r;
                    r = temp;
                }
                qmatrix[l][l] += 1;         qmatrix[l][r] += -2;        qmatrix[r][r] += 1;
                qmatrix[l+1][l+1] += 1;     qmatrix[l+1][r+1] += -2;    qmatrix[r+1][r+1] += 1;
                qmatrix[l+2][l+2] += 1;     qmatrix[l+2][r+2] += -2;    qmatrix[r+2][r+2] += 1;
            }
            interaction = interaction->next;
        }
    }

    // add the break and clash lower bound constraints in the objective, so increment the corresponding coefficients
    // This didn't work, CPLEX gives the following message:
    //  Dual infeasible due to empty column 's0'.
    //      Presolve - Unbounded or infeasible.

    // write the objective function
    fprintf (output, "\t[ ");
    int first = 1;
    int this_row = 0;
    for (int i=0; i < num_variables; i++)
    {
        for (int j=i; j < num_variables; j++)
        {
            if (qmatrix[i][j] != 0)
            {
                fprintf (output, "%+d x%d * x%d ", qmatrix[i][j], i, j);
                this_row++;
                if (this_row == 10)
                {
                    fprintf (output, "\n\t");
                    this_row = 0;
                }
            }
        }
    }
    // NOTE: adding y0*y1 to the objective makes it not be positive semi-definite
    //fprintf (output, " + y0 * y1 ");


    fprintf (output, "\t ] / 2 \n\t");

    // Now add the slack variables s

    // deallocate the space for qmatrix;
    for (int i=0; i < num_variables; i++)   delete [] qmatrix[i];
    delete [] qmatrix;

    fprintf (output, "\nSUBJECT TO\n");

    // NOW WRITE THE CONSTRAINTS
    // first the sphere constraints
    // 0 <= d^2(p_i,0) <= 1, for all loci p_i
    for (int i=0; i<num_sphere_constraints*3; i+=3)
    {
        // the lower bound doesn't make sense, it's true anyway
        //fprintf (output, "\tsc_l%d: [ x%d^2 + x%d^2 + x%d^2 ] >= 0\n", i/3, i, i+1, i+2);
        fprintf (output, "\tsc_u%d: [ x%d^2 + x%d^2 + x%d^2 ] <= %e\n", i/3, i, i+1, i+2, sphere_radius*sphere_radius);
    }

    // the break constraints
    int i=0;
    int bvar = 0;
    int yvar = 0;
    if (num_break_constraints > 0)
    {
        for (int c=0; c < mygenome->get_num_chroms(); c++)
        {
            for (int l=0; l < mygenome->get_num_loci(c)-1; l++)
            {
                fprintf (output, "\n");

                fprintf (output, "\tbc_l1_%d: y%d + y%d + y%d + y%d + y%d + y%d >= %e\n", i/3,
                    yvar, yvar+1, yvar+2, yvar+3, yvar+4, yvar+5, min_dist);
                fprintf (output, "\tbc_l2_%d: x%d - x%d - y%d + y%d = 0\n", i/3, i, i+3, yvar, yvar+1);
                fprintf (output, "\tbc_l3_%d: x%d - x%d - y%d + y%d = 0\n", i/3, i+1, i+4, yvar+2, yvar+3);
                fprintf (output, "\tbc_l4_%d: x%d - x%d - y%d + y%d = 0\n", i/3, i+2, i+5, yvar+4, yvar+5);
                // add binary variables b
                fprintf (output, "\tbc_l5_%d: y%d - %e b%d <= 0\n",  i/3, yvar, sphere_radius, bvar);
                fprintf (output, "\tbc_l6_%d: y%d + %e b%d <= %e\n", i/3, yvar+1, sphere_radius, bvar, sphere_radius);
                fprintf (output, "\tbc_l7_%d: y%d - %e b%d <= 0\n",  i/3, yvar+2, sphere_radius, bvar+1);
                fprintf (output, "\tbc_l8_%d: y%d + %e b%d <= %e\n", i/3, yvar+3, sphere_radius, bvar+1, sphere_radius);
                fprintf (output, "\tbc_l9_%d: y%d - %e b%d <= 0\n",  i/3, yvar+4, sphere_radius, bvar+2);
                fprintf (output, "\tbc_l10_%d: y%d + %e b%d <= %e\n",i/3, yvar+5, sphere_radius, bvar+2, sphere_radius);
                yvar+=6;
                bvar+=3;

                fprintf (output, "\tbc_u_%d: [ x%d^2 - 2 x%d*x%d + x%d^2 ", i/3, i, i, i+3, i+3);
                fprintf (output, " + x%d^2 - 2 x%d*x%d + x%d^2 ", i+1, i+1, i+4, i+4);
                fprintf (output, " + x%d^2 - 2 x%d*x%d + x%d^2 ] <= %e\n", i+2, i+2, i+5, i+5, max_dist*max_dist);

                i+=3;
            }
            i+=3;
        }
    }

    // NOW ADD THE BOUNDS ON THE VARIABLES
    fprintf (output, "\nBOUNDS\n");
    for (int i=0; i < num_variables; i++)
    {
        fprintf (output, "\t-%e <= x%d <= %e\n", sphere_radius, i, sphere_radius);
    }

    fprintf (output, "\nBINARY\n");
    for (int i=0; i < bvar; i++)
    {
        fprintf (output, "\tb%d\n", i);
    }

    fprintf (output, "\nEND\n");
    fclose(output);
}
