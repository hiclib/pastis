#include <stdio.h>
#include <string.h>
#include "genome_ipopt_nlp.hpp"
#include "metric.hpp"

using namespace Ipopt;

Solver::Solver(GENOME * mygenome, float min_clash_dist,
                 float min_clash_dist_inter, char * output_pdb,
                 int sphere_radius, int use_weights,
                 int bp_per_locus, int rDNA_frequency_normalizer,
                 int weight_of_inter, float weight_unseen){
  this->mygenome = mygenome;
  this->min_clash_dist = min_clash_dist;
  this->min_clash_dist_inter = min_clash_dist_inter;
  this->output_pdb = output_pdb;
  this->sphere_radius = sphere_radius;
  this->use_weights = use_weights;
  this->bp_per_locus = bp_per_locus;
  this->rDNA_frequency_normalizer = rDNA_frequency_normalizer;
  this->weight_of_inter = weight_of_inter;
  this->weight_unseen = weight_unseen;
}

int Solver::fit(int poisson_model, double alpha, double beta, int max_iter){
  
  cout << "fitting metric phase" << endl;

  this->mynlp = new genome_ipopt_nlp(
                      mygenome, min_clash_dist, min_clash_dist_inter,
                      output_pdb, sphere_radius, use_weights,
                      bp_per_locus, rDNA_frequency_normalizer,
                      (float) weight_of_inter, weight_unseen, (bool) poisson_model, alpha,
                      beta);

  this->app = new IpoptApplication();

  // Change some options
  // Note: The following choices are only examples, they might not be
  //       suitable for your optimization problem.
  this->app->Options()->SetNumericValue("tol", 1e-4);
  app->Options()->SetNumericValue("acceptable_tol", 1e-1);
  app->Options()->SetNumericValue("constr_viol_tol", 1e-1);

  this->app->Options()->SetStringValue("mu_strategy", "adaptive");
  this->app->Options()->SetStringValue("output_file", "ipopt.out");
  this->app->Options()->SetStringValue("hessian_approximation",
                                       "limited-memory");
  this->app->Options()->SetIntegerValue("max_iter", max_iter);
  // The following overwrites the default name (ipopt.opt) of the
  // options file
  // app->Options()->SetStringValue("option_file_name", "hs071.opt");

  // Intialize the IpoptApplication and process the options
  ApplicationReturnStatus status;
  status = this->app->Initialize();
  if (status != Solve_Succeeded) {
    printf("\n\n*** Error during initialization!\n");
  }

  // Create a new instance of IpoptApplication
  //  (use a SmartPtr, not raw)
  // Ask Ipopt to solve the problem
  status = this->app->OptimizeTNLP(this->mynlp);

  //mygenome->save_interaction_achievement (interaction_achievement_matrix);
  //mygenome->save_optimal_distances (optimal_interaction_matrix);
}
