// Copyright (C) 2005, 2007 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Common Public License.
//
// $Id: genome_ipopt_nlp.hpp,v 1.4 2010/05/12 03:05:11 andrones Exp $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2005-08-09

#ifndef __GENOME_IPOPT_NLP_HPP__
#define __GENOME_IPOPT_NLP_HPP__

#include "IpTNLP.hpp"
#include "genome_model.hpp"

using namespace Ipopt;

/** C++ Example NLP for interfacing a problem with IPOPT.
 *  HS071_NLP implements a C++ example of problem 71 of the
 *  Hock-Schittkowski test suite. This example is designed to go
 *  along with the tutorial document and show how to interface
 *  with IPOPT through the TNLP interface.
 *
 * Problem hs071 looks like this
 *
 *     min   x1*x4*(x1 + x2 + x3)  +  x3
 *     s.t.  x1*x2*x3*x4                   >=  25
 *           x1**2 + x2**2 + x3**2 + x4**2  =  40
 *           1 <=  x1,x2,x3,x4  <= 5
 *
 *     Starting point:
 *        x = (1, 5, 5, 1)
 *
 *     Optimal solution:
 *        x = (1.00000000, 4.74299963, 3.82114998, 1.37940829)
 *
 *
 */
class genome_ipopt_nlp : public TNLP
{
public:
  /** default constructor */
  genome_ipopt_nlp();

  // my additional constructor
  genome_ipopt_nlp(GENOME* mygenome, double min_clash_dist,
                   double min_clash_dist_inter, char* output_pdb,
                   double sphere_radius, int use_frequencies, int bp_per_locus,
                   int rDNA_frequency_normalizer, float weight_of_inter,
                   double weight_unseen, bool poisson_model, double alpha,
                   double beta);

  /** default destructor */
  virtual ~genome_ipopt_nlp();

  /**@name Overloaded from TNLP */
  //@{
  /** Method to return some info about the nlp */
  virtual bool get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
                            Index& nnz_h_lag, IndexStyleEnum& index_style);

  /** Method to return the bounds for my problem */
  virtual bool get_bounds_info(Index n, Number* x_l, Number* x_u,
                               Index m, Number* g_l, Number* g_u);

  /** Method to return the starting point for the algorithm */
  virtual bool get_starting_point(Index n, bool init_x, Number* x,
                                  bool init_z, Number* z_L, Number* z_U,
                                  Index m, bool init_lambda,
                                  Number* lambda);

  /** Method to return the objective value */
  virtual bool eval_f(Index n, const Number* x, bool new_x, Number& obj_value);
  virtual bool likelihood(Index n, const Number* x, bool new_x, Number& obj_value);
  virtual bool likelihood_power_law(Index n, const Number* x,
                                    bool new_x, Number& obj_value);
  virtual bool mds(Index n, const Number* x, bool new_x, Number& obj_value);

  /** Method to return the gradient of the objective */
  virtual bool eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f);
  virtual bool grad_likelihood(Index n, const Number* x, bool new_x, Number* grad_f);
  virtual bool grad_likelihood_power_law(Index n, const Number* x, bool new_x,
                                         Number* grad_f);
  virtual bool grad_mds(Index n, const Number* x, bool new_x, Number* grad_f);

  /** Method to return the constraint residuals */
  virtual bool eval_g(Index n, const Number* x, bool new_x, Index m, Number* g);

  /** Method to return:
   *   1) The structure of the jacobian (if "values" is NULL)
   *   2) The values of the jacobian (if "values" is not NULL)
   */
  virtual bool eval_jac_g(Index n, const Number* x, bool new_x,
                          Index m, Index nele_jac, Index* iRow, Index *jCol,
                          Number* values);

  /** Method to return:
   *   1) The structure of the hessian of the lagrangian (if "values" is NULL)
   *   2) The values of the hessian of the lagrangian (if "values" is not NULL)
   */
  virtual bool eval_h(Index n, const Number* x, bool new_x,
                      Number obj_factor, Index m, const Number* lambda,
                      bool new_lambda, Index nele_hess, Index* iRow,
                      Index* jCol, Number* values);
  virtual bool eval_h_likelihood_power_law(Index n, const Number* x, bool new_x,
                       Number obj_factor, Index m, const Number* lambda,
                       bool new_lambda, Index nele_hess, Index* iRow,
                       Index* jCol, Number* values);

  //@}

  /** @name Solution Methods */
  //@{
  /** This method is called when the algorithm is complete so the TNLP can store/write the solution */
  virtual void finalize_solution(SolverReturn status,
                                 Index n, const Number* x, const Number* z_L, const Number* z_U,
                                 Index m, const Number* g, const Number* lambda,
                                 Number obj_value,
				 const IpoptData* ip_data,
				 IpoptCalculatedQuantities* ip_cq);
  //@}


  bool write_cplex_input (char *cplex_input_filename);
  void set_alpha(float alpha);
  void set_beta(float beta);

private:
  /**@name Methods to block default compiler methods.
   * The compiler automatically generates the following three methods.
   *  Since the default compiler implementation is generally not what
   *  you want (for all but the most simple classes), we usually
   *  put the declarations of these methods in the private section
   *  and never implement them. This prevents the compiler from
   *  implementing an incorrect "default" behavior without us
   *  knowing. (See Scott Meyers book, "Effective C++")
   *
   */
  //@{
  //  HS071_NLP();
    genome_ipopt_nlp(const genome_ipopt_nlp&);
    genome_ipopt_nlp& operator=(const genome_ipopt_nlp&);

    int is_inter_arms (INTERACTION_T *interaction);
    // return 1 if this interaction is between 2 arms (of the same or different chrom)
    // return 0 if this is intra within the same arm

    int is_inter_centro_telo (INTERACTION_T *interaction);
    // return 1 if this interaction is between the centromere area and the telomere area of the same chromosome
    // return 0 if it's not

    GENOME *mygenome;
    char *output_pdb;

    int num_variables;
    int num_loci;
    int num_interactions;
    int use_weights;
    float weight_of_inter;    // the weight used, in case use_weights is 5 or 6
    int weighted_objective;

    int num_constraints;
    int num_sphere_constraints;
    int num_break_constraints;
    int num_clash_constraints;
    int num_small_sphere_constraints;

    int num_dist_constraints;
    double min_dist;
    double max_dist;
    double min_clash_dist;
    double min_clash_dist_inter;
    bool usedelta;
    double sphere_radius;
    int bp_per_locus;

    int iteration;
    int rDNA_frequency_normalizer;
    double weight_unseen;

    // For Poisson Model
    double alpha;
    double beta;

    bool poisson_model;

  //@}
};


#endif
