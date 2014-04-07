#ifndef METRIC_HPP
#define METRIC_HPP
#include <stdio.h>
#include "genome_model.hpp"
#endif

#include "IpIpoptApplication.hpp"
#include "genome_ipopt_nlp.hpp"

class Solver{
  public:
    Solver(GENOME * mygenome, float min_clash_dist,
            float min_clash_dist_inter, char * output_pdb,
            int sphere_radius, int use_weights,
            int bp_per_locus, int rDNA_frequency_normalizer,
            int weight_of_inter, float weight_unseen);
    ~Solver();
    SmartPtr<TNLP> * init_metric(GENOME * mygenome, float min_clash_dist,
                      float min_clash_dist_inter, char * output_pdb,
                      int sphere_radius, int use_weights,
                      int bp_per_locus, int rDNA_frequency_normalizer,
                      float weight_of_inter, float weight_unseen);

    int fit(int poisson_model, double alpha, double beta, int max_iter);

  private:
    GENOME * mygenome;
    float min_clash_dist;
    float min_clash_dist_inter;
    char * output_pdb;
    int sphere_radius;
    int use_weights;
    int bp_per_locus;
    int rDNA_frequency_normalizer;
    int weight_of_inter;
    float weight_unseen;
    genome_ipopt_nlp* mynlp;
    IpoptApplication * app;
};
