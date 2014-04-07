/**
 * This module defines the mapping from IF to distances.
 *
 **/

#include "frequencies.hpp"
#include <iostream>
#include "utils.h"
#define MAX_LINE 1000 // Maximum number of characters in one line.

using namespace std;


Mapping::Mapping(char * wish_dist_filename)
{
  // Let's read the mapping from a file
  FILE * mapping_file;
  if(open_file(wish_dist_filename,
               "r", TRUE,
               "wishdist", "wishdist",
               &mapping_file) == 0){
    exit(1);
  }

  double prev_freq = 0;
  float dist, pdist;
  float freq;
  char one_line[MAX_LINE];

  this->min_key = NULL;
  this->max_key = NULL;

  while(1){
    if(fgets(one_line, MAX_LINE, mapping_file) == NULL){
      break;
    }
    int num_scanned = sscanf(one_line, "%e %e", &freq, &dist);
    if(this->min_key == NULL or freq < this->min_key){
      this->min_key = (double) freq;
    }

    if(this->max_key == NULL or freq > this->max_key){
      this->max_key = (double) freq;
    } 
    this->distances[(double) freq] = dist;
    prev_freq = freq;
  }
}


Mapping::Mapping(){
  this->min_key = 1;
  this->max_key = 248;

  this->distances[248.] = 19.230769;
  this->distances[170.] = 38.221154;
  this->distances[153.] = 57.211538;
  this->distances[112.] = 76.201923;
  this->distances[103.] = 95.192308;
  this->distances[84.] = 114.182692;
  this->distances[67] = 133.173077;
  this->distances[66] = 152.163462;
  this->distances[54] = 171.153846;
  this->distances[48] = 190.144231;
  this->distances[45] = 209.134615;
  this->distances[39] = 228.125000;
  this->distances[36] = 247.115385;
  this->distances[32] = 266.105769;
  this->distances[29] = 285.096154;
  this->distances[28] = 304.086538;
  this->distances[25] = 323.076923;
  this->distances[23] = 342.067308;
  this->distances[20] = 380.048077;
  this->distances[18] = 418.028846;
  this->distances[17] = 437.019231;
  this->distances[16] = 456.009615;
  this->distances[15] = 475.000000;
  this->distances[14] = 493.990385;
  this->distances[13] = 531.971154;
  this->distances[12] = 550.961538;
  this->distances[11] = 588.942308;
  this->distances[10] = 626.923077;
  this->distances[9] = 683.894231;
  this->distances[8] = 740.865385;
  this->distances[7] = 835.817308;
  this->distances[6] = 949.759615;
  this->distances[5] = 1158.653846;
  this->distances[4] = 1272.596154;
  this->distances[3] = 1519.471154;
  this->distances[1] = 1538.461538;
}


float Mapping::get_wish_dist(double freq){

  if(freq < this->min_key){
    return NULL;
  }

  std::map<double, float>::iterator it;
  float old = 0;
  for(it = this->distances.begin(); it != this->distances.end(); it++){
    if(freq > it->first){
      old = it->second;
      continue;
    }else{
      return old;
    }
  }

  return old;
}
