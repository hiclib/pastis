#include <vector>
#include <map>

class Mapping{
  public:

    Mapping();
    Mapping(char *  wish_dist_filename);
    float get_wish_dist(double freq);

  private:
    std::map<double, float> distances;
    double min_key;
    double max_key;

};
