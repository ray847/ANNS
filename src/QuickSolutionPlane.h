#pragma once

#include "QuickSolutionMatrix.h" // solution::Vec

namespace solution {
class Plane {
public:
  /* Variables */
  Vec coeff_;
  float intercept_;
  /* Constructor */
  Plane(size_t dim) : coeff_(dim) {}
  Plane(const Vec& coeff, float intercept): 
    coeff_(coeff),
    intercept_(intercept) {}
  /* Functions */
  auto dim() const {
    return coeff_.n();
  }
  float operator()(const Vec& vec) const {
    float res = intercept_;
    for (size_t i = 0; i < dim(); ++i) {
      res += coeff_[i][0] * vec[i][0];
    }
    return res;
  }
};

} // namespace solution
