#pragma once

#include <math.h> // sqrt

#include <algorithm> // 

#include "SolutionConcept.h" // IsSolution

/**
 * An example of what a solution class must contain.
 * 
 * Functions:
 * `void build(int d, const std::vector<float>& base)`
 * `void search(const std::vector<float>& base, int d)`
 */
class Solution {
public:
  /**
   * Load & preprocessthe vector dataset.
   * @param d The number of dimensions for each vector.
   * @param base The dataset in with all vectors concatenated.
   */
  inline void build(int d, const std::vector<float>& base) {
    d_ = d;
    n_ = base.size() / d;
    base_ = base;
  }
  /**
   * Search for the 10 closest vectors in the dataset.
   *
   * The standard for considering the distance between vectors is the **L2**
   * distance: $$ ||x - y||_2 $$
   *
   * @param [in] query The vector to search for. This vector should be of `d`
   * dimensional.
   * @param [out] res The place to put the results. Memory is pre-allocated.
   */
  inline void search(const std::vector<float>& query, int* res) {
    std::vector<std::pair<float, size_t>> dis;
    for (size_t i = 0; i < n_; ++i) {
      dis.emplace_back(L2(query, i), i);
    }
    std::partial_sort(dis.begin(), dis.begin() + 10, dis.end());
    for (int i = 0; i < 10; ++i) {
      res[i] = dis[i].second;
    }
  }
private:
  /* Variables */
  size_t d_;
  size_t n_;
  std::vector<float> base_;
  /* Functions */
  float L2(const std::vector<float>& vec, size_t n) {
    float tmp = 0;
    for (size_t i = 0; i < d_; ++i) {
      tmp += (vec[i] - base_[n * d_ + i]) * (vec[i] - base_[n * d_ + i]);
    }
    return sqrtf(tmp);
  }
};

static_assert(IsSolution<Solution>);
