#pragma once

#include <chrono> // std::chrono::milliseconds
#include <thread> // std::this_thread::sleep_for

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
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  /**
   * Search for the 10 closest vectors in the dataset.
   *
   * The standard for considering the distance between vectors is the **L2**
   * distance: \[||x - y||_2\]
   *
   * @param [in] query The vector to search for. This vector should be of `d`
   * dimensional.
   * @param [out] res The place to put the results. Memory is pre-allocated.
   */
  inline void search(const std::vector<float>& query, int* res) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
};

static_assert(IsSolution<Solution>);
