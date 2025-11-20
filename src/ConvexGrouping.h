#pragma once

#include <memory> // std::unique_ptr
#include <algorithm> // std::copy_n

#include "Global.h" // global::kCRITERION
#include "SolutionConcept.h" // IsSolution
#include "ConvexGroupingSolver.h" // solution::Solver

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
    solver = std::make_unique<solution::Solver<>>(d, base);
    solver->build();
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
    solution::Vec<float> query_vec(query.size(), 1, query.data());
    auto res_tmp = solver->search(query_vec);
    std::copy_n(res_tmp.begin(), global::kCRITERION, res);
  }
private:
  std::unique_ptr<solution::Solver<>> solver;
};

static_assert(IsSolution<Solution>);
