#pragma once

#include <vector> // std::vector
#include <memory> // std::unique_ptr

#include "SolutionConcept.h" // IsSolution
#include "QuickSolutionSolver.h" // solution::Solver

/**
 * An example of what a solution class must contain.
 * 
 * Functions:
 * `void build(int d, const std::vector<float>& base)`
 * `void search(const std::vector<float>& base, int d)`
 */
class Solution {
public:
  using SolverT = solution::Solver<1>;
  /* Functions */
  /**
   * Load & preprocessthe vector dataset.
   * @param d The number of dimensions for each vector.
   * @param base The dataset in with all vectors concatenated.
   */
  inline void build(int d, const std::vector<float>& base) {
    solution::Mat base_mat(d, base.size() / d);
    for (size_t i = 0; i < d; ++i) {
      for (size_t j = 0; j < base.size() / d; ++j) {
        base_mat[i][j] = base[i + j * d];
      }
    }
    solver_ = std::make_unique<SolverT>(base_mat);
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
  inline void search(const std::vector<float>& query, int* res) {}
private:
  std::unique_ptr<SolverT> solver_;
};
static_assert(IsSolution<Solution>);

