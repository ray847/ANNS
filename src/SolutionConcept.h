#pragma once

#include <vector> // std::vector

/**
 * This concept set all the necesssary requirment for a valid Solution class.
 *
 * Required functions:
 * `void build(int d, const std::vector<float>& base)`
 * `void search(const std::vector<float>& base, int d)`
 *
 * Check the declaration of `ExampleSolution` to see what they do respectively.
 * @see ExampleSolution
 */
template<typename SolutionT>
concept IsSolution = requires (SolutionT solution,
                               int d, const std::vector<float>& base,
                               const std::vector<float>& query, int* res) {
  {solution.build(d, base)} -> std::same_as<void>;
  {solution.search(query, res)} -> std::same_as<void>;
};

/**
 * An example of what a solution class must contain.
 * 
 * Functions:
 * `void build(int d, const std::vector<float>& base)`
 * `void search(const std::vector<float>& base, int d)`
 */
class ExampleSolution {
public:
  /**
   * Load & preprocessthe vector dataset.
   * @param d The number of dimensions for each vector.
   * @param base The dataset in with all vectors concatenated.
   */
  void build(int d, const std::vector<float>& base);
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
  void search(const std::vector<float>& query, int* res);
};

/**
 * You can safely ignore this.
 * It is just to avoid no definition errors.
 */
inline void ExampleSolution::build(int d, const std::vector<float>& base) {}
inline void ExampleSolution::search(const std::vector<float>& query, int* res) {}

static_assert(IsSolution<ExampleSolution>);
