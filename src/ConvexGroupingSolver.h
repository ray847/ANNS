#pragma once

#include <numeric> // std::iota
#include <vector> // std::vector
#include <algorithm> // std::sort
#include <fstream> // std::ofstream

#include "LinAlg.h" // Mat, Vec
#include "ConvexGroupingGraph.h" // solution::Graph

namespace solution {
/* Solver Class Definition */
template<size_t GROUP_SIZE = 100>
class Solver {
public:
  /* Constructor */
  Solver(size_t dim, const std::vector<float>& base):
    base_(dim, base.size() / dim, base.data()), 
    order_(base.size() / dim) {
    std::iota(order_.begin(), order_.end(), 0);
  }
  /* Destructor */
  ~Solver() {
    if constexpr (global::kDEBUG) {
      /* Save the order and grouping. */
      {
        std::ofstream os("base_sorted.txt");
        for (size_t i = 0; i < n_base(); ++i) {
          for (size_t j = 0; j < dim(); ++j) {
            os << base_vec(i)[j] << ' ';
          }
          os << '\n';
        }
        os.close();
      } {
        std::ofstream os("split_indicies.txt");
        for (auto ele : convex_groups_) {
          os << ele << ' ';
        }
        os.close();
      }
    }
  }
  /* Functions */
  void build() {
    /* Construct the convex groups. */
    size_t st = 0;
    do {
      convex_groups_.push_back(st);
    } while ((st = peal(st)) != n_base());
    convex_groups_.push_back(n_base());
  };
  void search();
private:
  /* Variables */
  const Mat<float, false> base_;
  std::vector<size_t> order_;
  std::vector<size_t> convex_groups_;
  //Graph<Vec<float>> graph_;
  /* Functions */
  /**
   * Find the largest outer convex group and rearrange the base vectors.
   *
   * @param st Index of the first base vector to construct the convex group.
   */
  size_t peal(size_t st) {
    size_t ed = st; //< Index to the first vector NOT belonging to the convex
                    //< group.
    for (size_t g = 0; g < GROUP_SIZE; ++g) {
      /* Select a random direction. */
      Vec<float> dir(dim());
      random_uniform(dir.data(), dim());
      /* Find the 2 base vectors with the max and min product. */
      std::vector<float> products_data(n_base() - st);
      float* products = products_data.data() - st;
      for (size_t i = st; i < n_base(); ++i)
        products[i] = dot(dir.data(), base_vec(i), dim());
      auto [min_iter, max_iter] = std::minmax_element(products + st, 
                                                    products + n_base());
      size_t min_idx = min_iter - products;
      size_t max_idx = max_iter - products;
      /* Put the vectors into the group. */
      if (min_idx == max_idx) {
        if (min_idx >= ed) swap_base_vec(ed++, min_idx);
      } else {
        if (min_idx >= ed) swap_base_vec(ed++, min_idx);
        if (max_idx >= ed) swap_base_vec(ed++, max_idx);
      }
    }
    return ed;
  }
  void construct_graph(size_t st, size_t ed) {

  }
  /* Utility Functions */
  size_t dim() const {return base_.n();}
  size_t n_base() const {return base_.m();}
  const float* base_vec(size_t i) const {
    return base_.data() + order_[i] * dim();
  }
  void swap_base_vec(size_t i, size_t j) {
    std::swap(order_[i], order_[j]);
  }
};
} // namespace solution
