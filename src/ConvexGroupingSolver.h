#pragma once

#include <limits> // std::numeric_limits
#include <numeric> // std::iota
#include <vector> // std::vector
#include <algorithm> // std::sort
#include <fstream> // std::ofstream
#include <unordered_set> // std::set
#include <iostream> // std::clog

#include "Global.h"
#include "LinAlg.h" // Mat, Vec
#include "ConvexGroupingGraph.h" // solution::Graph

namespace solution {
/* Solver Class Definition */
template<size_t GROUP_SIZE = 2000>
class Solver {
public:
  /* Constructor */
  Solver(size_t dim, const std::vector<float>& base):
    base_(dim, base.size() / dim, base.data()), 
    order_(base.size() / dim),
    graph_(base.size() / dim) {
    std::iota(order_.begin(), order_.end(), 0);
  }
  /* Destructor */
  ~Solver() {
    if constexpr (global::kDEBUG) {
      /* Save the order and grouping. */
      {
        std::ofstream os("tmp/base_sorted.txt");
        for (size_t i = 0; i < n_base(); ++i) {
          for (size_t j = 0; j < dim(); ++j) {
            os << base_vec(i)[j] << ' ';
          }
          os << '\n';
        }

        os.close();
      } {
        std::ofstream os("tmp/grouping.txt");
        for (auto ele : convex_groups_) {
          os << ele << ' ';
        }
        os.close();
      } {
        std::ofstream os("tmp/graph.txt");
        for (size_t from = 0; from < n_base(); ++from) {
          for (auto to : graph_.adj(from)) {
            os << from << ' ' << to << '\n';
          }
        }
      }
    }
  }
  /* Functions */
  void build() {
    /* Construct the convex groups. */
    size_t st = 0;
    do {
      convex_groups_.push_back(st);
    } while ((st = build_partial(st)) != n_base());
    convex_groups_.push_back(n_base());
  };
  std::vector<size_t> search(const Vec<float>& query) {
    std::unordered_set<size_t> ignore;
    /* Navigate the graph. */
    std::vector<std::pair<float, size_t>> points;
    for (size_t seed : seeds_) {
      points.push_back(navigate_graph(seed, query, ignore));
    }
    /* Get the first kCRITERION points. */
    std::vector<size_t> res;
    for (size_t i = 0; i < global::kCRITERION; ++i) {
      auto global_min = std::min_element(points.begin(), points.end());
      res.push_back(actual_order(global_min->second));
      ignore.insert(global_min->second);
     *global_min = navigate_graph(global_min->second, query, ignore);
    }
    return res;
  }
private:
  /* Variables */
  const Mat<float, false> base_;
  std::vector<size_t> order_;
  std::vector<size_t> convex_groups_;
  Graph graph_;
  std::vector<size_t> seeds_;
  /* Functions */
  /**
   * Find the largest outer convex group and rearrange the base vectors.
   *
   * @param st Index of the first base vector to construct the convex group.
   */
  size_t build_partial(size_t st) {
    size_t ed = st; //< Index to the first vector NOT belonging to the convex
    //< group.
    /* Select points for the convex group. */
    std::vector<Vec<float>> random_dirs;
    for (size_t g = 0; g < GROUP_SIZE; ++g) {
      /* Select a random direction. */
      Vec<float> dir(dim());
      random_uniform(dir.data(), dim());
      /* Find the base vector with the max dot product. */
      float max_prod = std::numeric_limits<float>::min();
      size_t max_idx = 0;
      for (size_t i = st; i < n_base(); ++i) {
        float prod = dot(dir.data(), base_vec(i), dim());
        if (prod > max_prod) {
          max_prod = prod;
          max_idx = i;
        }
      }
      /* Put the vectors into the group. */
      if (max_idx >= ed) {
        swap_base_vec(max_idx, ed++);
        random_dirs.push_back(dir);
      }
    }
    /* Construct the graph on the direction. */
    const size_t kFANOUT = 15;
    Mat<std::pair<float, size_t>> distance(ed - st, ed - st);
    for (size_t from = st; from < ed; ++from) {
      for (size_t to = st; to < from; ++to) {
        float tmp = L2(base_vec(from), base_vec(to), dim());
        distance.at(from - st, to - st) = {tmp, to};
        distance.at(to - st, from - st) = {tmp, from};
      }
    }
    for (size_t from = st; from < ed; ++from) {
      std::partial_sort(&distance.at(from - st, 0),
                        std::min(&distance.at(from - st, 0) + kFANOUT + 1,
                                 &distance.at(from - st, ed - st)),
                        &distance.at(from - st, ed - st));
      for (size_t i = 1; i < std::min(kFANOUT + 1, ed - st); ++i) {
        size_t to = distance.at(from - st, i).second;
        graph_.connect(from, to);
      }
    }
    /* Seed selection. */
    seeds_.push_back(st);
    return ed;
  }
  std::pair<float, size_t> navigate_graph(
    size_t st,
    const Vec<float>& query,
    const std::unordered_set<size_t>& ignore
  ) {
    if (ignore.count(st)) {
      for (size_t adj : graph_.adj(st)) {
        if (!ignore.count(adj)) {
          st = adj;
          break;
        }
      }
    }
    float dis_st = L2(query.data(), base_vec(st), dim());
    bool found = true;
    while (found) {
      found = false;
      for (size_t adj : graph_.adj(st)) {
        if (ignore.count(adj)) continue;
        float dis = L2(query.data(), base_vec(adj), dim());
        if (dis < dis_st) {
          st = adj;
          dis_st = dis;
          found = true;
          break;
        }
      }
    }
    return {dis_st, st};
  }
  /* Utility Functions */
  size_t dim() const {return base_.n();}
  size_t n_base() const {return base_.m();}
  size_t actual_order(size_t i) {return order_[i];}
  const float* base_vec(size_t i) const {
    return base_.data() + order_[i] * dim();
  }
  void swap_base_vec(size_t i, size_t j) {
    std::swap(order_[i], order_[j]);
  }
};
} // namespace solution
