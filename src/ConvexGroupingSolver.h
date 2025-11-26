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
#include "Graph.h" // solution::Graph

namespace solution {
/* Solver Class Definition */
template<size_t GROUP_SIZE = 10000>
class Solver {
public:
  /* Constructor */
  Solver(size_t dim, const std::vector<float>& base):
    base_(dim, base.size() / dim, base.data()), 
    order_(base.size() / dim),
    graph_() {
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
    /* Navigate the graph. */
    std::vector<std::pair<float, size_t>> points;
    for (size_t seed : seeds_) {
      std::pair<float, size_t> point = {
        L2(query.data(), base_vec(seed), dim()),
        seed
      };
      bool found = true;
      while (found) {
        found = false;
        for (size_t adj : graph_.adj(point.second)) {
          float dis = L2(query.data(), base_vec(adj), dim());
          if (point.first > dis) {
            point = {dis, adj};
            found = true;
            break;
          }
        }
      }
      points.push_back(point);
    }
    /* Get the first kCRITERION points. */
    std::vector<size_t> res;
    std::unordered_set<size_t> res_set;
    for (size_t i = 0; i < global::kCRITERION; ++i) {
      auto global_min = std::min_element(points.begin(), points.end());
      res.push_back(actual_order(global_min->second));
      res_set.insert(global_min->second);
      std::vector<std::pair<float, size_t>> adj_points;
      for (size_t adj : graph_.adj(global_min->second)) {
        if (res_set.count(adj)) continue;
        adj_points.emplace_back(L2(query.data(), base_vec(adj), dim()), adj);
      }
      if (adj_points.empty()) {
        global_min->first = std::numeric_limits<float>::max();
        continue;
      }
      auto group_min = std::min_element(adj_points.begin(), adj_points.end());
      *global_min = *group_min;
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
        std::clog << ed << std::endl;
        swap_base_vec(max_idx, ed++);
        random_dirs.push_back(dir);
      }
    }
    /* Construct the graph on the direction. */
    const size_t kFANOUT = 30;
    for (size_t from = st; from < ed; ++from) {
      std::vector<std::pair<float, size_t>> products;
      for (size_t to = st; to < ed; ++to) {
        if (from == to) continue;
        float product = dot(random_dirs[from - st].data(),
                            random_dirs[to - st].data(),
                            dim());
        products.emplace_back(product, to);
      }
      std::partial_sort(products.begin(),
                        std::min(products.begin() + kFANOUT, products.end()),
                        products.end(),
                        std::greater<std::pair<float, size_t>>{});
      for (size_t i = 0; i < std::min(kFANOUT, products.size()); ++i) {
        size_t to = products[i].second;
        graph_.connect(from, to);
      }
    }
    /* Seed selection. */
    seeds_.push_back(st);
    return ed;
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
