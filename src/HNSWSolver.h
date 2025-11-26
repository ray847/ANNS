#pragma once

#include <cstddef> // size_t
#include <cmath> // log

#include <random> // std::normal_distribution
#include <vector> // std::vector
#include <utility> // std::pair
#include <algorithm> // std::partial_sort

#include "LinAlg.h" // solution::Mat
#include "Graph.h" // solution::Graph
#include "Global.h" // rng

namespace solution {
template<
size_t kM,
size_t kEF_CONSTRUCTION,
size_t kEF_SEARCH,
size_t kML = log(kM)
>
class Solver {
public:
  Solver(const Mat<float, false>& base): base_(base) {
    for (size_t i = 0; i < n_base(); ++i) insert_vec(i);
  }
private:
  const Mat<float, false> base_;
  std::vector<Graph> layer_graphs_;
  /* Functions */
  void insert_vec(size_t idx) {
    size_t top_layer = random_layer();
    layer_graphs_.resize(top_layer + 1);
    for (size_t layer = 0; layer <= top_layer; ++layer) {
      insert_into_graph(idx, layer);
    }
  }
  void insert_into_graph(size_t vec_idx, size_t layer) {
    auto& graph = layer_graphs_[layer];
    std::vector<std::pair<float, size_t>> dis;
    for (auto node : graph.nodes()) {
      dis.emplace_back(L2(base_vec(node), base_vec(vec_idx), dim()), node);
    }
    size_t n_candidates = std::min(kEF_CONSTRUCTION, dis.size());
    std::partial_sort(dis.begin(),
                      dis.begin() + n_candidates,
                      dis.end());
    for (size_t candidate_idx = 0;
    candidate_idx < n_candidates;
    ++candidate_idx) {
      bool insert = true;
      for (auto neighbor : graph.adj(vec_idx)) {
        if (L2(base_vec(candidate_idx), base_vec(neighbor)) 
          < dis[candidate_idx].first) {
          insert = false;
          break;
        }
      }
      if (insert) {
        graph.connect(vec_idx, candidate_idx);
      }
    }
  }
  /* Utility Functions */
  size_t random_layer() {
    std::uniform_real_distribution<float> uniform_dis(0.0f, 1.0f);
    return static_cast<size_t>(-log(1 - uniform_dis(global::rng)) * kML);
  }
  size_t dim() const {
    return base_.n();
  }
  size_t n_base() {
    return base_.m();
  }
  float* base_vec(size_t idx) {
    return &base_.at(0, idx);
  }
};
} // namespace solution
