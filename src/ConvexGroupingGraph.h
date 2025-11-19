#pragma once

#include <vector> // std::vector

namespace solution {
template<typename WT>
class Graph {
public:
  using EdgeT = std::pair<size_t, WT>;
  Graph(size_t n = 0) : adj_(n) {}
  void connect(size_t i, size_t j, WT w) {
    adj_[i].emplace_back(j, w);
  }
  const std::vector<EdgeT>& Adj(size_t i) {
    return adj_[i];
  }
private:
  std::vector<std::vector<EdgeT>> adj_;
};
} // namespace solution
