#pragma once

#include <vector> // std::vector

namespace solution {
class Graph {
public:
  Graph(size_t n = 0) : adj_(n) {}
  void connect(size_t i, size_t j) {
    adj_[i].push_back(j);
  }
  const auto& adj(size_t i) const {
    return adj_[i];
  }
private:
  std::vector<std::vector<size_t>> adj_;
};
} // namespace solution
