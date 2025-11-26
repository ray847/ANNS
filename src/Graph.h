#pragma once

#include <unordered_set>
#include <vector> // std::vector
#include <unordered_map> // std::unordered_map

namespace solution {
class Graph {
public:
  void connect(size_t i, size_t j) {
    nodes_.insert(i);
    nodes_.insert(j);
    adj_[i].push_back(j);
    adj_[j].push_back(i);
  }
  const auto& adj(size_t i) const {
    return adj_.at(i);
  }
  const auto& nodes() const {
    return nodes_;
  }
private:
  std::unordered_set<size_t> nodes_;
  std::unordered_map<size_t, std::vector<size_t>> adj_;
};
} // namespace solution
