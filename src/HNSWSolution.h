#pragma once

#include "SolutionConcept.h"
#include <vector>
#include <random>
#include <algorithm>
#include <queue>
#include <cmath>
#include <unordered_set>

// Helper structure for priority queue
using F_I = std::pair<float, int>;

class Solution {
public:
  void build(int d, const std::vector<float>& base);
  void search(const std::vector<float>& query, int* res);

private:
  struct Node {
    std::vector<float> vec;
    int level;
    std::vector<std::vector<int>> neighbors;
  };

  int d_ = 0;
  int m_ = 48; // Max connections per layer
  int ef_construction_ = 1000; // Size of dynamic candidate list
  int ef_search_ = 500; // Size of dynamic candidate list for search
  int entry_point_ = -1;
  double level_mult_ = 1.0 / log(1.0 * m_);
  std::vector<Node> graph_;
  std::mt19937 rng_;

  inline float distance(const std::vector<float>& v1, const std::vector<float>& v2);
  inline int getRandomLevel();
  inline std::priority_queue<F_I> search_layer(const std::vector<float>& query, int entry_point, int level, int ef);
};

static_assert(IsSolution<Solution>);

inline void Solution::build(int d, const std::vector<float>& base) {
  d_ = d;
  int n = base.size() / d;
  graph_.resize(n);
  rng_.seed(42);

  for (int i = 0; i < n; ++i) {
    graph_[i].vec.assign(base.begin() + i * d, base.begin() + (i + 1) * d);
    graph_[i].level = getRandomLevel();
    graph_[i].neighbors.resize(graph_[i].level + 1);
  }

  if (n == 0) return;

  entry_point_ = 0;
  for (int i = 1; i < n; ++i) {
    if (graph_[i].level > graph_[entry_point_].level) {
      entry_point_ = i;
    }
  }

  for (int i = 0; i < n; ++i) {
    if (i == entry_point_) continue;

    int current_ep = entry_point_;
    int max_level = graph_[entry_point_].level;
    int target_level = graph_[i].level;

    for (int level = max_level; level > target_level; --level) {
      std::priority_queue<F_I> candidates;
      candidates.push({-distance(graph_[i].vec, graph_[current_ep].vec), current_ep});

      std::unordered_set<int> visited;
      visited.insert(current_ep);

      while (!candidates.empty()) {
        auto top = candidates.top();
        candidates.pop();
        int current_node_idx = top.second;

        bool found_better = false;
        for (int neighbor_idx : graph_[current_node_idx].neighbors[level]) {
          if (visited.find(neighbor_idx) == visited.end()) {
            visited.insert(neighbor_idx);
            float dist = distance(graph_[i].vec, graph_[neighbor_idx].vec);
            if (-dist > top.first) {
              candidates.push({-dist, neighbor_idx});
              found_better = true;
            }
          }
        }
        if(found_better) {
          current_ep = candidates.top().second;
        } else {
          break;
        }
      }
    }


    for (int level = std::min(target_level, max_level); level >= 0; --level) {
      std::priority_queue<F_I> search_result = search_layer(graph_[i].vec, current_ep, level, ef_construction_);

      int count = 0;
      std::vector<int> neighbors;
      while(!search_result.empty() && count < m_){
        neighbors.push_back(search_result.top().second);
        search_result.pop();
        count++;
      }

      for(int neighbor_idx : neighbors) {
        graph_[i].neighbors[level].push_back(neighbor_idx);
        graph_[neighbor_idx].neighbors[level].push_back(i);
      }
    }
    if (target_level > graph_[entry_point_].level) {
      entry_point_ = i;
    }
  }
}

inline void Solution::search(const std::vector<float>& query, int* res) {
  int current_ep = entry_point_;
  if (current_ep == -1) return;

  int max_level = graph_[entry_point_].level;

  for (int level = max_level; level > 0; --level) {
    std::priority_queue<F_I> candidates;
    candidates.push({-distance(query, graph_[current_ep].vec), current_ep});

    std::unordered_set<int> visited;
    visited.insert(current_ep);

    while (!candidates.empty()) {
      auto top = candidates.top();
      candidates.pop();
      int current_node_idx = top.second;

      bool found_better = false;
      for (int neighbor_idx : graph_[current_node_idx].neighbors[level]) {
        if (visited.find(neighbor_idx) == visited.end()) {
          visited.insert(neighbor_idx);
          float dist = distance(query, graph_[neighbor_idx].vec);
          if (-dist > top.first) {
            candidates.push({-dist, neighbor_idx});
            found_better = true;
          }
        }
      }
      if(found_better) {
        current_ep = candidates.top().second;
      } else {
        break;
      }
    }
  }

  std::priority_queue<F_I> result_pq = search_layer(query, current_ep, 0, ef_search_);

  std::vector<int> result_vec;
  while(!result_pq.empty()){
    result_vec.push_back(result_pq.top().second);
    result_pq.pop();
  }
  std::reverse(result_vec.begin(), result_vec.end());

  for(size_t i=0; i < result_vec.size() && i < 10; ++i){
    res[i] = result_vec[i];
  }
}

inline std::priority_queue<F_I> Solution::search_layer(const std::vector<float>& query, int entry_point, int level, int ef) {
  std::priority_queue<F_I> candidates;
  candidates.push({-distance(query, graph_[entry_point].vec), entry_point});

  std::priority_queue<F_I> result;
  result.push({distance(query, graph_[entry_point].vec), entry_point});

  std::unordered_set<int> visited;
  visited.insert(entry_point);

  while(!candidates.empty()){
    auto top = candidates.top();
    candidates.pop();

    if (-top.first > result.top().first && result.size() >= ef) {
      break;
    }

    int current_node_idx = top.second;
    for (int neighbor_idx : graph_[current_node_idx].neighbors[level]) {
      if (visited.find(neighbor_idx) == visited.end()) {
        visited.insert(neighbor_idx);
        float dist = distance(query, graph_[neighbor_idx].vec);
        if (dist < result.top().first || result.size() < ef) {
          candidates.push({-dist, neighbor_idx});
          result.push({dist, neighbor_idx});
          if (result.size() > ef) {
            result.pop();
          }
        }
      }
    }
  }
  return result;
}

inline float Solution::distance(const std::vector<float>& v1, const std::vector<float>& v2) {
  float dist = 0.0f;
  for (int i = 0; i < d_; ++i) {
    float diff = v1[i] - v2[i];
    dist += diff * diff;
  }
  return sqrt(dist);
}

inline int Solution::getRandomLevel() {
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  return static_cast<int>(-log(dist(rng_)) * level_mult_);
}
