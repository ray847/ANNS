#pragma once

#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>
#include <queue>
#include <functional>

#include "TunableHNSW/Distance.h"

namespace Standalone {

// Enum to select the hard-coded parameter set
enum class Dataset { SIFT, GLOVE };

template <Dataset D>
class HNSW {
 public:
  HNSW();

  void Build(const std::vector<float>& base_data);
  void Search(const std::vector<float>& query, int k, int* result_indices);

 private:
  struct Node {
    int level;
    size_t offset;
  };

  struct VisitedList {
    std::vector<bool> tags;
    void resize(size_t n) { tags.resize(n, false); }
    void reset() { std::fill(tags.begin(), tags.end(), false); }
    inline bool visit(int id) {
      if (tags[id]) return true;
      tags[id] = true;
      return false;
    }
  };

  // --- Hard-coded Parameters ---
  static constexpr int dim_ = (D == Dataset::SIFT) ? 128 : 100;
  static constexpr int M_ = (D == Dataset::SIFT) ? 32 : 48;
  static constexpr int M0_ = (D == Dataset::SIFT) ? 64 : 96;
  static constexpr int ef_construction_ = (D == Dataset::SIFT) ? 500 : 800;
  static constexpr int patience_ = (D == Dataset::SIFT) ? 100 : 200;
  static constexpr int max_ef_search_ = (D == Dataset::SIFT) ? 500 : 2000;
  
  const double level_mult_;
  int max_level_ = -1;
  int entry_point_ = -1;

  std::vector<float> data_storage_;
  const float* data_ptr_ = nullptr;
  size_t num_points_ = 0;
  std::vector<Node> nodes_;
  std::vector<int> flat_graph_;
  std::vector<int> link_counts_;
  
  using Dist = TunableHNSW::Distance<dim_, true>;

  // --- Private Methods ---
  int GetRandomLevel_(std::mt19937& rng);
  void Insert_(int node_id, 
               std::vector<std::vector<std::vector<int>>>& temp_graph,
               std::vector<std::mutex>& locks);
  std::priority_queue<std::pair<float, int>> SearchLayer_(
      const float* query, int entry_point, int ef, int level,
      const std::vector<std::vector<std::vector<int>>>& temp_graph,
      std::vector<std::mutex>& locks, VisitedList& visited);
  void SelectNeighbors_(
      std::priority_queue<std::pair<float, int>>& candidates,
      std::vector<int>& target_list, int M);
  void FlattenGraph_(
      const std::vector<std::vector<std::vector<int>>>& temp_graph);
};

// --- Implementation ---

template <Dataset D>
HNSW<D>::HNSW() : level_mult_(1.0 / std::log(1.0 * M_)) {}

template <Dataset D>
void HNSW<D>::Build(const std::vector<float>& base_data) {
  data_storage_ = base_data;
  data_ptr_ = data_storage_.data();
  num_points_ = base_data.size() / dim_;
  nodes_.resize(num_points_);
  
  std::mt19937 rng(100);
  int max_level = 0;
  for (size_t i = 0; i < num_points_; ++i) {
    nodes_[i].level = GetRandomLevel_(rng);
    if (nodes_[i].level > max_level) max_level = nodes_[i].level;
  }
  max_level_ = max_level;
  
  if (num_points_ > 0) {
    entry_point_ = 0;
    for(size_t i = 1; i < num_points_; ++i) {
      if(nodes_[i].level > nodes_[entry_point_].level) entry_point_ = i;
    }
  }

  std::vector<std::vector<std::vector<int>>> temp_graph(num_points_);
  for(size_t i = 0; i < num_points_; ++i) {
      temp_graph[i].resize(nodes_[i].level + 1);
  }
  std::vector<std::mutex> locks(num_points_);
  std::atomic<size_t> atomic_idx{0};

  unsigned int num_threads = std::thread::hardware_concurrency();
  auto worker_func = [&]() {
    while (true) {
      size_t current_node_id = atomic_idx.fetch_add(1);
      if (current_node_id >= num_points_) break;
      Insert_(current_node_id, temp_graph, locks);
    }
  };

  std::vector<std::thread> threads;
  for (unsigned int i = 0; i < num_threads; ++i) {
    threads.emplace_back(worker_func);
  }
  for (auto& t : threads) {
    t.join();
  }

  FlattenGraph_(temp_graph);
}

template <Dataset D>
int HNSW<D>::GetRandomLevel_(std::mt19937& rng) {
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  return static_cast<int>(-std::log(dist(rng)) * level_mult_);
}

template <Dataset D>
void HNSW<D>::Insert_(int node_id,
                           std::vector<std::vector<std::vector<int>>>& temp_graph,
                           std::vector<std::mutex>& locks) {
  
  thread_local VisitedList visited;
  if(visited.tags.size() != num_points_) visited.resize(num_points_);
  
  const float* query_vector = data_ptr_ + node_id * dim_;
  int current_ep = entry_point_;
  int node_level = nodes_[node_id].level;
  
  if (current_ep == -1) return;

  for (int level = max_level_; level > node_level; --level) {
    bool changed = true;
    while(changed) {
        changed = false;
        float min_dist = Dist::L2Sq(query_vector, data_ptr_ + current_ep * dim_);
        std::lock_guard<std::mutex> lock(locks[current_ep]);
        if (level >= temp_graph[current_ep].size()) continue;
        const auto& neighbors = temp_graph[current_ep][level];
        for (int neighbor_id : neighbors) {
            float dist = Dist::L2Sq(query_vector, data_ptr_ + neighbor_id * dim_);
            if (dist < min_dist) {
                min_dist = dist;
                current_ep = neighbor_id;
                changed = true;
            }
        }
    }
  }

  for (int level = std::min(node_level, max_level_); level >= 0; --level) {
    visited.reset();
    auto top_candidates = SearchLayer_(query_vector, current_ep, ef_construction_, level, temp_graph, locks, visited);
    
    int max_M = (level == 0) ? M0_ : M_;
    std::vector<int> neighbors;
    neighbors.reserve(max_M);
    SelectNeighbors_(top_candidates, neighbors, max_M);
    
    {
        std::lock_guard<std::mutex> lock(locks[node_id]);
        temp_graph[node_id][level] = neighbors;
    }

    for (int neighbor_id : neighbors) {
      std::lock_guard<std::mutex> lock(locks[neighbor_id]);
      if (level >= temp_graph[neighbor_id].size()) continue;

      auto& neighbor_links = temp_graph[neighbor_id][level];
      int neighbor_M = (level == 0) ? M0_ : M_;

      if (neighbor_links.size() < neighbor_M) {
        neighbor_links.push_back(node_id);
      } else {
        float new_node_dist = Dist::L2Sq(data_ptr_ + neighbor_id * dim_, query_vector);
        std::priority_queue<std::pair<float, int>> temp_pq;
        for(int link : neighbor_links) {
            temp_pq.push({Dist::L2Sq(data_ptr_ + neighbor_id * dim_, data_ptr_ + link * dim_), link});
        }
        if (new_node_dist < temp_pq.top().first) {
            temp_pq.pop();
            temp_pq.push({new_node_dist, node_id});
            neighbor_links.clear();
            while(!temp_pq.empty()) {
                neighbor_links.push_back(temp_pq.top().second);
                temp_pq.pop();
            }
        }
      }
    }
    if(!top_candidates.empty()) {
        current_ep = top_candidates.top().second;
    }
  }
}

template <Dataset D>
std::priority_queue<std::pair<float, int>> HNSW<D>::SearchLayer_(
    const float* query, int entry_point, int ef, int level,
    const std::vector<std::vector<std::vector<int>>>& temp_graph,
    std::vector<std::mutex>& locks, VisitedList& visited) {
  
  using QueueItem = std::pair<float, int>;
  std::priority_queue<QueueItem> top_results;
  std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> candidates;

  float initial_dist = Dist::L2Sq(query, data_ptr_ + entry_point * dim_);
  candidates.push({initial_dist, entry_point});
  top_results.push({initial_dist, entry_point});
  visited.visit(entry_point);

  while (!candidates.empty()) {
    auto [dist, id] = candidates.top();
    candidates.pop();

    if (dist > top_results.top().first && top_results.size() >= (size_t)ef) break;
    
    std::lock_guard<std::mutex> lock(locks[id]);
    if (level >= temp_graph[id].size()) continue;

    const auto& neighbors = temp_graph[id][level];
    for (int neighbor_id : neighbors) {
      if (!visited.visit(neighbor_id)) {
        float neighbor_dist = Dist::L2Sq(query, data_ptr_ + neighbor_id * dim_);
        if (top_results.size() < (size_t)ef || neighbor_dist < top_results.top().first) {
          candidates.push({neighbor_dist, neighbor_id});
          top_results.push({neighbor_dist, neighbor_id});
          if (top_results.size() > (size_t)ef) top_results.pop();
        }
      }
    }
  }
  return top_results;
}

template <Dataset D>
void HNSW<D>::SelectNeighbors_(
    std::priority_queue<std::pair<float, int>>& candidates,
    std::vector<int>& target_list, int M) {
  
  if (candidates.empty()) return;

  std::vector<std::pair<float, int>> candidate_vec;
  candidate_vec.reserve(candidates.size());
  while(!candidates.empty()) {
    candidate_vec.push_back(candidates.top());
    candidates.pop();
  }
  std::reverse(candidate_vec.begin(), candidate_vec.end());

  for (const auto& cand : candidate_vec) {
    if (target_list.size() >= (size_t)M) break;
    bool is_good = true;
    for (int selected_neighbor : target_list) {
      if (Dist::L2Sq(data_ptr_ + cand.second * dim_, data_ptr_ + selected_neighbor * dim_) < cand.first) {
        is_good = false;
        break;
      }
    }
    if (is_good) {
      target_list.push_back(cand.second);
    }
  }
}

template <Dataset D>
void HNSW<D>::FlattenGraph_(const std::vector<std::vector<std::vector<int>>>& temp_graph) {
  link_counts_.resize(num_points_ * (max_level_ + 1), 0);
  size_t total_links = 0;
  for (size_t i = 0; i < num_points_; ++i) {
    if(nodes_[i].level > max_level_) continue;
    for (int level = 0; level <= nodes_[i].level; ++level) {
      if (level < temp_graph[i].size()) {
        total_links += temp_graph[i][level].size();
      }
    }
  }
  
  flat_graph_.resize(total_links);
  size_t current_offset = 0;

  for (size_t i = 0; i < num_points_; ++i) {
    nodes_[i].offset = current_offset;
    if(nodes_[i].level > max_level_) continue;
    for (int level = 0; level <= nodes_[i].level; ++level) {
      if (level < temp_graph[i].size()) {
        const auto& neighbors = temp_graph[i][level];
        link_counts_[i * (max_level_ + 1) + level] = neighbors.size();
        for (int neighbor : neighbors) {
          flat_graph_[current_offset++] = neighbor;
        }
      }
    }
  }
}

template <Dataset D>
void HNSW<D>::Search(const std::vector<float>& query, int k, int* result_indices) {
  thread_local VisitedList visited;
  if (visited.tags.size() != num_points_) visited.resize(num_points_);
  visited.reset();

  const float* query_data = query.data();
  int current_ep = entry_point_;

  if (current_ep == -1) {
    for(int i = 0; i < k; ++i) result_indices[i] = -1;
    return;
  }

  float current_dist = Dist::L2Sq(query_data, data_ptr_ + current_ep * dim_);

  for (int level = max_level_; level > 0; --level) {
    bool changed = true;
    while (changed) {
      changed = false;
      size_t node_offset = nodes_[current_ep].offset;
      int link_offset_level = 0;
      for (int i = 0; i < level; ++i) {
          link_offset_level += link_counts_[current_ep * (max_level_ + 1) + i];
      }
      const int* neighbors = flat_graph_.data() + node_offset + link_offset_level;
      int count = link_counts_[current_ep * (max_level_ + 1) + level];
      for (int i = 0; i < count; ++i) {
        int neighbor_id = neighbors[i];
        float dist = Dist::L2Sq(query_data, data_ptr_ + neighbor_id * dim_);
        if (dist < current_dist) {
          current_dist = dist;
          current_ep = neighbor_id;
          changed = true;
        }
      }
    }
  }

  using QueueItem = std::pair<float, int>;
  std::priority_queue<QueueItem> top_candidates;
  std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> candidates;
  top_candidates.push({current_dist, current_ep});
  candidates.push({current_dist, current_ep});
  visited.visit(current_ep);

  int patience = patience_;
  float best_dist_so_far = current_dist;

  while(!candidates.empty()) {
    auto [dist, id] = candidates.top();
    candidates.pop();

    if (dist > best_dist_so_far) {
        if (--patience == 0) break;
    }
    
    size_t node_offset = nodes_[id].offset;
    const int* neighbors = flat_graph_.data() + node_offset;
    int count = link_counts_[id * (max_level_ + 1) + 0];

    for (int i = 0; i < count; ++i) {
        int neighbor_id = neighbors[i];
        if (i + 4 < count) { 
            _mm_prefetch((const char*)(data_ptr_ + neighbors[i+4] * dim_), _MM_HINT_T0);
        }
        if(!visited.visit(neighbor_id)) {
            float neighbor_dist = Dist::L2Sq(query_data, data_ptr_ + neighbor_id * dim_);
            if (top_candidates.size() < (size_t)max_ef_search_ || neighbor_dist < best_dist_so_far) {
                candidates.push({neighbor_dist, neighbor_id});
                top_candidates.push({neighbor_dist, neighbor_id});
                if (top_candidates.size() > (size_t)max_ef_search_) {
                    top_candidates.pop();
                }
                best_dist_so_far = top_candidates.top().first;
            }
        }
    }
  }

  size_t result_count = 0;
  std::vector<QueueItem> sorted_results;
  sorted_results.reserve(top_candidates.size());
  while(!top_candidates.empty()) {
    sorted_results.push_back(top_candidates.top());
    top_candidates.pop();
  }
  std::reverse(sorted_results.begin(), sorted_results.end());

  for (const auto& p : sorted_results) {
    if (result_count >= (size_t)k) break;
    result_indices[result_count++] = p.second;
  }
  while(result_count < (size_t)k) {
    result_indices[result_count++] = -1;
  }
}

} // namespace Standalone