// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <random>
#include <thread>
#include <vector>

#include "../Global.h"
#include "Config.h"
#include "Distance.h"
#include "Metrics.h"
#include "OPQ.h"

namespace TunableHNSW {

template <typename Config>
class HNSW {
 public:
  using Dist = Distance<Config::kDimVal, Config::kUseSIMDVal>;
  using PQ = std::conditional_t<
      Config::kQuantizationVal == QuantizationStrategy::kPQ || Config::kQuantizationVal == QuantizationStrategy::kOPQ,
      OptimizedProductQuantizer<Config>,
      std::nullptr_t>;

  HNSW()
      : level_mult_(1.0 / std::log(1.0 * Config::kMVal)),
        metrics_() {
    if constexpr (Config::kQuantizationVal == QuantizationStrategy::kPQ || Config::kQuantizationVal == QuantizationStrategy::kOPQ) {
      pq_.emplace();
    }
  }

  ~HNSW() {
    if constexpr (global::kDEBUG) {
      PrintMetricsReport_();
    }
  }

  void Build(const std::vector<float>& base_data) {
    data_storage_ = base_data;
    data_ptr_ = data_storage_.data();
    num_points_ = base_data.size() / Config::kDimVal;
    nodes_.resize(num_points_);

    InitializeNodes_();
    
    if constexpr (global::kDEBUG) {
        std::cout << "Building HNSW graph for " << num_points_ << " vectors..." << std::endl;
    }
    nodes_processed_count_ = 1; // Initialize to 1 for the entry point (node 0)
    BuildGraphConcurrently_();

    if constexpr (global::kDEBUG) {
        std::cout << "HNSW graph build complete." << std::endl;
    }

    if constexpr (Config::kQuantizationVal == QuantizationStrategy::kPQ || Config::kQuantizationVal == QuantizationStrategy::kOPQ) {
      if constexpr (global::kDEBUG) {
          std::cout << "Training Product Quantizer and encoding vectors..." << std::endl;
      }
      pq_->Train(data_ptr_, num_points_);
      pq_codes_.resize(num_points_ * Config::kPQSubquantizersVal);
      for (size_t i = 0; i < num_points_; ++i) {
        auto code = pq_->Encode(data_ptr_ + i * Config::kDimVal);
        std::copy(code.begin(), code.end(),
                  pq_codes_.data() + i * Config::kPQSubquantizersVal);
      }
      if constexpr (global::kDEBUG) {
          std::cout << "Product Quantizer training and encoding complete." << std::endl;
      }
    }
  }

  void Search(const std::vector<float>& query, int* result_indices) {
    if constexpr (Config::kQuantizationVal == QuantizationStrategy::kPQ || Config::kQuantizationVal == QuantizationStrategy::kOPQ) {
      SearchPQ_(query, result_indices);
    } else {
      SearchFP_(query, result_indices);
    }
  }

  const HNSWMetrics<Config::kMaxLevelVal>& metrics() const { return metrics_; }

 private:
  struct Node {
    int level;
    std::vector<int> flat_links;
    std::vector<int> link_counts;
    std::unique_ptr<std::mutex> lock;
  };

  struct VisitedList {
    std::vector<unsigned short> tags;
    unsigned short current_tag = 0;
    void resize(size_t n) { tags.resize(n, 0); }
    void advance() {
      if (++current_tag == 0) {
        std::fill(tags.begin(), tags.end(), 0);
        current_tag = 1;
      }
    }
    inline bool visit(int id) {
      if (tags[id] == current_tag) return true;
      tags[id] = current_tag;
      return false;
    }
  };

  float DistSqFP_(int id_a, int id_b) {
    metrics_.increment_distance_calculations();
    return Dist::L2Sq(data_ptr_ + id_a * Config::kDimVal,
                      data_ptr_ + id_b * Config::kDimVal);
  }

  float QueryDistSqFP_(const float* query, int node_id) {
    metrics_.increment_distance_calculations();
    return Dist::L2Sq(query, data_ptr_ + node_id * Config::kDimVal);
  }

  void InitializeNodes_();
  void BuildGraphConcurrently_();
  int GetRandomLevel_(std::mt19937& rng);
  inline int GetLinkOffset_(int level) const;
      void Insert_(size_t current_node_id, VisitedList& visited);
  std::priority_queue<std::pair<float, int>> SearchLayerFP_(
      const float* query_data, int entry_point, int ef, int level,
      VisitedList& visited);
  void GetNeighborsHeuristic_(int source_node_id,
                              std::vector<std::pair<float, int>>& candidates,
                              int level, int* output_buffer,
                              int& output_count);
  void AddConnection_(int source_node_id, int dest_node_id, int level);
  void SearchFP_(const std::vector<float>& query, int* result_indices);
  void SearchPQ_(const std::vector<float>& query, int* result_indices);
  void PrintMetricsReport_() const;

  std::vector<float> data_storage_;
  const float* data_ptr_ = nullptr;
  size_t num_points_ = 0;

  std::optional<PQ> pq_;
  std::vector<uint8_t> pq_codes_;

  std::vector<Node> nodes_;
  int entry_point_ = -1;
  int max_level_ = -1;
  double level_mult_;
  std::mutex global_lock_;

  HNSWMetrics<Config::kMaxLevelVal> metrics_;
  std::atomic<size_t> nodes_processed_count_;
};

template <typename Config>
void HNSW<Config>::InitializeNodes_() {
  std::mt19937 rng(global::kSEED);
  for (size_t i = 0; i < num_points_; ++i) {
    int level = GetRandomLevel_(rng);
    nodes_[i].level = level;
    nodes_[i].link_counts.resize(level + 1, 0);
    nodes_[i].lock = std::make_unique<std::mutex>();
    size_t total_links =
        Config::kM0Val + (level > 0 ? (size_t)level * Config::kMVal : 0);
    nodes_[i].flat_links.resize(total_links);
  }
  entry_point_ = 0;
  max_level_ = nodes_[0].level;
}

template <typename Config>
void HNSW<Config>::BuildGraphConcurrently_() {
  std::atomic<size_t> atomic_idx{1};
  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) num_threads = 4;
  auto worker_func = [&](int thread_id) {
    VisitedList visited;
    visited.resize(num_points_);
    while (true) {
      size_t current_node_id = atomic_idx.fetch_add(1, std::memory_order_relaxed);
      if (current_node_id >= num_points_) break;
      Insert_(current_node_id, visited);
    }
  };
  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (unsigned int i = 0; i < num_threads; ++i)
    threads.emplace_back(worker_func, i);
  
  // Progress tracking loop
  if constexpr (global::kDEBUG) {
      size_t last_reported_progress = 0;
      while (nodes_processed_count_.load() < num_points_) {
          size_t current_progress = (nodes_processed_count_.load() * 100) / num_points_;
          if (current_progress > last_reported_progress) {
              std::cout << "\rBuilding HNSW: " << current_progress << "%" << std::flush;
              last_reported_progress = current_progress;
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      std::cout << "\rBuilding HNSW: 100%" << std::endl;
  }

  for (auto& t : threads) t.join();
}

template <typename Config>
int HNSW<Config>::GetRandomLevel_(std::mt19937& rng) {
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  return std::min(static_cast<int>(-std::log(dist(rng)) * level_mult_),
                  Config::kMaxLevelVal);
}

template <typename Config>
inline int HNSW<Config>::GetLinkOffset_(int level) const {
  return (level == 0) ? 0 : (Config::kM0Val + (level - 1) * Config::kMVal);
}

template <typename Config>
void HNSW<Config>::Insert_(size_t current_node_id, VisitedList& visited) {
  const float* current_vector = data_ptr_ + current_node_id * Config::kDimVal;
  int current_level = nodes_[current_node_id].level;
  int entry_point = entry_point_;
  int current_max_level = max_level_;

  for (int level = current_max_level; level > current_level; --level) {
    bool changed = true;
    while (changed) {
      changed = false;
      float entry_point_dist = QueryDistSqFP_(current_vector, entry_point);
      const Node& entry_point_node = nodes_[entry_point];
      if (level >= entry_point_node.link_counts.size()) break;
      int offset = GetLinkOffset_(level);
      int count = entry_point_node.link_counts[level];
      const int* links = entry_point_node.flat_links.data() + offset;
      for (int j = 0; j < count; ++j) {
        int neighbor_id = links[j];
        float neighbor_dist = QueryDistSqFP_(current_vector, neighbor_id);
        if (neighbor_dist < entry_point_dist) {
          entry_point = neighbor_id;
          entry_point_dist = neighbor_dist;
          changed = true;
        }
      }
    }
  }

  for (int level = std::min(current_level, current_max_level); level >= 0;
       --level) {
    auto top_candidates_pq = SearchLayerFP_(
        current_vector, entry_point, Config::kEfConstructionVal, level, visited);
    std::vector<std::pair<float, int>> top_candidates;
    top_candidates.reserve(Config::kEfConstructionVal + 1);
    while (!top_candidates_pq.empty()) {
      top_candidates.push_back(top_candidates_pq.top());
      top_candidates_pq.pop();
    }
    int offset = GetLinkOffset_(level);
    int* link_dest = nodes_[current_node_id].flat_links.data() + offset;
    int count = 0;
    GetNeighborsHeuristic_(current_node_id, top_candidates, level, link_dest,
                           count);
    nodes_[current_node_id].link_counts[level] = count;
    for (int j = 0; j < count; ++j)
      AddConnection_(link_dest[j], current_node_id, level);
    if (!top_candidates.empty()) entry_point = top_candidates[0].second;
  }

  if (current_level > max_level_) {
    std::lock_guard<std::mutex> lock(global_lock_);
    if (current_level > max_level_) {
      max_level_ = current_level;
      entry_point_ = current_node_id;
    }
  }
  if constexpr (global::kDEBUG) {
      nodes_processed_count_.fetch_add(1, std::memory_order_relaxed);
  }
}

template <typename Config>
std::priority_queue<std::pair<float, int>> HNSW<Config>::SearchLayerFP_(
    const float* query_data, int entry_point, int ef, int level,
    VisitedList& visited) {
  using QueueItem = std::pair<float, int>;
  std::priority_queue<QueueItem> top_candidates;
  std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>>
      candidates;
  visited.advance();
  float initial_dist = QueryDistSqFP_(query_data, entry_point);
  top_candidates.push({initial_dist, entry_point});
  candidates.push({initial_dist, entry_point});
  visited.visit(entry_point);
  while (!candidates.empty()) {
    auto [current_dist, current_id] = candidates.top();
    if (current_dist > top_candidates.top().first &&
        top_candidates.size() >= ef)
      break;
    candidates.pop();
    const Node& node = nodes_[current_id];
    int size = node.link_counts[level];
    int offset = GetLinkOffset_(level);
    const int* links = node.flat_links.data() + offset;
    for (int i = 0; i < size; ++i) {
      int neighbor_id = links[i];
      if (i + 1 < size)
        _mm_prefetch(
            (const char*)(data_ptr_ + links[i + 1] * Config::kDimVal),
            _MM_HINT_T0);
      if (!visited.visit(neighbor_id)) {
        float dist = QueryDistSqFP_(query_data, neighbor_id);
        if (top_candidates.size() < ef || dist < top_candidates.top().first) {
          candidates.push({dist, neighbor_id});
          top_candidates.push({dist, neighbor_id});
          if (top_candidates.size() > ef) top_candidates.pop();
        }
      }
    }
  }
  return top_candidates;
}

template <typename Config>
void HNSW<Config>::GetNeighborsHeuristic_(
    int source_node_id, std::vector<std::pair<float, int>>& candidates,
    int level, int* output_buffer, int& output_count) {
  size_t max_m = (level == 0) ? Config::kM0Val : Config::kMVal;
  output_count = 0;
  if (candidates.empty()) return;
  std::sort(candidates.begin(), candidates.end());
  for (const auto& cand : candidates) {
    if (output_count >= max_m) break;
    bool is_good = true;
    for (int j = 0; j < output_count; ++j) {
      if (DistSqFP_(cand.second, output_buffer[j]) < cand.first) {
        is_good = false;
        break;
      }
    }
    if (is_good) output_buffer[output_count++] = cand.second;
  }
}

template <typename Config>
void HNSW<Config>::AddConnection_(int source_node_id, int dest_node_id,
                                  int level) {
  Node& node = nodes_[source_node_id];
  std::lock_guard<std::mutex> lock(*node.lock);
  int count = node.link_counts[level];
  int offset = GetLinkOffset_(level);
  int* links_ptr = node.flat_links.data() + offset;
  for (int i = 0; i < count; ++i)
    if (links_ptr[i] == dest_node_id) return;
  size_t max_m = (level == 0) ? Config::kM0Val : Config::kMVal;
  if (count < max_m) {
    links_ptr[count] = dest_node_id;
    node.link_counts[level]++;
  } else {
    std::vector<std::pair<float, int>> candidates;
    candidates.reserve(max_m + 1);
    for (int i = 0; i < count; ++i)
      candidates.push_back({DistSqFP_(source_node_id, links_ptr[i]), links_ptr[i]});
    candidates.push_back({DistSqFP_(source_node_id, dest_node_id), dest_node_id});
    int new_count = 0;
    GetNeighborsHeuristic_(source_node_id, candidates, level, links_ptr,
                           new_count);
    node.link_counts[level] = new_count;
  }
}

template <typename Config>
void HNSW<Config>::SearchFP_(const std::vector<float>& query,
                             int* result_indices) {
  static thread_local VisitedList visited;
  if (visited.tags.size() != num_points_) visited.resize(num_points_);
  const float* query_data = query.data();
  int entry_point = entry_point_;
  float current_dist = QueryDistSqFP_(query_data, entry_point);

  for (int level = max_level_; level > 0; --level) {
    auto t_start_layer = std::chrono::high_resolution_clock::now();
    bool changed = true;
    while (changed) {
      changed = false;
      const Node& node = nodes_[entry_point];
      if (level >= node.link_counts.size()) break;
      int count = node.link_counts[level];
      int offset = GetLinkOffset_(level);
      const int* links = node.flat_links.data() + offset;
      for (int i = 0; i < count; ++i) {
        int neighbor = links[i];
        if (i + 1 < count)
          _mm_prefetch(
              (const char*)(data_ptr_ + links[i + 1] * Config::kDimVal),
              _MM_HINT_T0);
        float d = QueryDistSqFP_(query_data, neighbor);
        if (d < current_dist) {
          current_dist = d;
          entry_point = neighbor;
          changed = true;
        }
      }
    }
    auto t_end_layer = std::chrono::high_resolution_clock::now();
    metrics_.add_navigation_time(
        level, std::chrono::duration_cast<std::chrono::nanoseconds>(
                   t_end_layer - t_start_layer)
                   .count());
  }

  auto t_start_l0 = std::chrono::high_resolution_clock::now();
  using QueueItem = std::pair<float, int>;
  std::priority_queue<QueueItem> top_candidates;
  visited.advance();

  if constexpr (Config::kSearchStrategyVal == SearchStrategy::kStandard) {
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>>
        candidates;
    candidates.push({current_dist, entry_point});
    top_candidates.push({current_dist, entry_point});
    visited.visit(entry_point);
    while (!candidates.empty()) {
      auto [c_dist, c_id] = candidates.top();
      candidates.pop();
      if (c_dist > top_candidates.top().first &&
          top_candidates.size() >= Config::kEfSearchVal)
        break;
      const Node& node = nodes_[c_id];
      int size = node.link_counts[0];
      int offset = GetLinkOffset_(0);
      const int* links = node.flat_links.data() + offset;
      for (int i = 0; i < size; ++i) {
        int neighbor_id = links[i];
        if (!visited.visit(neighbor_id)) {
          float d = QueryDistSqFP_(query_data, neighbor_id);
          if (top_candidates.size() < Config::kEfSearchVal ||
              d < top_candidates.top().first) {
            candidates.push({d, neighbor_id});
            top_candidates.push({d, neighbor_id});
            if (top_candidates.size() > Config::kEfSearchVal)
              top_candidates.pop();
          }
        }
      }
    }
  } else {  // kDynamic
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>>
        newly_added;
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>>
        candidates;
    newly_added.push({current_dist, entry_point});
    top_candidates.push({current_dist, entry_point});
    visited.visit(entry_point);

    // New variables for dynamic search stopping condition
    float best_dist_in_top_candidates = top_candidates.top().first;
    int no_improvement_iterations = 0;

    while (!newly_added.empty() || !candidates.empty()) {
      QueueItem current_item;
      if (!newly_added.empty()) {
        current_item = newly_added.top();
        newly_added.pop();
      } else {
        current_item = candidates.top();
        candidates.pop();
      }
      auto [c_dist, c_id] = current_item;

      // Check for no improvement
      if (top_candidates.top().first < best_dist_in_top_candidates) {
          best_dist_in_top_candidates = top_candidates.top().first;
          no_improvement_iterations = 0; // Reset counter
      } else {
          no_improvement_iterations++;
      }

      if (no_improvement_iterations >= Config::kDynamicMaxNoImprovementIterationsVal) {
          break;
      }


      if (c_dist > top_candidates.top().first &&
          top_candidates.size() >= Config::kEfSearchVal)
        break;
      const Node& node = nodes_[c_id];
      int size = node.link_counts[0];
      int offset = GetLinkOffset_(0);
      const int* links = node.flat_links.data() + offset;
      for (int i = 0; i < size; ++i) {
        int neighbor_id = links[i];
        if (!visited.visit(neighbor_id)) {
          float d = QueryDistSqFP_(query_data, neighbor_id);
          if (top_candidates.size() < Config::kEfSearchVal ||
              d < top_candidates.top().first) {
            newly_added.push({d, neighbor_id});
            top_candidates.push({d, neighbor_id});
            if (top_candidates.size() > Config::kEfSearchVal)
              top_candidates.pop();
          }
        }
      }
    }
  }

  auto t_end_l0 = std::chrono::high_resolution_clock::now();
  metrics_.add_navigation_time(
      0, std::chrono::duration_cast<std::chrono::nanoseconds>(t_end_l0 -
                                                              t_start_l0)
             .count());

  size_t k_idx = 0;
  std::vector<QueueItem> sorted_results;
  while (!top_candidates.empty()) {
    sorted_results.push_back(top_candidates.top());
    top_candidates.pop();
  }
  std::reverse(sorted_results.begin(), sorted_results.end());
  for (const auto& p : sorted_results) {
    if (k_idx >= global::kCRITERION) break;
    result_indices[k_idx++] = p.second;
  }
  while (k_idx < global::kCRITERION) result_indices[k_idx++] = -1;
}

template <typename Config>
void HNSW<Config>::SearchPQ_(const std::vector<float>& query,
                             int* result_indices) {
  static thread_local VisitedList visited;
  if (visited.tags.size() != num_points_) visited.resize(num_points_);

  const float* query_data = query.data();
  auto dist_table = pq_->BuildDistanceTable(query_data);

  auto query_dist_sq_pq = [&](int node_id) {
    metrics_.increment_distance_calculations();
    const uint8_t* code =
        pq_codes_.data() + node_id * Config::kPQSubquantizersVal;
    return pq_->GetDistanceFromTable(dist_table, code);
  };

  int entry_point = entry_point_;
  float current_dist = query_dist_sq_pq(entry_point);

  // Phase 1: Upper Layers (Greedy Descent)
  for (int level = max_level_; level > 0; --level) {
    auto t_start_layer = std::chrono::high_resolution_clock::now();
    bool changed = true;
    while (changed) {
      changed = false;
      const Node& node = nodes_[entry_point];
      if (level >= node.link_counts.size()) break;
      int count = node.link_counts[level];
      int offset = GetLinkOffset_(level);
      const int* links = node.flat_links.data() + offset;
      for (int i = 0; i < count; ++i) {
        int neighbor = links[i];
        float d = query_dist_sq_pq(neighbor);
        if (d < current_dist) {
          current_dist = d;
          entry_point = neighbor;
          changed = true;
        }
      }
    }
    auto t_end_layer = std::chrono::high_resolution_clock::now();
    metrics_.add_navigation_time(
        level, std::chrono::duration_cast<std::chrono::nanoseconds>(
                   t_end_layer - t_start_layer)
                   .count());
  }

  // Phase 2: Layer 0 (Fine-grained Search)
  auto t_start_l0 = std::chrono::high_resolution_clock::now();
  using QueueItem = std::pair<float, int>;
  std::priority_queue<QueueItem> top_candidates;
  std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>>
      candidates;
  visited.advance();
  candidates.push({current_dist, entry_point});
  top_candidates.push({current_dist, entry_point});
  visited.visit(entry_point);

  while (!candidates.empty()) {
    auto [c_dist, c_id] = candidates.top();
    candidates.pop();
    if (c_dist > top_candidates.top().first &&
        top_candidates.size() >= Config::kEfSearchVal)
      break;
    const Node& node = nodes_[c_id];
    int size = node.link_counts[0];
    int offset = GetLinkOffset_(0);
    const int* links = node.flat_links.data() + offset;
    for (int i = 0; i < size; ++i) {
      int neighbor_id = links[i];
      if (!visited.visit(neighbor_id)) {
        float d = query_dist_sq_pq(neighbor_id);
        if (top_candidates.size() < Config::kEfSearchVal ||
            d < top_candidates.top().first) {
          candidates.push({d, neighbor_id});
          top_candidates.push({d, neighbor_id});
          if (top_candidates.size() > Config::kEfSearchVal)
            top_candidates.pop();
        }
      }
    }
  }
  auto t_end_l0 = std::chrono::high_resolution_clock::now();
  metrics_.add_navigation_time(
      0, std::chrono::duration_cast<std::chrono::nanoseconds>(t_end_l0 -
                                                              t_start_l0)
             .count());

  // Reranking Step
  std::vector<QueueItem> candidates_to_rerank;
  candidates_to_rerank.reserve(top_candidates.size());
  while (!top_candidates.empty()) {
    candidates_to_rerank.push_back(top_candidates.top());
    top_candidates.pop();
  }

  for (const auto& item : candidates_to_rerank) {
    int candidate_id = item.second;
    float exact_dist = QueryDistSqFP_(query_data, candidate_id);
    if (top_candidates.size() < global::kCRITERION ||
        exact_dist < top_candidates.top().first) {
      top_candidates.push({exact_dist, candidate_id});
      if (top_candidates.size() > global::kCRITERION) {
        top_candidates.pop();
      }
    }
  }

  size_t k_idx = 0;
  std::vector<QueueItem> sorted_results;
  while (!top_candidates.empty()) {
    sorted_results.push_back(top_candidates.top());
    top_candidates.pop();
  }
  std::reverse(sorted_results.begin(), sorted_results.end());
  for (const auto& p : sorted_results) {
    if (k_idx >= global::kCRITERION) break;
    result_indices[k_idx++] = p.second;
  }
  while (k_idx < global::kCRITERION) result_indices[k_idx++] = -1;
}

template <typename Config>
void HNSW<Config>::PrintMetricsReport_() const {
  if constexpr (global::kDEBUG) {
    long long total_dist_calcs = metrics_.distance_calculations.load();
    std::cout << "\n======================================================\n";
    std::cout << "              TunableHNSW METRICS REPORT              \n";
    std::cout << "======================================================\n";
    std::cout << "Total Distance Calculations: " << total_dist_calcs << "\n";
    std::cout << "------------------------------------------------------\n";
    std::cout << "Total Navigation Time per Layer (ns):\n";
    for (int l = max_level_; l >= 0; --l) {
      long long layer_time = metrics_.navigation_time_per_layer[l].load();
      if (layer_time > 0 || l == 0) {
        std::cout << "  Layer " << std::setw(2) << l << ": " << std::setw(12)
                  << layer_time << " ns\n";
      }
    }
    std::cout << "======================================================\n";
  }
}

}  // namespace TunableHNSW

