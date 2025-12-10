#pragma once

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <vector>

#include "Global.h"
#include "SolutionConcept.h"

// Calculates the squared Euclidean distance between two vectors using AVX2.
// Returns the squared distance as a float.
__attribute__((target("avx2,fma"))) inline float SquaredDistance(
    const float* a, const float* b, int d) {
  __m256 sum = _mm256_setzero_ps();
  const float* end_safe = a + (d & ~7);

  while (a < end_safe) {
    __m256 v_a = _mm256_loadu_ps(a);
    __m256 v_b = _mm256_loadu_ps(b);
    __m256 diff = _mm256_sub_ps(v_a, v_b);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    a += 8;
    b += 8;
  }

  __m128 sum_low = _mm256_castps256_ps128(sum);
  __m128 sum_high = _mm256_extractf128_ps(sum, 1);
  __m128 v_res = _mm_add_ps(sum_low, sum_high);
  v_res = _mm_hadd_ps(v_res, v_res);
  v_res = _mm_hadd_ps(v_res, v_res);
  float res = _mm_cvtss_f32(v_res);

  int remainder = d & 7;
  for (int i = 0; i < remainder; ++i) {
    float diff = a[i] - b[i];
    res += diff * diff;
  }

  return res;
}

// Encapsulates the HNSW graph topology and connectivity logic.
template <int kMaxNeighbors, int kMaxLayer0Neighbors, int kMaxLevel>
class HnswGraph {
 public:
  struct Node {
    int level;
    // Flattened Adjacency List:
    // [Layer0 (M0) | Layer1 (M) | ... | LayerN (M)]
    std::vector<int> flat_links;
    std::vector<int> link_counts;
  };

  HnswGraph() { level_mult_ = 1.0 / std::log(1.0 * kMaxNeighbors); }

  void Initialize(size_t n, std::mt19937& rng) {
    nodes_.resize(n);
    for (size_t i = 0; i < n; ++i) {
      int level = GetRandomLevel(rng);
      nodes_[i].level = level;
      nodes_[i].link_counts.resize(level + 1, 0);
      size_t total_links = kMaxLayer0Neighbors;
      if (level > 0) total_links += (size_t)level * kMaxNeighbors;
      nodes_[i].flat_links.resize(total_links);
    }
    entry_point_ = 0;
    max_level_ = nodes_[0].level;
  }

  // --- Topology Accessors ---

  inline int GetLinkOffset(int level) const {
    return (level == 0) ? 0
                        : (kMaxLayer0Neighbors + (level - 1) * kMaxNeighbors);
  }

  inline int* GetNeighborsPtr(int node_id, int level) {
    return nodes_[node_id].flat_links.data() + GetLinkOffset(level);
  }

  inline const int* GetNeighborsPtr(int node_id, int level) const {
    return nodes_[node_id].flat_links.data() + GetLinkOffset(level);
  }

  inline int GetNeighborCount(int node_id, int level) const {
    return nodes_[node_id].link_counts[level];
  }

  bool HasNeighbor(int node_id, int level, int target_id) const {
    const int* links = GetNeighborsPtr(node_id, level);
    int count = GetNeighborCount(node_id, level);
    for (int i = 0; i < count; ++i) {
      if (links[i] == target_id) return true;
    }
    return false;
  }

  // --- Topology Modifiers ---

  void AppendNeighbor(int node_id, int level, int target_id) {
    int count = nodes_[node_id].link_counts[level];
    int* links = GetNeighborsPtr(node_id, level);
    links[count] = target_id;
    nodes_[node_id].link_counts[level] = count + 1;
  }

  void SetNeighbors(int node_id, int level, const int* new_neighbors,
                    int new_count) {
    int* links = GetNeighborsPtr(node_id, level);
    for (int i = 0; i < new_count; ++i) {
      links[i] = new_neighbors[i];
    }
    nodes_[node_id].link_counts[level] = new_count;
  }

  // --- Getters/Setters for Global State ---

  int GetNodeLevel(int node_id) const { return nodes_[node_id].level; }

  int GetEntryPoint() const { return entry_point_; }
  void SetEntryPoint(int entry_point) { entry_point_ = entry_point; }

  int GetMaxLevel() const { return max_level_; }
  void SetMaxLevel(int max_level) { max_level_ = max_level; }

 private:
  int GetRandomLevel(std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = -std::log(dist(rng)) * level_mult_;
    return std::min(static_cast<int>(r), kMaxLevel);
  }

  std::vector<Node> nodes_;
  int entry_point_ = -1;
  int max_level_ = -1;
  double level_mult_;
};

// Main Solution class implementing HNSW construction and search.
class Solution {
 public:
  // --- Hyperparameters ---
  static constexpr int kM = 32;
  static constexpr int kM0 = 64;  // kM * 2
  static constexpr int kEfConstruction = 600;
  static constexpr int kEfSearch = 100;
  static constexpr int kMaxLevel = 16;

  ~Solution() {
    if constexpr (global::kDEBUG) {
      if (total_searches_ > 0) {
        std::cout << "  - Avg Distance Calcs per Search: "
                  << static_cast<double>(total_dist_calcs_) / total_searches_
                  << "\n";
      }
    }
  }

  void build(int d, const std::vector<float>& base) {
    d_ = d;
    data_storage_ = base;
    data_ptr_ = data_storage_.data();
    n_ = base.size() / d_;
    visited_list_.Resize(n_);

    graph_.Initialize(n_, global::rng);

    if constexpr (global::kDEBUG) {
      std::cout << "Building HNSW (d=" << d_ << ", M=" << kM
                << ") for " << n_ << " vectors...\n";
    }

    for (size_t curr_obj = 1; curr_obj < n_; ++curr_obj) {
      InsertVec(curr_obj, visited_list_);

      if constexpr (global::kDEBUG) {
        if (curr_obj % 1000 == 0) {
          std::cout << "Build: " << static_cast<int>(100.0 * curr_obj / n_)
                    << "% \r" << std::flush;
        }
      }
    }

    if constexpr (global::kDEBUG) {
      std::cout << "Build: 100% - Done.\n";
      total_dist_calcs_ = 0;
    }
  }

  void search(const std::vector<float>& query, int* res) {
    if (visited_list_.Size() != n_) visited_list_.Resize(n_);

    int curr_ep = graph_.GetEntryPoint();
    int m_level = graph_.GetMaxLevel();
    const float* q_data = query.data();

    curr_ep = GreedySearch(q_data, curr_ep, m_level, 0);
    auto top = BeamSearch(q_data, curr_ep, kEfSearch, 0, visited_list_);

    size_t k = 0;
    std::vector<std::pair<float, int>> sorted;
    sorted.reserve(top.size());
    while (!top.empty()) {
      sorted.push_back(top.top());
      top.pop();
    }
    std::sort(sorted.begin(), sorted.end());

    for (const auto& p : sorted) {
      if (k >= 10) break;
      res[k++] = p.second;
    }
    while (k < 10) res[k++] = -1;

    if constexpr (global::kDEBUG) {
      total_searches_++;
    }
  }

 private:
  HnswGraph<kM, kM0, kMaxLevel> graph_;

  size_t total_dist_calcs_ = 0;
  size_t total_searches_ = 0;

  int d_ = 0;
  size_t n_ = 0;

  const float* data_ptr_ = nullptr;
  std::vector<float> data_storage_;

  struct VisitedList {
    std::vector<unsigned short> tags;
    unsigned short current_tag = 0;

    void Resize(size_t n) { tags.resize(n, 0); }

    size_t Size() const { return tags.size(); }

    void Advance() {
      current_tag++;
      if (current_tag == 0) {
        std::fill(tags.begin(), tags.end(), 0);
        current_tag = 1;
      }
    }

    inline bool Visit(int id) {
      if (tags[id] == current_tag) return true;
      tags[id] = current_tag;
      return false;
    }
  };

  VisitedList visited_list_;

  inline float CalcDist(const float* a, const float* b) {
    if constexpr (global::kDEBUG) {
      total_dist_calcs_++;
    }
    return SquaredDistance(a, b, d_);
  }

  // --- Heuristic Neighbor Selection ---
  void GetNeighborsHeuristic(int src,
                             std::vector<std::pair<float, int>>& candidates,
                             int level, int* output_buffer,
                             int& output_count) {
    int max_m = (level == 0) ? kM0 : kM;
    output_count = 0;
    if (candidates.empty()) return;

    std::sort(candidates.begin(), candidates.end());

    for (const auto& cand : candidates) {
      if (output_count >= max_m) break;
      int cand_id = cand.second;
      float dist_to_src = cand.first;

      bool good = true;
      for (int j = 0; j < output_count; ++j) {
        float d_neighbor = CalcDist(data_ptr_ + cand_id * d_,
                                    data_ptr_ + output_buffer[j] * d_);
        if (d_neighbor < dist_to_src) {
          good = false;
          break;
        }
      }
      if (good) output_buffer[output_count++] = cand_id;
    }
  }

  // --- Connection Logic ---
  void AddConnection(int src, int dest, int level) {
    if (graph_.HasNeighbor(src, level, dest)) return;

    int count = graph_.GetNeighborCount(src, level);
    int max_m = (level == 0) ? kM0 : kM;

    // 1. Simple Case: Append if space is available.
    if (count < max_m) {
      graph_.AppendNeighbor(src, level, dest);
    }
    // 2. Pruning Case: Graph is full, run heuristic.
    else {
      std::vector<std::pair<float, int>> candidates;
      candidates.reserve(max_m + 1);

      const int* links_ptr = graph_.GetNeighborsPtr(src, level);
      for (int i = 0; i < count; ++i) {
        float d = CalcDist(data_ptr_ + src * d_,
                           data_ptr_ + links_ptr[i] * d_);
        candidates.push_back({d, links_ptr[i]});
      }

      float d_dest = CalcDist(data_ptr_ + src * d_, data_ptr_ + dest * d_);
      candidates.push_back({d_dest, dest});

      std::vector<int> new_links(max_m);
      int new_count = 0;
      GetNeighborsHeuristic(src, candidates, level, new_links.data(),
                            new_count);

      graph_.SetNeighbors(src, level, new_links.data(), new_count);
    }
  }

  int InsertIntoLayer(int new_vec_id, int seed, int level,
                      VisitedList& visited) {
    const float* curr_vec = data_ptr_ + new_vec_id * d_;
    auto top_candidates =
        BeamSearch(curr_vec, seed, kEfConstruction, level, visited);

    std::vector<std::pair<float, int>> potential;
    potential.reserve(kEfConstruction + 1);

    int best_next_seed = seed;
    float min_dist = std::numeric_limits<float>::max();

    while (!top_candidates.empty()) {
      auto p = top_candidates.top();
      top_candidates.pop();
      potential.push_back(p);
      if (p.first < min_dist) {
        min_dist = p.first;
        best_next_seed = p.second;
      }
    }

    std::vector<int> link_dst_buffer(kM0);  // Buffer for heuristic result
    int count = 0;

    // Select neighbors for the new node.
    GetNeighborsHeuristic(new_vec_id, potential, level,
                          link_dst_buffer.data(), count);

    // Set neighbors for the new node (it was empty, so simple set).
    graph_.SetNeighbors(new_vec_id, level, link_dst_buffer.data(), count);

    // Add back-links.
    for (int j = 0; j < count; ++j) {
      AddConnection(link_dst_buffer[j], new_vec_id, level);
    }

    return best_next_seed;
  }

  int GreedySearch(const float* query_data, int entry_point, int start_level,
                   int stop_level) {
    int curr_node = entry_point;

    for (int l = start_level; l > stop_level; l--) {
      bool changed = true;
      while (changed) {
        changed = false;
        float dist_min =
            CalcDist(query_data, data_ptr_ + curr_node * d_);
        
        // Check if we can descend at this node
        if (l > graph_.GetNodeLevel(curr_node)) break;

        int count = graph_.GetNeighborCount(curr_node, l);
        const int* links = graph_.GetNeighborsPtr(curr_node, l);

        for (int i = 0; i < count; ++i) {
          int neighbor = links[i];
          if (i + 1 < count) {
            _mm_prefetch(
                reinterpret_cast<const char*>(data_ptr_ + links[i + 1] * d_),
                _MM_HINT_T0);
          }

          float d = CalcDist(query_data, data_ptr_ + neighbor * d_);
          if (d < dist_min) {
            curr_node = neighbor;
            dist_min = d;
            changed = true;
          }
        }
      }
    }
    return curr_node;
  }

  std::priority_queue<std::pair<float, int>> BeamSearch(
      const float* query_data, int entry_point, int ef, int level,
      VisitedList& visited) {
    using QueueItem = std::pair<float, int>;
    std::priority_queue<QueueItem> top_candidates;
    std::priority_queue<QueueItem, std::vector<QueueItem>,
                        std::greater<>>
        candidates;

    visited.Advance();
    float initial_dist =
        CalcDist(query_data, data_ptr_ + entry_point * d_);
    top_candidates.push({initial_dist, entry_point});
    candidates.push({initial_dist, entry_point});
    visited.Visit(entry_point);

    while (!candidates.empty()) {
      auto [curr_dist, curr_id] = candidates.top();
      if (curr_dist > top_candidates.top().first &&
          top_candidates.size() >= ef)
        break;
      candidates.pop();

      int size = graph_.GetNeighborCount(curr_id, level);
      const int* links = graph_.GetNeighborsPtr(curr_id, level);

      for (int i = 0; i < size; ++i) {
        int neighbor_id = links[i];
        if (i + 1 < size) {
          _mm_prefetch(
              reinterpret_cast<const char*>(data_ptr_ + links[i + 1] * d_),
              _MM_HINT_T0);
        }

        if (!visited.Visit(neighbor_id)) {
          float d =
              CalcDist(query_data, data_ptr_ + neighbor_id * d_);
          if (top_candidates.size() < ef ||
              d < top_candidates.top().first) {
            candidates.push({d, neighbor_id});
            top_candidates.push({d, neighbor_id});
            if (top_candidates.size() > ef) top_candidates.pop();
          }
        }
      }
    }
    return top_candidates;
  }

  std::priority_queue<std::pair<float, int>> BeamSearchDynamic(
      const float* query_data, int entry_point, int ef, int level,
      VisitedList& visited) {
    using QueueItem = std::pair<float, int>;
    std::priority_queue<QueueItem> top_candidates;
    std::priority_queue<QueueItem, std::vector<QueueItem>,
                        std::greater<>>
        candidates;

    visited.Advance();
    float initial_dist =
        CalcDist(query_data, data_ptr_ + entry_point * d_);
    top_candidates.push({initial_dist, entry_point});
    candidates.push({initial_dist, entry_point});
    visited.Visit(entry_point);

    while (!candidates.empty()) {
      auto [curr_dist, curr_id] = candidates.top();
      if (curr_dist > top_candidates.top().first &&
          top_candidates.size() >= ef)
        break;
      candidates.pop();

      int size = graph_.GetNeighborCount(curr_id, level);
      const int* links = graph_.GetNeighborsPtr(curr_id, level);

      for (int i = 0; i < size; ++i) {
        int neighbor_id = links[i];
        if (i + 1 < size) {
          _mm_prefetch(
              reinterpret_cast<const char*>(data_ptr_ + links[i + 1] * d_),
              _MM_HINT_T0);
        }

        if (!visited.Visit(neighbor_id)) {
          float d =
              CalcDist(query_data, data_ptr_ + neighbor_id * d_);
          if (top_candidates.size() < ef ||
              d < top_candidates.top().first) {
            candidates.push({d, neighbor_id});
            top_candidates.push({d, neighbor_id});
            if (top_candidates.size() > ef) top_candidates.pop();
          }
        }
      }
    }
    return top_candidates;
  }

  void InsertVec(int new_vec_id, VisitedList& visited) {
    int curr_level = graph_.GetNodeLevel(new_vec_id);
    int curr_ep = graph_.GetEntryPoint();
    int curr_max = graph_.GetMaxLevel();
    const float* curr_vec = data_ptr_ + new_vec_id * d_;

    curr_ep = GreedySearch(curr_vec, curr_ep, curr_max, curr_level);

    for (int l = std::min(curr_level, curr_max); l >= 0; l--) {
      curr_ep = InsertIntoLayer(new_vec_id, curr_ep, l, visited);
    }

    if (curr_level > graph_.GetMaxLevel()) {
      graph_.SetMaxLevel(curr_level);
      graph_.SetEntryPoint(new_vec_id);
    }
  }
};

static_assert(IsSolution<Solution>);
