#pragma once

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <queue>
#include <random>
#include <thread>
#include <vector>

#include "Global.h"
#include "SolutionConcept.h"

// Standard AVX2 Distance Calculation
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

// Reverted to Standard Layout (Better Cache Locality for Graph Traversal)
template <int kMaxNeighbors, int kMaxLayer0Neighbors, int kMaxLevel>
class HnswGraph {
public:
  struct Node {
    int level;
    // Back to standard vectors. 
    // The allocator often places these near the Node struct, reducing TLB misses 
    // compared to a single massive array.
    std::vector<int> flat_links;
    std::vector<int> link_counts;
  };

  HnswGraph() { 
    level_mult_ = 1.0 / std::log(1.0 * kMaxNeighbors);
    node_locks_ = std::vector<std::mutex>(8192); 
  }

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
    std::memcpy(links, new_neighbors, new_count * sizeof(int));
    nodes_[node_id].link_counts[level] = new_count;
  }

  // --- Locks & Global State ---

  std::mutex& GetLock(int node_id) const {
    return node_locks_[node_id % node_locks_.size()];
  }
  std::mutex& GetGlobalLock() const { return global_lock_; }

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

  mutable std::vector<std::mutex> node_locks_;
  mutable std::mutex global_lock_;
};

class Solution {
public:
  static constexpr int kM = 64;
  static constexpr int kM0 = 128;
  static constexpr int kEfConstruction = 600;
  static constexpr int kEfSearch = 300;
  static constexpr int kMaxLevel = 16;

  Solution() {
    layer_times_ns_ = std::vector<std::atomic<long long>>(kMaxLevel + 1);
    for(auto& x : layer_times_ns_) x = 0;
  }

  ~Solution() {
    if constexpr (global::kDEBUG) {
      long long total_s = total_searches_.load();
      if (total_s > 0) {
        std::cout << "\n=== HNSW Profiling Stats (Optimized Topology) ===\n";
        std::cout << "Total Searches: " << total_s << "\n";
        std::cout << "Avg Distance Calcs: "
          << static_cast<double>(total_dist_calcs_.load()) / total_s
          << "\n";

        std::cout << "--- Avg Time per Layer ---\n";
        long long total_time = 0;
        for (int l = kMaxLevel; l >= 0; --l) {
          long long t = layer_times_ns_[l].load();
          if (t > 0) {
            double avg_us = (double)t / total_s / 1000.0;
            std::cout << "Layer " << l << ": " << avg_us << " us\n";
            total_time += t;
          }
        }
        std::cout << "Total Avg Latency: " << (double)total_time / total_s / 1000.0 << " us\n";
        std::cout << "==========================\n";
      }
    }
  }

  void build(int d, const std::vector<float>& base) {
    d_ = d;
    data_storage_ = base;
    data_ptr_ = data_storage_.data();
    n_ = base.size() / d_;

    graph_.Initialize(n_, global::rng);

    if constexpr (global::kDEBUG) {
      std::cout << "Building HNSW (Optimized Topology: M=" << kM
        << ", M0=" << kM0 << ") using " 
        << std::thread::hardware_concurrency() << " threads...\n";
    }

    std::atomic<size_t> atomic_curr_obj(1);
    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i) {
      threads.emplace_back([&]() {
        VisitedList local_visited; 
        local_visited.Resize(n_);

        while(true) {
          size_t curr_obj = atomic_curr_obj.fetch_add(1);
          if (curr_obj >= n_) break;
          InsertVec(curr_obj, local_visited);
          if constexpr (global::kDEBUG) {
            if (curr_obj % 5000 == 0 && curr_obj % num_threads == 0) {
              std::cout << "Build: " << static_cast<int>(100.0 * curr_obj / n_)
                << "% \r" << std::flush;
            }
          }
        }
      });
    }

    for (auto& t : threads) {
      if (t.joinable()) t.join();
    }

    if constexpr (global::kDEBUG) {
      std::cout << "Build: 100% - Done.\n";
      total_dist_calcs_ = 0;
    }
  }

  void search(const std::vector<float>& query, int* res) {
    thread_local VisitedList visited_list;
    if (visited_list.Size() != n_) visited_list.Resize(n_);

    int curr_ep = graph_.GetEntryPoint();
    int m_level = graph_.GetMaxLevel();
    const float* q_data = query.data();

    // 1. Upper Layers: Greedy Search
    curr_ep = GreedySearch(q_data, curr_ep, m_level, 0, true);

    // 2. Layer 0: Beam Search
    {
      auto t_start = std::chrono::high_resolution_clock::now();

      auto top = BeamSearchDynamic(q_data, curr_ep, kEfSearch, 0, visited_list);

      auto t_end = std::chrono::high_resolution_clock::now();
      if constexpr (global::kDEBUG) {
        layer_times_ns_[0].fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count());
      }

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
    }

    if constexpr (global::kDEBUG) {
      total_searches_.fetch_add(1);
    }
  }

private:
  HnswGraph<kM, kM0, kMaxLevel> graph_;

  std::atomic<size_t> total_dist_calcs_{0};
  std::atomic<size_t> total_searches_{0};
  std::vector<std::atomic<long long>> layer_times_ns_;

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
    inline unsigned short* GetTagPtr(int id) {
      return &tags[id];
    }
  };

  inline float CalcDist(const float* a, const float* b) {
    if constexpr (global::kDEBUG) {
      total_dist_calcs_.fetch_add(1, std::memory_order_relaxed);
    }
    return SquaredDistance(a, b, d_);
  }

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

  void AddConnection(int src, int dest, int level) {
    if (graph_.HasNeighbor(src, level, dest)) return;
    std::lock_guard<std::mutex> lock(graph_.GetLock(src));

    int count = graph_.GetNeighborCount(src, level);
    int max_m = (level == 0) ? kM0 : kM;

    if (count < max_m) {
      graph_.AppendNeighbor(src, level, dest);
    } else {
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

    std::vector<int> link_dst_buffer(kM0);
    int count = 0;
    GetNeighborsHeuristic(new_vec_id, potential, level,
                          link_dst_buffer.data(), count);

    graph_.SetNeighbors(new_vec_id, level, link_dst_buffer.data(), count);

    for (int j = 0; j < count; ++j) {
      AddConnection(link_dst_buffer[j], new_vec_id, level);
    }
    return best_next_seed;
  }

  int GreedySearch(const float* query_data, int entry_point, int start_level,
                   int stop_level, bool record_profiling = false) {
    int curr_node = entry_point;
    for (int l = start_level; l > stop_level; l--) {
      auto t_start = std::chrono::high_resolution_clock::now();
      bool changed = true;
      while (changed) {
        changed = false;
        float dist_min = CalcDist(query_data, data_ptr_ + curr_node * d_);

        if (l > graph_.GetNodeLevel(curr_node)) break;

        int count = graph_.GetNeighborCount(curr_node, l);
        const int* links = graph_.GetNeighborsPtr(curr_node, l);

        if(count > 0) {
          _mm_prefetch(reinterpret_cast<const char*>(data_ptr_ + links[0] * d_), _MM_HINT_T0);
        }

        for (int i = 0; i < count; ++i) {
          int neighbor = links[i];
          if (i + 1 < count) {
            _mm_prefetch(reinterpret_cast<const char*>(data_ptr_ + links[i + 1] * d_), _MM_HINT_T0);
          }

          float d = CalcDist(query_data, data_ptr_ + neighbor * d_);
          if (d < dist_min) {
            curr_node = neighbor;
            dist_min = d;
            changed = true;
          }
        }
      }
      if (record_profiling && global::kDEBUG) {
        auto t_end = std::chrono::high_resolution_clock::now();
        layer_times_ns_[l].fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count());
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
    float initial_dist = CalcDist(query_data, data_ptr_ + entry_point * d_);
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

      // PREFETCHING OPTIMIZATION
      if(size > 0) {
        _mm_prefetch(reinterpret_cast<const char*>(data_ptr_ + links[0] * d_), _MM_HINT_T0);
        // Prefetch the visited tag for the first neighbor too
        _mm_prefetch(reinterpret_cast<const char*>(visited.GetTagPtr(links[0])), _MM_HINT_T0);
      }

      for (int i = 0; i < size; ++i) {
        int neighbor_id = links[i];
        if (i + 1 < size) {
          _mm_prefetch(reinterpret_cast<const char*>(data_ptr_ + links[i + 1] * d_), _MM_HINT_T0);
          _mm_prefetch(reinterpret_cast<const char*>(visited.GetTagPtr(links[i + 1])), _MM_HINT_T0);
        }

        if (!visited.Visit(neighbor_id)) {
          float d = CalcDist(query_data, data_ptr_ + neighbor_id * d_);
          if (top_candidates.size() < ef || d < top_candidates.top().first) {
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
    const float* query_data,
    int entry_point,
    int ef,
    int level,
    VisitedList& visited
  ) {
    using QueueItem = std::pair<float, int>;
    
    // Result Set (Max Heap): Keeps track of the best 'ef' nodes found
    std::priority_queue<QueueItem> top_candidates;
    
    // Global Queue (Min Heap): Standard pool of unexpanded nodes
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> global_candidates;

    // Local Queue (Min Heap): The "Newly Added" nodes
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> local_candidates;

    visited.Advance();
    float initial_dist = CalcDist(query_data, data_ptr_ + entry_point * d_);
    
    top_candidates.push({initial_dist, entry_point});
    
    // Start with the entry point as "newly added"
    local_candidates.push({initial_dist, entry_point});
    visited.Visit(entry_point);

    while (!global_candidates.empty() || !local_candidates.empty()) {
      QueueItem curr_item;
      bool is_local_search = false;

      // 1. Selection Logic: Prioritize "Newly Added" (Local)
      if (!local_candidates.empty()) {
        is_local_search = true;
        curr_item = local_candidates.top();
        local_candidates.pop();

        // Demote siblings: Since we are descending into 'curr_item', 
        // its siblings are no longer the "newest" focus. Move them to global.
        while(!local_candidates.empty()) {
            global_candidates.push(local_candidates.top());
            local_candidates.pop();
        }
      } else {
        // Fallback to global if the "new" path dried up
        curr_item = global_candidates.top();
        global_candidates.pop();
      }

      auto [curr_dist, curr_id] = curr_item;

      // 2. Stopping Condition
      // If the current node is worse than the worst node in our result buffer,
      // we generally stop.
      if (top_candidates.size() >= ef && curr_dist > top_candidates.top().first) {
        if (is_local_search) {
          // If we were following a "new" path and it got bad, strictly stop this path
          // but DO NOT terminate the whole search, because 'global' might still have
          // good nodes we skipped.
          continue; 
        } else {
          // If the best node in 'global' is bad, then everything is bad. We can stop.
          break;
        }
      }

      // 3. Expansion
      int size = graph_.GetNeighborCount(curr_id, level);
      const int* links = graph_.GetNeighborsPtr(curr_id, level);

      // Prefetching Optimization
      if(size > 0) {
        _mm_prefetch(reinterpret_cast<const char*>(data_ptr_ + links[0] * d_), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(visited.GetTagPtr(links[0])), _MM_HINT_T0);
      }

      for (int i = 0; i < size; ++i) {
        int neighbor_id = links[i];
        if (i + 1 < size) {
          _mm_prefetch(reinterpret_cast<const char*>(data_ptr_ + links[i + 1] * d_), _MM_HINT_T0);
          _mm_prefetch(reinterpret_cast<const char*>(visited.GetTagPtr(links[i + 1])), _MM_HINT_T0);
        }

        if (!visited.Visit(neighbor_id)) {
          float d = CalcDist(query_data, data_ptr_ + neighbor_id * d_);
          
          if (top_candidates.size() < ef || d < top_candidates.top().first) {
            // Push to LOCAL (Newly Added) queue to investigate immediately next
            local_candidates.push({d, neighbor_id});
            
            top_candidates.push({d, neighbor_id});
            if (top_candidates.size() > ef) top_candidates.pop();
          }
        }
      }
    }
    return top_candidates;
  }

  void InsertVec(int new_vec_id, VisitedList& visited) {
    int curr_ep = graph_.GetEntryPoint();
    int curr_max = graph_.GetMaxLevel();
    int curr_level = graph_.GetNodeLevel(new_vec_id);
    const float* curr_vec = data_ptr_ + new_vec_id * d_;

    {
      std::lock_guard<std::mutex> lock(graph_.GetGlobalLock());
      if (curr_ep == -1) {
        graph_.SetEntryPoint(new_vec_id);
        graph_.SetMaxLevel(curr_level);
        return;
      }
    }

    curr_ep = graph_.GetEntryPoint();
    curr_max = graph_.GetMaxLevel();
    curr_ep = GreedySearch(curr_vec, curr_ep, curr_max, curr_level, false);

    for (int l = std::min(curr_level, curr_max); l >= 0; l--) {
      curr_ep = InsertIntoLayer(new_vec_id, curr_ep, l, visited);
    }

    if (curr_level > curr_max) {
      std::lock_guard<std::mutex> lock(graph_.GetGlobalLock());
      if (curr_level > graph_.GetMaxLevel()) {
        graph_.SetMaxLevel(curr_level);
        graph_.SetEntryPoint(new_vec_id);
      }
    }
  }
};

static_assert(IsSolution<Solution>);
