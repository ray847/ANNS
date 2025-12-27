#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <queue>
#include <cmath>
#include <cstring>
#include <mutex>
#include <memory>
#include <thread>
#include <atomic>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <immintrin.h>
#include <memory_resource> // [PMR] Required

#include "Global.h"

class Solution {
private:
  // --- CONSTANTS ---
  static constexpr int M = 48;
  static constexpr int M0 = 96;
  static constexpr int ef_construction = 500;
  static constexpr int ef_search = 300;
  static constexpr int MAX_LEVEL = 16;

  // --- DATA ---
  int d_ = 0;
  size_t n_ = 0;

  // [PMR] 1. Define the monotonic resource. 
  // Declared before nodes_ to ensure correct destruction order.
  std::pmr::monotonic_buffer_resource memory_pool_;

  // Scalar Quantization Data
  std::vector<uint16_t> data_sq_;
  const uint16_t* data_sq_ptr_ = nullptr;

  // Quantization Parameters (per dimension)
  std::vector<float> min_vals_;
  std::vector<float> scale_vals_;    
  std::vector<float> scale_sq_vals_; 

  struct Node {
    int level;
    // [PMR] 2. Use pmr::vector for graph structure
    std::pmr::vector<int> flat_links;
    std::pmr::vector<int> link_counts;
    std::unique_ptr<std::mutex> lock;

    // [PMR] 3. Constructor accepts memory resource
    Node(int lvl, int total_links, std::pmr::memory_resource* mr)
      : level(lvl),
      flat_links(total_links, mr),
      link_counts(lvl + 1, 0, mr),
      lock(std::make_unique<std::mutex>()) 
    {}

    // Move-only
    Node(Node&&) = default;
    Node& operator=(Node&&) = default;
    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;
  };
  std::vector<Node> nodes_;

  int entry_point_ = -1;
  int max_level_ = -1;
  double level_mult_;
  std::mutex global_lock_;

  // --- DIAGNOSTICS & STATS ---
  struct SearchStats {
    long long count = 0;
    long long total_layer0_hops = 0;
    double sum_ratios = 0.0;
    std::vector<long long> layer_times_ns;

    SearchStats() : layer_times_ns(MAX_LEVEL + 1, 0) {}
  };

  mutable SearchStats stats_;
  mutable std::mutex stats_mutex_;

  // --- HELPER STRUCT: VISITED LIST ---
  struct VisitedList {
    std::vector<unsigned short> tags;
    unsigned short current_tag = 0;

    void resize(size_t n) { tags.resize(n, 0); }

    void advance() {
      current_tag++;
      if (current_tag == 0) {
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

public:
  // --- DESTRUCTOR ---
  ~Solution() {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    if (stats_.count == 0) return;

    double avg_hops = (double)stats_.total_layer0_hops / stats_.count;
    double avg_ratio = stats_.sum_ratios / stats_.count;

    std::cout << "\n======================================================\n";
    std::cout << "              HNSW SQ16 SEARCH PERFORMANCE            \n";
    std::cout << "======================================================\n";
    std::cout << "Total Queries:           " << stats_.count << "\n";
    std::cout << "Avg Layer 0 Hops:        " << std::fixed << std::setprecision(1) << avg_hops << "\n";
    std::cout << "Avg Entry Point Ratio:   " << std::setprecision(3) << avg_ratio << " (1.0 is perfect)\n";
    std::cout << "------------------------------------------------------\n";
    std::cout << "Avg Time per Layer (ns):\n";

    long long total_avg_time = 0;
    for (int l = MAX_LEVEL; l >= 0; l--) {
      if (l > max_level_ && stats_.layer_times_ns[l] == 0) continue;

      double avg_ns = (double)stats_.layer_times_ns[l] / stats_.count;
      total_avg_time += (long long)avg_ns;

      std::cout << "  Layer " << std::setw(2) << l << ": "
        << std::setw(8) << (long long)avg_ns << " ns";

      if (l == 0) std::cout << " (Dense Search)";
      else if (l == max_level_) std::cout << " (Entry)";
      std::cout << "\n";
    }
    std::cout << "------------------------------------------------------\n";
    std::cout << "Total Avg Latency:       " << total_avg_time / 1000.0 << " us\n";
    std::cout << "======================================================\n";
  }

  // --- BUILD FUNCTION ---
  void build(int d, const std::vector<float>& base) {
    d_ = d;
    n_ = base.size() / d_;

    // 1. Compute Statistics (Min, Max) for Quantization
    min_vals_.assign(d, std::numeric_limits<float>::max());
    std::vector<float> max_vals(d, std::numeric_limits<float>::lowest());

    for (size_t i = 0; i < n_; ++i) {
      const float* vec = base.data() + i * d;
      for (int j = 0; j < d; ++j) {
        if (vec[j] < min_vals_[j]) min_vals_[j] = vec[j];
        if (vec[j] > max_vals[j]) max_vals[j] = vec[j];
      }
    }

    scale_vals_.resize(d);
    scale_sq_vals_.resize(d);
    for (int j = 0; j < d; ++j) {
      float range = max_vals[j] - min_vals_[j];
      if (range < 1e-9f) range = 1e-9f;
      scale_vals_[j] = range / 65535.0f;
      scale_sq_vals_[j] = scale_vals_[j] * scale_vals_[j];
    }

    // 2. Quantize Data
    data_sq_.resize(n_ * d);
    data_sq_ptr_ = data_sq_.data();

    for (size_t i = 0; i < n_; ++i) {
      const float* src = base.data() + i * d;
      uint16_t* dst = data_sq_.data() + i * d;
      for (int j = 0; j < d; ++j) {
        float val = (src[j] - min_vals_[j]) / scale_vals_[j];
        dst[j] = static_cast<uint16_t>(std::round(val));
      }
    }

    // 3. Initialize Graph with PMR
    nodes_.clear();
    nodes_.reserve(n_); // Reserve main vector to avoid resizing/copying Nodes
    level_mult_ = 1.0 / std::log(1.0 * M);

    std::mt19937 rng_init(42);

    // [PMR] Serial Allocation Loop
    // Allocating sequentially here ensures the monotonic buffer is packed tightly 
    // in order of node ID, optimizing spatial locality.
    for (size_t i = 0; i < n_; ++i) {
      int level = get_random_level(rng_init);
      size_t total_links = M0;
      if (level > 0) total_links += (size_t)level * M;

      // Emplace back passes arguments to Node constructor
      nodes_.emplace_back(level, total_links, &memory_pool_);
    }

    entry_point_ = 0;
    max_level_ = nodes_[0].level;

    std::atomic<size_t> atomic_idx{ 1 };

    if constexpr (global::kDEBUG) {
      std::cout << "Building HNSW (d=" << d_ << ", M=" << M << ") for " << n_ << " vectors (SQ16)..." << std::endl;
    }

    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;

    auto worker_func = [&](int thread_id) {
      VisitedList visited;
      visited.resize(n_);
      std::mt19937 rng(42 + thread_id);

      // Thread-local temporary buffer for transformed vectors
      std::vector<float> curr_vec_trans(d_);

      while (true) {
        size_t curr_obj = atomic_idx.fetch_add(1, std::memory_order_relaxed);
        if (curr_obj >= n_) break;

        int curr_level = nodes_[curr_obj].level;
        int curr_ep = entry_point_;
        int curr_max_level = max_level_;

        // Reconstruct float-like query from SQ data for the current object
        const uint16_t* curr_sq = data_sq_ptr_ + curr_obj * d_;
        for(int j=0; j<d_; ++j) {
          curr_vec_trans[j] = (float)curr_sq[j];
        }

        // 1. Descent
        for (int l = curr_max_level; l > curr_level; l--) {
          bool changed = true;
          while (changed) {
            changed = false;
            float dist_ep = dist_query_sq(curr_vec_trans.data(), curr_ep);
            const Node& node_ep = nodes_[curr_ep];
            if (l >= node_ep.link_counts.size()) break;

            int offset = get_link_offset(l);
            int count = node_ep.link_counts[l];
            // [PMR] Accessing pmr::vector data
            const int* links = node_ep.flat_links.data() + offset;

            for (int j = 0; j < count; ++j) {
              int neighbor = links[j];
              float d = dist_query_sq(curr_vec_trans.data(), neighbor);
              if (d < dist_ep) {
                curr_ep = neighbor;
                dist_ep = d;
                changed = true;
              }
            }
          }
        }

        // 2. Construction
        for (int l = std::min(curr_level, curr_max_level); l >= 0; l--) {
          auto top_candidates = search_layer_build(curr_vec_trans.data(), curr_ep, ef_construction, l, visited);
          std::vector<std::pair<float, int>> potential;
          potential.reserve(ef_construction + 1);
          while (!top_candidates.empty()) {
            potential.push_back(top_candidates.top());
            top_candidates.pop();
          }

          int offset = get_link_offset(l);
          // [PMR] Accessing pmr::vector data
          int* link_dst = nodes_[curr_obj].flat_links.data() + offset;
          int count = 0;
          get_neighbors_heuristic(curr_obj, potential, l, link_dst, count);
          nodes_[curr_obj].link_counts[l] = count;

          for (int j = 0; j < count; ++j) add_connection(link_dst[j], curr_obj, l);
          if (!potential.empty()) curr_ep = potential[0].second;
        }

        if (curr_level > max_level_) {
          std::lock_guard<std::mutex> lock(global_lock_);
          if (curr_level > max_level_) {
            max_level_ = curr_level;
            entry_point_ = curr_obj;
          }
        }
      }
    };

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (unsigned int i = 0; i < num_threads; ++i) {
      threads.emplace_back(worker_func, i);
    }

    std::thread progress_thread([&]() {
      while (true) {
        size_t curr = atomic_idx.load(std::memory_order_relaxed);
        if (curr > n_) curr = n_;
        float progress = (n_ > 0) ? ((float)curr / n_ * 100.0f) : 100.0f;
        std::cout << "\rBuild Progress: " << std::fixed << std::setprecision(1) << progress << "%" << std::flush;
        if (curr >= n_) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
      }
      std::cout << "\rBuild Progress: 100.0%       " << std::endl;
    });

    for (auto& t : threads) {
      t.join();
    }
    progress_thread.join();
    if constexpr (global::kDEBUG) std::cout << "Build: 100% - Done.\n";
  }

  // --- SEARCH FUNCTION (Instrumented) ---
  void search(const std::vector<float>& query, int* res) {
    static thread_local VisitedList visited;
    if (visited.tags.size() != n_) visited.resize(n_);

    std::vector<long long> local_layer_times(MAX_LEVEL + 1, 0);
    int local_hops = 0;

    int curr_ep = entry_point_;
    const float* q_data = query.data();

    // Transform query: q_trans = (q - min) / scale
    std::vector<float> q_trans(d_);
    for(int i = 0; i < d_; ++i) {
      q_trans[i] = (q_data[i] - min_vals_[i]) / scale_vals_[i];
    }

    float cur_dist = dist_query_sq(q_trans.data(), curr_ep);

    visited.advance();
    visited.visit(curr_ep);

    // --- Phase 1: Upper Layers ---
    for (int l = max_level_; l > 0; l--) {
      auto t_start = std::chrono::high_resolution_clock::now();

      bool changed = true;
      while (changed) {
        changed = false;
        const Node& node = nodes_[curr_ep];
        if (l >= node.link_counts.size()) break;

        int count = node.link_counts[l];
        int offset = get_link_offset(l);
        // [PMR]
        const int* links = node.flat_links.data() + offset;

        for (int i = 0; i < count; ++i) {
          int neighbor = links[i];
          if (i + 1 < count) _mm_prefetch((const char*)(data_sq_ptr_ + links[i + 1] * d_), _MM_HINT_T0);

          float d = dist_query_sq(q_trans.data(), neighbor);
          if (d < cur_dist) {
            cur_dist = d;
            curr_ep = neighbor;
            changed = true;
          }
        }
      }
      auto t_end = std::chrono::high_resolution_clock::now();
      local_layer_times[l] = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();
    }

    float entry_dist = cur_dist;

    // --- Phase 2: Layer 0 ---
    auto t0_start = std::chrono::high_resolution_clock::now();

    using QueueItem = std::pair<float, int>;
    std::priority_queue<QueueItem> top_candidates;
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> candidates;

    top_candidates.push({ cur_dist, curr_ep });
    candidates.push({ cur_dist, curr_ep });
    visited.visit(curr_ep);

    while (!candidates.empty()) {
      auto [c_dist, c_id] = candidates.top();
      candidates.pop();

      if (c_dist > top_candidates.top().first && top_candidates.size() >= ef_search) break;

      const Node& node = nodes_[c_id];
      int size = node.link_counts[0];
      int offset = get_link_offset(0);
      const int* links = node.flat_links.data() + offset;

      for (int i = 0; i < size; ++i) {
        int neighbor_id = links[i];
        if (i + 1 < size) _mm_prefetch((const char*)(data_sq_ptr_ + links[i + 1] * d_), _MM_HINT_T0);

        if (!visited.visit(neighbor_id)) {
          local_hops++;
          float d = dist_query_sq(q_trans.data(), neighbor_id);
          if (top_candidates.size() < ef_search || d < top_candidates.top().first) {
            candidates.push({ d, neighbor_id });
            top_candidates.push({ d, neighbor_id });
            if (top_candidates.size() > ef_search) top_candidates.pop();
          }
        }
      }
    }

    auto t0_end = std::chrono::high_resolution_clock::now();
    local_layer_times[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t0_end - t0_start).count();

    // --- Fill Results ---
    size_t k_idx = 0;
    std::vector<std::pair<float, int>> sorted;
    while (!top_candidates.empty()) {
      sorted.push_back(top_candidates.top());
      top_candidates.pop();
    }
    std::sort(sorted.begin(), sorted.end());

    float final_dist = sorted.empty() ? -1.0f : sorted[0].first;
    float ratio = (final_dist > 1e-9) ? (entry_dist / final_dist) : 1.0f;

    for (const auto& p : sorted) {
      if (k_idx >= 10) break;
      res[k_idx++] = p.second;
    }
    while (k_idx < 10) res[k_idx++] = -1;

    {
      std::lock_guard<std::mutex> lock(stats_mutex_);
      stats_.count++;
      stats_.total_layer0_hops += local_hops;
      stats_.sum_ratios += ratio;
      for (int i = 0; i <= MAX_LEVEL; ++i) {
        stats_.layer_times_ns[i] += local_layer_times[i];
      }
    }
  }

private:
  // --- AVX2 Distance (Transformed Query Float vs Node SQ16) ---
  __attribute__((target("avx2,fma")))
  inline float dist_query_sq(const float* q_trans, int id_node) const {
    const uint16_t* node_ptr = data_sq_ptr_ + id_node * d_;
    const float* scale_sq_ptr = scale_sq_vals_.data();

    __m256 sum = _mm256_setzero_ps();
    int d = d_;
    int i = 0;

    for (; i <= d - 8; i += 8) {
      __m128i v_u16 = _mm_loadu_si128((const __m128i*)(node_ptr + i));
      __m256 v_f32_node = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(v_u16));
      __m256 v_q = _mm256_loadu_ps(q_trans + i);
      __m256 diff = _mm256_sub_ps(v_q, v_f32_node);
      __m256 diff_sq = _mm256_mul_ps(diff, diff);
      __m256 v_scale_sq = _mm256_loadu_ps(scale_sq_ptr + i);
      sum = _mm256_fmadd_ps(diff_sq, v_scale_sq, sum);
    }

    __m128 sum_low = _mm256_castps256_ps128(sum);
    __m128 sum_high = _mm256_extractf128_ps(sum, 1);
    __m128 v_res = _mm_add_ps(sum_low, sum_high);
    v_res = _mm_hadd_ps(v_res, v_res);
    v_res = _mm_hadd_ps(v_res, v_res);
    float res = _mm_cvtss_f32(v_res);

    for (; i < d; ++i) {
      float val = (float)node_ptr[i];
      float diff = q_trans[i] - val;
      res += diff * diff * scale_sq_ptr[i];
    }
    return res;
  }

  // --- AVX2 Distance (Node SQ16 vs Node SQ16) ---
  __attribute__((target("avx2,fma")))
  inline float dist_sq(int id_a, int id_b) const {
    const uint16_t* ptr_a = data_sq_ptr_ + id_a * d_;
    const uint16_t* ptr_b = data_sq_ptr_ + id_b * d_;
    const float* scale_sq_ptr = scale_sq_vals_.data();

    __m256 sum = _mm256_setzero_ps();
    int d = d_;
    int i = 0;

    for (; i <= d - 8; i += 8) {
      __m128i v_u16_a = _mm_loadu_si128((const __m128i*)(ptr_a + i));
      __m128i v_u16_b = _mm_loadu_si128((const __m128i*)(ptr_b + i));

      __m256 v_a = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(v_u16_a));
      __m256 v_b = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(v_u16_b));

      __m256 diff = _mm256_sub_ps(v_a, v_b);
      __m256 diff_sq = _mm256_mul_ps(diff, diff);

      __m256 v_scale_sq = _mm256_loadu_ps(scale_sq_ptr + i);

      sum = _mm256_fmadd_ps(diff_sq, v_scale_sq, sum);
    }

    __m128 sum_low = _mm256_castps256_ps128(sum);
    __m128 sum_high = _mm256_extractf128_ps(sum, 1);
    __m128 v_res = _mm_add_ps(sum_low, sum_high);
    v_res = _mm_hadd_ps(v_res, v_res);
    v_res = _mm_hadd_ps(v_res, v_res);
    float res = _mm_cvtss_f32(v_res);

    for (; i < d; ++i) {
      float d_val = (float)ptr_a[i] - (float)ptr_b[i];
      res += d_val * d_val * scale_sq_ptr[i];
    }
    return res;
  }


  inline int get_link_offset(int level) const {
    return (level == 0) ? 0 : (M0 + (level - 1) * M);
  }

  int get_random_level(std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = -std::log(dist(rng)) * level_mult_;
    return std::min(static_cast<int>(r), MAX_LEVEL);
  }

  std::priority_queue<std::pair<float, int>> search_layer_build(
    const float* query_data, int entry_point, int ef, int level, VisitedList& visited
  ) {
    using QueueItem = std::pair<float, int>;
    std::priority_queue<QueueItem> top_candidates;
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> candidates;

    visited.advance();
    float initial_dist = dist_query_sq(query_data, entry_point);
    top_candidates.push({ initial_dist, entry_point });
    candidates.push({ initial_dist, entry_point });
    visited.visit(entry_point);

    while (!candidates.empty()) {
      auto [curr_dist, curr_id] = candidates.top();
      if (curr_dist > top_candidates.top().first && top_candidates.size() >= ef) break;
      candidates.pop();

      const Node& node = nodes_[curr_id];
      int size = node.link_counts[level];
      int offset = get_link_offset(level);
      // [PMR] Accessing pmr::vector
      const int* links = node.flat_links.data() + offset;

      for (int i = 0; i < size; ++i) {
        int neighbor_id = links[i];
        if (i + 1 < size) _mm_prefetch((const char*)(data_sq_ptr_ + links[i + 1] * d_), _MM_HINT_T0);

        if (!visited.visit(neighbor_id)) {
          float d = dist_query_sq(query_data, neighbor_id);
          if (top_candidates.size() < ef || d < top_candidates.top().first) {
            candidates.push({ d, neighbor_id });
            top_candidates.push({ d, neighbor_id });
            if (top_candidates.size() > ef) top_candidates.pop();
          }
        }
      }
    }
    return top_candidates;
  }

  void get_neighbors_heuristic(
    int src,
    std::vector<std::pair<float, int>>& candidates,
    int level,
    int* output_buffer,
    int& output_count
  ) {
    int max_m = (level == 0) ? M0 : M;
    output_count = 0;
    if (candidates.empty()) return;
    std::sort(candidates.begin(), candidates.end());

    for (const auto& cand : candidates) {
      if (output_count >= max_m) break;
      int cand_id = cand.second;
      float dist_to_src = cand.first;
      bool good = true;
      for (int j = 0; j < output_count; ++j) {
        if (dist_sq(cand_id, output_buffer[j]) < dist_to_src) {
          good = false; break;
        }
      }
      if (good) output_buffer[output_count++] = cand_id;
    }
  }

  void add_connection(int src, int dest, int level) {
    Node& node = nodes_[src];
    std::lock_guard<std::mutex> lock(*node.lock);

    int count = node.link_counts[level];
    int offset = get_link_offset(level);
    // [PMR] Accessing pmr::vector
    int* links_ptr = node.flat_links.data() + offset;

    for (int i = 0; i < count; ++i) if (links_ptr[i] == dest) return;

    int max_m = (level == 0) ? M0 : M;

    if (count < max_m) {
      float dest_dist = dist_sq(src, dest);
      int insert_pos = count;
      for (int i = 0; i < count; ++i) {
        if (dist_sq(src, links_ptr[i]) > dest_dist) {
          insert_pos = i; break;
        }
      }
      for (int j = count; j > insert_pos; --j) links_ptr[j] = links_ptr[j - 1];
      links_ptr[insert_pos] = dest;
      __atomic_store_n(&node.link_counts[level], count + 1, __ATOMIC_RELEASE);
    }
    else {
      // Temporary vectors for heuristic logic do NOT use the monotonic resource
      // because they are short-lived. We let them use the default allocator.
      std::vector<std::pair<float, int>> candidates;
      candidates.reserve(max_m + 1);
      for (int i = 0; i < count; ++i) candidates.push_back({ dist_sq(src, links_ptr[i]), links_ptr[i] });
      candidates.push_back({ dist_sq(src, dest), dest });

      std::vector<int> new_links(max_m);
      int new_count = 0;
      get_neighbors_heuristic(src, candidates, level, new_links.data(), new_count);

      for (int i = 0; i < new_count; ++i) links_ptr[i] = new_links[i];
      if (new_count != count) __atomic_store_n(&node.link_counts[level], new_count, __ATOMIC_RELEASE);
    }
  }
};
