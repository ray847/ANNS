#pragma once

#include <cstdlib>
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
#include <immintrin.h>

#include "Global.h"
#include "SolutionConcept.h"

class Solution {
public:
  // Destructor to print stats
  ~Solution() {
    if constexpr (global::kDEBUG) {
      size_t s = total_searches_.load();
      if (s > 0) {
        size_t d = total_dist_calcs_.load();
        std::cout << "[DEBUG] Average distance calculations per search: " 
                  << (double)d / s << " (" << s << " searches)" << std::endl;
      }
    }
  }

  void build(int d, const std::vector<float>& base);
  void search(const std::vector<float>& query, int* res);

private:
  static constexpr int M = 64;               
  static constexpr int M0 = 128;              
  static constexpr int ef_construction = 600; 
  static constexpr int ef_search = 150;       
  static constexpr int MAX_LEVEL = 16;       

  // --- Statistics Counters ---
  // Using inline static to define them in header (C++17 feature)
  inline static std::atomic<size_t> total_dist_calcs_{0};
  inline static std::atomic<size_t> total_searches_{0};
  // Thread local to avoid atomic overhead on every distance calc
  inline static thread_local size_t local_dist_count_{0};

  int d_ = 0;
  size_t n_ = 0;

  const float* data_ptr_ = nullptr; 
  std::vector<float> data_storage_;

  struct Node {
    int level;
    std::vector<int> flat_links; 
    std::vector<int> link_counts;
    std::unique_ptr<std::mutex> lock;
  };

  std::vector<Node> nodes_;

  int entry_point_ = -1;
  int max_level_ = -1;
  double level_mult_;
  std::mutex global_lock_;

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

  // --- AVX2 Distance ---
  __attribute__((target("avx2,fma")))
  inline float dist_func_sq(const float* a, const float* b, int d) const {
    if constexpr (global::kDEBUG) {
      local_dist_count_++;
    }

    __m256 sum = _mm256_setzero_ps();
    const float* end_safe = a + (d & ~7);
    while (a < end_safe) {
      __m256 v_a = _mm256_loadu_ps(a);
      __m256 v_b = _mm256_loadu_ps(b);
      __m256 diff = _mm256_sub_ps(v_a, v_b);
      sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
      a += 8; b += 8;
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

  inline float dist_sq(int id_a, int id_b) const {
    return dist_func_sq(data_ptr_ + id_a * d_, data_ptr_ + id_b * d_, d_);
  }

  inline float dist_query_sq(const float* query, int id_node) const {
    return dist_func_sq(query, data_ptr_ + id_node * d_, d_);
  }

  inline int get_link_offset(int level) const {
    return (level == 0) ? 0 : (M0 + (level - 1) * M);
  }

  int get_random_level(std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = -std::log(dist(rng)) * level_mult_;
    return std::min(static_cast<int>(r), MAX_LEVEL);
  }

  // --- Build Phase Search (Standard) ---
  std::priority_queue<std::pair<float, int>> search_layer(
    const float* query_data, int entry_point, int ef, int level, VisitedList& visited
  ) {
    using QueueItem = std::pair<float, int>;
    std::priority_queue<QueueItem> top_candidates;
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> candidates;

    visited.advance();
    float initial_dist = dist_query_sq(query_data, entry_point);
    top_candidates.push({initial_dist, entry_point});
    candidates.push({initial_dist, entry_point});
    visited.visit(entry_point);

    while (!candidates.empty()) {
      auto [curr_dist, curr_id] = candidates.top();
      if (curr_dist > top_candidates.top().first && top_candidates.size() >= ef) break;
      candidates.pop();

      const Node& node = nodes_[curr_id];
      int size = node.link_counts[level]; 
      int offset = get_link_offset(level);
      const int* links = node.flat_links.data() + offset;

      for (int i = 0; i < size; ++i) {
        int neighbor_id = links[i];
        if (i + 1 < size) _mm_prefetch((const char*)(data_ptr_ + links[i+1] * d_), _MM_HINT_T0);

        if (!visited.visit(neighbor_id)) {
          float d = dist_query_sq(query_data, neighbor_id);
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

  // --- Combined Search Helper (WITH MOVE-TO-FRONT) ---
  std::priority_queue<std::pair<float, int>> search_combined_layers(
    const float* query_data, int entry_point, int ef, VisitedList& visited
  ) {
    using QueueItem = std::pair<float, int>;
    std::priority_queue<QueueItem> top_candidates;
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> candidates;

    visited.advance();
    float initial_dist = dist_query_sq(query_data, entry_point);

    top_candidates.push({initial_dist, entry_point});
    candidates.push({initial_dist, entry_point});
    visited.visit(entry_point);

    while (!candidates.empty()) {
      auto [curr_dist, curr_id] = candidates.top();
      if (curr_dist > top_candidates.top().first && top_candidates.size() >= ef) break;
      candidates.pop();

      const Node& node = nodes_[curr_id];
      int node_max_level = node.level;

      for (int l = 0; l <= node_max_level; ++l) {
        int size = node.link_counts[l];
        int offset = get_link_offset(l);

        // We need non-const access to perform Move-To-Front
        int* links = const_cast<int*>(node.flat_links.data() + offset);

        for (int i = 0; i < size; ++i) {
          int neighbor_id = links[i];
          if (i + 1 < size) _mm_prefetch((const char*)(data_ptr_ + links[i+1] * d_), _MM_HINT_T0);

          if (!visited.visit(neighbor_id)) {
            float d = dist_query_sq(query_data, neighbor_id);

            // Check if this neighbor is "good enough" to be in top candidates
            if (top_candidates.size() < ef || d < top_candidates.top().first) {
              candidates.push({d, neighbor_id});
              top_candidates.push({d, neighbor_id});
              if (top_candidates.size() > ef) top_candidates.pop();

              if (i > 0) {
                std::swap(links[0], links[i]);
              }
            }
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

  // --- Thread-Safe Connection (WITH STRICT SORTING) ---
  void add_connection(int src, int dest, int level) {
    Node& node = nodes_[src];
    std::lock_guard<std::mutex> lock(*node.lock);

    int count = node.link_counts[level];
    int offset = get_link_offset(level);
    int* links_ptr = node.flat_links.data() + offset;

    for (int i = 0; i < count; ++i) if (links_ptr[i] == dest) return;

    int max_m = (level == 0) ? M0 : M;

    if (count < max_m) {
      float dest_dist = dist_sq(src, dest);
      int insert_pos = count;

      for(int i = 0; i < count; ++i) {
        float d = dist_sq(src, links_ptr[i]);
        if (d > dest_dist) {
          insert_pos = i;
          break;
        }
      }

      for (int j = count; j > insert_pos; --j) {
        links_ptr[j] = links_ptr[j-1];
      }

      links_ptr[insert_pos] = dest;
      __atomic_store_n(&node.link_counts[level], count + 1, __ATOMIC_RELEASE);

    } else {
      std::vector<std::pair<float, int>> candidates;
      candidates.reserve(max_m + 1);
      for (int i = 0; i < count; ++i) candidates.push_back({dist_sq(src, links_ptr[i]), links_ptr[i]});
      candidates.push_back({dist_sq(src, dest), dest});

      std::vector<int> new_links(max_m);
      int new_count = 0;
      get_neighbors_heuristic(src, candidates, level, new_links.data(), new_count);

      for(int i=0; i<new_count; ++i) links_ptr[i] = new_links[i];
      if (new_count != count) __atomic_store_n(&node.link_counts[level], new_count, __ATOMIC_RELEASE);
    }
  }
};

static_assert(IsSolution<Solution>);

inline void Solution::build(int d, const std::vector<float>& base) {
  d_ = d;
  data_storage_ = base;
  data_ptr_ = data_storage_.data();
  n_ = base.size() / d_;
  nodes_.resize(n_);
  level_mult_ = 1.0 / std::log(1.0 * M);

  std::mt19937 rng_init(42);
  for (size_t i = 0; i < n_; ++i) {
    int level = get_random_level(rng_init);
    nodes_[i].level = level;
    nodes_[i].link_counts.resize(level + 1, 0);
    nodes_[i].lock = std::make_unique<std::mutex>(); 
    size_t total_links = M0;
    if (level > 0) total_links += (size_t)level * M;
    nodes_[i].flat_links.resize(total_links);
  }

  entry_point_ = 0;
  max_level_ = nodes_[0].level;

  std::atomic<size_t> atomic_idx{1};       
  std::atomic<size_t> progress_counter{0}; 
  size_t total_work = n_ - 1;

  if constexpr (global::kDEBUG) {
    std::cout << "Building HNSW Optimized (d=" << d_ << ", M=" << M << ") for " << n_ << " vectors..." << std::endl;
  }

  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) num_threads = 4;

  auto worker_func = [&](int thread_id) {
    VisitedList visited;
    visited.resize(n_);
    std::mt19937 rng(42 + thread_id); 

    while (true) {
      size_t curr_obj = atomic_idx.fetch_add(1, std::memory_order_relaxed);
      if (curr_obj >= n_) break;

      int curr_level = nodes_[curr_obj].level;
      int curr_ep = entry_point_;
      int curr_max_level = max_level_;
      const float* curr_vec = data_ptr_ + curr_obj * d_;

      for (int l = curr_max_level; l > curr_level; l--) {
        bool changed = true;
        while (changed) {
          changed = false;
          float dist_ep = dist_query_sq(curr_vec, curr_ep);
          const Node& node_ep = nodes_[curr_ep];
          if (l >= node_ep.link_counts.size()) break;

          int offset = get_link_offset(l);
          int count = node_ep.link_counts[l];
          const int* links = node_ep.flat_links.data() + offset;

          for(int j=0; j<count; ++j) {
            int neighbor = links[j];
            float d = dist_query_sq(curr_vec, neighbor);
            if(d < dist_ep) {
              curr_ep = neighbor;
              dist_ep = d;
              changed = true;
            }
          }
        }
      }

      for (int l = std::min(curr_level, curr_max_level); l >= 0; l--) {
        auto top_candidates = search_layer(curr_vec, curr_ep, ef_construction, l, visited);
        std::vector<std::pair<float, int>> potential;
        potential.reserve(ef_construction + 1);
        while(!top_candidates.empty()) {
          potential.push_back(top_candidates.top());
          top_candidates.pop();
        }

        int offset = get_link_offset(l);
        int* link_dst = nodes_[curr_obj].flat_links.data() + offset;
        int count = 0;
        get_neighbors_heuristic(curr_obj, potential, l, link_dst, count);
        nodes_[curr_obj].link_counts[l] = count;

        for(int j=0; j<count; ++j) add_connection(link_dst[j], curr_obj, l);
        if (!potential.empty()) curr_ep = potential[0].second;
      }

      if (curr_level > max_level_) {
        std::lock_guard<std::mutex> lock(global_lock_);
        if (curr_level > max_level_) {
          max_level_ = curr_level;
          entry_point_ = curr_obj;
        }
      }
      size_t num_done = progress_counter.fetch_add(1, std::memory_order_relaxed) + 1;
      if constexpr (global::kDEBUG) {
        if (total_work > 100 && num_done % (total_work / 100) == 0) {
          // Use global_lock_ to prevent threads from writing over each other's output
          std::lock_guard<std::mutex> lock(global_lock_);
          int percent = (num_done * 100) / total_work;
          std::cout << "Building: " << percent << "% \r" << std::flush;
        }
      }
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for(unsigned int i = 0; i < num_threads; ++i) {
    threads.emplace_back(worker_func, i);
  }
  for(auto& t : threads) {
    t.join();
  }
  if constexpr (global::kDEBUG) std::cout << "Build: 100% - Done.\n";
}

inline void Solution::search(const std::vector<float>& query, int* res) {
  // Reset thread-local counter at start of search to ignore build-time calls
  if constexpr (global::kDEBUG) {
    local_dist_count_ = 0;
  }

  static thread_local VisitedList visited;
  if (visited.tags.size() != n_) visited.resize(n_);

  int curr_ep = entry_point_;
  const float* q_data = query.data();

  auto top_candidates = search_combined_layers(q_data, curr_ep, ef_search, visited);

  size_t k = 0;
  std::vector<std::pair<float, int>> sorted;
  sorted.reserve(top_candidates.size());

  while(!top_candidates.empty()) {
    sorted.push_back(top_candidates.top());
    top_candidates.pop();
  }
  std::sort(sorted.begin(), sorted.end());

  for (const auto& p : sorted) {
    if (k >= 10) break;
    res[k++] = p.second;
  }
  while (k < 10) res[k++] = -1;

  // Aggregate stats at end of search
  if constexpr (global::kDEBUG) {
    total_dist_calcs_ += local_dist_count_;
    total_searches_++;
  }
}
