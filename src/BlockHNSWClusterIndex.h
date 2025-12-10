#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <thread>
#include <iostream>
#include <chrono>
#include <atomic>
#include <random>
#include <numeric>
#include <mutex>
#include <deque>

#include "Global.h" 

// Standalone AVX2 Distance function
__attribute__((target("avx2,fma"), always_inline)) 
inline float ClusterDist(const float* a, const float* b, int d) {
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

class ClusterIndex {
public:
  struct BucketInfo {
    int start_idx; 
    int count;
  };

  ClusterIndex() {
    total_scan_time_ns_ = 0;
    total_scans_ = 0;
  }

  ~ClusterIndex() {
    if constexpr (global::kDEBUG) {
      long long total_ns = total_scan_time_ns_.load();
      size_t count = total_scans_.load();
      if (count > 0) {
        double avg_us = (double)total_ns / count / 1000.0;
        std::cout << "\n=== ClusterIndex Profiling ===\n"
          << "Total Scans: " << count << "\n"
          << "Avg Scan Time: " << avg_us << " us\n"
          << "==============================\n";
      }
    }
  }

  void Build(int d, const std::vector<float>& base_data, int target_bucket_size = 64) {
    d_ = d;
    size_t total_n = base_data.size() / d;

    // STRATEGY: 
    // 1. Bisect to find roughly balanced starting points.
    // 2. Refine centroids with K-Means (5 iters) to center them perfectly.
    // 3. Store data using Top-2 replication.
    int SPLIT_LIMIT = target_bucket_size / 2; 
    if (SPLIT_LIMIT < 16) SPLIT_LIMIT = 16;

    if constexpr (global::kDEBUG) {
      std::cout << "[ClusterIndex] Starting Bisecting Init (Limit " << SPLIT_LIMIT << ")...\n";
    }

    // --- PHASE 1: Bisecting K-Means Initialization ---
    struct ClusterChunk { std::vector<int> indices; };
    std::deque<ClusterChunk> queue;

    ClusterChunk root;
    root.indices.resize(total_n);
    std::iota(root.indices.begin(), root.indices.end(), 0);
    queue.push_back(std::move(root));

    std::vector<std::vector<int>> leaves;

    // Lambda for AVX2 split logic
    auto run_split_logic = [&](const std::vector<int>& indices, 
                               std::vector<int>& out_A, 
                               std::vector<int>& out_B) 
                               __attribute__((target("avx2,fma"))) 
    {
        int p1_idx = indices[global::rng() % indices.size()];
        const float* c1 = &base_data[p1_idx * d];
        float max_d = -1.0f;
        int p2_idx = indices[0];

        int samples = std::min((int)indices.size(), 64);
        for(int k=0; k<samples; ++k) {
            int cand = indices[k];
            float dist = ClusterDist(&base_data[cand * d], c1, d);
            if (dist > max_d) { max_d = dist; p2_idx = cand; }
        }
        
        std::vector<float> local_cents(2 * d);
        std::memcpy(&local_cents[0], &base_data[p1_idx * d], d * sizeof(float));
        std::memcpy(&local_cents[d], &base_data[p2_idx * d], d * sizeof(float));

        for(int iter=0; iter<5; ++iter) {
            out_A.clear(); out_B.clear();
            float* cA = &local_cents[0];
            float* cB = &local_cents[d];

            for(int idx : indices) {
                float d1 = ClusterDist(&base_data[idx * d], cA, d);
                float d2 = ClusterDist(&base_data[idx * d], cB, d);
                if (d1 < d2) out_A.push_back(idx); else out_B.push_back(idx);
            }
            
            if (out_A.empty() || out_B.empty()) {
                out_A.clear(); out_B.clear();
                size_t mid = indices.size()/2;
                for(size_t i=0; i<mid; ++i) out_A.push_back(indices[i]);
                for(size_t i=mid; i<indices.size(); ++i) out_B.push_back(indices[i]);
                return;
            }

            if (iter < 4) {
                std::fill(local_cents.begin(), local_cents.end(), 0.0f);
                for(int idx : out_A) {
                    const float* v = &base_data[idx * d];
                    for(int k=0; k<d; ++k) local_cents[k] += v[k];
                }
                for(int idx : out_B) {
                    const float* v = &base_data[idx * d];
                    for(int k=0; k<d; ++k) local_cents[d+k] += v[k];
                }
                float iA = 1.0f/out_A.size(), iB = 1.0f/out_B.size();
                for(int k=0; k<d; ++k) local_cents[k] *= iA;
                for(int k=0; k<d; ++k) local_cents[d+k] *= iB;
            }
        }
    };

    while(!queue.empty()) {
      ClusterChunk chunk = std::move(queue.front());
      queue.pop_front();

      if (chunk.indices.size() <= (size_t)SPLIT_LIMIT) {
        leaves.push_back(std::move(chunk.indices));
        continue;
      }

      std::vector<int> A, B;
      A.reserve(chunk.indices.size()); B.reserve(chunk.indices.size());
      run_split_logic(chunk.indices, A, B);
      queue.push_back({std::move(A)});
      queue.push_back({std::move(B)});
    }

    // Initialize Centroids from Leaves
    num_centroids_ = leaves.size();
    centroids_.resize(num_centroids_ * d);
    
    int n_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    size_t l_chunk = (num_centroids_ + n_threads - 1) / n_threads;

    for(int t=0; t<n_threads; ++t) {
      threads.emplace_back([&, t]() {
        size_t start = t * l_chunk;
        size_t end = std::min(start + l_chunk, num_centroids_);
        for(size_t c=start; c<end; ++c) {
          const auto& idxs = leaves[c];
          float* center = &centroids_[c * d];
          std::fill(center, center + d, 0.0f);
          for(int idx : idxs) {
            const float* v = &base_data[idx * d];
            for(int k=0; k<d; ++k) center[k] += v[k];
          }
          if (!idxs.empty()) {
            float inv = 1.0f / idxs.size();
            for(int k=0; k<d; ++k) center[k] *= inv;
          }
        }
      });
    }
    for(auto& t : threads) t.join();
    threads.clear();

    // --- PHASE 1.5: Centroid Refinement (Standard K-Means) ---
    // Bisecting puts centroids in "ok" spots. We refine them to "perfect" spots.
    // This dramatically improves the spherical geometry for pruning.
    
    size_t train_size = std::min(total_n, (size_t)150000);
    std::vector<int> train_indices(total_n);
    std::iota(train_indices.begin(), train_indices.end(), 0);
    std::shuffle(train_indices.begin(), train_indices.end(), global::rng);
    
    if constexpr (global::kDEBUG) {
        std::cout << "[ClusterIndex] Refining Centroids (5 iters on " << train_size << " points)...\n";
    }

    std::vector<int> assignments_refine(train_size);
    size_t r_chunk = (train_size + n_threads - 1) / n_threads;

    for(int iter=0; iter<5; ++iter) {
        // Assign
        for(int t=0; t<n_threads; ++t) {
            threads.emplace_back([&, t]() __attribute__((target("avx2,fma"))) {
                size_t start = t * r_chunk;
                size_t end = std::min(start + r_chunk, train_size);
                for(size_t i=start; i<end; ++i) {
                    int real_idx = train_indices[i];
                    const float* vec = &base_data[real_idx * d];
                    float min_d = std::numeric_limits<float>::max();
                    int best_c = 0;
                    for(int c=0; c<num_centroids_; ++c) {
                        float dist = ClusterDist(vec, &centroids_[c*d], d);
                        if(dist < min_d) { min_d = dist; best_c = c; }
                    }
                    assignments_refine[i] = best_c;
                }
            });
        }
        for(auto& t : threads) t.join();
        threads.clear();

        // Update
        std::vector<int> counts(num_centroids_, 0);
        std::fill(centroids_.begin(), centroids_.end(), 0.0f);
        for(size_t i=0; i<train_size; ++i) {
            int c = assignments_refine[i];
            int idx = train_indices[i];
            counts[c]++;
            const float* v = &base_data[idx * d];
            for(int k=0; k<d; ++k) centroids_[c*d+k] += v[k];
        }
        for(int c=0; c<num_centroids_; ++c) {
            if(counts[c] > 0) {
                float inv = 1.0f/counts[c];
                for(int k=0; k<d; ++k) centroids_[c*d+k] *= inv;
            }
        }
    }

    // --- PHASE 2: Top-2 Multiple Assignment ---
    if constexpr (global::kDEBUG) {
      std::cout << "[ClusterIndex] Running Top-2 Assignment...\n";
    }

    int repl_factor = 2;
    std::vector<std::vector<int>> assignments(total_n);
    size_t p_chunk = (total_n + n_threads - 1) / n_threads;

    for(int t=0; t<n_threads; ++t) {
      threads.emplace_back([&, t]() __attribute__((target("avx2,fma"))) {
        size_t start = t * p_chunk;
        size_t end = std::min(start + p_chunk, total_n);
        std::vector<std::pair<float, int>> local_top; 
        local_top.reserve(num_centroids_);

        for(size_t i=start; i<end; ++i) {
          const float* vec = &base_data[i * d];
          local_top.clear();
          
          for(int c=0; c<num_centroids_; ++c) {
            float dist = ClusterDist(vec, &centroids_[c*d], d);
            local_top.push_back({dist, c});
          }
          std::partial_sort(local_top.begin(), local_top.begin() + repl_factor, local_top.end());

          assignments[i].reserve(repl_factor);
          for(int k=0; k<repl_factor; ++k) assignments[i].push_back(local_top[k].second);
        }
      });
    }
    for(auto& t : threads) t.join();
    threads.clear();

    // --- PHASE 3: Reorder ---
    buckets_.resize(num_centroids_);
    std::vector<int> bucket_counts(num_centroids_, 0);
    for(size_t i=0; i<total_n; ++i) {
      for(int c : assignments[i]) bucket_counts[c]++;
    }

    int current_offset = 0;
    int max_bucket_sz = 0;
    for(int c=0; c<num_centroids_; ++c) {
      buckets_[c].start_idx = current_offset;
      buckets_[c].count = 0; 
      current_offset += bucket_counts[c];
      if (bucket_counts[c] > max_bucket_sz) max_bucket_sz = bucket_counts[c];
    }

    reordered_data_.resize(total_n * repl_factor * d);
    original_indices_.resize(total_n * repl_factor);

    for(size_t i=0; i<total_n; ++i) {
      for(int c : assignments[i]) {
        int idx = buckets_[c].start_idx + buckets_[c].count;
        std::memcpy(&reordered_data_[idx*d], &base_data[i*d], d*sizeof(float));
        original_indices_[idx] = (int)i;
        buckets_[c].count++;
      }
    }

    if constexpr (global::kDEBUG) {
      std::cout << "[ClusterIndex] Build Complete. Max Bucket: " << max_bucket_sz 
        << " (Avg: " << (total_n * repl_factor / num_centroids_) << ")\n";
    }
  }

  __attribute__((target("avx2,fma"), noinline))
  void Scan(const std::vector<float>& query, const int* bucket_indices, int num_buckets, int* res) const {
    auto t_start = std::chrono::high_resolution_clock::now();

    const float* q_ptr = query.data();
    const float* data_start = reordered_data_.data();

    std::vector<std::pair<float, int>> active_buckets;
    active_buckets.reserve(num_buckets);
    float min_centroid_dist = std::numeric_limits<float>::max();

    for(int i=0; i<num_buckets; ++i) {
      int c_id = bucket_indices[i];
      if (c_id < 0 || c_id >= num_centroids_) continue;

      float d = ClusterDist(q_ptr, &centroids_[c_id * d_], d_);
      active_buckets.push_back({d, c_id});
      if (d < min_centroid_dist) min_centroid_dist = d;
    }
    std::sort(active_buckets.begin(), active_buckets.end());

    float pruning_limit = min_centroid_dist * 2.0f; 

    std::vector<int> filtered_buckets;
    filtered_buckets.reserve(active_buckets.size());
    for(const auto& p : active_buckets) {
      if (p.first <= pruning_limit) filtered_buckets.push_back(p.second);
      else break; 
    }

    // Dynamic Scheduling
    std::atomic<int> next_task_idx(0);
    int n_scan_tasks = filtered_buckets.size();
    int n_threads = std::min(n_scan_tasks, (int)std::thread::hardware_concurrency());
    if (n_threads < 1) n_threads = 1;

    std::vector<std::vector<std::pair<float, int>>> thread_candidates(n_threads);
    for(auto& v : thread_candidates) v.reserve(n_scan_tasks * 64 / n_threads);

    std::vector<std::thread> threads;
    for(int t=0; t<n_threads; ++t) {
      threads.emplace_back([&, t]() __attribute__((target("avx2,fma"))) {
        auto& local_cands = thread_candidates[t];
        while(true) {
          int i = next_task_idx.fetch_add(1, std::memory_order_relaxed);
          if (i >= n_scan_tasks) break;

          int c_id = filtered_buckets[i];
          const BucketInfo& bucket = buckets_[c_id];
          const float* vec_ptr = data_start + (bucket.start_idx * d_);
          int cnt = bucket.count; 

          for(int j=0; j<cnt; ++j) {
            if (j + 1 < cnt) _mm_prefetch(reinterpret_cast<const char*>(vec_ptr + d_), _MM_HINT_T0);
            float d = ClusterDist(q_ptr, vec_ptr, d_);
            local_cands.push_back({d, original_indices_[bucket.start_idx + j]});
            vec_ptr += d_;
          }
        }
      });
    }
    for(auto& t : threads) t.join();

    // Merge
    std::vector<std::pair<float, int>> all_candidates;
    size_t total_size = 0;
    for(const auto& v : thread_candidates) total_size += v.size();
    all_candidates.reserve(total_size);

    for(const auto& v : thread_candidates) {
      all_candidates.insert(all_candidates.end(), v.begin(), v.end());
    }

    if constexpr (global::kDEBUG) {
      auto t_end = std::chrono::high_resolution_clock::now();
      total_scan_time_ns_.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count(), std::memory_order_relaxed);
      total_scans_.fetch_add(1, std::memory_order_relaxed);
    }

    size_t k = 10;
    if (all_candidates.size() > k * 2) {
      std::partial_sort(all_candidates.begin(), all_candidates.begin() + k * 2, all_candidates.end());
      all_candidates.resize(k * 2);
    } else {
      std::sort(all_candidates.begin(), all_candidates.end());
    }

    // Deduplicate
    int out = 0;
    for (size_t i = 0; i < all_candidates.size() && out < (int)k; ++i) {
        if (out > 0 && all_candidates[i].second == res[out-1]) continue;
        res[out++] = all_candidates[i].second;
    }
    while (out < (int)k) res[out++] = -1;
  }

  const std::vector<float>& GetCentroids() const { return centroids_; }

private:
  int d_;
  size_t num_centroids_;
  std::vector<float> centroids_; 

  std::vector<BucketInfo> buckets_;
  std::vector<float> reordered_data_;
  std::vector<int> original_indices_; 

  mutable std::atomic<long long> total_scan_time_ns_;
  mutable std::atomic<size_t> total_scans_;
};
