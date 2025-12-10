#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <atomic>
#include <chrono>

#include "Global.h"

// Reuse your AVX2 distance function
__attribute__((target("avx2,fma"), always_inline)) 
inline float SolverDist(const float* a, const float* b, int d) {
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

class FlatSolver {
public:
  FlatSolver() {
    total_time_ns_ = 0;
    total_searches_ = 0;
  }

  ~FlatSolver() {
    if constexpr (global::kDEBUG) {
      long long total_ns = total_time_ns_.load();
      size_t count = total_searches_.load();
      if (count > 0) {
        std::cout << "\n=== FlatSolver (Stage 1) Stats ===\n"
          << "Avg Time: " << (double)total_ns / count / 1000.0 << " us\n"
          << "==================================\n";
      }
    }
  }

  void build(int d, const std::vector<float>& base) {
    d_ = d;
    data_ = base; // Keep a contiguous copy
    n_ = base.size() / d_;
    if constexpr (global::kDEBUG) {
      std::cout << "[FlatSolver] Indexed " << n_ << " centroids (Linear Scan).\n";
    }
  }

  __attribute__((target("avx2,fma")))
  void search(const std::vector<float>& query, int* res, int k_results) {
    auto t_start = std::chrono::high_resolution_clock::now();

    // 1. Calculate ALL distances
    // Since n_ is small (~15k), we can do this on the stack or a recycled vector.
    // Using a thread_local vector to avoid allocation overhead.
    static thread_local std::vector<std::pair<float, int>> candidates;
    if (candidates.size() != n_) candidates.resize(n_);

    const float* q_ptr = query.data();
    const float* d_ptr = data_.data();

    // Unrolling helps AVX pipeline
    for (size_t i = 0; i < n_; ++i) {
      candidates[i].first = SolverDist(q_ptr, d_ptr + i * d_, d_);
      candidates[i].second = i;
    }

    // 2. Partial Sort to get top K
    // Using partial_sort is usually faster than nth_element + sort 
    // when K is small relative to N (256 vs 15000).
    if ((size_t)k_results < n_) {
      std::partial_sort(candidates.begin(), candidates.begin() + k_results, candidates.end());
    } else {
      std::sort(candidates.begin(), candidates.end());
    }

    // 3. Write results
    for(int i=0; i<k_results; ++i) {
      res[i] = candidates[i].second;
    }

    if constexpr (global::kDEBUG) {
      auto t_end = std::chrono::high_resolution_clock::now();
      total_time_ns_.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count(), std::memory_order_relaxed);
      total_searches_.fetch_add(1, std::memory_order_relaxed);
    }
  }

private:
  int d_;
  size_t n_;
  std::vector<float> data_;

  std::atomic<long long> total_time_ns_;
  std::atomic<size_t> total_searches_;
};
