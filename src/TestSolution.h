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
#include <optional>
#include <immintrin.h>
#include <type_traits>
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <numeric>

#include "Eigen/Dense"
#include "Global.h"

namespace FinalSolution {

namespace Sift {
  
// --- Distance ---
template <int kDim, bool kEnableSIMD = false, typename = void>
struct Distance {
  static_assert(kDim > 0, "Dimension must be positive.");
  inline static float L2Sq(const float* vector_a, const float* vector_b) {
    float result = 0.0f;
    for (int i = 0; i < kDim; ++i) {
      const float diff = vector_a[i] - vector_b[i];
      result += diff * diff;
    }
    return result;
  }
};

template <>
struct Distance<128, true> {
  __attribute__((target("avx2,fma"))) inline static float L2Sq(
      const float* vector_a, const float* vector_b) {
    __m256 sum_256 = _mm256_setzero_ps();
    for (int i = 0; i < 16; ++i) {
      __m256 vector_a_256 = _mm256_loadu_ps(vector_a + i * 8);
      __m256 vector_b_256 = _mm256_loadu_ps(vector_b + i * 8);
      __m256 diff_256 = _mm256_sub_ps(vector_a_256, vector_b_256);
      sum_256 = _mm256_fmadd_ps(diff_256, diff_256, sum_256);
    }
    __m128 sum_128_low = _mm256_castps256_ps128(sum_256);
    __m128 sum_128_high = _mm256_extractf128_ps(sum_256, 1);
    __m128 result_128 = _mm_add_ps(sum_128_low, sum_128_high);
    result_128 = _mm_hadd_ps(result_128, result_128);
    result_128 = _mm_hadd_ps(result_128, result_128);
    return _mm_cvtss_f32(result_128);
  }
};

// --- Solution ---
class Solution {
 public:
  static constexpr int kDim = 128;
  static constexpr int kM = 32;
  static constexpr int kM0 = 64;
  static constexpr int kEfConstruction = 500;
  static constexpr int kPatience = 100;
  static constexpr int kMaxEfSearch = 500;

  Solution();
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

  using Dist = Distance<128, true>;

  const double level_mult_;
  int max_level_ = -1;
  int entry_point_ = -1;
  std::vector<float> data_storage_;
  const float* data_ptr_ = nullptr;
  size_t num_points_ = 0;
  std::vector<Node> nodes_;
  std::vector<int> flat_graph_;
  std::vector<int> link_counts_;
  
  int GetRandomLevel_(std::mt19937& rng);
  void Insert_(int node_id, 
               std::vector<std::vector<std::vector<int>>>& temp_graph,
               std::vector<std::mutex>& locks);
  std::priority_queue<std::pair<float, int>> SearchLayerFP_(
      const float* query, int entry_point, int ef, int level,
      const std::vector<std::vector<std::vector<int>>>& temp_graph,
      std::vector<std::mutex>& locks, VisitedList& visited);
  void SelectNeighbors_(
      std::priority_queue<std::pair<float, int>>& candidates,
      std::vector<int>& target_list);
  void FlattenGraph_(
      const std::vector<std::vector<std::vector<int>>>& temp_graph);
  void SearchFP_(const std::vector<float>& query, int k, int* result_indices);
};

} // namespace Sift

namespace Glove {

template <int kDim, bool kEnableSIMD = false, typename = void>
struct Distance {
  static_assert(kDim > 0, "Dimension must be positive.");

  inline static float L2Sq(const float* vector_a, const float* vector_b) {
    float result = 0.0f;
    for (int i = 0; i < kDim; ++i) {
      const float diff = vector_a[i] - vector_b[i];
      result += diff * diff;
    }
    return result;
  }
};

template <>
struct Distance<100, true> {
  __attribute__((target("avx2,fma"))) inline static float L2Sq(
      const float* vector_a, const float* vector_b) {
    __m256 sum_256 = _mm256_setzero_ps();

    for (int i = 0; i < 12; ++i) {
      __m256 vector_a_256 = _mm256_loadu_ps(vector_a + i * 8);
      __m256 vector_b_256 = _mm256_loadu_ps(vector_b + i * 8);
      __m256 diff_256 = _mm256_sub_ps(vector_a_256, vector_b_256);
      sum_256 = _mm256_fmadd_ps(diff_256, diff_256, sum_256);
    }

    __m128 sum_128_low = _mm256_castps256_ps128(sum_256);
    __m128 sum_128_high = _mm256_extractf128_ps(sum_256, 1);
    __m128 result_128 = _mm_add_ps(sum_128_low, sum_128_high);
    result_128 = _mm_hadd_ps(result_128, result_128);
    result_128 = _mm_hadd_ps(result_128, result_128);
    float result = _mm_cvtss_f32(result_128);

    for (int i = 96; i < 100; ++i) {
      const float diff = vector_a[i] - vector_b[i];
      result += diff * diff;
    }
    return result;
  }
};

class KMeans {
 public:
  KMeans(int num_clusters, int num_iterations);
  std::vector<float> Train(const float* data, int num_points, int dim,
                           int stride);
 private:
  void InitializeCentroids_(const float* data, int num_points, int dim,
                            int stride, std::vector<float>* centroids);
  float L2SqSlice_(const float* vector_a, const float* vector_b, int dim);
  int num_clusters_;
  int num_iterations_;
};

class OptimizedProductQuantizer {
 public:
  OptimizedProductQuantizer();
  void Train(const float* data, int num_points);
  std::vector<uint8_t> Encode(const float* vector) const;
  std::vector<float> BuildDistanceTable(const float* query) const;
  float GetDistanceFromTable(const std::vector<float>& table,
                             const uint8_t* code) const;
 private:
  void TrainCodebooks(const float* data, int num_points);
  float L2SqSlice_(const float* vector_a, const float* vector_b, int dim) const;

  Eigen::MatrixXf R_;
  std::vector<float> codebooks_;
};

class Solution {
 public:
  static constexpr int kDim = 100;
  static constexpr int kM = 48;
  static constexpr int kM0 = 96;
  static constexpr int kEfConstruction = 800;
  static constexpr int kPatience = 250; 
  static constexpr int kMaxEfSearch = 1600;
  // OPQ
  static constexpr int kPQSubquantizers = 20; 
  static constexpr size_t kOPQTrainSampleSize = 25000;

  Solution();

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

  using Dist = Distance<100, true>;
  using PQ = OptimizedProductQuantizer;

  const double level_mult_;
  int max_level_ = -1;
  int entry_point_ = -1;

  std::vector<float> data_storage_;
  const float* data_ptr_ = nullptr;
  size_t num_points_ = 0;

  std::vector<Node> nodes_;
  std::vector<int> flat_graph_;
  std::vector<int> link_counts_;
  
  std::optional<PQ> pq_;
  std::vector<uint8_t> pq_codes_;

  int GetRandomLevel_(std::mt19937& rng);
  void Insert_(int node_id, 
               std::vector<std::vector<std::vector<int>>>& temp_graph,
               std::vector<std::mutex>& locks);
  std::priority_queue<std::pair<float, int>> SearchLayerFP_(
      const float* query, int entry_point, int ef, int level,
      const std::vector<std::vector<std::vector<int>>>& temp_graph,
      std::vector<std::mutex>& locks, VisitedList& visited);
  void SelectNeighbors_(
      std::priority_queue<std::pair<float, int>>& candidates,
      std::vector<int>& target_list);
  void FlattenGraph_(
      const std::vector<std::vector<std::vector<int>>>& temp_graph);
  
  void SearchPQ_(const std::vector<float>& query, int k, int* result_indices);
};

} // namespace Glove

} // namespace FinalSolution
