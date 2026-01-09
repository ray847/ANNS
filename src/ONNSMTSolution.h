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
#include <thread>
#include <mutex>
#include <atomic>
#include <numeric>
#include <memory>

#include "Global.h"
#include "SolutionConcept.h"

// =================================================================================
// SIMD Distance
// =================================================================================
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

// =================================================================================
// Parallel Utilities
// =================================================================================
template<typename Func>
void ParallelFor(int start, int end, Func func) {
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    std::vector<std::thread> threads;
    std::atomic<int> idx(start);
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            while (true) {
                int i = idx.fetch_add(1, std::memory_order_relaxed);
                if (i >= end) break;
                func(i, t); // Pass thread index for thread-local storage
            }
        });
    }
    for (auto& t : threads) t.join();
}

// =================================================================================
// Minimal Matrix Lib
// =================================================================================
struct Matrix {
  int rows, cols;
  std::vector<float> data;

  Matrix() : rows(0), cols(0) {}
  Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0f) {}

  float& operator()(int r, int c) { return data[r * cols + c]; }
  const float& operator()(int r, int c) const { return data[r * cols + c]; }

  static Matrix Identity(int d) {
    Matrix m(d, d);
    for (int i = 0; i < d; ++i) m(i, i) = 1.0f;
    return m;
  }

  Matrix transpose() const {
    Matrix res(cols, rows);
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < cols; ++c)
        res(c, r) = (*this)(r, c);
    return res;
  }
};

inline Matrix MatMul(const Matrix& A, const Matrix& B) {
  Matrix C(A.rows, B.cols);
  for (int i = 0; i < A.rows; ++i) {
    for (int k = 0; k < A.cols; ++k) {
      float val = A(i, k);
      if (val == 0.0f) continue;
      for (int j = 0; j < B.cols; ++j) {
        C(i, j) += val * B(k, j);
      }
    }
  }
  return C;
}

inline void ComputeSVD(const Matrix& A, Matrix& U, Matrix& V) {
  int m = A.rows;
  int n = A.cols;
  U = A;
  V = Matrix::Identity(n);
  const int max_iters = 80;
  
  for (int iter = 0; iter < max_iters; ++iter) {
    float max_error = 0.0f;
    for (int i = 0; i < n - 1; ++i) {
      for (int j = i + 1; j < n; ++j) {
        float alpha = 0.0f, beta = 0.0f, gamma = 0.0f;
        for (int k = 0; k < m; ++k) {
          float u_ki = U(k, i);
          float u_kj = U(k, j);
          alpha += u_ki * u_ki;
          beta  += u_kj * u_kj;
          gamma += u_ki * u_kj;
        }
        max_error = std::max(max_error, std::abs(gamma) / std::sqrt(alpha * beta + 1e-20f));
        if (std::abs(gamma) < 1e-15f) continue;

        float zeta = (beta - alpha) / (2.0f * gamma);
        float t = std::copysign(1.0f / (std::abs(zeta) + std::sqrt(1.0f + zeta * zeta)), zeta);
        float c = 1.0f / std::sqrt(1.0f + t * t);
        float s = c * t;

        for (int k = 0; k < m; ++k) {
          float t1 = U(k, i);
          float t2 = U(k, j);
          U(k, i) = c * t1 - s * t2;
          U(k, j) = s * t1 + c * t2;
        }
        for (int k = 0; k < n; ++k) {
          float t1 = V(k, i);
          float t2 = V(k, j);
          V(k, i) = c * t1 - s * t2;
          V(k, j) = s * t1 + c * t2;
        }
      }
    }
    if (max_error < 1e-5f) break;
  }
  for (int i = 0; i < n; ++i) {
    float norm = 0.0f;
    for (int k = 0; k < m; ++k) norm += U(k, i) * U(k, i);
    norm = std::sqrt(norm);
    if (norm > 1e-10f) {
      float inv_norm = 1.0f / norm;
      for (int k = 0; k < m; ++k) U(k, i) *= inv_norm;
    }
  }
}

// =================================================================================
// Simple K-Means
// =================================================================================
class SimpleKMeans {
 public:
  SimpleKMeans(int k, int max_iters) : k_(k), max_iters_(max_iters) {}

  std::vector<float> Train(const float* data, int n, int dim) {
    std::vector<float> centroids(k_ * dim);
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), global::rng);

    for (int i = 0; i < k_; ++i) {
      std::copy_n(data + indices[i] * dim, dim, centroids.begin() + i * dim);
    }

    std::vector<int> assignments(n);
    std::vector<int> counts(k_);
    std::vector<float> new_centroids(k_ * dim);

    for (int iter = 0; iter < max_iters_; ++iter) {
      std::fill(counts.begin(), counts.end(), 0);
      std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);

      for (int i = 0; i < n; ++i) {
        const float* vec = data + i * dim;
        int best_c = 0;
        float min_dist = std::numeric_limits<float>::max();

        for (int c = 0; c < k_; ++c) {
          float dist = SquaredDistance(vec, centroids.data() + c * dim, dim);
          if (dist < min_dist) {
            min_dist = dist;
            best_c = c;
          }
        }
        assignments[i] = best_c;
        counts[best_c]++;
        
        float* cent_vec = new_centroids.data() + best_c * dim;
        for (int d = 0; d < dim; ++d) {
          cent_vec[d] += vec[d];
        }
      }

      for (int c = 0; c < k_; ++c) {
        if (counts[c] > 0) {
          float inv_count = 1.0f / counts[c];
          for (int d = 0; d < dim; ++d) {
            new_centroids[c * dim + d] *= inv_count;
          }
        } else {
             int rand_idx = indices[c % n];
             std::copy_n(data + rand_idx * dim, dim, new_centroids.begin() + c * dim);
        }
      }
      if (iter > 0 && centroids == new_centroids) break;
      centroids = new_centroids;
    }
    return centroids;
  }
 private:
  int k_;
  int max_iters_;
};

// =================================================================================
// Optimized Product Quantizer (OPQ)
// =================================================================================
class OPQ {
 public:
  static constexpr int kNumSubQuantizers = 16;
  static constexpr int kNumCentroids = 256;
  static constexpr int kOpqIterations = 5;
  static constexpr int kKMeansIterations = 10;

  void Train(const float* data, int n, int d, int sample_size = 20000) {
    d_ = d;
    sub_dim_ = (d + kNumSubQuantizers - 1) / kNumSubQuantizers;
    d_padded_ = sub_dim_ * kNumSubQuantizers;
    R_ = Matrix::Identity(d_padded_);

    int train_n = n;
    const float* train_data = data;
    std::vector<float> sample_buffer;

    if (sample_size > 0 && sample_size < n) {
      train_n = sample_size;
      sample_buffer.resize(train_n * d);
      std::vector<int> indices(n);
      std::iota(indices.begin(), indices.end(), 0);
      std::shuffle(indices.begin(), indices.end(), global::rng);
      for (int i = 0; i < train_n; ++i) {
        std::copy_n(data + indices[i] * d, d, sample_buffer.begin() + i * d);
      }
      train_data = sample_buffer.data();
    }

    Matrix X(train_n, d_padded_);
    for (int i = 0; i < train_n; ++i) {
      for (int j = 0; j < d; ++j) {
        X(i, j) = train_data[i * d + j];
      }
    }

    codebooks_.resize(kNumSubQuantizers * kNumCentroids * sub_dim_);

    for (int iter = 0; iter < kOpqIterations; ++iter) {
      Matrix Rt = R_.transpose();
      Matrix Y = MatMul(X, Rt);

      std::vector<std::thread> threads;
      for (int m = 0; m < kNumSubQuantizers; ++m) {
        threads.emplace_back([&, m]() {
          std::vector<float> sub_data(train_n * sub_dim_);
          for (int i = 0; i < train_n; ++i) {
            for (int j = 0; j < sub_dim_; ++j) {
              sub_data[i * sub_dim_ + j] = Y(i, m * sub_dim_ + j);
            }
          }
          SimpleKMeans kmeans(kNumCentroids, kKMeansIterations);
          std::vector<float> centroids = kmeans.Train(sub_data.data(), train_n, sub_dim_);
          std::copy(centroids.begin(), centroids.end(), 
                    codebooks_.begin() + m * kNumCentroids * sub_dim_);
        });
      }
      for (auto& t : threads) t.join();

      if (iter < kOpqIterations - 1) {
        Matrix X_hat(train_n, d_padded_);
        
        // Parallelize assignment step
        ParallelFor(0, train_n, [&](int i, int thread_idx) {
           for (int m = 0; m < kNumSubQuantizers; ++m) {
              const float* y_sub = &Y(i, m * sub_dim_);
              const float* cb_start = &codebooks_[m * kNumCentroids * sub_dim_];
              int best_k = 0;
              float min_dist = std::numeric_limits<float>::max();
              for(int k=0; k<kNumCentroids; ++k) {
                  float dist = SquaredDistance(y_sub, cb_start + k * sub_dim_, sub_dim_);
                  if (dist < min_dist) { min_dist = dist; best_k = k; }
              }
              const float* best_cent = cb_start + best_k * sub_dim_;
              for(int j=0; j<sub_dim_; ++j) {
                  X_hat(i, m * sub_dim_ + j) = best_cent[j];
              }
           }
        });

        Matrix Xt = X.transpose();
        Matrix M = MatMul(Xt, X_hat);
        Matrix U, V;
        ComputeSVD(M, U, V);
        Matrix Ut = U.transpose();
        R_ = MatMul(V, Ut);
      }
    }
  }

  std::vector<uint8_t> Encode(const float* vector) {
    Matrix v_vec(d_padded_, 1);
    for(int i=0; i<d_; ++i) v_vec(i, 0) = vector[i];
    Matrix v_rot = MatMul(R_, v_vec);

    std::vector<uint8_t> code(kNumSubQuantizers);
    for (int m = 0; m < kNumSubQuantizers; ++m) {
        const float* sub_vec = &v_rot(m * sub_dim_, 0);
        const float* cb = &codebooks_[m * kNumCentroids * sub_dim_];
        int best_k = 0;
        float min_dist = std::numeric_limits<float>::max();
        for(int k=0; k<kNumCentroids; ++k) {
            float d = SquaredDistance(sub_vec, cb + k * sub_dim_, sub_dim_);
            if(d < min_dist) { min_dist = d; best_k = k; }
        }
        code[m] = (uint8_t)best_k;
    }
    return code;
  }

  std::vector<float> BuildDistanceTable(const float* query) const {
    Matrix q_vec(d_padded_, 1);
    for(int i=0; i<d_; ++i) q_vec(i, 0) = query[i];
    Matrix q_rot = MatMul(R_, q_vec);
    std::vector<float> dist_table(kNumSubQuantizers * kNumCentroids);
    
    for (int m = 0; m < kNumSubQuantizers; ++m) {
        const float* q_sub = &q_rot(m * sub_dim_, 0);
        const float* cb = &codebooks_[m * kNumCentroids * sub_dim_];
        for (int k = 0; k < kNumCentroids; ++k) {
            dist_table[m * kNumCentroids + k] = SquaredDistance(q_sub, cb + k * sub_dim_, sub_dim_);
        }
    }
    return dist_table;
  }
  
  inline float Dist(const std::vector<float>& dist_table, const uint8_t* code) const {
      float dist = 0.0f;
      for(int m=0; m<kNumSubQuantizers; ++m) {
          dist += dist_table[m * kNumCentroids + code[m]];
      }
      return dist;
  }
  
  int GetSubDim() const { return sub_dim_; }
  int GetD() const { return d_; }

 private:
  int d_ = 0;
  int d_padded_ = 0;
  int sub_dim_ = 0;
  Matrix R_;
  std::vector<float> codebooks_;
};

// =================================================================================
// HNSW Graph Template
// =================================================================================
template <int kMaxNeighbors, int kMaxLayer0Neighbors, int kMaxLevel>
class HnswGraph {
 public:
  struct Node {
    int level;
    std::vector<int> flat_links;
    std::vector<int> link_counts;
    std::unique_ptr<std::mutex> lock;
  };

  HnswGraph() { level_mult_ = 1.0 / std::log(1.0 * kMaxNeighbors); }

  void Initialize(size_t n, std::mt19937& rng) {
    nodes_.resize(n);
    for (size_t i = 0; i < n; ++i) {
      int level = GetRandomLevel(rng);
      nodes_[i].level = level;
      nodes_[i].link_counts.resize(level + 1, 0);
      nodes_[i].lock = std::make_unique<std::mutex>();
      size_t total_links = kMaxLayer0Neighbors;
      if (level > 0) total_links += (size_t)level * kMaxNeighbors;
      nodes_[i].flat_links.resize(total_links);
    }
    entry_point_ = 0;
    max_level_ = nodes_[0].level;
  }

  inline int GetLinkOffset(int level) const {
    return (level == 0) ? 0 : (kMaxLayer0Neighbors + (level - 1) * kMaxNeighbors);
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
    for (int i = 0; i < count; ++i) if (links[i] == target_id) return true;
    return false;
  }
  void AppendNeighbor(int node_id, int level, int target_id) {
    int count = nodes_[node_id].link_counts[level];
    int* links = GetNeighborsPtr(node_id, level);
    links[count] = target_id;
    nodes_[node_id].link_counts[level] = count + 1;
  }
  void SetNeighbors(int node_id, int level, const int* new_neighbors, int new_count) {
    int* links = GetNeighborsPtr(node_id, level);
    std::copy_n(new_neighbors, new_count, links);
    nodes_[node_id].link_counts[level] = new_count;
  }
  int GetNodeLevel(int node_id) const { return nodes_[node_id].level; }
  int GetEntryPoint() const { return entry_point_; }
  void SetEntryPoint(int entry_point) { entry_point_ = entry_point; }
  int GetMaxLevel() const { return max_level_; }
  void SetMaxLevel(int max_level) { max_level_ = max_level; }
  
  std::mutex& GetNodeLock(int node_id) { return *nodes_[node_id].lock; }

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

// =================================================================================
// Solution Class
// =================================================================================
class Solution {
 public:
  static constexpr int kM = 48;
  static constexpr int kM0 = 96;
  static constexpr int kEfConstruction = 600;
  static constexpr int kEfSearch = 300;
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
    
    if constexpr (global::kDEBUG) std::cout << "Training OPQ..." << std::endl;
    opq_.Train(data_ptr_, n_, d_, 20000);
    
    if constexpr (global::kDEBUG) std::cout << "Encoding data..." << std::endl;
    codes_.resize(n_ * OPQ::kNumSubQuantizers);
    
    // Parallel Encode
    ParallelFor(0, n_, [&](int i, int) {
        auto code = opq_.Encode(data_ptr_ + i * d_);
        std::copy(code.begin(), code.end(), codes_.begin() + i * OPQ::kNumSubQuantizers);
    });
    
    if constexpr (global::kDEBUG) {
      std::cout << "Building HNSW (d=" << d_ << ", M=" << kM
                << ") for " << n_ << " vectors...\n";
    }

    std::atomic<int> progress = 0;
    
    // Parallel Build
    // Thread-local visited lists are handled by lambda capture or passing
    int num_threads = std::thread::hardware_concurrency();
    if(num_threads == 0) num_threads = 4;
    std::vector<VisitedList> thread_visited_lists(num_threads);
    for(auto& vl : thread_visited_lists) vl.Resize(n_);

    ParallelFor(1, n_, [&](int curr_obj, int thread_idx) {
        this->InsertVec(curr_obj, thread_visited_lists[thread_idx]);
        
        if constexpr (global::kDEBUG) {
            int p = progress.fetch_add(1, std::memory_order_relaxed);
            if (p % 1000 == 0) {
                // Stdout might be messy, but acceptable for debug
                std::cout << "Build: " << static_cast<int>(100.0 * p / n_) << "% \r" << std::flush;
            }
        }
    });

    if constexpr (global::kDEBUG) {
      std::cout << "Build: 100% - Done.\n";
      total_dist_calcs_ = 0;
    }
  }

  void search(const std::vector<float>& query, int* res) {
    // Thread-local visited list for search is static thread_local
    static thread_local VisitedList visited_list;
    if (visited_list.Size() != n_) visited_list.Resize(n_);

    int curr_ep = graph_.GetEntryPoint();
    int m_level = graph_.GetMaxLevel();
    const float* q_data = query.data();
    
    auto dist_table = opq_.BuildDistanceTable(q_data);

    curr_ep = GreedySearch(q_data, curr_ep, m_level, 0, dist_table);
    
    auto top = BeamSearch(q_data, curr_ep, kEfSearch, 0, visited_list, dist_table);

    size_t k = 0;
    std::vector<std::pair<float, int>> sorted;
    sorted.reserve(top.size());
    while (!top.empty()) {
      sorted.push_back(top.top());
      top.pop();
    }
    std::sort(sorted.begin(), sorted.end());

    std::vector<std::pair<float, int>> final_res;
    final_res.reserve(sorted.size());
    for(auto& p : sorted) {
        float exact_dist = SquaredDistance(q_data, data_ptr_ + p.second * d_, d_);
        final_res.push_back({exact_dist, p.second});
    }
    std::sort(final_res.begin(), final_res.end());

    for (const auto& p : final_res) {
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
  OPQ opq_;
  std::vector<uint8_t> codes_;
  std::mutex global_lock_; // For entry point updates

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

  VisitedList visited_list_; // Kept for member compatibility, but unused in parallel build

  inline float CalcDistExact(const float* a, const float* b) {
    if constexpr (global::kDEBUG) total_dist_calcs_++;
    return SquaredDistance(a, b, d_);
  }
  
  inline float CalcDistOPQ(const std::vector<float>& table, int id) {
      if constexpr (global::kDEBUG) total_dist_calcs_++;
      return opq_.Dist(table, codes_.data() + id * OPQ::kNumSubQuantizers);
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
      output_buffer[output_count++] = cand.second;
    }
  }

  void AddConnection(int src, int dest, int level) {
    // LOCKING needed for node 'src'
    std::lock_guard<std::mutex> lock(graph_.GetNodeLock(src));
    
    if (graph_.HasNeighbor(src, level, dest)) return;

    int count = graph_.GetNeighborCount(src, level);
    int max_m = (level == 0) ? kM0 : kM;

    if (count < max_m) {
      graph_.AppendNeighbor(src, level, dest);
    } else {
      std::vector<std::pair<float, int>> candidates;
      candidates.reserve(max_m + 1);

      const int* links_ptr = graph_.GetNeighborsPtr(src, level);
      for (int i = 0; i < count; ++i) {
        float d = CalcDistExact(data_ptr_ + src * d_, data_ptr_ + links_ptr[i] * d_);
        candidates.push_back({d, links_ptr[i]});
      }

      float d_dest = CalcDistExact(data_ptr_ + src * d_, data_ptr_ + dest * d_);
      candidates.push_back({d_dest, dest});

      std::vector<int> new_links(max_m);
      int new_count = 0;
      GetNeighborsHeuristic(src, candidates, level, new_links.data(), new_count);
      graph_.SetNeighbors(src, level, new_links.data(), new_count);
    }
  }

  int InsertIntoLayer(int new_vec_id, int seed, int level, VisitedList& visited) {
    const float* curr_vec = data_ptr_ + new_vec_id * d_;
    auto top_candidates = BeamSearchBuild(curr_vec, seed, kEfConstruction, level, visited);

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
    GetNeighborsHeuristic(new_vec_id, potential, level, link_dst_buffer.data(), count);
    
    // Set neighbors for new node. NO LOCK needed for new_vec_id as only this thread sees it yet.
    graph_.SetNeighbors(new_vec_id, level, link_dst_buffer.data(), count);
    
    for (int j = 0; j < count; ++j) {
      AddConnection(link_dst_buffer[j], new_vec_id, level);
    }

    return best_next_seed;
  }

  int GreedySearch(const float* query_data, int entry_point, int start_level,
                   int stop_level, const std::vector<float>& dist_table) {
    int curr_node = entry_point;
    for (int l = start_level; l > stop_level; l--) {
      bool changed = true;
      while (changed) {
        changed = false;
        float dist_min = CalcDistOPQ(dist_table, curr_node);
        if (l > graph_.GetNodeLevel(curr_node)) break;
        int count = graph_.GetNeighborCount(curr_node, l);
        const int* links = graph_.GetNeighborsPtr(curr_node, l);
        for (int i = 0; i < count; ++i) {
          int neighbor = links[i];
          float d = CalcDistOPQ(dist_table, neighbor);
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
      VisitedList& visited, const std::vector<float>& dist_table) {
    using QueueItem = std::pair<float, int>;
    std::priority_queue<QueueItem> top_candidates;
    std::priority_queue<QueueItem, std::vector<QueueItem>,
                        std::greater<>>
        candidates;

    visited.Advance();
    float initial_dist = CalcDistOPQ(dist_table, entry_point);
    top_candidates.push({initial_dist, entry_point});
    candidates.push({initial_dist, entry_point});
    visited.Visit(entry_point);

    while (!candidates.empty()) {
      auto [curr_dist, curr_id] = candidates.top();
      if (curr_dist > top_candidates.top().first && top_candidates.size() >= ef) break;
      candidates.pop();

      int size = graph_.GetNeighborCount(curr_id, level);
      const int* links = graph_.GetNeighborsPtr(curr_id, level);

      for (int i = 0; i < size; ++i) {
        int neighbor_id = links[i];
        if (!visited.Visit(neighbor_id)) {
          float d = CalcDistOPQ(dist_table, neighbor_id);
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

  std::priority_queue<std::pair<float, int>> BeamSearchBuild(
      const float* query_data, int entry_point, int ef, int level,
      VisitedList& visited) {
    using QueueItem = std::pair<float, int>;
    std::priority_queue<QueueItem> top_candidates;
    std::priority_queue<QueueItem, std::vector<QueueItem>,
                        std::greater<>>
        candidates;

    visited.Advance();
    float initial_dist = CalcDistExact(query_data, data_ptr_ + entry_point * d_);
    top_candidates.push({initial_dist, entry_point});
    candidates.push({initial_dist, entry_point});
    visited.Visit(entry_point);

    while (!candidates.empty()) {
      auto [curr_dist, curr_id] = candidates.top();
      if (curr_dist > top_candidates.top().first && top_candidates.size() >= ef) break;
      candidates.pop();

      int size = graph_.GetNeighborCount(curr_id, level);
      const int* links = graph_.GetNeighborsPtr(curr_id, level);

      for (int i = 0; i < size; ++i) {
        int neighbor_id = links[i];
        if (i + 1 < size) {
           _mm_prefetch((const char*)(data_ptr_ + links[i + 1] * d_), _MM_HINT_T0);
        }
        if (!visited.Visit(neighbor_id)) {
          float d = CalcDistExact(query_data, data_ptr_ + neighbor_id * d_);
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

  void InsertVec(int new_vec_id, VisitedList& visited) {
    int curr_level = graph_.GetNodeLevel(new_vec_id);
    int curr_ep = graph_.GetEntryPoint();
    int curr_max = graph_.GetMaxLevel();
    const float* curr_vec = data_ptr_ + new_vec_id * d_;

    // Use EXACT search for finding entry point during build
    curr_ep = GreedySearchBuild(curr_vec, curr_ep, curr_max, curr_level);

    for (int l = std::min(curr_level, curr_max); l >= 0; l--) {
      curr_ep = InsertIntoLayer(new_vec_id, curr_ep, l, visited);
    }

    if (curr_level > graph_.GetMaxLevel()) {
      std::lock_guard<std::mutex> lock(global_lock_);
      if (curr_level > graph_.GetMaxLevel()) {
        graph_.SetMaxLevel(curr_level);
        graph_.SetEntryPoint(new_vec_id);
      }
    }
  }

  int GreedySearchBuild(const float* query_data, int entry_point, int start_level, int stop_level) {
      int curr_node = entry_point;
      for (int l = start_level; l > stop_level; l--) {
        bool changed = true;
        while (changed) {
          changed = false;
          float dist_min = CalcDistExact(query_data, data_ptr_ + curr_node * d_);
          if (l > graph_.GetNodeLevel(curr_node)) break;
          int count = graph_.GetNeighborCount(curr_node, l);
          const int* links = graph_.GetNeighborsPtr(curr_node, l);
          for (int i = 0; i < count; ++i) {
            int neighbor = links[i];
            float d = CalcDistExact(query_data, data_ptr_ + neighbor * d_);
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
};

static_assert(IsSolution<Solution>);
