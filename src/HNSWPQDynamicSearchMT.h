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

// -----------------------------------------------------------------------------
// AVX2 Distance Primitives
// -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// Product Quantization (PQ) Engine
// -----------------------------------------------------------------------------

class ProductQuantizer {
public:
    static constexpr int kCentroids = 256; // Fixed to fit in uint8_t

    ProductQuantizer() = default;

    bool IsTrained() const { return trained_; }
    int GetM() const { return m_; }

    // Train using K-Means clustering on subspaces
    void Train(const float* data, size_t n, int d, int m_subquantizers) {
        n_ = n;
        d_ = d;
        m_ = m_subquantizers;
        d_sub_ = d_ / m_;

        if (d_ % m_ != 0) {
            std::cerr << "PQ Error: Dimension not divisible by M\n";
            return;
        }

        centroids_.resize(m_ * kCentroids * d_sub_);
        
        // Train each subspace independently
        std::vector<std::thread> threads;
        int num_threads = std::min((int)std::thread::hardware_concurrency(), m_);
        
        std::atomic<int> subspace_counter(0);

        auto train_worker = [&]() {
            while (true) {
                int m_idx = subspace_counter.fetch_add(1);
                if (m_idx >= m_) break;
                TrainSubspace(data, m_idx);
            }
        };

        for (int i = 0; i < num_threads; ++i) threads.emplace_back(train_worker);
        for (auto& t : threads) t.join();

        trained_ = true;
    }

    // Convert all vectors to PQ codes
    std::vector<uint8_t> EncodeBatch(const float* data, size_t n) {
        std::vector<uint8_t> codes(n * m_);
        
        // Simple parallel encoding
        #pragma omp parallel for schedule(dynamic) if(n > 1000)
        for (size_t i = 0; i < n; ++i) {
            EncodeSingle(data + i * d_, codes.data() + i * m_);
        }
        return codes;
    }

    // Precompute Distance Table for ADC (Asymmetric Distance Computation)
    // Table size: M * 256
    void ComputeDistanceTable(const float* query, float* table) const {
        for (int m = 0; m < m_; ++m) {
            const float* q_sub = query + m * d_sub_;
            const float* c_base = centroids_.data() + m * kCentroids * d_sub_;
            float* t_base = table + m * kCentroids;

            // Compute distance from query sub-vector to all 256 centroids
            // This is small enough that compiler autovectorization usually handles it well,
            // but we use the existing SquaredDistance helper.
            for (int k = 0; k < kCentroids; ++k) {
                t_base[k] = SquaredDistance(q_sub, c_base + k * d_sub_, d_sub_);
            }
        }
    }

    // Fast ADC Distance Lookup
    inline float DistPQ(const float* table, const uint8_t* code) const {
        float dist = 0.0f;
        // Unrolling hint for compiler
        #pragma unroll
        for (int m = 0; m < m_; ++m) {
            dist += table[m * kCentroids + code[m]];
        }
        return dist;
    }

private:
    bool trained_ = false;
    int d_ = 0;
    int m_ = 0;
    int d_sub_ = 0;
    size_t n_ = 0;
    std::vector<float> centroids_; // [m][256][d_sub] flattened

    void TrainSubspace(const float* data, int m_idx) {
        // Pointers for this subspace
        float* sub_centroids = centroids_.data() + m_idx * kCentroids * d_sub_;
        int offset = m_idx * d_sub_;

        // 1. Initialize Centroids (Random Sample)
        // Use a deterministic seed for reproducibility in debug
        std::mt19937 rng(1234 + m_idx);
        std::uniform_int_distribution<size_t> dist(0, n_ - 1);
        
        for (int k = 0; k < kCentroids; ++k) {
            size_t idx = dist(rng);
            std::memcpy(sub_centroids + k * d_sub_, data + idx * d_ + offset, d_sub_ * sizeof(float));
        }

        // 2. K-Means Iterations
        const int max_iters = 15; // Sufficient for PQ
        std::vector<int> counts(kCentroids);
        std::vector<float> new_sums(kCentroids * d_sub_);
        
        // To speed up training, we can sample a subset of data if N is huge
        size_t train_n = std::min(n_, (size_t)25000); // Train on up to 25k points per subspace
        std::vector<size_t> indices(train_n);
        if (train_n < n_) {
             for(size_t i=0; i<train_n; ++i) indices[i] = dist(rng);
        } else {
             for(size_t i=0; i<train_n; ++i) indices[i] = i;
        }

        for (int iter = 0; iter < max_iters; ++iter) {
            std::fill(counts.begin(), counts.end(), 0);
            std::fill(new_sums.begin(), new_sums.end(), 0.0f);

            // Assign step
            for (size_t i = 0; i < train_n; ++i) {
                const float* vec = data + indices[i] * d_ + offset;
                int best_k = -1;
                float min_dist = std::numeric_limits<float>::max();

                for (int k = 0; k < kCentroids; ++k) {
                    float d = SquaredDistance(vec, sub_centroids + k * d_sub_, d_sub_);
                    if (d < min_dist) {
                        min_dist = d;
                        best_k = k;
                    }
                }
                
                counts[best_k]++;
                float* sum_ptr = new_sums.data() + best_k * d_sub_;
                for (int j = 0; j < d_sub_; ++j) sum_ptr[j] += vec[j];
            }

            // Update step
            for (int k = 0; k < kCentroids; ++k) {
                if (counts[k] > 0) {
                    float inv = 1.0f / counts[k];
                    for (int j = 0; j < d_sub_; ++j) {
                        sub_centroids[k * d_sub_ + j] = new_sums[k * d_sub_ + j] * inv;
                    }
                }
            }
        }
    }

    void EncodeSingle(const float* vec, uint8_t* code) {
        for (int m = 0; m < m_; ++m) {
            const float* sub_vec = vec + m * d_sub_;
            const float* sub_centroids = centroids_.data() + m * kCentroids * d_sub_;
            
            int best_k = 0;
            float min_dist = std::numeric_limits<float>::max();

            // This inner loop is critical during encoding, but happens only once per vector
            for (int k = 0; k < kCentroids; ++k) {
                float d = SquaredDistance(sub_vec, sub_centroids + k * d_sub_, d_sub_);
                if (d < min_dist) {
                    min_dist = d;
                    best_k = k;
                }
            }
            code[m] = (uint8_t)best_k;
        }
    }
};

// -----------------------------------------------------------------------------
// HNSW Graph (Standard Layout)
// -----------------------------------------------------------------------------
template <int kMaxNeighbors, int kMaxLayer0Neighbors, int kMaxLevel>
class HnswGraph {
public:
  struct Node {
    int level;
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

  void SetNeighbors(int node_id, int level, const int* new_neighbors, int new_count) {
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

// -----------------------------------------------------------------------------
// Main Solution
// -----------------------------------------------------------------------------

class Solution {
public:
  static constexpr int kM = 64;
  static constexpr int kM0 = 128;
  static constexpr int kEfConstruction = 600;
  static constexpr int kEfSearch = 300;
  static constexpr int kMaxLevel = 16;
  
  // PQ Parameters
  static constexpr int kPQM = 32; // Number of subquantizers (must divide d)

  Solution() {
    layer_times_ns_ = std::vector<std::atomic<long long>>(kMaxLevel + 1);
    for(auto& x : layer_times_ns_) x = 0;
  }

  ~Solution() {
    if constexpr (global::kDEBUG) {
      long long total_s = total_searches_.load();
      if (total_s > 0) {
        std::cout << "\n=== HNSW Profiling Stats (Optimized + PQ) ===\n";
        std::cout << "Total Searches: " << total_s << "\n";
        std::cout << "Avg Distance Calcs: "
          << static_cast<double>(total_dist_calcs_.load()) / total_s
          << "\n";
        std::cout << "PQ Enabled: " << (pq_.IsTrained() ? "Yes" : "No") << "\n";
      }
    }
  }

  void build(int d, const std::vector<float>& base) {
    d_ = d;
    data_storage_ = base;
    data_ptr_ = data_storage_.data();
    n_ = base.size() / d_;

    graph_.Initialize(n_, global::rng);

    // 1. Build Graph using High-Precision Floats
    if constexpr (global::kDEBUG) {
      std::cout << "Building HNSW Graph...\n";
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
                std::cout << "Build: " << static_cast<int>(100.0 * curr_obj / n_) << "% \r" << std::flush;
            }
          }
        }
      });
    }

    for (auto& t : threads) {
      if (t.joinable()) t.join();
    }

    // 2. Train and Compress with PQ for Search
    if (d_ % kPQM == 0) {
        if constexpr (global::kDEBUG) std::cout << "\nTraining Product Quantizer (M=" << kPQM << ")...\n";
        pq_.Train(data_ptr_, n_, d_, kPQM);
        if constexpr (global::kDEBUG) std::cout << "Encoding Data...\n";
        pq_codes_ = pq_.EncodeBatch(data_ptr_, n_);
        pq_data_ptr_ = pq_codes_.data();
        
        // Optional: Release float memory if strict memory constraint
        // data_storage_.clear(); 
        // data_storage_.shrink_to_fit();
        // data_ptr_ = nullptr;
    } else {
        if constexpr (global::kDEBUG) std::cout << "Skipping PQ (Dimension not divisible by " << kPQM << ")\n";
    }

    if constexpr (global::kDEBUG) {
      std::cout << "Build Complete.\n";
      total_dist_calcs_ = 0;
    }
  }

  void search(const std::vector<float>& query, int* res) {
    thread_local VisitedList visited_list;
    if (visited_list.Size() != n_) visited_list.Resize(n_);

    int curr_ep = graph_.GetEntryPoint();
    int m_level = graph_.GetMaxLevel();
    const float* q_data = query.data();

    // Prepare PQ Distance Table if available
    float* dist_table = nullptr;
    // Use stack memory for table if small enough, otherwise vector
    alignas(32) float local_dist_table[kPQM * 256]; 
    
    if (pq_.IsTrained()) {
        pq_.ComputeDistanceTable(q_data, local_dist_table);
        dist_table = local_dist_table;
    }

    // 1. Upper Layers: Greedy Search
    curr_ep = GreedySearch(q_data, curr_ep, m_level, 0, dist_table);

    // 2. Layer 0: Beam Search
    {
      auto top = BeamSearchDynamic(q_data, curr_ep, kEfSearch, 0, visited_list, dist_table);

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
  
  // PQ Components
  ProductQuantizer pq_;
  std::vector<uint8_t> pq_codes_;
  const uint8_t* pq_data_ptr_ = nullptr;

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

  // Unified Distance Calculator
  // If dist_table is provided, it uses ADC (PQ). Otherwise, uses exact float distance.
  inline float CalcDist(const float* q_raw, const float* dist_table, int node_id) {
    if constexpr (global::kDEBUG) {
      total_dist_calcs_.fetch_add(1, std::memory_order_relaxed);
    }

    if (dist_table && pq_data_ptr_) {
        // Fast PQ Lookup
        return pq_.DistPQ(dist_table, pq_data_ptr_ + node_id * kPQM);
    } else {
        // Fallback or Build-phase exact distance
        return SquaredDistance(q_raw, data_ptr_ + node_id * d_, d_);
    }
  }

  // Used only during construction (comparing two stored nodes)
  inline float CalcDistNodeNode(int id_a, int id_b) {
      return SquaredDistance(data_ptr_ + id_a * d_, data_ptr_ + id_b * d_, d_);
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
        // Heuristic always uses high precision during build for better graph quality
        float d_neighbor = CalcDistNodeNode(cand_id, output_buffer[j]);
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
        float d = CalcDistNodeNode(src, links_ptr[i]);
        candidates.push_back({d, links_ptr[i]});
      }

      float d_dest = CalcDistNodeNode(src, dest);
      candidates.push_back({d_dest, dest});

      std::vector<int> new_links(max_m);
      int new_count = 0;
      GetNeighborsHeuristic(src, candidates, level, new_links.data(),
                            new_count);
      graph_.SetNeighbors(src, level, new_links.data(), new_count);
    }
  }

  int InsertIntoLayer(int new_vec_id, int seed, int level, VisitedList& visited) {
    const float* curr_vec = data_ptr_ + new_vec_id * d_;
    
    // Build phase: always pass nullptr for dist_table to force Exact Distance
    auto top_candidates = BeamSearch(curr_vec, seed, kEfConstruction, level, visited, nullptr);

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
                   int stop_level, const float* dist_table) {
    int curr_node = entry_point;
    for (int l = start_level; l > stop_level; l--) {
      bool changed = true;
      while (changed) {
        changed = false;
        float dist_min = CalcDist(query_data, dist_table, curr_node);

        if (l > graph_.GetNodeLevel(curr_node)) break;

        int count = graph_.GetNeighborCount(curr_node, l);
        const int* links = graph_.GetNeighborsPtr(curr_node, l);

        if(count > 0 && pq_data_ptr_ && dist_table) {
             // PQ Prefetching
             _mm_prefetch(reinterpret_cast<const char*>(pq_data_ptr_ + links[0] * kPQM), _MM_HINT_T0);
        } else if (count > 0) {
             // Float Prefetching
             _mm_prefetch(reinterpret_cast<const char*>(data_ptr_ + links[0] * d_), _MM_HINT_T0);
        }

        for (int i = 0; i < count; ++i) {
          int neighbor = links[i];
          if (i + 1 < count) {
            if(pq_data_ptr_ && dist_table)
                _mm_prefetch(reinterpret_cast<const char*>(pq_data_ptr_ + links[i + 1] * kPQM), _MM_HINT_T0);
            else
                _mm_prefetch(reinterpret_cast<const char*>(data_ptr_ + links[i + 1] * d_), _MM_HINT_T0);
          }

          float d = CalcDist(query_data, dist_table, neighbor);
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
    VisitedList& visited, const float* dist_table) {
    using QueueItem = std::pair<float, int>;
    std::priority_queue<QueueItem> top_candidates;
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> candidates;

    visited.Advance();
    float initial_dist = CalcDist(query_data, dist_table, entry_point);
    top_candidates.push({initial_dist, entry_point});
    candidates.push({initial_dist, entry_point});
    visited.Visit(entry_point);

    while (!candidates.empty()) {
      auto [curr_dist, curr_id] = candidates.top();
      if (curr_dist > top_candidates.top().first && top_candidates.size() >= ef)
        break;
      candidates.pop();

      int size = graph_.GetNeighborCount(curr_id, level);
      const int* links = graph_.GetNeighborsPtr(curr_id, level);

      // PREFETCHING OPTIMIZATION
      if(size > 0) {
        if(pq_data_ptr_ && dist_table)
            _mm_prefetch(reinterpret_cast<const char*>(pq_data_ptr_ + links[0] * kPQM), _MM_HINT_T0);
        else
            _mm_prefetch(reinterpret_cast<const char*>(data_ptr_ + links[0] * d_), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(visited.GetTagPtr(links[0])), _MM_HINT_T0);
      }

      for (int i = 0; i < size; ++i) {
        int neighbor_id = links[i];
        if (i + 1 < size) {
           if(pq_data_ptr_ && dist_table)
             _mm_prefetch(reinterpret_cast<const char*>(pq_data_ptr_ + links[i + 1] * kPQM), _MM_HINT_T0);
           else
             _mm_prefetch(reinterpret_cast<const char*>(data_ptr_ + links[i + 1] * d_), _MM_HINT_T0);
           _mm_prefetch(reinterpret_cast<const char*>(visited.GetTagPtr(links[i + 1])), _MM_HINT_T0);
        }

        if (!visited.Visit(neighbor_id)) {
          float d = CalcDist(query_data, dist_table, neighbor_id);
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
    VisitedList& visited,
    const float* dist_table
  ) {
    using QueueItem = std::pair<float, int>;
    std::priority_queue<QueueItem> top_candidates;
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> global_candidates;
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> local_candidates;

    visited.Advance();
    float initial_dist = CalcDist(query_data, dist_table, entry_point);
    
    top_candidates.push({initial_dist, entry_point});
    local_candidates.push({initial_dist, entry_point});
    visited.Visit(entry_point);

    while (!global_candidates.empty() || !local_candidates.empty()) {
      QueueItem curr_item;
      bool is_local_search = false;

      if (!local_candidates.empty()) {
        is_local_search = true;
        curr_item = local_candidates.top();
        local_candidates.pop();
        while(!local_candidates.empty()) {
            global_candidates.push(local_candidates.top());
            local_candidates.pop();
        }
      } else {
        curr_item = global_candidates.top();
        global_candidates.pop();
      }

      auto [curr_dist, curr_id] = curr_item;

      if (top_candidates.size() >= ef && curr_dist > top_candidates.top().first) {
        if (is_local_search) continue; 
        else break;
      }

      int size = graph_.GetNeighborCount(curr_id, level);
      const int* links = graph_.GetNeighborsPtr(curr_id, level);

      // Prefetching Optimization
      if(size > 0) {
        if(pq_data_ptr_ && dist_table)
            _mm_prefetch(reinterpret_cast<const char*>(pq_data_ptr_ + links[0] * kPQM), _MM_HINT_T0);
        else
            _mm_prefetch(reinterpret_cast<const char*>(data_ptr_ + links[0] * d_), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(visited.GetTagPtr(links[0])), _MM_HINT_T0);
      }

      for (int i = 0; i < size; ++i) {
        int neighbor_id = links[i];
        if (i + 1 < size) {
          if(pq_data_ptr_ && dist_table)
            _mm_prefetch(reinterpret_cast<const char*>(pq_data_ptr_ + links[i + 1] * kPQM), _MM_HINT_T0);
          else
            _mm_prefetch(reinterpret_cast<const char*>(data_ptr_ + links[i + 1] * d_), _MM_HINT_T0);
          _mm_prefetch(reinterpret_cast<const char*>(visited.GetTagPtr(links[i + 1])), _MM_HINT_T0);
        }

        if (!visited.Visit(neighbor_id)) {
          float d = CalcDist(query_data, dist_table, neighbor_id);
          
          if (top_candidates.size() < ef || d < top_candidates.top().first) {
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
    // Build phase: use exact distance (dist_table = nullptr)
    curr_ep = GreedySearch(curr_vec, curr_ep, curr_max, curr_level, nullptr);

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
