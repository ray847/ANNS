#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <mutex>
#include <immintrin.h>
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <thread>
#include <atomic>
#include <functional>

#include "Eigen/Dense"
#include "StackHeap.h"

constexpr int kPrefetchDis = 8;

namespace TestSolution {

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
    for (int i = 0; i < 16; ++i) { // 128 / 8 = 16
      __m256 vector_a_256 = _mm256_loadu_ps(vector_a + i * 8);
      __m256 vector_b_256 = _mm256_loadu_ps(vector_b + i * 8);
      __m256 diff_256 = _mm256_sub_ps(vector_a_256, vector_b_256);
      sum_256 = _mm256_add_ps(_mm256_mul_ps(diff_256, diff_256), sum_256);
    }
    // Horizontal sum of the 8 floats in sum_256
    __m128 sum_128 = _mm256_extractf128_ps(sum_256, 0); // Low 128 bits
    sum_128 = _mm_add_ps(sum_128, _mm256_extractf128_ps(sum_256, 1)); // High 128 bits
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    return _mm_cvtss_f32(sum_128);
  }
};

template <>
struct Distance<100, true> {
  __attribute__((target("avx2,fma"))) inline static float L2Sq(
      const float* vector_a, const float* vector_b) {
    __m256 sum_256 = _mm256_setzero_ps();
    // Process 96 dimensions (12 * 8)
    for (int i = 0; i < 12; ++i) { 
      __m256 vector_a_256 = _mm256_loadu_ps(vector_a + i * 8);
      __m256 vector_b_256 = _mm256_loadu_ps(vector_b + i * 8);
      __m256 diff_256 = _mm256_sub_ps(vector_a_256, vector_b_256);
      sum_256 = _mm256_add_ps(_mm256_mul_ps(diff_256, diff_256), sum_256);
    }
    
    // Horizontal sum for the 96 dimensions processed by AVX2
    __m128 sum_128 = _mm256_extractf128_ps(sum_256, 0);
    sum_128 = _mm_add_ps(sum_128, _mm256_extractf128_ps(sum_256, 1));
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    float result = _mm_cvtss_f32(sum_128);

    // Process the remaining 4 dimensions (100 - 96) using scalar operations
    for (int i = 96; i < 100; ++i) {
      const float diff = vector_a[i] - vector_b[i];
      result += diff * diff;
    }
    return result;
  }
};

// --- GenericSolution ---
template <int kDim, int kM, int kM0, int kEfConstruction, int kEfSearch>
class GenericSolution {
 public:
  GenericSolution();
  void Build(const std::vector<float>& base_data);
  void Search(const std::vector<float>& query, int k, int* result_indices);

 private:
  struct Node {
    int level;
    size_t offset;
  };
  
  struct VisitedList {
    std::vector<uint16_t> tags;
    uint16_t curr = 0;
    void resize(size_t n) { tags.resize(n, 0); }
    void reset() {
      curr++;
      if (curr == 0) {
        std::fill(tags.begin(), tags.end(), 0);
        curr = 1;
      }
    }
    inline bool visit(int id) {
      if (tags[id] == curr) return true;
      tags[id] = curr;
      return false;
    }
  };

  using Dist = Distance<kDim, true>;

  const double level_mult_;
  int max_level_ = -1;
  int entry_point_ = -1;
  std::vector<float> data_storage_;
  const float* data_ptr_ = nullptr;
  size_t num_points_ = 0;
  std::vector<Node> nodes_;
  std::vector<std::vector<std::vector<int>>> graph_data_;
  
  int GetRandomLevel_(std::mt19937& rng);
  void Insert_(int node_id, 
               std::vector<std::vector<std::vector<int>>>& temp_graph,
               std::vector<std::mutex>& locks);
  StackHeap<std::pair<float, int>, kEfConstruction + 1> SearchLayerFP_(
      const float* query, int entry_point, int ef, int level,
      const std::vector<std::vector<std::vector<int>>>& temp_graph,
      std::vector<std::mutex>& locks, VisitedList& visited);
  void SelectNeighbors_(
      StackHeap<std::pair<float, int>, kEfConstruction + 1>& candidates,
      std::vector<int>& target_list);
  void PruneGraph_(std::vector<std::vector<std::vector<int>>>& temp_graph);
  void SearchFP_(const std::vector<float>& query, int k, int* result_indices);
  void SearchAdaptiveFP_(const std::vector<float>& query, int k, int* result_indices);

  static constexpr float kGamma = 1.19f;
};

// Implementation

template <int kDim, int kM, int kM0, int kEfConstruction, int kEfSearch>
GenericSolution<kDim, kM, kM0, kEfConstruction, kEfSearch>::GenericSolution()
    : level_mult_(1.0 / std::log(1.0 * kM)) {}

template <int kDim, int kM, int kM0, int kEfConstruction, int kEfSearch>
void GenericSolution<kDim, kM, kM0, kEfConstruction, kEfSearch>::Build(const std::vector<float>& base_data) {
  data_storage_ = base_data;
  data_ptr_ = data_storage_.data();
  num_points_ = base_data.size() / kDim;
  nodes_.resize(num_points_);

  std::mt19937 rng(100);
  int max_level = 0;
  for (size_t i = 0; i < num_points_; ++i) {
    nodes_[i].level = GetRandomLevel_(rng);
    if (nodes_[i].level > max_level) max_level = nodes_[i].level;
  }
  max_level_ = max_level;

  if (num_points_ > 0) {
    entry_point_ = 0;
    for(size_t i = 1; i < num_points_; ++i) {
      if(nodes_[i].level > nodes_[entry_point_].level) entry_point_ = i;
    }
  }

  std::vector<std::vector<std::vector<int>>> temp_graph(num_points_);
  for(size_t i = 0; i < num_points_; ++i) {
    temp_graph[i].resize(nodes_[i].level + 1);
  }
  std::vector<std::mutex> locks(num_points_);
  std::atomic<size_t> atomic_idx{0};

  unsigned int num_threads = std::thread::hardware_concurrency();
  auto worker_func = [&]() {
    while (true) {
      size_t current_node_id = atomic_idx.fetch_add(1);
      if (current_node_id >= num_points_) break;
      Insert_(current_node_id, temp_graph, locks);
    }
  };

  std::vector<std::thread> threads;
  for (unsigned int i = 0; i < num_threads; ++i) {
    threads.emplace_back(worker_func);
  }
  for (auto& t : threads) {
    t.join();
  }
  PruneGraph_(temp_graph);
  graph_data_ = std::move(temp_graph);
}

template <int kDim, int kM, int kM0, int kEfConstruction, int kEfSearch>
void GenericSolution<kDim, kM, kM0, kEfConstruction, kEfSearch>::Search(const std::vector<float>& query, int k, int* result_indices) {
  SearchFP_(query, k, result_indices);
}

template <int kDim, int kM, int kM0, int kEfConstruction, int kEfSearch>
int GenericSolution<kDim, kM, kM0, kEfConstruction, kEfSearch>::GetRandomLevel_(std::mt19937& rng) {
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  return static_cast<int>(-std::log(dist(rng)) * level_mult_);
}

template <int kDim, int kM, int kM0, int kEfConstruction, int kEfSearch>
void GenericSolution<kDim, kM, kM0, kEfConstruction, kEfSearch>::Insert_(int node_id,
                       std::vector<std::vector<std::vector<int>>>& temp_graph,
                       std::vector<std::mutex>& locks) {

  thread_local VisitedList visited;
  if(visited.tags.size() != num_points_) visited.resize(num_points_);

  const float* query_vector = data_ptr_ + node_id * kDim;
  int current_ep = entry_point_;
  int node_level = nodes_[node_id].level;

  if (current_ep == -1) return;

  for (int level = max_level_; level > node_level; --level) {
    bool changed = true;
    while(changed) {
      changed = false;
      float min_dist = Dist::L2Sq(query_vector, data_ptr_ + current_ep * kDim);
      std::lock_guard<std::mutex> lock(locks[current_ep]);
      if (level >= temp_graph[current_ep].size()) continue;
      const auto& neighbors = temp_graph[current_ep][level];
      for (int neighbor_id : neighbors) {
        float dist = Dist::L2Sq(query_vector, data_ptr_ + neighbor_id * kDim);
        if (dist < min_dist) {
          min_dist = dist;
          current_ep = neighbor_id;
          changed = true;
        }
      }
    }
  }

  for (int level = std::min(node_level, max_level_); level >= 0; --level) {
    visited.reset();
    auto top_candidates = SearchLayerFP_(query_vector, current_ep, kEfConstruction, level, temp_graph, locks, visited);

    std::vector<int> neighbors;
    neighbors.reserve(kM0);
    SelectNeighbors_(top_candidates, neighbors);

    {
      std::lock_guard<std::mutex> lock(locks[node_id]);
      temp_graph[node_id][level] = neighbors;
    }

    for (int neighbor_id : neighbors) {
      std::lock_guard<std::mutex> lock(locks[neighbor_id]);
      if (level >= temp_graph[neighbor_id].size()) continue;
      auto& neighbor_links = temp_graph[neighbor_id][level];
      int neighbor_M = (level == 0) ? kM0 : kM;
      if (neighbor_links.size() < (size_t)neighbor_M) {
        neighbor_links.push_back(node_id);
      } else {
        float new_node_dist = Dist::L2Sq(data_ptr_ + neighbor_id * kDim, query_vector);
        StackHeap<std::pair<float, int>, kM0 + 1> temp_pq;
        for(int link : neighbor_links) {
          temp_pq.push({Dist::L2Sq(data_ptr_ + neighbor_id * kDim, data_ptr_ + link * kDim), link});
        }
        if (new_node_dist < temp_pq.top().first) {
          temp_pq.pop();
          temp_pq.push({new_node_dist, node_id});
          neighbor_links.clear();
          while(!temp_pq.empty()) {
            neighbor_links.push_back(temp_pq.top().second);
            temp_pq.pop();
          }
        }
      }
    }
    if(!top_candidates.empty()) {
      current_ep = top_candidates.top().second;
    }
  }
}

template <int kDim, int kM, int kM0, int kEfConstruction, int kEfSearch>
StackHeap<std::pair<float, int>, kEfConstruction + 1> GenericSolution<kDim, kM, kM0, kEfConstruction, kEfSearch>::SearchLayerFP_(
  const float* query, int entry_point, int ef, int level,
  const std::vector<std::vector<std::vector<int>>>& temp_graph,
  std::vector<std::mutex>& locks, VisitedList& visited) {

  using QueueItem = std::pair<float, int>;
  StackHeap<QueueItem, kEfConstruction + 1> top_results;
  StackHeap<QueueItem, kEfConstruction * 2, std::greater<QueueItem>> candidates;

  float initial_dist = Dist::L2Sq(query, data_ptr_ + entry_point * kDim);
  candidates.push({initial_dist, entry_point});
  top_results.push({initial_dist, entry_point});
  visited.visit(entry_point);

  while (!candidates.empty()) {
    auto [dist, id] = candidates.top();
    candidates.pop();
    if (dist > top_results.top().first && top_results.size() >= (size_t)ef) break;

    std::lock_guard<std::mutex> lock(locks[id]);
    if (level >= temp_graph[id].size()) continue;

    const auto& neighbors = temp_graph[id][level];
    for (int neighbor_id : neighbors) {
      if (!visited.visit(neighbor_id)) {
        float neighbor_dist = Dist::L2Sq(query, data_ptr_ + neighbor_id * kDim);
        if (top_results.size() < (size_t)ef || neighbor_dist < top_results.top().first) {
          candidates.push({neighbor_dist, neighbor_id});
          top_results.push({neighbor_dist, neighbor_id});
          if (top_results.size() > (size_t)ef) top_results.pop();
        }
      }
    }
  }
  return top_results;
}

template <int kDim, int kM, int kM0, int kEfConstruction, int kEfSearch>
void GenericSolution<kDim, kM, kM0, kEfConstruction, kEfSearch>::SelectNeighbors_(
  StackHeap<std::pair<float, int>, kEfConstruction + 1>& candidates,
  std::vector<int>& target_list) {

  if (candidates.empty()) return;

  std::vector<std::pair<float, int>> candidate_vec;
  candidate_vec.reserve(candidates.size());
  while(!candidates.empty()) {
    candidate_vec.push_back(candidates.top());
    candidates.pop();
  }
  std::reverse(candidate_vec.begin(), candidate_vec.end());

  for (const auto& cand : candidate_vec) {
    if (target_list.size() >= (size_t)kM) break;
    bool is_good = true;
    for (int selected_neighbor : target_list) {
      if (Dist::L2Sq(data_ptr_ + cand.second * kDim, data_ptr_ + selected_neighbor * kDim) < cand.first) {
        is_good = false;
        break;
      }
    }
    if (is_good) {
      target_list.push_back(cand.second);
    }
  }
}



template <int kDim, int kM, int kM0, int kEfConstruction, int kEfSearch>
void GenericSolution<kDim, kM, kM0, kEfConstruction, kEfSearch>::PruneGraph_(std::vector<std::vector<std::vector<int>>>& temp_graph) {
  std::atomic<size_t> atomic_idx{0};
  unsigned int num_threads = std::thread::hardware_concurrency();

  auto worker_func = [&]() {
    while(true) {
      size_t i = atomic_idx.fetch_add(1);
      if (i >= num_points_) break;
      
      for (int level = 0; level <= nodes_[i].level; ++level) {
        if (level >= temp_graph[i].size()) continue;
        
        std::vector<int>& neighbors = temp_graph[i][level];
        if (neighbors.empty()) continue;
        
        std::vector<std::pair<float, int>> candidates;
        candidates.reserve(neighbors.size());
        for (int n : neighbors) {
          candidates.push_back({Dist::L2Sq(data_ptr_ + i * kDim, data_ptr_ + n * kDim), n});
        }
        
        std::sort(candidates.begin(), candidates.end());
        
        std::vector<int> new_neighbors;
        new_neighbors.reserve(neighbors.size());
        
        int max_m = (level == 0) ? kM0 : kM;
        
        for (const auto& cand : candidates) {
          if (new_neighbors.size() >= (size_t)max_m) break;
          
          bool keep = true;
          for (int existing : new_neighbors) {
            if (Dist::L2Sq(data_ptr_ + cand.second * kDim, data_ptr_ + existing * kDim) < cand.first) {
              keep = false;
              break;
            }
          }
          if (keep) new_neighbors.push_back(cand.second);
        }
        neighbors = std::move(new_neighbors);
      }
    }
  };

  std::vector<std::thread> threads;
  for(unsigned int i=0; i<num_threads; ++i) threads.emplace_back(worker_func);
  for(auto& t : threads) t.join();
}

template <int kDim, int kM, int kM0, int kEfConstruction, int kEfSearch>
void GenericSolution<kDim, kM, kM0, kEfConstruction, kEfSearch>::SearchFP_(const std::vector<float>& query, int k, int* result_indices) {
  thread_local VisitedList visited;
  if (visited.tags.size() != num_points_) visited.resize(num_points_);
  visited.reset();

  const float* query_data = query.data();
  int current_ep = entry_point_;

  if (current_ep == -1) {
    for(int i = 0; i < k; ++i) result_indices[i] = -1;
    return;
  }

  float current_dist = Dist::L2Sq(query_data, data_ptr_ + current_ep * kDim);

  for (int level = max_level_; level > 0; --level) {
    bool changed = true;
    while (changed) {
      changed = false;
      if (level >= graph_data_[current_ep].size()) continue; // Check if level exists for current_ep
      const auto& neighbors = graph_data_[current_ep][level];
      for (int neighbor_id : neighbors) {
        float dist = Dist::L2Sq(query_data, data_ptr_ + neighbor_id * kDim);
        if (dist < current_dist) {
          current_dist = dist;
          current_ep = neighbor_id;
          changed = true;
        }
      }
    }
  }

  using QueueItem = std::pair<float, int>;
  StackHeap<QueueItem, kEfSearch + 1> top_candidates;
  StackHeap<QueueItem, kEfSearch * 2, std::greater<QueueItem>> candidates;

  top_candidates.push({current_dist, current_ep});
  candidates.push({current_dist, current_ep});
  visited.visit(current_ep);

  float best_dist_so_far = current_dist;

  while(!candidates.empty()) {
    auto [dist, id] = candidates.top();
    candidates.pop();

    if (dist > best_dist_so_far && top_candidates.size() >= (size_t)kEfSearch) {
      break;
    }

        if (0 >= graph_data_[id].size()) continue; // Ensure level 0 exists for node id
        const auto& neighbors = graph_data_[id][0];
    
        for (int i = 0; i < neighbors.size(); ++i) {
          int neighbor_id = neighbors[i];
          if (i + kPrefetchDis < neighbors.size()) {
            int next_neighbor = neighbors[i + kPrefetchDis];
            _mm_prefetch((const char*)(data_ptr_ + next_neighbor * kDim), _MM_HINT_T0);
            // Prefetch the tag!
            _mm_prefetch((const char*)&visited.tags[next_neighbor], _MM_HINT_T0);
          }
      if(!visited.visit(neighbor_id)) {
        float neighbor_dist = Dist::L2Sq(query_data, data_ptr_ + neighbor_id * kDim);
        if (top_candidates.size() < (size_t)kEfSearch || neighbor_dist < best_dist_so_far) {
          candidates.push({neighbor_dist, neighbor_id});
          top_candidates.push({neighbor_dist, neighbor_id});
          if (top_candidates.size() > (size_t)kEfSearch) top_candidates.pop();
          best_dist_so_far = top_candidates.top().first;
        }
      }
    }
  }

  size_t result_count = 0;
  std::vector<QueueItem> sorted_results;
  sorted_results.reserve(top_candidates.size());
  while(!top_candidates.empty()) {
    sorted_results.push_back(top_candidates.top());
    top_candidates.pop();
  }
  std::reverse(sorted_results.begin(), sorted_results.end());

  for (const auto& p : sorted_results) {
    if (result_count >= (size_t)k) break;
    result_indices[result_count++] = p.second;
  }
  while(result_count < (size_t)k) result_indices[result_count++] = -1;
}

template <int kDim, int kM, int kM0, int kEfConstruction, int kEfSearch>
void GenericSolution<kDim, kM, kM0, kEfConstruction, kEfSearch>::SearchAdaptiveFP_(const std::vector<float>& query, int k, int* result_indices) {
  thread_local VisitedList visited;
  if (visited.tags.size() != num_points_) visited.resize(num_points_);
  visited.reset();

  const float* query_data = query.data();
  int current_ep = entry_point_;

  if (current_ep == -1) {
    for(int i = 0; i < k; ++i) result_indices[i] = -1;
    return;
  }

  float current_dist = Dist::L2Sq(query_data, data_ptr_ + current_ep * kDim);

  for (int level = max_level_; level > 0; --level) {
    bool changed = true;
    while (changed) {
      changed = false;
      if (level >= graph_data_[current_ep].size()) continue; // Check if level exists for current_ep
      const auto& neighbors = graph_data_[current_ep][level];
      for (int neighbor_id : neighbors) {
        float dist = Dist::L2Sq(query_data, data_ptr_ + neighbor_id * kDim);
        if (dist < current_dist) {
          current_dist = dist;
          current_ep = neighbor_id;
          changed = true;
        }
      }
    }
  }

  using QueueItem = std::pair<float, int>;
  StackHeap<QueueItem, kEfSearch + 1> top_candidates;
  StackHeap<QueueItem, kEfSearch * 2, std::greater<QueueItem>> candidates;

  top_candidates.push({current_dist, current_ep});
  candidates.push({current_dist, current_ep});
  visited.visit(current_ep);

  size_t target_k = std::min((size_t)k, (size_t)kEfSearch);

  while(!candidates.empty()) {
    auto [dist, id] = candidates.top();
    candidates.pop();

    if (top_candidates.size() >= target_k && dist > top_candidates.top().first * kGamma) break;

    if (0 >= graph_data_[id].size()) continue; // Ensure level 0 exists for node id
    const auto& neighbors = graph_data_[id][0];

    for (int i = 0; i < neighbors.size(); ++i) {
      int neighbor_id = neighbors[i];
      if (i + kPrefetchDis < neighbors.size()) {
        int next_neighbor = neighbors[i + kPrefetchDis];
        _mm_prefetch((const char*)(data_ptr_ + next_neighbor * kDim), _MM_HINT_T0);
        _mm_prefetch((const char*)&visited.tags[next_neighbor], _MM_HINT_T0); 
      }

      if(!visited.visit(neighbor_id)) {
        float neighbor_dist = Dist::L2Sq(query_data, data_ptr_ + neighbor_id * kDim);
        
        if (top_candidates.size() < target_k || neighbor_dist < top_candidates.top().first * kGamma) {
           candidates.push({neighbor_dist, neighbor_id});
        }

        if (top_candidates.size() < target_k || neighbor_dist < top_candidates.top().first) {
          top_candidates.push({neighbor_dist, neighbor_id});
          if (top_candidates.size() > target_k) top_candidates.pop();
        }
      }
    }
  }

  size_t result_count = 0;
  std::vector<QueueItem> sorted_results;
  sorted_results.reserve(top_candidates.size());
  while(!top_candidates.empty()) {
    sorted_results.push_back(top_candidates.top());
    top_candidates.pop();
  }
  std::reverse(sorted_results.begin(), sorted_results.end());

  for (const auto& p : sorted_results) {
    if (result_count >= (size_t)k) break;
    result_indices[result_count++] = p.second;
  }
  while(result_count < (size_t)k) result_indices[result_count++] = -1;
}

} // namespace TestSolution

class Solution {
public:
  /**
   * Load & preprocessthe vector dataset.
   * @param d The number of dimensions for each vector.
   * @param base The dataset in with all vectors concatenated.
   */
  inline void build(int d, const std::vector<float>& base) {
    if (d == 128) {
      is_sift_ = true;
      sift_solution_.Build(base);
    } else {
      is_sift_ = false;
      glove_solution_.Build(base);
    }
  }
  /**
   * Search for the 10 closest vectors in the dataset.
   *
   * The standard for considering the distance between vectors is the **L2**
   * distance: \[||x - y||_2\]
   *
   * @param [in] query The vector to search for. This vector should be of `d`
   * dimensional.
   * @param [out] res The place to put the results. Memory is pre-allocated.
   */
  inline void search(const std::vector<float>& query, int* res) {
    if (is_sift_) sift_solution_.Search(query, 10, res);
    else glove_solution_.Search(query, 10, res);
  }
private:
  bool is_sift_;
  TestSolution::GenericSolution<
    128, // kDim
    32,  // kM
    64,  // kM0
    256, // kEfConstruction
    200  // kEfSearch
  > sift_solution_;
  TestSolution::GenericSolution<
    100, // kDim
    48,  // kM
    96,  // kM0
    512, // kEfConstruction
    600  // kEfSearch
  > glove_solution_;
};
