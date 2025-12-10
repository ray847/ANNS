#include "TestSolution.h"

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

Solution::Solution() : level_mult_(1.0 / std::log(1.0 * kM)) {}

void Solution::Build(const std::vector<float>& base_data) {
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
  FlattenGraph_(temp_graph);
}

void Solution::Search(const std::vector<float>& query, int k, int* result_indices) {
    SearchFP_(query, k, result_indices);
}

int Solution::GetRandomLevel_(std::mt19937& rng) {
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  return static_cast<int>(-std::log(dist(rng)) * level_mult_);
}

void Solution::Insert_(int node_id,
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
        std::priority_queue<std::pair<float, int>> temp_pq;
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

std::priority_queue<std::pair<float, int>> Solution::SearchLayerFP_(
    const float* query, int entry_point, int ef, int level,
    const std::vector<std::vector<std::vector<int>>>& temp_graph,
    std::vector<std::mutex>& locks, VisitedList& visited) {
  
  using QueueItem = std::pair<float, int>;
  std::priority_queue<QueueItem> top_results;
  std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> candidates;

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

void Solution::SelectNeighbors_(
    std::priority_queue<std::pair<float, int>>& candidates,
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

void Solution::FlattenGraph_(const std::vector<std::vector<std::vector<int>>>& temp_graph) {
  link_counts_.resize(num_points_ * (max_level_ + 1), 0);
  size_t total_links = 0;
  for (size_t i = 0; i < num_points_; ++i) {
    if(nodes_[i].level > max_level_) continue;
    for (int level = 0; level <= nodes_[i].level; ++level) {
      if (level < temp_graph[i].size()) total_links += temp_graph[i][level].size();
    }
  }
  
  flat_graph_.resize(total_links);
  size_t current_offset = 0;

  for (size_t i = 0; i < num_points_; ++i) {
    nodes_[i].offset = current_offset;
    if(nodes_[i].level > max_level_) continue;
    for (int level = 0; level <= nodes_[i].level; ++level) {
      if (level < temp_graph[i].size()) {
        const auto& neighbors = temp_graph[i][level];
        link_counts_[i * (max_level_ + 1) + level] = neighbors.size();
        for (int neighbor : neighbors) flat_graph_[current_offset++] = neighbor;
      }
    }
  }
}

void Solution::SearchFP_(const std::vector<float>& query, int k, int* result_indices) {
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
      size_t node_offset = nodes_[current_ep].offset;
      int link_offset_level = 0;
      for (int i = 0; i < level; ++i) link_offset_level += link_counts_[current_ep * (max_level_ + 1) + i];
      const int* neighbors = flat_graph_.data() + node_offset + link_offset_level;
      int count = link_counts_[current_ep * (max_level_ + 1) + level];

      for (int i = 0; i < count; ++i) {
        int neighbor_id = neighbors[i];
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
  std::priority_queue<QueueItem> top_candidates;
  std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> candidates;

  top_candidates.push({current_dist, current_ep});
  candidates.push({current_dist, current_ep});
  visited.visit(current_ep);

  int patience = kPatience;
  float best_dist_so_far = current_dist;

  while(!candidates.empty()) {
    auto [dist, id] = candidates.top();
    candidates.pop();

    if (dist > best_dist_so_far && top_candidates.size() >= (size_t)kMaxEfSearch) {
        if (--patience == 0) break;
    }
    
    size_t node_offset = nodes_[id].offset;
    const int* neighbors = flat_graph_.data() + node_offset;
    int count = link_counts_[id * (max_level_ + 1) + 0];

    for (int i = 0; i < count; ++i) {
        int neighbor_id = neighbors[i];
        if (i + 4 < count) _mm_prefetch((const char*)(data_ptr_ + neighbors[i+4] * kDim), _MM_HINT_T0);
        
        if(!visited.visit(neighbor_id)) {
            float neighbor_dist = Dist::L2Sq(query_data, data_ptr_ + neighbor_id * kDim);
            if (top_candidates.size() < (size_t)kMaxEfSearch || neighbor_dist < best_dist_so_far) {
                candidates.push({neighbor_dist, neighbor_id});
                top_candidates.push({neighbor_dist, neighbor_id});
                if (top_candidates.size() > (size_t)kMaxEfSearch) top_candidates.pop();
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

} // namespace Sift

namespace Glove {

// KMeans implementation
KMeans::KMeans(int num_clusters, int num_iterations)
    : num_clusters_(num_clusters), num_iterations_(num_iterations) {}

std::vector<float> KMeans::Train(const float* data, int num_points, int dim,
                         int stride) {
    std::vector<float> centroids(num_clusters_ * dim);
    InitializeCentroids_(data, num_points, dim, stride, &centroids);

    std::vector<int> assignments(num_points);
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    std::vector<std::thread> threads;

    for (int iter = 0; iter < num_iterations_; ++iter) {
      auto assignment_worker = [&](int start, int end) {
        for (int i = start; i < end; ++i) {
          float min_dist_sq = FLT_MAX;
          int best_cluster_id = -1;
          for (int j = 0; j < num_clusters_; ++j) {
            float dist_sq =
                L2SqSlice_(data + i * stride, centroids.data() + j * dim, dim);
            if (dist_sq < min_dist_sq) {
              min_dist_sq = dist_sq;
              best_cluster_id = j;
            }
          }
          assignments[i] = best_cluster_id;
        }
      };

      threads.clear();
      threads.reserve(num_threads);
      int block_size = num_points / num_threads;
      for (unsigned int i = 0; i < num_threads; ++i) {
          int start = i * block_size;
          int end = (i == num_threads - 1) ? num_points : start + block_size;
          threads.emplace_back(assignment_worker, start, end);
      }
      for (auto& t : threads) {
          t.join();
      }

      std::vector<float> new_centroids(num_clusters_ * dim, 0.0f);
      std::vector<int> cluster_counts(num_clusters_, 0);
      std::vector<std::vector<float>> thread_new_centroids(num_threads, std::vector<float>(num_clusters_ * dim, 0.0f));
      std::vector<std::vector<int>> thread_cluster_counts(num_threads, std::vector<int>(num_clusters_, 0));

      auto update_worker = [&](int thread_id, int start, int end) {
        for (int i = start; i < end; ++i) {
          int cluster_id = assignments[i];
          if (cluster_id != -1) {
            thread_cluster_counts[thread_id][cluster_id]++;
            for (int d = 0; d < dim; ++d) {
                thread_new_centroids[thread_id][cluster_id * dim + d] += data[i * stride + d];
            }
          }
        }
      };
    
      threads.clear();
      for (unsigned int i = 0; i < num_threads; ++i) {
          int start = i * block_size;
          int end = (i == num_threads - 1) ? num_points : start + block_size;
          threads.emplace_back(update_worker, i, start, end);
      }
      for (auto& t : threads) {
          t.join();
      }

      for (unsigned int i = 0; i < num_threads; ++i) {
        for (int j = 0; j < num_clusters_; ++j) {
            cluster_counts[j] += thread_cluster_counts[i][j];
            for (int d = 0; d < dim; ++d) {
                new_centroids[j * dim + d] += thread_new_centroids[i][j * dim + d];
            }
        }
      }

      for (int j = 0; j < num_clusters_; ++j) {
        if (cluster_counts[j] > 0) {
          for (int d = 0; d < dim; ++d) {
            new_centroids[j * dim + d] /= cluster_counts[j];
          }
        } else {
          int rand_idx =
              std::uniform_int_distribution<int>(0, num_points - 1)(global::rng);
          const float* random_point = data + rand_idx * stride;
          std::copy(random_point, random_point + dim,
                    new_centroids.data() + j * dim);
        }
      }
      centroids = new_centroids;
    }
    return centroids;
  }

void KMeans::InitializeCentroids_(const float* data, int num_points, int dim,
                            int stride, std::vector<float>* centroids) {
    std::vector<int> indices(num_points);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), global::rng);

    for (int i = 0; i < num_clusters_; ++i) {
      const float* src = data + indices[i] * stride;
      float* dst = centroids->data() + i * dim;
      std::copy(src, src + dim, dst);
    }
  }

float KMeans::L2SqSlice_(const float* vector_a, const float* vector_b, int dim) {
    float result = 0.0f;
    for (int i = 0; i < dim; ++i) {
      float diff = vector_a[i] - vector_b[i];
      result += diff * diff;
    }
    return result;
}


// OptimizedProductQuantizer implementation
OptimizedProductQuantizer::OptimizedProductQuantizer() : R_(Eigen::MatrixXf::Identity(Solution::kDim, Solution::kDim)) {}

void OptimizedProductQuantizer::Train(const float* data, int num_points) {
    using MatrixXfRow = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    
    Eigen::Map<const MatrixXfRow> full_dataset(data, num_points, Solution::kDim);
    MatrixXfRow train_dataset;

    constexpr size_t sample_size = Solution::kOPQTrainSampleSize;

    if (sample_size > 0 && sample_size < num_points) {
        train_dataset.resize(sample_size, Solution::kDim);
        std::vector<int> indices(num_points);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), global::rng);
        for (size_t i = 0; i < sample_size; ++i) {
            train_dataset.row(i) = full_dataset.row(indices[i]);
        }
    } else {
        train_dataset = full_dataset;
    }

    const int num_iterations = 10;
    for (int i = 0; i < num_iterations; ++i) {
      MatrixXfRow rotated_data = train_dataset * R_.transpose();

      TrainCodebooks(rotated_data.data(), train_dataset.rows());

      MatrixXfRow C(train_dataset.rows(), Solution::kDim);
      
      auto worker = [&](int start, int end) {
        for (int j = start; j < end; ++j) {
                    const float* vec = rotated_data.row(j).data();
                  
                      for (int m = 0; m < Solution::kPQSubquantizers; ++m) {             const float* sub_vec = vec + m * (Solution::kDim / Solution::kPQSubquantizers);
             const float* codebook = codebooks_.data() + m * 256 * (Solution::kDim / Solution::kPQSubquantizers);
             
             int best_id = 0;
             float min_dist = FLT_MAX;
             
             for (int k = 0; k < 256; ++k) {
                 float d = L2SqSlice_(sub_vec, codebook + k * (Solution::kDim / Solution::kPQSubquantizers), (Solution::kDim / Solution::kPQSubquantizers));
                 if(d < min_dist) { min_dist = d; best_id = k; }
             }

             const float* best_centroid = codebook + best_id * (Solution::kDim / Solution::kPQSubquantizers);
             for(int d=0; d<(Solution::kDim / Solution::kPQSubquantizers); ++d) {
                 C(j, m * (Solution::kDim / Solution::kPQSubquantizers) + d) = best_centroid[d];
             }
          }
        }
      };

      unsigned int num_threads = std::thread::hardware_concurrency();
      if (num_threads == 0) num_threads = 4;
      std::vector<std::thread> threads;
      threads.reserve(num_threads);
      int block_size = train_dataset.rows() / num_threads;
      for (unsigned int i = 0; i < num_threads; ++i) {
          int start = i * block_size;
          int end = (i == num_threads - 1) ? train_dataset.rows() : start + block_size;
          threads.emplace_back(worker, start, end);
      }

      for (auto& t : threads) {
          t.join();
      }

      Eigen::MatrixXf M = train_dataset.transpose() * C;
      Eigen::JacobiSVD<Eigen::MatrixXf> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
      
      R_ = svd.matrixV() * svd.matrixU().transpose();
    }
}

std::vector<uint8_t> OptimizedProductQuantizer::Encode(const float* vector) const {
    Eigen::Map<const Eigen::Matrix<float, Solution::kDim, 1>> v(vector);
    Eigen::Matrix<float, Solution::kDim, 1> rotated_v = R_ * v;
    
    std::vector<uint8_t> code(Solution::kPQSubquantizers);
    for (int m = 0; m < Solution::kPQSubquantizers; ++m) {
      const float* sub_vector = rotated_v.data() + m * (Solution::kDim / Solution::kPQSubquantizers);
      const float* codebook =
          codebooks_.data() + m * 256 * (Solution::kDim / Solution::kPQSubquantizers);

      float min_dist_sq = FLT_MAX;
      uint8_t best_centroid_id = 0;
      for (int k = 0; k < 256; ++k) {
        float dist_sq =
            L2SqSlice_(sub_vector, codebook + k * (Solution::kDim / Solution::kPQSubquantizers), (Solution::kDim / Solution::kPQSubquantizers));
        if (dist_sq < min_dist_sq) {
          min_dist_sq = dist_sq;
          best_centroid_id = k;
        }
      }
      code[m] = best_centroid_id;
    }
    return code;
}

std::vector<float> OptimizedProductQuantizer::BuildDistanceTable(const float* query) const {
    Eigen::Map<const Eigen::Matrix<float, Solution::kDim, 1>> q(query);
    Eigen::Matrix<float, Solution::kDim, 1> rotated_q = R_ * q;

    std::vector<float> table(Solution::kPQSubquantizers * 256);
    for (int m = 0; m < Solution::kPQSubquantizers; ++m) {
      const float* query_sub_vector = rotated_q.data() + m * (Solution::kDim / Solution::kPQSubquantizers);
      const float* codebook =
          codebooks_.data() + m * 256 * (Solution::kDim / Solution::kPQSubquantizers);
      for (int k = 0; k < 256; ++k) {
        table[m * 256 + k] = L2SqSlice_(
            query_sub_vector, codebook + k * (Solution::kDim / Solution::kPQSubquantizers), (Solution::kDim / Solution::kPQSubquantizers));
      }
    }
    return table;
}

float OptimizedProductQuantizer::GetDistanceFromTable(const std::vector<float>& table,
                             const uint8_t* code) const {
    float dist = 0.0f;
    for (int m = 0; m < Solution::kPQSubquantizers; ++m) {
      dist += table[m * 256 + code[m]];
    }
    return dist;
}

void OptimizedProductQuantizer::TrainCodebooks(const float* data, int num_points) {
    codebooks_.resize(Solution::kPQSubquantizers * 256 * (Solution::kDim / Solution::kPQSubquantizers));
    std::vector<std::thread> threads;
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;

    auto train_kmeans = [&](int m) {
        KMeans kmeans(256, 20);
        const float* sub_data_start = data + m * (Solution::kDim / Solution::kPQSubquantizers);
        auto sub_centroids =
            kmeans.Train(sub_data_start, num_points, (Solution::kDim / Solution::kPQSubquantizers), Solution::kDim);
        float* codebook_dst =
            codebooks_.data() + m * 256 * (Solution::kDim / Solution::kPQSubquantizers);
        std::copy(sub_centroids.begin(), sub_centroids.end(), codebook_dst);
            };
    
        for (int m = 0; m < Solution::kPQSubquantizers; ++m) {        threads.emplace_back(train_kmeans, m);
        if (threads.size() >= num_threads) {
            for (auto& t : threads) t.join();
            threads.clear();
        }
    }
    for (auto& t : threads) t.join();
}

float OptimizedProductQuantizer::L2SqSlice_(const float* vector_a, const float* vector_b, int dim) const {
    float result = 0.0f;
    for (int i = 0; i < dim; ++i) {
      float diff = vector_a[i] - vector_b[i];
      result += diff * diff;
    }
    return result;
}


// Glove Solution implementation
Solution::Solution() : level_mult_(1.0 / std::log(1.0 * kM)) {
  pq_.emplace();
}

void Solution::Build(const std::vector<float>& base_data) {
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

  FlattenGraph_(temp_graph);

  pq_->Train(data_ptr_, num_points_);
  pq_codes_.resize(num_points_ * kPQSubquantizers);
  auto encode_worker = [&](size_t start, size_t end) {
      for(size_t i = start; i < end; ++i) {
          auto code = pq_->Encode(data_ptr_ + i * kDim);
          std::copy(code.begin(), code.end(), pq_codes_.data() + i * kPQSubquantizers);
      }
  };
  threads.clear();
  size_t block_size = num_points_ / num_threads;
  for (unsigned int i = 0; i < num_threads; ++i) {
      size_t start = i * block_size;
      size_t end = (i == num_threads - 1) ? num_points_ : start + block_size;
      threads.emplace_back(encode_worker, start, end);
  }
  for(auto& t : threads) t.join();
}

void Solution::Search(const std::vector<float>& query, int k, int* result_indices) {
    SearchPQ_(query, k, result_indices);
}

int Solution::GetRandomLevel_(std::mt19937& rng) {
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  return static_cast<int>(-std::log(dist(rng)) * level_mult_);
}

void Solution::Insert_(int node_id,
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
        std::priority_queue<std::pair<float, int>> temp_pq;
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

std::priority_queue<std::pair<float, int>> Solution::SearchLayerFP_(
    const float* query, int entry_point, int ef, int level,
    const std::vector<std::vector<std::vector<int>>>& temp_graph,
    std::vector<std::mutex>& locks, VisitedList& visited) {
  
  using QueueItem = std::pair<float, int>;
  std::priority_queue<QueueItem> top_results;
  std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> candidates;

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

void Solution::SelectNeighbors_(
    std::priority_queue<std::pair<float, int>>& candidates,
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

void Solution::FlattenGraph_(const std::vector<std::vector<std::vector<int>>>& temp_graph) {
  link_counts_.resize(num_points_ * (max_level_ + 1), 0);
  size_t total_links = 0;
  for (size_t i = 0; i < num_points_; ++i) {
    if(nodes_[i].level > max_level_) continue;
    for (int level = 0; level <= nodes_[i].level; ++level) {
      if (level < temp_graph[i].size()) total_links += temp_graph[i][level].size();
    }
  }
  
  flat_graph_.resize(total_links);
  size_t current_offset = 0;

  for (size_t i = 0; i < num_points_; ++i) {
    nodes_[i].offset = current_offset;
    if(nodes_[i].level > max_level_) continue;
    for (int level = 0; level <= nodes_[i].level; ++level) {
      if (level < temp_graph[i].size()) {
        const auto& neighbors = temp_graph[i][level];
        link_counts_[i * (max_level_ + 1) + level] = neighbors.size();
        for (int neighbor : neighbors) flat_graph_[current_offset++] = neighbor;
      }
    }
  }
}

void Solution::SearchPQ_(const std::vector<float>& query, int k, int* result_indices) {
    thread_local VisitedList visited;
    if(visited.tags.size() != num_points_) visited.resize(num_points_);
    visited.reset();

    const float* query_data = query.data();
    auto dist_table = pq_->BuildDistanceTable(query_data);
    auto query_dist_sq_pq = [&](int node_id) {
        const uint8_t* code = pq_codes_.data() + node_id * kPQSubquantizers;
        return pq_->GetDistanceFromTable(dist_table, code);
    };

    int current_ep = entry_point_;
    if(current_ep == -1) { return; }

    float current_dist = query_dist_sq_pq(current_ep);

    for (int level = max_level_; level > 0; --level) {
        bool changed = true;
        while(changed) {
            changed = false;
            size_t node_offset = nodes_[current_ep].offset;
            int link_offset_level = 0;
            for(int i=0; i<level; ++i) link_offset_level += link_counts_[current_ep * (max_level_ + 1) + i];
            const int* neighbors = flat_graph_.data() + node_offset + link_offset_level;
            int count = link_counts_[current_ep * (max_level_ + 1) + level];
            for (int i = 0; i < count; ++i) {
                if (neighbors[i] < 0 || (size_t)neighbors[i] >= num_points_) continue;
                float dist = query_dist_sq_pq(neighbors[i]);
                if (dist < current_dist) {
                    current_dist = dist;
                    current_ep = neighbors[i];
                    changed = true;
                }
            }
        }
    }

    using QueueItem = std::pair<float, int>;
    std::priority_queue<QueueItem> top_candidates;
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> candidates;

    top_candidates.push({current_dist, current_ep});
    candidates.push({current_dist, current_ep});
    visited.visit(current_ep);

    int patience = kPatience;
    float best_dist_so_far = current_dist;

    while(!candidates.empty()) {
        auto [dist, id] = candidates.top();
        candidates.pop();
        if (dist > best_dist_so_far) {
            if (--patience == 0) break;
        }
        
        size_t node_offset = nodes_[id].offset;
        const int* neighbors = flat_graph_.data() + node_offset;
        int count = link_counts_[id * (max_level_ + 1) + 0];

        for (int i = 0; i < count; ++i) {
            int neighbor_id = neighbors[i];
            if (i + 4 < count) _mm_prefetch((const char*)(pq_codes_.data() + neighbors[i+4] * kPQSubquantizers), _MM_HINT_T0);
            if(!visited.visit(neighbor_id)) {
                float neighbor_dist = query_dist_sq_pq(neighbor_id);
                if (top_candidates.size() < (size_t)kMaxEfSearch || neighbor_dist < best_dist_so_far) {
                    candidates.push({neighbor_dist, neighbor_id});
                    top_candidates.push({neighbor_dist, neighbor_id});
                    if (top_candidates.size() > (size_t)kMaxEfSearch) top_candidates.pop();
                    best_dist_so_far = top_candidates.top().first;
                }
            }
        }
    }
    
    std::vector<QueueItem> candidates_to_rerank;
    candidates_to_rerank.reserve(top_candidates.size());
    while(!top_candidates.empty()) {
        candidates_to_rerank.push_back(top_candidates.top());
        top_candidates.pop();
    }
    
    for (const auto& item : candidates_to_rerank) {
        float exact_dist = Dist::L2Sq(query_data, data_ptr_ + item.second * kDim);
        if (top_candidates.size() < (size_t)k || exact_dist < top_candidates.top().first) {
            top_candidates.push({exact_dist, item.second});
            if (top_candidates.size() > (size_t)k) top_candidates.pop();
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

} // namespace Glove

} // namespace FinalSolution
