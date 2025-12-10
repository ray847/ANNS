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

#include "TunableHNSW/Distance.h"
#include "TunableHNSW/OPQ.h"

// This config header defines the structs for the different strategies
// to be tested by the final solution.
namespace MySolution {

// The strategies to test, defined as config structs.
// This combines the previous IntegratedConfig.h and the user's new hybrid config.

enum class QuantizationStrategy { kNone, kOPQ };
enum class SearchStrategy { kStandard, kDynamic };

struct SIFT_Final_Config {
    static constexpr int kDim = 128;
    static constexpr int kDimVal = 128;
    static constexpr int kM = 32;
    static constexpr int kM0 = 64;
    static constexpr int kEfConstruction = 500;
    
    static constexpr QuantizationStrategy kQuantizationVal = QuantizationStrategy::kNone;
    static constexpr SearchStrategy kSearchStrategyVal = SearchStrategy::kDynamic;
    
    static constexpr int kPatience = 100;
    static constexpr int kMaxEfSearch = 500;

    // Unused PQ params
    static constexpr int kPQSubquantizersVal = 0;
    static constexpr size_t kQuantizerTrainSampleSizeVal = 0;
};

struct GLOVE_Final_Config {
    static constexpr int kDim = 100;
    static constexpr int kDimVal = 100;
    static constexpr int kM = 48;
    static constexpr int kM0 = 96;
    static constexpr int kEfConstruction = 800;

    static constexpr QuantizationStrategy kQuantizationVal = QuantizationStrategy::kNone;
    static constexpr SearchStrategy kSearchStrategyVal = SearchStrategy::kDynamic;

    static constexpr int kPatience = 200;
    static constexpr int kMaxEfSearch = 2000;

    // Unused PQ params
    static constexpr int kPQSubquantizersVal = 0;
    static constexpr size_t kQuantizerTrainSampleSizeVal = 0;
};

struct GLOVE_OPQ_Hybrid_Config {
    static constexpr int kDim = 100;
    static constexpr int kDimVal = 100;
    static constexpr int kM = 48;
    static constexpr int kM0 = 96;
    static constexpr int kEfConstruction = 800;

    static constexpr QuantizationStrategy kQuantizationVal = QuantizationStrategy::kOPQ;
    static constexpr SearchStrategy kSearchStrategyVal = SearchStrategy::kDynamic;
    
    // OPQ Params
    static constexpr int kPQSubquantizersVal = 20; 
    static constexpr size_t kQuantizerTrainSampleSizeVal = 25000;

    // Dynamic Search Params for noisy graph
    static constexpr int kPatience = 250;
    static constexpr int kMaxEfSearch = 1600;
};

} // namespace MySolution


// The final, integrated HNSW implementation
namespace MySolution {

template <typename Config>
class HNSW {
 public:
  HNSW();

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

  using Dist = TunableHNSW::Distance<Config::kDim, true>;
  using PQ = std::conditional_t<
      Config::kQuantizationVal == QuantizationStrategy::kOPQ,
      TunableHNSW::OptimizedProductQuantizer<Config>,
      std::nullptr_t>;

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
  
  void SearchFP_(const std::vector<float>& query, int k, int* result_indices);
  void SearchPQ_(const std::vector<float>& query, int k, int* result_indices);
};

// --- Implementation ---



template <typename Config>

HNSW<Config>::HNSW() : level_mult_(1.0 / std::log(1.0 * Config::kM)) {

  if constexpr (Config::kQuantizationVal != QuantizationStrategy::kNone) {

    pq_.emplace();

  }

}



template <typename Config>

void HNSW<Config>::Build(const std::vector<float>& base_data) {

  data_storage_ = base_data;

  data_ptr_ = data_storage_.data();

  num_points_ = base_data.size() / Config::kDim;

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



  if constexpr (Config::kQuantizationVal != QuantizationStrategy::kNone) {

      pq_->Train(data_ptr_, num_points_);

      pq_codes_.resize(num_points_ * Config::kPQSubquantizersVal);

      auto encode_worker = [&](size_t start, size_t end) {

          for(size_t i = start; i < end; ++i) {

              auto code = pq_->Encode(data_ptr_ + i * Config::kDim);

              std::copy(code.begin(), code.end(), pq_codes_.data() + i * Config::kPQSubquantizersVal);

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

}



template <typename Config>

void HNSW<Config>::Search(const std::vector<float>& query, int k, int* result_indices) {

    if constexpr (Config::kQuantizationVal != QuantizationStrategy::kNone) {

        SearchPQ_(query, k, result_indices);

    } else {

        SearchFP_(query, k, result_indices);

    }

}



template <typename Config>

int HNSW<Config>::GetRandomLevel_(std::mt19937& rng) {

  std::uniform_real_distribution<double> dist(0.0, 1.0);

  return static_cast<int>(-std::log(dist(rng)) * level_mult_);

}



template <typename Config>

void HNSW<Config>::Insert_(int node_id,

                           std::vector<std::vector<std::vector<int>>>& temp_graph,

                           std::vector<std::mutex>& locks) {

  

  thread_local VisitedList visited;

  if(visited.tags.size() != num_points_) visited.resize(num_points_);

  

  const float* query_vector = data_ptr_ + node_id * Config::kDim;

  int current_ep = entry_point_;

  int node_level = nodes_[node_id].level;

  

  if (current_ep == -1) return;



  for (int level = max_level_; level > node_level; --level) {

    bool changed = true;

    while(changed) {

        changed = false;

        float min_dist = Dist::L2Sq(query_vector, data_ptr_ + current_ep * Config::kDim);

        std::lock_guard<std::mutex> lock(locks[current_ep]);

        if (level >= temp_graph[current_ep].size()) continue;

        const auto& neighbors = temp_graph[current_ep][level];

        for (int neighbor_id : neighbors) {

            float dist = Dist::L2Sq(query_vector, data_ptr_ + neighbor_id * Config::kDim);

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

    auto top_candidates = SearchLayerFP_(query_vector, current_ep, Config::kEfConstruction, level, temp_graph, locks, visited);

    

    std::vector<int> neighbors;

    neighbors.reserve(Config::kM0); // Reserve max possible

    SelectNeighbors_(top_candidates, neighbors);

    

    {

        std::lock_guard<std::mutex> lock(locks[node_id]);

        temp_graph[node_id][level] = neighbors;

    }



    for (int neighbor_id : neighbors) {

      std::lock_guard<std::mutex> lock(locks[neighbor_id]);

      if (level >= temp_graph[neighbor_id].size()) continue;

      auto& neighbor_links = temp_graph[neighbor_id][level];

      int neighbor_M = (level == 0) ? Config::kM0 : Config::kM;

      if (neighbor_links.size() < (size_t)neighbor_M) {

        neighbor_links.push_back(node_id);

      } else {

        float new_node_dist = Dist::L2Sq(data_ptr_ + neighbor_id * Config::kDim, query_vector);

        std::priority_queue<std::pair<float, int>> temp_pq;

        for(int link : neighbor_links) {

            temp_pq.push({Dist::L2Sq(data_ptr_ + neighbor_id * Config::kDim, data_ptr_ + link * Config::kDim), link});

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



template <typename Config>

std::priority_queue<std::pair<float, int>> HNSW<Config>::SearchLayerFP_(

    const float* query, int entry_point, int ef, int level,

    const std::vector<std::vector<std::vector<int>>>& temp_graph,

    std::vector<std::mutex>& locks, VisitedList& visited) {

  

  using QueueItem = std::pair<float, int>;

  std::priority_queue<QueueItem> top_results;

  std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> candidates;



  float initial_dist = Dist::L2Sq(query, data_ptr_ + entry_point * Config::kDim);

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

        float neighbor_dist = Dist::L2Sq(query, data_ptr_ + neighbor_id * Config::kDim);

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



template <typename Config>

void HNSW<Config>::SelectNeighbors_(

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

    if (target_list.size() >= (size_t)Config::kM) break;

    bool is_good = true;

    for (int selected_neighbor : target_list) {

      if (Dist::L2Sq(data_ptr_ + cand.second * Config::kDim, data_ptr_ + selected_neighbor * Config::kDim) < cand.first) {

        is_good = false;

        break;

      }

    }

    if (is_good) {

      target_list.push_back(cand.second);

    }

  }

}



template <typename Config>

void HNSW<Config>::FlattenGraph_(const std::vector<std::vector<std::vector<int>>>& temp_graph) {

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



template<typename Config>

void HNSW<Config>::SearchFP_(const std::vector<float>& query, int k, int* result_indices) {

  thread_local VisitedList visited;

  if (visited.tags.size() != num_points_) visited.resize(num_points_);

  visited.reset();



  const float* query_data = query.data();

  int current_ep = entry_point_;



  if (current_ep == -1) {

    for(int i = 0; i < k; ++i) result_indices[i] = -1;

    return;

  }



  float current_dist = Dist::L2Sq(query_data, data_ptr_ + current_ep * Config::kDim);



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

        float dist = Dist::L2Sq(query_data, data_ptr_ + neighbor_id * Config::kDim);

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



  int patience = Config::kPatience;

  float best_dist_so_far = current_dist;



  while(!candidates.empty()) {

    auto [dist, id] = candidates.top();

    candidates.pop();



    if (dist > best_dist_so_far && top_candidates.size() >= (size_t)Config::kMaxEfSearch) {

        if (--patience == 0) break;

    }

    

    size_t node_offset = nodes_[id].offset;

    const int* neighbors = flat_graph_.data() + node_offset;

    int count = link_counts_[id * (max_level_ + 1) + 0];



    for (int i = 0; i < count; ++i) {

        int neighbor_id = neighbors[i];

        if (i + 4 < count) _mm_prefetch((const char*)(data_ptr_ + neighbors[i+4] * Config::kDim), _MM_HINT_T0);

        

        if(!visited.visit(neighbor_id)) {

            float neighbor_dist = Dist::L2Sq(query_data, data_ptr_ + neighbor_id * Config::kDim);

            if (top_candidates.size() < (size_t)Config::kMaxEfSearch || neighbor_dist < best_dist_so_far) {

                candidates.push({neighbor_dist, neighbor_id});

                top_candidates.push({neighbor_dist, neighbor_id});

                if (top_candidates.size() > (size_t)Config::kMaxEfSearch) top_candidates.pop();

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

  while(result_count < (size_t)k) {

    result_indices[result_count++] = -1;

  }

}



template<typename Config>

void HNSW<Config>::SearchPQ_(const std::vector<float>& query, int k, int* result_indices) {

    thread_local VisitedList visited;

    if(visited.tags.size() != num_points_) visited.resize(num_points_);

    visited.reset();



    const float* query_data = query.data();

    auto dist_table = pq_->BuildDistanceTable(query_data);

    auto query_dist_sq_pq = [&](int node_id) {

        const uint8_t* code = pq_codes_.data() + node_id * Config::kPQSubquantizersVal;

        return pq_->GetDistanceFromTable(dist_table, code);

    };



    int current_ep = entry_point_;

    if(current_ep == -1) { /* ... handle empty ... */ return; }



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



    int patience = Config::kPatience;

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

            if (i + 4 < count) _mm_prefetch((const char*)(pq_codes_.data() + neighbors[i+4] * Config::kPQSubquantizersVal), _MM_HINT_T0);

            if(!visited.visit(neighbor_id)) {

                float neighbor_dist = query_dist_sq_pq(neighbor_id);

                if (top_candidates.size() < (size_t)Config::kMaxEfSearch || neighbor_dist < best_dist_so_far) {

                    candidates.push({neighbor_dist, neighbor_id});

                    top_candidates.push({neighbor_dist, neighbor_id});

                    if (top_candidates.size() > (size_t)Config::kMaxEfSearch) top_candidates.pop();

                    best_dist_so_far = top_candidates.top().first;

                }

            }

        }

    }

    

    // Reranking Step

    std::vector<QueueItem> candidates_to_rerank;

    candidates_to_rerank.reserve(top_candidates.size());

    while(!top_candidates.empty()) {

        candidates_to_rerank.push_back(top_candidates.top());

        top_candidates.pop();

    }

    

    for (const auto& item : candidates_to_rerank) {

        float exact_dist = Dist::L2Sq(query_data, data_ptr_ + item.second * Config::kDim);

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
