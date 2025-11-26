#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <queue>
#include <cmath>
#include <cstring>
#include <limits>

class Solution {
public:
  void build(int d, const std::vector<float>& base);
  void search(const std::vector<float>& query, int* res);

private:
  // --- Optimized Parameters for GloVe ---
  // M=32 offers better recall for high-dim data (100d+) than 24
  static constexpr int M = 32;               
  static constexpr int M0 = M * 2;           
  static constexpr int ef_construction = 500; 
  static constexpr int ef_search = 800;       
  static constexpr int MAX_LEVEL = 32;       

  int d_ = 0;
  size_t n_ = 0;
  std::vector<float> data_;                 
  std::vector<int> levels_;
  // Flattening this vector in the future would provide further speedups, 
  // but keeping your structure for now.
  std::vector<std::vector<std::vector<int>>> graph_; 

  int entry_point_ = -1;
  int max_level_ = -1;
  double level_mult_;
  std::mt19937 rng_{42};

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
  } visited_;

  // OPTIMIZATION: Removed sqrt. 
  // Squared L2 preserves relative order (a < b iff a^2 < b^2) but is faster.
  inline float dist_func_sq(const float* a, const float* b, int d) {
    float res = 0;
    // Manual unrolling or SIMD could go here, but compilers usually 
    // auto-vectorize this loop well if d is known or simple.
    for (int i = 0; i < d; ++i) {
      float diff = a[i] - b[i];
      res += diff * diff;
    }
    return res;
  }

  inline float dist_sq(int id_a, int id_b) {
    return dist_func_sq(&data_[id_a * d_], &data_[id_b * d_], d_);
  }

  inline float dist_query_sq(const std::vector<float>& query, int id_node) {
    return dist_func_sq(query.data(), &data_[id_node * d_], d_);
  }

  int get_random_level() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = -std::log(dist(rng_)) * level_mult_;
    return std::min(static_cast<int>(r), MAX_LEVEL);
  }

  std::priority_queue<std::pair<float, int>> search_layer(
    const float* query_data, 
    int entry_point, 
    int ef, 
    int level
  ) {
    using QueueItem = std::pair<float, int>;
    std::priority_queue<QueueItem> top_candidates; // max-heap (stores ef closest)
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> candidates; // min-heap (exploration queue)

    visited_.advance();

    float initial_dist = dist_func_sq(query_data, &data_[entry_point * d_], d_);
    
    top_candidates.push({initial_dist, entry_point});
    candidates.push({initial_dist, entry_point});
    visited_.visit(entry_point);

    while (!candidates.empty()) {
      auto [curr_dist, curr_id] = candidates.top();
      candidates.pop();

      // CRITICAL OPTIMIZATION: 'break' instead of 'continue'.
      // If the closest candidate in the queue is farther than our worst 'ef' result,
      // we can stop safely because all subsequent candidates in min-heap are also farther.
      if (curr_dist > top_candidates.top().first && top_candidates.size() >= ef) {
        break; 
      }

      const auto& neighbors = graph_[curr_id][level];
      
      // OPTIMIZATION: Prefetch memory for the neighbor list
      // This brings the neighbor IDs into cache before we iterate
      #ifdef __GNUC__
        __builtin_prefetch(neighbors.data(), 0, 3);
      #endif

      for (int neighbor : neighbors) {
        if (!visited_.visit(neighbor)) {
          // OPTIMIZATION: Prefetch the actual vector data for this neighbor
          #ifdef __GNUC__
            __builtin_prefetch(&data_[neighbor * d_], 0, 0);
          #endif

          float d = dist_func_sq(query_data, &data_[neighbor * d_], d_);
          
          if (top_candidates.size() < ef || d < top_candidates.top().first) {
            candidates.push({d, neighbor});
            top_candidates.push({d, neighbor});
            
            if (top_candidates.size() > ef) {
              top_candidates.pop();
            }
          }
        }
      }
    }
    return top_candidates;
  }

  void get_neighbors_heuristic(int src, std::vector<std::pair<float, int>>& candidates, int level) {
    int max_m = (level == 0) ? M0 : M;
    
    std::sort(candidates.begin(), candidates.end());

    auto& neighbors = graph_[src][level];
    neighbors.clear();
    neighbors.reserve(max_m);

    for (size_t i = 0; i < candidates.size() && neighbors.size() < max_m; ++i) {
      int candidate_id = candidates[i].second;
      float dist_to_src = candidates[i].first;
      
      bool good = true;
      for (int existing_neighbor : neighbors) {
        float d_neighbor = dist_sq(candidate_id, existing_neighbor);
        
        // REVERTED TO STRICT HEURISTIC:
        // Removing the 0.8 factor ensures better graph navigation properties (Triangle Inequality).
        // While 0.8 keeps more edges, strict checking usually yields better recall at high EF.
        if (d_neighbor < dist_to_src) {
          good = false;
          break;
        }
      }
      if (good) {
        neighbors.push_back(candidate_id);
      }
    }
    
    // Backfill if heuristic pruned too much (keeping connectivity)
    // Removed the "min(3, max_m)" check to allow fuller graphs
    if (neighbors.size() < max_m) { 
        for (size_t i = 0; i < candidates.size() && neighbors.size() < max_m; ++i) {
            int candidate_id = candidates[i].second;
            bool found = false;
            for(int existing : neighbors) {
                if(existing == candidate_id) { found = true; break; }
            }
            if (!found) {
                neighbors.push_back(candidate_id);
            }
        }
    }
  }

  void add_connection(int src, int dest, int level) {
    auto& neighbors = graph_[src][level];
    for (int n : neighbors) {
      if (n == dest) return;
    }

    neighbors.push_back(dest);

    int max_m = (level == 0) ? M0 : M;
    if (neighbors.size() > max_m) {
      std::vector<std::pair<float, int>> candidates;
      candidates.reserve(neighbors.size());
      for (int n : neighbors) {
        candidates.push_back({dist_sq(src, n), n});
      }
      get_neighbors_heuristic(src, candidates, level);
    }
  }
};

inline void Solution::build(int d, const std::vector<float>& base) {
  d_ = d;
  data_ = base;
  n_ = base.size() / d;
  
  levels_.resize(n_);
  graph_.resize(n_);
  visited_.resize(n_);
  level_mult_ = 1.0 / std::log(1.0 * M);

  if (n_ == 0) return;

  for (size_t i = 0; i < n_; ++i) {
    int level = get_random_level();
    levels_[i] = level;
    graph_[i].resize(level + 1);
    for (int l = 0; l <= level; ++l) {
      graph_[i][l].reserve((l == 0 ? M0 : M) + 1);
    }
  }

  entry_point_ = 0;
  max_level_ = levels_[0];

  for (size_t i = 1; i < n_; ++i) {
    int curr_obj = i;
    int curr_level = levels_[i];
    int curr_ep = entry_point_;
    const float* curr_vec = &data_[curr_obj * d_];

    // 1. Greedy descent
    // Uses squared distance now
    for (int l = max_level_; l > curr_level; l--) {
      bool changed = true;
      while (changed) {
        changed = false;
        float dist_ep = dist_sq(curr_obj, curr_ep);
        for (int neighbor : graph_[curr_ep][l]) {
          float d = dist_sq(curr_obj, neighbor);
          if (d < dist_ep) {
            curr_ep = neighbor;
            dist_ep = d;
            changed = true;
          }
        }
      }
    }

    // 2. Construction
    for (int l = std::min(curr_level, max_level_); l >= 0; l--) {
      // search_layer now handles visited_.advance() internally
      auto top_candidates = search_layer(curr_vec, curr_ep, ef_construction, l);

      std::vector<std::pair<float, int>> potential_neighbors;
      potential_neighbors.reserve(ef_construction);
      while (!top_candidates.empty()) {
        potential_neighbors.push_back(top_candidates.top());
        top_candidates.pop();
      }

      get_neighbors_heuristic(curr_obj, potential_neighbors, l);

      for (int neighbor : graph_[curr_obj][l]) {
        add_connection(neighbor, curr_obj, l);
      }

      if (!potential_neighbors.empty()) {
        curr_ep = potential_neighbors[0].second;
      }
    }

    if (curr_level > max_level_) {
      max_level_ = curr_level;
      entry_point_ = curr_obj;
    }
  }
}

inline void Solution::search(const std::vector<float>& query, int* res) {
  if (n_ == 0) return;

  int curr_ep = entry_point_;

  // Greedy descent
  for (int l = max_level_; l > 0; l--) {
    bool changed = true;
    while (changed) {
      changed = false;
      float dist_ep = dist_query_sq(query, curr_ep);
      for (int neighbor : graph_[curr_ep][l]) {
        float d = dist_query_sq(query, neighbor);
        if (d < dist_ep) {
          curr_ep = neighbor;
          dist_ep = d;
          changed = true;
        }
      }
    }
  }

  // Beam search at layer 0
  auto top_candidates = search_layer(query.data(), curr_ep, ef_search, 0);

  std::vector<std::pair<float, int>> results;
  results.reserve(ef_search);
  while (!top_candidates.empty()) {
    results.push_back(top_candidates.top());
    top_candidates.pop();
  }

  // Sort by distance (smallest first)
  // Note: Standard sort on pairs sorts by first element (distance) ascending by default
  // But priority_queue was a Max Heap, so we popped them largest-first?
  // Actually, search_layer returns a max-heap (farthest of the k-nearest at top).
  // We need to reverse or sort.
  std::sort(results.begin(), results.end());

  for (size_t i = 0; i < 10 && i < results.size(); ++i) {
    res[i] = results[i].second;
  }
  
  for (size_t i = results.size(); i < 10; ++i) {
    res[i] = -1;
  }
}
