#pragma once

#include <vector>
#include <mutex>
#include <cmath>
#include <random>
#include <cstring>
#include <algorithm>

template <int kMaxNeighbors, int kMaxLayer0Neighbors, int kMaxLevel>
class HnswGraph {
 public:
  struct Node {
    int level;
    // Standard vector layout (Best Cache Locality for this Graph)
    std::vector<int> flat_links;
    std::vector<int> link_counts;
  };

  HnswGraph() { 
      level_mult_ = 1.0 / std::log(1.0 * kMaxNeighbors);
      // Initialize 8192 sharded locks for fine-grained concurrency
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

  // --- Locking & Global State ---

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
