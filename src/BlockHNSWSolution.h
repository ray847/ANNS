#pragma once

#include <vector>

#include "BlockHNSWSolver.h"
#include "SolutionConcept.h"
#include "BlockHNSWClusterIndex.h"

class Solution {
public:
  void build(int d, const std::vector<float>& base) {
    // Adjust target_bucket_size based on dataset size
    // 1. Build Index (Data Organization)
    cluster_index_.Build(d, base, 64);
    // 2. Get Centroids and Index them
    // We use FlatSolver because linear scanning 15k vectors is faster/more accurate
    // than navigating an HNSW graph.
    std::vector<float> centroids = cluster_index_.GetCentroids();
    solver_.build(d, centroids);
  }

  void search(const std::vector<float>& query, int* res) {
    // --- TUNING KNOB ---
    constexpr int kNumProbes = 100; 
    
    std::vector<int> nearest_centroids(kNumProbes);
    solver_.search(query, nearest_centroids.data(), kNumProbes);
    cluster_index_.Scan(query, nearest_centroids.data(), kNumProbes, res);
  }

private:
  HNSWSolver solver_; // Replaced HNSWSolver
  ClusterIndex cluster_index_;
};

static_assert(IsSolution<Solution>);
