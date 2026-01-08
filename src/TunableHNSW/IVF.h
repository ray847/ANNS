#pragma once

#include <vector>
#include <memory>
#include <mutex>
#include <numeric>
#include <algorithm>
#include <thread>
#include <memory_resource>
#include "../Global.h"
#include "Config.h"
#include "HNSW.h"
#include "KMeans.h"
#include "OPQ.h"

namespace TunableHNSW {

template <typename MainConfig>
class IVF {
 public:
  using CentroidHNSWConfig = HNSWConfig<
      MainConfig::kDimVal, MainConfig::kUseSIMDVal, SearchStrategy::kStandard,
      QuantizationStrategy::kNone, IndexStrategy::kHNSW,
      8,    // M
      16,   // M0
      50,   // EfConstruction
      MainConfig::kIVFNProbeVal,
      10,   // MaxLevel
      0,    // PQSubquantizers
      0,    // IVF_NumClusters
      0,    // IVF_NProbe
      0,    // IVF_TrainSampleSize
      MainConfig::kUsePMRVal
      >;

  using CentroidHNSW = HNSW<CentroidHNSWConfig>;

  IVF() = default;

  void Build(const std::vector<float>& base_data) {
    if constexpr (MainConfig::kUsePMRVal) {
      pmr_resource_.emplace();
      data_storage_ = Vector<float>(&*pmr_resource_);
      coarse_centroids_ = Vector<float>(&*pmr_resource_);
      inverted_lists_ = Vector<Vector<int>>(&*pmr_resource_);
      pq_codes_ = Vector<uint8_t>(&*pmr_resource_);
    }
    data_storage_.assign(base_data.begin(), base_data.end());
    const float* data_ptr = data_storage_.data();
    size_t num_points = base_data.size() / MainConfig::kDimVal;

    const float* kmeans_train_data = data_ptr;
    size_t kmeans_train_points = num_points;
    std::vector<float> sampled_data;
    if constexpr (MainConfig::kIVFTrainSampleSizeVal > 0) {
        if (MainConfig::kIVFTrainSampleSizeVal < num_points) {
            constexpr size_t sample_size = MainConfig::kIVFTrainSampleSizeVal;
            sampled_data.reserve(sample_size * MainConfig::kDimVal);
            std::vector<int> indices(num_points);
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), global::rng);

            for (size_t i = 0; i < sample_size; ++i) {
                const float* src = data_ptr + indices[i] * MainConfig::kDimVal;
                sampled_data.insert(sampled_data.end(), src, src + MainConfig::kDimVal);
            }
            kmeans_train_data = sampled_data.data();
            kmeans_train_points = sample_size;
        }
    }

    KMeans kmeans(MainConfig::kIVFNumClustersVal, 25);
    auto trained_centroids =
        kmeans.Train(kmeans_train_data, kmeans_train_points, MainConfig::kDimVal, MainConfig::kDimVal);
    coarse_centroids_.assign(trained_centroids.begin(), trained_centroids.end());

    centroid_hnsw_ = std::make_unique<CentroidHNSW>();
    centroid_hnsw_->Build(coarse_centroids_);

    inverted_lists_.resize(MainConfig::kIVFNumClustersVal);
    std::vector<std::mutex> locks(MainConfig::kIVFNumClustersVal);
    
    auto worker1 = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; ++i) {
        float min_dist_sq = FLT_MAX;
        int best_cluster_id = -1;
        for (int j = 0; j < MainConfig::kIVFNumClustersVal; ++j) {
          float dist_sq =
              Distance<MainConfig::kDimVal, MainConfig::kUseSIMDVal>::L2Sq(
                  data_ptr + i * MainConfig::kDimVal,
                  coarse_centroids_.data() + j * MainConfig::kDimVal);
          if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            best_cluster_id = j;
          }
        }
        if (best_cluster_id != -1) {
          std::lock_guard<std::mutex> lock(locks[best_cluster_id]);
          inverted_lists_[best_cluster_id].push_back(i);
        }
      }
    };

    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    size_t block_size = num_points / num_threads;
    for (unsigned int i = 0; i < num_threads; ++i) {
        size_t start = i * block_size;
        size_t end = (i == num_threads - 1) ? num_points : start + block_size;
        threads.emplace_back(worker1, start, end);
    }
    for (auto& t : threads) {
        t.join();
    }

    if constexpr (MainConfig::kQuantizationVal == QuantizationStrategy::kPQ || MainConfig::kQuantizationVal == QuantizationStrategy::kOPQ) {
      pq_.emplace();
      pq_->Train(data_ptr, num_points);
      pq_codes_.resize(num_points * MainConfig::kPQSubquantizersVal);
      
      auto worker2 = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
          auto code = pq_->Encode(data_ptr + i * MainConfig::kDimVal);
          std::copy(code.begin(), code.end(),
                    pq_codes_.data() + i * MainConfig::kPQSubquantizersVal);
        }
      };

      std::vector<std::thread> threads2;
      threads2.reserve(num_threads);
      for (unsigned int i = 0; i < num_threads; ++i) {
          size_t start = i * block_size;
          size_t end = (i == num_threads - 1) ? num_points : start + block_size;
          threads2.emplace_back(worker2, start, end);
      }
      for (auto& t : threads2) {
          t.join();
      }
    }
  }

  void Search(const std::vector<float>& query, int* result_indices) {
    std::vector<int> probe_indices(MainConfig::kIVFNProbeVal);
    centroid_hnsw_->Search(query, probe_indices.data());

    using QueueItem = std::pair<float, int>;
    std::priority_queue<QueueItem> top_candidates;

    if constexpr (MainConfig::kQuantizationVal == QuantizationStrategy::kPQ || MainConfig::kQuantizationVal == QuantizationStrategy::kOPQ) {
      constexpr size_t rerank_candidates_count = global::kCRITERION * 4;
      auto dist_table = pq_->BuildDistanceTable(query.data());
      for (int i = 0; i < MainConfig::kIVFNProbeVal; ++i) {
        int cluster_id = probe_indices[i];
        if (cluster_id == -1) continue;
        const auto& list = inverted_lists_[cluster_id];
        for (int idx : list) {
          const uint8_t* code =
              pq_codes_.data() + idx * MainConfig::kPQSubquantizersVal;
          float dist = pq_->GetDistanceFromTable(dist_table, code);
          if (top_candidates.size() < rerank_candidates_count ||
              dist < top_candidates.top().first) {
            top_candidates.push({dist, idx});
            if (top_candidates.size() > rerank_candidates_count) {
              top_candidates.pop();
            }
          }
        }
      }

      std::vector<QueueItem> candidates_to_rerank;
      candidates_to_rerank.reserve(top_candidates.size());
      while (!top_candidates.empty()) {
          candidates_to_rerank.push_back(top_candidates.top());
          top_candidates.pop();
      }

      for(const auto& item : candidates_to_rerank) {
          int candidate_id = item.second;
          float exact_dist = Distance<MainConfig::kDimVal, MainConfig::kUseSIMDVal>::L2Sq(
              query.data(), data_storage_.data() + candidate_id * MainConfig::kDimVal);
          
          if (top_candidates.size() < global::kCRITERION || exact_dist < top_candidates.top().first) {
              top_candidates.push({exact_dist, candidate_id});
              if (top_candidates.size() > global::kCRITERION) {
                  top_candidates.pop();
              }
          }
      }

    } else {
      for (int i = 0; i < MainConfig::kIVFNProbeVal; ++i) {
        int cluster_id = probe_indices[i];
        if (cluster_id == -1) continue;
        const auto& list = inverted_lists_[cluster_id];
        for (int idx : list) {
          float dist = Distance<MainConfig::kDimVal, MainConfig::kUseSIMDVal>::L2Sq(
              query.data(), data_storage_.data() + idx * MainConfig::kDimVal);
          if (top_candidates.size() < global::kCRITERION ||
              dist < top_candidates.top().first) {
            top_candidates.push({dist, idx});
            if (top_candidates.size() > global::kCRITERION) {
              top_candidates.pop();
            }
          }
        }
      }
    }

    size_t k_idx = 0;
    std::vector<QueueItem> sorted_results;
    while (!top_candidates.empty()) {
      sorted_results.push_back(top_candidates.top());
      top_candidates.pop();
    }
    std::reverse(sorted_results.begin(), sorted_results.end());
    for (const auto& p : sorted_results) {
      if (k_idx >= global::kCRITERION) break;
      result_indices[k_idx++] = p.second;
    }
    while (k_idx < global::kCRITERION) result_indices[k_idx++] = -1;
  }

 private:
  template <typename T>
  using Vector = std::conditional_t<MainConfig::kUsePMRVal, std::pmr::vector<T>, std::vector<T>>;

  using PQ = std::conditional_t<
      MainConfig::kQuantizationVal == QuantizationStrategy::kPQ || MainConfig::kQuantizationVal == QuantizationStrategy::kOPQ,
      OptimizedProductQuantizer<MainConfig>,
      std::nullptr_t>;

  std::optional<std::pmr::synchronized_pool_resource> pmr_resource_;
  Vector<float> coarse_centroids_;
  Vector<Vector<int>> inverted_lists_;
  std::unique_ptr<CentroidHNSW> centroid_hnsw_;

  Vector<float> data_storage_;
  std::optional<PQ> pq_;
  Vector<uint8_t> pq_codes_;
};

}  // namespace TunableHNSW
