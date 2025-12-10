#pragma once

#include <cstddef>

namespace TunableHNSW {

enum class SearchStrategy { kStandard, kDynamic };
enum class QuantizationStrategy { kNone, kPQ, kOPQ };
enum class IndexStrategy { kHNSW, kIVF_HNSW };

template <int kDim, bool kUseSIMD, SearchStrategy kSearchStrategy,
          QuantizationStrategy kQuantization, IndexStrategy kIndexStrategy,
          size_t kM, size_t kM0, size_t kEfConstruction, size_t kEfSearch,
          int kMaxLevel, int kDynamicMaxNoImprovementIterations = 100, int kPQSubquantizers = 0,
          size_t kQuantizerTrainSampleSize = 0, int kIVFNumClusters = 0,
          int kIVFNProbe = 0, size_t kIVFTrainSampleSize = 0>
struct HNSWConfig {
  static constexpr int kDimVal = kDim;
  static constexpr bool kUseSIMDVal = kUseSIMD;
  static constexpr SearchStrategy kSearchStrategyVal = kSearchStrategy;
  static constexpr QuantizationStrategy kQuantizationVal = kQuantization;
  static constexpr IndexStrategy kIndexStrategyVal = kIndexStrategy;
  static constexpr size_t kMVal = kM;
  static constexpr size_t kM0Val = kM0;
  static constexpr size_t kEfConstructionVal = kEfConstruction;
  static constexpr size_t kEfSearchVal = kEfSearch;
  static constexpr int kMaxLevelVal = kMaxLevel;
  static constexpr int kDynamicMaxNoImprovementIterationsVal = kDynamicMaxNoImprovementIterations;
  static constexpr int kPQSubquantizersVal = kPQSubquantizers;
  static constexpr size_t kQuantizerTrainSampleSizeVal = kQuantizerTrainSampleSize;
  static constexpr int kIVFNumClustersVal = kIVFNumClusters;
  static constexpr int kIVFNProbeVal = kIVFNProbe;
  static constexpr size_t kIVFTrainSampleSizeVal = kIVFTrainSampleSize;

  static_assert(kDimVal > 0, "Dimension must be positive.");
  static_assert(kMVal > 0, "M must be positive.");
  static_assert(kM0Val > 0, "M0 must be positive.");
  static_assert(kEfConstructionVal > 0, "efConstruction must be positive.");
  static_assert(kEfSearchVal > 0, "efSearch must be positive.");
  static_assert(kMaxLevelVal > 0, "MaxLevel must be positive.");

  static constexpr bool Validate() {
    if constexpr (kQuantizationVal == QuantizationStrategy::kPQ || kQuantizationVal == QuantizationStrategy::kOPQ) {
      if (kPQSubquantizersVal <= 0) return false;
      if (kDimVal % kPQSubquantizersVal != 0) return false;
      if (kQuantizerTrainSampleSizeVal <= 0) return false;
    }
    if constexpr (kIndexStrategyVal == IndexStrategy::kIVF_HNSW) {
      if (kIVFNumClustersVal <= 0) return false;
      if (kIVFNProbeVal <= 0) return false;
    }
    return true;
  }
  static_assert(Validate(), "Invalid HNSWConfig parameters");
};

}  // namespace TunableHNSW
