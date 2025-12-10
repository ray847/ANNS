// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "HNSW.h"
#include "IVF.h"
#include "Config.h"
#include "../Global.h"

// This file provides a concrete implementation of the Solution concept
// for the evaluation framework. It uses the HNSWConfig structs to define
// different combinations of optimizations and dispatches to the correct
// index type (HNSW or IVF) at compile time.

// Configuration for 100-dimensional vectors (e.g., GLOVE).
using HNSWConfig100 = TunableHNSW::HNSWConfig<
    /*kDim=*/100,
    /*kUseSIMD=*/true,
    /*kSearchStrategy=*/TunableHNSW::SearchStrategy::kDynamic,
    /*kQuantization=*/TunableHNSW::QuantizationStrategy::kPQ,
    /*kIndexStrategy=*/TunableHNSW::IndexStrategy::kIVF_HNSW,
    /*kM=*/32,
    /*kM0=*/64,
    /*kEfConstruction=*/400,
    /*kEfSearch=*/200,
    /*kMaxLevel=*/16,
    /*kPQSubquantizers=*/25,
    /*kPQTrainSampleSize=*/20000, // Sample 20k vectors for PQ training
    /*kIVFNumClusters=*/1024,
    /*kIVFNProbe=*/10>;

// Configuration for 128-dimensional vectors (e.g., SIFT).
using HNSWConfig128 = TunableHNSW::HNSWConfig<
    /*kDim=*/128,
    /*kUseSIMD=*/true,
    /*kSearchStrategy=*/TunableHNSW::SearchStrategy::kDynamic,
    /*kQuantization=*/TunableHNSW::QuantizationStrategy::kPQ,
    /*kIndexStrategy=*/TunableHNSW::IndexStrategy::kIVF_HNSW,
    /*kM=*/32,
    /*kM0=*/64,
    /*kEfConstruction=*/400,
    /*kEfSearch=*/200,
    /*kMaxLevel=*/16,
    /*kPQSubquantizers=*/32,
    /*kPQTrainSampleSize=*/20000, // Sample 20k vectors for PQ training
    /*kIVFNumClusters=*/1024,
    /*kIVFNProbe=*/10>;

class TunableHNSWSolution {
 private:
  template <typename Config>
  using Index = std::conditional_t<
      Config::kIndexStrategyVal == TunableHNSW::IndexStrategy::kIVF_HNSW,
      TunableHNSW::IVF<Config>, TunableHNSW::HNSW<Config>>;

  using Index100 = Index<HNSWConfig100>;
  using Index128 = Index<HNSWConfig128>;

  std::unique_ptr<Index100> index100_;
  std::unique_ptr<Index128> index128_;
  int dims_ = 0;

 public:
  TunableHNSWSolution() = default;

  void build(int d, const std::vector<float>& base) {
    dims_ = d;
    if (dims_ == 100) {
      index100_ = std::make_unique<Index100>();
      index100_->Build(base);
    } else if (dims_ == 128) {
      index128_ = std::make_unique<Index128>();
      index128_->Build(base);
    }
  }

  void search(const std::vector<float>& query, int* res) {
    if (dims_ == 100 && index100_) {
      index100_->Search(query, res);
    } else if (dims_ == 128 && index128_) {
      index128_->Search(query, res);
    }
  }
};
