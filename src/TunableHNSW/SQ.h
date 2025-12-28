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

#include <vector>
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <numeric>
#include "Config.h"

namespace TunableHNSW {

template <int kDim>
class ScalarQuantizer {
 public:
  ScalarQuantizer() {
    min_.resize(kDim);
    diff_.resize(kDim);
  }

  void Train(const float* data, int num_points) {
    std::fill(min_.begin(), min_.end(), FLT_MAX);
    std::vector<float> max_val(kDim, -FLT_MAX);

    for (int i = 0; i < num_points; ++i) {
      const float* vec = data + i * kDim;
      for (int d = 0; d < kDim; ++d) {
        if (vec[d] < min_[d]) min_[d] = vec[d];
        if (vec[d] > max_val[d]) max_val[d] = vec[d];
      }
    }

    for (int d = 0; d < kDim; ++d) {
      diff_[d] = max_val[d] - min_[d];
      if (diff_[d] < 1e-9f) diff_[d] = 1.0f; // Prevent division by zero
    }
  }

  void Encode(const float* vector, uint16_t* code) const {
    for (int d = 0; d < kDim; ++d) {
      float val = vector[d];
      float normalized = (val - min_[d]) / diff_[d];
      if (normalized < 0.0f) normalized = 0.0f;
      if (normalized > 1.0f) normalized = 1.0f;
      code[d] = static_cast<uint16_t>(normalized * 65535.0f);
    }
  }

  float L2Sq(const float* query, const uint16_t* code) const {
    float dist = 0.0f;
    for (int d = 0; d < kDim; ++d) {
      float reconstructed = (static_cast<float>(code[d]) / 65535.0f) * diff_[d] + min_[d];
      float diff = query[d] - reconstructed;
      dist += diff * diff;
    }
    return dist;
  }

 private:
  std::vector<float> min_;
  std::vector<float> diff_;
};

}  // namespace TunableHNSW
