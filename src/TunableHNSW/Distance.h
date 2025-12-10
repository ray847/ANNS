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

#include <immintrin.h>
#include <type_traits>

namespace TunableHNSW {

// Provides compile-time selectable implementations for L2 distance calculation.
//
// The generic template provides a portable fallback. Template specializations
// are used to provide optimized versions for specific dimensions and for
// SIMD (AVX2) instruction sets.

// Generic L2 distance squared, acts as a fallback.
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

// Specialization for AVX-enabled distance calculations.
template <int kDim>
struct Distance<kDim, true, std::enable_if_t<(kDim > 0)>> {
  static_assert(kDim > 0, "Dimension must be positive.");

  __attribute__((target("avx2,fma"))) inline static float L2Sq(
      const float* vector_a, const float* vector_b) {
    __m256 sum_256 = _mm256_setzero_ps();
    int i = 0;

    for (; i <= kDim - 8; i += 8) {
      __m256 vector_a_256 = _mm256_loadu_ps(vector_a + i);
      __m256 vector_b_256 = _mm256_loadu_ps(vector_b + i);
      __m256 diff_256 = _mm256_sub_ps(vector_a_256, vector_b_256);
      sum_256 = _mm256_fmadd_ps(diff_256, diff_256, sum_256);
    }

    __m128 sum_128_low = _mm256_castps256_ps128(sum_256);
    __m128 sum_128_high = _mm256_extractf128_ps(sum_256, 1);
    __m128 result_128 = _mm_add_ps(sum_128_low, sum_128_high);
    result_128 = _mm_hadd_ps(result_128, result_128);
    result_128 = _mm_hadd_ps(result_128, result_128);
    float result = _mm_cvtss_f32(result_128);

    for (; i < kDim; ++i) {
      const float diff = vector_a[i] - vector_b[i];
      result += diff * diff;
    }
    return result;
  }
};

// Specialization for Dim=100 with SIMD.
template <>
struct Distance<100, true> {
  __attribute__((target("avx2,fma"))) inline static float L2Sq(
      const float* vector_a, const float* vector_b) {
    __m256 sum_256 = _mm256_setzero_ps();

    for (int i = 0; i < 12; ++i) {
      __m256 vector_a_256 = _mm256_loadu_ps(vector_a + i * 8);
      __m256 vector_b_256 = _mm256_loadu_ps(vector_b + i * 8);
      __m256 diff_256 = _mm256_sub_ps(vector_a_256, vector_b_256);
      sum_256 = _mm256_fmadd_ps(diff_256, diff_256, sum_256);
    }

    __m128 sum_128_low = _mm256_castps256_ps128(sum_256);
    __m128 sum_128_high = _mm256_extractf128_ps(sum_256, 1);
    __m128 result_128 = _mm_add_ps(sum_128_low, sum_128_high);
    result_128 = _mm_hadd_ps(result_128, result_128);
    result_128 = _mm_hadd_ps(result_128, result_128);
    float result = _mm_cvtss_f32(result_128);

    for (int i = 96; i < 100; ++i) {
      const float diff = vector_a[i] - vector_b[i];
      result += diff * diff;
    }
    return result;
  }
};

// Specialization for Dim=128 with SIMD.
template <>
struct Distance<128, true> {
  __attribute__((target("avx2,fma"))) inline static float L2Sq(
      const float* vector_a, const float* vector_b) {
    __m256 sum_256 = _mm256_setzero_ps();

    for (int i = 0; i < 16; ++i) {
      __m256 vector_a_256 = _mm256_loadu_ps(vector_a + i * 8);
      __m256 vector_b_256 = _mm256_loadu_ps(vector_b + i * 8);
      __m256 diff_256 = _mm256_sub_ps(vector_a_256, vector_b_256);
      sum_256 = _mm256_fmadd_ps(diff_256, diff_256, sum_256);
    }

    __m128 sum_128_low = _mm256_castps256_ps128(sum_256);
    __m128 sum_128_high = _mm256_extractf128_ps(sum_256, 1);
    __m128 result_128 = _mm_add_ps(sum_128_low, sum_128_high);
    result_128 = _mm_hadd_ps(result_128, result_128);
    result_128 = _mm_hadd_ps(result_128, result_128);

    return _mm_cvtss_f32(result_128);
  }
};

}  // namespace TunableHNSW
