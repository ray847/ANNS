#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <cfloat>
#include "KMeans.h"

namespace TunableHNSW {

// Implements Product Quantization for vector compression.
//
// The vector space is partitioned into a Cartesian product of low-dimensional
// subspaces, and each subspace is quantized separately. A vector is represented
// by a short code of centroid indices.
template <int kDim, int kSubquantizers, int kSubvectorBits = 8>
class ProductQuantizer {
 public:
  static constexpr int kNumCentroids = 1 << kSubvectorBits;
  static constexpr int kSubvectorDim = kDim / kSubquantizers;

  static_assert(
      kDim % kSubquantizers == 0,
      "Dimension must be divisible by the number of sub-quantizers.");

  ProductQuantizer() = default;

  // Trains the codebooks for each sub-quantizer using K-Means.
  // If `sample_size` is less than `num_points`, training is done on a random
  // sample of the data.
  void Train(const float* data, int num_points, size_t sample_size) {
    codebooks_.resize(kSubquantizers * kNumCentroids * kSubvectorDim);
    KMeans kmeans(kNumCentroids, 20);

    const float* train_data = data;
    int actual_train_points = num_points;
    std::vector<float> sampled_data;

    if (sample_size > 0 && sample_size < num_points) {
      sampled_data.reserve(sample_size * kDim);
      std::vector<int> indices(num_points);
      std::iota(indices.begin(), indices.end(), 0);
      std::shuffle(indices.begin(), indices.end(), global::rng);

      for (size_t i = 0; i < sample_size; ++i) {
        std::copy(data + indices[i] * kDim, data + (indices[i] + 1) * kDim,
                  std::back_inserter(sampled_data));
      }
      train_data = sampled_data.data();
      actual_train_points = sample_size;
    }

    for (int m = 0; m < kSubquantizers; ++m) {
      const float* sub_data_start = train_data + m * kSubvectorDim;
      auto sub_centroids =
          kmeans.Train(sub_data_start, actual_train_points, kSubvectorDim, kDim);

      float* codebook_dst =
          codebooks_.data() + m * kNumCentroids * kSubvectorDim;
      std::copy(sub_centroids.begin(), sub_centroids.end(), codebook_dst);
    }
  }

  // Encodes a full-precision vector into a PQ code.
  std::vector<uint8_t> Encode(const float* vector) const {
    std::vector<uint8_t> code(kSubquantizers);
    for (int m = 0; m < kSubquantizers; ++m) {
      const float* sub_vector = vector + m * kSubvectorDim;
      const float* codebook =
          codebooks_.data() + m * kNumCentroids * kSubvectorDim;

      float min_dist_sq = FLT_MAX;
      uint8_t best_centroid_id = 0;
      for (int k = 0; k < kNumCentroids; ++k) {
        float dist_sq =
            L2SqSlice_(sub_vector, codebook + k * kSubvectorDim, kSubvectorDim);
        if (dist_sq < min_dist_sq) {
          min_dist_sq = dist_sq;
          best_centroid_id = k;
        }
      }
      code[m] = best_centroid_id;
    }
    return code;
  }

  // Pre-computes a distance table for a given query vector.
  // The table stores the squared L2 distance between each sub-vector of the
  // query and each centroid in the corresponding codebook.
  std::vector<float> BuildDistanceTable(const float* query) const {
    std::vector<float> table(kSubquantizers * kNumCentroids);
    for (int m = 0; m < kSubquantizers; ++m) {
      const float* query_sub_vector = query + m * kSubvectorDim;
      const float* codebook =
          codebooks_.data() + m * kNumCentroids * kSubvectorDim;
      for (int k = 0; k < kNumCentroids; ++k) {
        table[m * kNumCentroids + k] = L2SqSlice_(
            query_sub_vector, codebook + k * kSubvectorDim, kSubvectorDim);
      }
    }
    return table;
  }

  // Calculates the Asymmetric Distance from the pre-computed table and a PQ code.
  float GetDistanceFromTable(const std::vector<float>& table,
                             const uint8_t* code) const {
    float dist = 0.0f;
    for (int m = 0; m < kSubquantizers; ++m) {
      dist += table[m * kNumCentroids + code[m]];
    }
    return dist;
  }

 private:
  // Calculates L2 squared distance on a slice of data.
  float L2SqSlice_(const float* vector_a, const float* vector_b, int dim) const {
    float result = 0.0f;
    for (int i = 0; i < dim; ++i) {
      float diff = vector_a[i] - vector_b[i];
      result += diff * diff;
    }
    return result;
  }

  // Stores all codebooks contiguously.
  // Layout: [subquantizer_0, subquantizer_1, ..., subquantizer_m]
  // Each subquantizer block has kNumCentroids * kSubvectorDim floats.
  std::vector<float> codebooks_;
};

}  // namespace TunableHNSW
