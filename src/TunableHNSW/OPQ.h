#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <numeric>
#include <thread>
#include "KMeans.h"
#include "../Eigen/Dense"
#include "Config.h"
#include "../Global.h"

namespace TunableHNSW {

template <typename Config>
class OptimizedProductQuantizer {
 public:
  static constexpr int kDim = Config::kDimVal;
  static constexpr int kSubquantizers = Config::kPQSubquantizersVal;
  static constexpr int kSubvectorBits = 8;
  static constexpr int kNumCentroids = 1 << kSubvectorBits;
  static constexpr int kSubvectorDim = kDim / kSubquantizers;

  static_assert(
      kDim % kSubquantizers == 0,
      "Dimension must be divisible by the number of sub-quantizers.");

  OptimizedProductQuantizer() : R_(Eigen::MatrixXf::Identity(kDim, kDim)) {}

  void Train(const float* data, int num_points) {
    using MatrixXfRow = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    
    Eigen::Map<const MatrixXfRow> full_dataset(data, num_points, kDim);
    MatrixXfRow train_dataset;

    constexpr size_t sample_size = Config::kQuantizerTrainSampleSizeVal;

    if (sample_size > 0 && sample_size < num_points) {
        train_dataset.resize(sample_size, kDim);
        std::vector<int> indices(num_points);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), global::rng);
        for (size_t i = 0; i < sample_size; ++i) {
            train_dataset.row(i) = full_dataset.row(indices[i]);
        }
    } else {
        train_dataset = full_dataset;
    }

    if constexpr (Config::kQuantizationVal == QuantizationStrategy::kOPQ) {
        const int num_iterations = 10;
        for (int i = 0; i < num_iterations; ++i) {
          MatrixXfRow rotated_data = train_dataset * R_.transpose();

          TrainCodebooks(rotated_data.data(), train_dataset.rows());

          MatrixXfRow C(train_dataset.rows(), kDim);
          
          auto worker = [&](int start, int end) {
            for (int j = start; j < end; ++j) {
              const float* vec = rotated_data.row(j).data();
            
              for (int m = 0; m < kSubquantizers; ++m) {
                 const float* sub_vec = vec + m * kSubvectorDim;
                 const float* codebook = codebooks_.data() + m * kNumCentroids * kSubvectorDim;
                 
                 int best_id = 0;
                 float min_dist = FLT_MAX;
                 
                 for (int k = 0; k < kNumCentroids; ++k) {
                     float d = L2SqSlice_(sub_vec, codebook + k * kSubvectorDim, kSubvectorDim);
                     if(d < min_dist) { min_dist = d; best_id = k; }
                 }

                 const float* best_centroid = codebook + best_id * kSubvectorDim;
                 for(int d=0; d<kSubvectorDim; ++d) {
                     C(j, m * kSubvectorDim + d) = best_centroid[d];
                 }
              }
            }
          };

          unsigned int num_threads = std::thread::hardware_concurrency();
          if (num_threads == 0) num_threads = 4;
          std::vector<std::thread> threads;
          threads.reserve(num_threads);
          int block_size = train_dataset.rows() / num_threads;
          for (unsigned int i = 0; i < num_threads; ++i) {
              int start = i * block_size;
              int end = (i == num_threads - 1) ? train_dataset.rows() : start + block_size;
              threads.emplace_back(worker, start, end);
          }

          for (auto& t : threads) {
              t.join();
          }

          Eigen::MatrixXf M = train_dataset.transpose() * C;
          Eigen::JacobiSVD<Eigen::MatrixXf> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
          
          R_ = svd.matrixV() * svd.matrixU().transpose();
        }
    } else {
        R_ = Eigen::MatrixXf::Identity(kDim, kDim);
        TrainCodebooks(train_dataset.data(), train_dataset.rows());
    }
  }

  std::vector<uint8_t> Encode(const float* vector) const {
    Eigen::Map<const Eigen::Matrix<float, kDim, 1>> v(vector);
    Eigen::Matrix<float, kDim, 1> rotated_v = R_ * v;
    
    std::vector<uint8_t> code(kSubquantizers);
    for (int m = 0; m < kSubquantizers; ++m) {
      const float* sub_vector = rotated_v.data() + m * kSubvectorDim;
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

  std::vector<float> BuildDistanceTable(const float* query) const {
    Eigen::Map<const Eigen::Matrix<float, kDim, 1>> q(query);
    Eigen::Matrix<float, kDim, 1> rotated_q = R_ * q;

    std::vector<float> table(kSubquantizers * kNumCentroids);
    for (int m = 0; m < kSubquantizers; ++m) {
      const float* query_sub_vector = rotated_q.data() + m * kSubvectorDim;
      const float* codebook =
          codebooks_.data() + m * kNumCentroids * kSubvectorDim;
      for (int k = 0; k < kNumCentroids; ++k) {
        table[m * kNumCentroids + k] = L2SqSlice_(
            query_sub_vector, codebook + k * kSubvectorDim, kSubvectorDim);
      }
    }
    return table;
  }

  float GetDistanceFromTable(const std::vector<float>& table,
                             const uint8_t* code) const {
    float dist = 0.0f;
    for (int m = 0; m < kSubquantizers; ++m) {
      dist += table[m * kNumCentroids + code[m]];
    }
    return dist;
  }

 private:
  void TrainCodebooks(const float* data, int num_points) {
    codebooks_.resize(kSubquantizers * kNumCentroids * kSubvectorDim);
    std::vector<std::thread> threads;
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;

    auto train_kmeans = [&](int m) {
        KMeans kmeans(kNumCentroids, 20);
        const float* sub_data_start = data + m * kSubvectorDim;
        auto sub_centroids =
            kmeans.Train(sub_data_start, num_points, kSubvectorDim, kDim);
        float* codebook_dst =
            codebooks_.data() + m * kNumCentroids * kSubvectorDim;
        std::copy(sub_centroids.begin(), sub_centroids.end(), codebook_dst);
    };

    for (int m = 0; m < kSubquantizers; ++m) {
        threads.emplace_back(train_kmeans, m);
        if (threads.size() >= num_threads) {
            for (auto& t : threads) t.join();
            threads.clear();
        }
    }
    for (auto& t : threads) t.join();
  }

  float L2SqSlice_(const float* vector_a, const float* vector_b, int dim) const {
    float result = 0.0f;
    for (int i = 0; i < dim; ++i) {
      float diff = vector_a[i] - vector_b[i];
      result += diff * diff;
    }
    return result;
  }

  Eigen::MatrixXf R_;
  std::vector<float> codebooks_;
};

}  // namespace TunableHNSW
