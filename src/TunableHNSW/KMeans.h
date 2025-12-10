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
#include <random>
#include <algorithm>
#include <cfloat>
#include <thread>
#include "../Global.h"

namespace TunableHNSW {

// A simple K-Means implementation for use by the ProductQuantizer.
class KMeans {
 public:
  KMeans(int num_clusters, int num_iterations)
      : num_clusters_(num_clusters), num_iterations_(num_iterations) {}

  // Trains the K-Means model on a slice of the dataset.
  // This is used by the ProductQuantizer to train a codebook on each
  // set of sub-vectors.
  //
  // `data`: Pointer to the beginning of the full-dimensional data.
  // `num_points`: The total number of vectors to train on.
  // `dim`: The dimension of the sub-vectors for this k-means instance.
  // `stride`: The stride in floats to get from one sub-vector to the next.
  std::vector<float> Train(const float* data, int num_points, int dim,
                           int stride) {
    std::vector<float> centroids(num_clusters_ * dim);
    InitializeCentroids_(data, num_points, dim, stride, &centroids);

    std::vector<int> assignments(num_points);
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    std::vector<std::thread> threads;

    for (int iter = 0; iter < num_iterations_; ++iter) {
      auto assignment_worker = [&](int start, int end) {
        for (int i = start; i < end; ++i) {
          float min_dist_sq = FLT_MAX;
          int best_cluster_id = -1;
          for (int j = 0; j < num_clusters_; ++j) {
            float dist_sq =
                L2SqSlice_(data + i * stride, centroids.data() + j * dim, dim);
            if (dist_sq < min_dist_sq) {
              min_dist_sq = dist_sq;
              best_cluster_id = j;
            }
          }
          assignments[i] = best_cluster_id;
        }
      };

      threads.clear();
      threads.reserve(num_threads);
      int block_size = num_points / num_threads;
      for (unsigned int i = 0; i < num_threads; ++i) {
          int start = i * block_size;
          int end = (i == num_threads - 1) ? num_points : start + block_size;
          threads.emplace_back(assignment_worker, start, end);
      }
      for (auto& t : threads) {
          t.join();
      }

      std::vector<float> new_centroids(num_clusters_ * dim, 0.0f);
      std::vector<int> cluster_counts(num_clusters_, 0);
      std::vector<std::vector<float>> thread_new_centroids(num_threads, std::vector<float>(num_clusters_ * dim, 0.0f));
      std::vector<std::vector<int>> thread_cluster_counts(num_threads, std::vector<int>(num_clusters_, 0));

      auto update_worker = [&](int thread_id, int start, int end) {
        for (int i = start; i < end; ++i) {
          int cluster_id = assignments[i];
          if (cluster_id != -1) {
            thread_cluster_counts[thread_id][cluster_id]++;
            for (int d = 0; d < dim; ++d) {
                thread_new_centroids[thread_id][cluster_id * dim + d] += data[i * stride + d];
            }
          }
        }
      };
    
      threads.clear();
      for (unsigned int i = 0; i < num_threads; ++i) {
          int start = i * block_size;
          int end = (i == num_threads - 1) ? num_points : start + block_size;
          threads.emplace_back(update_worker, i, start, end);
      }
      for (auto& t : threads) {
          t.join();
      }

      for (unsigned int i = 0; i < num_threads; ++i) {
        for (int j = 0; j < num_clusters_; ++j) {
            cluster_counts[j] += thread_cluster_counts[i][j];
            for (int d = 0; d < dim; ++d) {
                new_centroids[j * dim + d] += thread_new_centroids[i][j * dim + d];
            }
        }
      }

      for (int j = 0; j < num_clusters_; ++j) {
        if (cluster_counts[j] > 0) {
          for (int d = 0; d < dim; ++d) {
            new_centroids[j * dim + d] /= cluster_counts[j];
          }
        } else {
          int rand_idx =
              std::uniform_int_distribution<int>(0, num_points - 1)(global::rng);
          const float* random_point = data + rand_idx * stride;
          std::copy(random_point, random_point + dim,
                    new_centroids.data() + j * dim);
        }
      }
      centroids = new_centroids;
    }
    return centroids;
  }

 private:
  // Initializes centroids by picking k unique random points from the data.
  void InitializeCentroids_(const float* data, int num_points, int dim,
                            int stride, std::vector<float>* centroids) {
    std::vector<int> indices(num_points);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), global::rng);

    for (int i = 0; i < num_clusters_; ++i) {
      const float* src = data + indices[i] * stride;
      float* dst = centroids->data() + i * dim;
      std::copy(src, src + dim, dst);
    }
  }

  // Calculates L2 squared distance on a slice of data.
  float L2SqSlice_(const float* vector_a, const float* vector_b, int dim) {
    float result = 0.0f;
    for (int i = 0; i < dim; ++i) {
      float diff = vector_a[i] - vector_b[i];
      result += diff * diff;
    }
    return result;
  }

  int num_clusters_;
  int num_iterations_;
};

}  // namespace TunableHNSW
