#include <iostream> // std:cerr
#include <stdexcept> // std::runtime_error
#include <vector> // std::vector
#include <random> // std::random_device
#include <fstream> // std::ofstream
#include <string> // std::string
#include <format> // std::format

#include "NaiveSolution.h" // Solution
#include "Global.h" // global::kDATA_SET_INFOS

void generate(const global::DataSetInfo& info);

int main() {
  for (auto dataset_info : global::kDATA_SET_INFOS) {
    try {
      generate(dataset_info);
    } catch (std::runtime_error e) {
      std::cerr << e.what() << std::endl;
    }
  }
}

std::vector<float> load_floats(const std::string& file, size_t count) {
  std::vector<float> data(count); 
  std::ifstream is(file);
  if (!is.is_open())
    throw std::runtime_error("load_dataset: Failed to open file.");
  for (auto& ele : data) is >> ele;
  return data;
}
std::vector<int> generate_label(
  const global::DataSetInfo& info,
  const std::vector<std::vector<float>>& queries
) { 
  /* Unpack the dataset info. */
  auto name = info.name;
  auto n_data_points = info.n_data_points;
  auto dataset_file = info.dataset_file;
  auto dims = info.dims;
  auto n_queries = info.n_queries;
  std::cout << "------------------------------------------------------------\n";
  std::cout << std::format("Generate Label for: {}\n", name);
  /* Load the dataset. */
  auto dataset = load_floats(std::string{dataset_file}, dims * n_data_points);
  /* Run the solver. */
  Solution solution;
  std::vector<int> res(n_queries * 10, 0);
  solution.build(dims, dataset);
  for (int i = 0; i < n_queries; ++i) {
    solution.search(queries[i], res.data() + 10 * i);
  }
  /* Save the labels. */
  std::cout << std::format("Done.\n");
  return res;
}
std::vector<std::vector<float>> generate_query(
  const global::DataSetInfo& info
) {
  auto n_queries = info.n_queries;
  auto dims = info.dims;
  std::vector<std::vector<float>> data(n_queries);
  {
    std::normal_distribution<float> dis(0.0f, 1.0f);
    for (size_t i = 0; i < n_queries; ++i) {
      for (size_t j = 0; j < dims; ++j) {
        data[i].emplace_back(dis(global::rng));
      }
    }
  }
  return data;
}
void generate(const global::DataSetInfo& info) {
  /* Unpack the dataset info. */
  auto n_queries = info.n_queries;
  auto query_file = info.sample_file;
  auto label_file = info.label_file;
  auto dims = info.dims;
  auto queries = generate_query(info);
  auto labels = generate_label(info, queries);
  /* Save the results. */
  {
    std::ofstream os(std::string{query_file});
    for (size_t i = 0; i < n_queries; ++i) {
      for (size_t j = 0; j < dims; ++j) {
        os << queries[i][j] << ' ';
      }
      os << '\n';
    }
  }
  {
    std::ofstream os(std::string{label_file});
    for (size_t i = 0; i < n_queries; ++i) {
      for (size_t j = 0; j < global::kCRITERION; ++j) {
        os << labels[i * global::kCRITERION + j] << ' ';
      } 
      os << '\n';
    }
  }
}
