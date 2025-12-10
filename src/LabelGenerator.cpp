#include <iostream> // std:cerr
#include <stdexcept> // std::runtime_error
#include <vector> // std::vector
#include <fstream> // std::ofstream
#include <string> // std::string
#include <format> // std::format

#include "NaiveSolution.h" // Solution
#include "Global.h" // global::kDATA_SET_INFOS

void generate(const global::DataSetInfo& info);

int main() {
  generate(global::kMINISIFT_INFO);
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
  const std::vector<float>& queries
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
    std::vector<float> query(dims);
    std::copy_n(&queries[i * dims], dims, query.begin());
    solution.search(query, res.data() + 10 * i);
  }
  /* Save the labels. */
  std::cout << std::format("Done.\n");
  return res;
}
void generate(const global::DataSetInfo& info) {
  /* Unpack the dataset info. */
  auto n_queries = info.n_queries;
  auto query_file = info.sample_file;
  auto label_file = info.label_file;
  auto dims = info.dims;
  auto queries = load_floats(std::string(query_file), dims * n_queries);
  auto labels = generate_label(info, queries);
  /* Save the results. */
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
