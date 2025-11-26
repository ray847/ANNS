#include <iostream> // std::cout
#include <vector> // std::vector
#include <string> // std::string
#include <fstream> // std::ifstream
#include <chrono> // std::chrono
#include <format> // std::format
#include <unordered_set> // std::unordered_set

#include "Global.h" // global::kDATA_SET_INFOS
//#include "EmptySolution.h" // Solution
//#include "NaiveSolution.h" // Solution
//#include "QuickSolution.h" // Solution
//#include "ConvexGrouping.h" // Solution
#include "HnswSolution.h" // Solution

/* Function Declarations */
void run(const global::DataSetInfo& dataset);

/* Main Function */
int main() {
  run(global::kGLOVE_INFO);
  //run(global::kTEST_INFO);
  return 0;
}

struct Sample {
  std::vector<float> query;
  std::vector<int> label;
};

/* Function Definitions */
std::vector<float> load_base(const global::DataSetInfo& info) {
  auto base_file = info.dataset_file;
  auto dims = info.dims;
  auto n_base = info.n_data_points;
  std::ifstream is(std::string{base_file});
  std::vector<float> base(dims * n_base);
  for (auto& ele : base) is >> ele;
  is.close();
  return base;
}
std::vector<std::vector<float>> load_samples(const global::DataSetInfo& info) {
  auto sample_file = info.sample_file;
  auto dims = info.dims;
  auto n_samples = info.n_queries;
  std::ifstream is(std::string{sample_file});
  std::vector<std::vector<float>> samples(n_samples, std::vector<float>(dims));
  for (auto& sample : samples) {
    for (auto& dim : sample) {
      is >> dim;
    }
  }
  is.close();
  return samples;
}
std::vector<std::vector<size_t>> load_labels(const global::DataSetInfo& info) {
  auto label_file = info.label_file;
  auto n_samples = info.n_queries;
  std::ifstream is(std::string{label_file});
  std::vector<std::vector<size_t>> labels(
    n_samples,
    std::vector<size_t>(global::kCRITERION)
  );
  for (auto& label : labels) {
    for (auto& ele : label) {
      is >> ele;
    }
  }
  is.close();
  return labels;
}
void save_res(const std::vector<std::vector<int>>& res) {
  std::ofstream os ("tmp/search_res.txt");
  for (size_t i = 0; i < res.size(); ++i) {
    for (size_t j = 0; j < global::kCRITERION; ++j) {
      os << res[i][j] << ' ';
    }
    os << '\n';
  }
}
void run(const global::DataSetInfo& info) { 
  /* Unpack the dataset info. */
  auto name = info.name;
  auto n_queries = info.n_queries;
  auto dims = info.dims;
  std::cout << "------------------------------------------------------------\n";
  std::cout << std::format("Running Test: {}\n", name);
  /* Load the dataset. */
  auto base = load_base(info);
  auto samples = load_samples(info);
  auto labels = load_labels(info);
  /* Time the test. */
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::high_resolution_clock;
  Solution solution;
  std::vector<std::vector<int>> res(n_queries,
                                    std::vector<int>(global::kCRITERION));
  auto st = high_resolution_clock::now();
  solution.build(dims, base);
  auto build_ed = high_resolution_clock::now();
  for (int i = 0; i < n_queries; ++i) {
    solution.search(samples[i], res[i].data());
  }
  auto search_ed = high_resolution_clock::now();
  /* Analyze the accuracy. */
  size_t correct_count = 0;
  for (size_t i = 0; i < n_queries; ++i) {
    std::unordered_set<int> res_set(res[i].begin(), res[i].end());
    for (size_t j = 0; j < global::kCRITERION; ++j) {
      if (res_set.count(labels[i][j])) correct_count++;
    }
  }
  double precision = (double)correct_count / (n_queries * global::kCRITERION);
  /* Output the result. */
  std::cout << std::format(
    "Total Time: {}\n",
    duration_cast<duration<double>>(search_ed - st)
  );
  std::cout << std::format(
    "Build Time: {}\n",
    duration_cast<duration<double>>(build_ed - st)
  );
  std::cout << std::format(
    "Total Search Time: {}\n",
    duration_cast<duration<double>>(search_ed - build_ed)
  );
  std::cout << std::format(
    "Average Search Time: {}\n",
    duration_cast<duration<double>>(search_ed - build_ed) / n_queries
  );
  std::cout << std::format("Precision: {}\n", precision);
  if constexpr (global::kDEBUG) save_res(res);
}
