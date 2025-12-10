#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <format>
#include <unordered_set>
#include <string_view>
#include <memory>

#include "Global.h" 
#include "MySolution.h"

namespace {

// --- Data Structure for Pre-loaded Data ---
struct LoadedDataset {
  std::string name;
  int dims;
  int n_queries;
  std::vector<float> base;
  std::vector<std::vector<float>> samples;
  std::vector<std::vector<size_t>> labels;
};

// --- File Loading Utilities ---
std::vector<float> LoadBaseFromFile(const global::DataSetInfo& info) {
  std::cout << "  [IO] Loading Base: " << info.dataset_file << "..." << std::flush;
  std::ifstream is(std::string{info.dataset_file});
  if (!is) throw std::runtime_error("Could not open dataset file: " + std::string{info.dataset_file});
  
  std::vector<float> base(info.dims * info.n_data_points);
  for (auto& ele : base) is >> ele;
  std::cout << " Done." << std::endl;
  return base;
}

std::vector<std::vector<float>> LoadSamplesFromFile(const global::DataSetInfo& info) {
  std::cout << "  [IO] Loading Queries: " << info.sample_file << "..." << std::flush;
  std::ifstream is(std::string{info.sample_file});
  if (!is) throw std::runtime_error("Could not open sample file: " + std::string{info.sample_file});

  std::vector<std::vector<float>> samples(info.n_queries, std::vector<float>(info.dims));
  for (auto& sample : samples) {
    for (auto& dim : sample) is >> dim;
  }
  std::cout << " Done." << std::endl;
  return samples;
}

std::vector<std::vector<size_t>> LoadLabelsFromFile(const global::DataSetInfo& info) {
  std::cout << "  [IO] Loading Labels: " << info.label_file << "..." << std::flush;
  std::ifstream is(std::string{info.label_file});
  if (!is) throw std::runtime_error("Could not open label file: " + std::string{info.label_file});

  std::vector<std::vector<size_t>> labels(info.n_queries, std::vector<size_t>(global::kCRITERION));
  for (auto& label : labels) {
    for (auto& ele : label) is >> ele;
  }
  std::cout << " Done." << std::endl;
  return labels;
}

// Unified Loader
LoadedDataset LoadDataset(const global::DataSetInfo& info) {
    std::cout << "\n>>> Loading Dataset into RAM: " << info.name << " <<<
";
    LoadedDataset ds;
    ds.name = info.name;
    ds.dims = info.dims;
    ds.n_queries = info.n_queries;
    
    ds.base = LoadBaseFromFile(info);
    ds.samples = LoadSamplesFromFile(info);
    ds.labels = LoadLabelsFromFile(info);
    
    std::cout << ">>> Load Complete. Memory Ready. <<<
\n";
    return ds;
}

// --- Test Runner for the Final Solution ---
template<typename Config>
void RunTest(const LoadedDataset& data, std::string_view config_name) {
  if (data.dims != Config::kDim) {
    std::cout << "Skipping mismatch: " << config_name << " (Dim " << data.dims << " vs " << Config::kDim << ")\n";
    return;
  }

  std::cout << "------------------------------------------------------------\n";
  std::cout << std::format("TEST: {:<45}", config_name) << std::endl;
  std::cout << "------------------------------------------------------------\n";

  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;

  MySolution::HNSW<Config> index;
  std::vector<std::vector<int>> results(data.n_queries, std::vector<int>(global::kCRITERION));

  std::cout << "Building Index... " << std::flush;
  auto st = high_resolution_clock::now();
  index.Build(data.base); 
  auto build_ed = high_resolution_clock::now();
  std::cout << "Done." << std::endl;

  std::cout << "Searching...      " << std::flush;
  auto search_st = high_resolution_clock::now();
  for (int i = 0; i < data.n_queries; ++i) {
    index.Search(data.samples[i], global::kCRITERION, results[i].data());
  }
  auto search_ed = high_resolution_clock::now();
  std::cout << "Done." << std::endl;

  size_t correct_count = 0;
  for (size_t i = 0; i < data.n_queries; ++i) {
    std::unordered_set<int> res_set(results[i].begin(), results[i].end());
    for (size_t j = 0; j < global::kCRITERION; ++j) {
      if (res_set.count(data.labels[i][j])) correct_count++;
    }
  }
  double precision = (double)correct_count / (data.n_queries * global::kCRITERION);

  std::cout << std::format("Build Time:       {:.4f} s\n", duration_cast<duration<double>>(build_ed - st).count());
  std::cout << std::format("Avg Search Time:  {:.4f} ms\n", duration_cast<duration<double>>(search_ed - search_st).count() * 1000 / data.n_queries);
  std::cout << std::format("Recall@{{}}:        {{:.4f}}\n", global::kCRITERION, precision);
  std::cout << std::endl;
}

} // namespace

int main() {
  try {
    // --- SIFT BATTLE ---
    {
        LoadedDataset siftData = LoadDataset(global::kSIFT_INFO);
        std::cout << "\n=== FINAL SIFT BENCHMARK ===\n";
        RunTest<MySolution::SIFT_Final_Config>(siftData, "Final Solution | SIFT | Dynamic Search");
    }

    // --- GLOVE BATTLE ---
    {
        LoadedDataset gloveData = LoadDataset(global::kGLOVE_INFO);
        std::cout << "\n=== FINAL GLOVE BENCHMARKS ===\n";
        RunTest<MySolution::GLOVE_Final_Config>(gloveData, "Final Solution | GLOVE | Dynamic Search");
        RunTest<MySolution::GLOVE_OPQ_Hybrid_Config>(gloveData, "Final Solution | GLOVE | OPQ+Dynamic+Rerank");
    }

  } catch (const std::exception& e) {
    std::cerr << "CRITICAL ERROR: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}