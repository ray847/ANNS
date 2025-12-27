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
#include "TunableHNSW/HNSW.h"
#include "TunableHNSW/IVF.h"
#include "TunableHNSW/Config.h"

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
    std::string line;
    std::getline(is, line);
    std::stringstream ss{line};
    for (auto& ele : label) ss >> ele;
  }
  std::cout << " Done." << std::endl;
  return labels;
}

// Unified Loader
LoadedDataset LoadDataset(const global::DataSetInfo& info) {
    std::cout << "\n>>> Loading Dataset into RAM: " << info.name << " <<<" << std::endl;
    LoadedDataset ds;
    ds.name = info.name;
    ds.dims = info.dims;
    ds.n_queries = info.n_queries;
    
    // Load all components
    ds.base = LoadBaseFromFile(info);
    ds.samples = LoadSamplesFromFile(info);
    ds.labels = LoadLabelsFromFile(info);
    
    std::cout << ">>> Load Complete. Memory Ready. <<<\n" << std::endl;
    return ds;
}

// --- Test Runner ---
template<typename Config>
void RunTest(const LoadedDataset& data, std::string_view config_name) {
  // Safety Check: Dimension Mismatch
  if (data.dims != Config::kDimVal) {
    std::cout << "Skipping mismatch: " << config_name << " (Dim " << data.dims << " vs " << Config::kDimVal << ")\n";
    return;
  }

  std::cout << "------------------------------------------------------------\n";
  std::cout << std::format("TEST: {:<45}", config_name) << std::endl;
  std::cout << "------------------------------------------------------------\n";

  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;

  using IndexType = std::conditional_t<
    Config::kIndexStrategyVal == TunableHNSW::IndexStrategy::kIVF_HNSW,
    TunableHNSW::IVF<Config>,
    TunableHNSW::HNSW<Config>
  >;

  auto index = std::make_unique<IndexType>();
  std::vector<std::vector<int>> results(data.n_queries, std::vector<int>(global::kCRITERION));

  std::cout << "Building Index... " << std::flush;
  auto st = high_resolution_clock::now();
  
  // Pass the pre-loaded base vector
  index->Build(data.base); 
  
  auto build_ed = high_resolution_clock::now();
  std::cout << "Done." << std::endl;

  std::cout << "Searching...      " << std::flush;
  auto search_st = high_resolution_clock::now();
  for (int i = 0; i < data.n_queries; ++i) {
    index->Search(data.samples[i], results[i].data());
  }
  auto search_ed = high_resolution_clock::now();
  std::cout << "Done." << std::endl;

  // Compute Metrics
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
  std::cout << std::format("Recall@{}:        {:.4f}\n", global::kCRITERION, precision);
  std::cout << std::endl;
}

}  // namespace

int main() {
  using TunableHNSW::HNSWConfig;
  using TunableHNSW::SearchStrategy;
  using TunableHNSW::QuantizationStrategy;
  using TunableHNSW::IndexStrategy;

  // --- THE "PERFECT 6" RUNS ---

  // *** RUN 1 & 2: SIFT SHOWDOWN (Target: >99%) ***
  // Goal: Prove that Dynamic Search maintains 99% recall faster than Standard.
  
  // 1. SIFT REFERENCE (Standard)
  using SIFT_STD_99 = HNSWConfig<128, true, SearchStrategy::kStandard, QuantizationStrategy::kNone, IndexStrategy::kHNSW, 
                                 32, 64, 500, 300, 16>;

  // 2. SIFT CHALLENGER (Dynamic)
  // Patience=100 ensures we don't drop below 99%.
  using SIFT_DYN_99 = HNSWConfig<128, true, SearchStrategy::kDynamic, QuantizationStrategy::kNone, IndexStrategy::kHNSW, 
                                 32, 64, 500, 500, 16, 100>;


  // *** RUN 3 & 4: THE "COST OF QUALITY" (GloVe) ***
  // Goal: Show how much slower it is to go from 95% to 99% (The "Pareto Frontier").
  
  // 3. GLOVE BASELINE (Target: ~95%)
  // Uses standard params (M=16). Good for comparison.
  using GLOVE_STD_95 = HNSWConfig<100, true, SearchStrategy::kStandard, QuantizationStrategy::kNone, IndexStrategy::kHNSW, 
                                  16, 32, 200, 300, 16>;

  // 4. GLOVE HIGH ACCURACY (Target: >99%)
  // Uses M=48 (Brute Force). Compare this time vs Run 3 to show the cost.
  using GLOVE_STD_99 = HNSWConfig<100, true, SearchStrategy::kStandard, QuantizationStrategy::kNone, IndexStrategy::kHNSW, 
                                  48, 96, 800, 1500, 16>;


  // *** RUN 5 & 6: GLOVE STRATEGY BATTLE (Target: >99%) ***
  // Goal: Can advanced algos beat the Brute Force approach of Run 4?
  
  // 5. GLOVE DYNAMIC (Target: >99%)
  // Can we exit early on easy queries while hitting 99%?
  using GLOVE_DYN_99 = HNSWConfig<100, true, SearchStrategy::kDynamic, QuantizationStrategy::kNone, IndexStrategy::kHNSW, 
                                  48, 96, 800, 1600, 16, 150>;

  // 6. GLOVE PQ + RERANK (Target: ~99%)
  // Compresses graph to fit in cache, but reranks with float. 
  // Fast graph traversal vs. expensive reranking step.
  using GLOVE_PQ_99 = HNSWConfig<100, true, SearchStrategy::kStandard, QuantizationStrategy::kPQ, IndexStrategy::kHNSW, 
                                 48, 96, 800, 1600, 16, 
                                 100, 20, 25000>; // PQ specific params


try {
    // --- SIFT BATTLE ---
    {
        LoadedDataset siftData = LoadDataset(global::kSIFT_INFO);
        std::cout << "=== SIFT RUNS (Target 99%) ===\n";
        RunTest<SIFT_STD_99>(siftData, "[1] SIFT | Standard | 99% Reference");
        RunTest<SIFT_DYN_99>(siftData, "[2] SIFT | Dynamic  | 99% Challenger");
    }

    // --- GLOVE BATTLE ---
    {
        LoadedDataset gloveData = LoadDataset(global::kGLOVE_INFO);
        std::cout << "\n=== GLOVE RUNS (Cost & Strategy) ===\n";
        RunTest<GLOVE_STD_95>(gloveData, "[3] GloVe | Standard | 95% Baseline");
        RunTest<GLOVE_STD_99>(gloveData, "[4] GloVe | Standard | 99% High Accuracy (M=48)");
        RunTest<GLOVE_DYN_99>(gloveData, "[5] GloVe | Dynamic  | 99% Optimization");
        RunTest<GLOVE_PQ_99>(gloveData,  "[6] GloVe | PQ+Rerank| 99% Memory Opt");
    }

  } catch (const std::exception& e) {
    std::cerr << "CRITICAL ERROR: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
