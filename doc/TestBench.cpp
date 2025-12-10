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
    for (auto& ele : label) is >> ele;
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

  // --- TUNED CONFIGURATIONS ---

  // *** SIFT CONFIGURATIONS ***
  // Baseline: Standard HNSW (High Accuracy)
  using SIFT_HNSW_SIMD = HNSWConfig<128, true, SearchStrategy::kStandard, QuantizationStrategy::kNone, IndexStrategy::kHNSW, 16, 32, 200, 100, 16>;
  
  // Baseline 2: Fast Standard HNSW (Lower Accuracy Baseline)
  // Helps compare if Dynamic is truly better or just "looser"
  using SIFT_HNSW_SIMD_FAST = HNSWConfig<128, true, SearchStrategy::kStandard, QuantizationStrategy::kNone, IndexStrategy::kHNSW, 16, 32, 200, 40, 16>;

  // Dynamic: Increased Patience to 64 (was 20) to restore Recall
  using SIFT_HNSW_SIMD_DYN = HNSWConfig<128, true, SearchStrategy::kDynamic, QuantizationStrategy::kNone, IndexStrategy::kHNSW, 16, 32, 200, 400, 16, 64>;

  // PQ: Standard (Approximate)
  using SIFT_PQ_SIMD = HNSWConfig<128, true, SearchStrategy::kStandard, QuantizationStrategy::kPQ, IndexStrategy::kHNSW, 16, 32, 200, 100, 16, 100, 16, 25000>;
  
  // IVF: Increased NProbe to 256 (was 64) for better recall
  using SIFT_IVF_SIMD = HNSWConfig<128, true, SearchStrategy::kStandard, QuantizationStrategy::kNone, IndexStrategy::kIVF_HNSW, 16, 32, 200, 100, 16, 100, 0, 0, 4096, 256, 100000>;
  using SIFT_IVF_SIMD_DYN = HNSWConfig<128, true, SearchStrategy::kDynamic, QuantizationStrategy::kNone, IndexStrategy::kIVF_HNSW, 16, 32, 200, 400, 16, 64, 0, 0, 4096, 256, 100000>;
  
  using SIFT_IVF_PQ_SIMD = HNSWConfig<128, true, SearchStrategy::kStandard, QuantizationStrategy::kPQ, IndexStrategy::kIVF_HNSW, 16, 32, 200, 100, 16, 100, 16, 25000, 4096, 256, 100000>;


  // *** GLOVE CONFIGURATIONS ***
  using GLOVE_HNSW_SIMD = HNSWConfig<100, true, SearchStrategy::kStandard, QuantizationStrategy::kNone, IndexStrategy::kHNSW, 16, 32, 200, 100, 16>;
  
  // Dynamic: Increased Patience to 64 (was 30) for GloVe difficulty
  using GLOVE_HNSW_SIMD_DYN = HNSWConfig<100, true, SearchStrategy::kDynamic, QuantizationStrategy::kNone, IndexStrategy::kHNSW, 16, 32, 200, 400, 16, 64>;

  using GLOVE_PQ_SIMD = HNSWConfig<100, true, SearchStrategy::kStandard, QuantizationStrategy::kPQ, IndexStrategy::kHNSW, 16, 32, 200, 100, 16, 100, 20, 25000>;

  // IVF: Increased NProbe to 256
  using GLOVE_IVF_SIMD = HNSWConfig<100, true, SearchStrategy::kStandard, QuantizationStrategy::kNone, IndexStrategy::kIVF_HNSW, 16, 32, 200, 100, 16, 100, 0, 0, 4096, 256, 118351>;
  using GLOVE_IVF_SIMD_DYN = HNSWConfig<100, true, SearchStrategy::kDynamic, QuantizationStrategy::kNone, IndexStrategy::kIVF_HNSW, 16, 32, 200, 400, 16, 64, 0, 0, 4096, 256, 118351>;
  
  using GLOVE_IVF_PQ_SIMD = HNSWConfig<100, true, SearchStrategy::kStandard, QuantizationStrategy::kPQ, IndexStrategy::kIVF_HNSW, 16, 32, 200, 100, 16, 100, 20, 25000, 4096, 256, 118351>;


  try {
    // --- PART 1: Run SIFT Tests ---
    {
        LoadedDataset siftData = LoadDataset(global::kSIFT_INFO);
        
        std::cout << "=== RUNNING SIFT BENCHMARK ===\n";
        RunTest<SIFT_HNSW_SIMD>(siftData, "SIFT | HNSW | SIMD | Std (High Recall)");
        RunTest<SIFT_HNSW_SIMD_FAST>(siftData, "SIFT | HNSW | SIMD | Std (Fast Baseline)"); // New check
        RunTest<SIFT_HNSW_SIMD_DYN>(siftData, "SIFT | HNSW | SIMD | Dynamic (Patience=64)");

        RunTest<SIFT_PQ_SIMD>(siftData, "SIFT | HNSW | PQ | SIMD | Std");

        RunTest<SIFT_IVF_SIMD>(siftData, "SIFT | HNSW | IVF | SIMD | Std (Probe=256)");
        RunTest<SIFT_IVF_SIMD_DYN>(siftData, "SIFT | HNSW | IVF | SIMD | Dynamic");

        RunTest<SIFT_IVF_PQ_SIMD>(siftData, "SIFT | HNSW | IVF | PQ | SIMD | Std");
    }

    // --- PART 2: Run GloVe Tests ---
    {
        LoadedDataset gloveData = LoadDataset(global::kGLOVE_INFO);

        std::cout << "\n=== RUNNING GLOVE BENCHMARK ===\n";
        RunTest<GLOVE_HNSW_SIMD>(gloveData, "GloVe | HNSW | SIMD | Std");
        RunTest<GLOVE_HNSW_SIMD_DYN>(gloveData, "GloVe | HNSW | SIMD | Dynamic (Patience=64)");

        RunTest<GLOVE_PQ_SIMD>(gloveData, "GloVe | HNSW | PQ | SIMD | Std");

        RunTest<GLOVE_IVF_SIMD>(gloveData, "GloVe | HNSW | IVF | SIMD | Std (Probe=256)");
        RunTest<GLOVE_IVF_SIMD_DYN>(gloveData, "GloVe | HNSW | IVF | SIMD | Dynamic");

        RunTest<GLOVE_IVF_PQ_SIMD>(gloveData, "GloVe | HNSW | IVF | PQ | SIMD | Std");
    }

  } catch (const std::exception& e) {
    std::cerr << "CRITICAL ERROR: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
