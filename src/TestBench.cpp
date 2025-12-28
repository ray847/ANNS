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
    std::cout << "\n>>> Loading Dataset into RAM: " << info.name << " <<<";
    LoadedDataset ds;
    ds.name = info.name;
    ds.dims = info.dims;
    ds.n_queries = info.n_queries;
    
    // Load all components
    ds.base = LoadBaseFromFile(info);
    ds.samples = LoadSamplesFromFile(info);
    ds.labels = LoadLabelsFromFile(info);
    
    std::cout << ">>> Load Complete. Memory Ready. <<<\n";
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

  // --- SIFT Configurations (Dim 128) ---
  using SIFT_DEFAULT = HNSWConfig<128, true, SearchStrategy::kStandard, QuantizationStrategy::kNone, IndexStrategy::kHNSW, 32, 64, 300, 100, 16>;
  using SIFT_DYNAMIC = HNSWConfig<128, true, SearchStrategy::kDynamic, QuantizationStrategy::kNone, IndexStrategy::kHNSW, 32, 64, 300, 150, 16, 75>;
  using SIFT_PQ      = HNSWConfig<128, true, SearchStrategy::kStandard, QuantizationStrategy::kPQ, IndexStrategy::kHNSW, 32, 64, 300, 200, 16, 100, 16, 25000>;
  using SIFT_OPQ     = HNSWConfig<128, true, SearchStrategy::kStandard, QuantizationStrategy::kOPQ, IndexStrategy::kHNSW, 32, 64, 300, 185, 16, 100, 16, 25000>;
  using SIFT_SQ      = HNSWConfig<128, true, SearchStrategy::kStandard, QuantizationStrategy::kSQ, IndexStrategy::kHNSW, 32, 64, 300, 100, 16>;
  using SIFT_IVF     = HNSWConfig<128, true, SearchStrategy::kStandard, QuantizationStrategy::kNone, IndexStrategy::kIVF_HNSW, 32, 64, 300, 150, 16, 100, 0, 0, 1024, 128, 25000>;
  using SIFT_PMR     = HNSWConfig<128, true, SearchStrategy::kStandard, QuantizationStrategy::kNone, IndexStrategy::kHNSW, 32, 64, 300, 100, 16, 100, 0, 0, 0, 0, 0, true>;

  // --- GloVe Configurations (Dim 100) ---
  // M = 48, M0 = 96, EfConstruction = 600, EfSearch = 450, MaxLayers = 16
  using GLOVE_DEFAULT = HNSWConfig<100, true, SearchStrategy::kStandard, QuantizationStrategy::kNone, IndexStrategy::kHNSW, 48, 96, 600, 400, 16>;
  using GLOVE_DYNAMIC = HNSWConfig<100, true, SearchStrategy::kDynamic, QuantizationStrategy::kNone, IndexStrategy::kHNSW, 48, 96, 600, 450, 16, 200>;
  using GLOVE_PQ      = HNSWConfig<100, true, SearchStrategy::kStandard, QuantizationStrategy::kPQ, IndexStrategy::kHNSW, 48, 96, 600, 500, 16, 100, 20, 25000>;
  using GLOVE_OPQ     = HNSWConfig<100, true, SearchStrategy::kStandard, QuantizationStrategy::kOPQ, IndexStrategy::kHNSW, 48, 96, 600, 500, 16, 100, 20, 25000>;
  using GLOVE_SQ      = HNSWConfig<100, true, SearchStrategy::kStandard, QuantizationStrategy::kSQ, IndexStrategy::kHNSW, 48, 96, 600, 300, 16>;
  using GLOVE_IVF     = HNSWConfig<100, true, SearchStrategy::kStandard, QuantizationStrategy::kNone, IndexStrategy::kIVF_HNSW, 48, 96, 600, 600, 16, 100, 0, 0, 1024, 256, 25000>;
  using GLOVE_PMR     = HNSWConfig<100, true, SearchStrategy::kStandard, QuantizationStrategy::kNone, IndexStrategy::kHNSW, 48, 96, 600, 400, 16, 100, 0, 0, 0, 0, 0, true>;

  try {
    // --- SIFT RUNS ---
    {
        LoadedDataset siftData = LoadDataset(global::kMINISIFT_INFO);
        std::cout << "=== SIFT RUNS (Target 99%) ===\n";
        //RunTest<SIFT_DEFAULT>(siftData, "[1] SIFT | Default HNSW");
        //RunTest<SIFT_DYNAMIC>(siftData, "[2] SIFT | Dynamic Search");
        //RunTest<SIFT_PQ>(siftData,      "[3] SIFT | PQ Enabled");
        //RunTest<SIFT_OPQ>(siftData,     "[4] SIFT | OPQ Enabled");
        //RunTest<SIFT_SQ>(siftData,      "[5] SIFT | SQ Enabled");
        //RunTest<SIFT_IVF>(siftData,     "[6] SIFT | IVF Solution");
        RunTest<SIFT_PMR>(siftData,     "[7] SIFT | PMR Enabled");
    }

    // --- GLOVE RUNS ---
    {
        LoadedDataset gloveData = LoadDataset(global::kMINIGLOVE_INFO);
        std::cout << "\n=== GLOVE RUNS (Target 98%) ===\n";
        //RunTest<GLOVE_DEFAULT>(gloveData, "[1] GloVe | Default HNSW");
        //RunTest<GLOVE_DYNAMIC>(gloveData, "[2] GloVe | Dynamic Search");
        //RunTest<GLOVE_PQ>(gloveData,      "[3] GloVe | PQ Enabled");
        //RunTest<GLOVE_OPQ>(gloveData,     "[4] GloVe | OPQ Enabled");
        //RunTest<GLOVE_SQ>(gloveData,      "[5] GloVe | SQ Enabled");
        //RunTest<GLOVE_IVF>(gloveData,     "[6] GloVe | IVF Solution");
        //RunTest<GLOVE_PMR>(gloveData,     "[7] GloVe | PMR Enabled");
    }

  } catch (const std::exception& e) {
    std::cerr << "CRITICAL ERROR: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
