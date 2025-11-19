#include <random> // std::random_device
#include <fstream> // std::ofstream
#include <string> // std::string

#include "Global.h" // Global::kTEST_INFO

int main() {
  std::ofstream os(std::string(global::kTEST_INFO.dataset_file));
  std::normal_distribution<float> dis(0.0f, 1.0f);
  for (size_t i = 0; i < global::kTEST_INFO.n_data_points; ++i) {
    for (size_t j = 0; j < global::kTEST_INFO.dims; ++j) {
       os << dis(global::rng) << ' ';
    }
    os << '\n';
  }
  os.close();
  return 0;
}
