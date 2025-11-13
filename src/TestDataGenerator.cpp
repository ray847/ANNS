#include <random> // std::random_device
#include <fstream> // std::ofstream
#include <string> // std::string

#include "Public.h" // kTEST_INFO

int main() {
  std::ofstream os(std::string(kTEST_INFO.dataset_file));
  std::random_device rd{};
  std::mt19937 mt{rd()};
  mt.seed(kSEED);
  std::normal_distribution<float> dis(0.0f, 1.0f);
  for (size_t i = 0; i < kTEST_INFO.n_data_points; ++i) {
    for (size_t j = 0; j < kTEST_INFO.dims; ++j) {
       os << dis(mt) << ' ';
    }
    os << '\n';
  }
  os.close();
  return 0;
}
