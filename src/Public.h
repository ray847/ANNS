#pragma once

#include <stdint.h> // uint64_t

#include <string_view> // std::stirng_view
#include <array> // std::array

constexpr bool kDEBUG = true;

/**
 * Information on a single data set.
 *
 * Info: file name, dimensions, number of data points.
 */
struct DataSetInfo {
  std::string_view name; //< Dataset Name
  std::string_view dataset_file; //< File name
  std::string_view sample_file; //< File name
  std::string_view label_file; //< File name
  int dims; //< Dimensions
  int n_data_points; //< Number of data points
  int n_queries; //< Number of test queries
                 //< This is directly used by the query generator
};
/* Constants */
constexpr DataSetInfo kGLOVE_INFO{
  "Glove",
  "./data_o/glove/base.txt",
  "./data_o/glove/sample.txt",
  "./data_o/glove/label.txt",
  100,
  1183514,
  100,
};
constexpr DataSetInfo kSIFT_INFO{
  "Sift",
  "./data_o/sift/base.txt",
  "./data_o/sift/smaple.txt",
  "./data_o/sift/label.txt",
  128, 
  1000000,
  100,
};
constexpr DataSetInfo kTEST_INFO {
  "TEST",
  "./data_o/test/base.txt",
  "./data_o/test/sample.txt",
  "./data_o/test/label.txt",
  3,
  1000,
  10,
};
constexpr std::array<DataSetInfo, 3> kDATA_SET_INFOS{
  kGLOVE_INFO,
  kSIFT_INFO,
  kTEST_INFO
};
constexpr uint32_t kSEED = 42; //< Seed for rng
constexpr size_t kCRITERION = 10; //< Number of neighboring vectors to consider
