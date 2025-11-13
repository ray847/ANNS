cmake_minimum_required(VERSION 3.15)

project(VectorSearch)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(main
  src/main.cpp
)

add_executable(query_generator
  src/QueryGenerator.cpp
)

add_executable(test_data_genrator
  src/TestDataGenerator.cpp
)
