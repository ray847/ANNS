#pragma once

// A simple, header-only configuration file to hold the optimal parameters
// for the final integrated HNSW solution.

namespace MySolution {

struct SIFT_Config {
    static constexpr int kDim = 128;
    static constexpr int kM = 32;
    static constexpr int kM0 = 64;
    static constexpr int kEfConstruction = 500;
    
    // Dynamic Search Params
    static constexpr int kPatience = 100;
    static constexpr int kMaxEfSearch = 500;
};

struct GLOVE_Config {
    static constexpr int kDim = 100;
    static constexpr int kM = 48;
    static constexpr int kM0 = 96;
    static constexpr int kEfConstruction = 800;
    
    // Dynamic Search Params
    static constexpr int kPatience = 200;
    static constexpr int kMaxEfSearch = 2000;
};

// --- Toggle this typedef to switch the entire engine ---
using CurrentConfig = SIFT_Config; 
// using CurrentConfig = GLOVE_Config;

} // namespace MySolution
