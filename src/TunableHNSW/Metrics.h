#pragma once

#include <vector>
#include <chrono>
#include <atomic>
#include "../Global.h" // For global::kDEBUG

namespace TunableHNSW {

template<int MaxLevel_>
struct HNSWMetrics {
    static constexpr int MaxLevel = MaxLevel_;
    std::atomic<long long> distance_calculations{0};
    std::vector<std::atomic<long long>> navigation_time_per_layer;

    HNSWMetrics() : navigation_time_per_layer(MaxLevel + 1) {
        reset();
    }

    void reset() {
        if constexpr (global::kDEBUG) {
            distance_calculations = 0;
            for (int i = 0; i <= MaxLevel; ++i) {
                navigation_time_per_layer[i] = 0;
            }
        }
    }

    inline void increment_distance_calculations(long long count = 1) {
        if constexpr (global::kDEBUG) {
            distance_calculations.fetch_add(count, std::memory_order_relaxed);
        }
    }

    inline void add_navigation_time(int layer, long long nanoseconds) {
        if constexpr (global::kDEBUG) {
            if (layer >= 0 && layer <= MaxLevel) {
                navigation_time_per_layer[layer].fetch_add(nanoseconds, std::memory_order_relaxed);
            }
        }
    }
};

} // namespace TunableHNSW