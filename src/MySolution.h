#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <queue>
#include <cmath>
#include <cstring>
#include <mutex>
#include <memory>
#include <thread>
#include <atomic>
#include <immintrin.h>

class Solution {
private:
    // --- CONSTANTS ---
    static constexpr int M = 64;
    static constexpr int M0 = 256;
    static constexpr int ef_construction = 600;
    static constexpr int ef_search = 300;
    static constexpr int MAX_LEVEL = 8;

    // --- DATA ---
    int d_ = 0;
    size_t n_ = 0;
    
    // Scalar Quantization Data
    std::vector<uint16_t> data_sq_;
    const uint16_t* data_sq_ptr_ = nullptr;
    
    // Quantization Parameters (per dimension)
    std::vector<float> min_vals_;
    std::vector<float> scale_vals_;    // (max - min) / 65535
    std::vector<float> scale_sq_vals_; // scale_vals_ ^ 2

    struct Node {
        int level;
        std::vector<int> flat_links;
        std::vector<int> link_counts;
        std::unique_ptr<std::mutex> lock;
    };
    std::vector<Node> nodes_;

    int entry_point_ = -1;
    int max_level_ = -1;
    double level_mult_;
    std::mutex global_lock_;

    // --- HELPER STRUCT: VISITED LIST ---
    struct VisitedList {
        std::vector<unsigned short> tags;
        unsigned short current_tag = 0;

        void resize(size_t n) { tags.resize(n, 0); }

        void advance() {
            current_tag++;
            if (current_tag == 0) {
                std::fill(tags.begin(), tags.end(), 0);
                current_tag = 1;
            }
        }

        inline bool visit(int id) {
            if (tags[id] == current_tag) return true;
            tags[id] = current_tag;
            return false;
        }
    };

public:
    // --- BUILD FUNCTION ---
    void build(int d, const std::vector<float>& base) {
        d_ = d;
        n_ = base.size() / d_;
        
        // 1. Compute Statistics (Min, Max) for Quantization
        min_vals_.assign(d, std::numeric_limits<float>::max());
        std::vector<float> max_vals(d, std::numeric_limits<float>::lowest());
        
        for (size_t i = 0; i < n_; ++i) {
            const float* vec = base.data() + i * d;
            for (int j = 0; j < d; ++j) {
                if (vec[j] < min_vals_[j]) min_vals_[j] = vec[j];
                if (vec[j] > max_vals[j]) max_vals[j] = vec[j];
            }
        }
        
        scale_vals_.resize(d);
        scale_sq_vals_.resize(d);
        for (int j = 0; j < d; ++j) {
            float range = max_vals[j] - min_vals_[j];
            if (range < 1e-9f) range = 1e-9f; // Avoid division by zero
            scale_vals_[j] = range / 65535.0f;
            scale_sq_vals_[j] = scale_vals_[j] * scale_vals_[j];
        }

        // 2. Quantize Data
        data_sq_.resize(n_ * d);
        data_sq_ptr_ = data_sq_.data();
        
        for (size_t i = 0; i < n_; ++i) {
            const float* src = base.data() + i * d;
            uint16_t* dst = data_sq_.data() + i * d;
            for (int j = 0; j < d; ++j) {
                float val = (src[j] - min_vals_[j]) / scale_vals_[j];
                dst[j] = static_cast<uint16_t>(std::round(val));
            }
        }

        // 3. Initialize Graph
        nodes_.resize(n_);
        level_mult_ = 1.0 / std::log(1.0 * M);

        std::mt19937 rng_init(42);
        for (size_t i = 0; i < n_; ++i) {
            int level = get_random_level(rng_init);
            nodes_[i].level = level;
            nodes_[i].link_counts.resize(level + 1, 0);
            nodes_[i].lock = std::make_unique<std::mutex>();
            size_t total_links = M0;
            if (level > 0) total_links += (size_t)level * M;
            nodes_[i].flat_links.resize(total_links);
        }

        entry_point_ = 0;
        max_level_ = nodes_[0].level;

        std::atomic<size_t> atomic_idx{ 1 };
        
        unsigned int num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;

        auto worker_func = [&](int thread_id) {
            VisitedList visited;
            visited.resize(n_);
            std::mt19937 rng(42 + thread_id);

            while (true) {
                size_t curr_obj = atomic_idx.fetch_add(1, std::memory_order_relaxed);
                if (curr_obj >= n_) break;

                int curr_level = nodes_[curr_obj].level;
                int curr_ep = entry_point_;
                int curr_max_level = max_level_;
                
                // Note: We use the *quantized* data for distance calculations during build too
                // to act as the source of truth, though we could pass the float vector here.
                // However, the distance functions are now updated to use stored SQ data.
                // For the "current vector", we need to be careful.
                // dist_query_sq expects a float query. We can recover the float approximation
                // of curr_obj or use the original base if we had access.
                // To keep it simple and consistent, we'll dequantize curr_obj on the fly 
                // into a thread-local float buffer or use a specific dist function.
                
                // Let's create a temporary float buffer for the current object to act as "query"
                // Optimization: The dist_query_sq now expects a transformed query (q-min)/scale.
                // For the current object (which is a node), the transformed query is simply its SQ values cast to float.
                std::vector<float> curr_vec_trans(d_);
                const uint16_t* curr_sq = data_sq_ptr_ + curr_obj * d_;
                for(int j=0; j<d_; ++j) {
                     curr_vec_trans[j] = (float)curr_sq[j];
                }

                // 1. Descent (Greedy)
                for (int l = curr_max_level; l > curr_level; l--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        float dist_ep = dist_query_sq(curr_vec_trans.data(), curr_ep);
                        const Node& node_ep = nodes_[curr_ep];
                        // Safety check for concurrency race on max_level
                        if (l >= node_ep.link_counts.size()) break;

                        int offset = get_link_offset(l);
                        int count = node_ep.link_counts[l];
                        const int* links = node_ep.flat_links.data() + offset;

                        for (int j = 0; j < count; ++j) {
                            int neighbor = links[j];
                            float d = dist_query_sq(curr_vec_trans.data(), neighbor);
                            if (d < dist_ep) {
                                curr_ep = neighbor;
                                dist_ep = d;
                                changed = true;
                            }
                        }
                    }
                }

                // 2. Construction
                for (int l = std::min(curr_level, curr_max_level); l >= 0; l--) {
                    auto top_candidates = search_layer_build(curr_vec_trans.data(), curr_ep, ef_construction, l, visited);
                    std::vector<std::pair<float, int>> potential;
                    potential.reserve(ef_construction + 1);
                    while (!top_candidates.empty()) {
                        potential.push_back(top_candidates.top());
                        top_candidates.pop();
                    }

                    int offset = get_link_offset(l);
                    int* link_dst = nodes_[curr_obj].flat_links.data() + offset;
                    int count = 0;
                    get_neighbors_heuristic(curr_obj, potential, l, link_dst, count);
                    nodes_[curr_obj].link_counts[l] = count;

                    for (int j = 0; j < count; ++j) add_connection(link_dst[j], curr_obj, l);
                    if (!potential.empty()) curr_ep = potential[0].second;
                }

                if (curr_level > max_level_) {
                    std::lock_guard<std::mutex> lock(global_lock_);
                    if (curr_level > max_level_) {
                        max_level_ = curr_level;
                        entry_point_ = curr_obj;
                    }
                }
            }
        };

        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for (unsigned int i = 0; i < num_threads; ++i) {
            threads.emplace_back(worker_func, i);
        }

        for (auto& t : threads) {
            t.join();
        }
    }

    // --- SEARCH FUNCTION (Instrumented) ---
    void search(const std::vector<float>& query, int* res) {
        static thread_local VisitedList visited;
        if (visited.tags.size() != n_) visited.resize(n_);

        int curr_ep = entry_point_;
        const float* q_data = query.data();
        
        // Transform query: q_trans = (q - min) / scale
        std::vector<float> q_trans(d_);
        for(int i = 0; i < d_; ++i) {
            q_trans[i] = (q_data[i] - min_vals_[i]) / scale_vals_[i];
        }
        
        float cur_dist = dist_query_sq(q_trans.data(), curr_ep);

        visited.advance();
        visited.visit(curr_ep);

        // --- Phase 1: Upper Layers (Greedy Descent) ---
        for (int l = max_level_; l > 0; l--) {
            bool changed = true;
            while (changed) {
                changed = false;
                const Node& node = nodes_[curr_ep];
                if (l >= node.link_counts.size()) break;

                int count = node.link_counts[l];
                int offset = get_link_offset(l);
                const int* links = node.flat_links.data() + offset;

                for (int i = 0; i < count; ++i) {
                    int neighbor = links[i];
                    if (i + 1 < count) _mm_prefetch((const char*)(data_sq_ptr_ + links[i + 1] * d_), _MM_HINT_T0);

                    float d = dist_query_sq(q_trans.data(), neighbor);
                    if (d < cur_dist) {
                        cur_dist = d;
                        curr_ep = neighbor;
                        changed = true;
                    }
                }
            }
        }

        // --- Phase 2: Layer 0 (Fine-grained Search) ---
        using QueueItem = std::pair<float, int>;
        std::priority_queue<QueueItem> top_candidates;
        std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> candidates;

        top_candidates.push({ cur_dist, curr_ep });
        candidates.push({ cur_dist, curr_ep });
        visited.visit(curr_ep);

        while (!candidates.empty()) {
            auto [c_dist, c_id] = candidates.top();
            candidates.pop();

            if (c_dist > top_candidates.top().first && top_candidates.size() >= ef_search) break;

            const Node& node = nodes_[c_id];
            int size = node.link_counts[0];
            int offset = get_link_offset(0);
            const int* links = node.flat_links.data() + offset;

            for (int i = 0; i < size; ++i) {
                int neighbor_id = links[i];
                if (i + 1 < size) _mm_prefetch((const char*)(data_sq_ptr_ + links[i + 1] * d_), _MM_HINT_T0);

                if (!visited.visit(neighbor_id)) {
                    float d = dist_query_sq(q_trans.data(), neighbor_id);
                    if (top_candidates.size() < ef_search || d < top_candidates.top().first) {
                        candidates.push({ d, neighbor_id });
                        top_candidates.push({ d, neighbor_id });
                        if (top_candidates.size() > ef_search) top_candidates.pop();
                    }
                }
            }
        }

        // --- Fill Results ---
        size_t k_idx = 0;
        std::vector<std::pair<float, int>> sorted;
        while (!top_candidates.empty()) {
            sorted.push_back(top_candidates.top());
            top_candidates.pop();
        }
        std::sort(sorted.begin(), sorted.end());

        float final_dist = sorted.empty() ? -1.0f : sorted[0].first;
        
        for (const auto& p : sorted) {
            if (k_idx >= 10) break;
            res[k_idx++] = p.second;
        }
        while (k_idx < 10) res[k_idx++] = -1;
    }

private:
    // --- AVX2 Distance (Transformed Query Float vs Node SQ16) ---
    // Expects q_trans[i] = (q[i] - min[i]) / scale[i]
    // Computes Sum( (q_trans[i] - node[i])^2 * scale_sq[i] )
    __attribute__((target("avx2,fma")))
    inline float dist_query_sq(const float* q_trans, int id_node) const {
        const uint16_t* node_ptr = data_sq_ptr_ + id_node * d_;
        const float* scale_sq_ptr = scale_sq_vals_.data();
        
        __m256 sum = _mm256_setzero_ps();
        int d = d_;
        int i = 0;
        
        // Process 8 elements at a time
        for (; i <= d - 8; i += 8) {
            // Load 8 uint16 values
            __m128i v_u16 = _mm_loadu_si128((const __m128i*)(node_ptr + i));
            // Convert to 8 floats
            __m256 v_f32_node = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(v_u16));
            
            // Load transformed query
            __m256 v_q = _mm256_loadu_ps(q_trans + i);
            
            // Diff: q_trans - node
            __m256 diff = _mm256_sub_ps(v_q, v_f32_node);
            
            // Square: diff * diff
            __m256 diff_sq = _mm256_mul_ps(diff, diff);
            
            // Load scale squared
            __m256 v_scale_sq = _mm256_loadu_ps(scale_sq_ptr + i);
            
            // Weighted sum
            sum = _mm256_fmadd_ps(diff_sq, v_scale_sq, sum);
        }
        
        // Reduction
        __m128 sum_low = _mm256_castps256_ps128(sum);
        __m128 sum_high = _mm256_extractf128_ps(sum, 1);
        __m128 v_res = _mm_add_ps(sum_low, sum_high);
        v_res = _mm_hadd_ps(v_res, v_res);
        v_res = _mm_hadd_ps(v_res, v_res);
        float res = _mm_cvtss_f32(v_res);
        
        // Handle remainder
        for (; i < d; ++i) {
            float val = (float)node_ptr[i];
            float diff = q_trans[i] - val;
            res += diff * diff * scale_sq_ptr[i];
        }
        return res;
    }

    // --- AVX2 Distance (Node SQ16 vs Node SQ16) ---
    // Used during build for heuristic neighbor selection
    __attribute__((target("avx2,fma")))
    inline float dist_sq(int id_a, int id_b) const {
        const uint16_t* ptr_a = data_sq_ptr_ + id_a * d_;
        const uint16_t* ptr_b = data_sq_ptr_ + id_b * d_;
        const float* scale_sq_ptr = scale_sq_vals_.data();
        
        __m256 sum = _mm256_setzero_ps();
        int d = d_;
        int i = 0;
        
        for (; i <= d - 8; i += 8) {
            __m128i v_u16_a = _mm_loadu_si128((const __m128i*)(ptr_a + i));
            __m128i v_u16_b = _mm_loadu_si128((const __m128i*)(ptr_b + i));
            
            __m256 v_a = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(v_u16_a));
            __m256 v_b = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(v_u16_b));
            
            __m256 diff = _mm256_sub_ps(v_a, v_b);
            __m256 diff_sq = _mm256_mul_ps(diff, diff);
            
            __m256 v_scale_sq = _mm256_loadu_ps(scale_sq_ptr + i);
            
            sum = _mm256_fmadd_ps(diff_sq, v_scale_sq, sum);
        }
        
        __m128 sum_low = _mm256_castps256_ps128(sum);
        __m128 sum_high = _mm256_extractf128_ps(sum, 1);
        __m128 v_res = _mm_add_ps(sum_low, sum_high);
        v_res = _mm_hadd_ps(v_res, v_res);
        v_res = _mm_hadd_ps(v_res, v_res);
        float res = _mm_cvtss_f32(v_res);
        
        for (; i < d; ++i) {
            float d_val = (float)ptr_a[i] - (float)ptr_b[i];
            res += d_val * d_val * scale_sq_ptr[i];
        }
        return res;
    }


    inline int get_link_offset(int level) const {
        return (level == 0) ? 0 : (M0 + (level - 1) * M);
    }

    int get_random_level(std::mt19937& rng) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double r = -std::log(dist(rng)) * level_mult_;
        return std::min(static_cast<int>(r), MAX_LEVEL);
    }

    // Standard search for build phase (isolated from the instrumented public search)
    std::priority_queue<std::pair<float, int>> search_layer_build(
        const float* query_data, int entry_point, int ef, int level, VisitedList& visited
    ) {
        using QueueItem = std::pair<float, int>;
        std::priority_queue<QueueItem> top_candidates;
        std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> candidates;

        visited.advance();
        float initial_dist = dist_query_sq(query_data, entry_point);
        top_candidates.push({ initial_dist, entry_point });
        candidates.push({ initial_dist, entry_point });
        visited.visit(entry_point);

        while (!candidates.empty()) {
            auto [curr_dist, curr_id] = candidates.top();
            if (curr_dist > top_candidates.top().first && top_candidates.size() >= ef) break;
            candidates.pop();

            const Node& node = nodes_[curr_id];
            int size = node.link_counts[level];
            int offset = get_link_offset(level);
            const int* links = node.flat_links.data() + offset;

            for (int i = 0; i < size; ++i) {
                int neighbor_id = links[i];
                if (i + 1 < size) _mm_prefetch((const char*)(data_sq_ptr_ + links[i + 1] * d_), _MM_HINT_T0);

                if (!visited.visit(neighbor_id)) {
                    float d = dist_query_sq(query_data, neighbor_id);
                    if (top_candidates.size() < ef || d < top_candidates.top().first) {
                        candidates.push({ d, neighbor_id });
                        top_candidates.push({ d, neighbor_id });
                        if (top_candidates.size() > ef) top_candidates.pop();
                    }
                }
            }
        }
        return top_candidates;
    }

    void get_neighbors_heuristic(
        int src,
        std::vector<std::pair<float, int>>& candidates,
        int level,
        int* output_buffer,
        int& output_count
    ) {
        int max_m = (level == 0) ? M0 : M;
        output_count = 0;
        if (candidates.empty()) return;
        std::sort(candidates.begin(), candidates.end());

        for (const auto& cand : candidates) {
            if (output_count >= max_m) break;
            int cand_id = cand.second;
            float dist_to_src = cand.first;
            bool good = true;
            for (int j = 0; j < output_count; ++j) {
                if (dist_sq(cand_id, output_buffer[j]) < dist_to_src) {
                    good = false; break;
                }
            }
            if (good) output_buffer[output_count++] = cand_id;
        }
    }

    void add_connection(int src, int dest, int level) {
        Node& node = nodes_[src];
        std::lock_guard<std::mutex> lock(*node.lock);

        int count = node.link_counts[level];
        int offset = get_link_offset(level);
        int* links_ptr = node.flat_links.data() + offset;

        for (int i = 0; i < count; ++i) if (links_ptr[i] == dest) return;

        int max_m = (level == 0) ? M0 : M;

        if (count < max_m) {
            // Strict sorting insert
            float dest_dist = dist_sq(src, dest);
            int insert_pos = count;
            for (int i = 0; i < count; ++i) {
                if (dist_sq(src, links_ptr[i]) > dest_dist) {
                    insert_pos = i; break;
                }
            }
            for (int j = count; j > insert_pos; --j) links_ptr[j] = links_ptr[j - 1];
            links_ptr[insert_pos] = dest;
            __atomic_store_n(&node.link_counts[level], count + 1, __ATOMIC_RELEASE);
        }
        else {
            std::vector<std::pair<float, int>> candidates;
            candidates.reserve(max_m + 1);
            for (int i = 0; i < count; ++i) candidates.push_back({ dist_sq(src, links_ptr[i]), links_ptr[i] });
            candidates.push_back({ dist_sq(src, dest), dest });

            std::vector<int> new_links(max_m);
            int new_count = 0;
            get_neighbors_heuristic(src, candidates, level, new_links.data(), new_count);

            for (int i = 0; i < new_count; ++i) links_ptr[i] = new_links[i];
            if (new_count != count) __atomic_store_n(&node.link_counts[level], new_count, __ATOMIC_RELEASE);
        }
    }
};
