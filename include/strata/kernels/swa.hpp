// include/strata/kernels/swa.hpp - High-Performance Sliding Window Attention (SWA) Optimization Engine
//
// Optimizes Sliding Window Attention (e.g. MiMo-V2.6, Mistral, Long-Context Hybrid Models):
// 1. Circular Ring-Buffer KV Cache: Stores only active rolling window W (e.g. 4096 tokens),
//    yielding up to 250x KV cache memory reduction over 1M+ token contexts (O(W) vs O(T)).
// 2. Causal Range & Block-Tile Pruning: Skips computation and memory loads outside [pos - W, pos].
// 3. Fast Vectorized / Bounded Softmax Decoding: Reduces decode attention latency to constant O(W).
#pragma once

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::kernels {

struct SwaWindowConfig {
    int64_t window_size = 4096;
    int64_t n_head = 48;
    int64_t n_head_kv = 8;
    int64_t head_dim = 128;
    float scale = 0.0883883476f; // 1.0f / sqrt(128)

    int64_t kv_stride_per_token() const {
        return n_head_kv * head_dim;
    }

    int64_t q_stride_per_token() const {
        return n_head * head_dim;
    }
};

struct SwaCausalRange {
    int64_t start_pos = 0;
    int64_t end_pos = 0;       // inclusive
    int64_t active_length = 0;

    bool is_in_window(int64_t key_pos) const {
        return key_pos >= start_pos && key_pos <= end_pos;
    }
};

// ------------------- Circular Ring-Buffer KV Cache -------------------
class SlidingWindowKVCache {
public:
    explicit SlidingWindowKVCache(const SwaWindowConfig& cfg);
    ~SlidingWindowKVCache();

    // Store a single token's KV projection into the circular ring buffer
    void append(int64_t token_pos, const float* k_token, const float* v_token);

    // Retrieve physical circular slot index for a given global logical token position
    inline int64_t get_ring_slot(int64_t token_pos) const {
        return token_pos % cfg_.window_size;
    }

    // Read specific key/value vectors from the circular buffer
    const float* get_key(int64_t token_pos) const;
    const float* get_value(int64_t token_pos) const;

    // Direct pointers to internal ring buffers
    const float* key_buffer() const { return k_ring_.data(); }
    const float* value_buffer() const { return v_ring_.data(); }

    const SwaWindowConfig& config() const { return cfg_; }
    int64_t total_tokens_appended() const { return total_tokens_; }
    int64_t active_window_tokens() const;

    // Memory footprint & savings
    size_t memory_bytes() const;
    double memory_reduction_ratio(int64_t total_tokens) const;

    void reset();

private:
    SwaWindowConfig cfg_;
    int64_t total_tokens_ = 0;
    int64_t slot_stride_ = 0;

    // Contiguous fixed-size ring buffers: [window_size * n_head_kv * head_dim]
    std::vector<float> k_ring_;
    std::vector<float> v_ring_;
};

// ------------------- SWA Execution & Pruning Engine -------------------
class SWAExecutionEngine {
public:
    SWAExecutionEngine() = default;

    // Calculate causal sliding-window boundary for query_pos
    static SwaCausalRange compute_causal_range(int64_t query_pos, int64_t window_size);

    // Block-level tile pruning for 2D prefill attention matrices
    // Returns true if tile (row_range x col_range) can be completely skipped (0 FLOPs)
    static bool should_skip_prefill_tile(int64_t row_start, int64_t row_end,
                                         int64_t col_start, int64_t col_end,
                                         int64_t window_size);

    // Optimized Decode Step: Computes attention over the sliding window [query_pos - W, query_pos]
    // using circular ring buffer KV access and constant O(W) work.
    static void decode_step(const float* q_step,
                            const SlidingWindowKVCache& kv_cache,
                            int64_t query_pos,
                            const SwaWindowConfig& cfg,
                            float* out_attn);
};

} // namespace strata::kernels
