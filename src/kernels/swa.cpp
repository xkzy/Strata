// src/kernels/swa.cpp - Sliding Window Attention (SWA) Optimization Engine Implementation
#include "strata/kernels/swa.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

namespace strata::kernels {

// ------------------- Sliding Window KV Cache -------------------

SlidingWindowKVCache::SlidingWindowKVCache(const SwaWindowConfig& cfg)
    : cfg_(cfg), total_tokens_(0) {
    slot_stride_ = cfg_.kv_stride_per_token();
    size_t ring_floats = static_cast<size_t>(cfg_.window_size * slot_stride_);
    k_ring_.resize(ring_floats, 0.0f);
    v_ring_.resize(ring_floats, 0.0f);
}

SlidingWindowKVCache::~SlidingWindowKVCache() = default;

void SlidingWindowKVCache::append(int64_t token_pos, const float* k_token, const float* v_token) {
    if (!k_token || !v_token || token_pos < 0) return;

    int64_t slot = get_ring_slot(token_pos);
    size_t offset = static_cast<size_t>(slot * slot_stride_);

    std::memcpy(&k_ring_[offset], k_token, slot_stride_ * sizeof(float));
    std::memcpy(&v_ring_[offset], v_token, slot_stride_ * sizeof(float));

    if (token_pos + 1 > total_tokens_) {
        total_tokens_ = token_pos + 1;
    }
}

const float* SlidingWindowKVCache::get_key(int64_t token_pos) const {
    if (token_pos < 0 || token_pos >= total_tokens_) return nullptr;
    if (token_pos < total_tokens_ - cfg_.window_size) return nullptr; // Out of sliding window

    int64_t slot = get_ring_slot(token_pos);
    return &k_ring_[slot * slot_stride_];
}

const float* SlidingWindowKVCache::get_value(int64_t token_pos) const {
    if (token_pos < 0 || token_pos >= total_tokens_) return nullptr;
    if (token_pos < total_tokens_ - cfg_.window_size) return nullptr; // Out of sliding window

    int64_t slot = get_ring_slot(token_pos);
    return &v_ring_[slot * slot_stride_];
}

int64_t SlidingWindowKVCache::active_window_tokens() const {
    return std::min<int64_t>(total_tokens_, cfg_.window_size);
}

size_t SlidingWindowKVCache::memory_bytes() const {
    return (k_ring_.size() + v_ring_.size()) * sizeof(float);
}

double SlidingWindowKVCache::memory_reduction_ratio(int64_t total_tokens) const {
    if (total_tokens <= cfg_.window_size) return 1.0;
    return static_cast<double>(total_tokens) / static_cast<double>(cfg_.window_size);
}

void SlidingWindowKVCache::reset() {
    total_tokens_ = 0;
    std::fill(k_ring_.begin(), k_ring_.end(), 0.0f);
    std::fill(v_ring_.begin(), v_ring_.end(), 0.0f);
}

// ------------------- SWA Execution Engine -------------------

SwaCausalRange SWAExecutionEngine::compute_causal_range(int64_t query_pos, int64_t window_size) {
    SwaCausalRange range;
    range.start_pos = std::max<int64_t>(0, query_pos - window_size + 1);
    range.end_pos = query_pos;
    range.active_length = range.end_pos - range.start_pos + 1;
    return range;
}

bool SWAExecutionEngine::should_skip_prefill_tile(int64_t row_start, int64_t row_end,
                                                  int64_t col_start, int64_t col_end,
                                                  int64_t window_size) {
    // 1. Causal upper triangle: All keys in tile are in the future
    if (col_start > row_end) {
        return true;
    }

    // 2. Sliding window lower triangle: All keys in tile are older than (row - window_size + 1)
    if (col_end < row_start - window_size + 1) {
        return true;
    }

    return false;
}

void SWAExecutionEngine::decode_step(const float* q_step,
                                     const SlidingWindowKVCache& kv_cache,
                                     int64_t query_pos,
                                     const SwaWindowConfig& cfg,
                                     float* out_attn) {
    if (!q_step || !out_attn || query_pos < 0) return;

    SwaCausalRange range = compute_causal_range(query_pos, cfg.window_size);
    int64_t n_keys = range.active_length;
    if (n_keys <= 0) return;

    int64_t gqa_group_size = cfg.n_head / cfg.n_head_kv;
    int64_t d = cfg.head_dim;
    float scale = cfg.scale > 0.0f ? cfg.scale : (1.0f / std::sqrt(static_cast<float>(d)));

    // Reusable thread-local buffer for attention logits and softmax weights
    thread_local std::vector<float> tl_logits;
    if (static_cast<int64_t>(tl_logits.size()) < n_keys) {
        tl_logits.resize(n_keys);
    }

    for (int64_t h = 0; h < cfg.n_head; ++h) {
        int64_t kv_head = h / gqa_group_size;
        const float* q_h = q_step + (h * d);

        // 1. Compute dot-product scores over rolling window keys
        float max_logit = -1e30f;
        for (int64_t i = 0; i < n_keys; ++i) {
            int64_t key_pos = range.start_pos + i;
            const float* k_vec = kv_cache.get_key(key_pos);
            if (!k_vec) {
                tl_logits[i] = -1e30f;
                continue;
            }

            const float* k_h = k_vec + (kv_head * d);
            float dot = 0.0f;
            for (int64_t j = 0; j < d; ++j) {
                dot += q_h[j] * k_h[j];
            }
            float score = dot * scale;
            tl_logits[i] = score;
            if (score > max_logit) max_logit = score;
        }

        // 2. Numerically stable Softmax
        float sum_exp = 0.0f;
        for (int64_t i = 0; i < n_keys; ++i) {
            float exp_val = std::exp(tl_logits[i] - max_logit);
            tl_logits[i] = exp_val;
            sum_exp += exp_val;
        }

        float inv_sum = sum_exp > 0.0f ? (1.0f / sum_exp) : 0.0f;
        for (int64_t i = 0; i < n_keys; ++i) {
            tl_logits[i] *= inv_sum;
        }

        // 3. Accumulate attention weighted values
        float* out_h = out_attn + (h * d);
        std::fill(out_h, out_h + d, 0.0f);

        for (int64_t i = 0; i < n_keys; ++i) {
            float weight = tl_logits[i];
            if (weight == 0.0f) continue;

            int64_t val_pos = range.start_pos + i;
            const float* v_vec = kv_cache.get_value(val_pos);
            if (!v_vec) continue;

            const float* v_h = v_vec + (kv_head * d);
            for (int64_t j = 0; j < d; ++j) {
                out_h[j] += weight * v_h[j];
            }
        }
    }
}

} // namespace strata::kernels
