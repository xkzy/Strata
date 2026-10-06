// include/strata/kernels/fast_moe.hpp
#pragma once

#include "strata/kernels/math_policy.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace strata::kernels {

class FastMoE {
private:
#if defined(__AVX2__) || defined(__AVX__) || defined(_MSC_VER)
    static inline float hsum256_ps(__m256 v) {
        __m128 vlow = _mm256_castps256_ps128(v);
        __m128 vhigh = _mm256_extractf128_ps(v, 1);
        __m128 sum128 = _mm_add_ps(vlow, vhigh);
        __m128 sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        __m128 sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x55));
        return _mm_cvtss_f32(sum32);
    }
#endif

#if defined(__AVX512F__)
    static inline float hsum512_ps(__m512 v) {
        return _mm512_reduce_add_ps(v);
    }
#endif

    static inline void accumulate_weighted(float* __restrict__ out,
                                           const float* __restrict__ v,
                                           float weight,
                                           size_t dim) {
        if (!out || !v || weight == 0.0f) return;
        size_t d = 0;
#if defined(__AVX512F__)
        __m512 w_v = _mm512_set1_ps(weight);
        for (; d + 15 < dim; d += 16) {
            __m512 vo = _mm512_loadu_ps(out + d);
            __m512 vv = _mm512_loadu_ps(v + d);
            _mm512_storeu_ps(out + d, _mm512_fmadd_ps(w_v, vv, vo));
        }
#elif defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER))
        __m256 w_v = _mm256_set1_ps(weight);
        for (; d + 7 < dim; d += 8) {
            __m256 vo = _mm256_loadu_ps(out + d);
            __m256 vv = _mm256_loadu_ps(v + d);
            _mm256_storeu_ps(out + d, _mm256_fmadd_ps(w_v, vv, vo));
        }
#endif
        for (; d < dim; ++d) {
            out[d] += weight * v[d];
        }
    }

public:
    // ------------------------------------------------------------------------
    // Top-K Router: Partial-sort Top-K selection with numerical bounds sanitization
    // and stable softmax normalization over selected experts.
    // Complexity: O(E + K log K)
    // ------------------------------------------------------------------------
    static void route_topk(
        const float* __restrict__ logits,
        size_t num_experts,
        size_t top_k,
        int32_t* __restrict__ out_indices,
        float* __restrict__ out_weights,
        const MathContext& ctx = MathContext()) {
        if (!logits || !out_indices || !out_weights || num_experts == 0 || top_k == 0) return;
        top_k = std::min(top_k, num_experts);

        struct ExpertScore {
            int32_t index;
            float score;
        };

        std::vector<ExpertScore> scores(num_experts);
        for (size_t i = 0; i < num_experts; ++i) {
            scores[i] = {static_cast<int32_t>(i), sanitize_logit(logits[i], ctx)};
        }

        // Partial quick-select for top-k: O(N + K log K), stable tie-breaking
        std::partial_sort(
            scores.begin(), scores.begin() + top_k, scores.end(),
            [](const ExpertScore& a, const ExpertScore& b) {
                if (a.score != b.score) return a.score > b.score;
                return a.index < b.index;
            }
        );

        float max_top_score = scores[0].score;
        float sum_exp = 0.0f;
        for (size_t k = 0; k < top_k; ++k) {
            float e = std::exp(scores[k].score - max_top_score);
            out_weights[k] = e;
            sum_exp += e;
            out_indices[k] = scores[k].index;
        }

        const float inv_sum = sum_exp > 0.0f ? (1.0f / (sum_exp + 1e-9f)) : 0.0f;
        for (size_t k = 0; k < top_k; ++k) {
            out_weights[k] *= inv_sum;
        }
    }

    // ------------------------------------------------------------------------
    // Batched Top-K Router: routes multiple tokens across experts.
    // ------------------------------------------------------------------------
    static void route_topk_batch(
        const float* __restrict__ logits,
        size_t num_tokens,
        size_t num_experts,
        size_t top_k,
        int32_t* __restrict__ out_indices,
        float* __restrict__ out_weights,
        const MathContext& ctx = MathContext()) {
        if (!logits || !out_indices || !out_weights || num_tokens == 0 || num_experts == 0 || top_k == 0) return;

        for (size_t t = 0; t < num_tokens; ++t) {
            const float* tok_logits = logits + t * num_experts;
            int32_t* tok_indices = out_indices + t * top_k;
            float* tok_weights = out_weights + t * top_k;

            route_topk(tok_logits, num_experts, top_k, tok_indices, tok_weights, ctx);
        }
    }

    // ------------------------------------------------------------------------
    // Weighted Expert Aggregation (Array of Pointers):
    // Accumulates top_k expert output rows into out [dim].
    // Vectorized with AVX-512 / AVX2 FMA.
    // ------------------------------------------------------------------------
    static void weighted_aggregate(
        const float* const* __restrict__ expert_outputs,
        const float* __restrict__ weights,
        size_t top_k,
        size_t dim,
        float* __restrict__ out) {
        if (!expert_outputs || !weights || !out || top_k == 0 || dim == 0) return;
        std::fill(out, out + dim, 0.0f);

        for (size_t k = 0; k < top_k; ++k) {
            const float* exp_out = expert_outputs[k];
            if (!exp_out) continue;
            accumulate_weighted(out, exp_out, weights[k], dim);
        }
    }

    // ------------------------------------------------------------------------
    // Weighted Expert Aggregation (Contiguous Buffer):
    // expert_outputs is shape [top_k, dim].
    // ------------------------------------------------------------------------
    static void weighted_aggregate(
        const float* __restrict__ expert_outputs,
        const float* __restrict__ weights,
        size_t top_k,
        size_t dim,
        float* __restrict__ out) {
        if (!expert_outputs || !weights || !out || top_k == 0 || dim == 0) return;
        std::fill(out, out + dim, 0.0f);

        for (size_t k = 0; k < top_k; ++k) {
            const float* exp_out = expert_outputs + k * dim;
            accumulate_weighted(out, exp_out, weights[k], dim);
        }
    }

    // Alias for design-specification naming parity
    static inline void weighted_expert_aggregate(
        const float* const* __restrict__ expert_outputs,
        const float* __restrict__ weights,
        size_t top_k,
        size_t dim,
        float* __restrict__ out) {
        weighted_aggregate(expert_outputs, weights, top_k, dim, out);
    }

    static inline void weighted_expert_aggregate(
        const float* __restrict__ expert_outputs,
        const float* __restrict__ weights,
        size_t top_k,
        size_t dim,
        float* __restrict__ out) {
        weighted_aggregate(expert_outputs, weights, top_k, dim, out);
    }

    // ------------------------------------------------------------------------
    // Batched Weighted Expert Aggregation:
    // expert_outputs [num_tokens, top_k, dim], weights [num_tokens, top_k], out [num_tokens, dim]
    // ------------------------------------------------------------------------
    static void weighted_aggregate_batch(
        const float* __restrict__ expert_outputs,
        const float* __restrict__ weights,
        size_t num_tokens,
        size_t top_k,
        size_t dim,
        float* __restrict__ out) {
        if (!expert_outputs || !weights || !out || num_tokens == 0 || top_k == 0 || dim == 0) return;

        for (size_t t = 0; t < num_tokens; ++t) {
            const float* tok_exp = expert_outputs + t * top_k * dim;
            const float* tok_w = weights + t * top_k;
            float* tok_out = out + t * dim;

            weighted_aggregate(tok_exp, tok_w, top_k, dim, tok_out);
        }
    }

    // ------------------------------------------------------------------------
    // Token Dispatch Permutation Tables:
    // Computes per-expert token routing counts, prefix offsets, and token permutation
    // indices without redundant allocations.
    // ------------------------------------------------------------------------
    static void compute_dispatch_permutation(
        const int32_t* __restrict__ expert_indices, // [num_tokens * top_k]
        size_t num_tokens,
        size_t top_k,
        size_t num_experts,
        int32_t* __restrict__ out_expert_counts,     // [num_experts]
        int32_t* __restrict__ out_expert_offsets,    // [num_experts + 1]
        int32_t* __restrict__ out_token_indices,     // [num_tokens * top_k]
        int32_t* __restrict__ out_slot_indices = nullptr) {
        if (!expert_indices || !out_expert_counts || !out_expert_offsets ||
            !out_token_indices || num_tokens == 0 || top_k == 0 || num_experts == 0) {
            return;
        }

        std::fill(out_expert_counts, out_expert_counts + num_experts, 0);
        const size_t total_assignments = num_tokens * top_k;
        std::fill(out_token_indices, out_token_indices + total_assignments, -1);
        if (out_slot_indices) {
            std::fill(out_slot_indices, out_slot_indices + total_assignments, -1);
        }

        // 1. Histogram of tokens per expert
        for (size_t i = 0; i < total_assignments; ++i) {
            int32_t e = expert_indices[i];
            if (e >= 0 && static_cast<size_t>(e) < num_experts) {
                out_expert_counts[e]++;
            }
        }

        // 2. Prefix sum for expert offsets
        out_expert_offsets[0] = 0;
        for (size_t e = 0; e < num_experts; ++e) {
            out_expert_offsets[e + 1] = out_expert_offsets[e] + out_expert_counts[e];
        }

        // 3. Populate dispatch permutation indices
        std::vector<int32_t> running_offset(out_expert_offsets, out_expert_offsets + num_experts);
        for (size_t t = 0; t < num_tokens; ++t) {
            for (size_t k = 0; k < top_k; ++k) {
                int32_t e = expert_indices[t * top_k + k];
                if (e >= 0 && static_cast<size_t>(e) < num_experts) {
                    int32_t pos = running_offset[e]++;
                    out_token_indices[pos] = static_cast<int32_t>(t);
                    if (out_slot_indices) {
                        out_slot_indices[pos] = static_cast<int32_t>(k);
                    }
                }
            }
        }
    }

    // Alias for design-specification naming parity
    static inline void dispatch_permutation(
        const int32_t* __restrict__ expert_indices,
        size_t num_tokens,
        size_t top_k,
        size_t num_experts,
        int32_t* __restrict__ out_expert_counts,
        int32_t* __restrict__ out_expert_offsets,
        int32_t* __restrict__ out_token_indices,
        int32_t* __restrict__ out_slot_indices = nullptr) {
        compute_dispatch_permutation(
            expert_indices, num_tokens, top_k, num_experts,
            out_expert_counts, out_expert_offsets,
            out_token_indices, out_slot_indices
        );
    }

    // ------------------------------------------------------------------------
    // Token Permute:
    // Gathers token embeddings according to token_indices layout:
    // tokens [num_tokens, dim] -> out_permuted [total_dispatched, dim]
    // ------------------------------------------------------------------------
    static void permute_tokens(
        const float* __restrict__ tokens,
        const int32_t* __restrict__ token_indices,
        size_t total_dispatched,
        size_t dim,
        float* __restrict__ out_permuted) {
        if (!tokens || !token_indices || !out_permuted || total_dispatched == 0 || dim == 0) return;

        for (size_t i = 0; i < total_dispatched; ++i) {
            int32_t t = token_indices[i];
            const float* src = tokens + static_cast<size_t>(t) * dim;
            float* dst = out_permuted + i * dim;
            std::memcpy(dst, src, dim * sizeof(float));
        }
    }

    static inline void dispatch_permute(
        const float* __restrict__ tokens,
        const int32_t* __restrict__ token_indices,
        size_t total_dispatched,
        size_t dim,
        float* __restrict__ out_permuted) {
        permute_tokens(tokens, token_indices, total_dispatched, dim, out_permuted);
    }

    // ------------------------------------------------------------------------
    // Combine Dispatched:
    // Aggregates dispatched expert outputs [total_dispatched, dim] back to out_tokens [num_tokens, dim],
    // weighted by expert router probabilities.
    // ------------------------------------------------------------------------
    static void combine_dispatched(
        const float* __restrict__ dispatched_outputs,
        const int32_t* __restrict__ token_indices,
        const int32_t* __restrict__ slot_indices,
        const float* __restrict__ weights,
        size_t total_dispatched,
        size_t num_tokens,
        size_t top_k,
        size_t dim,
        float* __restrict__ out_tokens) {
        if (!dispatched_outputs || !token_indices || !weights || !out_tokens ||
            total_dispatched == 0 || num_tokens == 0 || dim == 0) {
            return;
        }

        std::fill(out_tokens, out_tokens + num_tokens * dim, 0.0f);

        for (size_t i = 0; i < total_dispatched; ++i) {
            int32_t t = token_indices[i];
            if (t < 0 || static_cast<size_t>(t) >= num_tokens) continue;

            float w = 1.0f;
            if (slot_indices) {
                int32_t k = slot_indices[i];
                if (k >= 0 && static_cast<size_t>(k) < top_k) {
                    w = weights[static_cast<size_t>(t) * top_k + static_cast<size_t>(k)];
                }
            } else {
                w = weights[i];
            }

            const float* exp_out = dispatched_outputs + i * dim;
            float* tok_out = out_tokens + static_cast<size_t>(t) * dim;
            accumulate_weighted(tok_out, exp_out, w, dim);
        }
    }

    static inline void unpermute_tokens(
        const float* __restrict__ dispatched_outputs,
        const int32_t* __restrict__ token_indices,
        const int32_t* __restrict__ slot_indices,
        const float* __restrict__ weights,
        size_t total_dispatched,
        size_t num_tokens,
        size_t top_k,
        size_t dim,
        float* __restrict__ out_tokens) {
        combine_dispatched(dispatched_outputs, token_indices, slot_indices, weights,
                           total_dispatched, num_tokens, top_k, dim, out_tokens);
    }
};

} // namespace strata::kernels
