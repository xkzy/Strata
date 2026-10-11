// include/strata/kernels/fast_speculative.hpp
#pragma once

#include "strata/kernels/math_policy.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace strata::kernels {

enum class SpeculativeMode {
    kGreedy = 0,
    kRejectionSampling = 1
};

struct SpeculativeResult {
    std::vector<int32_t> accepted_tokens; // Draft tokens accepted by the target model
    int32_t terminal_token = -1;          // Correction token (if mismatch) or bonus token (if all accepted)
    size_t drafts_proposed = 0;           // K
    size_t drafts_accepted = 0;           // Number of draft tokens verified
    float acceptance_rate = 0.0f;         // drafts_accepted / drafts_proposed
    float effective_tokens_per_step = 1.0f; // drafts_accepted + 1
};

class FastSpeculativeVerifier {
public:
    /// Fast SIMD-accelerated argmax across vocabulary logits
    static inline int32_t fast_argmax(const float* logits, size_t vocab_size) {
        if (!logits || vocab_size == 0) return 0;

        int32_t best_idx = 0;
        float best_val = logits[0];

        size_t v = 0;
#if defined(__AVX2__)
        __m256 max_val_v = _mm256_set1_ps(-1e30f);
        __m256i max_idx_v = _mm256_setzero_si256();
        __m256i step_idx_v = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
        const __m256i eight_v = _mm256_set1_epi32(8);

        for (; v + 7 < vocab_size; v += 8) {
            __m256 val = _mm256_loadu_ps(logits + v);
            __m256 mask = _mm256_cmp_ps(val, max_val_v, _CMP_GT_OQ);
            max_val_v = _mm256_max_ps(max_val_v, val);
            max_idx_v = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(max_idx_v),
                                                              _mm256_castsi256_ps(step_idx_v),
                                                              mask));
            step_idx_v = _mm256_add_epi32(step_idx_v, eight_v);
        }

        alignas(32) float vals[8];
        alignas(32) int32_t idxs[8];
        _mm256_store_ps(vals, max_val_v);
        _mm256_store_si256(reinterpret_cast<__m256i*>(idxs), max_idx_v);

        for (int i = 0; i < 8; ++i) {
            if (vals[i] > best_val) {
                best_val = vals[i];
                best_idx = idxs[i];
            }
        }
#endif
        for (; v < vocab_size; ++v) {
            if (logits[v] > best_val) {
                best_val = logits[v];
                best_idx = static_cast<int32_t>(v);
            }
        }

        return best_idx;
    }

    /// Verifies draft tokens using greedy argmax comparison.
    /// target_logits: [ (num_drafts + 1) * vocab_size ]
    /// - target_logits[k * vocab_size] is the distribution predicting token (k+1) given prompt + drafts[0..k-1]
    /// - target_logits[num_drafts * vocab_size] is the bonus distribution if all K drafts are accepted
    static SpeculativeResult verify_greedy(
        const int32_t* draft_tokens,
        size_t num_drafts,
        const float* target_logits,
        size_t vocab_size) {
        SpeculativeResult result;
        result.drafts_proposed = num_drafts;

        if (!draft_tokens || num_drafts == 0 || !target_logits || vocab_size == 0) {
            return result;
        }

        size_t accepted_count = 0;
        int32_t terminal_tok = -1;

        for (size_t k = 0; k < num_drafts; ++k) {
            const float* logits_k = target_logits + k * vocab_size;
            int32_t target_tok = fast_argmax(logits_k, vocab_size);

            if (target_tok == draft_tokens[k]) {
                result.accepted_tokens.push_back(target_tok);
                accepted_count++;
            } else {
                // Mismatch: emit target_tok as the correction token and halt verification
                terminal_tok = target_tok;
                break;
            }
        }

        if (accepted_count == num_drafts) {
            // All drafts accepted! Sample the bonus token from the (num_drafts)-th position
            const float* bonus_logits = target_logits + num_drafts * vocab_size;
            terminal_tok = fast_argmax(bonus_logits, vocab_size);
        }

        result.terminal_token = terminal_tok;
        result.drafts_accepted = accepted_count;
        result.acceptance_rate = (num_drafts > 0) ? (static_cast<float>(accepted_count) / static_cast<float>(num_drafts)) : 0.0f;
        result.effective_tokens_per_step = static_cast<float>(accepted_count + 1);

        return result;
    }

    /// Verifies draft tokens using stochastic rejection sampling with temperature.
    /// target_probs: [ (num_drafts + 1) * vocab_size ]
    /// draft_probs: [ num_drafts * vocab_size ]
    static SpeculativeResult verify_rejection_sampling(
        const int32_t* draft_tokens,
        size_t num_drafts,
        const float* target_probs,
        const float* draft_probs,
        size_t vocab_size,
        uint64_t seed = 42) {
        SpeculativeResult result;
        result.drafts_proposed = num_drafts;

        if (!draft_tokens || num_drafts == 0 || !target_probs || !draft_probs || vocab_size == 0) {
            return result;
        }

        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<float> dist_uni(0.0f, 1.0f);

        size_t accepted_count = 0;
        int32_t terminal_tok = -1;

        for (size_t k = 0; k < num_drafts; ++k) {
            int32_t d_tok = draft_tokens[k];
            if (d_tok < 0 || static_cast<size_t>(d_tok) >= vocab_size) break;

            float p_target = target_probs[k * vocab_size + d_tok];
            float p_draft = draft_probs[k * vocab_size + d_tok];

            float acceptance_prob = (p_draft > 1e-9f) ? std::min(1.0f, p_target / p_draft) : 1.0f;
            float r = dist_uni(rng);

            if (r <= acceptance_prob) {
                result.accepted_tokens.push_back(d_tok);
                accepted_count++;
            } else {
                // Rejection: Sample correction token from max(0, p_target - p_draft)
                std::vector<float> adjusted(vocab_size, 0.0f);
                float sum_adj = 0.0f;
                for (size_t v = 0; v < vocab_size; ++v) {
                    float diff = target_probs[k * vocab_size + v] - draft_probs[k * vocab_size + v];
                    if (diff > 0.0f) {
                        adjusted[v] = diff;
                        sum_adj += diff;
                    }
                }

                if (sum_adj > 1e-9f) {
                    float u = dist_uni(rng) * sum_adj;
                    float cum = 0.0f;
                    terminal_tok = 0;
                    for (size_t v = 0; v < vocab_size; ++v) {
                        cum += adjusted[v];
                        if (cum >= u) {
                            terminal_tok = static_cast<int32_t>(v);
                            break;
                        }
                    }
                } else {
                    terminal_tok = fast_argmax(target_probs + k * vocab_size, vocab_size);
                }
                break;
            }
        }

        if (accepted_count == num_drafts) {
            // Bonus token from target_probs[num_drafts]
            const float* bonus_p = target_probs + num_drafts * vocab_size;
            float u = dist_uni(rng);
            float cum = 0.0f;
            terminal_tok = static_cast<int32_t>(vocab_size - 1);
            for (size_t v = 0; v < vocab_size; ++v) {
                cum += bonus_p[v];
                if (cum >= u) {
                    terminal_tok = static_cast<int32_t>(v);
                    break;
                }
            }
        }

        result.terminal_token = terminal_tok;
        result.drafts_accepted = accepted_count;
        result.acceptance_rate = (num_drafts > 0) ? (static_cast<float>(accepted_count) / static_cast<float>(num_drafts)) : 0.0f;
        result.effective_tokens_per_step = static_cast<float>(accepted_count + 1);

        return result;
    }
};

} // namespace strata::kernels
