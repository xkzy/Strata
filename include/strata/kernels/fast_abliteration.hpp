// include/strata/kernels/fast_abliteration.hpp - High-Performance Online Abliteration Kernel Engine
//
// Implements inference-time representation intervention (Arditi et al. / Maxime Labonne "Uncensor any LLM with abliteration"):
// Refusal in aligned LLMs is mediated by a low-dimensional direction in the residual stream.
// By projecting activations onto the refusal direction and subtracting this projection in real time,
// the model's refusal behavior is inhibited online without requiring offline weight modifications or retraining.
//
// Mathematical formulation:
//   proj = <x, r_hat> = sum_{i=0}^{D-1} x[i] * r_hat[i]
//   x <- x - alpha * proj * r_hat  (orthogonal ablation)
//   or
//   x <- x - alpha * max(0, proj) * r_hat  (positive refusal axis suppression)
//
// Supported architectures:
//   - Qwen3.8-Flash-Next / Qwen2.5 (D = 2560, hc = 4)
//   - Llama 3 / 3.1 / 3.2 (D = 2048 / 4096 / 8192)
//   - Mistral / Mixtral (D = 4096)
//   - DeepSeek V2 / V3 (D = 5120)
#pragma once

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "strata/kernels/fast_parallel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace strata::kernels {

enum class AbliterationMode : uint8_t {
    kOrthogonal = 0,   // Full projection removal: x <- x - alpha * <x, r> * r
    kPositiveOnly = 1  // One-sided refusal inhibition: x <- x - alpha * max(0, <x, r>) * r
};

class FastAbliteration {
public:
    // ------------------------------------------------------------------------
    // Normalizes a vector in-place to unit Euclidean length
    // ------------------------------------------------------------------------
    static inline float normalize_vector(float* __restrict__ vec, int64_t D) {
        if (!vec || D <= 0) return 0.0f;
        double sum_sq = 0.0;
        int64_t i = 0;

#if defined(__AVX2__)
        __m256 acc = _mm256_setzero_ps();
        for (; i <= D - 8; i += 8) {
            __m256 v = _mm256_loadu_ps(vec + i);
            acc = _mm256_fmadd_ps(v, v, acc);
        }
        alignas(32) float buf[8];
        _mm256_storeu_ps(buf, acc);
        for (int j = 0; j < 8; ++j) sum_sq += buf[j];
#endif
        for (; i < D; ++i) {
            sum_sq += static_cast<double>(vec[i]) * vec[i];
        }

        float norm = static_cast<float>(std::sqrt(sum_sq));
        if (norm > 1e-12f) {
            float inv_norm = 1.0f / norm;
            i = 0;
#if defined(__AVX2__)
            __m256 v_inv = _mm256_set1_ps(inv_norm);
            for (; i <= D - 8; i += 8) {
                __m256 v = _mm256_loadu_ps(vec + i);
                _mm256_storeu_ps(vec + i, _mm256_mul_ps(v, v_inv));
            }
#endif
            for (; i < D; ++i) {
                vec[i] *= inv_norm;
            }
        }
        return norm;
    }

    // ------------------------------------------------------------------------
    // Computes dot product: dot = <x, r>
    // ------------------------------------------------------------------------
    static inline float dot_product(const float* __restrict__ a, const float* __restrict__ b, int64_t D) {
        float sum = 0.0f;
        int64_t i = 0;

#if defined(__AVX512F__)
        __m512 acc = _mm512_setzero_ps();
        for (; i <= D - 16; i += 16) {
            __m512 va = _mm512_loadu_ps(a + i);
            __m512 vb = _mm512_loadu_ps(b + i);
            acc = _mm512_fmadd_ps(va, vb, acc);
        }
        sum = _mm512_reduce_add_ps(acc);
#elif defined(__AVX2__)
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        for (; i <= D - 16; i += 16) {
            __m256 va0 = _mm256_loadu_ps(a + i);
            __m256 vb0 = _mm256_loadu_ps(b + i);
            __m256 va1 = _mm256_loadu_ps(a + i + 8);
            __m256 vb1 = _mm256_loadu_ps(b + i + 8);
            acc0 = _mm256_fmadd_ps(va0, vb0, acc0);
            acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
        }
        __m256 acc = _mm256_add_ps(acc0, acc1);
        alignas(32) float buf[8];
        _mm256_storeu_ps(buf, acc);
        for (int j = 0; j < 8; ++j) sum += buf[j];
#endif
        for (; i < D; ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    // ------------------------------------------------------------------------
    // In-place abliteration intervention for a single token hidden state vector:
    // x <- x - alpha * proj * r_hat
    // ------------------------------------------------------------------------
    static inline void project_and_subtract_token(
        float* __restrict__ x,
        const float* __restrict__ r_dir,
        int64_t D,
        float strength = 1.0f,
        AbliterationMode mode = AbliterationMode::kOrthogonal) {
        if (!x || !r_dir || D <= 0 || strength <= 0.0f) return;

        float proj = dot_product(x, r_dir, D);

        if (mode == AbliterationMode::kPositiveOnly && proj <= 0.0f) {
            return; // no positive refusal component to inhibit
        }

        const float scale = strength * proj;
        int64_t i = 0;

#if defined(__AVX512F__)
        __m512 v_scale = _mm512_set1_ps(scale);
        for (; i <= D - 16; i += 16) {
            __m512 vx = _mm512_loadu_ps(x + i);
            __m512 vr = _mm512_loadu_ps(r_dir + i);
            vx = _mm512_fnmadd_ps(v_scale, vr, vx);
            _mm512_storeu_ps(x + i, vx);
        }
#elif defined(__AVX2__)
        __m256 v_scale = _mm256_set1_ps(scale);
        for (; i <= D - 8; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vr = _mm256_loadu_ps(r_dir + i);
            vx = _mm256_fnmadd_ps(v_scale, vr, vx);
            _mm256_storeu_ps(x + i, vx);
        }
#endif
        for (; i < D; ++i) {
            x[i] -= scale * r_dir[i];
        }
    }

    // ------------------------------------------------------------------------
    // Multi-token & Multi-channel Residual Tensor Abliteration:
    // Applies online abliteration across all T tokens and hc channels of residual stack R:
    // R: [T, hc, D] (row-major)
    // ------------------------------------------------------------------------
    static void abliterate_layer_residual(
        float* __restrict__ R,
        int64_t T,
        int64_t hc,
        int64_t D,
        const float* __restrict__ r_dir,
        float strength = 1.0f,
        AbliterationMode mode = AbliterationMode::kOrthogonal) {
        if (!R || !r_dir || T <= 0 || hc <= 0 || D <= 0 || strength <= 0.0f) return;

        const int64_t total_vectors = T * hc;

        #pragma omp parallel for schedule(static) if (total_vectors >= 4)
        for (int64_t idx = 0; idx < total_vectors; ++idx) {
            float* vec = R + idx * D;
            project_and_subtract_token(vec, r_dir, D, strength, mode);
        }
    }

    // ------------------------------------------------------------------------
    // Contrastive Refusal Direction Estimation:
    // Given harmful and harmless activation batches, computes:
    //   r = mean(harmful) - mean(harmless)
    //   r_hat = r / ||r||
    // ------------------------------------------------------------------------
    static bool compute_contrastive_refusal_vector(
        const std::vector<std::vector<float>>& harmful_acts,
        const std::vector<std::vector<float>>& harmless_acts,
        int64_t D,
        std::vector<float>& out_dir) {
        if (harmful_acts.empty() || harmless_acts.empty() || D <= 0) return false;

        out_dir.assign(D, 0.0f);
        std::vector<double> mean_harmful(D, 0.0);
        std::vector<double> mean_harmless(D, 0.0);

        for (const auto& act : harmful_acts) {
            if (static_cast<int64_t>(act.size()) < D) continue;
            for (int64_t i = 0; i < D; ++i) mean_harmful[i] += act[i];
        }
        for (const auto& act : harmless_acts) {
            if (static_cast<int64_t>(act.size()) < D) continue;
            for (int64_t i = 0; i < D; ++i) mean_harmless[i] += act[i];
        }

        const double inv_n_harm = 1.0 / static_cast<double>(harmful_acts.size());
        const double inv_n_harmless = 1.0 / static_cast<double>(harmless_acts.size());

        for (int64_t i = 0; i < D; ++i) {
            double h = mean_harmful[i] * inv_n_harm;
            double u = mean_harmless[i] * inv_n_harmless;
            out_dir[i] = static_cast<float>(h - u);
        }

        normalize_vector(out_dir.data(), D);
        return true;
    }

    // (A former `generate_canonical_refusal_direction` - a synthetic sine/cosine basis presented as a refusal
    // direction - was removed with docs/ABLITERATION_PLAN.md phase 0: no direction may be fabricated.  Every
    // direction applied to the model must come from measured activations (`compute_contrastive_refusal_vector`)
    // or from a file the user chose.)

    // ------------------------------------------------------------------------
    // Binary Serialization / Deserialization of Refusal Vectors:
    // Format: Magic 0x41424C52 ("ABLR"), D (int64), L (int64), float data[L * D]
    // ------------------------------------------------------------------------
    static bool save_vector_file(
        const std::string& path,
        const std::vector<float>& dir_data,
        int64_t D,
        int64_t L) {
        std::ofstream ofs(path, std::ios::binary);
        if (!ofs.is_open()) return false;

        const uint32_t magic = 0x41424C52; // 'ABLR'
        ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        ofs.write(reinterpret_cast<const char*>(&D), sizeof(D));
        ofs.write(reinterpret_cast<const char*>(&L), sizeof(L));

        const size_t total_bytes = dir_data.size() * sizeof(float);
        ofs.write(reinterpret_cast<const char*>(dir_data.data()), total_bytes);
        return ofs.good();
    }

    static bool load_vector_file(
        const std::string& path,
        std::vector<float>& out_dir,
        int64_t& out_D,
        int64_t& out_L) {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs.is_open()) return false;

        uint32_t magic = 0;
        ifs.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (magic != 0x41424C52) return false;

        ifs.read(reinterpret_cast<char*>(&out_D), sizeof(out_D));
        ifs.read(reinterpret_cast<char*>(&out_L), sizeof(out_L));
        if (out_D <= 0 || out_L <= 0 || out_D > 65536 || out_L > 256) return false;

        size_t total_elems = static_cast<size_t>(out_D * out_L);
        out_dir.resize(total_elems);
        ifs.read(reinterpret_cast<char*>(out_dir.data()), total_elems * sizeof(float));
        return ifs.good();
    }
};

} // namespace strata::kernels
