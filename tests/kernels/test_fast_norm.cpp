// tests/kernels/test_fast_norm.cpp - Verification suite for Fast Normalization Suite
#include "strata/kernels/fast_norm.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cassert>

using namespace strata::kernels;

static int g_failed = 0;

#define CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            std::printf("  FAILED: %s (line %d)\n", msg, __LINE__); \
            g_failed++; \
        } \
    } while (0)

#define CHECK_NEAR(a, b, eps, msg) \
    do { \
        float diff = std::fabs((float)(a) - (float)(b)); \
        if (diff > (eps)) { \
            std::printf("  FAILED: %s (got %f, want %f, diff %e > %e, line %d)\n", \
                        msg, (float)(a), (float)(b), diff, (float)(eps), __LINE__); \
            g_failed++; \
        } \
    } while (0)

void test_fast_rmsnorm() {
    std::printf("[Test 1/4] Testing Fast RMSNorm...\n");
    constexpr size_t dim = 256;
    std::vector<float> x(dim), w(dim), out(dim), out_ref(dim);

    float sum_sq = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        x[i] = ((float)(i % 50) - 25.0f) * 0.1f;
        w[i] = 1.0f + ((float)(i % 20) * 0.01f);
        sum_sq += x[i] * x[i];
    }
    float scale = 1.0f / std::sqrt(sum_sq / (float)dim + 1e-6f);
    for (size_t i = 0; i < dim; ++i) {
        out_ref[i] = x[i] * scale * w[i];
    }

    FastNorm::rmsnorm(x.data(), w.data(), out.data(), dim, 1e-6f);

    for (size_t i = 0; i < dim; ++i) {
        CHECK_NEAR(out[i], out_ref[i], 1e-4f, "rmsnorm matches reference");
    }
    std::printf("  Passed. Fast RMSNorm validated.\n");
}

void test_fused_rmsnorm_residual() {
    std::printf("[Test 2/4] Testing Fused RMSNorm + Residual Addition...\n");
    constexpr size_t dim = 256;
    std::vector<float> residual(dim), x(dim), w(dim), out(dim), out_ref(dim), res_ref(dim);

    float sum_sq = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        residual[i] = ((float)(i % 30) - 15.0f) * 0.05f;
        x[i] = ((float)(i % 40) - 20.0f) * 0.05f;
        w[i] = 1.0f + ((float)(i % 10) * 0.02f);

        res_ref[i] = residual[i] + x[i];
        sum_sq += res_ref[i] * res_ref[i];
    }
    float scale = 1.0f / std::sqrt(sum_sq / (float)dim + 1e-6f);
    for (size_t i = 0; i < dim; ++i) {
        out_ref[i] = res_ref[i] * scale * w[i];
    }

    FastNorm::fused_rmsnorm_residual(residual.data(), x.data(), w.data(), out.data(), dim, 1e-6f);

    for (size_t i = 0; i < dim; ++i) {
        CHECK_NEAR(residual[i], res_ref[i], 1e-4f, "residual updated correctly in-place");
        CHECK_NEAR(out[i], out_ref[i], 1e-4f, "fused rmsnorm matches reference");
    }
    std::printf("  Passed. Fused RMSNorm + Residual validated.\n");
}

void test_gemma_rmsnorm() {
    std::printf("[Test 3/4] Testing Gemma-style (1 + gamma) RMSNorm...\n");
    constexpr size_t dim = 128;
    std::vector<float> x(dim), w(dim), out(dim);

    float sum_sq = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        x[i] = ((float)(i % 20) - 10.0f) * 0.1f;
        w[i] = 0.05f * (float)(i % 5);
        sum_sq += x[i] * x[i];
    }
    float scale = 1.0f / std::sqrt(sum_sq / (float)dim + 1e-6f);

    FastNorm::gemma_rmsnorm(x.data(), w.data(), out.data(), dim, 1e-6f);

    for (size_t i = 0; i < dim; ++i) {
        float expected = x[i] * scale * (1.0f + w[i]);
        CHECK_NEAR(out[i], expected, 1e-4f, "gemma rmsnorm matches");
    }
    std::printf("  Passed. Gemma RMSNorm validated.\n");
}

void test_head_rmsnorm() {
    std::printf("[Test 4/4] Testing Per-Head QK-Norm...\n");
    constexpr size_t n_heads = 4;
    constexpr size_t head_dim = 64;
    std::vector<float> qk(n_heads * head_dim);

    for (size_t i = 0; i < qk.size(); ++i) {
        qk[i] = ((float)(i % 35) - 17.0f) * 0.1f;
    }

    FastNorm::head_rmsnorm(qk.data(), n_heads, head_dim, nullptr, 1e-6f);

    // Each head should have unit RMS norm (~1.0)
    for (size_t h = 0; h < n_heads; ++h) {
        float sum_sq = 0.0f;
        for (size_t d = 0; d < head_dim; ++d) {
            float v = qk[h * head_dim + d];
            sum_sq += v * v;
        }
        float rms = std::sqrt(sum_sq / (float)head_dim);
        CHECK_NEAR(rms, 1.0f, 1e-3f, "head has unit RMS");
    }
    std::printf("  Passed. Per-Head QK-Norm validated.\n");
}

int main() {
    std::printf("=================================================================\n");
    std::printf("   RUNNING STRATA FAST NORMALIZATION TEST SUITE                  \n");
    std::printf("=================================================================\n");

    test_fast_rmsnorm();
    test_fused_rmsnorm_residual();
    test_gemma_rmsnorm();
    test_head_rmsnorm();

    std::printf("=================================================================\n");
    if (g_failed == 0) {
        std::printf("   ALL FAST NORMALIZATION TESTS PASSED (4/4)                     \n");
        std::printf("=================================================================\n");
        return 0;
    } else {
        std::printf("   %d FAST NORMALIZATION TESTS FAILED!                           \n", g_failed);
        std::printf("=================================================================\n");
        return 1;
    }
}
