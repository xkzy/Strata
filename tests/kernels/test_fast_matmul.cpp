// tests/kernels/test_fast_matmul.cpp - Verification suite for Fast MatMul (GEMM / GEMV)
#include "strata/kernels/fast_matmul.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cassert>
#include <chrono>

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

void test_fast_gemv() {
    std::printf("[Test 1/6] Testing Fast GEMV (Matrix-Vector Multiply)...\n");
    constexpr int64_t M = 128;
    constexpr int64_t K = 256;
    std::vector<float> A(M * K), x(K), y(M), y_ref(M);

    for (int64_t i = 0; i < M * K; ++i) A[i] = ((float)(i % 100) - 50.0f) * 0.01f;
    for (int64_t k = 0; k < K; ++k) x[k] = ((float)(k % 50) - 25.0f) * 0.02f;

    // Reference
    for (int64_t i = 0; i < M; ++i) {
        float sum = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
            sum += A[i * K + k] * x[k];
        }
        y_ref[i] = sum;
    }

    FastMatMul::gemv_fp32(A.data(), x.data(), y.data(), M, K);

    for (int64_t i = 0; i < M; ++i) {
        CHECK_NEAR(y[i], y_ref[i], 1e-4f, "gemv matches reference");
    }
    std::printf("  Passed. Fast GEMV validated.\n");
}

void test_batch_gemv_multi_token() {
    std::printf("[Test 2/6] Testing Multi-Token Batch GEMV (Speculative/Decoding Window)...\n");
    constexpr int64_t T = 4;
    constexpr int64_t N = 640;
    constexpr int64_t K = 2560;

    std::vector<float> X(T * K), W(N * K), Y(T * N), Y_ref(T * N);
    for (int64_t i = 0; i < T * K; ++i) X[i] = ((float)(i % 70) - 35.0f) * 0.01f;
    for (int64_t i = 0; i < N * K; ++i) W[i] = ((float)(i % 80) - 40.0f) * 0.01f;

    for (int64_t t = 0; t < T; ++t) {
        for (int64_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += X[t * K + k] * W[n * K + k];
            }
            Y_ref[t * N + n] = sum;
        }
    }

    FastMatMul::batch_gemv_fp32(X.data(), K, W.data(), K, Y.data(), N, T, N, K);

    for (int64_t i = 0; i < T * N; ++i) {
        CHECK_NEAR(Y[i], Y_ref[i], 1e-3f, "batch gemv matches reference");
    }
    std::printf("  Passed. Multi-token batch GEMV validated.\n");
}

void test_tiled_gemm() {
    std::printf("[Test 3/6] Testing Tiled Cache-Blocked GEMM...\n");
    constexpr int64_t M = 128;
    constexpr int64_t N = 128;
    constexpr int64_t K = 128;

    std::vector<float> A(M * K), B(K * N), C(M * N), C_ref(M * N);
    for (int64_t i = 0; i < M * K; ++i) A[i] = ((float)(i % 50) - 25.0f) * 0.02f;
    for (int64_t i = 0; i < K * N; ++i) B[i] = ((float)(i % 60) - 30.0f) * 0.02f;

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C_ref[i * N + j] = sum;
        }
    }

    FastMatMul::gemm_fp32(A.data(), B.data(), C.data(), M, N, K);

    for (int64_t i = 0; i < M * N; ++i) {
        CHECK_NEAR(C[i], C_ref[i], 1e-3f, "gemm matches reference");
    }
    std::printf("  Passed. Tiled GEMM validated.\n");
}

void test_fused_matmul_activation() {
    std::printf("[Test 4/6] Testing Fused MatMul + Activation (SiLU / GELU / ReLU)...\n");
    constexpr int64_t M = 64;
    constexpr int64_t N = 64;
    constexpr int64_t K = 64;

    std::vector<float> A(M * K), B(K * N), C_silu(M * N), C_gelu(M * N);
    for (int64_t i = 0; i < M * K; ++i) A[i] = ((float)(i % 40) - 20.0f) * 0.05f;
    for (int64_t i = 0; i < K * N; ++i) B[i] = ((float)(i % 50) - 25.0f) * 0.05f;

    FastMatMul::gemm_fp32(A.data(), B.data(), C_silu.data(), M, N, K, 1.0f, 0.0f, ActivationType::kSiLU);
    FastMatMul::gemm_fp32(A.data(), B.data(), C_gelu.data(), M, N, K, 1.0f, 0.0f, ActivationType::kGELU);

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float raw_dot = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                raw_dot += A[i * K + k] * B[k * N + j];
            }
            float expected_silu = fast_math::fast_silu(raw_dot);
            float expected_gelu = fast_math::fast_gelu(raw_dot);

            CHECK_NEAR(C_silu[i * N + j], expected_silu, 1e-3f, "fused gemm silu matches");
            CHECK_NEAR(C_gelu[i * N + j], expected_gelu, 1e-3f, "fused gemm gelu matches");
        }
    }
    std::printf("  Passed. Fused MatMul + Activation validated.\n");
}

void test_fused_swiglu_gemm() {
    std::printf("[Test 5/6] Testing Fused SwiGLU GEMM (Dual Projection in One Pass)...\n");
    constexpr int64_t M = 8;
    constexpr int64_t N = 640;
    constexpr int64_t K = 2560;

    std::vector<float> X(M * K), W_gate(N * K), W_up(N * K), Out(M * N), Out_ref(M * N);
    for (int64_t i = 0; i < M * K; ++i) X[i] = ((float)(i % 55) - 27.0f) * 0.02f;
    for (int64_t i = 0; i < N * K; ++i) {
        W_gate[i] = ((float)(i % 65) - 32.0f) * 0.01f;
        W_up[i] = ((float)(i % 75) - 37.0f) * 0.01f;
    }

    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            float g_sum = 0.0f, u_sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                g_sum += X[m * K + k] * W_gate[n * K + k];
                u_sum += X[m * K + k] * W_up[n * K + k];
            }
            Out_ref[m * N + n] = fast_math::fast_swiglu(g_sum, u_sum);
        }
    }

    FastMatMul::fused_swiglu_gemm(X.data(), W_gate.data(), W_up.data(), Out.data(), M, N, K);

    for (int64_t i = 0; i < M * N; ++i) {
        CHECK_NEAR(Out[i], Out_ref[i], 1e-3f, "fused swiglu gemm matches reference");
    }
    std::printf("  Passed. Fused SwiGLU GEMM validated.\n");
}

void test_matmul_benchmark_gflops() {
    std::printf("[Test 6/6] Benchmark Throughput & GFLOPS...\n");
    constexpr int64_t M = 256;
    constexpr int64_t N = 256;
    constexpr int64_t K = 256;

    std::vector<float> A(M * K, 0.01f), B(K * N, 0.02f), C(M * N, 0.0f);

    auto t0 = std::chrono::high_resolution_clock::now();
    constexpr int ITERS = 10;
    for (int iter = 0; iter < ITERS; ++iter) {
        FastMatMul::gemm_fp32(A.data(), B.data(), C.data(), M, N, K);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double avg_ms = total_ms / ITERS;
    double ops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = (ops / (avg_ms * 1e-3)) / 1e9;

    std::printf("  Matrix Shape: [%ld x %ld] * [%ld x %ld]\n", (long)M, (long)K, (long)K, (long)N);
    std::printf("  Avg Compute Time: %.3f ms per GEMM (%.2f GFLOPS)\n", avg_ms, gflops);
    std::printf("  Passed. Throughput benchmark validated.\n");
}

int main() {
    std::printf("=================================================================\n");
    std::printf("   RUNNING STRATA FAST MATMUL (GEMM / GEMV) TEST SUITE           \n");
    std::printf("=================================================================\n");

    test_fast_gemv();
    test_batch_gemv_multi_token();
    test_tiled_gemm();
    test_fused_matmul_activation();
    test_fused_swiglu_gemm();
    test_matmul_benchmark_gflops();

    std::printf("=================================================================\n");
    if (g_failed == 0) {
        std::printf("   ALL FAST MATMUL TESTS PASSED (6/6)                            \n");
        std::printf("=================================================================\n");
        return 0;
    } else {
        std::printf("   %d FAST MATMUL TESTS FAILED!                                  \n", g_failed);
        std::printf("=================================================================\n");
        return 1;
    }
}
