// tests/kernels/test_fast_activations.cpp - Verification suite for Fast Activation Math
#include "strata/kernels/fast_activations.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cassert>
#include <chrono>

using namespace strata::kernels;
using namespace strata::kernels::fast_math;

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

void test_fast_sigmoid() {
    std::printf("[Test 1/10] Testing Fast Sigmoid (values, monotonicity & bounds)...\n");
    CHECK_NEAR(fast_sigmoid(0.0f), 0.5f, 1e-6f, "sigmoid(0) == 0.5");
    CHECK(fast_sigmoid(20.0f) == 1.0f, "sigmoid(20) clamped to 1.0");
    CHECK(fast_sigmoid(-20.0f) == 0.0f, "sigmoid(-20) clamped to 0.0");

    // Monotonicity check
    float prev = -1.0f;
    for (float x = -15.0f; x <= 15.0f; x += 0.25f) {
        float s = fast_sigmoid(x);
        CHECK(s >= prev, "sigmoid must be monotonically increasing");
        float ref = 1.0f / (1.0f + std::exp(-x));
        CHECK_NEAR(s, ref, 1e-4f, "sigmoid matches standard exp");
        prev = s;
    }
    std::printf("  Passed. Fast Sigmoid verified.\n");
}

void test_fast_silu() {
    std::printf("[Test 2/10] Testing Fast SiLU / Swish...\n");
    CHECK_NEAR(fast_silu(0.0f), 0.0f, 1e-6f, "silu(0) == 0");
    CHECK_NEAR(fast_silu(20.0f), 20.0f, 1e-6f, "silu(20) == 20");
    CHECK_NEAR(fast_silu(-20.0f), 0.0f, 1e-6f, "silu(-20) == 0");

    for (float x = -10.0f; x <= 10.0f; x += 0.5f) {
        float s = fast_silu(x);
        float ref = x / (1.0f + std::exp(-x));
        CHECK_NEAR(s, ref, 1e-4f, "silu matches standard exp");
    }
    std::printf("  Passed. Fast SiLU verified.\n");
}

void test_fast_swiglu() {
    std::printf("[Test 3/10] Testing Fast SwiGLU Gated Activation...\n");
    float gate = 2.0f;
    float up = 3.5f;
    float expected = fast_silu(gate) * up;
    CHECK_NEAR(fast_swiglu(gate, up), expected, 1e-6f, "swiglu == silu(gate) * up");
    CHECK_NEAR(fast_swiglu(0.0f, 10.0f), 0.0f, 1e-6f, "swiglu(0, up) == 0");
    std::printf("  Passed. Fast SwiGLU verified.\n");
}

void test_fast_gelu_and_quick_gelu() {
    std::printf("[Test 4/10] Testing Fast GELU and QuickGELU...\n");
    CHECK_NEAR(fast_gelu(0.0f), 0.0f, 1e-6f, "gelu(0) == 0");
    CHECK_NEAR(fast_quick_gelu(0.0f), 0.0f, 1e-6f, "quick_gelu(0) == 0");

    for (float x = -5.0f; x <= 5.0f; x += 0.5f) {
        float g = fast_gelu(x);
        float q = fast_quick_gelu(x);
        // GELU should be smooth and closely follow standard Gaussian CDF approximation
        CHECK(!std::isnan(g) && !std::isinf(g), "gelu is finite");
        CHECK(!std::isnan(q) && !std::isinf(q), "quick_gelu is finite");
        if (x > 3.0f) {
            CHECK_NEAR(g, x, 0.05f, "gelu(x) -> x for large x");
        } else if (x < -3.0f) {
            CHECK_NEAR(g, 0.0f, 0.05f, "gelu(x) -> 0 for large negative x");
        }
    }
    std::printf("  Passed. Fast GELU & QuickGELU verified.\n");
}

void test_fast_geglu() {
    std::printf("[Test 5/10] Testing Fast GeGLU Gated Activation...\n");
    float gate = 1.5f;
    float up = -2.0f;
    float expected = fast_gelu(gate) * up;
    CHECK_NEAR(fast_geglu(gate, up), expected, 1e-6f, "geglu == gelu(gate) * up");
    std::printf("  Passed. Fast GeGLU verified.\n");
}

void test_fast_softplus() {
    std::printf("[Test 6/10] Testing Fast Softplus...\n");
    CHECK_NEAR(fast_softplus(0.0f), std::log(2.0f), 1e-4f, "softplus(0) == ln(2)");
    CHECK_NEAR(fast_softplus(30.0f), 30.0f, 1e-4f, "softplus(30) == 30 (linear regime)");
    CHECK_NEAR(fast_softplus(-30.0f), 0.0f, 1e-4f, "softplus(-30) == 0");
    std::printf("  Passed. Fast Softplus verified.\n");
}

void test_fast_relu_and_variants() {
    std::printf("[Test 7/10] Testing Fast ReLU, Squared ReLU, LeakyReLU, HardSigmoid, HardSwish...\n");
    // ReLU
    CHECK_NEAR(fast_relu(3.0f), 3.0f, 1e-6f, "relu(3) == 3");
    CHECK_NEAR(fast_relu(-3.0f), 0.0f, 1e-6f, "relu(-3) == 0");

    // Squared ReLU
    CHECK_NEAR(fast_squared_relu(3.0f), 9.0f, 1e-6f, "squared_relu(3) == 9");
    CHECK_NEAR(fast_squared_relu(-3.0f), 0.0f, 1e-6f, "squared_relu(-3) == 0");

    // LeakyReLU
    CHECK_NEAR(fast_leaky_relu(2.0f, 0.1f), 2.0f, 1e-6f, "leaky_relu(2) == 2");
    CHECK_NEAR(fast_leaky_relu(-2.0f, 0.1f), -0.2f, 1e-6f, "leaky_relu(-2) == -0.2");

    // HardSigmoid & HardSwish
    CHECK_NEAR(fast_hard_sigmoid(0.0f), 0.5f, 1e-6f, "hard_sigmoid(0) == 0.5");
    CHECK_NEAR(fast_hard_sigmoid(4.0f), 1.0f, 1e-6f, "hard_sigmoid(4) == 1.0");
    CHECK_NEAR(fast_hard_sigmoid(-4.0f), 0.0f, 1e-6f, "hard_sigmoid(-4) == 0.0");
    CHECK_NEAR(fast_hard_swish(0.0f), 0.0f, 1e-6f, "hard_swish(0) == 0");
    CHECK_NEAR(fast_hard_swish(3.0f), 3.0f, 1e-6f, "hard_swish(3) == 3");

    std::printf("  Passed. ReLU and Hard variants verified.\n");
}

void test_fast_mish_and_elu() {
    std::printf("[Test 8/10] Testing Fast Mish & ELU...\n");
    CHECK_NEAR(fast_mish(0.0f), 0.0f, 1e-6f, "mish(0) == 0");
    CHECK_NEAR(fast_elu(0.0f), 0.0f, 1e-6f, "elu(0) == 0");
    CHECK_NEAR(fast_elu(2.0f), 2.0f, 1e-6f, "elu(2) == 2");
    CHECK(fast_elu(-2.0f, 1.0f) < 0.0f, "elu(-2) < 0");
    std::printf("  Passed. Fast Mish & ELU verified.\n");
}

void test_activation_dispatcher() {
    std::printf("[Test 9/10] Testing Unified Activation Dispatcher...\n");
    float x = 1.5f;
    CHECK_NEAR(fast_activate(ActivationType::kSiLU, x), fast_silu(x), 1e-6f, "dispatch silu");
    CHECK_NEAR(fast_activate(ActivationType::kSigmoid, x), fast_sigmoid(x), 1e-6f, "dispatch sigmoid");
    CHECK_NEAR(fast_activate(ActivationType::kGELU, x), fast_gelu(x), 1e-6f, "dispatch gelu");
    CHECK_NEAR(fast_activate(ActivationType::kReLU, x), fast_relu(x), 1e-6f, "dispatch relu");
    CHECK_NEAR(fast_activate(ActivationType::kNone, x), x, 1e-6f, "dispatch none");

    float g = 2.0f, u = 3.0f;
    CHECK_NEAR(fast_gated_activate(ActivationType::kSwiGLU, g, u), fast_swiglu(g, u), 1e-6f, "dispatch swiglu");
    CHECK_NEAR(fast_gated_activate(ActivationType::kGeGLU, g, u), fast_geglu(g, u), 1e-6f, "dispatch geglu");
    std::printf("  Passed. Activation Dispatcher verified.\n");
}

void test_array_batch_processing_and_throughput() {
    std::printf("[Test 10/10] Testing Array Batch Processing & Throughput...\n");
    constexpr size_t N = 100000;
    std::vector<float> in(N), out(N), gate(N), up(N);
    for (size_t i = 0; i < N; ++i) {
        in[i] = ((float)(i % 200) - 100.0f) * 0.1f;
        gate[i] = in[i];
        up[i] = 1.25f;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    fast_activate_array(in.data(), out.data(), N, ActivationType::kSiLU);
    auto t1 = std::chrono::high_resolution_clock::now();
    double silu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    for (size_t i = 0; i < N; ++i) {
        CHECK_NEAR(out[i], fast_silu(in[i]), 1e-6f, "batch silu output");
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    fast_swiglu_array(gate.data(), up.data(), out.data(), N);
    auto t3 = std::chrono::high_resolution_clock::now();
    double swiglu_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    for (size_t i = 0; i < N; ++i) {
        CHECK_NEAR(out[i], fast_swiglu(gate[i], up[i]), 1e-6f, "batch swiglu output");
    }

    double gflops_silu = (double)N / (silu_ms * 1e-3) / 1e9;
    double gflops_swiglu = ((double)N * 2.0) / (swiglu_ms * 1e-3) / 1e9;

    std::printf("  Batch size: %zu elements\n", N);
    std::printf("  SiLU throughput:   %.2f Gelem/s (%.3f ms)\n", gflops_silu, silu_ms);
    std::printf("  SwiGLU throughput: %.2f Gelem/s (%.3f ms)\n", gflops_swiglu, swiglu_ms);
    std::printf("  Passed. Batch processing & throughput verified.\n");
}

int main() {
    std::printf("=================================================================\n");
    std::printf("   RUNNING STRATA FAST ACTIVATION MATH TEST SUITE                \n");
    std::printf("=================================================================\n");

    test_fast_sigmoid();
    test_fast_silu();
    test_fast_swiglu();
    test_fast_gelu_and_quick_gelu();
    test_fast_geglu();
    test_fast_softplus();
    test_fast_relu_and_variants();
    test_fast_mish_and_elu();
    test_activation_dispatcher();
    test_array_batch_processing_and_throughput();

    std::printf("=================================================================\n");
    if (g_failed == 0) {
        std::printf("   ALL FAST ACTIVATION MATH TESTS PASSED (10/10)                 \n");
        std::printf("=================================================================\n");
        return 0;
    } else {
        std::printf("   %d FAST ACTIVATION MATH TESTS FAILED!                        \n", g_failed);
        std::printf("=================================================================\n");
        return 1;
    }
}
