// tests/kernels/test_fast_softmax.cpp - Verification suite for Fast Softmax & Logits Processor
#include "strata/kernels/fast_softmax.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cassert>
#include <numeric>

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

void test_softmax_probability_sum_and_monotonicity() {
    std::printf("[Test 1/5] Testing Softmax Normalization & Probability Sum...\n");
    std::vector<float> logits = {2.0f, 1.0f, 0.1f, -1.0f, 5.0f, 3.5f};
    FastSoftmax::softmax_inplace(logits.data(), logits.size(), 1.0f);

    float sum = 0.0f;
    for (float p : logits) {
        CHECK(p >= 0.0f && p <= 1.0f, "probabilities must be in [0, 1]");
        sum += p;
    }
    CHECK_NEAR(sum, 1.0f, 1e-5f, "softmax probability sum == 1.0");
    CHECK(logits[4] > logits[5] && logits[5] > logits[0], "highest logit produces highest probability");
    std::printf("  Passed. Softmax normalization validated.\n");
}

void test_softmax_temperature_scaling() {
    std::printf("[Test 2/5] Testing Temperature Scaling & Greedy Limit...\n");
    std::vector<float> logits = {1.0f, 4.0f, 2.0f};

    // Greedy mode (temp = 0.0)
    std::vector<float> greedy = logits;
    FastSoftmax::softmax_inplace(greedy.data(), greedy.size(), 0.0f);
    CHECK_NEAR(greedy[1], 1.0f, 1e-6f, "greedy mode assigns probability 1.0 to argmax");
    CHECK_NEAR(greedy[0], 0.0f, 1e-6f, "greedy mode assigns probability 0.0 to others");

    // High temperature (smooth distribution)
    std::vector<float> warm = logits;
    FastSoftmax::softmax_inplace(warm.data(), warm.size(), 2.0f);
    CHECK(warm[1] < 0.85f, "higher temperature softens peak probability");
    std::printf("  Passed. Temperature scaling validated.\n");
}

void test_log_sum_exp_stability() {
    std::printf("[Test 3/5] Testing Numerically Stable Log-Sum-Exp (LSE)...\n");
    // Large values that would overflow exp(1000) if not stabilized
    std::vector<float> large_logits = {1000.0f, 1002.0f, 999.0f};
    float lse = FastSoftmax::log_sum_exp(large_logits.data(), large_logits.size());

    CHECK(!std::isnan(lse) && !std::isinf(lse), "LSE remains finite on extreme logits");
    CHECK(lse > 1002.0f, "LSE > max logit");
    CHECK_NEAR(lse, 1002.0f + std::log(1.0f + std::exp(-2.0f) + std::exp(-3.0f)), 1e-4f, "exact LSE calculation");
    std::printf("  Passed. Log-Sum-Exp numerical stability validated.\n");
}

void test_fast_argmax() {
    std::printf("[Test 4/5] Testing Fast ArgMax Token Selection...\n");
    std::vector<float> logits = {-1.5f, 0.2f, 8.4f, 3.1f, -0.9f};
    int32_t best = FastSoftmax::argmax(logits.data(), logits.size());
    CHECK(best == 2, "argmax finds index 2");
    std::printf("  Passed. Fast ArgMax validated.\n");
}

void test_min_p_filter() {
    std::printf("[Test 5/5] Testing Min-P Truncation Filter...\n");
    std::vector<float> probs = {0.60f, 0.30f, 0.08f, 0.02f}; // p_max = 0.60, min_p = 0.1 -> threshold = 0.06
    FastSoftmax::min_p_filter(probs.data(), probs.size(), 0.10f);

    CHECK(probs[0] > 0.0f, "0.60 preserved");
    CHECK(probs[1] > 0.0f, "0.30 preserved");
    CHECK(probs[2] > 0.0f, "0.08 preserved");
    CHECK_NEAR(probs[3], 0.0f, 1e-6f, "0.02 clipped to 0");

    float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
    CHECK_NEAR(sum, 1.0f, 1e-5f, "min-p filtered distribution renormalizes to 1.0");
    std::printf("  Passed. Min-P filter validated.\n");
}

int main() {
    std::printf("=================================================================\n");
    std::printf("   RUNNING STRATA FAST SOFTMAX & LOGITS TEST SUITE               \n");
    std::printf("=================================================================\n");

    test_softmax_probability_sum_and_monotonicity();
    test_softmax_temperature_scaling();
    test_log_sum_exp_stability();
    test_fast_argmax();
    test_min_p_filter();

    std::printf("=================================================================\n");
    if (g_failed == 0) {
        std::printf("   ALL FAST SOFTMAX TESTS PASSED (5/5)                           \n");
        std::printf("=================================================================\n");
        return 0;
    } else {
        std::printf("   %d FAST SOFTMAX TESTS FAILED!                                 \n", g_failed);
        std::printf("=================================================================\n");
        return 1;
    }
}
