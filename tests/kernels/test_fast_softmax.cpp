// tests/kernels/test_fast_softmax.cpp - Verification suite for Fast Softmax & Logits Processor
#include "strata/kernels/fast_softmax.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <functional>
#include <random>

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
    std::printf("[Test 1/8] Testing Softmax Normalization & Probability Sum...\n");
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
    std::printf("[Test 2/8] Testing Temperature Scaling & Greedy Limit...\n");
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
    std::printf("[Test 3/8] Testing Numerically Stable Log-Sum-Exp (LSE)...\n");
    // Large values that would overflow exp(1000) if not stabilized
    std::vector<float> large_logits = {1000.0f, 1002.0f, 999.0f};
    float lse = FastSoftmax::log_sum_exp(large_logits.data(), large_logits.size());

    CHECK(!std::isnan(lse) && !std::isinf(lse), "LSE remains finite on extreme logits");
    CHECK(lse > 1002.0f, "LSE > max logit");
    CHECK_NEAR(lse, 1002.0f + std::log(1.0f + std::exp(-2.0f) + std::exp(-3.0f)), 1e-4f, "exact LSE calculation");
    std::printf("  Passed. Log-Sum-Exp numerical stability validated.\n");
}

void test_fast_argmax() {
    std::printf("[Test 4/8] Testing Fast ArgMax Token Selection...\n");
    std::vector<float> logits = {-1.5f, 0.2f, 8.4f, 3.1f, -0.9f};
    int32_t best = FastSoftmax::argmax(logits.data(), logits.size());
    CHECK(best == 2, "argmax finds index 2");
    std::printf("  Passed. Fast ArgMax validated.\n");
}

void test_min_p_filter() {
    std::printf("[Test 5/8] Testing Min-P Truncation Filter...\n");
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

// Exact reference: k-th largest by full sort; everything below it becomes -inf.
static void reference_top_k(std::vector<float>& l, int k) {
    std::vector<float> s = l;
    std::sort(s.begin(), s.end(), std::greater<float>());
    const float th = s[k - 1];
    for (float& v : l) if (v < th) v = -INFINITY;
}

void test_top_k_filter_matches_exact() {
    std::printf("[Test 6/8] Testing Top-K Filter against exact sort (ties, -inf, odd lengths)...\n");
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (size_t n : {2u, 9u, 64u, 257u, 5000u, 20000u}) {
        for (int k : {1, 2, 8, 20, 40, 100, 1000}) {
            if (static_cast<size_t>(k) >= n) continue;
            for (int variant = 0; variant < 3; ++variant) {
                std::vector<float> a(n);
                for (float& v : a) v = dist(rng);
                if (variant == 1) for (float& v : a) v = std::round(v);                   // many ties
                if (variant == 2) for (size_t i = 0; i < n; i += 3) a[i] = -INFINITY;     // pre-masked
                std::vector<float> want = a, got = a;
                reference_top_k(want, k);
                FastSoftmax::top_k_filter(got.data(), n, k);
                if (want != got) {
                    std::printf("  mismatch n=%zu k=%d variant=%d\n", n, k, variant);
                    CHECK(false, "top_k_filter equals exact reference");
                }
            }
        }
    }
    std::printf("  Passed. Top-K filter validated.\n");
}

void test_softmax_masked_and_edge_lengths() {
    std::printf("[Test 7/8] Testing Softmax -inf masking & non-multiple-of-8 lengths...\n");
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> dist(-30.0f, 30.0f);
    for (size_t n : {1u, 7u, 8u, 9u, 63u, 65u, 1000u}) {
        std::vector<float> a(n);
        for (float& v : a) v = dist(rng);
        if (n > 10) { a[3] = -INFINITY; a[n - 2] = -INFINITY; }
        std::vector<double> ref(n);
        double mx = -INFINITY, sum = 0.0;
        for (float v : a) mx = std::max(mx, (double)v);
        for (size_t i = 0; i < n; ++i) { ref[i] = std::exp((double)a[i] - mx); sum += ref[i]; }
        std::vector<float> got = a;
        FastSoftmax::softmax_inplace(got.data(), n);
        for (size_t i = 0; i < n; ++i) {
            if (std::isinf(a[i])) CHECK(got[i] == 0.0f, "masked logit has probability exactly 0");
            else if (ref[i] / sum > 1e-12) CHECK_NEAR(got[i], ref[i] / sum, 1e-4 * (ref[i] / sum) + 1e-12, "softmax matches double reference");
        }
    }
    std::printf("  Passed. Masking and edge lengths validated.\n");
}

void test_argmax_ties_nan_and_long() {
    std::printf("[Test 8/8] Testing ArgMax ties, NaN and long inputs...\n");
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (size_t n : {2u, 63u, 64u, 65u, 300u, 1000u, 152064u}) {
        std::vector<float> a(n);
        for (float& v : a) v = dist(rng);
        int ref = 0;
        for (size_t i = 1; i < n; ++i) if (a[i] > a[ref]) ref = (int)i;
        CHECK(FastSoftmax::argmax(a.data(), n) == ref, "argmax matches scalar scan");
        if (n > 10) {
            a[n / 3] = 100.0f; a[n - 1] = 100.0f;
            CHECK(FastSoftmax::argmax(a.data(), n) == (int)(n / 3), "first maximum wins ties");
            a[n / 3] = NAN; a[5] = -INFINITY;
            CHECK(FastSoftmax::argmax(a.data(), n) == (int)(n - 1), "NaN never wins");
        }
    }
    std::printf("  Passed. ArgMax edge cases validated.\n");
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
    test_top_k_filter_matches_exact();
    test_softmax_masked_and_edge_lengths();
    test_argmax_ties_nan_and_long();

    std::printf("=================================================================\n");
    if (g_failed == 0) {
        std::printf("   ALL FAST SOFTMAX TESTS PASSED (8/8)                           \n");
        std::printf("=================================================================\n");
        return 0;
    } else {
        std::printf("   %d FAST SOFTMAX TESTS FAILED!                                 \n", g_failed);
        std::printf("=================================================================\n");
        return 1;
    }
}
