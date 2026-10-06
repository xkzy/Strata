// tests/kernels/test_fast_rope.cpp - Verification suite for Fast RoPE Suite
#include "strata/kernels/fast_rope.hpp"

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

void test_rope_identity_at_pos_zero() {
    std::printf("[Test 1/5] Testing RoPE Identity at Position 0...\n");
    constexpr int64_t rot_dim = 64;
    std::vector<float> inv_freqs(rot_dim / 2), cos_tab(rot_dim / 2), sin_tab(rot_dim / 2);
    FastRoPE::compute_inv_freqs(inv_freqs.data(), rot_dim, 10000.0f);
    FastRoPE::compute_cos_sin(0, inv_freqs.data(), cos_tab.data(), sin_tab.data(), rot_dim / 2);

    for (size_t i = 0; i < rot_dim / 2; ++i) {
        CHECK_NEAR(cos_tab[i], 1.0f, 1e-6f, "cos(0) == 1");
        CHECK_NEAR(sin_tab[i], 0.0f, 1e-6f, "sin(0) == 0");
    }

    std::vector<float> vec(rot_dim), original(rot_dim);
    for (size_t i = 0; i < rot_dim; ++i) {
        vec[i] = ((float)(i % 20) - 10.0f) * 0.1f;
        original[i] = vec[i];
    }

    FastRoPE::apply_neox(vec.data(), cos_tab.data(), sin_tab.data(), rot_dim);

    for (size_t i = 0; i < rot_dim; ++i) {
        CHECK_NEAR(vec[i], original[i], 1e-6f, "RoPE at pos 0 is exact identity");
    }
    std::printf("  Passed. Position 0 identity verified.\n");
}

void test_rope_norm_preservation() {
    std::printf("[Test 2/5] Testing RoPE Orthogonal Norm Preservation...\n");
    constexpr int64_t rot_dim = 64;
    std::vector<float> inv_freqs(rot_dim / 2), cos_tab(rot_dim / 2), sin_tab(rot_dim / 2);
    FastRoPE::compute_inv_freqs(inv_freqs.data(), rot_dim, 10000.0f);
    FastRoPE::compute_cos_sin(42, inv_freqs.data(), cos_tab.data(), sin_tab.data(), rot_dim / 2);

    std::vector<float> vec(rot_dim);
    float orig_l2_sq = 0.0f;
    for (size_t i = 0; i < rot_dim; ++i) {
        vec[i] = ((float)(i % 30) - 15.0f) * 0.05f;
        orig_l2_sq += vec[i] * vec[i];
    }

    FastRoPE::apply_neox(vec.data(), cos_tab.data(), sin_tab.data(), rot_dim);

    float rot_l2_sq = 0.0f;
    for (size_t i = 0; i < rot_dim; ++i) {
        rot_l2_sq += vec[i] * vec[i];
    }

    CHECK_NEAR(std::sqrt(rot_l2_sq), std::sqrt(orig_l2_sq), 1e-4f, "2D rotation preserves vector Euclidean norm");
    std::printf("  Passed. Norm preservation verified.\n");
}

void test_rope_interleaved_vs_neox() {
    std::printf("[Test 3/5] Testing Interleaved vs NeoX Rotation Logic...\n");
    constexpr int64_t rot_dim = 4;
    float cos_val = std::cos(0.5f);
    float sin_val = std::sin(0.5f);
    std::vector<float> cos_tab = {cos_val, cos_val};
    std::vector<float> sin_tab = {sin_val, sin_val};

    // Interleaved pair: [x0, x1, x2, x3] -> rotate (x0, x1) and (x2, x3)
    std::vector<float> inter_vec = {1.0f, 2.0f, 3.0f, 4.0f};
    FastRoPE::apply_interleaved(inter_vec.data(), cos_tab.data(), sin_tab.data(), rot_dim);

    float exp_0 = 1.0f * cos_val - 2.0f * sin_val;
    float exp_1 = 1.0f * sin_val + 2.0f * cos_val;
    float exp_2 = 3.0f * cos_val - 4.0f * sin_val;
    float exp_3 = 3.0f * sin_val + 4.0f * cos_val;

    CHECK_NEAR(inter_vec[0], exp_0, 1e-5f, "interleaved 0");
    CHECK_NEAR(inter_vec[1], exp_1, 1e-5f, "interleaved 1");
    CHECK_NEAR(inter_vec[2], exp_2, 1e-5f, "interleaved 2");
    CHECK_NEAR(inter_vec[3], exp_3, 1e-5f, "interleaved 3");

    std::printf("  Passed. Interleaved RoPE verified.\n");
}

void test_multi_head_neox_rope() {
    std::printf("[Test 4/5] Testing Multi-Head NeoX RoPE...\n");
    constexpr int64_t n_heads = 4;
    constexpr int64_t head_dim = 64;
    constexpr int64_t rot_dim = 64;

    std::vector<float> inv_freqs(rot_dim / 2), cos_tab(rot_dim / 2), sin_tab(rot_dim / 2);
    FastRoPE::compute_inv_freqs(inv_freqs.data(), rot_dim, 10000.0f);
    FastRoPE::compute_cos_sin(128, inv_freqs.data(), cos_tab.data(), sin_tab.data(), rot_dim / 2);

    std::vector<float> qk(n_heads * head_dim);
    for (size_t i = 0; i < qk.size(); ++i) qk[i] = ((float)(i % 25) - 12.0f) * 0.1f;

    std::vector<float> qk_ref = qk;
    for (int64_t h = 0; h < n_heads; ++h) {
        FastRoPE::apply_neox(qk_ref.data() + h * head_dim, cos_tab.data(), sin_tab.data(), rot_dim);
    }

    FastRoPE::apply_multi_head_neox(qk.data(), n_heads, head_dim, rot_dim, cos_tab.data(), sin_tab.data());

    for (size_t i = 0; i < qk.size(); ++i) {
        CHECK_NEAR(qk[i], qk_ref[i], 1e-4f, "multi-head matches per-head rotation");
    }
    std::printf("  Passed. Multi-Head RoPE verified.\n");
}

void test_yarn_context_scaling() {
    std::printf("[Test 5/5] Testing YaRN Long-Context RoPE Frequency Scaling...\n");
    constexpr int64_t rot_dim = 64;
    RoPEConfig cfg;
    cfg.freq_base = 10000.0f;
    cfg.beta_fast = 32.0f;
    cfg.beta_slow = 1.0f;

    std::vector<float> base_freqs(rot_dim / 2), yarn_freqs(rot_dim / 2);
    FastRoPE::compute_inv_freqs(base_freqs.data(), rot_dim, cfg.freq_base);
    FastRoPE::compute_yarn_freqs(yarn_freqs.data(), rot_dim, cfg, 4096.0f, 32768.0f);

    for (size_t i = 0; i < rot_dim / 2; ++i) {
        CHECK(yarn_freqs[i] > 0.0f, "YaRN frequencies must be strictly positive");
        CHECK(yarn_freqs[i] <= base_freqs[i], "interpolated frequencies <= base frequencies");
    }
    std::printf("  Passed. YaRN scaling verified.\n");
}

int main() {
    std::printf("=================================================================\n");
    std::printf("   RUNNING STRATA FAST ROPE TEST SUITE                           \n");
    std::printf("=================================================================\n");

    test_rope_identity_at_pos_zero();
    test_rope_norm_preservation();
    test_rope_interleaved_vs_neox();
    test_multi_head_neox_rope();
    test_yarn_context_scaling();

    std::printf("=================================================================\n");
    if (g_failed == 0) {
        std::printf("   ALL FAST ROPE TESTS PASSED (5/5)                              \n");
        std::printf("=================================================================\n");
        return 0;
    } else {
        std::printf("   %d FAST ROPE TESTS FAILED!                                    \n", g_failed);
        std::printf("=================================================================\n");
        return 1;
    }
}
