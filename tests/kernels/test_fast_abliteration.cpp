// tests/kernels/test_fast_abliteration.cpp - Unit and Parity Test Suite for Fast Online Abliteration Kernel
#include "strata/kernels/fast_abliteration.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
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

void test_normalization_and_dot_product() {
    std::printf("[Test 1/6] Testing Vector Normalization and SIMD Dot Product...\n");
    constexpr int64_t D = 2560; // Qwen hidden size
    std::vector<float> v(D);
    for (int64_t i = 0; i < D; ++i) {
        v[i] = std::sin(static_cast<float>(i) * 0.1f);
    }

    float orig_norm = FastAbliteration::normalize_vector(v.data(), D);
    CHECK(orig_norm > 0.0f, "original norm positive");

    float self_dot = FastAbliteration::dot_product(v.data(), v.data(), D);
    CHECK_NEAR(self_dot, 1.0f, 1e-5f, "unit vector self dot product == 1.0");

    std::printf("  Passed. Self dot product = %.6f (expected 1.0).\n", self_dot);
}

void test_orthogonal_projection_subtraction() {
    std::printf("[Test 2/6] Testing Orthogonal Projection Subtraction (x <- x - alpha * <x, r> * r)...\n");
    constexpr int64_t D = 2048;
    std::vector<float> r(D);
    for (int64_t i = 0; i < D; ++i) r[i] = std::cos(static_cast<float>(i) * 0.05f);
    FastAbliteration::normalize_vector(r.data(), D);

    // Create activation x = 3.0 * r + orthogonal_component
    std::vector<float> x(D);
    for (int64_t i = 0; i < D; ++i) {
        x[i] = 3.0f * r[i] + (i % 2 == 0 ? 0.5f : -0.5f);
    }

    // Measure initial projection
    float proj_before = FastAbliteration::dot_product(x.data(), r.data(), D);
    CHECK(std::fabs(proj_before) > 1.0f, "initial projection significant");

    // Perform full orthogonal abliteration (strength = 1.0)
    FastAbliteration::project_and_subtract_token(x.data(), r.data(), D, 1.0f, AbliterationMode::kOrthogonal);

    // The new projection onto r should be zero
    float proj_after = FastAbliteration::dot_product(x.data(), r.data(), D);
    CHECK_NEAR(proj_after, 0.0f, 1e-4f, "orthogonal projection removed");

    std::printf("  Passed. Projection before: %.4f, after: %.6f\n", proj_before, proj_after);
}

void test_positive_only_refusal_suppression() {
    std::printf("[Test 3/6] Testing Positive-Only Refusal Axis Suppression...\n");
    constexpr int64_t D = 1024;
    std::vector<float> r(D);
    for (int64_t i = 0; i < D; ++i) r[i] = (i % 3 == 0) ? 1.0f : 0.0f;
    FastAbliteration::normalize_vector(r.data(), D);

    // Negative projection case
    std::vector<float> x_neg(D);
    for (int64_t i = 0; i < D; ++i) x_neg[i] = -2.0f * r[i];
    std::vector<float> x_neg_copy = x_neg;

    FastAbliteration::project_and_subtract_token(x_neg.data(), r.data(), D, 1.0f, AbliterationMode::kPositiveOnly);
    for (int64_t i = 0; i < D; ++i) {
        CHECK_NEAR(x_neg[i], x_neg_copy[i], 1e-6f, "negative projection untouched in positive-only mode");
    }

    // Positive projection case
    std::vector<float> x_pos(D);
    for (int64_t i = 0; i < D; ++i) x_pos[i] = 2.0f * r[i];
    FastAbliteration::project_and_subtract_token(x_pos.data(), r.data(), D, 1.0f, AbliterationMode::kPositiveOnly);
    float proj_pos_after = FastAbliteration::dot_product(x_pos.data(), r.data(), D);
    CHECK_NEAR(proj_pos_after, 0.0f, 1e-4f, "positive projection inhibited");

    std::printf("  Passed. Positive component inhibited while negative component preserved.\n");
}

void test_layer_residual_abliteration() {
    std::printf("[Test 4/6] Testing Multi-Token Multi-Channel Residual Stream Abliteration...\n");
    constexpr int64_t T = 8;
    constexpr int64_t hc = 4;
    constexpr int64_t D = 2560;
    std::vector<float> R(T * hc * D);
    std::vector<float> r(D);
    for (int64_t i = 0; i < D; ++i) r[i] = 1.0f;
    FastAbliteration::normalize_vector(r.data(), D);

    for (size_t i = 0; i < R.size(); ++i) {
        R[i] = static_cast<float>(static_cast<int>(i % 100) - 50) * 0.01f;
    }


    FastAbliteration::abliterate_layer_residual(R.data(), T, hc, D, r.data(), 1.0f, AbliterationMode::kOrthogonal);

    for (int64_t t = 0; t < T; ++t) {
        for (int64_t c = 0; c < hc; ++c) {
            const float* vec = R.data() + (t * hc + c) * D;
            float proj = FastAbliteration::dot_product(vec, r.data(), D);
            CHECK_NEAR(proj, 0.0f, 1e-4f, "residual stream projection removed across all tokens and channels");
        }
    }
    std::printf("  Passed. Residual abliteration verified across %ld tokens x %ld channels.\n", T, hc);
}

void test_contrastive_refusal_vector() {
    std::printf("[Test 5/6] Testing Contrastive Refusal Vector Computation (mean_harm - mean_harmless)...\n");
    constexpr int64_t D = 512;
    std::vector<std::vector<float>> harmful(10, std::vector<float>(D, 0.0f));
    std::vector<std::vector<float>> harmless(10, std::vector<float>(D, 0.0f));

    for (size_t b = 0; b < 10; ++b) {
        for (int64_t i = 0; i < D; ++i) {
            harmful[b][i] = 5.0f + static_cast<float>(i % 5);
            harmless[b][i] = 1.0f + static_cast<float>(i % 5);
        }
    }

    std::vector<float> refusal_dir;
    bool ok = FastAbliteration::compute_contrastive_refusal_vector(harmful, harmless, D, refusal_dir);
    CHECK(ok, "contrastive calculation succeeded");
    CHECK((int64_t)refusal_dir.size() == D, "correct dimension");

    // All elements in harmful - harmless = 4.0; normalized unit vector has all elements equal to 1/sqrt(D)
    float expected_val = 1.0f / std::sqrt(static_cast<float>(D));
    for (int64_t i = 0; i < D; ++i) {
        CHECK_NEAR(refusal_dir[i], expected_val, 1e-4f, "contrastive direction element value");
    }

    std::printf("  Passed. Contrastive refusal vector matches analytical expectation.\n");
}

void test_serialization_round_trip() {
    std::printf("[Test 6/6] Testing Vector File Serialization Round Trip...\n");
    constexpr int64_t D = 2560;
    constexpr int64_t L = 36;
    std::vector<float> all_layers(L * D);

    // A fixed ramp stands in for a measured vector: the file path is what this test covers (a former
    // "canonical direction" generator was removed with docs/ABLITERATION_PLAN.md phase 0 - no direction may
    // be fabricated).
    for (int64_t l = 0; l < L; ++l) {
        for (int64_t i = 0; i < D; ++i) {
            all_layers[(size_t)(l * D + i)] = 0.25f + (float)((l * D + i) % 17) * 0.125f;
        }
    }

    std::string test_path = "/tmp/strata_test_ablr.bin";
    bool saved = FastAbliteration::save_vector_file(test_path, all_layers, D, L);
    CHECK(saved, "saved vector file");

    std::vector<float> loaded_data;
    int64_t loaded_D = 0, loaded_L = 0;
    bool loaded = FastAbliteration::load_vector_file(test_path, loaded_data, loaded_D, loaded_L);
    CHECK(loaded, "loaded vector file");
    CHECK(loaded_D == D, "loaded D match");
    CHECK(loaded_L == L, "loaded L match");
    CHECK(loaded_data.size() == all_layers.size(), "loaded size match");

    for (size_t i = 0; i < all_layers.size(); ++i) {
        CHECK_NEAR(loaded_data[i], all_layers[i], 1e-7f, "bit-exact serialization round trip");
    }

    std::remove(test_path.c_str());
    std::printf("  Passed. Binary serialization round trip verified bit-exact.\n");
}

int main() {
    std::printf("=== Strata Fast Online Abliteration Kernel Verification ===\n");
    test_normalization_and_dot_product();
    test_orthogonal_projection_subtraction();
    test_positive_only_refusal_suppression();
    test_layer_residual_abliteration();
    test_contrastive_refusal_vector();
    test_serialization_round_trip();

    if (g_failed > 0) {
        std::printf("=== %d TEST(S) FAILED ===\n", g_failed);
        return 1;
    }
    std::printf("=== All 6 Tests Passed Successfully ===\n");
    return 0;
}
