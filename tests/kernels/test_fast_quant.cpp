// tests/kernels/test_fast_quant.cpp - Verification suite for Fast Vectorized Quantization Engine
#include "strata/kernels/fast_quant.hpp"

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

void test_fp16_round_trip() {
    std::printf("[Test 1/4] Testing Vectorized FP32 <-> FP16 Round-Trip...\n");
    constexpr size_t N = 64;
    std::vector<float> orig(N), decoded(N);
    std::vector<uint16_t> fp16_data(N);

    for (size_t i = 0; i < N; ++i) {
        orig[i] = ((float)(i % 30) - 15.0f) * 0.25f;
    }

    FastQuant::f32_to_f16(orig.data(), fp16_data.data(), N);
    FastQuant::f16_to_f32(fp16_data.data(), decoded.data(), N);

    for (size_t i = 0; i < N; ++i) {
        CHECK_NEAR(decoded[i], orig[i], 1e-3f, "fp16 round trip");
    }
    std::printf("  Passed. FP16 vectorized conversions validated.\n");
}

void test_bf16_round_trip() {
    std::printf("[Test 2/4] Testing Vectorized FP32 <-> BF16 Round-Trip...\n");
    constexpr size_t N = 64;
    std::vector<float> orig(N), decoded(N);
    std::vector<uint16_t> bf16_data(N);

    for (size_t i = 0; i < N; ++i) {
        orig[i] = ((float)(i % 40) - 20.0f) * 0.125f;
    }

    FastQuant::f32_to_bf16(orig.data(), bf16_data.data(), N);
    FastQuant::bf16_to_f32(bf16_data.data(), decoded.data(), N);

    for (size_t i = 0; i < N; ++i) {
        CHECK_NEAR(decoded[i], orig[i], 1e-2f, "bf16 round trip");
    }
    std::printf("  Passed. BF16 vectorized conversions validated.\n");
}

void test_q8_0_quantization() {
    std::printf("[Test 3/4] Testing Q8_0 Block Quantization & Dequantization...\n");
    constexpr size_t n_blocks = 4;
    constexpr size_t N = n_blocks * 32;

    std::vector<float> orig(N), decoded(N), scales(n_blocks);
    std::vector<int8_t> quants(N);

    for (size_t i = 0; i < N; ++i) {
        orig[i] = ((float)(i % 50) - 25.0f) * 0.05f;
    }

    FastQuant::quantize_q8_0(orig.data(), quants.data(), scales.data(), n_blocks);
    FastQuant::dequantize_q8_0(quants.data(), scales.data(), decoded.data(), n_blocks);

    for (size_t i = 0; i < N; ++i) {
        CHECK_NEAR(decoded[i], orig[i], 0.05f, "q8_0 quantization error within 1 LSB");
    }
    std::printf("  Passed. Q8_0 block quantization validated.\n");
}

void test_q4_0_dequantization() {
    std::printf("[Test 4/4] Testing Q4_0 4-bit Nibble Unpacking & Scaling...\n");
    constexpr size_t n_elems = 32;
    std::vector<uint8_t> nibbles(16); // 16 bytes = 32 x 4-bit nibbles
    // Fill with byte 0x88 -> lo = 8-8 = 0, hi = 8-8 = 0
    // Byte 0x97 -> lo = 7-8 = -1, hi = 9-8 = +1
    nibbles[0] = 0x97;
    for (size_t i = 1; i < 16; ++i) nibbles[i] = 0x88;

    std::vector<float> decoded(n_elems);
    float scale = 0.5f;

    FastQuant::dequantize_q4_0(nibbles.data(), scale, decoded.data(), n_elems);

    CHECK_NEAR(decoded[0], -0.5f, 1e-6f, "q4_0 lo nibble: (7 - 8) * 0.5 = -0.5");
    CHECK_NEAR(decoded[1], 0.5f, 1e-6f, "q4_0 hi nibble: (9 - 8) * 0.5 = +0.5");
    CHECK_NEAR(decoded[2], 0.0f, 1e-6f, "q4_0 zero nibble: (8 - 8) * 0.5 = 0.0");
    std::printf("  Passed. Q4_0 nibble unpacking validated.\n");
}

int main() {
    std::printf("=================================================================\n");
    std::printf("   RUNNING STRATA FAST QUANTIZATION TEST SUITE                   \n");
    std::printf("=================================================================\n");

    test_fp16_round_trip();
    test_bf16_round_trip();
    test_q8_0_quantization();
    test_q4_0_dequantization();

    std::printf("=================================================================\n");
    if (g_failed == 0) {
        std::printf("   ALL FAST QUANTIZATION TESTS PASSED (4/4)                      \n");
        std::printf("=================================================================\n");
        return 0;
    } else {
        std::printf("   %d FAST QUANTIZATION TESTS FAILED!                            \n", g_failed);
        std::printf("=================================================================\n");
        return 1;
    }
}
