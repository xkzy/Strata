// tests/kernels/test_fast_mamba.cpp - Unit Tests for Fast Mamba SSM Engine
#include "strata/kernels/fast_mamba.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace strata::kernels;

void test_mamba_conv1d() {
    std::cout << "[Test 1] Fast Mamba 1D Depthwise Conv with SiLU..." << std::endl;
    MambaConfig cfg;
    cfg.d_model = 64;
    cfg.expand = 2; // d_inner = 128
    cfg.d_conv = 4;

    FastMambaConv1D conv(cfg);

    int64_t d_in = cfg.d_inner();
    std::vector<float> x_in(d_in, 1.0f);
    // Weights: [d_inner * 4]
    std::vector<float> weight(d_in * 4, 0.25f);
    std::vector<float> bias(d_in, 0.0f);
    std::vector<float> x_out(d_in, 0.0f);

    // Step 1: history is all zeros, so sum = 1.0 * 0.25 = 0.25
    conv.step(x_in.data(), weight.data(), bias.data(), x_out.data());

    float expected_sum = 0.25f;
    float expected_silu = expected_sum / (1.0f + std::exp(-expected_sum));
    for (int64_t c = 0; c < d_in; ++c) {
        float diff = std::abs(x_out[c] - expected_silu);
        assert(diff < 1e-5f);
    }

    // Step 2, 3, 4: verify rolling history accumulation
    for (int step = 0; step < 3; ++step) {
        conv.step(x_in.data(), weight.data(), bias.data(), x_out.data());
    }

    // At step 4, all 4 kernel taps are filled with 1.0f: sum = 4 * 0.25 = 1.0f
    expected_sum = 1.0f;
    expected_silu = expected_sum / (1.0f + std::exp(-expected_sum));
    for (int64_t c = 0; c < d_in; ++c) {
        float diff = std::abs(x_out[c] - expected_silu);
        assert(diff < 1e-5f);
    }
    std::cout << "  [PASSED] 1D Convolution correctly rolls history and computes SiLU." << std::endl;
}

void test_mamba_selective_scan_o1_memory() {
    std::cout << "[Test 2] Fast Mamba Selective Scan Constant O(1) Memory Footprint..." << std::endl;
    MambaConfig cfg;
    cfg.d_model = 64;
    cfg.expand = 2; // d_inner = 128
    cfg.d_state = 16;

    FastMambaSSMState ssm_state(cfg);
    assert(ssm_state.state_bytes() == 128 * 16 * sizeof(float)); // Fixed constant size!

    int64_t d_in = cfg.d_inner();
    int64_t N = cfg.d_state;

    std::vector<float> u(d_in, 0.5f);
    std::vector<float> delta_raw(d_in, 0.1f);
    std::vector<float> dt_bias(d_in, 0.0f);
    std::vector<float> A_log(d_in * N, 0.0f); // A = -exp(0) = -1.0f (stable exponential decay)
    std::vector<float> B(N, 1.0f);
    std::vector<float> C(N, 0.5f);
    std::vector<float> D(d_in, 0.1f);
    std::vector<float> y_out(d_in, 0.0f);

    // Run 500 steps of autoregressive decode
    // Memory state stays exactly at 128 * 16 * 4 bytes (8 KiB) forever!
    for (int t = 0; t < 500; ++t) {
        FastMambaSelectiveScan::step(ssm_state, u.data(), delta_raw.data(), dt_bias.data(),
                                    A_log.data(), B.data(), C.data(), D.data(), cfg, y_out.data());
    }

    assert(ssm_state.state_bytes() == 128 * 16 * sizeof(float));
    for (int64_t c = 0; c < d_in; ++c) {
        assert(!std::isnan(y_out[c]) && !std::isinf(y_out[c]));
        assert(y_out[c] > 0.0f);
    }
    std::cout << "  [PASSED] Selective scan executed 500 tokens with strictly constant 8 KiB state memory." << std::endl;
}

void test_mamba_parallel_scan_parity() {
    std::cout << "[Test 3] Fast Mamba Parallel Scan Parity (Prefill vs Sequential)..." << std::endl;
    MambaConfig cfg;
    cfg.d_model = 32;
    cfg.expand = 2; // d_inner = 64
    cfg.d_state = 8;

    int64_t T = 16;
    int64_t d_in = cfg.d_inner();
    int64_t N = cfg.d_state;

    std::vector<float> u_seq(T * d_in);
    std::vector<float> delta_seq(T * d_in);
    std::vector<float> B_seq(T * N);
    std::vector<float> C_seq(T * N);
    for (size_t i = 0; i < u_seq.size(); ++i) u_seq[i] = static_cast<float>((i % 5) + 1) * 0.1f;
    for (size_t i = 0; i < delta_seq.size(); ++i) delta_seq[i] = 0.05f;
    for (size_t i = 0; i < B_seq.size(); ++i) B_seq[i] = 0.2f;
    for (size_t i = 0; i < C_seq.size(); ++i) C_seq[i] = 0.3f;

    std::vector<float> dt_bias(d_in, 0.0f);
    std::vector<float> A_log(d_in * N, -0.5f);
    std::vector<float> D(d_in, 0.05f);

    // 1. Parallel Scan Sequence
    FastMambaSSMState parallel_state(cfg);
    std::vector<float> y_parallel(T * d_in, 0.0f);
    FastMambaSelectiveScan::parallel_scan_sequence(
        u_seq.data(), delta_seq.data(), dt_bias.data(), A_log.data(),
        B_seq.data(), C_seq.data(), D.data(), T, cfg, parallel_state, y_parallel.data());

    // 2. Sequential Step-by-Step
    FastMambaSSMState seq_state(cfg);
    std::vector<float> y_seq(T * d_in, 0.0f);
    for (int64_t t = 0; t < T; ++t) {
        const float* u_t = u_seq.data() + (t * d_in);
        const float* delta_t = delta_seq.data() + (t * d_in);
        const float* B_t = B_seq.data() + (t * N);
        const float* C_t = C_seq.data() + (t * N);
        float* y_t = y_seq.data() + (t * d_in);

        FastMambaSelectiveScan::step(seq_state, u_t, delta_t, dt_bias.data(),
                                    A_log.data(), B_t, C_t, D.data(), cfg, y_t);
    }

    // Verify parity
    float max_diff = 0.0f;
    for (size_t i = 0; i < y_parallel.size(); ++i) {
        float diff = std::abs(y_parallel[i] - y_seq[i]);
        if (diff > max_diff) max_diff = diff;
    }
    assert(max_diff < 1e-5f);

    // Verify final state parity
    for (size_t i = 0; i < static_cast<size_t>(d_in * N); ++i) {
        float diff = std::abs(parallel_state.data()[i] - seq_state.data()[i]);
        assert(diff < 1e-5f);
    }
    std::cout << "  [PASSED] Parallel prefill scan matched sequential scan perfectly (max_diff=" << max_diff << ")." << std::endl;
}

void test_mamba_gated_fusion() {
    std::cout << "[Test 4] Fast Mamba Fused Gated Branch Multiplication..." << std::endl;
    int64_t d_in = 64;
    std::vector<float> y_ssm(d_in, 2.0f);
    std::vector<float> z_gate(d_in, 0.5f);
    std::vector<float> out(d_in, 0.0f);

    FastMambaBlock::fuse_gated_output(y_ssm.data(), z_gate.data(), d_in, out.data());

    float expected_silu = 0.5f / (1.0f + std::exp(-0.5f));
    float expected_out = 2.0f * expected_silu;

    for (int64_t i = 0; i < d_in; ++i) {
        float diff = std::abs(out[i] - expected_out);
        assert(diff < 1e-5f);
    }
    std::cout << "  [PASSED] Gated fusion y_ssm * SiLU(z) computed accurately." << std::endl;
}

int main() {
    std::cout << "=== Running Fast Mamba SSM Engine Tests ===" << std::endl;
    test_mamba_conv1d();
    test_mamba_selective_scan_o1_memory();
    test_mamba_parallel_scan_parity();
    test_mamba_gated_fusion();
    std::cout << "All Mamba SSM optimization tests passed successfully!" << std::endl;
    return 0;
}
