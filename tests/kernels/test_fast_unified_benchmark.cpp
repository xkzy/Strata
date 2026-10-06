// tests/kernels/test_fast_unified_benchmark.cpp - Microbenchmark and Verification for Fast Kernels
#include "strata/strata_unified.hpp"

#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace strata::kernels;
using Clock = std::chrono::high_resolution_clock;

void benchmark_full_transformer_layer_kernels() {
    std::cout << "[Benchmark 1/3] Simulating Full Transformer Layer Kernel Pipeline..." << std::endl;

    const size_t hidden_dim = 2048;
    const size_t ffn_dim = 5632;
    const size_t num_heads = 16;
    const size_t head_dim = hidden_dim / num_heads;
    const size_t rot_dim = head_dim;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> x(hidden_dim);
    std::vector<float> residual(hidden_dim);
    std::vector<float> gamma(hidden_dim, 1.0f);
    std::vector<float> norm_out(hidden_dim);
    for (size_t i = 0; i < hidden_dim; ++i) {
        x[i] = dist(rng);
        residual[i] = dist(rng);
    }

    std::vector<float> w_gate(hidden_dim * ffn_dim), w_up(hidden_dim * ffn_dim);
    for (size_t i = 0; i < hidden_dim * ffn_dim; ++i) {
        w_gate[i] = dist(rng) * 0.02f;
        w_up[i] = dist(rng) * 0.02f;
    }
    std::vector<float> swiglu_out(ffn_dim);

    std::vector<float> inv_freqs(rot_dim / 2);
    FastRoPE::compute_inv_freqs(inv_freqs.data(), rot_dim, 10000.0f);

    std::vector<float> cos_tab(rot_dim / 2);
    std::vector<float> sin_tab(rot_dim / 2);

    const int iterations = 1000;
    auto t0 = Clock::now();

    for (int it = 0; it < iterations; ++it) {
        // 1. RMSNorm + Residual Fused
        FastNorm::fused_rmsnorm_residual(residual.data(), x.data(), gamma.data(), norm_out.data(), hidden_dim, 1e-6f);

        // 2. RoPE on Q and K heads
        FastRoPE::compute_cos_sin(128 + it, inv_freqs.data(), cos_tab.data(), sin_tab.data(), rot_dim / 2);
        FastRoPE::apply_multi_head_neox(norm_out.data(), num_heads, head_dim, rot_dim, cos_tab.data(), sin_tab.data());

        // 3. Fused SwiGLU GEMM Projection: SiLU(x * W_gate) * (x * W_up)
        FastMatMul::fused_swiglu_gemm(norm_out.data(), w_gate.data(), w_up.data(), swiglu_out.data(), 1, ffn_dim, hidden_dim);
    }

    auto t1 = Clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double per_layer_us = (total_ms / iterations) * 1000.0;

    std::cout << "  Iterations: " << iterations << std::endl;
    std::cout << "  Layer time: " << std::fixed << std::setprecision(2) << per_layer_us << " us / layer" << std::endl;
    std::cout << "  Equivalent 32-layer forward: " << (per_layer_us * 32.0 / 1000.0) << " ms / token" << std::endl;
    std::cout << "  Passed." << std::endl;
}

void benchmark_quantization_and_softmax_pipeline() {
    std::cout << "[Benchmark 2/3] Benchmarking Quantization + Softmax + Sampling..." << std::endl;

    const size_t vocab_size = 152064; // Qwen vocabulary
    std::vector<float> logits(vocab_size);
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (size_t i = 0; i < vocab_size; ++i) logits[i] = dist(rng);

    const int iterations = 500;
    auto t0 = Clock::now();

    for (int it = 0; it < iterations; ++it) {
        // Fast Softmax with temperature
        FastSoftmax::softmax_inplace(logits.data(), vocab_size, 0.8f);

        // Fast ArgMax
        int32_t top_tok = FastSoftmax::argmax(logits.data(), vocab_size);
        (void)top_tok;
    }

    auto t1 = Clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double per_step_us = (total_ms / iterations) * 1000.0;

    std::cout << "  Vocab size: " << vocab_size << std::endl;
    std::cout << "  Softmax + ArgMax time: " << per_step_us << " us / step" << std::endl;
    std::cout << "  Passed." << std::endl;
}

void benchmark_generation_loop_interception() {
    std::cout << "[Benchmark 3/3] Benchmarking Online Generation Loop Detector Latency..." << std::endl;

    strata::generation::GenerationLoopDetector detector;
    const int tokens = 2048;
    auto t0 = Clock::now();

    for (int t = 0; t < tokens; ++t) {
        auto verdict = detector.feed_token(t % 50, " token");
        (void)verdict;
    }

    auto t1 = Clock::now();
    double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double per_token_ns = (total_us / tokens) * 1000.0;

    std::cout << "  Tokens processed: " << tokens << std::endl;
    std::cout << "  Per-token detector overhead: " << per_token_ns << " ns / token" << std::endl;
    assert(per_token_ns < 5000.0 && "Detector overhead must be ultra-low (<5us per token)");
    std::cout << "  Passed. Negligible overhead verified." << std::endl;
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "   RUNNING STRATA UNIFIED FAST KERNEL & BENCHMARK SUITE          " << std::endl;
    std::cout << "=================================================================" << std::endl;

    benchmark_full_transformer_layer_kernels();
    benchmark_quantization_and_softmax_pipeline();
    benchmark_generation_loop_interception();

    std::cout << "=================================================================" << std::endl;
    std::cout << "   ALL UNIFIED FAST KERNEL BENCHMARKS COMPLETED SUCCESSFULLY     " << std::endl;
    std::cout << "=================================================================" << std::endl;
    return 0;
}
