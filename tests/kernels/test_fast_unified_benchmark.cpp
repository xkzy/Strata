// tests/kernels/test_fast_unified_benchmark.cpp - Microbenchmark and Verification for Fast Kernels
#include "strata/strata_unified.hpp"
#include "strata/kernels/fast_mimo.hpp"
#include "strata/kernels/fast_mamba.hpp"

#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace strata::kernels;
using Clock = std::chrono::high_resolution_clock;

// Benchmark 1: Dense Transformer Layer Kernel Pipeline (RMSNorm, RoPE, SwiGLU GEMM)
void benchmark_full_transformer_layer_kernels() {
    std::cout << "[Benchmark 1/6] Simulating Dense Transformer Layer Kernel Pipeline..." << std::endl;

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

    const int iterations = 200;
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
    assert(!std::isnan(norm_out[0]) && !std::isinf(norm_out[0]));
    std::cout << "  Passed." << std::endl;
}

// Benchmark 2: FastAttention Pipeline (Single-head & GQA KV Cache Decode)
void benchmark_fast_attention_pipeline() {
    std::cout << "[Benchmark 2/6] Benchmarking FastAttention Pipeline (Single-head & GQA KV Decode)..." << std::endl;

    const size_t num_heads = 16;
    const size_t num_kv_heads = 4;
    const size_t head_dim = 128;
    const size_t hidden_dim = num_heads * head_dim; // 2048
    const size_t seq_len = 512;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> q(hidden_dim);
    for (size_t i = 0; i < hidden_dim; ++i) q[i] = dist(rng);

    std::vector<float> k_cache(seq_len * num_kv_heads * head_dim);
    std::vector<float> v_cache(seq_len * num_kv_heads * head_dim);
    for (size_t i = 0; i < k_cache.size(); ++i) {
        k_cache[i] = dist(rng) * 0.1f;
        v_cache[i] = dist(rng) * 0.1f;
    }

    std::vector<float> attn_out(hidden_dim, 0.0f);
    MathContext ctx_balanced(MathPolicy::kBalanced);
    MathContext ctx_fast(MathPolicy::kFast);

    const int iterations = 200;

    // 1. GQA Multi-head decode benchmark (Balanced policy)
    auto t0 = Clock::now();
    for (int it = 0; it < iterations; ++it) {
        FastAttention::multi_head_scaled_dot_product_decode(
            q.data(), k_cache.data(), v_cache.data(), attn_out.data(),
            num_heads, num_kv_heads, seq_len, head_dim, scale, ctx_balanced
        );
    }
    auto t1 = Clock::now();
    double total_ms_gqa = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double per_step_us_gqa = (total_ms_gqa / iterations) * 1000.0;

    for (size_t i = 0; i < hidden_dim; ++i) {
        assert(!std::isnan(attn_out[i]) && !std::isinf(attn_out[i]));
    }

    // 2. Single-head decode benchmark (Fast policy)
    std::vector<float> single_q(head_dim);
    std::vector<float> single_k(seq_len * head_dim);
    std::vector<float> single_v(seq_len * head_dim);
    std::vector<float> single_out(head_dim);
    for (size_t i = 0; i < head_dim; ++i) single_q[i] = dist(rng);
    for (size_t i = 0; i < seq_len * head_dim; ++i) {
        single_k[i] = dist(rng) * 0.1f;
        single_v[i] = dist(rng) * 0.1f;
    }

    auto t2 = Clock::now();
    for (int it = 0; it < iterations; ++it) {
        FastAttention::scaled_dot_product_decode(
            single_q.data(), single_k.data(), single_v.data(), single_out.data(),
            seq_len, head_dim, scale, ctx_fast
        );
    }
    auto t3 = Clock::now();
    double total_ms_single = std::chrono::duration<double, std::milli>(t3 - t2).count();
    double per_step_us_single = (total_ms_single / iterations) * 1000.0;

    std::cout << "  KV Cache sequence length: " << seq_len << " tokens" << std::endl;
    std::cout << "  GQA Heads: " << num_heads << " Q-heads / " << num_kv_heads << " KV-heads (head_dim=" << head_dim << ")" << std::endl;
    std::cout << "  GQA Decode Latency: " << std::fixed << std::setprecision(2) << per_step_us_gqa << " us / step" << std::endl;
    std::cout << "  Single-Head Decode Latency: " << per_step_us_single << " us / head" << std::endl;
    std::cout << "  Passed." << std::endl;
}

// Benchmark 3: FastMoE Pipeline (Top-K Routing, Permutation & Aggregation)
void benchmark_fast_moe_pipeline() {
    std::cout << "[Benchmark 3/6] Benchmarking FastMoE Pipeline (Top-K Routing, Permutation & Aggregation)..." << std::endl;

    const size_t num_experts = 64;
    const size_t top_k = 8;
    const size_t hidden_dim = 2048;
    const size_t num_tokens = 8;

    std::mt19937 rng(54321);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

    // 1. Router logits
    std::vector<float> logits(num_tokens * num_experts);
    for (size_t i = 0; i < logits.size(); ++i) logits[i] = dist(rng);

    std::vector<int32_t> topk_indices(num_tokens * top_k);
    std::vector<float> topk_weights(num_tokens * top_k);

    const int iterations = 1000;

    // Benchmark Batched Top-K Router
    auto t0 = Clock::now();
    for (int it = 0; it < iterations; ++it) {
        FastMoE::route_topk_batch(
            logits.data(), num_tokens, num_experts, top_k,
            topk_indices.data(), topk_weights.data()
        );
    }
    auto t1 = Clock::now();
    double total_ms_router = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double per_token_us_router = (total_ms_router / (iterations * num_tokens)) * 1000.0;

    // Verify routing properties: weights sum to 1.0 per token
    for (size_t t = 0; t < num_tokens; ++t) {
        float sum_w = 0.0f;
        for (size_t k = 0; k < top_k; ++k) {
            sum_w += topk_weights[t * top_k + k];
            assert(topk_indices[t * top_k + k] >= 0 && topk_indices[t * top_k + k] < static_cast<int32_t>(num_experts));
        }
        assert(std::abs(sum_w - 1.0f) < 1e-4f);
    }

    // Benchmark Dispatch Permutation
    std::vector<int32_t> expert_counts(num_experts);
    std::vector<int32_t> expert_offsets(num_experts + 1);
    std::vector<int32_t> token_indices(num_tokens * top_k);
    std::vector<int32_t> slot_indices(num_tokens * top_k);

    auto t2 = Clock::now();
    for (int it = 0; it < iterations; ++it) {
        FastMoE::compute_dispatch_permutation(
            topk_indices.data(), num_tokens, top_k, num_experts,
            expert_counts.data(), expert_offsets.data(),
            token_indices.data(), slot_indices.data()
        );
    }
    auto t3 = Clock::now();
    double total_ms_dispatch = std::chrono::duration<double, std::milli>(t3 - t2).count();
    double per_batch_us_dispatch = (total_ms_dispatch / iterations) * 1000.0;

    // Benchmark Weighted Expert Aggregation
    std::vector<float> expert_outputs(top_k * hidden_dim);
    for (size_t i = 0; i < expert_outputs.size(); ++i) expert_outputs[i] = dist(rng);
    std::vector<float> aggregated_out(hidden_dim);

    auto t4 = Clock::now();
    for (int it = 0; it < iterations; ++it) {
        FastMoE::weighted_aggregate(
            expert_outputs.data(), topk_weights.data(), top_k, hidden_dim, aggregated_out.data()
        );
    }
    auto t5 = Clock::now();
    double total_ms_agg = std::chrono::duration<double, std::milli>(t5 - t4).count();
    double per_token_us_agg = (total_ms_agg / iterations) * 1000.0;

    for (size_t i = 0; i < hidden_dim; ++i) {
        assert(!std::isnan(aggregated_out[i]) && !std::isinf(aggregated_out[i]));
    }

    std::cout << "  Experts: " << num_experts << " (top_k=" << top_k << "), Hidden Dim: " << hidden_dim << std::endl;
    std::cout << "  Top-K Router Latency: " << std::fixed << std::setprecision(2) << per_token_us_router << " us / token" << std::endl;
    std::cout << "  Dispatch Permutation Latency: " << per_batch_us_dispatch << " us / batch-8" << std::endl;
    std::cout << "  Weighted Aggregation Latency: " << per_token_us_agg << " us / token" << std::endl;
    std::cout << "  Passed." << std::endl;
}

// Benchmark 4: End-to-End MoE Transformer Forward Layer Pipeline Simulation
void benchmark_full_moe_transformer_layer_kernels() {
    std::cout << "[Benchmark 4/6] Simulating End-to-End MoE Transformer Forward Layer Pipeline..." << std::endl;

    const size_t hidden_dim = 2048;
    const size_t num_heads = 16;
    const size_t num_kv_heads = 4;
    const size_t head_dim = hidden_dim / num_heads; // 128
    const size_t rot_dim = head_dim;
    const size_t seq_len = 256;
    const size_t num_experts = 8;
    const size_t top_k = 2;
    const size_t expert_ffn_dim = 2048;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::mt19937 rng(999);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> x(hidden_dim);
    std::vector<float> residual(hidden_dim);
    std::vector<float> gamma(hidden_dim, 1.0f);
    std::vector<float> norm_out(hidden_dim);
    for (size_t i = 0; i < hidden_dim; ++i) {
        x[i] = dist(rng);
        residual[i] = dist(rng);
    }

    // KV Cache
    std::vector<float> k_cache(seq_len * num_kv_heads * head_dim);
    std::vector<float> v_cache(seq_len * num_kv_heads * head_dim);
    for (size_t i = 0; i < k_cache.size(); ++i) {
        k_cache[i] = dist(rng) * 0.05f;
        v_cache[i] = dist(rng) * 0.05f;
    }
    std::vector<float> attn_out(hidden_dim);

    // RoPE tables
    std::vector<float> inv_freqs(rot_dim / 2);
    FastRoPE::compute_inv_freqs(inv_freqs.data(), rot_dim, 10000.0f);
    std::vector<float> cos_tab(rot_dim / 2);
    std::vector<float> sin_tab(rot_dim / 2);

    // MoE Router & Experts
    std::vector<float> router_logits(num_experts);
    for (size_t i = 0; i < num_experts; ++i) router_logits[i] = dist(rng);
    std::vector<int32_t> selected_experts(top_k);
    std::vector<float> selected_weights(top_k);

    // Top-k experts weights
    std::vector<float> w_gate(top_k * hidden_dim * expert_ffn_dim);
    std::vector<float> w_up(top_k * hidden_dim * expert_ffn_dim);
    for (size_t i = 0; i < w_gate.size(); ++i) {
        w_gate[i] = dist(rng) * 0.02f;
        w_up[i] = dist(rng) * 0.02f;
    }
    std::vector<float> expert_ffn_outs(top_k * expert_ffn_dim);
    std::vector<float> moe_combined(expert_ffn_dim);

    const int iterations = 50;
    auto t0 = Clock::now();

    for (int it = 0; it < iterations; ++it) {
        // 1. Pre-Attention Fused RMSNorm + Residual
        FastNorm::fused_rmsnorm_residual(residual.data(), x.data(), gamma.data(), norm_out.data(), hidden_dim, 1e-6f);

        // 2. RoPE on Q & K
        FastRoPE::compute_cos_sin(100 + it, inv_freqs.data(), cos_tab.data(), sin_tab.data(), rot_dim / 2);
        FastRoPE::apply_multi_head_neox(norm_out.data(), num_heads, head_dim, rot_dim, cos_tab.data(), sin_tab.data());

        // 3. FastAttention GQA Decode over KV cache
        FastAttention::multi_head_scaled_dot_product_decode(
            norm_out.data(), k_cache.data(), v_cache.data(), attn_out.data(),
            num_heads, num_kv_heads, seq_len, head_dim, scale
        );

        // 4. Post-Attention Fused RMSNorm + Residual
        FastNorm::fused_rmsnorm_residual(x.data(), attn_out.data(), gamma.data(), norm_out.data(), hidden_dim, 1e-6f);

        // 5. FastMoE Routing: Top-K Router
        FastMoE::route_topk(router_logits.data(), num_experts, top_k, selected_experts.data(), selected_weights.data());

        // 6. Expert FFN Execution (SwiGLU GEMM for each selected expert)
        for (size_t k = 0; k < top_k; ++k) {
            const float* wg = w_gate.data() + k * hidden_dim * expert_ffn_dim;
            const float* wu = w_up.data() + k * hidden_dim * expert_ffn_dim;
            float* e_out = expert_ffn_outs.data() + k * expert_ffn_dim;
            FastMatMul::fused_swiglu_gemm(norm_out.data(), wg, wu, e_out, 1, expert_ffn_dim, hidden_dim);
        }

        // 7. FastMoE Weighted Expert Aggregation
        FastMoE::weighted_aggregate(
            expert_ffn_outs.data(), selected_weights.data(), top_k, expert_ffn_dim, moe_combined.data()
        );
    }

    auto t1 = Clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double per_layer_us = (total_ms / iterations) * 1000.0;

    std::cout << "  Iterations: " << iterations << std::endl;
    std::cout << "  Full MoE Layer time (Norm + RoPE + GQA + Router + Experts + Agg): "
              << std::fixed << std::setprecision(2) << per_layer_us << " us / layer" << std::endl;
    std::cout << "  Equivalent 32-layer MoE forward: " << (per_layer_us * 32.0 / 1000.0) << " ms / token" << std::endl;
    assert(!std::isnan(moe_combined[0]) && !std::isinf(moe_combined[0]));
    std::cout << "  Passed." << std::endl;
}

// Benchmark 5: Quantization + Softmax + Sampling
void benchmark_quantization_and_softmax_pipeline() {
    std::cout << "[Benchmark 5/6] Benchmarking Quantization + Softmax + Sampling..." << std::endl;

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

// Benchmark 6: Online Generation Loop Detector Latency
void benchmark_generation_loop_interception() {
    std::cout << "[Benchmark 6/8] Benchmarking Online Generation Loop Detector Latency..." << std::endl;

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

// Benchmark 7: Xenolith-Inspired Quantized Q4_0 x Q8_0 GEMV & Speculative Decode GEMM
void benchmark_quantized_q4_q8_pipeline() {
    std::cout << "[Benchmark 7/8] Benchmarking Direct Quantized Q4_0 x Q8_0 GEMV & Multi-Token GEMM..." << std::endl;

    const size_t hidden_dim = 2048;
    const size_t ffn_dim = 5632;
    const size_t num_blocks = hidden_dim / 32;

    std::mt19937 rng(777);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> w_f32(ffn_dim * hidden_dim);
    for (auto& x : w_f32) x = dist(rng) * 0.05f;

    std::vector<FastQuant::BlockQ4_0> w_q4(ffn_dim * num_blocks);
    for (size_t i = 0; i < ffn_dim; ++i) {
        FastQuant::quantize_q4_0(w_f32.data() + i * hidden_dim, w_q4.data() + i * num_blocks, num_blocks);
    }

    std::vector<float> act_f32(hidden_dim);
    for (auto& x : act_f32) x = dist(rng);

    std::vector<FastQuant::BlockQ8_0> act_q8(num_blocks);
    FastQuant::quantize_q8_0_with_sum(act_f32.data(), act_q8.data(), num_blocks);

    std::vector<float> out(ffn_dim);

    const int iterations = 500;
    auto t0 = Clock::now();

    for (int it = 0; it < iterations; ++it) {
        FastMatMul::gemv_q4_0_q8_0(w_q4.data(), act_q8.data(), out.data(), ffn_dim, hidden_dim);
    }

    auto t1 = Clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double per_step_us = (total_ms / iterations) * 1000.0;

    // Speculative 4-token batched GEMM
    const size_t T = 4;
    std::vector<FastQuant::BlockQ8_0> batch_q8(T * num_blocks);
    for (size_t t = 0; t < T; ++t) {
        for (size_t b = 0; b < num_blocks; ++b) batch_q8[t * num_blocks + b] = act_q8[b];
    }
    std::vector<float> batch_out(T * ffn_dim);

    auto t2 = Clock::now();
    for (int it = 0; it < iterations; ++it) {
        FastMatMul::gemm_q4_0_q8_0(w_q4.data(), batch_q8.data(), batch_out.data(), ffn_dim, T, hidden_dim);
    }
    auto t3 = Clock::now();
    double total_ms_batch = std::chrono::duration<double, std::milli>(t3 - t2).count();
    double per_batch_us = (total_ms_batch / iterations) * 1000.0;

    // Benchmark Q4_1 and Q5_0
    std::vector<FastQuant::BlockQ4_1> w_q41(ffn_dim * num_blocks);
    std::vector<FastQuant::BlockQ8_1> act_q81(num_blocks);
    for (size_t i = 0; i < ffn_dim; ++i) {
        FastQuant::quantize_q4_1(w_f32.data() + i * hidden_dim, w_q41.data() + i * num_blocks, num_blocks);
    }
    FastQuant::quantize_q8_1(act_f32.data(), act_q81.data(), num_blocks);

    auto t4 = Clock::now();
    for (int it = 0; it < iterations; ++it) {
        FastMatMul::gemv_q4_1_q8_1(w_q41.data(), act_q81.data(), out.data(), ffn_dim, hidden_dim);
    }
    auto t5 = Clock::now();
    double total_ms_q41 = std::chrono::duration<double, std::milli>(t5 - t4).count();
    double per_step_us_q41 = (total_ms_q41 / iterations) * 1000.0;

    std::vector<FastQuant::BlockQ5_0> w_q50(ffn_dim * num_blocks);
    for (size_t i = 0; i < ffn_dim; ++i) {
        FastQuant::quantize_q5_0(w_f32.data() + i * hidden_dim, w_q50.data() + i * num_blocks, num_blocks);
    }
    auto t6 = Clock::now();
    for (int it = 0; it < iterations; ++it) {
        FastMatMul::gemv_q5_0_q8_0(w_q50.data(), act_q8.data(), out.data(), ffn_dim, hidden_dim);
    }
    auto t7 = Clock::now();
    double total_ms_q50 = std::chrono::duration<double, std::milli>(t7 - t6).count();
    double per_step_us_q50 = (total_ms_q50 / iterations) * 1000.0;

    std::cout << "  FFN Layer Shape: [" << ffn_dim << " x " << hidden_dim << "]" << std::endl;
    std::cout << "  Q4_0 x Q8_0 Single-Token GEMV: " << std::fixed << std::setprecision(2) << per_step_us << " us / step" << std::endl;
    std::cout << "  Q4_1 x Q8_1 Single-Token GEMV: " << per_step_us_q41 << " us / step" << std::endl;
    std::cout << "  Q5_0 x Q8_0 Single-Token GEMV: " << per_step_us_q50 << " us / step" << std::endl;
    std::cout << "  Q4_0 x Q8_0 4-Token Batched GEMM: " << per_batch_us << " us / step (" << (per_batch_us / T) << " us / token)" << std::endl;
    assert(!std::isnan(out[0]) && !std::isinf(out[0]));
    std::cout << "  Passed." << std::endl;
}

// Benchmark 8: Copy-on-Write (COW) Split-KV Attention, Zero-Allocation Prefill & Split-K Long Context
void benchmark_cow_split_kv_and_prefill() {
    std::cout << "[Benchmark 8/8] Benchmarking COW Split-KV Attention, Zero-Alloc Prefill & Split-K Long Context..." << std::endl;

    const size_t num_heads = 16;
    const size_t num_kv_heads = 4;
    const size_t head_dim = 128;
    const size_t hidden_dim = num_heads * head_dim; // 2048
    const size_t prefix_len = 1024;
    const size_t tail_len = 64;
    const size_t prefill_len = 256;
    const size_t long_seq_len = 4096;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::mt19937 rng(999);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> q(hidden_dim);
    for (auto& x : q) x = dist(rng);

    std::vector<float> prefix_k(prefix_len * num_kv_heads * head_dim);
    std::vector<float> prefix_v(prefix_len * num_kv_heads * head_dim);
    for (auto& x : prefix_k) x = dist(rng) * 0.1f;
    for (auto& x : prefix_v) x = dist(rng) * 0.1f;

    std::vector<float> tail_k(tail_len * num_kv_heads * head_dim);
    std::vector<float> tail_v(tail_len * num_kv_heads * head_dim);
    for (auto& x : tail_k) x = dist(rng) * 0.1f;
    for (auto& x : tail_v) x = dist(rng) * 0.1f;

    std::vector<float> attn_out(hidden_dim);

    const int iterations = 200;
    auto t0 = Clock::now();

    for (int it = 0; it < iterations; ++it) {
        FastAttention::cow_multi_head_scaled_dot_product_decode(
            q.data(), prefix_k.data(), prefix_v.data(), prefix_len,
            tail_k.data(), tail_v.data(), tail_len, attn_out.data(),
            num_heads, num_kv_heads, head_dim, scale
        );
    }

    auto t1 = Clock::now();
    double total_ms_cow = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double per_step_us_cow = (total_ms_cow / iterations) * 1000.0;

    // Zero-allocation prefill benchmark
    std::vector<float> pref_q(prefill_len * num_heads * head_dim);
    std::vector<float> pref_k(prefill_len * num_kv_heads * head_dim);
    std::vector<float> pref_v(prefill_len * num_kv_heads * head_dim);
    std::vector<float> pref_out(prefill_len * num_heads * head_dim);
    for (auto& x : pref_q) x = dist(rng);
    for (auto& x : pref_k) x = dist(rng) * 0.1f;
    for (auto& x : pref_v) x = dist(rng) * 0.1f;

    auto t2 = Clock::now();
    constexpr int PREF_ITERS = 20;
    for (int it = 0; it < PREF_ITERS; ++it) {
        FastAttention::flash_attention_prefill_mha(
            pref_q.data(), pref_k.data(), pref_v.data(), pref_out.data(),
            prefill_len, num_heads, num_kv_heads, head_dim, scale, true
        );
    }
    auto t3 = Clock::now();
    double total_ms_pref = std::chrono::duration<double, std::milli>(t3 - t2).count();
    double per_step_ms_pref = total_ms_pref / PREF_ITERS;

    // Split-K Long Context benchmark (4096 tokens)
    std::vector<float> long_k(long_seq_len * head_dim);
    std::vector<float> long_v(long_seq_len * head_dim);
    std::vector<float> long_out(head_dim);
    for (auto& x : long_k) x = dist(rng) * 0.1f;
    for (auto& x : long_v) x = dist(rng) * 0.1f;

    auto t4 = Clock::now();
    constexpr int LONG_ITERS = 50;
    for (int it = 0; it < LONG_ITERS; ++it) {
        FastAttention::split_k_scaled_dot_product_decode(
            q.data(), long_k.data(), long_v.data(), long_out.data(),
            long_seq_len, head_dim, scale, 256
        );
    }
    auto t5 = Clock::now();
    double total_ms_long = std::chrono::duration<double, std::milli>(t5 - t4).count();
    double per_step_us_long = (total_ms_long / LONG_ITERS) * 1000.0;

    std::cout << "  Split-KV Cache Context: " << prefix_len << " prefix + " << tail_len << " tail = " << (prefix_len + tail_len) << " tokens" << std::endl;
    std::cout << "  COW GQA Decode Latency: " << std::fixed << std::setprecision(2) << per_step_us_cow << " us / step" << std::endl;
    std::cout << "  Zero-Alloc FlashPrefill (" << prefill_len << " tokens): " << per_step_ms_pref << " ms / prefill" << std::endl;
    std::cout << "  Split-K Long Context (" << long_seq_len << " tokens): " << per_step_us_long << " us / head" << std::endl;
    assert(!std::isnan(attn_out[0]) && !std::isinf(attn_out[0]));
    std::cout << "  Passed." << std::endl;
}

void benchmark_online_flash_decode_and_adaptive_moe() {
    std::cout << "[Benchmark 9/9] Benchmarking 1-Pass Tiled Online Flash Decode & Adaptive MoE Pruning..." << std::endl;
    using Clock = std::chrono::high_resolution_clock;
    std::mt19937 rng(4242);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const size_t head_dim = 128;
    const size_t seq_len = 1024;
    std::vector<float> q(head_dim);
    std::vector<float> k(seq_len * head_dim);
    std::vector<float> v(seq_len * head_dim);
    std::vector<float> out_flash(head_dim);
    for (auto& x : q) x = dist(rng);
    for (auto& x : k) x = dist(rng) * 0.1f;
    for (auto& x : v) x = dist(rng) * 0.1f;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    constexpr int ITERS = 200;
    auto t0 = Clock::now();
    for (int it = 0; it < ITERS; ++it) {
        FastAttention::flash_decode_scaled_dot_product(
            q.data(), k.data(), v.data(), out_flash.data(), seq_len, head_dim, scale
        );
    }
    auto t1 = Clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double us_per_step = (total_ms / ITERS) * 1000.0;

    // Adaptive MoE Routing benchmark
    const size_t num_experts = 64;
    const size_t top_k = 8;
    std::vector<float> logits(num_experts);
    for (auto& l : logits) l = dist(rng) * 2.0f;
    logits[3] = 8.5f; // dominant expert
    logits[12] = 7.1f; // second expert
    std::vector<int32_t> indices(top_k);
    std::vector<float> weights(top_k);

    auto t2 = Clock::now();
    size_t active_experts = 0;
    for (int it = 0; it < 10000; ++it) {
        active_experts = FastMoE::route_topk_with_threshold(
            logits.data(), num_experts, top_k, indices.data(), weights.data(), 0.01f, 0.98f
        );
    }
    auto t3 = Clock::now();
    double moe_total_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    double moe_ns_per_route = (moe_total_ms / 10000.0) * 1000000.0;

    std::cout << "  1-Pass Tiled Online Flash Decode (1024 tokens): " << std::fixed << std::setprecision(2) << us_per_step << " us / head" << std::endl;
    std::cout << "  Adaptive MoE Router Latency: " << std::fixed << std::setprecision(2) << moe_ns_per_route << " ns / token (" << active_experts << " active experts chosen)" << std::endl;
    assert(!std::isnan(out_flash[0]));
    std::cout << "  Passed." << std::endl;
}

void benchmark_paged_attention_and_speculative_verification() {
    std::cout << "[Benchmark 10/10] Benchmarking PagedAttention, Speculative Verification & In-Flight Batching..." << std::endl;
    using Clock = std::chrono::high_resolution_clock;
    std::mt19937 rng(9999);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const size_t head_dim = 128;
    const size_t seq_len = 1024;
    const size_t num_kv_heads = 4;
    const size_t block_size = 16;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    PagedKVCacheConfig kv_cfg;
    kv_cfg.block_size = block_size;
    kv_cfg.num_kv_heads = num_kv_heads;
    kv_cfg.head_dim = head_dim;
    kv_cfg.initial_pool_blocks = 128;

    FastPagedKVCache cache(kv_cfg);
    uint64_t seq_id = 42;
    cache.create_sequence(seq_id);

    const size_t kv_stride = num_kv_heads * head_dim;
    for (size_t t = 0; t < seq_len; ++t) {
        std::vector<float> k_tok(kv_stride);
        std::vector<float> v_tok(kv_stride);
        for (size_t i = 0; i < kv_stride; ++i) {
            k_tok[i] = dist(rng) * 0.1f;
            v_tok[i] = dist(rng) * 0.1f;
        }
        cache.append_token(seq_id, k_tok.data(), v_tok.data());
    }

    std::vector<float> q(head_dim);
    for (auto& x : q) x = dist(rng);
    std::vector<float> paged_out(head_dim);

    // 1. Benchmark PagedAttention decoding over 1024 tokens
    constexpr int ITERS = 200;
    auto t0 = Clock::now();
    for (int it = 0; it < ITERS; ++it) {
        FastAttention::paged_scaled_dot_product_decode(
            q.data(), cache, seq_id, 0, paged_out.data(), scale
        );
    }
    auto t1 = Clock::now();
    double paged_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double paged_us_per_step = (paged_ms / ITERS) * 1000.0;

    // 2. Benchmark FastSpeculativeVerifier (4 drafts, 32k vocabulary)
    const size_t vocab_size = 32000;
    const size_t num_drafts = 4;
    std::vector<float> target_logits((num_drafts + 1) * vocab_size);
    for (auto& l : target_logits) l = dist(rng);
    for (size_t d = 0; d <= num_drafts; ++d) {
        target_logits[d * vocab_size + 100 + d] = 12.0f; // distinct peaks
    }
    int32_t draft_tokens[4] = {100, 101, 102, 103};

    auto t2 = Clock::now();
    constexpr int SPEC_ITERS = 500;
    for (int it = 0; it < SPEC_ITERS; ++it) {
        auto res = FastSpeculativeVerifier::verify_greedy(
            draft_tokens, num_drafts, target_logits.data(), vocab_size
        );
        (void)res;
    }
    auto t3 = Clock::now();
    double spec_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    double spec_us_per_round = (spec_ms / SPEC_ITERS) * 1000.0;

    // 3. InFlightBatcher Continuous Scheduling iteration benchmark
    strata::runtime::InFlightBatcherConfig b_cfg;
    b_cfg.max_batch_size = 16;
    b_cfg.prefill_chunk_size = 128;
    auto shared_cache = std::make_shared<FastPagedKVCache>(kv_cfg);
    strata::runtime::InFlightBatcher batcher(shared_cache, b_cfg);

    for (uint64_t r = 1; r <= 8; ++r) {
        strata::runtime::GenerationRequest req;
        req.request_id = r;
        req.prompt_tokens.resize(64, static_cast<int32_t>(r));
        req.max_new_tokens = 32;
        batcher.submit_request(req);
    }

    auto t4 = Clock::now();
    for (int step_idx = 0; step_idx < 10; ++step_idx) {
        batcher.step();
    }
    auto t5 = Clock::now();
    double batcher_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();
    double batcher_us_per_iter = (batcher_ms / 10.0) * 1000.0;

    std::cout << "  PagedAttention Decode (1024 tokens, 64 blocks): " << std::fixed << std::setprecision(2) << paged_us_per_step << " us / head" << std::endl;
    std::cout << "  FastSpeculativeVerifier (K=4, 32k vocab): " << std::fixed << std::setprecision(2) << spec_us_per_round << " us / round" << std::endl;
    std::cout << "  In-Flight Batcher Iteration Latency (8 streams): " << std::fixed << std::setprecision(2) << batcher_us_per_iter << " us / iter" << std::endl;

    assert(!std::isnan(paged_out[0]));
    std::cout << "  Passed." << std::endl;
}

// Benchmark 11: MiMo-V2.6-Flash Asymmetric SWA & Mamba SSM Selective Scan Pipeline
void benchmark_mimo_flash_and_mamba_ssm() {
    std::cout << "[Benchmark 11/11] MiMo-V2.6 Flash Architecture & Mamba SSM Engine..." << std::endl;

    // 1. Benchmark MiMo Asymmetric Head Dimensions (QK=192, V=128) Decode over 128 SWA window
    MiMoFlashConfig mimo_cfg;
    FastMiMoAsymmetricKVCache mimo_cache(mimo_cfg, /*is_swa=*/true);

    int64_t k_stride = mimo_cfg.swa_n_head_kv * mimo_cfg.head_dim_qk;
    int64_t v_stride = mimo_cfg.swa_n_head_kv * mimo_cfg.head_dim_v;
    std::vector<float> k_tok(k_stride, 0.05f);
    std::vector<float> v_tok(v_stride, 0.1f);

    for (int64_t t = 0; t < 128; ++t) {
        mimo_cache.append(t, k_tok.data(), v_tok.data());
    }

    std::vector<float> q_step(mimo_cfg.n_head * mimo_cfg.head_dim_qk, 0.05f);
    std::vector<float> mimo_out(mimo_cfg.n_head * mimo_cfg.head_dim_v, 0.0f);

    const int mimo_iters = 100;
    auto t0 = Clock::now();
    for (int i = 0; i < mimo_iters; ++i) {
        FastMiMoAttention::decode_step(q_step.data(), mimo_cache, 127, 1 /*SWA layer*/, mimo_cfg, mimo_out.data());
    }
    auto t1 = Clock::now();
    double mimo_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double mimo_us_per_step = (mimo_ms / mimo_iters) * 1000.0;

    // 2. Benchmark MiMo 256-Expert Top-8 Routing
    std::vector<float> router_logits(256, 0.1f);
    router_logits[15] = 5.0f;
    router_logits[30] = 6.0f;
    int32_t selected[8];
    float weights[8];

    const int route_iters = 1000;
    auto t2 = Clock::now();
    for (int i = 0; i < route_iters; ++i) {
        FastMiMoMoERouter::route_top8(router_logits.data(), selected, weights, true);
    }
    auto t3 = Clock::now();
    double route_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    double route_us_per_call = (route_ms / route_iters) * 1000.0;

    // 3. Benchmark Mamba Selective Scan (d_model=2048, d_inner=4096, d_state=16)
    MambaConfig mamba_cfg;
    mamba_cfg.d_model = 2048;
    mamba_cfg.expand = 2; // d_inner = 4096
    mamba_cfg.d_state = 16;
    FastMambaSSMState mamba_state(mamba_cfg);

    int64_t d_in = mamba_cfg.d_inner();
    int64_t N = mamba_cfg.d_state;
    std::vector<float> u(d_in, 0.1f);
    std::vector<float> delta(d_in, 0.05f);
    std::vector<float> dt_bias(d_in, 0.0f);
    std::vector<float> A_log(d_in * N, -0.5f);
    std::vector<float> B(N, 0.2f);
    std::vector<float> C(N, 0.3f);
    std::vector<float> D(d_in, 0.1f);
    std::vector<float> mamba_out(d_in, 0.0f);

    const int mamba_iters = 100;
    auto t4 = Clock::now();
    for (int i = 0; i < mamba_iters; ++i) {
        FastMambaSelectiveScan::step(mamba_state, u.data(), delta.data(), dt_bias.data(),
                                    A_log.data(), B.data(), C.data(), D.data(), mamba_cfg, mamba_out.data());
    }
    auto t5 = Clock::now();
    double mamba_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();
    double mamba_us_per_step = (mamba_ms / mamba_iters) * 1000.0;

    std::cout << "  MiMo-V2.6 Flash Decode Attention (QK=192, V=128, 64 heads, 128 window): "
              << std::fixed << std::setprecision(2) << mimo_us_per_step << " us / step" << std::endl;
    std::cout << "  MiMo-V2.6 Top-8 MoE Router (256 experts): "
              << std::fixed << std::setprecision(2) << route_us_per_call << " us / call" << std::endl;
    std::cout << "  Mamba Selective Scan Step (d_inner=4096, N=16, constant O(1) state): "
              << std::fixed << std::setprecision(2) << mamba_us_per_step << " us / step" << std::endl;

    assert(!std::isnan(mimo_out[0]));
    assert(!std::isnan(mamba_out[0]));
    std::cout << "  Passed." << std::endl;
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "   RUNNING STRATA UNIFIED FAST KERNEL & BENCHMARK SUITE          " << std::endl;
    std::cout << "=================================================================" << std::endl;

    benchmark_full_transformer_layer_kernels();
    benchmark_fast_attention_pipeline();
    benchmark_fast_moe_pipeline();
    benchmark_full_moe_transformer_layer_kernels();
    benchmark_quantization_and_softmax_pipeline();
    benchmark_generation_loop_interception();
    benchmark_quantized_q4_q8_pipeline();
    benchmark_cow_split_kv_and_prefill();
    benchmark_online_flash_decode_and_adaptive_moe();
    benchmark_paged_attention_and_speculative_verification();
    benchmark_mimo_flash_and_mamba_ssm();

    std::cout << "=================================================================" << std::endl;
    std::cout << "   ALL UNIFIED FAST KERNEL BENCHMARKS COMPLETED SUCCESSFULLY     " << std::endl;
    std::cout << "=================================================================" << std::endl;
    return 0;
}
