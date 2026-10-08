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
    std::cout << "[Benchmark 6/6] Benchmarking Online Generation Loop Detector Latency..." << std::endl;

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
    benchmark_fast_attention_pipeline();
    benchmark_fast_moe_pipeline();
    benchmark_full_moe_transformer_layer_kernels();
    benchmark_quantization_and_softmax_pipeline();
    benchmark_generation_loop_interception();

    std::cout << "=================================================================" << std::endl;
    std::cout << "   ALL UNIFIED FAST KERNEL BENCHMARKS COMPLETED SUCCESSFULLY     " << std::endl;
    std::cout << "=================================================================" << std::endl;
    return 0;
}
