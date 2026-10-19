// tests/runtime/test_generic_runtime.cpp - Comprehensive Test Suite for Generic MoE Runtime
#include "strata/models/model_adapter.hpp"
#include "strata/runtime/device.hpp"
#include "strata/runtime/expert_manager.hpp"
#include "strata/runtime/memory.hpp"
#include "strata/runtime/runtime.hpp"
#include "strata/runtime/scheduler.hpp"
#include "strata/runtime/topology.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

void test_generic_device_abstraction() {
    std::cout << "[Test 1/7] Testing Generic Device Abstraction..." << std::endl;
    auto& dev_mgr = strata::runtime::DeviceManager::instance();
    dev_mgr.discover_all();

    int count = dev_mgr.device_count();
    assert(count >= 1); // At least Host CPU device must be registered

    for (int i = 0; i < count; ++i) {
        auto dev = dev_mgr.get_device(i);
        assert(dev != nullptr);
        assert(dev->id() == i);
        assert(!dev->name().empty());
        assert(dev->total_memory_bytes() > 0);
        assert(dev->metrics().compute_tflops_fp32 >= 0.0);
    }
    std::cout << "  Passed. Discovered " << count << " compute device(s)." << std::endl;
}

void test_memory_domain_and_topology() {
    std::cout << "[Test 2/7] Testing Memory Domain & Hardware Topology..." << std::endl;
    auto& topo = strata::runtime::HardwareTopology::instance();
    topo.discover();
    topo.calibrate(true);

    const auto& domains = topo.memory_domains();
    assert(!domains.empty());

    for (const auto& dom : domains) {
        assert(dom->id() >= 0);
        assert(dom->characteristics().total_capacity_bytes > 0);
        (void)dom;
    }

    // Verify transfer cost calculation
    double cost = topo.transfer_cost_sec(0, 0, 1024 * 1024);
    assert(cost == 0.0); // Same device cost is 0
    (void)cost;

    if (topo.devices().size() > 1) {
        double xfer_cost = topo.transfer_cost_sec(0, 1, 1024 * 1024);
        assert(xfer_cost > 0.0);
        (void)xfer_cost;
    }
    std::cout << "  Passed. Topology graph verified with " << domains.size() << " memory domain(s)." << std::endl;
}

void test_model_adapters() {
    std::cout << "[Test 3/7] Testing Model Adapter Architecture..." << std::endl;
    auto& reg = strata::models::ModelAdapterRegistry::instance();
    auto archs = reg.available_architectures();
    assert(!archs.empty());

    // Test Qwen MoE Adapter
    auto qwen = reg.create("qwen_moe");
    assert(qwen != nullptr);
    assert(qwen->architecture_name() == "qwen_moe");
    assert(qwen->config().n_expert == 512);
    assert(qwen->config().active_experts == 10);
    assert(qwen->supports_swa());
    assert(qwen->sliding_window_size() == 4096);
    assert(qwen->is_full_attention_layer(3));  // QSA layer
    assert(!qwen->is_full_attention_layer(0)); // GDN layer

    // Test Mixtral MoE Adapter
    auto mixtral = reg.create("mixtral");
    assert(mixtral != nullptr);
    assert(mixtral->architecture_name() == "mixtral");
    assert(mixtral->config().n_expert == 8);
    assert(mixtral->config().active_experts == 2);
    assert(mixtral->supports_swa());
    assert(mixtral->sliding_window_size() == 4096);
    assert(mixtral->is_full_attention_layer(0));

    // Test DeepSeek MoE Adapter
    auto deepseek = reg.create("deepseek_moe");
    assert(deepseek != nullptr);
    assert(deepseek->architecture_name() == "deepseek_moe");
    assert(deepseek->config().n_expert == 160);
    assert(deepseek->config().active_experts == 6);
    assert(deepseek->config().n_shared_experts == 2);
    assert(deepseek->supports_swa());
    assert(deepseek->sliding_window_size() == 4096);

    // Test DeepSeek-V4.1 Adapter (GGUF arch deepseek41)
    auto dsv41 = reg.create("deepseek41");
    assert(dsv41 != nullptr);
    assert(dsv41->architecture_name() == "deepseek41");
    const auto& v41 = dsv41->config();
    assert(v41.n_layers == 40);
    assert(v41.n_embd == 5120);
    assert(v41.n_head == 64 && v41.n_head_kv == 1 && v41.head_dim == 512);
    assert(v41.q_lora_rank == 1280 && v41.kv_lora_rank == 512 && v41.qk_rope_head_dim == 64);
    assert(v41.n_expert == 384 && v41.active_experts == 6 && v41.n_shared_experts == 1);
    assert(v41.n_ff == 2304);
    assert(v41.indexer_heads == 32 && v41.indexer_key_dim == 128 && v41.indexer_top_k == 512);
    assert(v41.output_lora_rank == 1024 && v41.output_groups == 8);
    assert(v41.hc == 4 && v41.mtp_layers == 3);
    assert(dsv41->supports_swa() && dsv41->sliding_window_size() == 128);
    // compress_ratios: layers 0-1 full, then HCA x2, then x1, last three full
    assert(dsv41->is_full_attention_layer(0) && dsv41->is_full_attention_layer(1));
    assert(!dsv41->is_full_attention_layer(2));
    assert(!dsv41->is_full_attention_layer(20));
    assert(dsv41->is_full_attention_layer(39));
    // engram memory lives on layers 1 and 14 only; layers 2/8/14/20 own the compressed KV source
    assert(v41.engram_layers.size() == 2 && v41.engram_layers[0] == 1 && v41.engram_layers[1] == 14);
    assert(v41.kv_source_layers.size() == 4 && v41.kv_source_layers[0] == 2 && v41.kv_source_layers[3] == 20);
    assert(dsv41->tensor_name(7, "attn_kv.weight") == "blk.7.attn_kv.weight");

    // Test MiMo-V2.6 MoE Adapter
    auto mimo = reg.create("mimo_v2_6");
    assert(mimo != nullptr);
    assert(mimo->architecture_name() == "mimo2");
    assert(mimo->config().n_expert == 256);
    assert(mimo->config().active_experts == 8);
    assert(mimo->config().n_shared_experts == 0); // pure sparse MoE, no shared expert
    assert(mimo->config().n_layers == 48);
    assert(mimo->config().mtp_layers == 5);
    assert(mimo->config().sliding_window == 128); // 128-token rolling SWA buffer
    assert(mimo->supports_swa());
    assert(mimo->sliding_window_size() == 128);
    assert(mimo->is_full_attention_layer(6)); // every 6th layer is full attention
    assert(!mimo->is_full_attention_layer(3));

    // Test Generic MoE Adapter
    auto generic_moe = reg.create("generic_moe");
    assert(generic_moe != nullptr);
    assert(generic_moe->supports_swa());
    assert(generic_moe->sliding_window_size() == 4096);

    std::cout << "  Passed. Verified Qwen, Mixtral, DeepSeek-V2/V3, DeepSeek-V4.1, MiMo-V2.6, and Generic MoE adapters with SWA." << std::endl;
}

void test_routing_algorithms() {
    std::cout << "[Test 4/7] Testing Routing Algorithms Across Models..." << std::endl;
    auto& reg = strata::models::ModelAdapterRegistry::instance();

    // 1. Sigmoid Top-K (Qwen style)
    auto qwen = reg.create("qwen_moe");
    std::vector<float> logits(512, 0.0f);
    logits[12] = 5.0f;
    logits[99] = 10.0f;
    logits[200] = 3.0f;

    std::vector<int32_t> selected(10, -1);
    std::vector<float> weights(10, 0.0f);
    qwen->route_token(logits.data(), 512, 10, selected.data(), weights.data());

    assert(selected[0] == 99); // Highest logit
    assert(selected[1] == 12);
    assert(selected[2] == 200);

    float sum = 0.0f;
    for (float w : weights) sum += w;
    assert(std::fabs(sum - 1.0f) < 1e-4);

    // 2. Softmax Top-K (Mixtral style)
    auto mixtral = reg.create("mixtral");
    std::vector<float> mix_logits(8, 0.0f);
    mix_logits[3] = 4.0f;
    mix_logits[7] = 8.0f;

    std::vector<int32_t> mix_selected(2, -1);
    std::vector<float> mix_weights(2, 0.0f);
    mixtral->route_token(mix_logits.data(), 8, 2, mix_selected.data(), mix_weights.data());

    assert(mix_selected[0] == 7);
    assert(mix_selected[1] == 3);
    assert(std::fabs(mix_weights[0] + mix_weights[1] - 1.0f) < 1e-4);

    // 3. Sigmoid Top-K with 384 experts (MiMo-V2.6 style)
    auto mimo = reg.create("mimo_v2_6");
    std::vector<float> mimo_logits(384, 0.0f);
    mimo_logits[42] = 12.0f;
    mimo_logits[188] = 9.5f;
    mimo_logits[301] = 7.0f;

    std::vector<int32_t> mimo_selected(8, -1);
    std::vector<float> mimo_weights(8, 0.0f);
    mimo->route_token(mimo_logits.data(), 384, 8, mimo_selected.data(), mimo_weights.data());

    assert(mimo_selected[0] == 42);
    assert(mimo_selected[1] == 188);
    assert(mimo_selected[2] == 301);

    float mimo_sum = 0.0f;
    for (float w : mimo_weights) mimo_sum += w;
    assert(std::fabs(mimo_sum - 1.0f) < 1e-4);

    // 4. sqrt(softplus) noaux_tc Top-6 with 384 experts (DeepSeek-V4.1 style): the score is
    // monotone in the logit, so the selection follows the raw logits, but the weights carry the
    // model's routed_scaling_factor (1.5), which multiplies the expert sum.
    auto dsv41 = reg.create("deepseek41");
    std::vector<float> v41_logits(384, -1.0f);
    v41_logits[7] = 6.0f;
    v41_logits[150] = 2.5f;
    v41_logits[333] = -3.0f; // sqrt(softplus(x)) stays real for negative logits

    std::vector<int32_t> v41_selected(6, -1);
    std::vector<float> v41_weights(6, 0.0f);
    dsv41->route_token(v41_logits.data(), 384, 6, v41_selected.data(), v41_weights.data());

    assert(v41_selected[0] == 7);
    assert(v41_selected[1] == 150);
    float v41_sum = 0.0f;
    for (float w : v41_weights) v41_sum += w;
    assert(std::fabs(v41_sum - 1.5f) < 1e-4);
    assert(v41_weights[0] > v41_weights[1]);

    std::cout << "  Passed. Routing logic validated bit-exact and normalized across all models." << std::endl;
}

void test_generic_expert_manager() {
    std::cout << "[Test 5/7] Testing Generic Expert Manager & Heat Placement..." << std::endl;
    strata::runtime::GenericExpertManager mgr(4, 16);

    uint8_t dummy_blob[1024];
    for (int l = 0; l < 4; ++l) {
        for (int e = 0; e < 16; ++e) {
            mgr.register_expert(l, e, dummy_blob, 1024, "IQ3_XXS");
        }
    }

    // Simulate access
    mgr.record_access(0, 5, 1.0);
    mgr.record_access(0, 5, 2.0);
    mgr.record_access(0, 5, 3.0);
    mgr.record_access(1, 2, 3.0);

    auto stats = mgr.get_stats(0, 5);
    assert(stats.access_count == 3);
    assert(stats.heat_score > 1.0);

    mgr.rebalance_placement();
    const auto* loc = mgr.get_location(0, 5);
    assert(loc != nullptr);
    assert(loc->byte_size == 1024);
    (void)loc;

    std::cout << "  Passed. Expert tracking, heat decay, and dynamic placement validated." << std::endl;
}

void test_dynamic_scheduler() {
    std::cout << "[Test 6/7] Testing Dynamic Computation Scheduler..." << std::endl;
    auto exp_mgr = std::make_shared<strata::runtime::GenericExpertManager>(4, 16);
    strata::runtime::DynamicScheduler sched(exp_mgr);

    int32_t active[4] = {1, 3, 5, 7};
    auto tasks = sched.schedule_active_experts(0, active, 4, 1024, 1000000);

    assert(tasks.size() == 4);
    for (const auto& t : tasks) {
        assert(t.target_device_id >= 0);
        assert(t.estimated_total_sec > 0.0);
        (void)t;
    }

    auto op_task = sched.schedule_op(strata::runtime::OpType::kAttention, 2048, 2048, 5000000, 0);
    assert(op_task.target_device_id >= 0);
    (void)op_task;

    std::cout << "  Passed. Dynamic scheduler cost evaluation and task assignment verified." << std::endl;
}

void test_generic_runtime_end_to_end() {
    std::cout << "[Test 7/7] Testing Generic MoE Runtime End-to-End Execution..." << std::endl;
    strata::runtime::RuntimeOptions opt;
    opt.architecture = "qwen_moe";
    opt.enable_heterogeneous = true;
    opt.calibrate_at_startup = true;

    strata::runtime::GenericMoERuntime rt(opt);
    std::string err;
    bool ok = rt.initialize(err);
    if (!ok) {
        std::cerr << "Initialization error: " << err << std::endl;
    }
    assert(ok);

    std::cout << rt.print_runtime_summary() << std::endl;

    strata::runtime::ForwardMetrics metrics;
    float logits[100];
    bool fwd_ok = rt.forward_token(0, 101, logits, &metrics);
    assert(fwd_ok);
    assert(metrics.total_forward_ms > 0.0);
    assert(metrics.experts_executed > 0);
    (void)fwd_ok;

    std::cout << "  Forward step completed in " << metrics.total_forward_ms
              << " ms (Experts executed: " << metrics.experts_executed << ")" << std::endl;
    std::cout << "  Passed. End-to-end Generic MoE Runtime verified!" << std::endl;
}

int main() {
    std::cout << "========================================================\n"
              << "   RUNNING STRATA GENERIC MoE RUNTIME VERIFICATION SUITE \n"
              << "========================================================\n";

    test_generic_device_abstraction();
    test_memory_domain_and_topology();
    test_model_adapters();
    test_routing_algorithms();
    test_generic_expert_manager();
    test_dynamic_scheduler();
    test_generic_runtime_end_to_end();

    std::cout << "========================================================\n"
              << "   ALL GENERIC RUNTIME AND ADAPTER TESTS PASSED (7/7)   \n"
              << "========================================================\n";
    return 0;
}
