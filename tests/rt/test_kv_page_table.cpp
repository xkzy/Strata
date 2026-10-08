// tests/rt/test_kv_page_table.cpp - test suite for virtual KV page table and prefetch predictor
#include "strata/rt/kv_page_table.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::cerr << "CHECK failed at " << __FILE__ << ":" << __LINE__ << ": " << #cond << "\n"; \
        std::abort(); \
    } \
} while (0)

using namespace strata::rt;

static void test_key_invariance_and_rope() {
    std::cout << "[1] KVBlockKey RoPE position and content sensitivity\n";

    KVBlockKey k1{100, 1, 0x12345678, 0, 4096, 0, 32, KVFormat::kFP16, 1};
    KVBlockKey k2{100, 1, 0x12345678, 4096, 8192, 0, 32, KVFormat::kFP16, 1}; // Different start/end position
    KVBlockKey k3{100, 1, 0x87654321, 0, 4096, 0, 32, KVFormat::kFP16, 1}; // Different token content
    KVBlockKey k4{100, 1, 0x12345678, 0, 4096, 0, 32, KVFormat::kFP8_E4M3, 1}; // Different KV format
    KVBlockKey k1_copy = k1;

    CHECK(k1 == k1_copy);
    CHECK(!(k1 == k2));
    CHECK(!(k1 == k3));
    CHECK(!(k1 == k4));

    KVBlockKeyHasher hasher;
    CHECK(hasher(k1) == hasher(k1_copy));
    CHECK(hasher(k1) != hasher(k2));
    CHECK(hasher(k1) != hasher(k3));
    CHECK(hasher(k1) != hasher(k4));
}

static void test_hierarchical_tier_transitions() {
    std::cout << "[2] Hierarchical tier registration, promotion and demotion\n";

    KVTierBudgets budgets;
    VirtualKVPageTable table(budgets);

    auto p1 = std::make_shared<KVPageDescriptor>();
    p1->page_id = 101;
    p1->session_id = 1;
    p1->key = KVBlockKey{100, 1, 0xAAA, 0, 4096, 0, 32, KVFormat::kFP16, 1};
    p1->residency = KVResidency::kRecompute;
    p1->byte_size = 8 * 1024 * 1024; // 8 MB
    p1->token_count = 4096;
    p1->prefill_cost_ms = 120.0f;

    bool ok = table.register_page(p1);
    CHECK(ok);

    auto stats = table.stats();
    CHECK(stats.total_pages == 1);
    CHECK(stats.tier_pages[static_cast<size_t>(KVResidency::kRecompute)] == 1);

    // Promote to Host RAM
    table.update_residency(101, KVResidency::kHost_RAM, (void*)0x1000, -1);
    stats = table.stats();
    CHECK(stats.promotions == 1);
    CHECK(stats.tier_pages[static_cast<size_t>(KVResidency::kHost_RAM)] == 1);
    CHECK(stats.tier_pages[static_cast<size_t>(KVResidency::kRecompute)] == 0);

    // Promote to GPU VRAM
    table.update_residency(101, KVResidency::kGPU_VRAM, (void*)0x1000, 4);
    stats = table.stats();
    CHECK(stats.promotions == 2);
    CHECK(stats.tier_pages[static_cast<size_t>(KVResidency::kGPU_VRAM)] == 1);
    CHECK(p1->vram_slot == 4);

    // Lookup
    auto found = table.lookup(p1->key);
    CHECK(found != nullptr);
    CHECK(found->page_id == 101);
    stats = table.stats();
    CHECK(stats.lookups == 1);
    CHECK(stats.hits[static_cast<size_t>(KVResidency::kGPU_VRAM)] == 1);
}

static void test_cost_aware_eviction() {
    std::cout << "[3] Cost-aware eviction prioritizes high-prefill pages\n";

    KVTierBudgets budgets;
    VirtualKVPageTable table(budgets);

    // Cheap metadata page (10ms prefill cost)
    auto cheap = std::make_shared<KVPageDescriptor>();
    cheap->page_id = 201;
    cheap->key = KVBlockKey{100, 1, 0x111, 0, 1024, 0, 32, KVFormat::kFP16, 1};
    cheap->residency = KVResidency::kHost_RAM;
    cheap->byte_size = 2 * 1024 * 1024;
    cheap->prefill_cost_ms = 10.0f;
    cheap->access_frequency = 1;
    cheap->last_access_tick = 100;

    // Expensive context page (500ms prefill cost)
    auto expensive = std::make_shared<KVPageDescriptor>();
    expensive->page_id = 202;
    expensive->key = KVBlockKey{100, 1, 0x222, 1024, 8192, 0, 32, KVFormat::kFP16, 1};
    expensive->residency = KVResidency::kHost_RAM;
    expensive->byte_size = 8 * 1024 * 1024;
    expensive->prefill_cost_ms = 500.0f;
    expensive->access_frequency = 5;
    expensive->last_access_tick = 100;

    table.register_page(cheap);
    table.register_page(expensive);

    // Evict 2MB from Host RAM at tick 100
    size_t freed = table.evict_tier(KVResidency::kHost_RAM, 2 * 1024 * 1024, 100);
    CHECK(freed >= 2 * 1024 * 1024);

    // Cheap page should be demoted first to Compressed RAM
    CHECK(cheap->residency == KVResidency::kCompressed_RAM);
    // Expensive page should remain in Host RAM
    CHECK(expensive->residency == KVResidency::kHost_RAM);
}

static void test_pinned_pages() {
    std::cout << "[4] Pinned pages are never evicted\n";

    KVTierBudgets budgets;
    VirtualKVPageTable table(budgets);

    auto pinned_page = std::make_shared<KVPageDescriptor>();
    pinned_page->page_id = 301;
    pinned_page->key = KVBlockKey{100, 1, 0x333, 0, 2048, 0, 32, KVFormat::kFP16, 1};
    pinned_page->residency = KVResidency::kGPU_VRAM;
    pinned_page->byte_size = 4 * 1024 * 1024;
    pinned_page->pinned = true;

    table.register_page(pinned_page);

    size_t freed = table.evict_tier(KVResidency::kGPU_VRAM, 10 * 1024 * 1024, 500);
    CHECK(freed == 0);
    CHECK(pinned_page->residency == KVResidency::kGPU_VRAM);
}

static void test_working_set_estimation() {
    std::cout << "[5] Working-set estimation and recency windowing\n";

    KVTierBudgets budgets;
    VirtualKVPageTable table(budgets);

    auto p1 = std::make_shared<KVPageDescriptor>();
    p1->page_id = 401;
    p1->key = KVBlockKey{100, 1, 0x401, 0, 1024, 0, 32, KVFormat::kFP16, 1};
    p1->residency = KVResidency::kHost_RAM;
    p1->byte_size = 2 * 1024 * 1024;
    p1->prefill_cost_ms = 50.0f;
    p1->last_access_tick = 950;

    auto p2 = std::make_shared<KVPageDescriptor>();
    p2->page_id = 402;
    p2->key = KVBlockKey{100, 1, 0x402, 1024, 2048, 0, 32, KVFormat::kFP16, 1};
    p2->residency = KVResidency::kHost_RAM;
    p2->byte_size = 2 * 1024 * 1024;
    p2->prefill_cost_ms = 50.0f;
    p2->last_access_tick = 200; // Old access tick

    table.register_page(p1);
    table.register_page(p2);

    // Window of 100 ticks at current_tick = 1000: includes p1 (delta 50), excludes p2 (delta 800)
    auto ws = table.estimate_working_set(1000, 100);
    CHECK(ws.size() == 1);
    CHECK(ws[0] == 401);
}

static void test_prefetch_predictor() {
    std::cout << "[6] Context prefetch predictor (Graph AST + Sequential)\n";

    KVTierBudgets budgets;
    auto table = std::make_shared<VirtualKVPageTable>(budgets);

    // Active page: Function A
    auto fn_a = std::make_shared<KVPageDescriptor>();
    fn_a->page_id = 501;
    fn_a->key = KVBlockKey{100, 1, 0x501, 0, 1024, 0, 32, KVFormat::kFP16, 1};
    fn_a->residency = KVResidency::kGPU_VRAM;
    fn_a->dependent_pages = {510, 520}; // Callee functions
    table->register_page(fn_a);

    // Dependent callee 1 (in NVMe)
    auto callee1 = std::make_shared<KVPageDescriptor>();
    callee1->page_id = 510;
    callee1->key = KVBlockKey{100, 1, 0x510, 0, 1024, 0, 32, KVFormat::kFP16, 1};
    callee1->residency = KVResidency::kNVMe_Disk;
    table->register_page(callee1);

    // Sequential page 502 (in Recompute)
    auto seq_page = std::make_shared<KVPageDescriptor>();
    seq_page->page_id = 502;
    seq_page->key = KVBlockKey{100, 1, 0x502, 0, 1024, 0, 32, KVFormat::kFP16, 1};
    seq_page->residency = KVResidency::kRecompute;
    table->register_page(seq_page);

    ContextPrefetchPredictor predictor;
    auto candidates = predictor.predict(501, table, {500}, 8);

    CHECK(!candidates.empty());
    // First candidate should be graph dependency (callee 510)
    CHECK(candidates[0].page_id == 510);
    CHECK(candidates[0].priority >= 0.90f);
    CHECK(candidates[0].reason == "graph_dependency");

    // Sequential candidate should also be present
    bool has_seq = false;
    for (const auto& c : candidates) {
        if (c.page_id == 502) {
            has_seq = true;
            CHECK(c.reason == "sequential_stride");
        }
    }
    CHECK(has_seq);
}

static void test_concurrent_access() {
    std::cout << "[7] Concurrent thread safety\n";

    KVTierBudgets budgets;
    auto table = std::make_shared<VirtualKVPageTable>(budgets);

    std::vector<std::thread> workers;
    for (int t = 0; t < 8; ++t) {
        workers.emplace_back([table, t]() {
            for (int i = 0; i < 200; ++i) {
                uint64_t id = static_cast<uint64_t>(t * 1000 + i + 1);
                auto page = std::make_shared<KVPageDescriptor>();
                page->page_id = id;
                page->key = KVBlockKey{100, 1, id, 0, 512, 0, 32, KVFormat::kFP16, 1};
                page->residency = KVResidency::kHost_RAM;
                page->byte_size = 1024 * 1024;
                table->register_page(page);

                table->record_access(id, 100 + i);
                auto p = table->lookup(page->key);
                CHECK(p != nullptr);
            }
        });
    }

    for (auto& w : workers) {
        w.join();
    }

    auto stats = table->stats();
    CHECK(stats.total_pages == 1600);
    CHECK(stats.lookups == 1600);
}

int main() {
    std::cout << "Running Virtual KV Page Table tests...\n";
    test_key_invariance_and_rope();
    test_hierarchical_tier_transitions();
    test_cost_aware_eviction();
    test_pinned_pages();
    test_working_set_estimation();
    test_prefetch_predictor();
    test_concurrent_access();
    std::cout << "ALL PASSED\n";
    return 0;
}
