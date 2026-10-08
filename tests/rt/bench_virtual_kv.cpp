// tests/rt/bench_virtual_kv.cpp - performance benchmark for virtual KV memory page table
#include "strata/rt/kv_page_table.hpp"

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using namespace strata::rt;
using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

int main(int argc, char** argv) {
    const int N = argc > 1 ? std::atoi(argv[1]) : 100000;
    std::printf("Virtual KV Memory Benchmark (%d pages)\n", N);

    KVTierBudgets budgets;
    budgets.host_ram_bytes = 64ull << 30; // 64 GB
    budgets.nvme_bytes = 512ull << 30;   // 512 GB
    VirtualKVPageTable table(budgets);

    std::vector<std::shared_ptr<KVPageDescriptor>> pages;
    pages.reserve(N);

    std::mt19937 rng(42);

    auto t0 = Clock::now();
    for (int i = 0; i < N; ++i) {
        auto page = std::make_shared<KVPageDescriptor>();
        page->page_id = static_cast<uint64_t>(i + 1);
        page->session_id = 1;
        uint64_t content_hash = 0x100000000ULL + i;
        uint32_t start_pos = static_cast<uint32_t>(i * 512);
        uint32_t end_pos = start_pos + 512;
        page->key = KVBlockKey{100, 1, content_hash, start_pos, end_pos, 0, 32, KVFormat::kFP16, 1};
        page->residency = (i < N / 4) ? KVResidency::kHost_RAM : (i < N / 2) ? KVResidency::kCompressed_RAM : KVResidency::kNVMe_Disk;
        page->byte_size = 512 * 4096; // 2 MB per block
        page->token_count = 512;
        page->prefill_cost_ms = 25.0f + static_cast<float>(rng() % 100);
        page->access_frequency = static_cast<uint32_t>(rng() % 10);
        page->last_access_tick = static_cast<uint64_t>(rng() % 10000);
        if (i % 50 == 0) {
            page->pinned = true;
        }
        if (i > 2) {
            page->dependent_pages = {static_cast<uint64_t>(i), static_cast<uint64_t>(i - 1)};
        }
        pages.push_back(page);
        table.register_page(page);
    }
    double register_ms = ms_since(t0);
    std::printf("[1] Page Registration: %d pages in %.2f ms (%.2f ns/page)\n", N, register_ms, (register_ms * 1e6) / N);

    // Benchmark 2: Exact Key Lookups
    const int lookups = 100000;
    t0 = Clock::now();
    size_t hits = 0;
    for (int k = 0; k < lookups; ++k) {
        int idx = (k * 1337) % N;
        auto p = table.lookup(pages[idx]->key);
        if (p) hits++;
    }
    double lookup_ms = ms_since(t0);
    std::printf("[2] Key Lookups: %d lookups in %.2f ms (%.2f ns/lookup, %zu hits)\n",
                lookups, lookup_ms, (lookup_ms * 1e6) / lookups, hits);

    // Benchmark 3: Working Set Estimation
    t0 = Clock::now();
    auto ws = table.estimate_working_set(10000, 2000);
    double ws_ms = ms_since(t0);
    std::printf("[3] Working Set Estimation: found %zu pages in %.2f ms\n", ws.size(), ws_ms);

    // Benchmark 4: Cost-Aware Eviction
    t0 = Clock::now();
    size_t freed = table.evict_tier(KVResidency::kHost_RAM, 64 * 1024 * 1024, 10000);
    double evict_ms = ms_since(t0);
    std::printf("[4] Cost-Aware Eviction: freed %zu MB in %.2f ms\n", freed / (1024 * 1024), evict_ms);

    // Benchmark 5: Context Prefetch Predictor
    ContextPrefetchPredictor predictor;
    auto table_ptr = std::make_shared<VirtualKVPageTable>(budgets);
    for (int i = 0; i < 500; ++i) {
        table_ptr->register_page(pages[i]);
    }
    t0 = Clock::now();
    const int prefetch_queries = 10000;
    for (int k = 0; k < prefetch_queries; ++k) {
        uint64_t active_pid = static_cast<uint64_t>((k % 400) + 10);
        auto candidates = predictor.predict(active_pid, table_ptr, {active_pid - 1}, 8);
        (void)candidates;
    }
    double prefetch_ms = ms_since(t0);
    std::printf("[5] Predictive Prefetching: %d predictions in %.2f ms (%.2f ns/prediction)\n",
                prefetch_queries, prefetch_ms, (prefetch_ms * 1e6) / prefetch_queries);

    return 0;
}
