// tests/core/test_ram_expert_cache.cpp - Tests for Pinned Host RAM Expert Cache Tier
#include "strata/core/ram_expert_cache.hpp"

#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

void test_ram_expert_cache_basic() {
    std::cout << "[Test 1/5] Testing RAM Expert Cache Allocation & Open..." << std::endl;
    strata::core::RamExpertCache cache;
    std::string err;
    const int64_t n_slots = 64;
    const int64_t n_layers = 16;
    const int64_t n_experts = 32;
    const int64_t blob_bytes = 4096;

    bool ok = cache.open(n_slots, n_layers, n_experts, blob_bytes, false, &err);
    assert(ok && "RamExpertCache::open should succeed");
    assert(cache.valid());
    assert(cache.slots() == n_slots);
    assert(cache.bytes() == n_slots * blob_bytes);
    assert(cache.resident() == 0);
    std::cout << "  Passed. Cache opened with " << cache.slots() << " slots (" << cache.bytes() << " bytes)." << std::endl;
}

void test_ram_expert_cache_admission_and_lookup() {
    std::cout << "[Test 2/5] Testing RAM Expert Admission & O(1) Pointer Resolution..." << std::endl;
    strata::core::RamExpertCache cache;
    cache.open(32, 8, 16, 1024, false);

    std::vector<uint8_t> dummy_expert(1024, 0xAB);
    dummy_expert[0] = 42;
    dummy_expert[1023] = 99;

    int32_t slot = cache.admit(2, 5, dummy_expert.data(), 1024);
    assert(slot >= 0 && "Admission should return valid slot index");
    assert(cache.slot_of(2, 5) == slot);
    assert(cache.slot_of(2, 6) == strata::core::kRamNotResident);

    const uint8_t* ptr = cache.get_expert_ptr(2, 5);
    assert(ptr != nullptr);
    assert(ptr[0] == 42 && ptr[1023] == 99);
    assert(cache.hits() == 1);

    const uint8_t* missing = cache.get_expert_ptr(2, 6);
    assert(missing == nullptr);
    assert(cache.misses() == 1);
    std::cout << "  Passed. Direct RAM pointer lookup verified with hit/miss tracking." << std::endl;
}

void test_ram_expert_cache_eviction() {
    std::cout << "[Test 3/5] Testing RAM Expert Dynamic LRU/Heat Eviction..." << std::endl;
    strata::core::RamExpertCache cache;
    // Small cache of only 2 slots
    cache.open(2, 4, 8, 512, false);

    std::vector<uint8_t> exp0(512, 1);
    std::vector<uint8_t> exp1(512, 2);
    std::vector<uint8_t> exp2(512, 3);

    int32_t s0 = cache.admit(0, 1, exp0.data(), 512);
    int32_t s1 = cache.admit(0, 2, exp1.data(), 512);
    assert(cache.resident() == 2);

    // Make exp0 hot by recording multiple accesses
    for (int i = 0; i < 10; ++i) {
        cache.record_access(0, 1);
    }

    // Now admitting exp2 must evict exp1 (the cold one) rather than exp0
    int32_t s2 = cache.admit(0, 3, exp2.data(), 512);
    assert(s2 == s1 && "Should evict s1 (colder expert)");
    assert(cache.slot_of(0, 1) == s0 && "Hot expert should remain resident");
    assert(cache.slot_of(0, 2) == strata::core::kRamNotResident && "Cold expert should be evicted");
    assert(cache.slot_of(0, 3) == s2);

    const uint8_t* p0 = cache.get_expert_ptr(0, 1);
    assert(p0 != nullptr && p0[0] == 1);
    const uint8_t* p2 = cache.get_expert_ptr(0, 3);
    assert(p2 != nullptr && p2[0] == 3);
    std::cout << "  Passed. Heat-prioritized dynamic eviction correctly preserved hot experts." << std::endl;
}

void test_ram_expert_cache_sized_open() {
    std::cout << "[Test 4/5] Testing Sized RAM Expert Slots..." << std::endl;
    strata::core::RamExpertCache cache;
    std::vector<int64_t> sizes = {512, 1024, 2048, 4096};
    bool ok = cache.open_sized(sizes, 4, 8, false);
    assert(ok);
    assert(cache.slots() == 4);
    assert(cache.bytes() == (512 + 1024 + 2048 + 4096));

    std::vector<uint8_t> data(4096, 0x77);
    int32_t s0 = cache.admit(1, 0, data.data(), 512);
    assert(s0 == 0);
    int32_t s3 = cache.admit(1, 3, data.data(), 4096);
    assert(s3 == 1);
    std::cout << "  Passed. Sized RAM slots correctly allocated and populated." << std::endl;
}

void test_ram_expert_cache_preload() {
    std::cout << "[Test 5/5] Testing Bulk Preload into RAM Expert Cache..." << std::endl;
    strata::core::RamExpertCache cache;
    cache.open(16, 4, 4, 256, false);

    std::vector<uint8_t> dummy(256, 0xEE);
    auto get_fn = [&](int64_t l, int64_t e, int64_t& sz) -> const uint8_t* {
        sz = 256;
        return dummy.data();
    };

    int64_t loaded = cache.preload_all(4, 4, get_fn);
    assert(loaded == 16);
    assert(cache.resident() == 16);
    for (int64_t l = 0; l < 4; ++l) {
        for (int64_t e = 0; e < 4; ++e) {
            assert(cache.slot_of(l, e) >= 0);
        }
    }
    std::cout << "  Passed. 16 experts bulk preloaded into host RAM." << std::endl;
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "   RUNNING STRATA PINNED RAM EXPERT CACHE TIER TEST SUITE       " << std::endl;
    std::cout << "=================================================================" << std::endl;

    test_ram_expert_cache_basic();
    test_ram_expert_cache_admission_and_lookup();
    test_ram_expert_cache_eviction();
    test_ram_expert_cache_sized_open();
    test_ram_expert_cache_preload();

    std::cout << "=================================================================" << std::endl;
    std::cout << "   ALL RAM EXPERT CACHE TESTS PASSED (5/5)                       " << std::endl;
    std::cout << "=================================================================" << std::endl;
    return 0;
}
