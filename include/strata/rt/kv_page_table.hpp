// include/strata/rt/kv_page_table.hpp - virtual KV page table and hierarchical residency
//
// A persistent virtual KV address space that decouples logical context retrieval from physical KV residency.
// Maps logical context pages to hierarchical physical tiers (L1 VRAM, L2 Host RAM, L3 Compressed RAM, L4 NVMe, L5 Recompute).
//
// Invariants maintained:
//   * A KVBlockKey is strictly position-, model-, RoPE-, and content-aware; blocks are never reused across mismatched positions.
//   * Original tokens remain the authoritative backing store (L5); KV caches are derived and can be invalidated safely.
//   * Eviction is cost- and dependency-aware: expensive prefill pages are prioritized over cheap metadata.
//   * Predictive prefetching overlaps Host->Device PCIe transfers with GPU compute.
#pragma once

#include "strata/rt/types.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::rt {

enum class KVResidency : uint8_t {
    kGPU_VRAM = 0,       // L1: Resident in GPU VRAM slot (zero-copy attention)
    kHost_RAM = 1,       // L2: Resident in pinned Host RAM (DMA-ready)
    kCompressed_RAM = 2, // L3: Compressed/Quantized in Host RAM (FP8/INT8)
    kNVMe_Disk = 3,      // L4: Memory-mapped or direct I/O on NVMe SSD
    kRecompute = 4       // L5: Canonical tokens only (requires chunked prefill)
};

const char* to_string(KVResidency r);

enum class KVFormat : uint8_t {
    kFP16 = 0,
    kBF16 = 1,
    kFP8_E4M3 = 2,
    kINT8 = 3,
    kQ4_0 = 4,
    kHybrid = 5
};

const char* to_string(KVFormat f);

/// 64-byte aligned composite key ensuring mathematical & RoPE invariance
struct alignas(64) KVBlockKey {
    uint64_t model_hash = 0;          // Hash of model architecture and weight signature
    uint64_t tokenizer_id = 0;        // Tokenizer vocabulary/schema identifier
    uint64_t token_content_hash = 0;  // hash128 of the token sequence chunk
    uint32_t start_token_pos = 0;     // RoPE absolute sequence start position
    uint32_t end_token_pos = 0;       // RoPE absolute sequence end position
    uint16_t layer_start = 0;         // Layer range for layer-wise streaming
    uint16_t layer_end = 0;
    KVFormat format = KVFormat::kFP16;
    uint8_t  rope_type = 0;           // Base frequency & scaling parameters

    bool operator==(const KVBlockKey& o) const {
        return model_hash == o.model_hash &&
               tokenizer_id == o.tokenizer_id &&
               token_content_hash == o.token_content_hash &&
               start_token_pos == o.start_token_pos &&
               end_token_pos == o.end_token_pos &&
               layer_start == o.layer_start &&
               layer_end == o.layer_end &&
               format == o.format &&
               rope_type == o.rope_type;
    }
};

struct KVBlockKeyHasher {
    size_t operator()(const KVBlockKey& k) const noexcept {
        // Fast 64-bit mixer combining all fields
        uint64_t h = k.model_hash ^ (k.tokenizer_id * 0x9e3779b97f4a7c15ULL);
        h ^= (k.token_content_hash + 0x517cc1b727220a95ULL + (h << 6) + (h >> 2));
        h ^= ((static_cast<uint64_t>(k.start_token_pos) << 32) | k.end_token_pos);
        h ^= ((static_cast<uint64_t>(k.layer_start) << 48) |
              (static_cast<uint64_t>(k.layer_end) << 32) |
              (static_cast<uint64_t>(k.format) << 8) |
              static_cast<uint64_t>(k.rope_type));
        return static_cast<size_t>(h);
    }
};

struct KVPageDescriptor {
    uint64_t page_id = 0;
    uint64_t session_id = 0;
    KVBlockKey key;
    KVResidency residency = KVResidency::kRecompute;

    // Physical handles
    void* host_ptr = nullptr;         // Pinned UVA memory pointer if in Host RAM
    int32_t vram_slot = -1;           // VRAM slot index if in GPU VRAM
    uint64_t disk_offset = 0;         // NVMe file offset
    size_t byte_size = 0;             // Total bytes for this KV block
    size_t token_count = 0;

    // Access & cost telemetry
    uint64_t last_access_tick = 0;
    uint32_t access_frequency = 0;
    float reuse_probability = 1.0f;
    float prefill_cost_ms = 0.0f;     // Measured or estimated recomputation latency
    bool pinned = false;              // System instructions / prompt anchors

    // Structural dependencies
    std::vector<uint64_t> dependent_pages;
    std::string source_symbol;        // AST definition or file identifier

    // Compute eviction weight: higher score = retain longer
    double retention_score(uint64_t current_tick, double recency_lambda = 0.001) const;
};

struct KVTierBudgets {
    uint64_t vram_bytes = 4ull << 30;           // 4 GB VRAM KV pool
    uint64_t host_ram_bytes = 32ull << 30;       // 32 GB Warm Host RAM pool
    uint64_t compressed_ram_bytes = 16ull << 30; // 16 GB Compressed RAM pool
    uint64_t nvme_bytes = 256ull << 30;          // 256 GB NVMe SSD space
    size_t   vram_slots = 1024;
};

struct KVTableStats {
    uint64_t total_pages = 0;
    uint64_t tier_pages[5] = {0, 0, 0, 0, 0};
    uint64_t tier_bytes[5] = {0, 0, 0, 0, 0};
    uint64_t lookups = 0;
    uint64_t hits[5] = {0, 0, 0, 0, 0}; // L1 hits, L2 hits, etc.
    uint64_t promotions = 0;
    uint64_t demotions = 0;
    uint64_t evictions = 0;
    uint64_t page_faults = 0;
};

class IKVPageTable {
public:
    virtual ~IKVPageTable() = default;

    virtual std::shared_ptr<KVPageDescriptor> lookup(const KVBlockKey& key) = 0;
    virtual std::shared_ptr<KVPageDescriptor> get_page(uint64_t page_id) const = 0;
    virtual bool register_page(std::shared_ptr<KVPageDescriptor> page) = 0;
    virtual bool update_residency(uint64_t page_id, KVResidency target_tier, void* new_ptr = nullptr, int32_t new_slot = -1) = 0;
    virtual void pin(uint64_t page_id, bool is_pinned) = 0;
    virtual void record_access(uint64_t page_id, uint64_t current_tick) = 0;

    virtual std::vector<uint64_t> estimate_working_set(uint64_t current_tick, uint64_t window_ticks) const = 0;
    virtual size_t evict_tier(KVResidency tier, uint64_t bytes_to_free, uint64_t current_tick) = 0;
    virtual KVTableStats stats() const = 0;
};

class VirtualKVPageTable : public IKVPageTable {
public:
    explicit VirtualKVPageTable(KVTierBudgets budgets);
    ~VirtualKVPageTable() override = default;

    std::shared_ptr<KVPageDescriptor> lookup(const KVBlockKey& key) override;
    std::shared_ptr<KVPageDescriptor> get_page(uint64_t page_id) const override;
    bool register_page(std::shared_ptr<KVPageDescriptor> page) override;
    bool update_residency(uint64_t page_id, KVResidency target_tier, void* new_ptr = nullptr, int32_t new_slot = -1) override;
    void pin(uint64_t page_id, bool is_pinned) override;
    void record_access(uint64_t page_id, uint64_t current_tick) override;

    std::vector<uint64_t> estimate_working_set(uint64_t current_tick, uint64_t window_ticks) const override;
    size_t evict_tier(KVResidency tier, uint64_t bytes_to_free, uint64_t current_tick) override;
    KVTableStats stats() const override;

    const KVTierBudgets& budgets() const { return budgets_; }

private:
    KVTierBudgets budgets_;
    mutable std::mutex mu_;
    std::unordered_map<KVBlockKey, std::shared_ptr<KVPageDescriptor>, KVBlockKeyHasher> key_index_;
    std::unordered_map<uint64_t, std::shared_ptr<KVPageDescriptor>> page_index_;
    KVTableStats stats_;
};

// ---- Predictive Context Prefetcher ----

struct PrefetchCandidate {
    uint64_t page_id = 0;
    KVResidency target_tier = KVResidency::kHost_RAM;
    float priority = 0.0f;     // 0.0 to 1.0
    std::string reason;        // "graph_callee", "sequential", "conversation_tail"
};

class ContextPrefetchPredictor {
public:
    ContextPrefetchPredictor() = default;

    /// Generates priority-ordered prefetch candidates based on AST graph links, recency, and sequential offsets
    std::vector<PrefetchCandidate> predict(
        uint64_t active_page_id,
        const std::shared_ptr<IKVPageTable>& page_table,
        const std::vector<uint64_t>& recent_history,
        size_t max_candidates = 8
    ) const;
};

} // namespace strata::rt
