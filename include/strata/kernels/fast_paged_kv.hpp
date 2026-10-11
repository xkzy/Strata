// include/strata/kernels/fast_paged_kv.hpp
#pragma once

#include "strata/kernels/fast_parallel.hpp"
#include "strata/kernels/math_policy.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace strata::kernels {

/// Configuration for Paged KV Cache block allocation and physical geometry
struct PagedKVCacheConfig {
    size_t block_size = 16;         // Number of tokens per physical memory block (typically 16 or 32)
    size_t num_layers = 1;          // Number of transformer layers managed
    size_t num_kv_heads = 4;        // Number of Key/Value heads per layer (GQA/MHA)
    size_t head_dim = 128;          // Dimension per head (e.g. 64, 128, 256)
    size_t initial_pool_blocks = 256; // Initial pre-allocated physical blocks
    size_t max_pool_blocks = 4096;  // Upper bound on pool capacity
};

/// Represents a single fixed-size physical memory block storing K and V vectors.
/// Tokens inside a physical block are contiguous in memory for SIMD vectorization.
struct alignas(64) PagedBlock {
    uint32_t block_id = 0;
    std::atomic<uint32_t> ref_count{0}; // Reference counter for Copy-on-Write (COW) sharing
    std::vector<float> k_data;          // Layout: [block_size * num_kv_heads * head_dim]
    std::vector<float> v_data;          // Layout: [block_size * num_kv_heads * head_dim]

    PagedBlock() = default;

    PagedBlock(uint32_t id, size_t block_size, size_t num_kv_heads, size_t head_dim)
        : block_id(id), ref_count(0),
          k_data(block_size * num_kv_heads * head_dim, 0.0f),
          v_data(block_size * num_kv_heads * head_dim, 0.0f) {}

    // Movable
    PagedBlock(PagedBlock&& o) noexcept
        : block_id(o.block_id),
          ref_count(o.ref_count.load(std::memory_order_relaxed)),
          k_data(std::move(o.k_data)),
          v_data(std::move(o.v_data)) {}

    PagedBlock& operator=(PagedBlock&& o) noexcept {
        if (this != &o) {
            block_id = o.block_id;
            ref_count.store(o.ref_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            k_data = std::move(o.k_data);
            v_data = std::move(o.v_data);
        }
        return *this;
    }

    // Disable copy to preserve reference counting integrity
    PagedBlock(const PagedBlock&) = delete;
    PagedBlock& operator=(const PagedBlock&) = delete;

    inline const float* k_token_ptr(size_t token_offset, size_t kv_head, size_t num_kv_heads, size_t head_dim) const noexcept {
        return k_data.data() + (token_offset * num_kv_heads + kv_head) * head_dim;
    }

    inline float* k_token_ptr_mut(size_t token_offset, size_t kv_head, size_t num_kv_heads, size_t head_dim) noexcept {
        return k_data.data() + (token_offset * num_kv_heads + kv_head) * head_dim;
    }

    inline const float* v_token_ptr(size_t token_offset, size_t kv_head, size_t num_kv_heads, size_t head_dim) const noexcept {
        return v_data.data() + (token_offset * num_kv_heads + kv_head) * head_dim;
    }

    inline float* v_token_ptr_mut(size_t token_offset, size_t kv_head, size_t num_kv_heads, size_t head_dim) noexcept {
        return v_data.data() + (token_offset * num_kv_heads + kv_head) * head_dim;
    }
};

/// Sequence-level Block Table: maps logical token indices to physical block IDs
struct SequenceBlockTable {
    uint64_t seq_id = 0;
    size_t num_tokens = 0;                     // Logical context length in tokens
    std::vector<uint32_t> physical_block_ids; // Ordered list of physical block indices

    inline size_t num_blocks() const noexcept {
        return physical_block_ids.size();
    }

    inline uint32_t physical_block_at(size_t logical_block_idx) const {
        if (logical_block_idx >= physical_block_ids.size()) {
            throw std::out_of_range("Logical block index out of bounds in block table");
        }
        return physical_block_ids[logical_block_idx];
    }
};

/// Telemetry and memory fragmentation statistics
struct PagedCacheStats {
    size_t total_blocks = 0;
    size_t allocated_blocks = 0;
    size_t free_blocks = 0;
    size_t active_sequences = 0;
    size_t total_cached_tokens = 0;
    size_t memory_used_bytes = 0;
    size_t memory_pool_bytes = 0;
    size_t shared_cow_blocks = 0; // Number of physical blocks referenced by >1 sequence
    double internal_fragmentation_pct = 0.0; // Percentage of unused space in tail blocks
    double memory_saved_by_cow_mb = 0.0;     // Memory saved by shared prefix caching
};

/// High-performance Paged KV Cache Manager
/// Implements virtual memory paging for Key-Value attention tensors, eliminating
/// external fragmentation and enabling zero-copy prompt prefix sharing.
class FastPagedKVCache {
public:
    explicit FastPagedKVCache(const PagedKVCacheConfig& config = PagedKVCacheConfig())
        : config_(config) {
        if (config_.block_size == 0) config_.block_size = 16;
        if (config_.num_kv_heads == 0) config_.num_kv_heads = 1;
        if (config_.head_dim == 0) config_.head_dim = 128;
        if (config_.initial_pool_blocks == 0) config_.initial_pool_blocks = 16;
        if (config_.max_pool_blocks < config_.initial_pool_blocks) {
            config_.max_pool_blocks = config_.initial_pool_blocks;
        }

        initialize_pool(config_.initial_pool_blocks);
    }

    ~FastPagedKVCache() = default;

    const PagedKVCacheConfig& config() const noexcept { return config_; }

    // ------------------------------------------------------------------------
    // Sequence Lifecycle Management
    // ------------------------------------------------------------------------

    /// Registers a new logical sequence
    bool create_sequence(uint64_t seq_id) {
        std::lock_guard<std::mutex> lock(mu_);
        if (sequences_.find(seq_id) != sequences_.end()) {
            return false;
        }
        SequenceBlockTable table;
        table.seq_id = seq_id;
        table.num_tokens = 0;
        sequences_.emplace(seq_id, std::move(table));
        return true;
    }

    /// Destroys a sequence and releases its physical blocks back to the pool
    bool destroy_sequence(uint64_t seq_id) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = sequences_.find(seq_id);
        if (it == sequences_.end()) {
            return false;
        }

        // Decrement reference counts on all physical blocks owned by this sequence
        for (uint32_t block_id : it->second.physical_block_ids) {
            release_block_locked(block_id);
        }

        sequences_.erase(it);
        return true;
    }

    /// Fork a sequence using Copy-on-Write (COW). The child shares all current
    /// physical blocks with the parent. Memory is only allocated when modified.
    bool fork_sequence(uint64_t parent_seq_id, uint64_t child_seq_id) {
        std::lock_guard<std::mutex> lock(mu_);
        auto p_it = sequences_.find(parent_seq_id);
        if (p_it == sequences_.end() || sequences_.find(child_seq_id) != sequences_.end()) {
            return false;
        }

        SequenceBlockTable child_table;
        child_table.seq_id = child_seq_id;
        child_table.num_tokens = p_it->second.num_tokens;
        child_table.physical_block_ids = p_it->second.physical_block_ids;

        // Increment ref counts for all shared physical blocks
        for (uint32_t block_id : child_table.physical_block_ids) {
            blocks_[block_id].ref_count.fetch_add(1, std::memory_order_relaxed);
        }

        sequences_.emplace(child_seq_id, std::move(child_table));
        return true;
    }

    // ------------------------------------------------------------------------
    // Token Appending and Storage
    // ------------------------------------------------------------------------

    /// Appends a single token's KV representations for all KV heads
    /// Automatically handles block boundary allocation and Copy-on-Write duplication
    bool append_token(uint64_t seq_id, const float* k_all_heads, const float* v_all_heads) {
        if (!k_all_heads || !v_all_heads) return false;

        std::lock_guard<std::mutex> lock(mu_);
        auto it = sequences_.find(seq_id);
        if (it == sequences_.end()) return false;

        SequenceBlockTable& table = it->second;
        const size_t token_idx = table.num_tokens;
        const size_t block_idx = token_idx / config_.block_size;
        const size_t token_offset = token_idx % config_.block_size;

        if (block_idx >= table.physical_block_ids.size()) {
            // Need a new block
            uint32_t new_block_id = allocate_block_locked();
            if (new_block_id == kInvalidBlockId) return false;
            table.physical_block_ids.push_back(new_block_id);
        } else if (token_offset == 0) {
            // Start of a block slot
            uint32_t new_block_id = allocate_block_locked();
            if (new_block_id == kInvalidBlockId) return false;
            table.physical_block_ids.push_back(new_block_id);
        } else {
            // Writing into an existing block: Check if COW duplication is needed
            uint32_t current_block_id = table.physical_block_ids.back();
            if (blocks_[current_block_id].ref_count.load(std::memory_order_relaxed) > 1) {
                // Shared with another sequence! Perform Copy-on-Write split
                uint32_t fresh_block_id = allocate_block_locked();
                if (fresh_block_id == kInvalidBlockId) return false;

                // Copy previously populated tokens in this block
                const size_t copy_elems = token_offset * config_.num_kv_heads * config_.head_dim;
                std::memcpy(blocks_[fresh_block_id].k_data.data(),
                            blocks_[current_block_id].k_data.data(),
                            copy_elems * sizeof(float));
                std::memcpy(blocks_[fresh_block_id].v_data.data(),
                            blocks_[current_block_id].v_data.data(),
                            copy_elems * sizeof(float));

                // Release shared block and point table to the fresh private block
                release_block_locked(current_block_id);
                table.physical_block_ids.back() = fresh_block_id;
            }
        }

        // Write the token's KV data into the physical block
        uint32_t target_block_id = table.physical_block_ids.back();
        PagedBlock& target_block = blocks_[target_block_id];
        const size_t write_offset = token_offset * config_.num_kv_heads * config_.head_dim;
        const size_t write_count = config_.num_kv_heads * config_.head_dim;

        std::memcpy(target_block.k_data.data() + write_offset, k_all_heads, write_count * sizeof(float));
        std::memcpy(target_block.v_data.data() + write_offset, v_all_heads, write_count * sizeof(float));

        table.num_tokens++;
        return true;
    }

    /// Appends a contiguous chunk of tokens (e.g. during prompt prefill)
    bool append_prompt_chunk(uint64_t seq_id, const float* k_chunk, const float* v_chunk, size_t chunk_tokens) {
        if (!k_chunk || !v_chunk || chunk_tokens == 0) return false;
        const size_t token_stride = config_.num_kv_heads * config_.head_dim;

        for (size_t t = 0; t < chunk_tokens; ++t) {
            const float* k_t = k_chunk + t * token_stride;
            const float* v_t = v_chunk + t * token_stride;
            if (!append_token(seq_id, k_t, v_t)) {
                return false;
            }
        }
        return true;
    }

    // ------------------------------------------------------------------------
    // Read & Access API
    // ------------------------------------------------------------------------

    /// Resolves direct pointer to a specific token's Key vector
    const float* resolve_token_k(uint64_t seq_id, size_t token_idx, size_t kv_head) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = sequences_.find(seq_id);
        if (it == sequences_.end() || token_idx >= it->second.num_tokens || kv_head >= config_.num_kv_heads) {
            return nullptr;
        }

        const size_t block_idx = token_idx / config_.block_size;
        const size_t token_offset = token_idx % config_.block_size;
        uint32_t block_id = it->second.physical_block_ids[block_idx];
        return blocks_[block_id].k_token_ptr(token_offset, kv_head, config_.num_kv_heads, config_.head_dim);
    }

    /// Resolves direct pointer to a specific token's Value vector
    const float* resolve_token_v(uint64_t seq_id, size_t token_idx, size_t kv_head) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = sequences_.find(seq_id);
        if (it == sequences_.end() || token_idx >= it->second.num_tokens || kv_head >= config_.num_kv_heads) {
            return nullptr;
        }

        const size_t block_idx = token_idx / config_.block_size;
        const size_t token_offset = token_idx % config_.block_size;
        uint32_t block_id = it->second.physical_block_ids[block_idx];
        return blocks_[block_id].v_token_ptr(token_offset, kv_head, config_.num_kv_heads, config_.head_dim);
    }

    /// Returns a copy of the block table for a sequence (used by attention kernels)
    bool get_block_table(uint64_t seq_id, std::vector<uint32_t>& out_block_ids, size_t& out_num_tokens) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = sequences_.find(seq_id);
        if (it == sequences_.end()) return false;

        out_block_ids = it->second.physical_block_ids;
        out_num_tokens = it->second.num_tokens;
        return true;
    }

    /// Access physical block directly by ID (read-only)
    const PagedBlock& get_block(uint32_t block_id) const {
        if (block_id >= blocks_.size()) {
            throw std::out_of_range("Invalid physical block ID");
        }
        return blocks_[block_id];
    }

    /// Returns context length of a sequence
    size_t sequence_length(uint64_t seq_id) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = sequences_.find(seq_id);
        return (it != sequences_.end()) ? it->second.num_tokens : 0;
    }

    // ------------------------------------------------------------------------
    // Telemetry and Statistics
    // ------------------------------------------------------------------------

    PagedCacheStats stats() const {
        std::lock_guard<std::mutex> lock(mu_);
        PagedCacheStats st;
        st.total_blocks = blocks_.size();
        st.free_blocks = free_block_ids_.size();
        st.allocated_blocks = st.total_blocks - st.free_blocks;
        st.active_sequences = sequences_.size();

        const size_t block_bytes = (config_.block_size * config_.num_kv_heads * config_.head_dim * sizeof(float)) * 2;
        st.memory_pool_bytes = st.total_blocks * block_bytes;
        st.memory_used_bytes = st.allocated_blocks * block_bytes;

        size_t total_tokens = 0;
        size_t total_capacity_tokens = st.allocated_blocks * config_.block_size;

        for (const auto& [_, seq] : sequences_) {
            total_tokens += seq.num_tokens;
        }
        st.total_cached_tokens = total_tokens;

        if (total_capacity_tokens > 0) {
            size_t unused_tokens = (total_capacity_tokens > total_tokens) ? (total_capacity_tokens - total_tokens) : 0;
            st.internal_fragmentation_pct = (double)unused_tokens / (double)total_capacity_tokens * 100.0;
        }

        size_t shared_cow = 0;
        for (const auto& b : blocks_) {
            uint32_t rc = b.ref_count.load(std::memory_order_relaxed);
            if (rc > 1) {
                shared_cow += (rc - 1);
            }
        }
        st.shared_cow_blocks = shared_cow;
        st.memory_saved_by_cow_mb = (double)(shared_cow * block_bytes) / (1024.0 * 1024.0);

        return st;
    }

    static constexpr uint32_t kInvalidBlockId = 0xFFFFFFFF;

private:
    void initialize_pool(size_t count) {
        blocks_.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            uint32_t id = static_cast<uint32_t>(blocks_.size());
            blocks_.emplace_back(id, config_.block_size, config_.num_kv_heads, config_.head_dim);
            free_block_ids_.push_back(id);
        }
    }

    uint32_t allocate_block_locked() {
        if (!free_block_ids_.empty()) {
            uint32_t id = free_block_ids_.back();
            free_block_ids_.pop_back();
            blocks_[id].ref_count.store(1, std::memory_order_relaxed);
            return id;
        }

        // Expand pool if within limit
        if (blocks_.size() < config_.max_pool_blocks) {
            uint32_t id = static_cast<uint32_t>(blocks_.size());
            blocks_.emplace_back(id, config_.block_size, config_.num_kv_heads, config_.head_dim);
            blocks_[id].ref_count.store(1, std::memory_order_relaxed);
            return id;
        }

        // Out of memory blocks
        return kInvalidBlockId;
    }

    void release_block_locked(uint32_t block_id) {
        if (block_id >= blocks_.size()) return;
        uint32_t prev_rc = blocks_[block_id].ref_count.fetch_sub(1, std::memory_order_relaxed);
        if (prev_rc <= 1) {
            // Released completely, return to free list
            blocks_[block_id].ref_count.store(0, std::memory_order_relaxed);
            free_block_ids_.push_back(block_id);
        }
    }

    PagedKVCacheConfig config_;
    mutable std::mutex mu_;
    std::vector<PagedBlock> blocks_;
    std::vector<uint32_t> free_block_ids_;
    std::unordered_map<uint64_t, SequenceBlockTable> sequences_;
};

} // namespace strata::kernels
