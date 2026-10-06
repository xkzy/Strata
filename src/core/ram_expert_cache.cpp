// src/core/ram_expert_cache.cpp - Host DDR4/DDR5 RAM Expert Cache Tier Implementation
#include "strata/core/ram_expert_cache.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

#if defined(STRATA_ENABLE_CUDA)
#include <cuda_runtime.h>
#elif defined(STRATA_ENABLE_HIP) || defined(STRATA_HIP_GFX906)
#include <hip/hip_runtime.h>
#endif

#if defined(_WIN32)
#include <windows.h>
#include <memoryapi.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace strata::core {

RamExpertCache::RamExpertCache() = default;

RamExpertCache::~RamExpertCache() {
    close();
}

bool RamExpertCache::open(int64_t n_slots, int64_t n_layers, int64_t n_expert, int64_t blob_bytes,
                          bool pinned, std::string* err) {
    if (n_slots <= 0 || n_layers <= 0 || n_expert <= 0 || blob_bytes <= 0) {
        if (err) *err = "Invalid parameters for RamExpertCache::open";
        return false;
    }

    close();

    std::lock_guard<std::mutex> lock(mutex_);
    slots_ = n_slots;
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    blob_bytes_ = blob_bytes;
    total_bytes_ = n_slots * blob_bytes;
    pinned_ = false;

    // Attempt pinned host allocation if requested
#if defined(STRATA_ENABLE_CUDA)
    if (pinned) {
        cudaError_t st = cudaHostAlloc((void**)&base_, (size_t)total_bytes_, cudaHostAllocPortable);
        if (st == cudaSuccess) {
            pinned_ = true;
        }
    }
#elif defined(STRATA_ENABLE_HIP) || defined(STRATA_HIP_GFX906)
    if (pinned) {
        hipError_t st = hipHostMalloc((void**)&base_, (size_t)total_bytes_, hipHostMallocPortable);
        if (st == hipSuccess) {
            pinned_ = true;
        }
    }
#endif

    // Fallback to aligned host allocation
    if (!base_) {
#if defined(_WIN32)
        base_ = static_cast<uint8_t*>(_aligned_malloc(static_cast<size_t>(total_bytes_), 64));
#else
        void* ptr = nullptr;
        if (posix_memalign(&ptr, 64, static_cast<size_t>(total_bytes_)) == 0) {
            base_ = static_cast<uint8_t*>(ptr);
        }
#endif
        if (!base_) {
            if (err) *err = "Failed to allocate " + std::to_string(total_bytes_) + " bytes for RAM expert cache";
            return false;
        }

        // Advise kernel to keep memory warm
#if !defined(_WIN32)
        madvise(base_, static_cast<size_t>(total_bytes_), MADV_WILLNEED);
#endif
    }

    // Zero-initialize memory
    std::memset(base_, 0, static_cast<size_t>(total_bytes_));

    // Initialize tables
    residency_.assign(static_cast<size_t>(n_layers * n_expert), kRamNotResident);
    offsets_.resize(static_cast<size_t>(n_slots));
    slot_sizes_.resize(static_cast<size_t>(n_slots), static_cast<uint64_t>(blob_bytes));
    slot_owners_.resize(static_cast<size_t>(n_slots), {-1, -1});
    access_counts_.resize(static_cast<size_t>(n_slots), 0);
    heat_scores_.resize(static_cast<size_t>(n_slots), 0.0);

    for (int64_t i = 0; i < n_slots; ++i) {
        offsets_[static_cast<size_t>(i)] = static_cast<uint64_t>(i * blob_bytes);
    }

    resident_count_ = 0;
    next_free_slot_ = 0;
    hits_ = 0;
    misses_ = 0;

    return true;
}

bool RamExpertCache::open_sized(const std::vector<int64_t>& slot_bytes, int64_t n_layers, int64_t n_expert,
                              bool pinned, std::string* err) {
    if (slot_bytes.empty() || n_layers <= 0 || n_expert <= 0) {
        if (err) *err = "Invalid parameters for RamExpertCache::open_sized";
        return false;
    }

    close();

    std::lock_guard<std::mutex> lock(mutex_);
    slots_ = static_cast<int64_t>(slot_bytes.size());
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    blob_bytes_ = 0;
    pinned_ = false;

    offsets_.resize(static_cast<size_t>(slots_));
    slot_sizes_.resize(static_cast<size_t>(slots_));

    uint64_t accum = 0;
    for (size_t i = 0; i < slot_bytes.size(); ++i) {
        offsets_[i] = accum;
        slot_sizes_[i] = static_cast<uint64_t>(slot_bytes[i]);
        accum += slot_sizes_[i];
    }
    total_bytes_ = static_cast<int64_t>(accum);

#if defined(STRATA_ENABLE_CUDA)
    if (pinned) {
        if (cudaHostAlloc((void**)&base_, (size_t)total_bytes_, cudaHostAllocPortable) == cudaSuccess) {
            pinned_ = true;
        }
    }
#elif defined(STRATA_ENABLE_HIP) || defined(STRATA_HIP_GFX906)
    if (pinned) {
        if (hipHostMalloc((void**)&base_, (size_t)total_bytes_, hipHostMallocPortable) == hipSuccess) {
            pinned_ = true;
        }
    }
#endif

    if (!base_) {
#if defined(_WIN32)
        base_ = static_cast<uint8_t*>(_aligned_malloc(static_cast<size_t>(total_bytes_), 64));
#else
        void* ptr = nullptr;
        if (posix_memalign(&ptr, 64, static_cast<size_t>(total_bytes_)) == 0) {
            base_ = static_cast<uint8_t*>(ptr);
        }
#endif
        if (!base_) {
            if (err) *err = "Failed to allocate memory for sized RAM expert cache";
            return false;
        }
    }

    std::memset(base_, 0, static_cast<size_t>(total_bytes_));
    residency_.assign(static_cast<size_t>(n_layers * n_expert), kRamNotResident);
    slot_owners_.resize(static_cast<size_t>(slots_), {-1, -1});
    access_counts_.resize(static_cast<size_t>(slots_), 0);
    heat_scores_.resize(static_cast<size_t>(slots_), 0.0);

    resident_count_ = 0;
    next_free_slot_ = 0;
    hits_ = 0;
    misses_ = 0;

    return true;
}

void RamExpertCache::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (base_) {
#if defined(STRATA_ENABLE_CUDA)
        if (pinned_) {
            cudaFreeHost(base_);
        } else {
            free(base_);
        }
#elif defined(STRATA_ENABLE_HIP) || defined(STRATA_HIP_GFX906)
        if (pinned_) {
            hipHostFree(base_);
        } else {
            free(base_);
        }
#elif defined(_WIN32)
        _aligned_free(base_);
#else
        free(base_);
#endif
        base_ = nullptr;
    }

    slots_ = 0;
    n_layers_ = 0;
    n_expert_ = 0;
    blob_bytes_ = 0;
    total_bytes_ = 0;
    resident_count_ = 0;
    next_free_slot_ = 0;
    pinned_ = false;
    residency_.clear();
    offsets_.clear();
    slot_sizes_.clear();
    slot_owners_.clear();
    access_counts_.clear();
    heat_scores_.clear();
}

int32_t RamExpertCache::slot_of(int64_t layer, int64_t expert) const {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_ || !base_) {
        return kRamNotResident;
    }
    size_t idx = static_cast<size_t>(layer * n_expert_ + expert);
    return residency_[idx];
}

const uint8_t* RamExpertCache::host_slot(int32_t slot) const {
    if (slot < 0 || slot >= slots_ || !base_) {
        return nullptr;
    }
    return base_ + offsets_[static_cast<size_t>(slot)];
}

uint8_t* RamExpertCache::host_slot(int32_t slot) {
    if (slot < 0 || slot >= slots_ || !base_) {
        return nullptr;
    }
    return base_ + offsets_[static_cast<size_t>(slot)];
}

const uint8_t* RamExpertCache::get_expert_ptr(int64_t layer, int64_t expert) const {
    int32_t slot = slot_of(layer, expert);
    if (slot >= 0) {
        hits_++;
        return host_slot(slot);
    }
    misses_++;
    return nullptr;
}

int32_t RamExpertCache::admit(int64_t layer, int64_t expert, const uint8_t* src_blob, int64_t bytes) {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_ || !base_ || !src_blob) {
        return kRamNotResident;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    size_t table_idx = static_cast<size_t>(layer * n_expert_ + expert);
    int32_t existing_slot = residency_[table_idx];
    if (existing_slot >= 0) {
        return existing_slot;
    }

    int32_t target_slot = kRamNotResident;
    if (next_free_slot_ < slots_) {
        target_slot = static_cast<int32_t>(next_free_slot_++);
        resident_count_++;
    } else {
        // Find slot with lowest heat score for eviction
        double min_heat = std::numeric_limits<double>::max();
        size_t evict_idx = 0;
        for (size_t s = 0; s < static_cast<size_t>(slots_); ++s) {
            if (heat_scores_[s] < min_heat) {
                min_heat = heat_scores_[s];
                evict_idx = s;
            }
        }
        target_slot = static_cast<int32_t>(evict_idx);
        
        // Evict current owner
        auto old_owner = slot_owners_[evict_idx];
        if (old_owner.first >= 0 && old_owner.second >= 0) {
            size_t old_table_idx = static_cast<size_t>(old_owner.first * n_expert_ + old_owner.second);
            if (old_table_idx < residency_.size()) {
                residency_[old_table_idx] = kRamNotResident;
            }
        }
    }

    // Copy expert payload into RAM slot
    uint64_t slot_cap = slot_sizes_[static_cast<size_t>(target_slot)];
    uint64_t to_copy = (bytes > 0 && static_cast<uint64_t>(bytes) <= slot_cap)
                           ? static_cast<uint64_t>(bytes)
                           : slot_cap;
    uint8_t* dst = base_ + offsets_[static_cast<size_t>(target_slot)];
    std::memcpy(dst, src_blob, to_copy);

    // Update metadata
    residency_[table_idx] = target_slot;
    slot_owners_[static_cast<size_t>(target_slot)] = {static_cast<int32_t>(layer), static_cast<int32_t>(expert)};
    access_counts_[static_cast<size_t>(target_slot)] = 1;
    heat_scores_[static_cast<size_t>(target_slot)] = 1.0;

    return target_slot;
}

bool RamExpertCache::evict(int64_t layer, int64_t expert) {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_ || !base_) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    size_t table_idx = static_cast<size_t>(layer * n_expert_ + expert);
    int32_t slot = residency_[table_idx];
    if (slot < 0) return false;

    residency_[table_idx] = kRamNotResident;
    slot_owners_[static_cast<size_t>(slot)] = {-1, -1};
    heat_scores_[static_cast<size_t>(slot)] = 0.0;
    access_counts_[static_cast<size_t>(slot)] = 0;
    resident_count_ = std::max<int64_t>(0, resident_count_ - 1);
    return true;
}

int64_t RamExpertCache::preload_all(int64_t n_layers, int64_t n_expert,
                                   const std::function<const uint8_t*(int64_t, int64_t, int64_t&)>& get_blob_fn) {
    if (!base_ || !get_blob_fn) return 0;

    int64_t loaded = 0;
    for (int64_t l = 0; l < n_layers; ++l) {
        for (int64_t e = 0; e < n_expert; ++e) {
            int64_t blob_sz = 0;
            const uint8_t* src = get_blob_fn(l, e, blob_sz);
            if (src && blob_sz > 0) {
                if (admit(l, e, src, blob_sz) >= 0) {
                    loaded++;
                    if (loaded >= slots_) return loaded;
                }
            }
        }
    }
    return loaded;
}

void RamExpertCache::record_access(int64_t layer, int64_t expert) {
    int32_t slot = slot_of(layer, expert);
    if (slot >= 0 && slot < slots_) {
        std::lock_guard<std::mutex> lock(mutex_);
        access_counts_[static_cast<size_t>(slot)]++;
        heat_scores_[static_cast<size_t>(slot)] += 1.0;
    }
}

} // namespace strata::core
