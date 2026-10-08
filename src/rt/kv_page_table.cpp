// src/rt/kv_page_table.cpp - implementation of virtual KV page table and prefetch predictor
#include "strata/rt/kv_page_table.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace strata::rt {

const char* to_string(KVResidency r) {
    switch (r) {
        case KVResidency::kGPU_VRAM: return "GPU_VRAM";
        case KVResidency::kHost_RAM: return "HOST_RAM";
        case KVResidency::kCompressed_RAM: return "COMPRESSED_RAM";
        case KVResidency::kNVMe_Disk: return "NVME_DISK";
        case KVResidency::kRecompute: return "RECOMPUTE";
    }
    return "UNKNOWN";
}

const char* to_string(KVFormat f) {
    switch (f) {
        case KVFormat::kFP16: return "FP16";
        case KVFormat::kBF16: return "BF16";
        case KVFormat::kFP8_E4M3: return "FP8_E4M3";
        case KVFormat::kINT8: return "INT8";
        case KVFormat::kQ4_0: return "Q4_0";
        case KVFormat::kHybrid: return "HYBRID";
    }
    return "UNKNOWN";
}

double KVPageDescriptor::retention_score(uint64_t current_tick, double recency_lambda) const {
    if (pinned) {
        return std::numeric_limits<double>::infinity();
    }
    // High prefill cost and high frequency increase retention; large byte sizes slightly lower it per-byte
    double cost_weight = (prefill_cost_ms * static_cast<double>(access_frequency + 1));
    double size_kb = std::max<double>(1.0, static_cast<double>(byte_size) / 1024.0);
    double score = (cost_weight / size_kb);

    double delta_ticks = (current_tick >= last_access_tick) ? static_cast<double>(current_tick - last_access_tick) : 0.0;
    double recency_decay = std::exp(-recency_lambda * delta_ticks);

    score *= recency_decay * (1.0 + 0.15 * static_cast<double>(dependent_pages.size()));
    return score;
}

VirtualKVPageTable::VirtualKVPageTable(KVTierBudgets budgets)
    : budgets_(budgets) {}

std::shared_ptr<KVPageDescriptor> VirtualKVPageTable::lookup(const KVBlockKey& key) {
    std::lock_guard<std::mutex> lock(mu_);
    stats_.lookups++;
    auto it = key_index_.find(key);
    if (it == key_index_.end()) {
        stats_.page_faults++;
        return nullptr;
    }
    auto page = it->second;
    size_t tier_idx = static_cast<size_t>(page->residency);
    if (tier_idx < 5) {
        stats_.hits[tier_idx]++;
    }
    return page;
}

std::shared_ptr<KVPageDescriptor> VirtualKVPageTable::get_page(uint64_t page_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = page_index_.find(page_id);
    return (it != page_index_.end()) ? it->second : nullptr;
}

bool VirtualKVPageTable::register_page(std::shared_ptr<KVPageDescriptor> page) {
    if (!page || page->page_id == 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mu_);
    key_index_[page->key] = page;
    page_index_[page->page_id] = page;
    stats_.total_pages = page_index_.size();

    size_t tier_idx = static_cast<size_t>(page->residency);
    if (tier_idx < 5) {
        stats_.tier_pages[tier_idx]++;
        stats_.tier_bytes[tier_idx] += page->byte_size;
    }
    return true;
}

bool VirtualKVPageTable::update_residency(uint64_t page_id, KVResidency target_tier, void* new_ptr, int32_t new_slot) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = page_index_.find(page_id);
    if (it == page_index_.end()) {
        return false;
    }
    auto page = it->second;
    if (page->residency == target_tier && page->host_ptr == new_ptr && page->vram_slot == new_slot) {
        return true;
    }

    size_t old_tier = static_cast<size_t>(page->residency);
    size_t new_tier = static_cast<size_t>(target_tier);

    if (old_tier < 5) {
        if (stats_.tier_pages[old_tier] > 0) stats_.tier_pages[old_tier]--;
        if (stats_.tier_bytes[old_tier] >= page->byte_size) {
            stats_.tier_bytes[old_tier] -= page->byte_size;
        } else {
            stats_.tier_bytes[old_tier] = 0;
        }
    }

    if (new_tier < old_tier) {
        stats_.promotions++;
    } else if (new_tier > old_tier) {
        stats_.demotions++;
    }

    page->residency = target_tier;
    page->host_ptr = new_ptr;
    page->vram_slot = new_slot;

    if (new_tier < 5) {
        stats_.tier_pages[new_tier]++;
        stats_.tier_bytes[new_tier] += page->byte_size;
    }

    return true;
}

void VirtualKVPageTable::pin(uint64_t page_id, bool is_pinned) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = page_index_.find(page_id);
    if (it != page_index_.end()) {
        it->second->pinned = is_pinned;
    }
}

void VirtualKVPageTable::record_access(uint64_t page_id, uint64_t current_tick) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = page_index_.find(page_id);
    if (it != page_index_.end()) {
        it->second->last_access_tick = current_tick;
        it->second->access_frequency++;
    }
}

std::vector<uint64_t> VirtualKVPageTable::estimate_working_set(uint64_t current_tick, uint64_t window_ticks) const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<std::pair<double, uint64_t>> scored_pages;

    for (const auto& kv : page_index_) {
        const auto& page = kv.second;
        if (page->pinned) {
            scored_pages.push_back({std::numeric_limits<double>::infinity(), page->page_id});
            continue;
        }
        uint64_t delta = (current_tick >= page->last_access_tick) ? (current_tick - page->last_access_tick) : 0;
        if (delta <= window_ticks) {
            double score = page->retention_score(current_tick);
            scored_pages.push_back({score, page->page_id});
        }
    }

    std::sort(scored_pages.begin(), scored_pages.end(), [](const auto& a, const auto& b) {
        return a.first > b.first; // highest retention score first
    });

    std::vector<uint64_t> working_set;
    working_set.reserve(scored_pages.size());
    for (const auto& item : scored_pages) {
        working_set.push_back(item.second);
    }
    return working_set;
}

size_t VirtualKVPageTable::evict_tier(KVResidency tier, uint64_t bytes_to_free, uint64_t current_tick) {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<std::pair<double, std::shared_ptr<KVPageDescriptor>>> candidates;

    for (const auto& kv : page_index_) {
        const auto& page = kv.second;
        if (page->residency == tier && !page->pinned) {
            candidates.push_back({page->retention_score(current_tick), page});
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        return a.first < b.first; // lowest score evicted first
    });

    uint64_t freed = 0;
    size_t evict_count = 0;
    KVResidency colder_tier = static_cast<KVResidency>(std::min<int>(static_cast<int>(tier) + 1, static_cast<int>(KVResidency::kRecompute)));

    for (auto& item : candidates) {
        if (freed >= bytes_to_free) {
            break;
        }
        auto page = item.second;
        size_t old_tier = static_cast<size_t>(page->residency);
        size_t new_tier = static_cast<size_t>(colder_tier);

        if (old_tier < 5) {
            if (stats_.tier_pages[old_tier] > 0) stats_.tier_pages[old_tier]--;
            if (stats_.tier_bytes[old_tier] >= page->byte_size) {
                stats_.tier_bytes[old_tier] -= page->byte_size;
            } else {
                stats_.tier_bytes[old_tier] = 0;
            }
        }

        freed += page->byte_size;
        page->residency = colder_tier;
        page->host_ptr = nullptr;
        page->vram_slot = -1;

        if (new_tier < 5) {
            stats_.tier_pages[new_tier]++;
            stats_.tier_bytes[new_tier] += page->byte_size;
        }

        stats_.demotions++;
        stats_.evictions++;
        evict_count++;
    }

    return freed;
}

KVTableStats VirtualKVPageTable::stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    return stats_;
}

std::vector<PrefetchCandidate> ContextPrefetchPredictor::predict(
    uint64_t active_page_id,
    const std::shared_ptr<IKVPageTable>& page_table,
    const std::vector<uint64_t>& recent_history,
    size_t max_candidates
) const {
    std::vector<PrefetchCandidate> candidates;
    if (!page_table || active_page_id == 0) {
        return candidates;
    }

    auto active_page = page_table->get_page(active_page_id);
    if (!active_page) {
        return candidates;
    }

    // 1. Graph dependencies (AST callers, callees, definitions)
    for (uint64_t dep_id : active_page->dependent_pages) {
        auto dep_page = page_table->get_page(dep_id);
        if (dep_page && dep_page->residency != KVResidency::kGPU_VRAM && dep_page->residency != KVResidency::kHost_RAM) {
            candidates.push_back({dep_id, KVResidency::kHost_RAM, 0.90f, "graph_dependency"});
        }
    }

    // 2. Sequential proximity (conversation flow or contiguous code pages)
    for (uint64_t offset = 1; offset <= 3; ++offset) {
        uint64_t seq_id = active_page_id + offset;
        auto seq_page = page_table->get_page(seq_id);
        if (seq_page && seq_page->residency != KVResidency::kGPU_VRAM && seq_page->residency != KVResidency::kHost_RAM) {
            float prio = 0.85f - 0.10f * static_cast<float>(offset - 1);
            candidates.push_back({seq_id, KVResidency::kHost_RAM, prio, "sequential_stride"});
        }
    }

    // 3. Temporal access history recurrence
    if (!recent_history.empty()) {
        for (auto it = recent_history.rbegin(); it != recent_history.rend(); ++it) {
            uint64_t hist_id = *it;
            if (hist_id != active_page_id) {
                auto hist_page = page_table->get_page(hist_id);
                if (hist_page && hist_page->residency == KVResidency::kCompressed_RAM) {
                    candidates.push_back({hist_id, KVResidency::kHost_RAM, 0.65f, "temporal_recency"});
                    break;
                }
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        return a.priority > b.priority;
    });

    if (candidates.size() > max_candidates) {
        candidates.resize(max_candidates);
    }
    return candidates;
}

} // namespace strata::rt
