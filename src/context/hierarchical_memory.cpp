// src/context/hierarchical_memory.cpp - Hierarchical Context Memory Implementation
#include "strata/context/hierarchical_memory.hpp"

#include <algorithm>
#include <iostream>
#include <sstream>

namespace strata::context {

HierarchicalMemoryManager::HierarchicalMemoryManager() {
    l3_session_.node_id = 0;
    l3_session_.level = 3;
    l3_session_.summary_text = "Initial session state.";
    l3_session_.token_count = 10;
}

HierarchicalMemoryManager::~HierarchicalMemoryManager() = default;

void HierarchicalMemoryManager::add_raw_item(std::shared_ptr<ContextItem> item) {
    if (!item) return;
    std::lock_guard<std::mutex> lock(mutex_);
    int64_t id = item->id();
    raw_items_[id] = item;
    chronological_order_.push_back(id);
    total_raw_tokens_ += item->token_count();
}

std::shared_ptr<ContextItem> HierarchicalMemoryManager::get_source_item(int64_t item_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = raw_items_.find(item_id);
    if (it != raw_items_.end()) {
        return it->second;
    }
    return nullptr;
}

std::string HierarchicalMemoryManager::create_extractive_summary(
    const std::vector<std::shared_ptr<ContextItem>>& items, size_t max_sentences) const {

    std::ostringstream ss;
    size_t count = 0;
    for (const auto& it : items) {
        if (!it) continue;
        const auto& text = it->raw_content();
        size_t dot_pos = text.find('.');
        if (dot_pos != std::string::npos && dot_pos < 200) {
            ss << text.substr(0, dot_pos + 1) << " ";
        } else {
            ss << text.substr(0, std::min<size_t>(text.size(), 120)) << "... ";
        }
        count++;
        if (count >= max_sentences) break;
    }
    return ss.str();
}

void HierarchicalMemoryManager::update_hierarchy_incremental(int64_t chunk_token_threshold,
                                                            int64_t section_token_threshold) {
    std::lock_guard<std::mutex> lock(mutex_);

    // 1. Group L0 raw items into L1 chunks
    std::vector<std::shared_ptr<ContextItem>> current_chunk_items;
    int64_t current_chunk_tokens = 0;

    l1_chunks_.clear();
    for (int64_t id : chronological_order_) {
        auto item = raw_items_[id];
        if (!item) continue;

        current_chunk_items.push_back(item);
        current_chunk_tokens += item->token_count();

        if (current_chunk_tokens >= chunk_token_threshold) {
            HierarchyNode chunk;
            chunk.node_id = next_node_id_++;
            chunk.level = 1;
            chunk.summary_text = create_extractive_summary(current_chunk_items, 3);
            chunk.token_count = std::max<int64_t>(10, current_chunk_tokens / 10);
            for (const auto& ci : current_chunk_items) {
                chunk.child_item_ids.push_back(ci->id());
                ci->set_summary_l1(chunk.summary_text);
            }
            l1_chunks_.push_back(chunk);
            current_chunk_items.clear();
            current_chunk_tokens = 0;
        }
    }

    if (!current_chunk_items.empty()) {
        HierarchyNode chunk;
        chunk.node_id = next_node_id_++;
        chunk.level = 1;
        chunk.summary_text = create_extractive_summary(current_chunk_items, 3);
        chunk.token_count = std::max<int64_t>(10, current_chunk_tokens / 10);
        for (const auto& ci : current_chunk_items) {
            chunk.child_item_ids.push_back(ci->id());
            ci->set_summary_l1(chunk.summary_text);
        }
        l1_chunks_.push_back(chunk);
    }

    // 2. Aggregate L1 chunks into L2 sections
    l2_sections_.clear();
    std::vector<int64_t> current_section_chunks;
    int64_t current_sec_tokens = 0;
    std::string sec_summary_accum;

    for (const auto& chunk : l1_chunks_) {
        current_section_chunks.push_back(chunk.node_id);
        current_sec_tokens += chunk.token_count * 10;
        sec_summary_accum += chunk.summary_text + " ";

        if (current_sec_tokens >= section_token_threshold) {
            HierarchyNode sec;
            sec.node_id = next_node_id_++;
            sec.level = 2;
            sec.summary_text = sec_summary_accum.substr(0, std::min<size_t>(sec_summary_accum.size(), 300));
            sec.token_count = std::max<int64_t>(20, current_sec_tokens / 50);
            l2_sections_.push_back(sec);
            current_section_chunks.clear();
            current_sec_tokens = 0;
            sec_summary_accum.clear();
        }
    }

    if (!current_section_chunks.empty()) {
        HierarchyNode sec;
        sec.node_id = next_node_id_++;
        sec.level = 2;
        sec.summary_text = sec_summary_accum.substr(0, std::min<size_t>(sec_summary_accum.size(), 300));
        sec.token_count = std::max<int64_t>(20, current_sec_tokens / 50);
        l2_sections_.push_back(sec);
    }

    // 3. Update L3 Session summary
    std::ostringstream session_ss;
    for (const auto& sec : l2_sections_) {
        session_ss << sec.summary_text << " ";
    }
    std::string full_sess = session_ss.str();
    if (!full_sess.empty()) {
        l3_session_.summary_text = full_sess.substr(0, std::min<size_t>(full_sess.size(), 600));
        l3_session_.token_count = static_cast<int64_t>(l3_session_.summary_text.size() / 4);
    }

    // Calculate total summary tokens
    total_summary_tokens_ = 0;
    for (const auto& c : l1_chunks_) total_summary_tokens_ += c.token_count;
    for (const auto& s : l2_sections_) total_summary_tokens_ += s.token_count;
    total_summary_tokens_ += l3_session_.token_count;
}

std::string HierarchicalMemoryManager::get_session_summary() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return l3_session_.summary_text;
}

std::string HierarchicalMemoryManager::get_global_state() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return l4_global_state_;
}

void HierarchicalMemoryManager::set_global_state(const std::string& state) {
    std::lock_guard<std::mutex> lock(mutex_);
    l4_global_state_ = state;
}

std::vector<HierarchyNode> HierarchicalMemoryManager::get_level_nodes(int level) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (level == 1) return l1_chunks_;
    if (level == 2) return l2_sections_;
    if (level == 3) return {l3_session_};
    return {};
}

} // namespace strata::context
