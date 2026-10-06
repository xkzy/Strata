// src/context/context_item.cpp - ContextItem Implementation
#include "strata/context/context_item.hpp"

namespace strata::context {

ContextItem::ContextItem(int64_t id, ContextItemType type, const std::string& raw_content,
                         int64_t token_count, double timestamp_sec)
    : id_(id), type_(type), token_count_(token_count),
      timestamp_(timestamp_sec), last_accessed_(timestamp_sec),
      raw_content_(raw_content) {}

std::string ContextItem::get_effective_content() const {
    switch (state_) {
        case ResidencyState::kActive:
        case ResidencyState::kWorking:
            return raw_content_;
        case ResidencyState::kCompressed:
            if (!summary_l1_.empty()) return summary_l1_;
            if (!summary_l2_.empty()) return summary_l2_;
            if (!summary_l3_.empty()) return summary_l3_;
            return raw_content_.substr(0, std::min<size_t>(raw_content_.size(), 200)) + " ... [compressed]";
        case ResidencyState::kIndexed:
        case ResidencyState::kArchived:
            return "[Archived/Indexed item " + std::to_string(id_) + "]";
    }
    return raw_content_;
}

} // namespace strata::context
