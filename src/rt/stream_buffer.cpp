// src/rt/stream_buffer.cpp - bounded hold-back (see stream_buffer.hpp)
#include "strata/rt/stream_buffer.hpp"

#include "strata/rt/claim.hpp"

#include <algorithm>

namespace strata::rt {

void BoundedStreamBuffer::reset(const std::string& already_released_prefix) {
    text_ = already_released_prefix;
    released_ = checked_ = already_released_prefix.size();
    max_held_ = forced_ = 0;
}

void BoundedStreamBuffer::resume(const std::string& kept_text, size_t released, size_t checked) {
    text_ = kept_text;
    released_ = std::min(released, text_.size());
    checked_ = std::max(released_, std::min(checked, text_.size()));
    max_held_ = forced_ = 0;
}

bool BoundedStreamBuffer::truncate_to(size_t len) {
    if (len < released_ || len > text_.size()) return false;
    text_.resize(len);
    checked_ = std::min(checked_, len);
    return true;
}

bool BoundedStreamBuffer::replace_range(size_t begin, size_t end, const std::string& replacement) {
    if (begin < released_ || end < begin || end > text_.size()) return false;
    text_.replace(begin, end - begin, replacement);
    const long delta = static_cast<long>(replacement.size()) - static_cast<long>(end - begin);
    if (checked_ >= end) checked_ = static_cast<size_t>(static_cast<long>(checked_) + delta);
    else checked_ = std::min(checked_, begin);
    return true;
}

bool BoundedStreamBuffer::may_hold_claim(const std::string& s) { return ClaimDetector::may_contain_claims(s); }

std::string BoundedStreamBuffer::take_up_to(size_t end) {
    end = std::min(end, text_.size());
    if (end <= released_) return "";
    std::string out = text_.substr(released_, end - released_);
    released_ = end;
    return out;
}

std::string BoundedStreamBuffer::push(const std::string& text) {
    text_ += text;
    // Text is released once its sentence has been checked: a sentence that looks harmless now can still grow a claim
    // (digits, a path) in a later token, so nothing inside an open sentence is released early. The hold is bounded
    // by max_hold_chars below.
    size_t limit = checked_;
    // keep the unreleased tail within the bound
    if (text_.size() - std::max(limit, released_) > cfg_.max_hold_chars) {
        size_t forced_end = text_.size() - cfg_.max_hold_chars;
        if (forced_end > limit) { limit = forced_end; ++forced_; }
    }
    std::string out = take_up_to(limit);
    max_held_ = std::max(max_held_, held());
    return out;
}

void BoundedStreamBuffer::mark_checked(size_t offset) {
    checked_ = std::max(checked_, std::min(offset, text_.size()));
}

std::string BoundedStreamBuffer::flush() {
    checked_ = text_.size();
    return take_up_to(text_.size());
}

size_t BoundedStreamBuffer::retract_unreleased() {
    size_t dropped = text_.size() - released_;
    text_.resize(released_);
    checked_ = std::min(checked_, released_);
    return dropped;
}

} // namespace strata::rt
