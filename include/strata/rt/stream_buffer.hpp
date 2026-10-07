// include/strata/rt/stream_buffer.hpp - bounded hold-back between generation and the network
#pragma once

#include <cstddef>
#include <string>

namespace strata::rt {

struct StreamBufferConfig {
    size_t max_hold_chars = 400;    // never hold more than this: latency stays bounded
};

// Text is produced into the buffer and released to the caller once it is safe. Safe means: not inside a sentence
// that may contain a claim still awaiting verification. The buffer can retract everything not yet released, so a
// contradicted sentence is regenerated before the caller ever sees it.
class BoundedStreamBuffer {
public:
    explicit BoundedStreamBuffer(StreamBufferConfig cfg = StreamBufferConfig()) : cfg_(cfg) {}

    void reset(const std::string& already_released_prefix = "");
    // Resume with kept text: `released` chars of it already reached the caller, `checked` chars were already verified.
    void resume(const std::string& kept_text, size_t released, size_t checked);
    // Cuts the text back to `len` (len >= released): used to drop a repeated span before resuming.
    bool truncate_to(size_t len);
    // Replaces text[begin, end) by `replacement`; only unreleased text can change. Returns false if it was released.
    bool replace_range(size_t begin, size_t end, const std::string& replacement);
    // Appends text. Returns the newly releasable text (possibly empty).
    std::string push(const std::string& text);
    // The runtime calls this after each sentence it has finished checking.
    void mark_checked(size_t offset);
    // Releases everything (end of generation).
    std::string flush();
    // Drops all unreleased text and returns the length that was dropped. Released text stays.
    size_t retract_unreleased();

    const std::string& text() const { return text_; }
    size_t released() const { return released_; }
    size_t checked() const { return checked_; }
    size_t held() const { return text_.size() - released_; }
    size_t max_held_seen() const { return max_held_; }
    size_t forced_releases() const { return forced_; }   // times the bound forced out unchecked text

private:
    StreamBufferConfig cfg_;
    std::string text_;
    size_t released_ = 0;
    size_t checked_ = 0;
    size_t max_held_ = 0;
    size_t forced_ = 0;
    std::string take_up_to(size_t end);
    static bool may_hold_claim(const std::string& s);
};

} // namespace strata::rt
