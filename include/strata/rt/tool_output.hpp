// include/strata/rt/tool_output.hpp - big tool results never reach the model whole
#pragma once

#include "strata/rt/context_manager.hpp"
#include "strata/tools/tool_runtime.hpp"

#include <memory>
#include <mutex>

namespace strata::rt {

struct ToolOutputConfig {
    size_t inline_limit_chars = 1500;     // results up to this size pass through unchanged
    size_t evidence_max_chars = 262144;   // how much of a raw result is kept as evidence text
};

// The agent still runs its own tools. If it sends a huge result to the inference endpoint, this stores the raw
// result (content-addressed, versioned by hash), registers it as evidence for verification, and substitutes a
// compact observation for the prompt.
class ToolOutputManager {
public:
    ToolOutputManager(ToolOutputConfig cfg, std::shared_ptr<RequestEvidenceProvider> request_evidence);

    // Rewrites tool messages in place. Returns how many were compacted.
    size_t process(const RequestScope& scope, std::vector<Message>& messages);

    size_t stored_results() const;
    uint64_t raw_bytes() const;
    uint64_t prompt_bytes() const { return prompt_bytes_; }

private:
    ToolOutputConfig cfg_;
    std::shared_ptr<RequestEvidenceProvider> request_evidence_;
    mutable std::mutex mu_;
    tools::ToolRuntime runtime_;
    uint64_t prompt_bytes_ = 0;
};

} // namespace strata::rt
