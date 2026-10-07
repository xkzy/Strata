// src/rt/tool_output.cpp
#include "strata/rt/tool_output.hpp"

namespace strata::rt {

ToolOutputManager::ToolOutputManager(ToolOutputConfig cfg, std::shared_ptr<RequestEvidenceProvider> request_evidence)
    : cfg_(cfg), request_evidence_(std::move(request_evidence)), runtime_(nullptr) {}

size_t ToolOutputManager::process(const RequestScope& scope, std::vector<Message>& messages) {
    size_t compacted = 0;
    for (auto& m : messages) {
        if (m.role != "tool") continue;
        const std::string raw = m.content;
        // Even a small result is evidence for verification; only big ones are replaced in the prompt.
        Evidence ev;
        ev.kind = EvidenceKind::kToolResult;
        ev.text = raw.size() > cfg_.evidence_max_chars ? raw.substr(0, cfg_.evidence_max_chars) : raw;
        ev.hash = hash128(raw);
        ev.source_version = ev.hash;
        ev.source_id = "tool:" + (m.name.empty() ? std::string("result") : m.name) + ":" + ev.hash.substr(0, 12);
        ev.id = ev.source_id;
        if (request_evidence_) request_evidence_->add(scope.key(), ev);

        if (raw.size() <= cfg_.inline_limit_chars) {
            std::lock_guard<std::mutex> lock(mu_);
            prompt_bytes_ += raw.size();
            continue;
        }
        tools::ToolExecutionRequest req;
        req.tool_name = m.name.empty() ? "tool" : m.name;
        req.raw_output = raw;
        std::string observation;
        {
            std::lock_guard<std::mutex> lock(mu_);
            tools::ToolResult res = runtime_.process_tool_execution(req);
            observation = res.compact_observation;
            prompt_bytes_ += observation.size();
        }
        if (observation.empty() || observation.size() >= raw.size()) observation = raw.substr(0, cfg_.inline_limit_chars) + "\n[... output truncated; full result stored ...]";
        m.content = observation;
        ++compacted;
    }
    return compacted;
}

size_t ToolOutputManager::stored_results() const {
    std::lock_guard<std::mutex> lock(mu_);
    return runtime_.store().total_results();
}

uint64_t ToolOutputManager::raw_bytes() const {
    std::lock_guard<std::mutex> lock(mu_);
    return runtime_.store().total_raw_bytes();
}

} // namespace strata::rt
