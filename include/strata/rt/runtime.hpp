// include/strata/rt/runtime.hpp - the transparent inference runtime
//
// To a caller this is one model behind one endpoint: messages in, text out, optionally streamed. Everything below
// (context assembly, retrieval, memory, verification, loop protection, dynamic temperature, recovery, caches) is an
// internal mechanism. Nothing here is a tool the model or the agent has to call.
#pragma once

#include "strata/generation/dynamic_temperature.hpp"
#include "strata/generation/generation_loop_detector.hpp"
#include "strata/rt/context_manager.hpp"
#include "strata/rt/generator.hpp"
#include "strata/rt/memory.hpp"
#include "strata/rt/recovery.hpp"
#include "strata/rt/stream_buffer.hpp"
#include "strata/rt/tool_output.hpp"
#include "strata/rt/verifier.hpp"
#include "strata/rt/virtual_context.hpp"

#include <atomic>
#include <functional>
#include <memory>
#include <vector>

namespace strata::rt {

// ---- what the caller sees ----
struct InferenceRequest {
    RequestScope scope;
    std::string model;
    std::vector<Message> messages;
    Sampling sampling;
    bool stream = false;
    bool debug = false;       // returns a trace in the response (diagnostics only; also available via metrics/admin)
};

struct TraceEvent {
    double ms = 0.0;
    std::string kind;
    std::string detail;
};

struct InferenceResponse {
    std::string text;
    FinishReason finish = FinishReason::kStop;
    int prompt_tokens = 0;
    int completion_tokens = 0;
    std::vector<TraceEvent> trace;   // empty unless the request asked for debug
};

using TextSink = std::function<void(const std::string&)>;

// ---- configuration ----
struct RuntimeConfig {
    bool verification_enabled = true;
    ClaimRisk min_claim_risk = ClaimRisk::kMedium;
    RecoveryPolicy recovery;
    size_t stream_max_hold_chars = 400;
    ContextConfig context;
    ToolOutputConfig tool_output;
    bool memory_auto_store = false;           // remember durable statements from user turns
    bool loop_protection = true;
    generation::GenerationLoopConfig loop;
    generation::DynamicTemperatureConfig temperature;   // base_temperature is replaced by the request's temperature
    double verification_budget_ms = 1000.0;   // total verification time per request
    bool hedge_with_verified_value = true;    // an exact computed/authoritative value may replace a contradicted one
    int max_attempts = 8;                     // hard cap on generation attempts per request
    VirtualContextConfig virtual_context;     // physical window, advertised virtual size, tiers
    bool persist_contexts = true;             // sessions survive a restart (content-addressed files + manifests)
    std::string context_storage_dir;          // empty: $STRATA_CONTEXT_DIR, else ~/.strata/contexts
};

struct RuntimeDeps {
    std::shared_ptr<IGenerator> generator;
    std::shared_ptr<ITokenizer> tokenizer;
    std::shared_ptr<IPromptTemplate> prompt_template;
    std::shared_ptr<ExactStateStore> exact_state;
    std::shared_ptr<context::MultiTenantContextManager> documents;
    std::shared_ptr<MemoryStore> memory;
    std::shared_ptr<ProgressiveVerifier> verifier;
    std::shared_ptr<VirtualContextStore> contexts;   // created from the config when absent
};

struct RuntimeMetrics {
    uint64_t requests = 0, attempts = 0, regenerations = 0, loop_recoveries = 0, retrievals = 0;
    uint64_t claims_checked = 0, contradictions = 0, late_contradictions = 0, hedged = 0, unknown_delivered = 0;
    uint64_t tool_results_compacted = 0, temperature_changes = 0, safe_stops = 0;
    uint64_t verification_skipped_low_risk = 0;
    uint64_t vc_selections = 0, vc_page_ins = 0, vc_page_outs = 0, vc_unavailable = 0, vc_rejected = 0;
    uint64_t kv_reused_tokens = 0, kv_rebuilt_tokens = 0;
};

class InferenceRuntime {
public:
    InferenceRuntime(RuntimeDeps deps, RuntimeConfig cfg = RuntimeConfig());

    // The only call a caller needs. With a sink, text is delivered as soon as it is safe; the return value always
    // holds the complete final text.
    InferenceResponse generate(const InferenceRequest& request, const TextSink& sink = nullptr);

    RuntimeMetrics metrics() const;
    ProgressiveVerifier& verifier() { return *deps_.verifier; }
    ContextManager& context() { return *context_; }
    RequestEvidenceProvider& request_evidence() { return *request_evidence_; }
    const RuntimeConfig& config() const { return cfg_; }
    VirtualContextStore& contexts() { return *deps_.contexts; }
    // Model metadata: physical vs. virtual window for this session (the virtual size is never claimed as physical).
    std::string model_info_json(const RequestScope& scope);

private:
    RuntimeDeps deps_;
    RuntimeConfig cfg_;
    std::shared_ptr<RequestEvidenceProvider> request_evidence_;
    std::unique_ptr<ContextManager> context_;
    std::unique_ptr<ToolOutputManager> tools_;
    ClaimDetector detector_;
    mutable std::mutex metrics_mu_;
    RuntimeMetrics metrics_;
    void bump(uint64_t RuntimeMetrics::*field, uint64_t n = 1);
};

} // namespace strata::rt
