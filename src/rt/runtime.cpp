// src/rt/runtime.cpp - the transparent inference runtime (see runtime.hpp)
#include "strata/rt/runtime.hpp"

#include "strata/rt/vc_evidence.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>

namespace strata::rt {

namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

LoopState to_loop_state(generation::LoopConfidence c) {
    switch (c) {
        case generation::LoopConfidence::kSuspicious: return LoopState::kSuspicious;
        case generation::LoopConfidence::kProbableLoop: return LoopState::kProbableLoop;
        case generation::LoopConfidence::kConfirmedLoop: return LoopState::kConfirmedLoop;
        default: return LoopState::kNormal;
    }
}

// replaces the first occurrence of `from` inside text[begin,end) by `to`; returns the new sentence, empty if absent
std::string substitute_value(const std::string& text, size_t begin, size_t end, const std::string& from, const std::string& to) {
    if (from.empty()) return "";
    std::string sentence = text.substr(begin, end - begin);
    size_t p = sentence.find(from);
    if (p == std::string::npos) return "";
    sentence.replace(p, from.size(), to);
    return sentence;
}
} // namespace

InferenceRuntime::InferenceRuntime(RuntimeDeps deps, RuntimeConfig cfg)
    : deps_(std::move(deps)), cfg_(std::move(cfg)), request_evidence_(std::make_shared<RequestEvidenceProvider>()),
      detector_(DetectorConfig{cfg_.min_claim_risk, 8}) {
    if (!deps_.prompt_template) deps_.prompt_template = std::make_shared<ChatMLTemplate>();
    if (!deps_.verifier) {
        deps_.verifier = std::make_shared<ProgressiveVerifier>(std::make_shared<VerificationCache>());
        deps_.verifier->add_verifier(std::make_shared<ExactEvidenceVerifier>());
        deps_.verifier->add_verifier(std::make_shared<ArithmeticVerifier>());
    }
    if (!deps_.memory) deps_.memory = std::make_shared<MemoryStore>();
    if (cfg_.virtual_context.enabled) {
        if (!deps_.contexts) {
            std::string dir;
            if (cfg_.persist_contexts) {
                dir = cfg_.context_storage_dir;
                if (dir.empty()) {
                    const char* e = std::getenv("STRATA_CONTEXT_DIR");
                    const char* h = std::getenv("HOME");
                    dir = e ? e : (h ? std::string(h) + "/.strata/contexts" : "");
                }
            }
            deps_.contexts = std::make_shared<VirtualContextStore>(cfg_.virtual_context, nullptr, dir);
        }
        cfg_.context.max_prompt_tokens = static_cast<size_t>(cfg_.virtual_context.physical_tokens);
    }
    std::vector<std::shared_ptr<IEvidenceProvider>> providers;
    if (cfg_.virtual_context.enabled) providers.push_back(std::make_shared<VirtualContextEvidenceProvider>(deps_.contexts));
    if (deps_.exact_state) providers.push_back(deps_.exact_state);
    if (deps_.documents) providers.push_back(std::make_shared<DocumentEvidenceProvider>(deps_.documents));
    providers.push_back(request_evidence_);
    context_ = std::make_unique<ContextManager>(cfg_.context, std::move(providers), deps_.memory);
    tools_ = std::make_unique<ToolOutputManager>(cfg_.tool_output, request_evidence_);
}

void InferenceRuntime::bump(uint64_t RuntimeMetrics::*field, uint64_t n) {
    std::lock_guard<std::mutex> lock(metrics_mu_);
    metrics_.*field += n;
}

std::string InferenceRuntime::model_info_json(const RequestScope& scope) {
    ContextWindowInfo w;
    if (cfg_.virtual_context.enabled) w = deps_.contexts->open(scope.security)->window_info();
    else { w.physical = cfg_.virtual_context.physical_tokens; w.virtual_advertised = w.virtual_capacity = w.physical; }
    return w.to_json();
}

RuntimeMetrics InferenceRuntime::metrics() const {
    std::lock_guard<std::mutex> lock(metrics_mu_);
    return metrics_;
}

InferenceResponse InferenceRuntime::generate(const InferenceRequest& request, const TextSink& sink) {
    const auto t0 = Clock::now();
    InferenceResponse resp;
    auto trace = [&](const std::string& kind, const std::string& detail) {
        if (request.debug) resp.trace.push_back({ms_since(t0), kind, detail});
    };
    bump(&RuntimeMetrics::requests);

    // ---- 1. internal context: virtual window (paging/retrieval), memory, tool-output handling ----
    std::vector<Message> convo = request.messages;
    std::shared_ptr<VirtualContext> vc;
    std::set<uint64_t> current_items;
    std::string vc_query;
    SelectionRequest sel;
    auto reselect = [&](const std::string& extra) {
        if (!extra.empty()) sel.extra_queries.push_back(extra);
        WorkingSet ws = vc->select(sel);
        bump(&RuntimeMetrics::vc_selections);
        bump(&RuntimeMetrics::vc_page_ins, ws.paged_in);
        bump(&RuntimeMetrics::vc_page_outs, ws.paged_out);
        for (const auto& t : ws.trace) trace("context", t);
        convo = ws.messages;
        if (!ws.unavailable.empty()) {
            // honest failure state: the model is told what it does not have, it is not given a guess
            bump(&RuntimeMetrics::vc_unavailable);
            std::string n = "Part of the earlier context this request refers to is not available to you:";
            for (const auto& u : ws.unavailable) n += "\n- " + u;
            n += "\nSay plainly what you do not have; do not guess or invent it.";
            convo.push_back({"system", n, ""});
            trace("context", "required_context_unavailable");
        }
    };
    if (cfg_.virtual_context.enabled) {
        vc = deps_.contexts->open(request.scope.security);
        std::set<uint64_t> fresh;
        std::vector<uint64_t> ids = vc->ingest(request.messages, &fresh);   // one entry per message; 0 = not stored (capacity)
        size_t not_stored = 0;
        for (uint64_t id : ids) if (!id) ++not_stored;
        if (not_stored) { bump(&RuntimeMetrics::vc_rejected, not_stored); trace("context", std::to_string(not_stored) + " earlier message(s) exceeded the storage limit"); }
        // the present request: everything after the agent's last assistant turn (new user text, tool results)
        size_t first = 0;
        for (size_t i = request.messages.size(); i-- > 0;) if (request.messages[i].role == "assistant") { first = i + 1; break; }
        for (size_t i = first; i < ids.size(); ++i) if (ids[i]) current_items.insert(ids[i]);
        for (auto it = request.messages.rbegin(); it != request.messages.rend(); ++it)
            if (it->role == "user") { vc_query = it->content; break; }
        // tool output of this request is exact evidence for the verifier even though the window pages it
        { std::vector<Message> scratch = request.messages; const size_t c = tools_->process(request.scope, scratch); if (c) bump(&RuntimeMetrics::tool_results_compacted, c); }
        sel.query = vc_query;
        sel.current_items = current_items;
        sel.requester = request.scope.security;
        const int64_t reserve_out = std::min<int64_t>(request.sampling.max_tokens, cfg_.virtual_context.physical_tokens / 4);
        const int64_t reserve_evidence = static_cast<int64_t>((cfg_.context.snippet_chars * cfg_.context.rag_top_k) / 4) + 300;
        sel.budget_tokens = std::max<int64_t>(512, cfg_.virtual_context.physical_tokens - reserve_out - reserve_evidence);
        reselect("");
        if (not_stored) convo.push_back({"system", "Some earlier messages of this conversation exceeded the storage limit and are not available to you. Do not guess their content.", ""});
    } else {
        const size_t compacted = tools_->process(request.scope, convo);
        if (compacted) { bump(&RuntimeMetrics::tool_results_compacted, compacted); trace("tool_output", std::to_string(compacted) + " compacted"); }
    }
    if (cfg_.memory_auto_store) {
        for (auto it = request.messages.rbegin(); it != request.messages.rend(); ++it)
            if (it->role == "user") { deps_.memory->observe_turn(request.scope.security, it->content); break; }
    }
    PreparedContext ctx = context_->prepare(request.scope, convo);
    if (ctx.retrieval_used) { bump(&RuntimeMetrics::retrievals); trace("retrieval", std::to_string(ctx.evidence.items.size()) + " evidence items"); }
    EvidenceSet evidence = ctx.evidence;
    const std::string scope_key = request.scope.security.to_string();

    // ---- 2. generation with live checks ----
    BoundedStreamBuffer buffer(StreamBufferConfig{cfg_.stream_max_hold_chars});
    generation::GenerationLoopDetector loop(cfg_.loop);
    generation::DynamicTemperatureConfig tcfg = cfg_.temperature;
    tcfg.base_temperature = request.sampling.temperature;   // only loops move it; the caller's value is the baseline
    generation::DynamicTemperatureController temp(tcfg);
    temp.reset(request.sampling.temperature);

    auto emit = [&](const std::string& s) {
        if (s.empty()) return;
        resp.text += s;
        if (sink) sink(s);
    };

    std::vector<std::string> notes;       // internal corrections for the next attempt (never shown to the caller)
    std::vector<Claim> all_claims;
    int regenerations = 0, loop_recoveries = 0, attempts = 0, produced = 0;
    bool retrieval_tried = false;
    double verify_ms = 0.0;
    resp.finish = FinishReason::kStop;

    struct Edit { size_t begin, end; std::string repl; };

    // Verifies the complete sentences in (checked, limit]. Returns true when generation must restart.
    auto check_text = [&](size_t limit) -> bool {
        const std::string text = buffer.text();
        const size_t from = buffer.checked();
        if (!cfg_.verification_enabled || limit <= from) { buffer.mark_checked(limit); return false; }
        std::vector<Claim> claims = detector_.scan(text, from, limit, request.scope);
        std::vector<Edit> edits;
        for (Claim& c : claims) {
            if (verify_ms > cfg_.verification_budget_ms) {   // resource signal: skip, never block the stream
                bump(&RuntimeMetrics::verification_skipped_low_risk);
                continue;
            }
            const auto tv = Clock::now();
            ClaimVerdict v = deps_.verifier->verify(c, evidence, "", scope_key);
            verify_ms += ms_since(tv);
            bump(&RuntimeMetrics::claims_checked);
            all_claims.push_back(c);

            RecoveryInput in;
            in.signals.verification_state = v.state;
            in.signals.evidence_state = evidence.state();
            in.verdict = &v;
            in.retrieval_tried = retrieval_tried;
            in.claim_retractable = c.begin >= buffer.released();
            in.regenerations_used = regenerations;
            in.loop_recoveries_used = loop_recoveries;
            RecoveryDecision d = decide_recovery(in, cfg_.recovery);
            if (d.action == RecoveryAction::kRetrieveInternally) {
                retrieval_tried = true;
                bump(&RuntimeMetrics::retrievals);
                EvidenceSet more = context_->retrieve(request.scope, c.text, cfg_.context.rag_top_k);
                for (auto& e : more.items) {
                    bool have = false;
                    for (const auto& x : evidence.items) if (x.id == e.id) { have = true; break; }
                    if (!have) evidence.items.push_back(e);
                }
                v = deps_.verifier->verify(c, evidence, "", scope_key);
                in.signals.verification_state = v.state;
                in.signals.evidence_state = evidence.state();
                in.verdict = &v;
                in.retrieval_tried = true;
                d = decide_recovery(in, cfg_.recovery);
                trace("retrieve", std::string("re-verified -> ") + to_string(v.state));
            }
            if (vc && (v.state == VerificationState::kVerified || v.state == VerificationState::kSupported))
                for (const auto& eid : v.evidence_ids)
                    if (eid.rfind("vc:", 0) == 0) vc->note_used_as_evidence(std::strtoull(eid.c_str() + 3, nullptr, 10));
            trace("claim", std::string(to_string(c.kind)) + " '" + c.value + "' -> " + to_string(v.state));
            if (v.state == VerificationState::kContradicted) {
                bump(&RuntimeMetrics::contradictions);
                if (!in.claim_retractable) bump(&RuntimeMetrics::late_contradictions);
            }
            if (v.state == VerificationState::kUnknown) bump(&RuntimeMetrics::unknown_delivered);

            if (d.action == RecoveryAction::kRegenerate) {
                ++regenerations;
                bump(&RuntimeMetrics::regenerations);
                notes.push_back(v.explanation + (v.corrected_value.empty() ? "" : " Correct value: " + v.corrected_value + "."));
                buffer.truncate_to(std::max(buffer.released(), c.begin));
                trace("recovery", "regenerate: " + d.reason);
                return true;
            }
            if (d.action == RecoveryAction::kHedgeClaim) {
                bool dup = false;
                for (const auto& e : edits) if (e.begin == c.begin && e.end == c.end) dup = true;
                if (dup) continue;
                bump(&RuntimeMetrics::hedged);
                std::string repl;
                if (cfg_.hedge_with_verified_value && v.state == VerificationState::kContradicted && !v.corrected_value.empty())
                    repl = substitute_value(text, c.begin, c.end, c.value, v.corrected_value);
                if (repl.empty()) repl = "I could not verify this: " + c.text;
                edits.push_back({c.begin, c.end, repl});
                trace("recovery", "hedge: " + d.reason);
            }
        }
        long delta = 0;
        std::sort(edits.begin(), edits.end(), [](const Edit& a, const Edit& b) { return a.begin > b.begin; });
        for (const auto& e : edits)
            if (buffer.replace_range(e.begin, e.end, e.repl)) delta += static_cast<long>(e.repl.size()) - static_cast<long>(e.end - e.begin);
        buffer.mark_checked(static_cast<size_t>(static_cast<long>(limit) + delta));
        return false;
    };

    while (true) {
        if (++attempts > cfg_.max_attempts) {
            bump(&RuntimeMetrics::safe_stops);
            trace("recovery", "attempt cap reached: stopping safely");
            break;
        }
        bump(&RuntimeMetrics::attempts);

        std::vector<Message> msgs = ctx.messages;
        if (!notes.empty()) {
            std::string n = "Verified facts for this answer (state them as plain fact, do not mention checking):";
            for (const auto& x : notes) n += "\n- " + x;
            msgs.push_back({"system", n, ""});
        }
        const std::vector<int32_t> ids = deps_.tokenizer->encode(deps_.prompt_template->render(msgs, buffer.text()));
        resp.prompt_tokens = static_cast<int>(ids.size());
        if (vc) {   // the physical KV holds this working set only; reuse exactly the matching prefix
            KvReusePlan kp = vc->plan_kv(KvIdentity{deps_.tokenizer->id(), deps_.prompt_template->version()}, ids);
            bump(&RuntimeMetrics::kv_reused_tokens, kp.reusable_prefix);
            bump(&RuntimeMetrics::kv_rebuilt_tokens, kp.to_prefill);
            trace("kv", "reuse " + std::to_string(kp.reusable_prefix) + ", prefill " + std::to_string(kp.to_prefill));
        }
        Sampling samp = request.sampling;
        double cur_temp = temp.current_temperature();
        samp.temperature = cur_temp;
        auto session = deps_.generator->start(ids, samp, request.scope.key());
        loop.reset();

        bool restart = false, ended = false;
        std::vector<size_t> tok_lens;    // text length per token of this attempt, for loop trimming
        GenToken tok;
        while (produced < request.sampling.max_tokens) {
            if (!session->next(tok)) { ended = true; break; }
            ++produced;
            if (cfg_.loop_protection) {
                auto lv = loop.feed_token(tok.id, tok.text);
                const double nt = temp.update(lv, tok.id);
                if (nt != cur_temp) { session->set_temperature(nt); cur_temp = nt; bump(&RuntimeMetrics::temperature_changes); trace("temperature", std::to_string(nt)); }
                if (to_loop_state(lv.confidence) == LoopState::kConfirmedLoop) {
                    RecoveryInput in;
                    in.signals.generation_loop_state = LoopState::kConfirmedLoop;
                    in.loop_recoveries_used = loop_recoveries;
                    RecoveryDecision d = decide_recovery(in, cfg_.recovery);
                    // drop the repeated span (unreleased part only), keep the good text
                    size_t trim = 0;
                    const size_t n = std::min(lv.trim_token_count, tok_lens.size());
                    for (size_t i = 0; i < n; ++i) trim += tok_lens[tok_lens.size() - 1 - i];
                    const size_t keep = buffer.text().size() > trim ? buffer.text().size() - trim : 0;
                    buffer.truncate_to(std::max(buffer.released(), keep));
                    if (d.action == RecoveryAction::kRegenerate) {
                        ++loop_recoveries;
                        bump(&RuntimeMetrics::loop_recoveries);
                        trace("recovery", "loop: " + lv.reason);
                        if (vc) {   // a loop may mean the model lost the thread: look for earlier context it should have had
                            const std::string& bt = buffer.text();
                            reselect(bt.size() > 240 ? bt.substr(bt.size() - 240) : bt);
                            ctx = context_->prepare(request.scope, convo);
                        }
                        restart = true;
                    } else {
                        bump(&RuntimeMetrics::safe_stops);
                        trace("recovery", "loop persisted: stopping safely");
                        resp.finish = FinishReason::kStop;
                        ended = true;
                    }
                    session->cancel();
                    break;
                }
            }
            tok_lens.push_back(tok.text.size());
            emit(buffer.push(tok.text));
            // stop strings
            bool stopped = false;
            for (const auto& st : request.sampling.stop) {
                if (st.empty()) continue;
                const size_t from = buffer.text().size() > tok.text.size() + st.size() ? buffer.text().size() - tok.text.size() - st.size() : 0;
                const size_t p = buffer.text().find(st, std::max(from, buffer.released()));
                if (p != std::string::npos && buffer.truncate_to(p)) { stopped = true; break; }
            }
            if (stopped) { session->cancel(); ended = true; break; }
            const size_t limit = ClaimDetector::complete_sentence_end(buffer.text(), buffer.checked());
            if (limit > buffer.checked()) {
                if (check_text(limit)) { restart = true; session->cancel(); break; }
                emit(buffer.push(""));
            }
        }
        if (restart) continue;
        if (!ended && produced >= request.sampling.max_tokens) resp.finish = FinishReason::kLength;
        // end of the answer: the unfinished tail is a sentence too
        if (buffer.checked() < buffer.text().size()) {
            if (check_text(buffer.text().size())) continue;
        }
        break;
    }

    // ---- 3. final consistency pass over the whole answer ----
    {
        auto bad = find_inconsistent_claims(all_claims);
        if (!bad.empty()) {
            bump(&RuntimeMetrics::contradictions);
            trace("consistency", std::to_string(bad.size()) + " self-contradicting claim pair(s)");
        }
    }
    emit(buffer.flush());
    resp.completion_tokens = produced;
    if (vc && !resp.text.empty()) {   // the answer is part of the logical context from now on
        NewItem a;
        a.kind = ItemKind::kAssistant; a.role = "assistant"; a.content = resp.text;
        if (!vc->append(a)) bump(&RuntimeMetrics::vc_rejected);
    }
    if (vc && !deps_.contexts->persist(vc)) trace("context", "could not persist the context manifest");
    request_evidence_->drop(request.scope.key());
    trace("done", std::string(to_string(resp.finish)));
    return resp;
}

} // namespace strata::rt
