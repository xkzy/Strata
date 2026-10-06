// src/context/virtual_context.cpp - Virtual Context Manager Implementation
#include "strata/context/virtual_context.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace strata::context {

VirtualContextManager::VirtualContextManager(int64_t physical_context_limit) {
    TokenBudgetConfig cfg;
    cfg.physical_context_limit = physical_context_limit;
    budget_mgr_.set_config(cfg);

    scorer_ = std::make_shared<CodeAwareImportanceScorer>();
    index_ = std::make_shared<HierarchicalBM25Index>();
    memory_mgr_ = std::make_shared<HierarchicalMemoryManager>();
    compactor_ = std::make_shared<ContextCompactor>(scorer_, index_, memory_mgr_);
}

VirtualContextManager::~VirtualContextManager() = default;

int64_t VirtualContextManager::estimate_tokens(const std::string& text) const {
    if (text.empty()) return 0;
    // Fast approx: ~4 characters per token
    return std::max<int64_t>(1, static_cast<int64_t>(text.size() / 4));
}

double VirtualContextManager::current_time_seconds() const {
    auto now = std::chrono::system_clock::now().time_since_epoch();
    return std::chrono::duration<double>(now).count();
}

int64_t VirtualContextManager::append_system_prompt(const std::string& content, int64_t approx_tokens) {
    int64_t id = next_item_id_++;
    int64_t tokens = approx_tokens > 0 ? approx_tokens : estimate_tokens(content);
    system_prompt_item_ = std::make_shared<ContextItem>(id, ContextItemType::kSystemPrompt, content,
                                                        tokens, current_time_seconds());
    memory_mgr_->add_raw_item(system_prompt_item_);
    stats_.virtual_context_total_tokens += tokens;
    stats_.total_items_tracked++;
    return id;
}

int64_t VirtualContextManager::append_user_message(const std::string& content, int64_t approx_tokens) {
    int64_t id = next_item_id_++;
    int64_t tokens = approx_tokens > 0 ? approx_tokens : estimate_tokens(content);
    auto item = std::make_shared<ContextItem>(id, ContextItemType::kUserMessage, content,
                                              tokens, current_time_seconds());
    active_items_.push_back(item);
    memory_mgr_->add_raw_item(item);

    stats_.virtual_context_total_tokens += tokens;
    stats_.physical_active_tokens += tokens;
    stats_.total_items_tracked++;

    // Check compaction
    compactor_->compact_if_needed(active_items_, budget_mgr_, content, current_time_seconds());
    return id;
}

int64_t VirtualContextManager::append_assistant_message(const std::string& content, int64_t approx_tokens) {
    int64_t id = next_item_id_++;
    int64_t tokens = approx_tokens > 0 ? approx_tokens : estimate_tokens(content);
    auto item = std::make_shared<ContextItem>(id, ContextItemType::kAssistantMessage, content,
                                              tokens, current_time_seconds());
    active_items_.push_back(item);
    memory_mgr_->add_raw_item(item);

    stats_.virtual_context_total_tokens += tokens;
    stats_.physical_active_tokens += tokens;
    stats_.total_items_tracked++;

    return id;
}

int64_t VirtualContextManager::append_tool_result(const std::string& tool_name,
                                                 const std::string& command_or_call,
                                                 const std::string& output,
                                                 int exit_code,
                                                 int64_t approx_tokens) {
    int64_t id = next_item_id_++;
    int64_t tokens = approx_tokens > 0 ? approx_tokens : estimate_tokens(output);
    auto item = std::make_shared<ContextItem>(id, ContextItemType::kToolOutput, output,
                                              tokens, current_time_seconds());
    item->mutable_metadata().command = command_or_call;
    item->mutable_metadata().exit_code = exit_code;
    item->mutable_metadata().tags.push_back(tool_name);

    // If large tool output (>1500 tokens), index immediately and compress active representation
    if (tokens > 1500) {
        index_->index_item(*item);
        item->set_state(ResidencyState::kCompressed);
        item->set_summary_l1("[Tool " + tool_name + " executed: " + command_or_call +
                             " | Exit: " + std::to_string(exit_code) + " | Output: " +
                             output.substr(0, 300) + "... (" + std::to_string(tokens) + " tokens indexed)]");
        item->set_token_count(100);
        stats_.indexed_tokens += tokens;
        stats_.compressed_tokens += 100;
    }

    active_items_.push_back(item);
    memory_mgr_->add_raw_item(item);

    stats_.virtual_context_total_tokens += tokens;
    stats_.physical_active_tokens += item->token_count();
    stats_.total_items_tracked++;

    return id;
}

int64_t VirtualContextManager::append_file_content(const std::string& filename,
                                                  const std::string& content,
                                                  const std::vector<std::string>& symbols,
                                                  int64_t approx_tokens) {
    int64_t id = next_item_id_++;
    int64_t tokens = approx_tokens > 0 ? approx_tokens : estimate_tokens(content);
    auto item = std::make_shared<ContextItem>(id, ContextItemType::kFileContent, content,
                                              tokens, current_time_seconds());
    item->mutable_metadata().filename = filename;
    item->mutable_metadata().tags = symbols;

    // Index file content immediately into RAG
    index_->index_item(*item);
    memory_mgr_->add_raw_item(item);

    // Files are indexed into searchable memory
    item->set_state(ResidencyState::kIndexed);
    item->set_token_count(0);

    stats_.virtual_context_total_tokens += tokens;
    stats_.indexed_tokens += tokens;
    stats_.total_items_tracked++;

    return id;
}

int64_t VirtualContextManager::append_code_diff(const std::string& filename,
                                               const std::string& diff,
                                               const std::string& commit_hash,
                                               int64_t approx_tokens) {
    int64_t id = next_item_id_++;
    int64_t tokens = approx_tokens > 0 ? approx_tokens : estimate_tokens(diff);
    auto item = std::make_shared<ContextItem>(id, ContextItemType::kCodeChange, diff,
                                              tokens, current_time_seconds());
    item->mutable_metadata().filename = filename;
    item->mutable_metadata().commit_hash = commit_hash;

    active_items_.push_back(item);
    memory_mgr_->add_raw_item(item);
    index_->index_item(*item);

    stats_.virtual_context_total_tokens += tokens;
    stats_.physical_active_tokens += tokens;
    stats_.total_items_tracked++;

    return id;
}

void VirtualContextManager::update_working_memory(const std::string& working_state, int64_t approx_tokens) {
    int64_t tokens = approx_tokens > 0 ? approx_tokens : estimate_tokens(working_state);
    if (!working_memory_item_) {
        int64_t id = next_item_id_++;
        working_memory_item_ = std::make_shared<ContextItem>(id, ContextItemType::kWorkingMemory,
                                                             working_state, tokens, current_time_seconds());
        memory_mgr_->add_raw_item(working_memory_item_);
    } else {
        working_memory_item_ = std::make_shared<ContextItem>(working_memory_item_->id(),
                                                             ContextItemType::kWorkingMemory,
                                                             working_state, tokens, current_time_seconds());
    }
    memory_mgr_->set_global_state(working_state);
    stats_.working_memory_tokens = tokens;
}

std::vector<SearchResult> VirtualContextManager::retrieve_relevant(const std::string& query, size_t top_k) {
    auto t0 = std::chrono::high_resolution_clock::now();

    RetrievalQuery rq;
    rq.text = query;
    rq.top_k = top_k;
    rq.hierarchical_expand = true;

    auto results = index_->search(rq);

    auto t1 = std::chrono::high_resolution_clock::now();
    stats_.last_retrieval_latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    stats_.retrieval_queries++;
    if (!results.empty()) stats_.retrieval_hits += results.size();

    return results;
}

CompactionResult VirtualContextManager::trigger_compaction(const std::string& current_query) {
    auto res = compactor_->compact_if_needed(active_items_, budget_mgr_, current_query, current_time_seconds());
    if (res.triggered) {
        stats_.compaction_events++;
        stats_.total_tokens_freed_by_compaction += res.tokens_freed;
        stats_.last_compaction_latency_ms = res.duration_ms;
    }
    return res;
}

PromptPayload VirtualContextManager::select_and_assemble_context(const std::string& current_query,
                                                                 const std::string& system_override) {
    // 1. Compact active context if needed before assembly
    trigger_compaction(current_query);

    // 2. Retrieve relevant historical knowledge
    auto retrieved = retrieve_relevant(current_query, 4);

    int64_t req_sys = system_prompt_item_ ? system_prompt_item_->token_count() : estimate_tokens(system_override);
    int64_t req_wm = working_memory_item_ ? working_memory_item_->token_count() : 0;

    int64_t req_conv = 0;
    for (const auto& it : active_items_) {
        if (it) req_conv += it->token_count();
    }

    int64_t req_retrieval = 0;
    for (const auto& r : retrieved) {
        req_retrieval += r.token_count;
    }

    std::string sess_summary = memory_mgr_->get_session_summary();
    int64_t req_summary = estimate_tokens(sess_summary);

    // 3. Compute exact token allocations
    TokenAllocation alloc = budget_mgr_.compute_allocation(req_sys, req_conv, req_wm, req_retrieval, req_summary);

    // 4. Assemble final physical prompt payload
    PromptPayload payload;
    std::ostringstream ss;

    // Header / System Prompt
    if (!system_override.empty()) {
        ss << "<|system|>\n" << system_override << "\n";
        payload.assembled_prompt += ss.str();
    } else if (system_prompt_item_) {
        ss << "<|system|>\n" << system_prompt_item_->raw_content() << "\n";
    }

    // High-Level Summary Context (if available and within allocation)
    if (alloc.summary_tokens > 0 && !sess_summary.empty() && sess_summary != "Initial session state.") {
        ss << "\n[Historical Context Summary]:\n" << sess_summary << "\n";
        payload.has_summary = true;
    }

    // Working Memory / Task State
    if (alloc.working_memory_tokens > 0 && working_memory_item_) {
        ss << "\n[Current Working State / Task]:\n" << working_memory_item_->raw_content() << "\n";
        payload.has_working_memory = true;
    }

    // Retrieved Historical Knowledge (RAG)
    if (alloc.retrieval_tokens > 0 && !retrieved.empty()) {
        ss << "\n[Retrieved Context & Source Excerpts]:\n";
        int64_t cur_ret_tokens = 0;
        for (const auto& r : retrieved) {
            if (cur_ret_tokens + r.token_count <= alloc.retrieval_tokens) {
                if (!r.filename.empty()) {
                    ss << "--- File: " << r.filename << " ---\n";
                }
                ss << r.matched_content << "\n";
                cur_ret_tokens += r.token_count;
                payload.retrieved_chunks_included++;
            }
        }
    }

    // Active Recent Conversation & Tool Outputs
    ss << "\n[Active Conversation]:\n";
    int64_t cur_conv_tokens = 0;
    // Iterate in chronological order
    for (const auto& it : active_items_) {
        if (!it) continue;
        int64_t item_tok = it->token_count();
        if (cur_conv_tokens + item_tok <= alloc.active_conv_tokens || it->type() == ContextItemType::kUserMessage) {
            if (it->type() == ContextItemType::kUserMessage) {
                ss << "User: " << it->get_effective_content() << "\n";
            } else if (it->type() == ContextItemType::kAssistantMessage) {
                ss << "Assistant: " << it->get_effective_content() << "\n";
            } else if (it->type() == ContextItemType::kToolOutput) {
                ss << "Tool Result: " << it->get_effective_content() << "\n";
            } else {
                ss << it->get_effective_content() << "\n";
            }
            cur_conv_tokens += item_tok;
            payload.active_items_included++;
        }
    }

    payload.assembled_prompt = ss.str();
    payload.total_prompt_tokens = estimate_tokens(payload.assembled_prompt);

    return payload;
}

VirtualContextStats VirtualContextManager::get_stats() const {
    stats_.physical_active_tokens = 0;
    for (const auto& it : active_items_) {
        if (it && it->state() == ResidencyState::kActive) {
            stats_.physical_active_tokens += it->token_count();
        }
    }
    stats_.indexed_tokens = index_->total_indexed_tokens();
    return stats_;
}

std::string VirtualContextManager::print_diagnostics() const {
    auto st = get_stats();
    std::ostringstream ss;
    ss << "================ STRATA VIRTUAL CONTEXT DIAGNOSTICS ================\n";
    ss << "Virtual Context Total:   " << std::setw(10) << st.virtual_context_total_tokens << " tokens\n";
    ss << "Physical Active Window:  " << std::setw(10) << st.physical_active_tokens << " tokens (Limit: "
       << budget_mgr_.config().physical_context_limit << ")\n";
    ss << "Working Memory:          " << std::setw(10) << st.working_memory_tokens << " tokens\n";
    ss << "Compressed Context:      " << std::setw(10) << st.compressed_tokens << " tokens\n";
    ss << "Indexed RAG Knowledge:   " << std::setw(10) << st.indexed_tokens << " tokens\n";
    ss << "Compaction Events:       " << std::setw(10) << st.compaction_events
       << " | Freed: " << st.total_tokens_freed_by_compaction << " tokens\n";
    ss << "Retrieval Queries:       " << std::setw(10) << st.retrieval_queries
       << " | Hits: " << st.retrieval_hits << " (Avg Latency: "
       << std::fixed << std::setprecision(2) << st.last_retrieval_latency_ms << " ms)\n";
    ss << "====================================================================\n";
    return ss.str();
}

} // namespace strata::context
