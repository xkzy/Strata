// include/strata/context/context_item.hpp - Context Item & Lifecycle State
//
// Represents atomic units of context (messages, tool outputs, files, code diffs,
// summaries, working memory) across the virtual context hierarchy.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::context {

enum class ContextItemType {
    kSystemPrompt = 0,
    kUserMessage,
    kAssistantMessage,
    kToolCall,
    kToolOutput,
    kFileContent,
    kCommandOutput,
    kCodeChange,
    kErrorLog,
    kTestResult,
    kSummaryL1,
    kSummaryL2,
    kSummaryL3,
    kWorkingMemory
};

enum class ResidencyState {
    kActive = 0,    // Currently resident in the model's physical context window
    kWorking,       // In high-priority working memory
    kCompressed,    // Compacted/summarized in active context; full original indexed
    kIndexed,       // Stored and searchable in the hierarchical retrieval index
    kArchived       // Cold storage in persistent immutable archive
};

struct ContextMetadata {
    std::string filename;
    std::string symbol_name;
    std::string command;
    int exit_code = 0;
    std::string commit_hash;
    std::string task_id;
    std::vector<std::string> tags;
    std::unordered_map<std::string, std::string> custom_attributes;
};

class ContextItem {
public:
    ContextItem(int64_t id, ContextItemType type, const std::string& raw_content,
                int64_t token_count, double timestamp_sec);

    int64_t id() const { return id_; }
    ContextItemType type() const { return type_; }
    ResidencyState state() const { return state_; }
    void set_state(ResidencyState state) { state_ = state; }

    int64_t token_count() const { return token_count_; }
    void set_token_count(int64_t tc) { token_count_ = tc; }

    double timestamp() const { return timestamp_; }
    double last_accessed() const { return last_accessed_; }
    void record_access(double now_sec) {
        access_count_++;
        last_accessed_ = now_sec;
    }
    uint64_t access_count() const { return access_count_; }

    // Immutable Source of Truth (L0)
    const std::string& raw_content() const { return raw_content_; }

    // Hierarchical Summaries (L1 - L3)
    const std::string& summary_l1() const { return summary_l1_; }
    void set_summary_l1(const std::string& s) { summary_l1_ = s; }

    const std::string& summary_l2() const { return summary_l2_; }
    void set_summary_l2(const std::string& s) { summary_l2_ = s; }

    const std::string& summary_l3() const { return summary_l3_; }
    void set_summary_l3(const std::string& s) { summary_l3_ = s; }

    // Importance & Relevance Scores
    double importance_score() const { return importance_score_; }
    void set_importance_score(double s) { importance_score_ = s; }

    // Metadata & References
    const ContextMetadata& metadata() const { return metadata_; }
    ContextMetadata& mutable_metadata() { return metadata_; }

    const std::vector<int64_t>& references() const { return references_; }
    void add_reference(int64_t ref_id) { references_.push_back(ref_id); }

    int64_t parent_id() const { return parent_id_; }
    void set_parent_id(int64_t pid) { parent_id_ = pid; }

    // Effective content based on current residency state
    std::string get_effective_content() const;

private:
    int64_t id_;
    ContextItemType type_;
    ResidencyState state_ = ResidencyState::kActive;
    int64_t token_count_ = 0;
    double timestamp_ = 0.0;
    double last_accessed_ = 0.0;
    uint64_t access_count_ = 0;

    std::string raw_content_;    // Immutable L0 raw source
    std::string summary_l1_;     // L1 Chunk summary
    std::string summary_l2_;     // L2 Section summary
    std::string summary_l3_;     // L3 Session summary

    double importance_score_ = 1.0;
    int64_t parent_id_ = -1;
    std::vector<int64_t> references_;
    ContextMetadata metadata_;
};

} // namespace strata::context
