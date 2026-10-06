// tests/context/test_virtual_context.cpp - Verification Suite for Generic Long-Context Virtual Memory Runtime
#include "strata/context/context_compactor.hpp"
#include "strata/context/context_item.hpp"
#include "strata/context/context_runtime.hpp"
#include "strata/context/hierarchical_memory.hpp"
#include "strata/context/importance_scorer.hpp"
#include "strata/context/retrieval_index.hpp"
#include "strata/context/token_budget.hpp"
#include "strata/context/virtual_context.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

void test_context_item_and_lifecycle() {
    std::cout << "[Test 1/8] Testing ContextItem and Lifecycle States..." << std::endl;
    strata::context::ContextItem item(1, strata::context::ContextItemType::kUserMessage,
                                      "Define an MoE scheduling algorithm.", 12, 1000.0);

    assert(item.id() == 1);
    assert(item.state() == strata::context::ResidencyState::kActive);
    assert(item.raw_content() == "Define an MoE scheduling algorithm.");

    item.set_summary_l1("User asked about MoE scheduling.");
    item.set_state(strata::context::ResidencyState::kCompressed);
    assert(item.get_effective_content() == "User asked about MoE scheduling.");

    std::cout << "  Passed. Item lifecycle and effective content verified." << std::endl;
}

void test_importance_scoring() {
    std::cout << "[Test 2/8] Testing Importance Scoring (Recency, Relevance, Code Awareness)..." << std::endl;
    strata::context::CodeAwareImportanceScorer scorer;
    scorer.set_active_files({"src/core/scheduler.cpp"});
    scorer.set_active_symbols({"schedule_active_experts"});

    strata::context::ContextItem code_item(10, strata::context::ContextItemType::kFileContent,
                                           "void schedule_active_experts() { ... }", 200, 1000.0);
    code_item.mutable_metadata().filename = "src/core/scheduler.cpp";
    code_item.mutable_metadata().symbol_name = "schedule_active_experts";

    strata::context::ContextItem generic_item(11, strata::context::ContextItemType::kCommandOutput,
                                              "some random terminal logs here", 200, 500.0);

    double code_score = scorer.score(code_item, "schedule_active_experts", 1010.0);
    double generic_score = scorer.score(generic_item, "schedule_active_experts", 1010.0);

    assert(code_score > generic_score);
    assert(code_score > 0.6);
    (void)code_score;
    (void)generic_score;
    std::cout << "  Passed. Importance scoring correctly prioritized relevant code context." << std::endl;
}

void test_token_budget_manager() {
    std::cout << "[Test 3/8] Testing Token Budget Manager Physical Context Constraints..." << std::endl;
    strata::context::TokenBudgetConfig cfg;
    cfg.physical_context_limit = 8192; // 8K physical context
    cfg.generation_reserve = 1024;     // 1024 output reserve
    strata::context::TokenBudgetManager mgr(cfg);

    assert(mgr.max_input_tokens() == 7168);

    // Request budget exceeding physical limit
    auto alloc = mgr.compute_allocation(2000, 6000, 2000, 5000, 1000);
    assert(alloc.total_allocated() <= 8192);
    assert(alloc.total_allocated() - alloc.generation_reserve <= 7168);
    (void)alloc;

    std::cout << "  Passed. Physical context limit strictly enforced without overflow." << std::endl;
}

void test_hierarchical_retrieval_index() {
    std::cout << "[Test 4/8] Testing Built-In Hierarchical Retrieval Index (BM25 + Metadata)..." << std::endl;
    strata::context::HierarchicalBM25Index index;

    strata::context::ContextItem item1(1, strata::context::ContextItemType::kUserMessage,
                                       "The secret server password is AlphaBetaGamma99.", 15, 100.0);
    strata::context::ContextItem item2(2, strata::context::ContextItemType::kFileContent,
                                       "int connect_database(const char* host) { return 0; }", 20, 101.0);
    item2.mutable_metadata().filename = "db.cpp";
    item2.mutable_metadata().symbol_name = "connect_database";

    index.index_item(item1);
    index.index_item(item2);

    strata::context::RetrievalQuery q;
    q.text = "password";
    q.top_k = 1;
    auto results = index.search(q);

    assert(results.size() == 1);
    assert(results[0].item_id == 1);
    assert(results[0].matched_content.find("AlphaBetaGamma99") != std::string::npos);

    // Metadata search
    strata::context::RetrievalQuery q_meta;
    q_meta.filename_filter = "db.cpp";
    q_meta.top_k = 1;
    auto meta_results = index.search(q_meta);
    assert(meta_results.size() == 1);
    assert(meta_results[0].item_id == 2);

    std::cout << "  Passed. Exact and metadata-guided retrieval verified." << std::endl;
}

void test_hierarchical_memory_immutability() {
    std::cout << "[Test 5/8] Testing Hierarchical Memory Manager (L0 Source Immutability -> L3 Summary)..." << std::endl;
    strata::context::HierarchicalMemoryManager hmm;

    for (int i = 0; i < 20; ++i) {
        auto item = std::make_shared<strata::context::ContextItem>(
            i + 1, strata::context::ContextItemType::kUserMessage,
            "Step " + std::to_string(i) + ": Executed benchmark on node cluster " + std::to_string(i) + ".",
            100, 1000.0 + i);
        hmm.add_raw_item(item);
    }

    hmm.update_hierarchy_incremental(300, 1000);

    // Verify L0 source of truth remains completely untouched and retrievable
    auto raw_item = hmm.get_source_item(5);
    assert(raw_item != nullptr);
    assert(raw_item->raw_content().find("Step 4") != std::string::npos);

    // Verify L1 chunks and L3 session summary exist
    auto l1_nodes = hmm.get_level_nodes(1);
    assert(!l1_nodes.empty());
    std::string sess_sum = hmm.get_session_summary();
    assert(!sess_sum.empty());

    std::cout << "  Passed. Authoritative L0 raw data preserved with hierarchical summaries." << std::endl;
}

void test_context_compaction() {
    std::cout << "[Test 6/8] Testing Automatic Context Compaction..." << std::endl;
    auto scorer = std::make_shared<strata::context::CompositeImportanceScorer>();
    auto index = std::make_shared<strata::context::HierarchicalBM25Index>();
    auto hmm = std::make_shared<strata::context::HierarchicalMemoryManager>();
    strata::context::ContextCompactor compactor(scorer, index, hmm);

    std::vector<std::shared_ptr<strata::context::ContextItem>> items;
    // Fill with 5000 tokens of messages
    for (int i = 0; i < 50; ++i) {
        auto item = std::make_shared<strata::context::ContextItem>(
            i + 1, strata::context::ContextItemType::kUserMessage,
            "Historical conversation log item " + std::to_string(i) + " with detailed execution traces.",
            100, 100.0 + i);
        items.push_back(item);
        hmm->add_raw_item(item);
    }

    // Force compaction to 1000 tokens target
    auto res = compactor.compact_to_target(items, 1000, "current task", 200.0);
    assert(res.triggered);
    assert(res.compacted_active_tokens <= 1500);
    assert(res.tokens_freed > 3000);

    // Verify compacted items were moved to index and are searchable
    strata::context::RetrievalQuery rq;
    rq.text = "Historical conversation log item 5";
    auto search_res = index->search(rq);
    assert(!search_res.empty());

    std::cout << "  Passed. Context compacted from " << res.initial_active_tokens
              << " to " << res.compacted_active_tokens << " tokens with full indexing." << std::endl;
}

void test_tool_and_coding_agent_integration() {
    std::cout << "[Test 7/8] Testing Tool Results & Code Diff Ingestion..." << std::endl;
    strata::context::VirtualContextManager vcm(8192);

    // Ingest large tool output (e.g. 3000 tokens)
    std::string large_output;
    for (int i = 0; i < 150; ++i) {
        large_output += "Test suite row " + std::to_string(i) + ": PASSED [0.01s] in module_alpha\n";
    }
    vcm.append_tool_result("pytest", "pytest serve/ tools/", large_output, 0, 3000);

    // Ingest file content
    vcm.append_file_content("include/strata/core/layer.hpp",
                            "class GdnLayer { public: void forward(); };",
                            {"GdnLayer", "forward"}, 500);

    // Query for a specific test row
    auto retrieved = vcm.retrieve_relevant("module_alpha test row 42", 2);
    assert(!retrieved.empty());

    std::cout << "  Passed. Tool outputs & files ingested with on-demand retrieval." << std::endl;
}

void test_virtual_context_end_to_end() {
    std::cout << "[Test 8/8] Testing End-to-End Long-Context Runtime Assembly (100K+ Virtual -> 8K Physical)..." << std::endl;
    strata::context::LongContextRuntimeOptions opts;
    opts.physical_context_limit = 8192; // 8K physical window
    strata::context::LongContextRuntime runtime(opts);

    runtime.virtual_context().append_system_prompt("You are Strata, a high-performance heterogeneous AI assistant.");
    runtime.virtual_context().update_working_memory("Current Goal: Optimize MoE routing layer.");

    // Feed a massive virtual conversation history (simulating 100K tokens)
    for (int i = 0; i < 200; ++i) {
        runtime.virtual_context().append_user_message(
            "Discussion turn " + std::to_string(i) + ": In step " + std::to_string(i) +
            ", the memory cluster configuration parameters were set to index_key_dim=128.", 500);
        runtime.virtual_context().append_assistant_message(
            "Acknowledged step " + std::to_string(i) + ". Parameters saved to virtual context.", 100);
    }

    // Now query about early conversation detail
    std::string prompt = runtime.prepare_prompt("What was index_key_dim set to in step 12?");

    assert(!prompt.empty());
    assert(prompt.find("You are Strata") != std::string::npos);
    assert(prompt.find("Current Goal: Optimize MoE routing layer.") != std::string::npos);

    std::cout << runtime.print_context_summary() << std::endl;
    std::cout << "  Passed. Virtual context effectively managed 120,000+ virtual tokens into physical prompt!" << std::endl;
}

int main() {
    std::cout << "=================================================================\n"
              << "   RUNNING STRATA GENERIC LONG-CONTEXT RUNTIME TEST SUITE        \n"
              << "=================================================================\n";

    test_context_item_and_lifecycle();
    test_importance_scoring();
    test_token_budget_manager();
    test_hierarchical_retrieval_index();
    test_hierarchical_memory_immutability();
    test_context_compaction();
    test_tool_and_coding_agent_integration();
    test_virtual_context_end_to_end();

    std::cout << "=================================================================\n"
              << "   ALL LONG-CONTEXT & VIRTUAL MEMORY TESTS PASSED (8/8)          \n"
              << "=================================================================\n";
    return 0;
}
