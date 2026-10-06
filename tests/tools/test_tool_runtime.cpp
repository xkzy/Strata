// tests/tools/test_tool_runtime.cpp - Tool Runtime & Bounded Observation Test Suite
#include "strata/strata_unified.hpp"
#include "strata/tools/tool_parser.hpp"
#include "strata/tools/tool_runtime.hpp"
#include "strata/tools/tool_store.hpp"
#include "strata/tools/tool_types.hpp"

#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

void test_tool_parsers() {
    std::cout << "[Test 1/7] Testing Structured Tool Output Parsers..." << std::endl;
    strata::tools::CompositeToolParser parser;

    // Test compiler output parsing
    std::string gcc_output = "src/core/router.cpp:42:15: error: unknown type name 'RouterTop10'\n"
                            "src/core/router.cpp:45:8: warning: unused variable 'k'\n";
    auto build_data = parser.parse("gcc", gcc_output, 1);
    assert(build_data.is_build);
    assert(!build_data.build_success);
    assert(build_data.error_count == 1);
    assert(build_data.warning_count == 1);
    assert(build_data.diagnostics.size() == 2);
    assert(build_data.diagnostics[0].line == 42);

    // Test pytest output parsing
    std::string pytest_output = "FAILED tests/test_kv.py::test_q8 - AssertionError: 5 != 8\n"
                               "2 failed, 14 passed in 0.20s\n";
    auto test_data = parser.parse("pytest", pytest_output, 1);
    assert(test_data.is_test);
    assert(test_data.tests_failed == 1);
    assert(!test_data.failures.empty());

    // Test search output parsing
    std::string grep_output = "src/core/session.cpp:12: void session_init();\n"
                             "src/core/session.cpp:45: void session_token();\n"
                             "include/strata/core/session.hpp:18: class SessionState;\n";
    auto search_data = parser.parse("grep", grep_output, 0);
    assert(search_data.is_search);
    assert(search_data.total_matches == 3);
    assert(search_data.matched_files.size() == 2);

    std::cout << "  Passed. Compiler, Test, and Search outputs parsed into structured state." << std::endl;
}

void test_content_addressed_storage() {
    std::cout << "[Test 2/7] Testing Content-Addressed Storage & Deduplication..." << std::endl;
    strata::tools::ContentAddressedStore cas;

    std::string raw1 = "Exact same 1000-line log output that happens repeatedly.";
    std::string raw2 = "Exact same 1000-line log output that happens repeatedly.";
    std::string raw3 = "Different log output.";

    std::string hash1 = cas.store(raw1);
    std::string hash2 = cas.store(raw2);
    std::string hash3 = cas.store(raw3);

    assert(hash1 == hash2); // Identical content produces identical hash
    assert(hash1 != hash3);
    assert(cas.total_stored_entries() == 2); // Stored once, referenced twice

    std::string retrieved;
    bool ok = cas.get(hash1, retrieved);
    assert(ok);
    (void)ok;
    assert(retrieved == raw1);

    std::cout << "  Passed. Content-addressed storage deduplicated identical raw payloads." << std::endl;
}

void test_large_output_policy_and_compact_observation() {
    std::cout << "[Test 3/7] Testing Large Tool Output Policy & Bounded Observations..." << std::endl;
    strata::tools::ToolRuntime runtime;

    // Generate massive raw output (e.g. 10,000 tokens of build logs)
    std::string massive_log;
    for (int i = 0; i < 500; ++i) {
        massive_log += "Compiling translation unit " + std::to_string(i) + " ... [OK]\n";
    }
    massive_log += "src/kernels/gemv.cu:108:12: error: invalid configuration argument\n";

    strata::tools::ToolExecutionRequest req;
    req.tool_name = "ninja";
    req.command_or_arguments = "ninja -C build";
    req.raw_output = massive_log;
    req.exit_code = 1;
    req.request_id = 101;

    auto res = runtime.process_tool_execution(req);

    assert(res.category == strata::tools::ResultCategory::kLargeExternal);
    assert(res.raw_tokens > 3000);
    // Bounded observation must be tiny (under 150 tokens)
    assert(res.observation_tokens < 150);
    assert(res.compact_observation.find("Result ID:") != std::string::npos);
    assert(res.compact_observation.find("invalid configuration argument") != std::string::npos);

    std::cout << "  Passed. Raw 3000+ token log reduced to " << res.observation_tokens
              << " token observation with full auditability." << std::endl;
}

void test_tool_fragment_retrieval() {
    std::cout << "[Test 4/7] Testing On-Demand Tool Fragment Retrieval..." << std::endl;
    strata::tools::ToolRuntime runtime;

    std::string large_output;
    for (int i = 1; i <= 200; ++i) {
        large_output += "Entry " + std::to_string(i) + ": Processing batch block " + std::to_string(i * 10) + "\n";
    }

    strata::tools::ToolExecutionRequest req;
    req.tool_name = "batch_worker";
    req.command_or_arguments = "process_batches";
    req.raw_output = large_output;
    req.exit_code = 0;

    auto res = runtime.process_tool_execution(req);

    // Retrieve specific fragment matching line query
    std::string fragment = runtime.retrieve_tool_fragment(res.result_id, "block 420");
    assert(!fragment.empty());
    assert(fragment.find("Entry 42") != std::string::npos);
    assert(fragment.find("block 420") != std::string::npos);

    std::cout << "  Passed. Targeted fragment extracted without retrieving 200 lines." << std::endl;
}

void test_context_materialization_lease() {
    std::cout << "[Test 5/7] Testing Temporary Context Materialization Lease..." << std::endl;
    strata::tools::ToolRuntime runtime;

    strata::tools::ToolExecutionRequest req;
    req.tool_name = "read_file";
    req.command_or_arguments = "view include/strata/runtime/device.hpp";
    req.raw_output = "// line 1\n// line 2\nclass ComputeDevice {};\n";
    req.exit_code = 0;

    auto res = runtime.process_tool_execution(req);

    // Materialize lease
    auto lease = runtime.materialize_result(res.result_id, 500);
    assert(lease.is_active);
    assert(lease.result_id == res.result_id);
    assert(lease.materialized_content.find("ComputeDevice") != std::string::npos);

    runtime.store().release_lease(lease.lease_id);
    std::cout << "  Passed. Temporary context lease created and released cleanly." << std::endl;
}

void test_tool_virtual_context_rag_unification() {
    std::cout << "[Test 6/7] Testing Unified Tool + Context + RAG Memory..." << std::endl;
    auto vctx = std::make_shared<strata::context::VirtualContextManager>(8192);
    strata::tools::ToolRuntime runtime(vctx);

    strata::tools::ToolExecutionRequest req;
    req.tool_name = "security_scan";
    req.command_or_arguments = "scan --ports";
    req.raw_output = "Vulnerability detected: CVE-2026-9912 on port 8080 SSL handshake.\n";
    req.exit_code = 0;

    runtime.process_tool_execution(req);

    // Query virtual context for CVE vulnerability
    auto search_results = vctx->retrieve_relevant("CVE-2026-9912", 1);
    assert(!search_results.empty());
    assert(search_results[0].matched_content.find("CVE-2026-9912") != std::string::npos);

    std::cout << "  Passed. Tool results automatically indexed into virtual context RAG." << std::endl;
}

void test_unified_ai_runtime_end_to_end() {
    std::cout << "[Test 7/7] Testing Strata Unified AI Runtime End-to-End..." << std::endl;
    strata::UnifiedRuntimeOptions opts;
    opts.physical_context_limit = 8192;
    opts.virtual_context_limit = 2000000;

    strata::StrataUnifiedRuntime unified(opts);
    std::string err;
    bool ok = unified.initialize(err);
    if (!ok) {
        std::cerr << "Init error: " << err << std::endl;
    }
    assert(ok);

    // Execute tool through unified runtime
    strata::tools::ToolExecutionRequest req;
    req.tool_name = "benchmark";
    req.command_or_arguments = "run_benchmarks";
    req.raw_output = "Benchmark results: throughput=45.2 tok/s, first_token=28.4 ms\n";
    req.exit_code = 0;
    unified.tool_runtime().process_tool_execution(req);

    // Prepare prompt for user query
    std::string prompt = unified.prepare_prompt_for_query("What was the benchmark throughput?");
    assert(!prompt.empty());

    std::cout << unified.print_full_system_status() << std::endl;
    std::cout << "  Passed. Strata Unified AI Runtime operational across all layers!" << std::endl;
}

int main() {
    std::cout << "=================================================================\n"
              << "   RUNNING STRATA TOOL RUNTIME & UNIFIED SYSTEM TEST SUITE       \n"
              << "=================================================================\n";

    test_tool_parsers();
    test_content_addressed_storage();
    test_large_output_policy_and_compact_observation();
    test_tool_fragment_retrieval();
    test_context_materialization_lease();
    test_tool_virtual_context_rag_unification();
    test_unified_ai_runtime_end_to_end();

    std::cout << "=================================================================\n"
              << "   ALL TOOL RUNTIME & UNIFIED SYSTEM TESTS PASSED (7/7)          \n"
              << "=================================================================\n";
    return 0;
}
