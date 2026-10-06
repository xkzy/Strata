# serve/test_virtual_context.py - Comprehensive Unit & Integration Tests for Virtual Context Server Runtime
import json
import pytest
from serve.virtual_context import (
    CompilerOutputParser,
    ContextItem,
    ContextItemType,
    ContextMetadata,
    HierarchicalBM25Index,
    InformationClass,
    ResidencyState,
    SearchOutputParser,
    TestOutputParser,
    ToolResultCASStore,
    VirtualContextServerRuntime,
)


def test_bm25_retrieval_index():
    index = HierarchicalBM25Index()

    item1 = ContextItem(
        item_id=1,
        item_type=ContextItemType.USER_MESSAGE,
        raw_content="The database port is 5432 and user is admin.",
        token_count=12,
        timestamp=100.0,
    )
    item2 = ContextItem(
        item_id=2,
        item_type=ContextItemType.FILE_CONTENT,
        raw_content="class ConnectionPool { void connect(); };",
        token_count=10,
        timestamp=101.0,
        metadata=ContextMetadata(filename="pool.cpp"),
    )

    index.index_item(item1)
    index.index_item(item2)

    res = index.search("database port", top_k=1)
    assert len(res) == 1
    assert res[0][0].item_id == 1

    file_res = index.search("", filename_filter="pool.cpp", top_k=1)
    assert len(file_res) == 1
    assert file_res[0][0].item_id == 2


def test_structured_tool_parsers():
    # Compiler Parser
    compiler_parser = CompilerOutputParser()
    raw_compiler = (
        "src/engine.cpp:12:5: error: 'undefined_var' was not declared in this scope\n"
        "src/engine.cpp:15:1: warning: unused variable 'x' [-Wunused-variable]\n"
        "FAILED: [code=1] src/engine.cpp.o\n"
    )
    c_data = compiler_parser.parse("compiler", raw_compiler, exit_code=1)
    assert c_data.status == "failed"
    assert c_data.error_count == 2
    assert c_data.warning_count == 1
    assert len(c_data.key_excerpts) > 0

    # Test Parser
    test_parser = TestOutputParser()
    raw_test = "=== test session starts ===\nFAILED tests/test_core.py::test_init\n1 failed, 45 passed in 0.5s"
    t_data = test_parser.parse("pytest", raw_test, exit_code=1)
    assert t_data.status == "failed"
    assert t_data.pass_count == 45
    assert t_data.fail_count == 1
    assert any("FAILED" in exc for exc in t_data.key_excerpts)

    # Search Parser
    search_parser = SearchOutputParser()
    raw_search = (
        "src/runtime/device.cpp:45: class ComputeDevice;\n"
        "src/runtime/device.cpp:102: void init();\n"
        "include/device.hpp:10: struct Device;\n"
    )
    s_data = search_parser.parse("grep", raw_search, exit_code=0)
    assert s_data.status == "success"
    assert s_data.matches_count == 3
    assert s_data.fields["unique_files_count"] == 2


def test_tool_result_cas_store_and_bounded_policy():
    cas = ToolResultCASStore()

    # Small output (should inline directly)
    small_raw = "Build succeeded in 0.2s."
    rid1, data1, obs1 = cas.ingest("build", small_raw, exit_code=0)
    assert obs1 == small_raw
    assert data1.raw_tokens < 300

    # Large output (3000 tokens / 12000 chars)
    large_raw = "\n".join([f"log line {i}: trace info execution step {i}" for i in range(500)])
    rid2, data2, obs2 = cas.ingest("server_logs", large_raw, exit_code=0)
    assert rid2.startswith("tool_res_")
    assert len(obs2) < len(large_raw)
    assert "Metrics: 500 lines" in obs2
    assert rid2 in obs2

    # Verify CAS retrieval
    retrieved = cas.get_raw_by_id(rid2)
    assert retrieved == large_raw

    # Fragment retrieval
    frag = cas.get_fragment(rid2, start_line=10, end_line=12)
    assert frag == "log line 9: trace info execution step 9\nlog line 10: trace info execution step 10\nlog line 11: trace info execution step 11"

    # Deduplication
    rid3, _, _ = cas.ingest("server_logs", large_raw, exit_code=0)
    assert rid3 == rid2
    assert cas.stats()["total_stored_entries"] == 2  # small + large


def test_virtual_context_server_runtime_openai_tool_processing():
    runtime = VirtualContextServerRuntime(physical_context_limit=4096)

    large_compiler_log = "error: unknown type 'MyStruct'\n" + "\n".join([f"src/file{i}.cpp: note: in file included from here" for i in range(400)])
    messages = [
        {"role": "system", "content": "You are a software engineer."},
        {"role": "user", "content": "Run the compiler build."},
        {"role": "assistant", "content": "Running cmake --build ."},
        {"role": "tool", "name": "compiler", "content": large_compiler_log, "tool_call_id": "call_123"},
        {"role": "user", "content": "Fix the build errors."}
    ]

    processed = runtime.process_messages(messages, target_budget=2048)

    # System and user intent preserved
    assert processed[0]["content"] == "You are a software engineer."
    assert processed[-1]["content"] == "Fix the build errors."

    # Tool output replaced with compact observation
    tool_msg = next(m for m in processed if m.get("role") == "tool")
    assert "tool_res_" in tool_msg["content"]
    assert "Diagnostics: 1 errors" in tool_msg["content"]
    assert len(tool_msg["content"]) < 1000

    # Raw content still preserved in CAS
    stats = runtime.stats()
    assert stats["tool_tokens_saved"] > 300
    assert stats["cas_store"]["total_stored_entries"] >= 1


def test_virtual_context_server_runtime_anthropic_tool_processing():
    runtime = VirtualContextServerRuntime(physical_context_limit=4096)

    large_pytest_log = "1 failed, 100 passed\nFAILED test_auth.py::test_login - AssertionError\n" + "\n".join([f"traceback line {i}..." for i in range(300)])
    messages = [
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_456", "content": large_pytest_log}
        ]},
        {"role": "user", "content": "Analyze the test failure."}
    ]

    processed = runtime.process_messages(messages, target_budget=2048)
    block = processed[0]["content"][0]
    assert block["type"] == "tool_result"
    assert "tool_res_" in block["content"]
    assert "100 passed, 1 failed" in block["content"]


def test_progressive_conversation_compaction():
    # Small budget (500 tokens)
    runtime = VirtualContextServerRuntime(physical_context_limit=500, generation_reserve=100)

    messages = [
        {"role": "system", "content": "You are an AI assistant specialized in database engineering."},
    ]

    for i in range(30):
        messages.append({"role": "user", "content": f"Query {i}: We configured PostgreSQL connection pool max_connections=200 and shared_buffers=4GB in step {i}."})
        messages.append({"role": "assistant", "content": f"<think>Thinking deeply about database tuning for step {i}...</think>Acknowledged configuration step {i}."})

    messages.append({"role": "user", "content": "What was the PostgreSQL max_connections configured earlier?"})

    processed = runtime.process_messages(messages, target_budget=350)

    # 1. System prompt preserved
    assert processed[0]["role"] == "system"
    assert "specialized in database engineering" in processed[0]["content"]

    # 2. Latest user turn preserved intact
    assert processed[-1]["role"] == "user"
    assert processed[-1]["content"] == "What was the PostgreSQL max_connections configured earlier?"

    # 3. Earlier conversation compacted into a summary turn with retrieved context
    has_summary = any("earlier conversation" in m.get("content", "") for m in processed)
    assert has_summary

    # 4. Old reasoning <think> blocks stripped/compressed
    for m in processed:
        if m.get("role") == "assistant":
            assert "<think>Thinking deeply" not in m.get("content", "")

    stats = runtime.stats()
    assert stats["compaction_events"] > 0
    assert stats["virtual_tokens_processed"] > 1000
    assert stats["physical_tokens_emitted"] <= 350
