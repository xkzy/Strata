# serve/test_virtual_context.py - Tests for Python Virtual Context Service
from serve.virtual_context import (
    ContextItemType,
    HierarchicalBM25Index,
    ResidencyState,
    VirtualContextManager,
)


def test_bm25_retrieval_index():
    index = HierarchicalBM25Index()

    from serve.virtual_context import ContextItem, ContextMetadata

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


def test_virtual_context_compaction_and_budgeting():
    # 512 physical limit (max input = 384)
    vcm = VirtualContextManager(physical_context_limit=512, generation_reserve=128)
    vcm.set_system_prompt("System assistant prompt.")
    vcm.update_working_memory("Task: Refactor routing.")

    # Add messages exceeding budget
    for i in range(25):
        vcm.append_message("user", f"Turn {i}: Detailed log entry {i} containing extensive parameters, execution stack traces, metrics, and context.")
        vcm.append_message("assistant", f"Acknowledged turn {i}.")

    stats = vcm.stats()
    assert stats["virtual_tokens"] > 500
    assert stats["active_tokens"] <= vcm.max_input_tokens
    assert stats["compaction_events"] > 0

    # Query about compacted item
    prompt = vcm.assemble_prompt("Detailed log entry 2")
    assert "System assistant prompt." in prompt
    assert "Task: Refactor routing." in prompt
    assert "[Retrieved Historical Context]:" in prompt
