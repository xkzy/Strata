# serve/virtual_context.py - Python Virtual Context Service for Strata API Server
"""Manages large virtual contexts (1M-2M+ tokens) over small physical LLM contexts (8K-32K).

Provides:
- Dynamic token budget management
- Multi-tier hierarchical memory (L0 raw events -> L1-L3 summaries)
- Automatic incremental context compaction
- Built-in hierarchical keyword/metadata/symbol retrieval
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class ContextItemType(Enum):
    SYSTEM_PROMPT = "system"
    USER_MESSAGE = "user"
    ASSISTANT_MESSAGE = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_OUTPUT = "tool_output"
    FILE_CONTENT = "file"
    CODE_DIFF = "code_diff"
    ERROR_LOG = "error"
    TEST_RESULT = "test"
    WORKING_MEMORY = "working_memory"


class ResidencyState(Enum):
    ACTIVE = "active"          # Resident in model physical context
    WORKING = "working"        # High-priority working state / scratchpad
    COMPRESSED = "compressed"  # Summarized in active context; raw indexed
    INDEXED = "indexed"        # In searchable retrieval index
    ARCHIVED = "archived"      # Cold persistent storage


@dataclass
class ContextMetadata:
    filename: str = ""
    symbol_name: str = ""
    command: str = ""
    exit_code: int = 0
    commit_hash: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class ContextItem:
    item_id: int
    item_type: ContextItemType
    raw_content: str
    token_count: int
    timestamp: float
    state: ResidencyState = ResidencyState.ACTIVE
    summary_l1: str = ""
    summary_l2: str = ""
    importance_score: float = 1.0
    access_count: int = 0
    last_accessed: float = 0.0
    metadata: ContextMetadata = field(default_factory=ContextMetadata)

    def get_effective_content(self) -> str:
        if self.state in (ResidencyState.ACTIVE, ResidencyState.WORKING):
            return self.raw_content
        elif self.state == ResidencyState.COMPRESSED:
            return self.summary_l1 or (self.raw_content[:200] + " ... [compressed]")
        return f"[Archived item {self.item_id}]"


class HierarchicalBM25Index:
    """In-memory BM25 index for fast sub-linear historical context search."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_records: Dict[int, ContextItem] = {}
        self.doc_lengths: Dict[int, int] = {}
        self.inverted_index: Dict[str, Dict[int, int]] = {}  # term -> {doc_id: tf}
        self.file_index: Dict[str, Set[int]] = {}
        self.tag_index: Dict[str, Set[int]] = {}
        self.avg_doc_len = 0.0

    def _tokenize(self, text: str) -> List[str]:
        return [w.lower() for w in re.findall(r"[A-Za-z0-9_]{2,}", text)]

    def index_item(self, item: ContextItem) -> None:
        doc_id = item.item_id
        text = item.raw_content + " " + item.summary_l1
        tokens = self._tokenize(text)
        doc_len = len(tokens)

        self.doc_records[doc_id] = item
        self.doc_lengths[doc_id] = doc_len

        # Count TF
        tf_map: Dict[str, int] = {}
        for t in tokens:
            tf_map[t] = tf_map.get(t, 0) + 1

        for term, tf in tf_map.items():
            if term not in self.inverted_index:
                self.inverted_index[term] = {}
            self.inverted_index[term][doc_id] = tf

        # Metadata
        if item.metadata.filename:
            self.file_index.setdefault(item.metadata.filename, set()).add(doc_id)
        for tag in item.metadata.tags:
            self.tag_index.setdefault(tag, set()).add(doc_id)

        n = len(self.doc_records)
        self.avg_doc_len = sum(self.doc_lengths.values()) / max(1, n)

    def search(self, query: str, top_k: int = 5, filename_filter: str = "") -> List[Tuple[ContextItem, float]]:
        query_tokens = self._tokenize(query)
        if not query_tokens and not filename_filter:
            return []

        n_docs = len(self.doc_records)
        if n_docs == 0:
            return []

        scores: Dict[int, float] = {}

        for term in query_tokens:
            if term in self.inverted_index:
                postings = self.inverted_index[term]
                df = len(postings)
                idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

                for doc_id, tf in postings.items():
                    dl = self.doc_lengths.get(doc_id, 50)
                    num = tf * (self.k1 + 1.0)
                    den = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avg_doc_len))
                    scores[doc_id] = scores.get(doc_id, 0.0) + (idf * (num / den))

        if filename_filter and filename_filter in self.file_index:
            for doc_id in self.file_index[filename_filter]:
                scores[doc_id] = scores.get(doc_id, 0.0) + 5.0

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score in ranked[:top_k]:
            if doc_id in self.doc_records:
                results.append((self.doc_records[doc_id], score))
        return results


class VirtualContextManager:
    """Manages virtual context (1M-2M+ tokens) over a physical context window."""

    def __init__(self, physical_context_limit: int = 8192, generation_reserve: int = 1024):
        self.physical_context_limit = physical_context_limit
        self.generation_reserve = generation_reserve
        self.max_input_tokens = physical_context_limit - generation_reserve

        self.index = HierarchicalBM25Index()
        self.items: List[ContextItem] = []
        self.system_prompt: Optional[ContextItem] = None
        self.working_memory: Optional[ContextItem] = None
        self.next_item_id = 1

        self.virtual_tokens = 0
        self.active_tokens = 0
        self.compaction_events = 0
        self.tokens_freed = 0

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def set_system_prompt(self, content: str) -> None:
        tokens = self.estimate_tokens(content)
        self.system_prompt = ContextItem(
            item_id=self.next_item_id,
            item_type=ContextItemType.SYSTEM_PROMPT,
            raw_content=content,
            token_count=tokens,
            timestamp=time.time(),
        )
        self.next_item_id += 1
        self.virtual_tokens += tokens

    def update_working_memory(self, state: str) -> None:
        tokens = self.estimate_tokens(state)
        self.working_memory = ContextItem(
            item_id=self.next_item_id,
            item_type=ContextItemType.WORKING_MEMORY,
            raw_content=state,
            token_count=tokens,
            timestamp=time.time(),
        )
        self.next_item_id += 1

    def append_message(self, role: str, content: str, metadata: Optional[ContextMetadata] = None) -> int:
        tokens = self.estimate_tokens(content)
        itype = ContextItemType.USER_MESSAGE if role == "user" else ContextItemType.ASSISTANT_MESSAGE
        item = ContextItem(
            item_id=self.next_item_id,
            item_type=itype,
            raw_content=content,
            token_count=tokens,
            timestamp=time.time(),
            metadata=metadata or ContextMetadata(),
        )
        self.next_item_id += 1
        self.items.append(item)
        self.virtual_tokens += tokens
        self.active_tokens += tokens

        self._compact_if_needed()
        return item.item_id

    def append_tool_result(self, tool_name: str, output: str, command: str = "", exit_code: int = 0) -> int:
        tokens = self.estimate_tokens(output)
        meta = ContextMetadata(command=command, exit_code=exit_code, tags=[tool_name])
        item = ContextItem(
            item_id=self.next_item_id,
            item_type=ContextItemType.TOOL_OUTPUT,
            raw_content=output,
            token_count=tokens,
            timestamp=time.time(),
            metadata=meta,
        )
        self.next_item_id += 1

        if tokens > 1500:
            # Index immediately & summarize in active context
            self.index.index_item(item)
            item.state = ResidencyState.COMPRESSED
            item.summary_l1 = f"[Tool {tool_name} output ({tokens} tokens) indexed into memory]"
            item.token_count = 50

        self.items.append(item)
        self.virtual_tokens += tokens
        self.active_tokens += item.token_count

        self._compact_if_needed()
        return item.item_id

    def _compact_if_needed(self, threshold: float = 0.85) -> None:
        if self.active_tokens < (self.max_input_tokens * threshold):
            return

        target = int(self.max_input_tokens * 0.65)
        self.compaction_events += 1

        # Pass 1: Compress active items to summaries
        for item in self.items:
            if self.active_tokens <= target:
                break
            if item.state == ResidencyState.ACTIVE and item.item_type != ContextItemType.SYSTEM_PROMPT:
                self.index.index_item(item)
                old_tok = item.token_count
                item.state = ResidencyState.COMPRESSED
                item.summary_l1 = f"[{item.item_type.value}: {item.raw_content[:100]}...]"
                item.token_count = max(15, old_tok // 5)
                freed = old_tok - item.token_count
                self.active_tokens -= freed
                self.tokens_freed += freed

        # Pass 2: If still above target, move compressed items to indexed (0 active tokens)
        if self.active_tokens > target:
            for item in self.items:
                if self.active_tokens <= target:
                    break
                if item.state == ResidencyState.COMPRESSED and item.item_type != ContextItemType.SYSTEM_PROMPT:
                    freed = item.token_count
                    item.state = ResidencyState.INDEXED
                    item.token_count = 0
                    self.active_tokens -= freed
                    self.tokens_freed += freed

    def assemble_prompt(self, user_query: str) -> str:
        self._compact_if_needed()

        # Retrieve relevant historical knowledge
        retrieved = self.index.search(user_query, top_k=3)

        parts = []
        if self.system_prompt:
            parts.append(f"<|system|>\n{self.system_prompt.raw_content}\n")

        if self.working_memory:
            parts.append(f"[Working Memory / Task State]:\n{self.working_memory.raw_content}\n")

        if retrieved:
            parts.append("[Retrieved Historical Context]:")
            for doc, _ in retrieved:
                parts.append(f"- {doc.raw_content}")
            parts.append("")

        parts.append("[Conversation]:")
        for item in self.items:
            parts.append(f"{item.item_type.value}: {item.get_effective_content()}")

        return "\n".join(parts)

    def stats(self) -> Dict[str, Any]:
        return {
            "virtual_tokens": self.virtual_tokens,
            "active_tokens": self.active_tokens,
            "physical_context_limit": self.physical_context_limit,
            "compaction_events": self.compaction_events,
            "tokens_freed": self.tokens_freed,
            "total_items": len(self.items),
        }
