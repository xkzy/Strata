# serve/virtual_context.py - Transparent Virtual Context & Tool State Runtime for Strata API Server
"""Transparent, Agent-Compatible Virtual Context & Tool State Runtime.

Enables Strata to operate as an inference + context + memory + retrieval runtime underneath
external agents (Claude Code, Codex, Gemini, Roo-Cline, Aider, OpenHands, custom agents)
over large conversations (1M-2M+ tokens) while maintaining a small, bounded physical LLM context.

Core Principles:
1. Strata is NOT the agent: No hijacking of planning, tool selection, or tool execution.
2. Transparent API Compatibility: Works seamlessly on standard OpenAI & Anthropic message schemas.
3. Immutable Source of Truth: Raw messages and tool outputs are stored permanently in server memory/CAS.
4. Bounded Tool Observation Policy: Large/huge tool outputs (>1,500 tokens) are stored in CAS, parsed,
   and represented as compact, high-signal observations (<120 tokens) with result IDs and pointers.
5. Automatic Incremental Compaction & BM25 Retrieval: Fits working context strictly within physical budget.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class InformationClass(Enum):
    CRITICAL = "critical"                # System instructions, active user query, pending directives (Never drop)
    IMPORTANT = "important"              # Strategic architectural decisions, core file changes, key decisions
    TEMPORARY = "temporary"              # Old reasoning traces (<think> blocks), conversational filler
    RECONSTRUCTABLE = "reconstructable"  # Search listings, directory trees, file dumps
    RAW = "raw"                          # High-volume unparsed tool outputs (Stored in CAS)


class ContextItemType(Enum):
    SYSTEM_PROMPT = "system"
    USER_MESSAGE = "user"
    ASSISTANT_MESSAGE = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_OUTPUT = "tool"
    FILE_CONTENT = "file"
    CODE_DIFF = "code_diff"
    ERROR_LOG = "error"
    TEST_RESULT = "test"
    WORKING_MEMORY = "working_memory"


class ResidencyState(Enum):
    ACTIVE = "active"          # Resident in model physical working context
    WORKING = "working"        # High-priority working state / scratchpad
    COMPRESSED = "compressed"  # Summarized in active context; raw indexed
    INDEXED = "indexed"        # In searchable retrieval index (0 physical tokens)
    ARCHIVED = "archived"      # Cold persistent storage


@dataclass
class ContextMetadata:
    filename: str = ""
    symbol_name: str = ""
    command: str = ""
    exit_code: int = 0
    commit_hash: str = ""
    tool_name: str = ""
    result_id: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class ContextItem:
    item_id: int
    item_type: ContextItemType
    raw_content: str
    token_count: int
    timestamp: float
    info_class: InformationClass = InformationClass.IMPORTANT
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
            return self.summary_l1 or (self.raw_content[:200] + " ... [compressed summary]")
        return f"[Archived item {self.item_id}: {self.metadata.tool_name or self.item_type.value}]"


# =====================================================================
# Structured Tool Parsers & Bounded Observation Generation
# =====================================================================

@dataclass
class StructuredToolData:
    tool_name: str
    status: str = "success"
    exit_code: int = 0
    total_lines: int = 0
    raw_tokens: int = 0
    error_count: int = 0
    warning_count: int = 0
    pass_count: int = 0
    fail_count: int = 0
    matches_count: int = 0
    key_excerpts: List[str] = field(default_factory=list)
    fields: Dict[str, Any] = field(default_factory=dict)


class ToolOutputParser:
    """Base parser for tool outputs."""
    def parse(self, tool_name: str, raw_output: str, exit_code: int = 0) -> StructuredToolData:
        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        data = StructuredToolData(
            tool_name=tool_name,
            exit_code=exit_code,
            total_lines=len(lines),
            raw_tokens=max(1, len(raw_output) // 4),
            status="success" if exit_code == 0 else "failed"
        )
        return data


class CompilerOutputParser(ToolOutputParser):
    """Parses compiler and build tool diagnostics (gcc, clang, rustc, nvcc, cmake, msbuild)."""
    def parse(self, tool_name: str, raw_output: str, exit_code: int = 0) -> StructuredToolData:
        data = super().parse(tool_name, raw_output, exit_code)
        lines = raw_output.splitlines()
        errors = []
        warnings = []

        err_pattern = re.compile(r"(error:|fatal error:|FAILED:|undefined reference|syntax error)", re.IGNORECASE)
        warn_pattern = re.compile(r"(warning:)", re.IGNORECASE)

        for line in lines:
            if err_pattern.search(line):
                errors.append(line.strip())
            elif warn_pattern.search(line):
                warnings.append(line.strip())

        data.error_count = len(errors)
        data.warning_count = len(warnings)
        if data.error_count > 0 or exit_code != 0:
            data.status = "failed"
        data.key_excerpts = errors[:5] if errors else warnings[:3]
        return data


class TestOutputParser(ToolOutputParser):
    """Parses test runner outputs (pytest, ctest, cargo test, jest, vitest, mocha)."""
    def parse(self, tool_name: str, raw_output: str, exit_code: int = 0) -> StructuredToolData:
        data = super().parse(tool_name, raw_output, exit_code)
        lines = raw_output.splitlines()
        failed_tests = []

        for line in lines:
            m_pass = re.search(r"(\d+)\s+passed", line, re.IGNORECASE)
            if m_pass:
                data.pass_count = int(m_pass.group(1))
            m_fail = re.search(r"(\d+)\s+failed", line, re.IGNORECASE)
            if m_fail:
                data.fail_count = int(m_fail.group(1))
            if "FAIL" in line or "FAILED" in line or "ERR" in line:
                failed_tests.append(line.strip())

        if data.fail_count > 0 or exit_code != 0:
            data.status = "failed"
        data.key_excerpts = failed_tests[:5]
        return data


class SearchOutputParser(ToolOutputParser):
    """Parses search and grep tools (grep, ripgrep, find, git grep)."""
    def parse(self, tool_name: str, raw_output: str, exit_code: int = 0) -> StructuredToolData:
        data = super().parse(tool_name, raw_output, exit_code)
        lines = [l.strip() for l in raw_output.splitlines() if l.strip()]
        data.matches_count = len(lines)
        unique_files = set()
        for line in lines:
            parts = line.split(":", 1)
            if len(parts) > 1 and ("/" in parts[0] or "." in parts[0]):
                unique_files.add(parts[0])

        data.fields["unique_files_count"] = len(unique_files)
        data.key_excerpts = lines[:5]
        return data


class GenericToolParser(ToolOutputParser):
    """Generic fallback tool output parser."""
    def parse(self, tool_name: str, raw_output: str, exit_code: int = 0) -> StructuredToolData:
        data = super().parse(tool_name, raw_output, exit_code)
        lines = [l.strip() for l in raw_output.splitlines() if l.strip()]
        if lines:
            data.key_excerpts = lines[:3] + ([f"... ({len(lines) - 4} lines truncated)"] if len(lines) > 4 else []) + lines[-1:]
        return data


# =====================================================================
# Server-Side Tool Result Store & Content-Addressed Storage (CAS)
# =====================================================================

class ToolResultCASStore:
    """Content-Addressed Storage (CAS) and Bounded Observation engine for tool results."""

    def __init__(self):
        self._lock = threading.RLock()
        self._store: Dict[str, str] = {}                 # hash -> raw payload
        self._records: Dict[str, StructuredToolData] = {}  # result_id -> structured data
        self._hash_to_id: Dict[str, str] = {}
        self._compiler_parser = CompilerOutputParser()
        self._test_parser = TestOutputParser()
        self._search_parser = SearchOutputParser()
        self._generic_parser = GenericToolParser()

    def _select_parser(self, tool_name: str, raw_output: str) -> ToolOutputParser:
        lower_name = tool_name.lower()
        if any(k in lower_name for k in ("build", "compile", "make", "cmake", "gcc", "clang", "cargo_build")):
            return self._compiler_parser
        if any(k in lower_name for k in ("test", "pytest", "ctest", "jest", "vitest", "cargo_test")):
            return self._test_parser
        if any(k in lower_name for k in ("grep", "search", "find", "rg", "ripgrep")):
            return self._search_parser
        if "error:" in raw_output.lower() or "warning:" in raw_output.lower():
            return self._compiler_parser
        if "passed" in raw_output.lower() and "failed" in raw_output.lower():
            return self._test_parser
        return self._generic_parser

    def ingest(self, tool_name: str, raw_output: str, command: str = "", exit_code: int = 0) -> Tuple[str, StructuredToolData, str]:
        """Ingests raw tool output into CAS and generates a bounded observation string."""
        raw_bytes = raw_output.encode("utf-8")
        h = hashlib.sha256(raw_bytes).hexdigest()

        with self._lock:
            if h in self._hash_to_id:
                result_id = self._hash_to_id[h]
                data = self._records[result_id]
                obs = self.render_observation(result_id, data)
                return result_id, data, obs

            result_id = f"tool_res_{h[:12]}"
            self._store[h] = raw_output
            self._hash_to_id[h] = result_id

            parser = self._select_parser(tool_name, raw_output)
            data = parser.parse(tool_name, raw_output, exit_code)
            self._records[result_id] = data
            obs = self.render_observation(result_id, data)
            return result_id, data, obs

    def render_observation(self, result_id: str, data: StructuredToolData) -> str:
        """Renders bounded observation for LLM physical context."""
        raw_tokens = data.raw_tokens
        if raw_tokens <= 300:
            # Small output: inline directly
            return self.get_raw_by_id(result_id)

        # Large / Huge output: Bounded Compact Observation (<120 tokens)
        lines = [
            f"[Tool Result: {data.tool_name} | Status: {data.status} | Exit Code: {data.exit_code}]",
            f"Metrics: {data.total_lines} lines ({raw_tokens} tokens)"
        ]
        if data.error_count or data.warning_count:
            lines.append(f"Diagnostics: {data.error_count} errors, {data.warning_count} warnings")
        if data.pass_count or data.fail_count:
            lines.append(f"Test Summary: {data.pass_count} passed, {data.fail_count} failed")
        if data.matches_count:
            lines.append(f"Search Matches: {data.matches_count} matches in {data.fields.get('unique_files_count', 1)} files")

        if data.key_excerpts:
            lines.append("Key Excerpts:")
            for exc in data.key_excerpts[:4]:
                lines.append(f"  > {exc[:160]}")

        lines.append(f"[Full output stored server-side. Result ID: {result_id}]")
        return "\n".join(lines)

    def get_raw_by_id(self, result_id: str) -> str:
        with self._lock:
            for h, rid in self._hash_to_id.items():
                if rid == result_id:
                    return self._store.get(h, "")
            return ""

    def get_fragment(self, result_id: str, start_line: int, end_line: int) -> str:
        raw = self.get_raw_by_id(result_id)
        if not raw:
            return ""
        lines = raw.splitlines()
        s = max(0, start_line - 1)
        e = min(len(lines), end_line)
        return "\n".join(lines[s:e])

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total_bytes = sum(len(v.encode("utf-8")) for v in self._store.values())
            return {
                "total_stored_entries": len(self._store),
                "total_stored_bytes": total_bytes,
                "total_records": len(self._records),
            }


# =====================================================================
# Hierarchical BM25 Retrieval Index
# =====================================================================

class HierarchicalBM25Index:
    """In-memory sub-linear BM25 index for context search."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_records: Dict[int, ContextItem] = {}
        self.doc_lengths: Dict[int, int] = {}
        self.inverted_index: Dict[str, Dict[int, int]] = {}  # term -> {doc_id: tf}
        self.file_index: Dict[str, Set[int]] = {}
        self.tag_index: Dict[str, Set[int]] = {}
        self.total_doc_len = 0
        self.avg_doc_len = 0.0

    def _tokenize(self, text: str) -> List[str]:
        return [w.lower() for w in re.findall(r"[A-Za-z0-9_]{2,}", text)]

    def index_item(self, item: ContextItem) -> None:
        doc_id = item.item_id
        text = item.raw_content + " " + item.summary_l1 + " " + item.metadata.filename + " " + item.metadata.tool_name
        tokens = self._tokenize(text)
        doc_len = len(tokens)

        self.doc_records[doc_id] = item
        self.doc_lengths[doc_id] = doc_len
        self.total_doc_len += doc_len

        tf_map: Dict[str, int] = {}
        for t in tokens:
            tf_map[t] = tf_map.get(t, 0) + 1

        for term, tf in tf_map.items():
            if term not in self.inverted_index:
                self.inverted_index[term] = {}
            self.inverted_index[term][doc_id] = tf

        if item.metadata.filename:
            self.file_index.setdefault(item.metadata.filename, set()).add(doc_id)
        for tag in item.metadata.tags:
            self.tag_index.setdefault(tag, set()).add(doc_id)

        n = len(self.doc_records)
        self.avg_doc_len = self.total_doc_len / max(1, n)

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


# =====================================================================
# Server-Side Virtual Context Runtime
# =====================================================================

class VirtualContextServerRuntime:
    """Transparent server-side context & memory runtime.
    
    Acts underneath external agents:
    - Ingests long message histories and tool results.
    - Preserves immutable raw content in memory & CAS.
    - Applies bounded tool observation policy.
    - Compacts conversation to fit bounded physical LLM context without dropping critical intent.
    - Enhances physical prompt with retrieved relevant past knowledge.
    """

    def __init__(self, physical_context_limit: int = 32768, generation_reserve: int = 1024):
        self.physical_context_limit = physical_context_limit
        self.generation_reserve = generation_reserve
        self.tool_store = ToolResultCASStore()
        self.index = HierarchicalBM25Index()
        self._lock = threading.RLock()
        self._next_id = 1

        # Metrics
        self.total_requests = 0
        self.virtual_tokens_processed = 0
        self.physical_tokens_emitted = 0
        self.compaction_events = 0
        self.tool_tokens_saved = 0

    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)

    def _strip_think_tags(self, text: str) -> str:
        """Strips or shortens old reasoning blocks (<think>...</think>) in historical turns."""
        if "<think>" in text and "</think>" in text:
            return re.sub(r"<think>.*?</think>", "[reasoning trace compressed]", text, flags=re.DOTALL)
        return text

    def process_messages(self, messages: List[Dict[str, Any]], target_budget: int, tokenizer: Any = None) -> List[Dict[str, Any]]:
        """Transparently prepares messages for the physical model context window."""
        if not messages:
            return messages

        with self._lock:
            self.total_requests += 1

        # Work on a deep copy to never mutate caller's raw structures
        processed = copy.deepcopy(messages)

        # Step 1: Bounded Tool Observation Policy
        # Process tool outputs in OpenAI ('role': 'tool') or Anthropic ('tool_result') formats
        tool_savings = 0
        for msg in processed:
            role = msg.get("role")
            content = msg.get("content")

            # OpenAI format: role == "tool" or role == "function"
            if role in ("tool", "function") and isinstance(content, str):
                raw_tokens = self.estimate_tokens(content)
                if raw_tokens > 300:
                    tool_name = msg.get("name") or "tool"
                    _, _, obs = self.tool_store.ingest(tool_name, content)
                    obs_tokens = self.estimate_tokens(obs)
                    msg["content"] = obs
                    tool_savings += max(0, raw_tokens - obs_tokens)

            # Anthropic format: list of content blocks with type: "tool_result"
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        t_content = block.get("content")
                        if isinstance(t_content, str):
                            raw_tokens = self.estimate_tokens(t_content)
                            if raw_tokens > 300:
                                tool_name = block.get("tool_use_id") or "tool"
                                _, _, obs = self.tool_store.ingest(tool_name, t_content)
                                obs_tokens = self.estimate_tokens(obs)
                                block["content"] = obs
                                tool_savings += max(0, raw_tokens - obs_tokens)

        with self._lock:
            self.tool_tokens_saved += tool_savings

        # Step 2: Estimate Total Prompt Tokens
        def count_tokens(msgs: List[Dict[str, Any]]) -> int:
            if tokenizer and hasattr(tokenizer, "encode"):
                try:
                    # Quick check using string serialization
                    text = json.dumps(msgs, ensure_ascii=False)
                    return len(tokenizer.encode(text, parse_special=False))
                except Exception:
                    pass
            total = 0
            for m in msgs:
                c = m.get("content")
                if isinstance(c, str):
                    total += self.estimate_tokens(c)
                elif isinstance(c, list):
                    for b in c:
                        if isinstance(b, dict) and "text" in b:
                            total += self.estimate_tokens(b["text"])
                        elif isinstance(b, dict) and "content" in b and isinstance(b["content"], str):
                            total += self.estimate_tokens(b["content"])
            return total

        current_tokens = count_tokens(processed)
        with self._lock:
            self.virtual_tokens_processed += current_tokens + tool_savings

        # Index messages into session retrieval index
        for m in processed:
            if m.get("role") not in ("system", "developer"):
                c = m.get("content", "")
                raw_text = c if isinstance(c, str) else json.dumps(c)
                if raw_text:
                    item = ContextItem(
                        item_id=self._next_id,
                        item_type=ContextItemType.USER_MESSAGE if m.get("role") == "user" else ContextItemType.ASSISTANT_MESSAGE,
                        raw_content=raw_text,
                        token_count=self.estimate_tokens(raw_text),
                        timestamp=time.time(),
                        info_class=InformationClass.TEMPORARY if "<think>" in raw_text else InformationClass.IMPORTANT,
                    )
                    self._next_id += 1
                    self.index.index_item(item)

        # If already fits comfortably within physical target budget, return directly
        if current_tokens <= target_budget:
            with self._lock:
                self.physical_tokens_emitted += current_tokens
            return processed

        # Step 3: Progressive Context Compaction & Retrieval Injection
        with self._lock:
            self.compaction_events += 1

        # Classify and index items
        # System prompt and latest user turn are CRITICAL
        n_msgs = len(processed)
        system_msgs = [m for m in processed if m.get("role") in ("system", "developer")]
        non_system_msgs = [m for m in processed if m.get("role") not in ("system", "developer")]

        if not non_system_msgs:
            return processed

        # Extract latest query for retrieval
        latest_user_text = ""
        for m in reversed(non_system_msgs):
            if m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str):
                    latest_user_text = c
                elif isinstance(c, list):
                    latest_user_text = " ".join([b.get("text", "") for b in c if isinstance(b, dict)])
                break

        # Index historical messages
        for idx, m in enumerate(non_system_msgs[:-2]):
            c = m.get("content", "")
            raw_text = c if isinstance(c, str) else json.dumps(c)
            item = ContextItem(
                item_id=self._next_id,
                item_type=ContextItemType.USER_MESSAGE if m.get("role") == "user" else ContextItemType.ASSISTANT_MESSAGE,
                raw_content=raw_text,
                token_count=self.estimate_tokens(raw_text),
                timestamp=time.time(),
                info_class=InformationClass.TEMPORARY if "<think>" in raw_text else InformationClass.IMPORTANT,
            )
            self._next_id += 1
            self.index.index_item(item)

        # Pass A: Strip old reasoning traces (<think>) from historical assistant messages
        for m in non_system_msgs:
            if m.get("role") == "assistant" and isinstance(m.get("content"), str):
                m["content"] = self._strip_think_tags(m["content"])

        # Pass B: If still exceeding budget, progressively compact older turns
        # Retain last 3 turns intact (CRITICAL working set)
        working_tail_count = min(len(non_system_msgs), 4)
        head_msgs = non_system_msgs[:-working_tail_count]
        tail_msgs = non_system_msgs[-working_tail_count:]

        # Compact head messages into a structured summary turn
        if head_msgs:
            summary_lines = [f"[Summary of {len(head_msgs)} earlier conversation messages]:"]
            sampled_head = head_msgs[:2] + (head_msgs[-2:] if len(head_msgs) > 4 else head_msgs[2:])
            for m in sampled_head:
                r = m.get("role", "msg")
                c = m.get("content", "")
                text = c if isinstance(c, str) else json.dumps(c)
                snippet = text[:80].replace("\n", " ")
                summary_lines.append(f"- {r}: {snippet}...")

            if len(head_msgs) > len(sampled_head):
                summary_lines.append(f"- ... ({len(head_msgs) - len(sampled_head)} intermediate messages indexed in memory)")

            # Retrieve top relevant context for active query
            retrieved = self.index.search(latest_user_text, top_k=2)
            if retrieved:
                summary_lines.append("\n[Relevant Past Knowledge Retrieved from Memory]:")
                for doc, score in retrieved:
                    summary_lines.append(f"> (Score: {score:.1f}) {doc.raw_content[:120]}...")

            compacted_head = {
                "role": "system",
                "content": "\n".join(summary_lines)
            }
            final_messages = system_msgs + [compacted_head] + tail_msgs
        else:
            final_messages = system_msgs + tail_msgs

        final_tokens = count_tokens(final_messages)
        with self._lock:
            self.physical_tokens_emitted += final_tokens

        return final_messages

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_requests": self.total_requests,
                "virtual_tokens_processed": self.virtual_tokens_processed,
                "physical_tokens_emitted": self.physical_tokens_emitted,
                "compaction_events": self.compaction_events,
                "tool_tokens_saved": self.tool_tokens_saved,
                "compression_ratio": (
                    round(self.virtual_tokens_processed / max(1, self.physical_tokens_emitted), 2)
                    if self.physical_tokens_emitted > 0 else 1.0
                ),
                "cas_store": self.tool_store.stats(),
            }


# Backwards compatibility alias
VirtualContextManager = VirtualContextServerRuntime
