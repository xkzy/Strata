# serve/anti_loop.py - Server-Side Anti-Loop and Runaway Execution Manager
import hashlib
import json
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class EscalationState(str, Enum):
    NORMAL = "NORMAL"
    SUSPECTED = "SUSPECTED"
    THROTTLED = "THROTTLED"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"


class ActionKind(str, Enum):
    INFERENCE = "inference"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    RETRIEVAL = "retrieval"
    COMPACTION = "context_compaction"
    AGENT_DELEGATION = "agent_delegation"


class ErrorCategory(str, Enum):
    NONE = "NONE"
    TRANSIENT_NETWORK = "TRANSIENT_NETWORK"
    RATE_LIMITED = "RATE_LIMITED"
    SYNTAX_OR_SCHEMA = "SYNTAX_OR_SCHEMA"
    NOT_FOUND = "NOT_FOUND"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    EXECUTION_PANIC = "EXECUTION_PANIC"
    UNKNOWN = "UNKNOWN"


@dataclass
class ExecutionBudget:
    max_requests: int = 1000
    max_steps: int = 100
    max_tool_calls: int = 50
    max_retrievals: int = 40
    max_generations: int = 80
    max_wall_time_sec: float = 300.0
    max_tokens_budget: int = 2000000

    max_no_progress_steps: int = 5
    max_identical_tool_calls: int = 3
    max_identical_failures: int = 3
    max_identical_retrievals: int = 3
    max_pattern_repetitions: int = 3

    max_recursive_depth: int = 8
    max_fan_out_children: int = 16
    max_concurrent_operations: int = 8


@dataclass
class ActionRecord:
    action_id: str
    parent_id: str = ""
    tenant_id: str = "default_tenant"
    user_id: str = "default_user"
    agent_id: str = "primary_agent"
    session_id: str = "global_session"
    request_id: str = ""

    kind: ActionKind = ActionKind.INFERENCE
    target_name: str = ""
    normalized_payload: str = ""
    canonical_hash: str = ""
    state_fingerprint: str = ""

    timestamp_sec: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    is_error: bool = False
    error_category: ErrorCategory = ErrorCategory.NONE
    error_message: str = ""
    result_summary: str = ""


@dataclass
class GuardVerdict:
    state: EscalationState = EscalationState.NORMAL
    allowed: bool = True
    is_throttled: bool = False
    throttle_delay_ms: float = 0.0
    reason: str = ""
    suggested_remediation: str = ""
    structured_observation: str = ""
    progress_score: float = 1.0
    current_step: int = 0
    no_progress_streak: int = 0


class ExecutionGraph:
    """Directed Acyclic Graph tracking lineage, recursion depth, and cycles."""

    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.agent_delegations: Dict[str, List[str]] = defaultdict(list)

    def add_node(self, record: ActionRecord) -> bool:
        if not record.action_id:
            return False

        parent_depth = 0
        if record.parent_id and record.parent_id in self.nodes:
            parent_node = self.nodes[record.parent_id]
            parent_node["children"].append(record.action_id)
            parent_depth = parent_node["depth"] + 1

            parent_agent = parent_node["record"].agent_id
            if parent_agent and record.agent_id and parent_agent != record.agent_id:
                self.agent_delegations[parent_agent].append(record.agent_id)
        elif record.parent_id:
            parent_depth = 1

        self.nodes[record.action_id] = {
            "record": record,
            "children": [],
            "parent": record.parent_id,
            "depth": parent_depth,
        }
        return True

    def detect_cycle(self, action_id: str) -> Optional[List[str]]:
        if action_id not in self.nodes:
            return None

        visited: Set[str] = set()
        in_stack: Set[str] = set()
        path: List[str] = []

        def dfs(curr: str) -> Optional[List[str]]:
            visited.add(curr)
            in_stack.add(curr)
            path.append(curr)

            node = self.nodes.get(curr)
            if node:
                for child in node["children"]:
                    if child in in_stack:
                        path.append(child)
                        return list(path)
                    if child not in visited:
                        res = dfs(child)
                        if res:
                            return res

            path.pop()
            in_stack.remove(curr)
            return None

        return dfs(action_id)

    def detect_cross_agent_cycle(self, current_agent: str) -> Optional[List[str]]:
        if not current_agent or current_agent not in self.agent_delegations:
            return None

        visited: Set[str] = set()
        in_stack: Set[str] = set()
        path: List[str] = []

        def dfs(agent: str) -> Optional[List[str]]:
            visited.add(agent)
            in_stack.add(agent)
            path.append(agent)

            for nxt in self.agent_delegations.get(agent, []):
                if nxt in in_stack:
                    path.append(nxt)
                    return list(path)
                if nxt not in visited:
                    res = dfs(nxt)
                    if res:
                        return res

            path.pop()
            in_stack.remove(agent)
            return None

        return dfs(current_agent)

    def calculate_recursion_depth(self, action_id: str) -> int:
        node = self.nodes.get(action_id)
        return node["depth"] if node else 0

    def calculate_fan_out(self, parent_id: str) -> int:
        node = self.nodes.get(parent_id)
        return len(node["children"]) if node else 0

    def clear(self):
        self.nodes.clear()
        self.agent_delegations.clear()


class LoopDetector:
    """Canonical hashing, pattern cycle detection, and tool failure tracking."""

    def __init__(self, budget: ExecutionBudget):
        self.budget = budget
        self.history: deque = deque(maxlen=64)
        self.hash_history: deque = deque(maxlen=64)
        self.failure_counts: Dict[str, int] = defaultdict(int)
        self.last_failure_msg: Dict[str, str] = {}
        self.retrieval_counts: Dict[str, int] = defaultdict(int)
        self.consecutive_no_progress: int = 0
        self.current_progress_score: float = 1.0

    @staticmethod
    def normalize_text(text: str) -> str:
        s = re.sub(r"\s+", " ", text.strip().lower())
        s = s.replace("\\", "/")
        s = re.sub(r"/+", "/", s)
        return s

    @classmethod
    def canonicalize_payload(cls, target_name: str, payload: Any) -> Tuple[str, str]:
        if isinstance(payload, (dict, list)):
            try:
                norm_p = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            except Exception:
                norm_p = str(payload)
        else:
            norm_p = str(payload or "")

        norm_combined = f"{cls.normalize_text(target_name)}::{cls.normalize_text(norm_p)}"
        h = hashlib.sha256(norm_combined.encode("utf-8")).hexdigest()[:16]
        return norm_combined, h

    @classmethod
    def categorize_error(cls, err_msg: str) -> ErrorCategory:
        low = cls.normalize_text(err_msg)
        if any(x in low for x in ("timeout", "timed out", "connection reset", "econnreset")):
            return ErrorCategory.TRANSIENT_NETWORK
        if any(x in low for x in ("rate limit", "429", "too many requests")):
            return ErrorCategory.RATE_LIMITED
        if any(x in low for x in ("syntax", "parse error", "invalid json", "invalid argument", "unexpected token")):
            return ErrorCategory.SYNTAX_OR_SCHEMA
        if any(x in low for x in ("not found", "no such file", "enoent", "404")):
            return ErrorCategory.NOT_FOUND
        if any(x in low for x in ("permission", "access denied", "eacces", "unauthorized")):
            return ErrorCategory.PERMISSION_DENIED
        if any(x in low for x in ("panic", "segmentation fault", "abort")):
            return ErrorCategory.EXECUTION_PANIC
        return ErrorCategory.UNKNOWN

    def evaluate_action(self, action: ActionRecord) -> GuardVerdict:
        verdict = GuardVerdict()
        verdict.current_step = len(self.history) + 1
        verdict.no_progress_streak = self.consecutive_no_progress
        verdict.progress_score = self.current_progress_score

        # 1. Tool Failure Loop check (prioritized for structured observation)
        if action.canonical_hash in self.failure_counts:
            fails = self.failure_counts[action.canonical_hash]
            if fails >= self.budget.max_identical_failures:
                verdict.state = EscalationState.BLOCKED
                verdict.allowed = False
                verdict.reason = f"Repeated tool failure loop: failed {fails} times identically."
                verdict.structured_observation = (
                    f"Repeated failure detected.\nTool: {action.target_name}\n"
                    f"Attempts: {fails}\nError: {self.last_failure_msg.get(action.canonical_hash, '')}\n"
                    "Further identical execution suppressed. Please choose an alternative approach."
                )
                return verdict

        # 2. Pattern Cycle check (period 1 to 6)
        temp_seq = list(self.hash_history) + [action.canonical_hash]
        n = len(temp_seq)
        if n >= 4:
            for p in range(1, min(7, n // 2 + 1)):
                reps = 1
                match = True
                for rep in range(1, n // p):
                    for i in range(p):
                        curr_idx = n - 1 - i
                        prev_idx = n - 1 - i - (rep * p)
                        if temp_seq[curr_idx] != temp_seq[prev_idx]:
                            match = False
                            break
                    if match:
                        reps += 1
                    else:
                        break
                if reps >= self.budget.max_pattern_repetitions:
                    verdict.state = EscalationState.BLOCKED
                    verdict.allowed = False
                    verdict.reason = f"Cyclic pattern loop detected (period {p}, repeated {reps} times)."
                    verdict.suggested_remediation = "Break cyclic alternating dependency."
                    return verdict
                elif reps >= 2:
                    verdict.state = EscalationState.THROTTLED
                    verdict.is_throttled = True
                    verdict.throttle_delay_ms = 100.0

        # 3. Exact duplicate check
        repeat_count = 1
        for prev_h in reversed(self.hash_history):
            if prev_h == action.canonical_hash:
                repeat_count += 1
            else:
                break

        if repeat_count >= self.budget.max_identical_tool_calls:
            verdict.state = EscalationState.BLOCKED
            verdict.allowed = False
            verdict.reason = f"Identical operation repeated {repeat_count} times without progress."
            verdict.suggested_remediation = "Change arguments or select an alternative tool."
            return verdict
        elif repeat_count >= 2:
            verdict.state = EscalationState.SUSPECTED
            verdict.is_throttled = True
            verdict.throttle_delay_ms = 50.0 * repeat_count
            verdict.reason = f"Duplicate operation detected ({repeat_count}x)."

        # 4. Retrieval Loop check
        if action.kind == ActionKind.RETRIEVAL:
            ret_count = self.retrieval_counts.get(action.canonical_hash, 0)
            if ret_count >= self.budget.max_identical_retrievals:
                verdict.state = EscalationState.BLOCKED
                verdict.allowed = False
                verdict.reason = f"Redundant retrieval loop: query repeated {ret_count} times."
                return verdict

        # 5. No-Progress Streak
        if self.consecutive_no_progress >= self.budget.max_no_progress_steps:
            verdict.state = EscalationState.CANCELLED
            verdict.allowed = False
            verdict.reason = f"Execution cancelled: {self.consecutive_no_progress} steps with zero progress."
            return verdict

        return verdict

    def record_outcome(self, action: ActionRecord, state_changed: bool, progress_delta: float):
        self.history.append(action)
        self.hash_history.append(action.canonical_hash)

        if action.is_error:
            self.failure_counts[action.canonical_hash] += 1
            self.last_failure_msg[action.canonical_hash] = action.error_message
        else:
            self.failure_counts[action.canonical_hash] = 0

        if action.kind == ActionKind.RETRIEVAL:
            self.retrieval_counts[action.canonical_hash] += 1

        if state_changed or progress_delta > 0.0:
            self.consecutive_no_progress = 0
            self.current_progress_score = min(1.0, self.current_progress_score + max(0.1, progress_delta))
        else:
            self.consecutive_no_progress += 1
            self.current_progress_score = max(0.0, self.current_progress_score - 0.2)


class ExecutionGuard:
    """Server-side execution guard for a specific session/agent."""

    def __init__(self, budget: Optional[ExecutionBudget] = None):
        self.budget = budget or ExecutionBudget()
        self.graph = ExecutionGraph()
        self.detector = LoopDetector(self.budget)
        self.start_time = time.time()

        self.steps = 0
        self.tool_calls = 0
        self.retrievals = 0
        self.generations = 0
        self.tokens_used = 0

    def check_action(self, action: ActionRecord) -> GuardVerdict:
        # Check Circuit Breaker
        elapsed = time.time() - self.start_time
        if elapsed > self.budget.max_wall_time_sec:
            return GuardVerdict(
                state=EscalationState.CANCELLED,
                allowed=False,
                reason=f"Wall-clock timeout ({int(elapsed)}s > {int(self.budget.max_wall_time_sec)}s)",
            )
        if self.tokens_used > self.budget.max_tokens_budget:
            return GuardVerdict(
                state=EscalationState.CANCELLED,
                allowed=False,
                reason=f"Token budget exhausted ({self.tokens_used} > {self.budget.max_tokens_budget})",
            )
        if self.steps >= self.budget.max_steps:
            return GuardVerdict(
                state=EscalationState.CANCELLED,
                allowed=False,
                reason=f"Max execution steps reached ({self.steps} >= {self.budget.max_steps})",
            )

        # Graph checks
        self.graph.add_node(action)
        depth = self.graph.calculate_recursion_depth(action.action_id)
        if depth > self.budget.max_recursive_depth:
            return GuardVerdict(
                state=EscalationState.CANCELLED,
                allowed=False,
                reason=f"Max recursive depth exceeded ({depth} > {self.budget.max_recursive_depth})",
            )

        if action.parent_id:
            fan_out = self.graph.calculate_fan_out(action.parent_id)
            if fan_out > self.budget.max_fan_out_children:
                return GuardVerdict(
                    state=EscalationState.BLOCKED,
                    allowed=False,
                    reason=f"Fan-out explosion prevented: {fan_out} child operations",
                )

        if action.agent_id:
            cycle = self.graph.detect_cross_agent_cycle(action.agent_id)
            if cycle:
                return GuardVerdict(
                    state=EscalationState.BLOCKED,
                    allowed=False,
                    reason=f"Cross-agent delegation cycle detected: {' -> '.join(cycle)}",
                )

        # Evaluate pattern and repetition detector
        return self.detector.evaluate_action(action)

    def record_outcome(self, action: ActionRecord, state_changed: bool, progress_delta: float, tokens_used: int = 0):
        self.steps += 1
        self.tokens_used += tokens_used

        if action.kind == ActionKind.TOOL_CALL:
            self.tool_calls += 1
        elif action.kind == ActionKind.RETRIEVAL:
            self.retrievals += 1
        elif action.kind == ActionKind.INFERENCE:
            self.generations += 1

        self.detector.record_outcome(action, state_changed, progress_delta)


class AntiLoopManager:
    """Unified Server-Side Coordinator for Anti-Loop & Runaway Guards."""

    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = AntiLoopManager()
        return cls._instance

    def __init__(self, default_budget: Optional[ExecutionBudget] = None):
        self.default_budget = default_budget or ExecutionBudget()
        self.guards: Dict[str, ExecutionGuard] = {}
        self.cross_session_triggers: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)

    def _scope_key(self, tenant_id: str, session_id: str, agent_id: str) -> str:
        t = tenant_id or "default_tenant"
        s = session_id or "global_session"
        a = agent_id or "primary_agent"
        return f"{t}::{s}::{a}"

    def get_or_create_guard(self, tenant_id: str, session_id: str, agent_id: str = "") -> ExecutionGuard:
        key = self._scope_key(tenant_id, session_id, agent_id)
        if key not in self.guards:
            self.guards[key] = ExecutionGuard(self.default_budget)
        return self.guards[key]

    def evaluate_action(self, action: ActionRecord) -> GuardVerdict:
        guard = self.get_or_create_guard(action.tenant_id, action.session_id, action.agent_id)
        return guard.check_action(action)

    def record_action_outcome(self, action: ActionRecord, state_changed: bool, progress_delta: float, tokens_used: int = 0):
        guard = self.get_or_create_guard(action.tenant_id, action.session_id, action.agent_id)
        guard.record_outcome(action, state_changed, progress_delta, tokens_used)

    def check_cross_session_trigger(self, source_session: str, target_session: str, resource_key: str) -> bool:
        if not source_session or not target_session or not resource_key:
            return True

        now = time.time()
        history = self.cross_session_triggers[resource_key]
        self.cross_session_triggers[resource_key] = [r for r in history if (now - r[2]) < 60.0]

        count = sum(1 for r in self.cross_session_triggers[resource_key]
                    if (r[0] == target_session and r[1] == source_session) or
                       (r[0] == source_session and r[1] == target_session))

        self.cross_session_triggers[resource_key].append((source_session, target_session, now))
        return count <= 4

    def stats(self) -> Dict[str, Any]:
        total_evals = 0
        total_blocked = 0
        total_cancelled = 0
        for g in self.guards.values():
            total_evals += g.steps
            if g.detector.consecutive_no_progress >= g.budget.max_no_progress_steps:
                total_cancelled += 1

        return {
            "active_guards": len(self.guards),
            "total_evaluations": total_evals,
            "blocked_actions": total_blocked,
            "cancelled_runaways": total_cancelled,
        }

    def reset(self):
        self.guards.clear()
        self.cross_session_triggers.clear()
