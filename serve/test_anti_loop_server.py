# serve/test_anti_loop_server.py - Tests for Server-Side Anti-Loop and Execution Guard
import unittest
from serve.anti_loop import (
    ActionKind,
    ActionRecord,
    AntiLoopManager,
    ErrorCategory,
    EscalationState,
    ExecutionBudget,
    ExecutionGuard,
    LoopDetector,
)


class TestAntiLoopGuard(unittest.TestCase):
    def setUp(self):
        AntiLoopManager.instance().reset()

    def test_canonical_payload_and_hashing(self):
        p1, h1 = LoopDetector.canonicalize_payload("read_file", {"path": "src//core\\foo.cpp"})
        p2, h2 = LoopDetector.canonicalize_payload("read_file", {"path": "src/core/foo.cpp"})
        self.assertEqual(h1, h2)
        self.assertIn("read_file::", p1)

    def test_exact_duplicate_detection(self):
        budget = ExecutionBudget(max_identical_tool_calls=3)
        detector = LoopDetector(budget)

        _, h = LoopDetector.canonicalize_payload("grep", {"pattern": "foo"})
        action = ActionRecord(action_id="act_1", kind=ActionKind.TOOL_CALL, target_name="grep", canonical_hash=h)

        v1 = detector.evaluate_action(action)
        self.assertTrue(v1.allowed)
        self.assertEqual(v1.state, EscalationState.NORMAL)
        detector.record_outcome(action, False, 0.0)

        v2 = detector.evaluate_action(action)
        self.assertTrue(v2.allowed)
        self.assertEqual(v2.state, EscalationState.SUSPECTED)
        self.assertTrue(v2.is_throttled)
        detector.record_outcome(action, False, 0.0)

        v3 = detector.evaluate_action(action)
        self.assertFalse(v3.allowed)
        self.assertEqual(v3.state, EscalationState.BLOCKED)

    def test_pattern_loop_detection(self):
        budget = ExecutionBudget(max_pattern_repetitions=3)
        detector = LoopDetector(budget)

        _, hA = LoopDetector.canonicalize_payload("search", "query1")
        _, hB = LoopDetector.canonicalize_payload("view", "file1")

        actA = ActionRecord(action_id="aA", kind=ActionKind.TOOL_CALL, canonical_hash=hA)
        actB = ActionRecord(action_id="aB", kind=ActionKind.TOOL_CALL, canonical_hash=hB)

        for _ in range(2):
            detector.evaluate_action(actA); detector.record_outcome(actA, False, 0.0)
            detector.evaluate_action(actB); detector.record_outcome(actB, False, 0.0)

        detector.evaluate_action(actA); detector.record_outcome(actA, False, 0.0)
        v = detector.evaluate_action(actB)
        self.assertFalse(v.allowed)
        self.assertEqual(v.state, EscalationState.BLOCKED)

    def test_tool_failure_loop(self):
        budget = ExecutionBudget(max_identical_failures=2)
        detector = LoopDetector(budget)

        _, h = LoopDetector.canonicalize_payload("compile", "main.cpp")
        action = ActionRecord(
            action_id="act_fail",
            kind=ActionKind.TOOL_CALL,
            target_name="compile",
            canonical_hash=h,
            is_error=True,
            error_message="syntax error: missing semicolon",
        )

        detector.record_outcome(action, False, 0.0)
        detector.record_outcome(action, False, 0.0)

        v = detector.evaluate_action(action)
        self.assertFalse(v.allowed)
        self.assertIn("Repeated failure detected", v.structured_observation)
        self.assertIn("missing semicolon", v.structured_observation)

    def test_no_progress_streak(self):
        budget = ExecutionBudget(max_no_progress_steps=3)
        detector = LoopDetector(budget)

        for i in range(3):
            act = ActionRecord(action_id=f"step_{i}", canonical_hash=f"h_{i}")
            detector.record_outcome(act, False, 0.0)

        act_next = ActionRecord(action_id="step_next", canonical_hash="h_next")
        v = detector.evaluate_action(act_next)
        self.assertFalse(v.allowed)
        self.assertEqual(v.state, EscalationState.CANCELLED)

    def test_recursion_depth_and_fan_out(self):
        budget = ExecutionBudget(max_recursive_depth=2, max_fan_out_children=2)
        guard = ExecutionGuard(budget)

        r0 = ActionRecord(action_id="r0")
        r1 = ActionRecord(action_id="r1", parent_id="r0")
        r2 = ActionRecord(action_id="r2", parent_id="r1")
        r3 = ActionRecord(action_id="r3", parent_id="r2") # Depth 3 > 2

        guard.check_action(r0)
        guard.check_action(r1)
        guard.check_action(r2)
        v = guard.check_action(r3)
        self.assertFalse(v.allowed)
        self.assertEqual(v.state, EscalationState.CANCELLED)

    def test_cross_agent_cycle(self):
        budget = ExecutionBudget()
        guard = ExecutionGuard(budget)

        a1 = ActionRecord(action_id="a1", agent_id="agent_alpha")
        a2 = ActionRecord(action_id="a2", agent_id="agent_beta", parent_id="a1")
        a3 = ActionRecord(action_id="a3", agent_id="agent_alpha", parent_id="a2")

        guard.check_action(a1)
        guard.check_action(a2)
        v = guard.check_action(a3)
        self.assertFalse(v.allowed)
        self.assertEqual(v.state, EscalationState.BLOCKED)
        self.assertIn("Cross-agent delegation cycle", v.reason)

    def test_cross_session_trigger(self):
        mgr = AntiLoopManager.instance()
        s1, s2 = "sess_1", "sess_2"
        res = "shared_db"

        ok1 = mgr.check_cross_session_trigger(s1, s2, res)
        ok2 = mgr.check_cross_session_trigger(s2, s1, res)
        ok3 = mgr.check_cross_session_trigger(s1, s2, res)
        ok4 = mgr.check_cross_session_trigger(s2, s1, res)
        ok5 = mgr.check_cross_session_trigger(s1, s2, res)
        blocked = mgr.check_cross_session_trigger(s2, s1, res)

        self.assertTrue(ok1 and ok2 and ok3 and ok4 and ok5)
        self.assertFalse(blocked)


if __name__ == "__main__":
    unittest.main()
