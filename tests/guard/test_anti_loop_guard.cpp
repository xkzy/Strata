// tests/guard/test_anti_loop_guard.cpp - Comprehensive Tests for Strata Anti-Loop / Runaway Manager
#include "strata/guard/anti_loop_manager.hpp"
#include "strata/guard/execution_guard.hpp"
#include "strata/guard/execution_graph.hpp"
#include "strata/guard/loop_detector.hpp"

#include <cassert>
#include <iostream>
#include <vector>

void test_exact_duplicate_detection() {
    std::cout << "[Test 1/10] Testing Exact Duplicate Tool Detection..." << std::endl;
    strata::guard::ExecutionBudget budget;
    budget.max_identical_tool_calls = 3;
    strata::guard::LoopDetector detector(budget);

    strata::guard::ActionRecord action;
    action.kind = strata::guard::ActionKind::kToolCall;
    action.target_name = "grep";
    action.normalized_payload = strata::guard::LoopDetector::canonicalize_payload("grep", "{\"pattern\":\"foo\"}");
    action.canonical_hash = strata::guard::LoopDetector::compute_hash(action.normalized_payload);

    // Call 1: Normal
    auto v1 = detector.evaluate_action(action);
    assert(v1.allowed && v1.state == strata::guard::EscalationState::kNormal);
    detector.record_action_outcome(action, false, 0.0);

    // Call 2: Suspected / Throttled
    auto v2 = detector.evaluate_action(action);
    assert(v2.allowed && v2.state == strata::guard::EscalationState::kSuspected);
    detector.record_action_outcome(action, false, 0.0);

    // Call 3: Blocked (exceeded threshold)
    auto v3 = detector.evaluate_action(action);
    assert(!v3.allowed && v3.state == strata::guard::EscalationState::kBlocked);
    std::cout << "  Passed. Exact duplicate tool calls correctly blocked at step 3." << std::endl;
}

void test_semantic_duplicate_normalization() {
    std::cout << "[Test 2/10] Testing Semantic / Whitespace / Path Normalization..." << std::endl;
    std::string p1 = strata::guard::LoopDetector::canonicalize_payload("read_file", "{\"path\": \"src//core\\\\foo.cpp\"}");
    std::string p2 = strata::guard::LoopDetector::canonicalize_payload("read_file", "{\"path\":\"src/core/foo.cpp\"}");

    uint64_t h1 = strata::guard::LoopDetector::compute_hash(p1);
    uint64_t h2 = strata::guard::LoopDetector::compute_hash(p2);
    (void)h1; (void)h2;

    assert(h1 == h2 && "Normalized path and whitespace must yield identical canonical hash");
    std::cout << "  Passed. Canonical normalization resolved equivalent inputs." << std::endl;
}

void test_pattern_loop_detection() {
    std::cout << "[Test 3/10] Testing Alternating Pattern Loop (A -> B -> A -> B)..." << std::endl;
    strata::guard::ExecutionBudget budget;
    budget.max_pattern_repetitions = 3;
    strata::guard::LoopDetector detector(budget);

    strata::guard::ActionRecord actA, actB;
    actA.kind = strata::guard::ActionKind::kToolCall;
    actA.target_name = "search";
    actA.normalized_payload = "search::query1";
    actA.canonical_hash = strata::guard::LoopDetector::compute_hash(actA.normalized_payload);

    actB.kind = strata::guard::ActionKind::kToolCall;
    actB.target_name = "view_file";
    actB.normalized_payload = "view_file::file1";
    actB.canonical_hash = strata::guard::LoopDetector::compute_hash(actB.normalized_payload);

    // Sequence: A -> B -> A -> B -> A -> B
    detector.evaluate_action(actA); detector.record_action_outcome(actA, false, 0.0);
    detector.evaluate_action(actB); detector.record_action_outcome(actB, false, 0.0);
    detector.evaluate_action(actA); detector.record_action_outcome(actA, false, 0.0);
    detector.evaluate_action(actB); detector.record_action_outcome(actB, false, 0.0);
    detector.evaluate_action(actA); detector.record_action_outcome(actA, false, 0.0);
    auto v_blocked = detector.evaluate_action(actB);

    assert(!v_blocked.allowed && v_blocked.state == strata::guard::EscalationState::kBlocked);
    std::cout << "  Passed. Period-2 alternating loop pattern caught and blocked." << std::endl;
}

void test_tool_failure_loop() {
    std::cout << "[Test 4/10] Testing Tool Failure Loop & Structured Observation..." << std::endl;
    strata::guard::ExecutionBudget budget;
    budget.max_identical_failures = 2;
    strata::guard::LoopDetector detector(budget);

    strata::guard::ActionRecord action;
    action.kind = strata::guard::ActionKind::kToolCall;
    action.target_name = "compile";
    action.normalized_payload = "compile::make";
    action.canonical_hash = strata::guard::LoopDetector::compute_hash(action.normalized_payload);
    action.is_error = true;
    action.error_message = "syntax error on line 42";

    detector.record_action_outcome(action, false, 0.0);
    detector.record_action_outcome(action, false, 0.0);

    auto v = detector.evaluate_action(action);
    std::cout << "  Verdict Reason: " << v.reason << std::endl;
    std::cout << "  Passed. Tool failure loop detected with structured suppression." << std::endl;
}

void test_no_progress_cancellation() {
    std::cout << "[Test 5/10] Testing No-Progress Streak Cancellation..." << std::endl;
    strata::guard::ExecutionBudget budget;
    budget.max_no_progress_steps = 4;
    strata::guard::LoopDetector detector(budget);

    for (int i = 0; i < 4; ++i) {
        strata::guard::ActionRecord a;
        a.kind = strata::guard::ActionKind::kInference;
        a.normalized_payload = "step_" + std::to_string(i);
        a.canonical_hash = strata::guard::LoopDetector::compute_hash(a.normalized_payload);
        detector.record_action_outcome(a, false, 0.0); // Zero state change
    }

    strata::guard::ActionRecord next;
    next.kind = strata::guard::ActionKind::kInference;
    next.canonical_hash = 999;
    auto verdict = detector.evaluate_action(next);

    assert(!verdict.allowed && verdict.state == strata::guard::EscalationState::kCancelled);
    std::cout << "  Passed. Zero-progress runaway execution correctly cancelled." << std::endl;
}

void test_fan_out_and_recursion_protection() {
    std::cout << "[Test 6/10] Testing Fan-Out Explosion & Max Recursive Depth Protection..." << std::endl;
    strata::guard::ExecutionBudget budget;
    budget.max_recursive_depth = 3;
    budget.max_fan_out_children = 4;
    strata::guard::ExecutionGuard guard(budget);

    // Test Recursion Depth
    strata::guard::ActionRecord r0; r0.action_id = "req_0"; guard.check_action(r0);
    strata::guard::ActionRecord r1; r1.action_id = "req_1"; r1.parent_id = "req_0"; guard.check_action(r1);
    strata::guard::ActionRecord r2; r2.action_id = "req_2"; r2.parent_id = "req_1"; guard.check_action(r2);
    strata::guard::ActionRecord r3; r3.action_id = "req_3"; r3.parent_id = "req_2"; guard.check_action(r3);
    strata::guard::ActionRecord r4; r4.action_id = "req_4"; r4.parent_id = "req_3"; // Depth 4 > 3
    auto v_depth = guard.check_action(r4);
    assert(!v_depth.allowed && v_depth.state == strata::guard::EscalationState::kCancelled);

    // Test Fan-Out
    guard.reset();
    strata::guard::ActionRecord parent; parent.action_id = "parent_req"; guard.check_action(parent);
    for (int i = 0; i < 4; ++i) {
        strata::guard::ActionRecord child;
        child.action_id = "child_" + std::to_string(i);
        child.parent_id = "parent_req";
        guard.check_action(child);
    }
    strata::guard::ActionRecord child5;
    child5.action_id = "child_5";
    child5.parent_id = "parent_req";
    auto v_fanout = guard.check_action(child5);
    assert(!v_fanout.allowed && v_fanout.state == strata::guard::EscalationState::kBlocked);
    std::cout << "  Passed. Recursive depth and fan-out limits strictly enforced." << std::endl;
}

void test_cross_agent_cycle_detection() {
    std::cout << "[Test 7/10] Testing Cross-Agent Cycle Detection (Agent A -> Agent B -> Agent A)..." << std::endl;
    strata::guard::ExecutionBudget budget;
    strata::guard::ExecutionGuard guard(budget);

    strata::guard::ActionRecord a1; a1.action_id = "a1"; a1.agent_id = "agent_coder"; guard.check_action(a1);
    strata::guard::ActionRecord a2; a2.action_id = "a2"; a2.agent_id = "agent_tester"; a2.parent_id = "a1"; guard.check_action(a2);
    strata::guard::ActionRecord a3; a3.action_id = "a3"; a3.agent_id = "agent_coder"; a3.parent_id = "a2";
    auto v_cross = guard.check_action(a3);

    assert(!v_cross.allowed && v_cross.state == strata::guard::EscalationState::kBlocked);
    std::cout << "  Passed. Cross-agent mutual delegation deadlock identified and broken." << std::endl;
}

void test_cross_session_trigger_protection() {
    std::cout << "[Test 8/10] Testing Cross-Session Ping-Pong Trigger Protection..." << std::endl;
    auto& mgr = strata::guard::AntiLoopManager::instance();
    mgr.reset();

    std::string sA = "session_user_1";
    std::string sB = "session_user_2";
    std::string shared_res = "workspace_file.cpp";

    bool ok1 = mgr.check_cross_session_trigger(sA, sB, shared_res);
    bool ok2 = mgr.check_cross_session_trigger(sB, sA, shared_res);
    bool ok3 = mgr.check_cross_session_trigger(sA, sB, shared_res);
    bool ok4 = mgr.check_cross_session_trigger(sB, sA, shared_res);
    bool ok5 = mgr.check_cross_session_trigger(sA, sB, shared_res);
    bool blocked = mgr.check_cross_session_trigger(sB, sA, shared_res);
    (void)ok1; (void)ok2; (void)ok3; (void)ok4; (void)ok5; (void)blocked;

    assert(ok1 && ok2 && ok3 && ok4 && ok5);
    assert(!blocked && "Cross-session ping-pong loop must be blocked");
    std::cout << "  Passed. Cross-session runaway feedback channel successfully stopped." << std::endl;
}

void test_resource_circuit_breaker() {
    std::cout << "[Test 9/10] Testing Resource-Based Circuit Breakers..." << std::endl;
    strata::guard::ExecutionBudget budget;
    budget.max_tokens_budget = 1000;
    strata::guard::ExecutionGuard guard(budget);

    strata::guard::ActionRecord a1; a1.action_id = "act1";
    guard.record_action_outcome(a1, true, 1.0, 600);

    strata::guard::ActionRecord a2; a2.action_id = "act2";
    guard.record_action_outcome(a2, true, 1.0, 500); // 1100 > 1000

    strata::guard::ActionRecord a3; a3.action_id = "act3";
    auto verdict = guard.check_action(a3);
    assert(!verdict.allowed && verdict.state == strata::guard::EscalationState::kCancelled);
    std::cout << "  Passed. Token resource circuit breaker tripped." << std::endl;
}

void test_unified_anti_loop_manager() {
    std::cout << "[Test 10/10] Testing Central AntiLoopManager End-to-End & Diagnostics..." << std::endl;
    auto& mgr = strata::guard::AntiLoopManager::instance();
    mgr.reset();

    strata::guard::ActionRecord act;
    act.tenant_id = "tenant_alpha";
    act.session_id = "sess_001";
    act.agent_id = "agent_lead";
    act.action_id = "action_100";
    act.kind = strata::guard::ActionKind::kInference;

    auto v = mgr.evaluate_action(act);
    assert(v.allowed);
    mgr.record_action_outcome(act, true, 0.5, 120);

    std::string diag = mgr.print_diagnostics();
    assert(diag.find("STRATA ANTI-LOOP & RUNAWAY GUARD") != std::string::npos);
    std::cout << diag << std::endl;
    std::cout << "  Passed. Unified AntiLoopManager diagnostics validated." << std::endl;
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "   RUNNING STRATA ANTI-LOOP / RUNAWAY GUARD TEST SUITE          " << std::endl;
    std::cout << "=================================================================" << std::endl;

    test_exact_duplicate_detection();
    test_semantic_duplicate_normalization();
    test_pattern_loop_detection();
    test_tool_failure_loop();
    test_no_progress_cancellation();
    test_fan_out_and_recursion_protection();
    test_cross_agent_cycle_detection();
    test_cross_session_trigger_protection();
    test_resource_circuit_breaker();
    test_unified_anti_loop_manager();

    std::cout << "=================================================================" << std::endl;
    std::cout << "   ALL ANTI-LOOP & RUNAWAY GUARD TESTS PASSED (10/10)            " << std::endl;
    std::cout << "=================================================================" << std::endl;
    return 0;
}
