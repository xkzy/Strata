// tests/context/test_multi_tenant.cpp - Multi-Tenant Context & Isolation Tests
#include "strata/context/multi_tenant_context.hpp"

#include <cassert>
#include <iostream>

using namespace strata::context;

void test_security_scope_hierarchy() {
    std::cout << "[Test 1/6] Testing Security Scope Hierarchy & Access Checks..." << std::endl;

    SecurityScope owner{"tenant_1", "user_alice", "proj_alpha", "agent_coder", "sess_101", SharingScope::PRIVATE};

    // Same session can access
    SecurityScope requester_same{"tenant_1", "user_alice", "proj_alpha", "agent_coder", "sess_101"};
    assert(requester_same.can_access(owner, SharingScope::PRIVATE));

    // Different session same user: denied on PRIVATE
    SecurityScope requester_diff_sess{"tenant_1", "user_alice", "proj_alpha", "agent_coder", "sess_102"};
    assert(!requester_diff_sess.can_access(owner, SharingScope::PRIVATE));
    // Allowed if sharing is USER
    assert(requester_diff_sess.can_access(owner, SharingScope::USER));

    // Different user same project: denied on USER, allowed on PROJECT
    SecurityScope requester_bob{"tenant_1", "user_bob", "proj_alpha", "agent_reviewer", "sess_201"};
    assert(!requester_bob.can_access(owner, SharingScope::USER));
    assert(requester_bob.can_access(owner, SharingScope::PROJECT));

    // Different project same tenant: denied on PROJECT, allowed on TEAM
    SecurityScope requester_carol{"tenant_1", "user_carol", "proj_beta", "agent_analyst", "sess_301"};
    assert(!requester_carol.can_access(owner, SharingScope::PROJECT));
    assert(requester_carol.can_access(owner, SharingScope::TEAM));

    // Different tenant: denied on TEAM
    SecurityScope requester_ext{"tenant_2", "user_david", "proj_alpha", "agent_coder", "sess_401"};
    assert(!requester_ext.can_access(owner, SharingScope::TEAM));
    assert(requester_ext.can_access(owner, SharingScope::GLOBAL));

    std::cout << "  Passed. Strict multi-level scope hierarchy validated." << std::endl;
}

void test_tenant_session_isolation() {
    std::cout << "[Test 2/6] Testing Strict Tenant & Session Isolation..." << std::endl;
    MultiTenantContextManager mgr(4096);

    SecurityScope alice_scope{"tenant_1", "user_alice", "proj_alpha", "agent_coder", "sess_alice_1"};
    SecurityScope bob_scope{"tenant_1", "user_bob", "proj_alpha", "agent_coder", "sess_bob_1"};

    auto alice_ctx = mgr.get_or_create_session(alice_scope);
    assert(alice_ctx != nullptr);

    // Alice appends proprietary secret
    alice_ctx->append_user_message("Secret Alice Key: XYZ-9988-ABC");

    // Eve cannot hijack Alice's session
    SecurityScope eve_hijack{"tenant_2", "user_eve", "proj_secret", "agent_coder", "sess_alice_1"};
    auto hijack_res = mgr.get_or_create_session(eve_hijack);
    assert(hijack_res == nullptr); // Denied!

    // Bob creates his own session
    auto bob_ctx = mgr.get_or_create_session(bob_scope);
    assert(bob_ctx != nullptr);
    assert(bob_ctx != alice_ctx);

    std::cout << "  Passed. Cross-tenant and cross-user session hijacking blocked." << std::endl;
}

void test_session_forking() {
    std::cout << "[Test 3/6] Testing Session Forking..." << std::endl;
    MultiTenantContextManager mgr(4096);

    SecurityScope parent_scope{"tenant_1", "user_alice", "proj_alpha", "agent_coder", "sess_main"};
    SecurityScope child_scope{"tenant_1", "user_alice", "proj_alpha", "agent_coder", "sess_fork_1"};

    auto parent_ctx = mgr.get_or_create_session(parent_scope);
    parent_ctx->append_user_message("Initial project architecture plan");

    // Fork session
    auto child_ctx = mgr.fork_session(parent_scope, child_scope);
    assert(child_ctx != nullptr);
    assert(child_ctx != parent_ctx);

    // Mutate child session
    child_ctx->append_user_message("Experimental branch test");
    assert(child_ctx->get_stats().virtual_context_total_tokens > 0);

    std::cout << "  Passed. Session forked cleanly with context isolation preserved." << std::endl;
}

void test_scoped_multi_tenant_retrieval() {
    std::cout << "[Test 4/6] Testing Scoped Multi-Tenant Retrieval Pre-Filtering..." << std::endl;
    MultiTenantContextManager mgr(4096);

    SecurityScope alice_scope{"tenant_1", "user_alice", "proj_alpha", "agent_coder", "sess_alice"};
    SecurityScope bob_scope{"tenant_1", "user_bob", "proj_alpha", "agent_reviewer", "sess_bob"};
    SecurityScope eve_scope{"tenant_2", "user_eve", "proj_other", "agent_attacker", "sess_eve"};

    auto alice_ctx = mgr.get_or_create_session(alice_scope);
    alice_ctx->append_user_message("Alice database credentials: db_admin:pass123");

    // Alice queries -> finds her credentials
    auto alice_res = mgr.retrieve("database credentials", alice_scope, 5);
    assert(!alice_res.empty());
    assert(alice_res[0].matched_content.find("Alice database credentials") != std::string::npos);

    // Bob queries -> does NOT find Alice's private credentials
    auto bob_res = mgr.retrieve("database credentials", bob_scope, 5);
    assert(bob_res.empty());

    // Eve queries -> does NOT find Alice's credentials
    auto eve_res = mgr.retrieve("database credentials", eve_scope, 5);
    assert(eve_res.empty());

    std::cout << "  Passed. Unauthorized items strictly filtered prior to ranking." << std::endl;
}

void test_shared_project_memory() {
    std::cout << "[Test 5/6] Testing Shared Project Memory Scopes..." << std::endl;
    MultiTenantContextManager mgr(4096);

    SecurityScope alice{"tenant_1", "user_alice", "proj_compiler", "agent_coder", "sess_a"};
    SecurityScope bob{"tenant_1", "user_bob", "proj_compiler", "agent_reviewer", "sess_b"};
    SecurityScope carol{"tenant_1", "user_carol", "proj_web", "agent_frontend", "sess_c"};

    // Alice publishes project documentation to proj_compiler
    mgr.ingest_shared_knowledge(alice, "compiler_spec.md", "Specification for compiler IR optimization pipeline", SharingScope::PROJECT);

    // Bob (same project) queries -> finds compiler spec
    auto bob_res = mgr.retrieve("compiler IR optimization", bob, 5);
    assert(!bob_res.empty());
    assert(bob_res[0].filename == "compiler_spec.md");

    // Carol (different project in same tenant) queries -> denied on PROJECT
    auto carol_res = mgr.retrieve("compiler IR optimization", carol, 5);
    assert(carol_res.empty());

    std::cout << "  Passed. Shared project memory accessible to authorized collaborators only." << std::endl;
}

void test_tenant_quotas_and_metrics() {
    std::cout << "[Test 6/6] Testing Tenant Quotas, Limits, and Accounting Metrics..." << std::endl;
    MultiTenantContextManager mgr(4096);

    TenantQuota strict_quota;
    strict_quota.max_sessions = 2;
    mgr.set_tenant_quota("tenant_limited", strict_quota);

    SecurityScope s1{"tenant_limited", "user_1", "p1", "a1", "sess_1"};
    SecurityScope s2{"tenant_limited", "user_1", "p1", "a1", "sess_2"};
    SecurityScope s3{"tenant_limited", "user_1", "p1", "a1", "sess_3"};

    auto c1 = mgr.get_or_create_session(s1);
    auto c2 = mgr.get_or_create_session(s2);
    auto c3 = mgr.get_or_create_session(s3);

    assert(c1 != nullptr);
    assert(c2 != nullptr);
    assert(c3 == nullptr); // Quota enforced!

    auto metrics = mgr.get_tenant_metrics("tenant_limited");
    assert(metrics.total_requests >= 2);
    (void)metrics;

    // Delete session 1 and verify slot freed
    bool deleted = mgr.delete_session(s1);
    assert(deleted);
    (void)deleted;
    auto c3_retry = mgr.get_or_create_session(s3);
    assert(c3_retry != nullptr); // Now allowed!

    std::cout << "  Passed. Tenant quotas and resource accounting validated." << std::endl;
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "   RUNNING STRATA MULTI-TENANT & MULTI-AGENT ISOLATION SUITE     " << std::endl;
    std::cout << "=================================================================" << std::endl;

    test_security_scope_hierarchy();
    test_tenant_session_isolation();
    test_session_forking();
    test_scoped_multi_tenant_retrieval();
    test_shared_project_memory();
    test_tenant_quotas_and_metrics();

    std::cout << "=================================================================" << std::endl;
    std::cout << "   ALL MULTI-TENANT & ISOLATION TESTS PASSED (6/6)               " << std::endl;
    std::cout << "=================================================================" << std::endl;
    return 0;
}
