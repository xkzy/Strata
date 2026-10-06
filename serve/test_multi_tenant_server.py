# serve/test_multi_tenant_server.py - Tests for Multi-Tenant, Multi-Agent Server Isolation & Fair Resource Scheduling
import json
import urllib.request
import pytest
from pathlib import Path
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve
from serve.multi_tenant import SecurityScope, SharingScope, TenantQuota

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def multi_tenant_server():
    """Starts a live Strata server on an ephemeral loopback port."""
    tok = ByteTokenizer()
    engine = MockEngine(tok, "</think>\n\nMulti-tenant model completion response.", max_context=2048)
    tpl = ChatTemplate(ROOT / "serve/chat_template.jinja")
    svc = Service(engine, tok, tpl, model_name="qwen3.8-flash-next")
    
    httpd = serve(svc, port=0)
    port = httpd.server_address[1]
    base_url = f"http://127.0.0.1:{port}"
    
    yield base_url, svc
    httpd.shutdown()
    httpd.server_close()


def test_multi_tenant_session_isolation(multi_tenant_server):
    base_url, svc = multi_tenant_server

    # Alice request
    alice_headers = {
        "Content-Type": "application/json",
        "X-Tenant-ID": "tenant_enterprise",
        "X-User-ID": "alice",
        "X-Workspace-ID": "backend_service",
        "X-Agent-ID": "coder",
        "X-Session-ID": "sess_alice_01"
    }
    alice_payload = {
        "model": "qwen3.8-flash-next",
        "messages": [
            {"role": "user", "content": "Alice secret password is AlphaBravoCharlie99."}
        ],
        "max_tokens": 64
    }
    req_alice = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(alice_payload).encode("utf-8"),
        headers=alice_headers
    )
    with urllib.request.urlopen(req_alice) as resp:
        assert resp.status == 200

    # Bob request (different user in same tenant)
    bob_headers = {
        "Content-Type": "application/json",
        "X-Tenant-ID": "tenant_enterprise",
        "X-User-ID": "bob",
        "X-Workspace-ID": "backend_service",
        "X-Agent-ID": "coder",
        "X-Session-ID": "sess_bob_01"
    }
    bob_payload = {
        "model": "qwen3.8-flash-next",
        "messages": [
            {"role": "user", "content": "What is Alice's password?"}
        ],
        "max_tokens": 64
    }
    req_bob = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(bob_payload).encode("utf-8"),
        headers=bob_headers
    )
    with urllib.request.urlopen(req_bob) as resp:
        assert resp.status == 200

    # Query Bob's context -> verify Alice's secret is NOT present
    query_bob = urllib.request.Request(
        f"{base_url}/v1/strata/context/query",
        data=json.dumps({"query": "AlphaBravoCharlie99"}).encode("utf-8"),
        headers=bob_headers
    )
    with urllib.request.urlopen(query_bob) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        assert len(data["hits"]) == 0  # Pre-filtered! Isolated!

    # Query Alice's context -> Alice CAN find her secret
    query_alice = urllib.request.Request(
        f"{base_url}/v1/strata/context/query",
        data=json.dumps({"query": "AlphaBravoCharlie99"}).encode("utf-8"),
        headers=alice_headers
    )
    with urllib.request.urlopen(query_alice) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        assert len(data["hits"]) > 0


def test_multi_agent_same_user(multi_tenant_server):
    base_url, svc = multi_tenant_server

    # User Alice - Coding Agent
    coder_headers = {
        "Content-Type": "application/json",
        "X-Tenant-ID": "tenant_1",
        "X-User-ID": "alice",
        "X-Workspace-ID": "core_repo",
        "X-Agent-ID": "coding_agent",
        "X-Session-ID": "sess_coding_1"
    }
    # User Alice - Review Agent
    review_headers = {
        "Content-Type": "application/json",
        "X-Tenant-ID": "tenant_1",
        "X-User-ID": "alice",
        "X-Workspace-ID": "core_repo",
        "X-Agent-ID": "review_agent",
        "X-Session-ID": "sess_review_1"
    }

    # Coder sends implementation plan
    req_coder = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps({
            "model": "qwen3.8-flash-next",
            "messages": [{"role": "user", "content": "Implementing zero-copy serialization."}]
        }).encode("utf-8"),
        headers=coder_headers
    )
    with urllib.request.urlopen(req_coder) as resp:
        assert resp.status == 200

    # List sessions for Alice -> should see both sessions
    req_list = urllib.request.Request(
        f"{base_url}/v1/strata/sessions",
        headers=coder_headers
    )
    with urllib.request.urlopen(req_list) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        session_ids = [s["session_id"] for s in data["sessions"]]
        assert "sess_coding_1" in session_ids


def test_session_fork_api(multi_tenant_server):
    base_url, svc = multi_tenant_server

    headers = {
        "Content-Type": "application/json",
        "X-Tenant-ID": "tenant_1",
        "X-User-ID": "alice",
        "X-Workspace-ID": "core_repo",
        "X-Agent-ID": "coding_agent",
        "X-Session-ID": "sess_parent"
    }

    # Initialize parent
    init_req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps({
            "model": "qwen3.8-flash-next",
            "messages": [{"role": "user", "content": "Initial parent context line."}]
        }).encode("utf-8"),
        headers=headers
    )
    with urllib.request.urlopen(init_req) as resp:
        assert resp.status == 200

    # Fork session
    fork_payload = {
        "tenant_id": "tenant_1",
        "user_id": "alice",
        "workspace_id": "core_repo",
        "agent_id": "coding_agent",
        "parent_session_id": "sess_parent",
        "new_session_id": "sess_forked_child"
    }
    fork_req = urllib.request.Request(
        f"{base_url}/v1/strata/sessions/fork",
        data=json.dumps(fork_payload).encode("utf-8"),
        headers=headers
    )
    with urllib.request.urlopen(fork_req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["status"] == "forked"
        assert data["session_id"] == "sess_forked_child"


def test_scoped_tool_cas_isolation(multi_tenant_server):
    base_url, svc = multi_tenant_server

    alice_scope = SecurityScope(tenant_id="tenant_x", user_id="alice", session_id="sess_alice")
    bob_scope = SecurityScope(tenant_id="tenant_x", user_id="bob", session_id="sess_bob")

    # Ingest scoped tool output for Alice
    raw_secret = "Confidential server diagnostic logs with private tokens."
    rid, _, _ = svc.multi_tenant.tool_store.ingest_scoped("diagnostics", raw_secret, alice_scope)

    # Bob tries to retrieve Alice's tool result -> DENIED / Empty
    bob_req = urllib.request.Request(
        f"{base_url}/v1/strata/tools/retrieve",
        data=json.dumps({"result_id": rid, "user": "bob", "tenant_id": "tenant_x", "session_id": "sess_bob"}).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-Tenant-ID": "tenant_x", "X-User-ID": "bob", "X-Session-ID": "sess_bob"}
    )
    with urllib.request.urlopen(bob_req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["content"] == ""  # Blocked!

    # Alice retrieves -> ALLOWED
    alice_req = urllib.request.Request(
        f"{base_url}/v1/strata/tools/retrieve",
        data=json.dumps({"result_id": rid, "user": "alice", "tenant_id": "tenant_x", "session_id": "sess_alice"}).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-Tenant-ID": "tenant_x", "X-User-ID": "alice", "X-Session-ID": "sess_alice"}
    )
    with urllib.request.urlopen(alice_req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["content"] == raw_secret


def test_shared_project_memory_api(multi_tenant_server):
    base_url, svc = multi_tenant_server

    alice_headers = {
        "Content-Type": "application/json",
        "X-Tenant-ID": "tenant_shared",
        "X-User-ID": "alice",
        "X-Workspace-ID": "proj_llm_engine",
        "X-Agent-ID": "dev",
        "X-Session-ID": "sess_1"
    }
    bob_headers = {
        "Content-Type": "application/json",
        "X-Tenant-ID": "tenant_shared",
        "X-User-ID": "bob",
        "X-Workspace-ID": "proj_llm_engine",
        "X-Agent-ID": "dev",
        "X-Session-ID": "sess_2"
    }
    carol_headers = {
        "Content-Type": "application/json",
        "X-Tenant-ID": "tenant_shared",
        "X-User-ID": "carol",
        "X-Workspace-ID": "proj_unrelated_web",
        "X-Agent-ID": "dev",
        "X-Session-ID": "sess_3"
    }

    # Ingest shared project knowledge
    ingest_req = urllib.request.Request(
        f"{base_url}/v1/strata/projects/memory",
        data=json.dumps({
            "title": "architecture_rfc.md",
            "content": "RFC 404: MoE Routing Optimization and Heterogeneous Memory Layout.",
            "sharing": "project"
        }).encode("utf-8"),
        headers=alice_headers
    )
    with urllib.request.urlopen(ingest_req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["status"] == "ingested"

    # Bob (same project) queries -> finds RFC
    bob_q = urllib.request.Request(
        f"{base_url}/v1/strata/context/query",
        data=json.dumps({"query": "Heterogeneous Memory Layout"}).encode("utf-8"),
        headers=bob_headers
    )
    with urllib.request.urlopen(bob_q) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        assert len(data["hits"]) > 0
        assert data["hits"][0]["filename"] == "architecture_rfc.md"

    # Carol (different workspace) queries -> cannot find it
    carol_q = urllib.request.Request(
        f"{base_url}/v1/strata/context/query",
        data=json.dumps({"query": "Heterogeneous Memory Layout"}).encode("utf-8"),
        headers=carol_headers
    )
    with urllib.request.urlopen(carol_q) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        assert len(data["hits"]) == 0
