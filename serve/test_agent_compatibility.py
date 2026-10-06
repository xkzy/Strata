# serve/test_agent_compatibility.py - Tests for Server-Side Agent Compatibility & Virtual Context Integration
import json
import urllib.request
import pytest
from pathlib import Path
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def live_server():
    """Starts a live Strata server on an ephemeral loopback port using MockEngine."""
    tok = ByteTokenizer()
    engine = MockEngine(tok, "</think>\n\nHello from Strata Inference Runtime!", max_context=1024)
    tpl = ChatTemplate(ROOT / "serve/chat_template.jinja")
    svc = Service(engine, tok, tpl, model_name="qwen3.8-flash-next")
    
    httpd = serve(svc, port=0)
    port = httpd.server_address[1]
    base_url = f"http://127.0.0.1:{port}"
    
    yield base_url, svc
    httpd.shutdown()
    httpd.server_close()


def test_openai_large_tool_output_agent_compatibility(live_server):
    base_url, svc = live_server

    # Agent sends a 50K-character (12K+ token) tool log to a 1024-token physical model
    large_log = "error: test failed\n" + "\n".join([f"log trace line {i}: details..." for i in range(1000)])
    payload = {
        "model": "qwen3.8-flash-next",
        "messages": [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Run tests."},
            {"role": "assistant", "content": "Running test suite..."},
            {"role": "tool", "name": "pytest", "content": large_log, "tool_call_id": "call_1"},
            {"role": "user", "content": "Fix the issue."}
        ],
        "max_tokens": 128
    }

    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    # Must NOT fail with 400 "prompt leaves no room to answer in the context"
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "choices" in data
        assert len(data["choices"]) > 0

    # Verify context diagnostics endpoint
    with urllib.request.urlopen(f"{base_url}/v1/strata/context") as resp:
        assert resp.status == 200
        stats = json.loads(resp.read().decode("utf-8"))
        assert stats["total_requests"] >= 1
        assert stats["tool_tokens_saved"] > 0

    # Verify tool store diagnostics endpoint
    with urllib.request.urlopen(f"{base_url}/v1/strata/tools") as resp:
        assert resp.status == 200
        tool_stats = json.loads(resp.read().decode("utf-8"))
        assert tool_stats["total_stored_entries"] >= 1


def test_anthropic_large_tool_result_agent_compatibility(live_server):
    base_url, svc = live_server

    large_build_output = "FAILED: src/engine.o\nerror: undefined symbol\n" + "\n".join([f"building object {i}..." for i in range(800)])
    payload = {
        "model": "qwen3.8-flash-next",
        "messages": [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_abc", "content": large_build_output}
            ]},
            {"role": "user", "content": "Diagnose the build failure."}
        ],
        "max_tokens": 128
    }

    req = urllib.request.Request(
        f"{base_url}/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "content" in data


def test_strata_tool_retrieve_endpoint(live_server):
    base_url, svc = live_server

    # Ingest a tool output
    raw = "line 1: header\nline 2: target info\nline 3: footer"
    rid, _, _ = svc.context_runtime.tool_store.ingest("fetch", raw)

    # Query full tool payload
    req = urllib.request.Request(
        f"{base_url}/v1/strata/tools/retrieve",
        data=json.dumps({"result_id": rid}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["result_id"] == rid
        assert data["content"] == raw

    # Query fragment line 2
    req_frag = urllib.request.Request(
        f"{base_url}/v1/strata/tools/retrieve",
        data=json.dumps({"result_id": rid, "start_line": 2, "end_line": 2}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req_frag) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["content"] == "line 2: target info"
