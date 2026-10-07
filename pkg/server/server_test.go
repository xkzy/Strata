package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func setupTestServer() *StrataServer {
	cfg := ServerConfig{
		Port:         8080,
		ModelName:    "Qwen3.8-Flash-Next",
		MaxContext:   32768,
		VirtualLimit: 2000000,
	}
	return NewStrataServer(cfg)
}

func TestHealthAndMetricsEndpoints(t *testing.T) {
	srv := setupTestServer()
	ts := httptest.NewServer(srv.Router())
	defer ts.Close()

	// Test /health
	resp, err := http.Get(ts.URL + "/health")
	if err != nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("failed /health: %v", err)
	}

	var health map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&health)
	// no model is configured in this test: the server is up but says plainly that the model is not
	if health["status"] != "degraded" || health["loaded"] != false || health["error"] == nil {
		t.Errorf("an engine that is not running must be reported, got %v", health)
	}

	// Test /metrics
	resp2, err := http.Get(ts.URL + "/metrics")
	if err != nil || resp2.StatusCode != http.StatusOK {
		t.Fatalf("failed /metrics: %v", err)
	}

	var metrics map[string]interface{}
	json.NewDecoder(resp2.Body).Decode(&metrics)
	if _, ok := metrics["math"]; !ok {
		t.Errorf("expected 'math' section in /metrics")
	}
	if _, ok := metrics["hardware_static"]; !ok {
		t.Errorf("expected 'hardware_static' section in /metrics")
	}
	if _, ok := metrics["hardware"]; !ok {
		t.Errorf("expected 'hardware' section in /metrics")
	}
	if _, ok := metrics["live"]; !ok {
		t.Errorf("expected 'live' section in /metrics")
	}
}

func TestMathEvaluateEndpoint(t *testing.T) {
	srv := setupTestServer()
	ts := httptest.NewServer(srv.Router())
	defer ts.Close()

	// Exact rational calculation: 1/3 + 1/6 -> 1/2
	payload := `{"operation":"evaluate","expression":"1/3 + 1/6","mode":"exact"}`
	resp, err := http.Post(ts.URL+"/v1/strata/math/evaluate", "application/json", strings.NewReader(payload))
	if err != nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("failed math evaluate: %v", err)
	}

	var res map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&res)
	if res["exact_result"] != "1/2" {
		t.Errorf("expected 1/2, got %v", res["exact_result"])
	}

	// Matrix Determinant
	payload2 := `{"operation":"determinant","expression":"[[1,2],[3,4]]"}`
	resp2, _ := http.Post(ts.URL+"/v1/strata/math/evaluate", "application/json", strings.NewReader(payload2))
	var res2 map[string]interface{}
	json.NewDecoder(resp2.Body).Decode(&res2)
	if res2["exact_result"] != "-2" {
		t.Errorf("expected -2, got %v", res2["exact_result"])
	}
}

func TestMathVerifyEndpoint(t *testing.T) {
	srv := setupTestServer()
	ts := httptest.NewServer(srv.Router())
	defer ts.Close()

	payload := `{"llm_output":"The product is 17381744.","expected_expression":"2384 * 7291"}`
	resp, err := http.Post(ts.URL+"/v1/strata/math/verify", "application/json", strings.NewReader(payload))
	if err != nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("failed math verify: %v", err)
	}

	var res map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&res)
	if res["matches"] != true {
		t.Errorf("expected matches true")
	}
}

func TestMathInterceptEndpoint(t *testing.T) {
	srv := setupTestServer()
	ts := httptest.NewServer(srv.Router())
	defer ts.Close()

	payload := `{"text":"Please calculate 2384 * 7291."}`
	resp, err := http.Post(ts.URL+"/v1/strata/math/intercept", "application/json", strings.NewReader(payload))
	if err != nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("failed math intercept: %v", err)
	}

	var res map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&res)
	if res["intercepted_count"].(float64) < 1 {
		t.Errorf("expected at least 1 intercepted expression")
	}
}

func TestLogicVerifyEndpoints(t *testing.T) {
	srv := setupTestServer()
	ts := httptest.NewServer(srv.Router())
	defer ts.Close()

	// 1. POST /v1/strata/verify
	claimJSON := `{
		"claim_id": "api-test-1",
		"type": "proposition",
		"premises": ["P -> Q", "P"],
		"claimed_value": "Q"
	}`
	resp, err := http.Post(ts.URL+"/v1/strata/verify", "application/json", strings.NewReader(claimJSON))
	if err != nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("failed /v1/strata/verify: %v", err)
	}

	var res map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&res)
	if res["status"] != "PASS" {
		t.Errorf("expected PASS for Modus Ponens verification, got %v", res["status"])
	}

	// 2. POST /v1/strata/verify/claims
	claimsPayload := `{
		"text": "The result is 2384 * 7291 = 17381744. Also 1000 m == 1 km."
	}`
	resp2, err2 := http.Post(ts.URL+"/v1/strata/verify/claims", "application/json", strings.NewReader(claimsPayload))
	if err2 != nil || resp2.StatusCode != http.StatusOK {
		t.Fatalf("failed /v1/strata/verify/claims: %v", err2)
	}

	var res2 map[string]interface{}
	json.NewDecoder(resp2.Body).Decode(&res2)
	if res2["claim_count"].(float64) < 1 {
		t.Errorf("expected at least 1 extracted claim")
	}

	// 3. GET /v1/strata/verify/metrics
	resp3, err3 := http.Get(ts.URL + "/v1/strata/verify/metrics")
	if err3 != nil || resp3.StatusCode != http.StatusOK {
		t.Fatalf("failed /v1/strata/verify/metrics: %v", err3)
	}

	var metrics map[string]interface{}
	json.NewDecoder(resp3.Body).Decode(&metrics)
	if metrics["total_verifications"].(float64) == 0 {
		t.Errorf("expected > 0 total_verifications in metrics")
	}
}

func TestMcpEndpoints(t *testing.T) {
	srv := setupTestServer()
	ts := httptest.NewServer(srv.Router())
	defer ts.Close()

	// 1. GET /v1/mcp/tools
	resp, err := http.Get(ts.URL + "/v1/mcp/tools")
	if err != nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("failed /v1/mcp/tools: %v", err)
	}
	var toolsResp map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&toolsResp)
	tools := toolsResp["tools"].([]interface{})
	if len(tools) < 4 {
		t.Fatalf("expected >= 4 tools, got %d", len(tools))
	}

	// 2. POST /v1/mcp (initialize)
	initJSON := `{"jsonrpc":"2.0","id":1,"method":"initialize"}`
	resp2, err2 := http.Post(ts.URL+"/v1/mcp", "application/json", strings.NewReader(initJSON))
	if err2 != nil || resp2.StatusCode != http.StatusOK {
		t.Fatalf("failed /v1/mcp initialize: %v", err2)
	}

	var mcpResp map[string]interface{}
	json.NewDecoder(resp2.Body).Decode(&mcpResp)
	if mcpResp["jsonrpc"] != "2.0" {
		t.Errorf("expected jsonrpc 2.0")
	}
}

func TestResponsesEndpoint(t *testing.T) {
	srv := setupTestServer()
	ts := httptest.NewServer(srv.Router())
	defer ts.Close()

	payload := `{
		"model": "Qwen3.8-Flash-Next",
		"input": [{"role": "user", "content": "hello"}]
	}`
	resp, err := http.Post(ts.URL+"/v1/responses", "application/json", strings.NewReader(payload))
	if err != nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("failed /v1/responses: %v", err)
	}

	var res map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&res)
	if res["object"] != "response" || res["status"] != "completed" {
		t.Errorf("expected completed response object, got %+v", res)
	}
}

func TestTelemetryEndpoint(t *testing.T) {
	srv := setupTestServer()
	ts := httptest.NewServer(srv.Router())
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/v1/telemetry")
	if err != nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("failed /v1/telemetry: %v", err)
	}

	var res map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&res)
	if _, ok := res["current"]; !ok {
		t.Errorf("expected 'current' field in telemetry response")
	}
}

func TestConfigEndpoint(t *testing.T) {
	srv := setupTestServer()
	ts := httptest.NewServer(srv.Router())
	defer ts.Close()

	// GET /config
	resp, err := http.Get(ts.URL + "/config")
	if err != nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("failed GET /config: %v", err)
	}
	var cfg map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&cfg)
	if cfg["model"] != "Qwen3.8-Flash-Next" {
		t.Errorf("expected model Qwen3.8-Flash-Next, got %v", cfg["model"])
	}

	// POST /config
	updatePayload := `{"model": "Qwen3.8-Coder-Next", "port": 8080}`
	resp2, err2 := http.Post(ts.URL+"/config", "application/json", strings.NewReader(updatePayload))
	if err2 != nil || resp2.StatusCode != http.StatusOK {
		t.Fatalf("failed POST /config: %v", err2)
	}
	var res2 map[string]interface{}
	json.NewDecoder(resp2.Body).Decode(&res2)
	if res2["status"] != "updated" {
		t.Errorf("expected status updated, got %v", res2["status"])
	}
}
func TestSetupEndpoints(t *testing.T) {
	srv := setupTestServer()
	ts := httptest.NewServer(srv.Router())
	defer ts.Close()

	// 1. GET /api/setup/hardware
	resp, err := http.Get(ts.URL + "/api/setup/hardware")
	if err != nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("failed GET /api/setup/hardware: %v", err)
	}
	var hw map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&hw)
	if hw["cpu_cores"].(float64) <= 0 {
		t.Errorf("expected positive cpu_cores")
	}

	// 2. GET /api/setup/models
	resp2, err := http.Get(ts.URL + "/api/setup/models")
	if err != nil || resp2.StatusCode != http.StatusOK {
		t.Fatalf("failed GET /api/setup/models: %v", err)
	}
	var modelsResp map[string]interface{}
	json.NewDecoder(resp2.Body).Decode(&modelsResp)
	modelsList := modelsResp["models"].([]interface{})
	if len(modelsList) == 0 {
		t.Errorf("expected non-empty models list")
	}

	// 3. POST /api/setup/configure
	cfgPayload := `{"model_name": "Qwen3.8-Flash-Next-Q4_K_M", "max_context": 65536, "virtual_limit": 1048576}`
	resp3, err := http.Post(ts.URL+"/api/setup/configure", "application/json", strings.NewReader(cfgPayload))
	if err != nil || resp3.StatusCode != http.StatusOK {
		t.Fatalf("failed POST /api/setup/configure: %v", err)
	}
	var cfgRes map[string]interface{}
	json.NewDecoder(resp3.Body).Decode(&cfgRes)
	if cfgRes["status"] != "configured" {
		t.Errorf("expected status configured, got %v", cfgRes["status"])
	}

	// 4. GET /api/setup/status
	resp4, err := http.Get(ts.URL + "/api/setup/status")
	if err != nil || resp4.StatusCode != http.StatusOK {
		t.Fatalf("failed GET /api/setup/status: %v", err)
	}
	var statusRes map[string]interface{}
	json.NewDecoder(resp4.Body).Decode(&statusRes)
	if statusRes["active_model"] != "Qwen3.8-Flash-Next-Q4_K_M" {
		t.Errorf("expected active model updated, got %v", statusRes["active_model"])
	}

	// 5. POST /api/setup/cancel
	resp5, err := http.Post(ts.URL+"/api/setup/cancel", "application/json", strings.NewReader("{}"))
	if err != nil || resp5.StatusCode != http.StatusOK {
		t.Fatalf("failed POST /api/setup/cancel: %v", err)
	}
}

func TestSystemEndpoints(t *testing.T) {
	srv := setupTestServer()
	ts := httptest.NewServer(srv.Router())
	defer ts.Close()

	// /api/slots
	resp, _ := http.Get(ts.URL + "/api/slots")
	if resp.StatusCode != http.StatusOK {
		t.Errorf("expected 200 for /api/slots")
	}

	// /api/cache
	resp2, _ := http.Get(ts.URL + "/api/cache")
	if resp2.StatusCode != http.StatusOK {
		t.Errorf("expected 200 for /api/cache")
	}

	// /api/model
	resp3, _ := http.Get(ts.URL + "/api/model")
	if resp3.StatusCode != http.StatusOK {
		t.Errorf("expected 200 for /api/model")
	}

	// /api/cancel
	resp4, _ := http.Post(ts.URL+"/api/cancel", "application/json", strings.NewReader("{}"))
	if resp4.StatusCode != http.StatusOK {
		t.Errorf("expected 200 for /api/cancel")
	}
}

func TestAPIKeyAuthorization(t *testing.T) {
	cfg := ServerConfig{
		Port:       8080,
		APIKey:     "secret-token-123",
		ModelName:  "Qwen3.8-Flash-Next",
		MaxContext: 32768,
	}
	srv := NewStrataServer(cfg)
	ts := httptest.NewServer(srv.Router())
	defer ts.Close()

	// Request without token -> 401
	resp, _ := http.Post(ts.URL+"/v1/completions", "application/json", strings.NewReader(`{"prompt":"hi"}`))
	if resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("expected 401 Unauthorized, got %d", resp.StatusCode)
	}

	// Request with a valid token passes authorization (this test has no model, so the answer is 503, not 401)
	req, _ := http.NewRequest("POST", ts.URL+"/v1/completions", strings.NewReader(`{"prompt":"hi"}`))
	req.Header.Set("Authorization", "Bearer secret-token-123")
	req.Header.Set("Content-Type", "application/json")
	resp2, err := http.DefaultClient.Do(req)
	if err != nil || resp2.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("expected 503 (authorized, but no model) with a valid bearer token, got %v %v", resp2, err)
	}
}
