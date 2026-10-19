package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"strata/pkg/mathruntime"
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

func TestMathTheoremsEndpoints(t *testing.T) {
	srv := setupTestServer()
	ts := httptest.NewServer(srv.Router())
	defer ts.Close()

	// 1. GET /v1/strata/math/theorems
	resp1, err1 := http.Get(ts.URL + "/v1/strata/math/theorems")
	if err1 != nil || resp1.StatusCode != http.StatusOK {
		t.Fatalf("failed GET /v1/strata/math/theorems: %v", err1)
	}
	var listRes map[string]interface{}
	json.NewDecoder(resp1.Body).Decode(&listRes)
	if listRes["count"].(float64) < 5 {
		t.Errorf("expected at least 5 theorems listed")
	}

	// 2. POST /v1/strata/math/theorems/verify
	payload := `{"name":"fermat_little_theorem","args":["7","101"]}`
	resp2, err2 := http.Post(ts.URL+"/v1/strata/math/theorems/verify", "application/json", strings.NewReader(payload))
	if err2 != nil || resp2.StatusCode != http.StatusOK {
		t.Fatalf("failed POST /v1/strata/math/theorems/verify: %v", err2)
	}
	var verRes map[string]interface{}
	json.NewDecoder(resp2.Body).Decode(&verRes)
	if verRes["matches"] != true {
		t.Errorf("expected matches true for FLT (7, 101)")
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

// The MCP hub is gone: its tools run as interceptions now (intercept_tools_test.go). The old paths must not
// answer as MCP, and the server must not list "tools" for them.
func TestMcpEndpointsAreGone(t *testing.T) {
	srv := setupTestServer()
	ts := httptest.NewServer(srv.Router())
	defer ts.Close()

	for _, path := range []string{"/v1/mcp/tools", "/mcp"} {
		resp, err := http.Get(ts.URL + path)
		if err != nil {
			t.Fatal(err)
		}
		var body map[string]interface{}
		json.NewDecoder(resp.Body).Decode(&body)
		resp.Body.Close()
		if _, ok := body["tools"]; ok || body["servers"] != nil {
			t.Errorf("GET %s still answers as MCP: %v", path, body)
		}
	}
	resp, err := http.Post(ts.URL+"/v1/mcp", "application/json", strings.NewReader(`{"jsonrpc":"2.0","id":1,"method":"initialize"}`))
	if err != nil {
		t.Fatal(err)
	}
	var rpc map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&rpc)
	resp.Body.Close()
	if rpc["jsonrpc"] == "2.0" {
		t.Errorf("POST /v1/mcp still answers JSON-RPC: %v", rpc)
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

	// 5. POST /api/setup/resolve
	resolvePayload := `{"input": "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF"}`
	resp5, err := http.Post(ts.URL+"/api/setup/resolve", "application/json", strings.NewReader(resolvePayload))
	if err != nil || resp5.StatusCode != http.StatusOK {
		t.Fatalf("failed POST /api/setup/resolve: %v", err)
	}
	var resolveRes map[string]interface{}
	json.NewDecoder(resp5.Body).Decode(&resolveRes)
	if resolveRes["repo_id"] != "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF" {
		t.Errorf("expected repo_id in resolve response, got %v", resolveRes["repo_id"])
	}

	// 6. POST /api/setup/cancel
	resp6, err := http.Post(ts.URL+"/api/setup/cancel", "application/json", strings.NewReader("{}"))
	if err != nil || resp6.StatusCode != http.StatusOK {
		t.Fatalf("failed POST /api/setup/cancel: %v", err)
	}

	// 7. POST /api/setup/delete security rejection
	badDelete, err := http.Post(ts.URL+"/api/setup/delete", "application/json", strings.NewReader(`{"path": "/etc/passwd"}`))
	if err != nil || badDelete.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400 Bad Request for unsafe delete path, got %v", badDelete.StatusCode)
	}

	// 8. POST /api/setup/delete valid deletion
	os.MkdirAll("models", 0755)
	testModelFile := "models/test_server_delete.gguf"
	os.WriteFile(testModelFile, []byte("model payload bytes"), 0644)
	validDelete, err := http.Post(ts.URL+"/api/setup/delete", "application/json", strings.NewReader(`{"filename": "test_server_delete.gguf"}`))
	if err != nil || validDelete.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 OK for valid delete, got %v", validDelete.StatusCode)
	}
	var delRes map[string]interface{}
	json.NewDecoder(validDelete.Body).Decode(&delRes)
	if delRes["status"] != "deleted" {
		t.Errorf("expected status deleted, got %v", delRes["status"])
	}
	if _, err := os.Stat(testModelFile); !os.IsNotExist(err) {
		t.Errorf("expected test model file to be removed")
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

func TestCASEndpoints(t *testing.T) {
	srv := setupTestServer()
	ts := httptest.NewServer(srv.Router())
	defer ts.Close()

	// 1. GET /v1/strata/cas/constants
	resp1, err1 := http.Get(ts.URL + "/v1/strata/cas/constants")
	if err1 != nil || resp1.StatusCode != http.StatusOK {
		t.Fatalf("failed GET /v1/strata/cas/constants: %v", err1)
	}
	var constData map[string]interface{}
	json.NewDecoder(resp1.Body).Decode(&constData)
	if count, ok := constData["count"].(float64); !ok || count < 10 {
		t.Errorf("expected >= 10 constants, got %v", constData["count"])
	}

	// 2. GET /v1/strata/cas/formulas?domain=MECHANICS
	resp2, err2 := http.Get(ts.URL + "/v1/strata/cas/formulas?domain=MECHANICS")
	if err2 != nil || resp2.StatusCode != http.StatusOK {
		t.Fatalf("failed GET /v1/strata/cas/formulas: %v", err2)
	}
	var formMap map[string]interface{}
	json.NewDecoder(resp2.Body).Decode(&formMap)
	if count, ok := formMap["count"].(float64); !ok || count < 3 {
		t.Errorf("expected >= 3 mechanics formulas, got %v", formMap["count"])
	}

	// 3. POST /v1/strata/cas/solve (F = m*a)
	solvePayload := `{
		"domain": "MECHANICS",
		"formula_id": "newton_second_law",
		"variables": {
			"m": "10 kg",
			"a": "3 m/s^2"
		},
		"target_var": "F"
	}`
	resp3, err3 := http.Post(ts.URL+"/v1/strata/cas/solve", "application/json", strings.NewReader(solvePayload))
	if err3 != nil || resp3.StatusCode != http.StatusOK {
		t.Fatalf("failed POST /v1/strata/cas/solve: %v", err3)
	}
	var solveRes mathruntime.CASResult
	json.NewDecoder(resp3.Body).Decode(&solveRes)
	if solveRes.Status != mathruntime.StatusSuccess || !strings.Contains(solveRes.ExactResult, "30 N") {
		t.Errorf("expected 30 N from CAS solve, got %+v", solveRes)
	}
}

func TestModelSwitchEndpoint(t *testing.T) {
	srv := setupTestServer()
	ts := httptest.NewServer(srv.Router())
	defer ts.Close()

	// 1. Switch to a new model with abliterate enabled
	switchPayload := `{
		"model": "gpt-oss-20b",
		"abliterate": true,
		"abliteration_strength": 1.5
	}`
	resp, err := http.Post(ts.URL+"/api/models/switch", "application/json", strings.NewReader(switchPayload))
	if err != nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("failed POST /api/models/switch: %v", err)
	}

	var switchRes map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&switchRes)
	if switchRes["status"] != "switching" || switchRes["model"] != "gpt-oss-20b" {
		t.Errorf("unexpected switch response: %v", switchRes)
	}

	if srv.Config.ModelName != "gpt-oss-20b" {
		t.Errorf("expected server model name to be updated to gpt-oss-20b, got %s", srv.Config.ModelName)
	}

	if srv.ConfigManager != nil {
		cfg := srv.ConfigManager.Get()
		if !cfg.Abliterate {
			t.Errorf("expected abliterate to be true in config manager")
		}
		if cfg.AbliterationStrength != 1.5 {
			t.Errorf("expected abliteration_strength to be 1.5, got %v", cfg.AbliterationStrength)
		}
		if cfg.LastModel != "gpt-oss-20b" {
			t.Errorf("expected LastModel to be gpt-oss-20b, got %s", cfg.LastModel)
		}
	}

	// 2. Verify /health reflects the new model and uncensor settings
	hResp, err := http.Get(ts.URL + "/health")
	if err != nil || hResp.StatusCode != http.StatusOK {
		t.Fatalf("failed GET /health: %v", err)
	}
	var health map[string]interface{}
	json.NewDecoder(hResp.Body).Decode(&health)
	if health["model"] != "gpt-oss-20b" {
		t.Errorf("expected health.model to be gpt-oss-20b, got %v", health["model"])
	}
	if health["abliterate"] != true {
		t.Errorf("expected health.abliterate to be true, got %v", health["abliterate"])
	}
	if health["abliteration_strength"] != 1.5 {
		t.Errorf("expected health.abliteration_strength to be 1.5, got %v", health["abliteration_strength"])
	}
}

func TestLastSessionConfigRestore(t *testing.T) {
	// Create temporary directory for isolated config testing
	tmpDir, err := os.MkdirTemp("", "strata-last-session-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	origWd, _ := os.Getwd()
	if err := os.Chdir(tmpDir); err != nil {
		t.Fatalf("failed to chdir to temp dir: %v", err)
	}
	defer os.Chdir(origWd)

	// Write strata-config.json with LastConfigFile and LastModel
	dummyConfig := `{
		"last_config_file": "strata-gpt_oss_20b.json",
		"last_model": "GPT-OSS-20B-UD-Q4_K_XL",
		"max_context": 65536
	}`
	if err := os.WriteFile("strata-config.json", []byte(dummyConfig), 0644); err != nil {
		t.Fatalf("failed to write dummy config: %v", err)
	}

	// Initialize server with empty config file
	srv := NewStrataServer(ServerConfig{})
	if srv.Config.ConfigFile != "strata-gpt_oss_20b.json" {
		t.Errorf("expected ConfigFile to be restored to strata-gpt_oss_20b.json, got %s", srv.Config.ConfigFile)
	}
	if srv.Config.ModelName != "GPT-OSS-20B-UD-Q4_K_XL" {
		t.Errorf("expected ModelName to be restored to GPT-OSS-20B-UD-Q4_K_XL, got %s", srv.Config.ModelName)
	}
	if srv.Config.MaxContext != 65536 {
		t.Errorf("expected MaxContext to be 65536, got %d", srv.Config.MaxContext)
	}
}

// /health and /api/setup/status need no key, so with a key set they must not give out file paths or the engine's log
// text; the same request with the key still gets them.
func TestPublicStatusRoutesHidePathsWithoutTheKey(t *testing.T) {
	srv := NewStrataServer(ServerConfig{Port: 8080, APIKey: "secret-token-123", ModelName: "m", ModelPath: "/data/models/m.gguf", MaxContext: 4096})
	ts := httptest.NewServer(srv.Router())
	defer ts.Close()
	get := func(path string, auth bool) map[string]interface{} {
		req, _ := http.NewRequest("GET", ts.URL+path, nil)
		if auth {
			req.Header.Set("Authorization", "Bearer secret-token-123")
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil || resp.StatusCode != http.StatusOK {
			t.Fatalf("GET %s (auth %v): %v %v", path, auth, err, resp)
		}
		defer resp.Body.Close()
		var m map[string]interface{}
		json.NewDecoder(resp.Body).Decode(&m)
		return m
	}
	h := get("/health", false)
	for _, k := range publicHealthFields {
		if _, ok := h[k]; ok {
			t.Errorf("/health without the key has %q: %v", k, h)
		}
	}
	if h["engine"] == nil || h["loaded"] != false {
		t.Errorf("/health must still say whether the engine is up: %v", h)
	}
	if get("/health", true)["error"] == nil {
		t.Error("/health with the key lost the engine error")
	}
	st := get("/api/setup/status", false)
	if st["active_path"] != "" {
		t.Errorf("/api/setup/status without the key gives the model path: %v", st)
	}
	if get("/api/setup/status", true)["active_path"] != "/data/models/m.gguf" {
		t.Error("/api/setup/status with the key lost the model path")
	}
}

// `strata serve -config strata-<model>.json` gives the engine its argument spec; that file is not a run
// configuration. Saving the run configuration back to that path destroyed the owner's model config (2026-10-19,
// every field replaced by the server defaults), so the run config must go to strata-config.json instead.
func TestServeDoesNotOverwriteTheEngineSpec(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir) // the fallback run config (strata-config.json) lands here, not in the package directory
	specPath := filepath.Join(dir, "strata-model.json")
	spec := `{"args":["--pack","/packs/coder","--max-context","262144"]}`
	if err := os.WriteFile(specPath, []byte(spec), 0o600); err != nil {
		t.Fatal(err)
	}

	srv := NewStrataServer(ServerConfig{Port: 0, ConfigFile: specPath})
	if err := srv.ConfigManager.SaveToFile(""); err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(specPath); string(b) != spec {
		t.Fatalf("the engine spec was rewritten:\n%s", b)
	}
	if _, err := os.Stat(filepath.Join(dir, "strata-config.json")); err != nil {
		t.Errorf("the run config must be saved to strata-config.json: %v", err)
	}
}
