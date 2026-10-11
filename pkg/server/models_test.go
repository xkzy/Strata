// pkg/server/models_test.go - Unit Tests for Multi-Model Endpoints and Registry Integration
package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestModelsEndpoint_ListAndDetail(t *testing.T) {
	srv := NewStrataServer(ServerConfig{
		Port:         0,
		ModelName:    "qwen3.8-flash-next",
		MaxContext:   32768,
		VirtualLimit: 2000000,
	})

	handler := srv.Router()

	// 1. Test GET /v1/models list
	req := httptest.NewRequest("GET", "/v1/models", nil)
	req.Host = "127.0.0.1:8080"
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected HTTP 200 on /v1/models, got %d: %s", w.Code, w.Body.String())
	}

	var listResp struct {
		Object string                   `json:"object"`
		Data   []map[string]interface{} `json:"data"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &listResp); err != nil {
		t.Fatalf("failed to decode /v1/models json: %v", err)
	}

	if listResp.Object != "list" {
		t.Fatalf("expected object 'list', got %s", listResp.Object)
	}

	modelIDs := make(map[string]bool)
	for _, m := range listResp.Data {
		id, _ := m["id"].(string)
		modelIDs[strings.ToLower(id)] = true
	}

	expectedBuiltins := []string{
		"qwen3.8-flash-next",
		"mimo-v2.6-flash-rl",
		"mamba-2.8b",
		"deepseek-v3",
	}

	for _, expected := range expectedBuiltins {
		if !modelIDs[expected] {
			t.Errorf("expected model %q in /v1/models, got IDs: %+v", expected, modelIDs)
		}
	}

	// 2. Test GET /v1/models/mimo-v2.6-flash-rl
	reqDetail := httptest.NewRequest("GET", "/v1/models/mimo-v2.6-flash-rl", nil)
	reqDetail.Host = "127.0.0.1:8080"
	wDetail := httptest.NewRecorder()
	handler.ServeHTTP(wDetail, reqDetail)

	if wDetail.Code != http.StatusOK {
		t.Fatalf("expected HTTP 200 on /v1/models/mimo-v2.6-flash-rl, got %d", wDetail.Code)
	}

	var detailResp map[string]interface{}
	if err := json.Unmarshal(wDetail.Body.Bytes(), &detailResp); err != nil {
		t.Fatalf("failed to decode detail json: %v", err)
	}

	if detailResp["id"] != "mimo-v2.6-flash-rl" {
		t.Fatalf("expected id 'mimo-v2.6-flash-rl', got %v", detailResp["id"])
	}
	if detailResp["family"] != "mimo" {
		t.Fatalf("expected family 'mimo', got %v", detailResp["family"])
	}
	if detailResp["architecture"] != "asymmetric_moe" {
		t.Fatalf("expected architecture 'asymmetric_moe', got %v", detailResp["architecture"])
	}

	// 3. Test alias resolution via GET /v1/models/mamba
	reqAlias := httptest.NewRequest("GET", "/v1/models/mamba", nil)
	reqAlias.Host = "127.0.0.1:8080"
	wAlias := httptest.NewRecorder()
	handler.ServeHTTP(wAlias, reqAlias)

	if wAlias.Code != http.StatusOK {
		t.Fatalf("expected HTTP 200 on /v1/models/mamba, got %d", wAlias.Code)
	}

	var aliasResp map[string]interface{}
	if err := json.Unmarshal(wAlias.Body.Bytes(), &aliasResp); err != nil {
		t.Fatalf("failed to decode alias detail json: %v", err)
	}

	if aliasResp["id"] != "mamba-2.8b" {
		t.Fatalf("expected alias 'mamba' to resolve to 'mamba-2.8b', got %v", aliasResp["id"])
	}
	if aliasResp["architecture"] != "state_space_model" {
		t.Fatalf("expected architecture 'state_space_model', got %v", aliasResp["architecture"])
	}
}
