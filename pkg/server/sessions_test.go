// pkg/server/sessions_test.go - Comprehensive tests for chat session REST APIs.
package server

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"strata/pkg/session"
)

func setupTestServerWithSessions(t *testing.T) (*httptest.Server, string) {
	t.Helper()
	tmpDir, err := os.MkdirTemp("", "strata-server-sessions-*")
	if err != nil {
		t.Fatal(err)
	}

	cfg := ServerConfig{
		Port:       8080,
		ModelName:  "test-model",
		MaxContext: 2048,
		SessionDir: tmpDir,
	}

	srv := NewStrataServer(cfg)
	ts := httptest.NewServer(srv.Router())
	return ts, tmpDir
}

func TestSessionsRESTAPI(t *testing.T) {
	ts, tmpDir := setupTestServerWithSessions(t)
	defer ts.Close()
	defer os.RemoveAll(tmpDir)

	client := ts.Client()

	// 1. GET /api/sessions - Initially empty
	res, err := client.Get(ts.URL + "/api/sessions")
	if err != nil {
		t.Fatalf("GET /api/sessions failed: %v", err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("GET /api/sessions failed with %d", res.StatusCode)
	}
	var listResp struct {
		Sessions []session.SessionSummary `json:"sessions"`
		Total    int                      `json:"total"`
	}
	if err := json.NewDecoder(res.Body).Decode(&listResp); err != nil {
		t.Fatalf("unmarshal list failed: %v", err)
	}
	if listResp.Total != 0 || len(listResp.Sessions) != 0 {
		t.Fatalf("expected 0 sessions, got %d", listResp.Total)
	}

	// 2. POST /api/sessions - Create new session
	sessPayload := session.Session{
		ID:    "sess-test-1",
		Title: "Go Migration Chat",
		Messages: []session.Message{
			{Role: "user", Text: "How do we migrate python to go?"},
			{Role: "assistant", Text: "We compile native go packages."},
		},
	}
	payloadBytes, _ := json.Marshal(sessPayload)
	res, err = client.Post(ts.URL+"/api/sessions", "application/json", bytes.NewReader(payloadBytes))
	if err != nil {
		t.Fatalf("POST /api/sessions failed: %v", err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(res.Body)
		t.Fatalf("POST /api/sessions failed with %d: %s", res.StatusCode, string(body))
	}
	var created session.Session
	if err := json.NewDecoder(res.Body).Decode(&created); err != nil {
		t.Fatalf("unmarshal created session failed: %v", err)
	}
	if created.ID != "sess-test-1" || created.Title != "Go Migration Chat" || len(created.Messages) != 2 {
		t.Fatalf("unexpected created session: %+v", created)
	}

	// 3. GET /api/sessions/sess-test-1 - Retrieve full session
	res, err = client.Get(ts.URL + "/api/sessions/sess-test-1")
	if err != nil {
		t.Fatalf("GET /api/sessions/sess-test-1 failed: %v", err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("GET /api/sessions/sess-test-1 failed: %d", res.StatusCode)
	}
	var fetched session.Session
	if err := json.NewDecoder(res.Body).Decode(&fetched); err != nil {
		t.Fatalf("unmarshal fetched session failed: %v", err)
	}
	if fetched.Title != "Go Migration Chat" {
		t.Fatalf("expected title 'Go Migration Chat', got %q", fetched.Title)
	}

	// 4. PUT /api/sessions/sess-test-1 - Rename / update
	fetched.Title = "Renamed Chat"
	fetched.Messages = append(fetched.Messages, session.Message{Role: "user", Text: "Another message"})
	upBytes, _ := json.Marshal(fetched)
	req, _ := http.NewRequest(http.MethodPut, ts.URL+"/api/sessions/sess-test-1", bytes.NewReader(upBytes))
	req.Header.Set("Content-Type", "application/json")
	res, err = client.Do(req)
	if err != nil {
		t.Fatalf("PUT /api/sessions/sess-test-1 failed: %v", err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("PUT /api/sessions/sess-test-1 failed: %d", res.StatusCode)
	}

	// 5. POST /api/sessions/sess-test-1/fork - Fork branch
	forkPayload := map[string]interface{}{
		"new_session_id": "sess-test-forked",
		"message_index":  2,
		"new_title":      "Forked Branch",
	}
	forkBytes, _ := json.Marshal(forkPayload)
	res, err = client.Post(ts.URL+"/api/sessions/sess-test-1/fork", "application/json", bytes.NewReader(forkBytes))
	if err != nil {
		t.Fatalf("POST /api/sessions/.../fork failed: %v", err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(res.Body)
		t.Fatalf("POST /api/sessions/.../fork failed: %d: %s", res.StatusCode, string(body))
	}
	var forked session.Session
	if err := json.NewDecoder(res.Body).Decode(&forked); err != nil {
		t.Fatalf("unmarshal forked session failed: %v", err)
	}
	if forked.ID != "sess-test-forked" || len(forked.Messages) != 2 || forked.Title != "Forked Branch" {
		t.Fatalf("unexpected forked session: %+v", forked)
	}

	// 6. Search via GET /api/sessions?q=forked
	res, err = client.Get(ts.URL + "/api/sessions?q=forked")
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("search failed: %d", res.StatusCode)
	}
	var searchResp struct {
		Sessions []session.SessionSummary `json:"sessions"`
		Total    int                      `json:"total"`
	}
	json.NewDecoder(res.Body).Decode(&searchResp)
	if searchResp.Total != 1 || searchResp.Sessions[0].ID != "sess-test-forked" {
		t.Fatalf("expected 1 search result for 'forked', got %d", searchResp.Total)
	}

	// 7. Legacy Delete via GET /api/strata/sessions/delete?id=sess-test-forked
	res, err = client.Get(ts.URL + "/api/strata/sessions/delete?id=sess-test-forked")
	if err != nil {
		t.Fatalf("legacy delete failed: %v", err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("legacy delete failed: %d", res.StatusCode)
	}

	// Verify forked session is gone
	res, err = client.Get(ts.URL + "/api/sessions/sess-test-forked")
	if err != nil {
		t.Fatalf("get deleted session failed: %v", err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusNotFound {
		t.Fatalf("expected 404 for deleted session, got %d", res.StatusCode)
	}

	// 8. DELETE /api/sessions/sess-test-1 - Delete single session
	delReq, _ := http.NewRequest(http.MethodDelete, ts.URL+"/api/sessions/sess-test-1", nil)
	res, err = client.Do(delReq)
	if err != nil {
		t.Fatalf("DELETE /api/sessions/sess-test-1 failed: %v", err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("DELETE /api/sessions/sess-test-1 failed: %d", res.StatusCode)
	}

	// 9. DELETE /api/sessions - Clear all
	clearReq, _ := http.NewRequest(http.MethodDelete, ts.URL+"/api/sessions", nil)
	res, err = client.Do(clearReq)
	if err != nil {
		t.Fatalf("DELETE /api/sessions failed: %v", err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("DELETE /api/sessions failed: %d", res.StatusCode)
	}
}
