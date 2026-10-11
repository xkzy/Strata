// pkg/session/session_test.go - Unit tests for session management and persistence.
package session

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

func TestSessionIDGenerationAndSanitization(t *testing.T) {
	id1 := GenerateID()
	id2 := GenerateID()
	if id1 == "" || id2 == "" || id1 == id2 {
		t.Fatalf("expected distinct valid IDs, got %q and %q", id1, id2)
	}

	clean, err := SanitizeID("sess-123_abc-XYZ")
	if err != nil || clean != "sess-123_abc-XYZ" {
		t.Fatalf("expected valid clean ID, got %q, err: %v", clean, err)
	}

	// Path traversal attempts must be rejected or cleaned
	traversalCases := []string{
		"../../etc/passwd",
		"../../../root",
		"sub/dir/id",
		"..\\..\\windows\\system32",
		"",
		" ",
	}

	for _, tc := range traversalCases {
		_, err := SanitizeID(tc)
		if err == nil {
			t.Errorf("SanitizeID(%q) expected error, but got nil", tc)
		}
	}
}

func TestSessionJSONCompatibility(t *testing.T) {
	// 1. camelCase from JS
	rawJS := `{"id":"sess-js","title":"JS Chat","createdAt":1700000000123,"updatedAt":1700000000456,"messages":[{"role":"user","text":"Hello world"}]}`
	var s1 Session
	if err := json.Unmarshal([]byte(rawJS), &s1); err != nil {
		t.Fatalf("unmarshal camelCase failed: %v", err)
	}
	if s1.CreatedAt != 1700000000123 || s1.UpdatedAt != 1700000000456 {
		t.Fatalf("unexpected timestamps: created=%d updated=%d", s1.CreatedAt, s1.UpdatedAt)
	}

	// 2. snake_case
	rawSnake := `{"id":"sess-py","title":"Py Chat","created_at":1700000000789,"updated_at":1700000000999,"messages":[]}`
	var s2 Session
	if err := json.Unmarshal([]byte(rawSnake), &s2); err != nil {
		t.Fatalf("unmarshal snake_case failed: %v", err)
	}
	if s2.CreatedAt != 1700000000789 || s2.UpdatedAt != 1700000000999 {
		t.Fatalf("unexpected timestamps: created=%d updated=%d", s2.CreatedAt, s2.UpdatedAt)
	}

	// 3. Marshal outputs both
	bytes, err := json.Marshal(&s1)
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}
	str := string(bytes)
	var verify map[string]interface{}
	if err := json.Unmarshal(bytes, &verify); err != nil {
		t.Fatalf("unmarshal back failed: %v", err)
	}
	if _, ok := verify["createdAt"]; !ok {
		t.Errorf("missing createdAt in output: %s", str)
	}
	if _, ok := verify["created_at"]; !ok {
		t.Errorf("missing created_at in output: %s", str)
	}

	// 4. Summary Preview
	s1.Messages = append(s1.Messages, Message{Role: "assistant", Text: "Hello! How can I help?"})
	summary := s1.Summary()
	if summary.MessageCount != 2 {
		t.Errorf("expected 2 messages, got %d", summary.MessageCount)
	}
	if summary.Preview != "Hello world" {
		t.Errorf("expected preview 'Hello world', got %q", summary.Preview)
	}
}

func TestStoreOperations(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "strata-session-test-*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tempDir)

	store, err := NewStore(tempDir)
	if err != nil {
		t.Fatalf("NewStore failed: %v", err)
	}

	// Initial list should be empty
	list, err := store.List()
	if err != nil || len(list) != 0 {
		t.Fatalf("expected empty list, got len=%d err=%v", len(list), err)
	}

	// Save session 1
	sess1 := &Session{
		ID:        "sess-1",
		Title:     "First Session",
		CreatedAt: 1000,
		UpdatedAt: 1000,
		Messages: []Message{
			{Role: "user", Text: "First message"},
			{Role: "assistant", Text: "Response 1"},
		},
		Tags: []string{"test", "alpha"},
	}
	if err := store.Save(sess1); err != nil {
		t.Fatalf("save sess1 failed: %v", err)
	}

	// Verify file exists on disk
	expectedFile := filepath.Join(tempDir, "sess-1.json")
	if _, err := os.Stat(expectedFile); err != nil {
		t.Fatalf("session file %s does not exist: %v", expectedFile, err)
	}

	// Save session 2 (newer)
	time.Sleep(10 * time.Millisecond)
	sess2 := &Session{
		ID:        "sess-2",
		Title:     "Second Session",
		CreatedAt: 2000,
		UpdatedAt: 2000,
		Messages: []Message{
			{Role: "user", Text: "Second message"},
		},
		Tags: []string{"beta"},
	}
	if err := store.Save(sess2); err != nil {
		t.Fatalf("save sess2 failed: %v", err)
	}

	// List should return 2 sessions, sess2 first (since updated_at is newer)
	list, err = store.List()
	if err != nil || len(list) != 2 {
		t.Fatalf("expected 2 sessions, got len=%d err=%v", len(list), err)
	}
	if list[0].ID != "sess-2" || list[1].ID != "sess-1" {
		t.Fatalf("expected order [sess-2, sess-1], got [%s, %s]", list[0].ID, list[1].ID)
	}

	// Get session 1
	got1, err := store.Get("sess-1")
	if err != nil {
		t.Fatalf("get sess-1 failed: %v", err)
	}
	if got1.Title != "First Session" || len(got1.Messages) != 2 {
		t.Fatalf("unexpected session content: %+v", got1)
	}

	// Fork session 1 (at turn 1)
	forked, err := store.Fork("sess-1", "sess-1-fork", 1, "Forked Alpha")
	if err != nil {
		t.Fatalf("fork failed: %v", err)
	}
	if forked.ID != "sess-1-fork" || forked.Title != "Forked Alpha" || len(forked.Messages) != 1 {
		t.Fatalf("unexpected forked session: %+v", forked)
	}
	if forked.Messages[0].Text != "First message" {
		t.Fatalf("expected message 'First message', got %q", forked.Messages[0].Text)
	}

	// Search
	results, err := store.Search("second")
	if err != nil || len(results) != 1 || results[0].ID != "sess-2" {
		t.Fatalf("search 'second' expected sess-2, got %+v", results)
	}

	results, err = store.Search("alpha")
	if err != nil || len(results) < 1 {
		t.Fatalf("search tag 'alpha' failed, got %+v", results)
	}

	// Delete session 1
	deleted, err := store.Delete("sess-1")
	if err != nil || !deleted {
		t.Fatalf("delete sess-1 failed: deleted=%v err=%v", deleted, err)
	}
	_, err = store.Get("sess-1")
	if err == nil {
		t.Fatalf("expected error getting deleted session")
	}

	// Clear remaining
	if err := store.Clear(); err != nil {
		t.Fatalf("clear failed: %v", err)
	}
	list, err = store.List()
	if err != nil || len(list) != 0 {
		t.Fatalf("expected 0 sessions after clear, got len=%d", len(list))
	}
}

func TestStoreConcurrency(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "strata-session-concurrent-*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tempDir)

	store, err := NewStore(tempDir)
	if err != nil {
		t.Fatal(err)
	}

	var wg sync.WaitGroup
	workers := 10
	iterations := 20

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				sid := fmt.Sprintf("worker-%d-sess-%d", workerID, j)
				sess := &Session{
					ID:    sid,
					Title: fmt.Sprintf("Chat %d-%d", workerID, j),
					Messages: []Message{
						{Role: "user", Text: fmt.Sprintf("Query %d", j)},
					},
				}
				if err := store.Save(sess); err != nil {
					t.Errorf("worker %d save failed: %v", workerID, err)
				}
				_, _ = store.Get(sid)
				_, _ = store.List()
			}
		}(i)
	}

	wg.Wait()

	list, err := store.List()
	if err != nil {
		t.Fatalf("list failed: %v", err)
	}
	if len(list) != workers*iterations {
		t.Fatalf("expected %d sessions, got %d", workers*iterations, len(list))
	}
}
