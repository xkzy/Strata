package virtualcontext

import (
	"fmt"
	"strings"
	"testing"
)

func TestMemoryRecallsByWordsAndOnlyWithinTheScope(t *testing.T) {
	m := NewConversationMemory(100, 50)
	m.Remember("alice|s1", "user", "My dog is called Biscuit and he is a brown labrador.")
	m.Remember("alice|s1", "assistant", "Nice to meet Biscuit! Labradors need long daily walks.")
	m.Remember("bob|s9", "user", "My secret project codename is Falcon Nine.")

	hits := m.Recall("alice|s1", "What was the name of my dog, the brown labrador?", 3, nil)
	if len(hits) == 0 || !strings.Contains(hits[0].Text, "Biscuit") {
		t.Fatalf("expected the dog note first, got %+v", hits)
	}
	if got := m.Recall("alice|s1", "what is the secret project codename Falcon", 3, nil); len(got) != 0 {
		t.Errorf("another scope's note must never be recalled: %+v", got)
	}
	if got := m.Recall("bob|s9", "secret project codename", 3, nil); len(got) != 1 || got[0].Role != "user" {
		t.Errorf("bob's own note: %+v", got)
	}
	if got := m.Recall("alice|s1", "completely unrelated question about astronomy", 3, nil); len(got) != 0 {
		t.Errorf("no shared words, no hit: %+v", got)
	}
}

func TestMemoryDedupesSkipsExcludedAndCapsLength(t *testing.T) {
	m := NewConversationMemory(100, 50)
	for i := 0; i < 3; i++ {
		m.Remember("s", "user", "the quick brown fox jumps over the lazy dog")
	}
	if n := m.Len("s"); n != 1 {
		t.Errorf("the same text twice is one note, got %d", n)
	}
	skip := func(text string) bool { return strings.Contains(text, "quick brown fox") }
	if got := m.Recall("s", "quick brown fox jumps lazy dog", 3, skip); len(got) != 0 {
		t.Errorf("excluded text (already in the prompt) must not come back: %+v", got)
	}
	m.Remember("s", "user", strings.Repeat("longword ", 2000))
	for _, h := range m.Recall("s", "longword longword longword", 3, nil) {
		if len(h.Text) > MaxNoteChars+8 {
			t.Errorf("a recalled note is capped, got %d chars", len(h.Text))
		}
	}
}

func TestMemoryIsBounded(t *testing.T) {
	// per scope: the oldest notes go first
	m := NewConversationMemory(1000, 4)
	for i := 0; i < 30; i++ {
		m.Remember("one", "user", fmt.Sprintf("gardening tools note%d", i))
	}
	if n := m.Len("one"); n != 4 {
		t.Fatalf("per-scope cap 4, got %d", n)
	}
	if hits := m.Recall("one", "gardening tools note0", 10, nil); len(hits) == 0 || strings.Contains(hits[0].Text, "note0") {
		t.Errorf("the oldest note must be gone: %+v", hits)
	}
	if hits := m.Recall("one", "gardening tools note29", 1, nil); len(hits) != 1 || !strings.Contains(hits[0].Text, "note29") {
		t.Errorf("the newest note must stay: %+v", hits)
	}
	// over all scopes
	m = NewConversationMemory(10, 4)
	for i := 0; i < 40; i++ {
		m.Remember(fmt.Sprintf("scope%d", i), "user", fmt.Sprintf("unique topic words here %d", i))
	}
	if n := m.Total(); n != 10 {
		t.Errorf("total cap 10, got %d", n)
	}
}
