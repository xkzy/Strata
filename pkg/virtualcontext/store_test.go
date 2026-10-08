package virtualcontext

import (
	"fmt"
	"strings"
	"testing"
	"time"
)

func tok(n int) string { return strings.Repeat("abcd", n) } // estimateTokens: ~n tokens

func TestStoreIsBoundedByTheVirtualLimit(t *testing.T) {
	v := NewVirtualContextManager(1000, 5000)
	for i := 0; i < 200; i++ {
		v.AppendItem(CategoryUserMessage, fmt.Sprintf("note%d %s", i, tok(100)), nil)
	}
	st := v.Stats()
	if st["virtual_context_total_tokens"].(uint64) > 5000 {
		t.Errorf("stored tokens %v exceed the virtual limit 5000", st["virtual_context_total_tokens"])
	}
	if st["items_stored"].(int) >= 200 || st["evicted_items"].(uint64) == 0 {
		t.Errorf("old items must be evicted: %v", st)
	}
	if st["total_items_tracked"].(uint64) != 200 {
		t.Errorf("total tracked counts every item ever added: %v", st["total_items_tracked"])
	}
	// the newest survives, the oldest is gone
	if hits := v.Query("note199", 3); len(hits) != 1 {
		t.Errorf("newest item missing: %v", hits)
	}
	if hits := v.Query("note0", 3); len(hits) != 0 {
		t.Errorf("oldest item must be evicted: %v", hits)
	}
}

func TestPhysicalWindowHoldsOnlyTheNewestAndIsWhatThePromptUses(t *testing.T) {
	v := NewVirtualContextManager(500, 100000)
	for i := 0; i < 30; i++ {
		v.AppendItem(CategoryUserMessage, fmt.Sprintf("turn%d %s", i, tok(50)), nil)
	}
	st := v.Stats()
	if st["physical_active_tokens"].(uint64) > 500 {
		t.Errorf("physical tokens %v exceed the window 500", st["physical_active_tokens"])
	}
	if st["compressed_tokens"].(uint64) == 0 || st["compaction_events"].(uint64) == 0 {
		t.Errorf("older items must move out of the window: %v", st)
	}
	p := v.AssemblePrompt("next")
	if !strings.Contains(p, "turn29") || strings.Contains(p, "turn0 ") {
		t.Errorf("the prompt holds the newest turns only")
	}
	if n := strings.Count(p, "<|im_start|>user"); n > 12 {
		t.Errorf("the prompt has %d items, the window holds about 500/50", n)
	}
	// compressed items are still found by search
	if hits := v.Query("turn0", 3); len(hits) != 1 {
		t.Errorf("a compressed item must stay searchable: %v", hits)
	}
}

func TestQueryRanksByWordOverlap(t *testing.T) {
	v := NewVirtualContextManager(8192, 2000000)
	v.AppendItem(CategoryToolResult, "build finished with warnings about deprecated api usage", nil)
	v.AppendItem(CategoryToolResult, "deployment to the staging cluster failed: connection refused by database host", nil)
	v.AppendItem(CategoryToolResult, "unrelated note about lunch", nil)
	hits := v.Query("why did the staging deployment fail with database connection refused", 3)
	if len(hits) == 0 || !strings.Contains(hits[0]["content"].(string), "staging cluster failed") {
		t.Fatalf("best match first: %v", hits)
	}
	if s0 := hits[0]["score"].(float64); s0 <= 0 || s0 > 1 {
		t.Errorf("score must be in (0,1], got %v", s0)
	}
	if len(hits) > 1 {
		if hits[0]["score"].(float64) < hits[1]["score"].(float64) {
			t.Errorf("hits must be sorted by score: %v", hits)
		}
	}
	if h := v.Query("zzzzqqqq", 3); len(h) != 0 {
		t.Errorf("no shared word, no hit: %v", h)
	}
}

func TestQueryScopedNeverCrossesScopes(t *testing.T) {
	v := NewVirtualContextManager(8192, 2000000)
	v.AppendItem(CategoryVerificationAudit, "Verification FAIL arithmetic secretvalue12345", map[string]string{"scope": "tenantA|u|s"})
	v.AppendItem(CategoryVerificationAudit, "Verification PASS arithmetic othervalue", map[string]string{"scope": "tenantB|u|s"})
	if h := v.QueryScoped("tenantB|u|s", "secretvalue12345", 5); len(h) != 0 {
		t.Errorf("another scope's item leaked: %v", h)
	}
	if h := v.QueryScoped("tenantA|u|s", "secretvalue12345", 5); len(h) != 1 {
		t.Errorf("own item missing: %v", h)
	}
	if h := v.QueryScoped("", "secretvalue12345", 5); len(h) != 0 {
		t.Errorf("an empty scope must match nothing, not everything: %v", h)
	}
}

func TestQueryStaysFastOnALargeStore(t *testing.T) {
	v := NewVirtualContextManager(8192, 50_000_000)
	for i := 0; i < 20000; i++ {
		v.AppendItem(CategoryToolResult, fmt.Sprintf("log line %d about worker%d finishing task%d", i, i%50, i), nil)
	}
	start := time.Now()
	for i := 0; i < 200; i++ {
		v.Query(fmt.Sprintf("worker%d task%d", i%50, i), 5)
	}
	if d := time.Since(start); d > 500*time.Millisecond {
		t.Errorf("200 queries over 20,000 items took %v: the search must use the index, not scan", d)
	}
}

func TestEvictionKeepsTheIndexCorrect(t *testing.T) {
	v := NewVirtualContextManager(100, 600)
	for i := 0; i < 400; i++ {
		v.AppendItem(CategoryUserMessage, fmt.Sprintf("alpha%d beta gamma %s", i, tok(20)), nil)
	}
	for _, h := range v.Query("beta gamma", 50) {
		if !strings.Contains(h["content"].(string), "alpha") {
			t.Errorf("odd hit %v", h)
		}
	}
	if hits := v.Query("alpha399", 3); len(hits) != 1 {
		t.Errorf("newest missing after many evictions: %v", hits)
	}
	if hits := v.Query("alpha1", 3); len(hits) != 0 {
		t.Errorf("evicted item still found: %v", hits)
	}
}

func TestMemoryNotesExpireAndCanBeForgotten(t *testing.T) {
	now := time.Unix(1_000_000, 0)
	m := NewConversationMemory(100, 50)
	m.now = func() time.Time { return now }
	m.ttl = time.Hour
	m.Remember("s", "user", "my favourite colour is turquoise blue")
	m.Remember("t", "user", "my favourite colour is crimson red")
	if got := m.Recall("s", "what is my favourite colour turquoise", 3, nil); len(got) != 1 {
		t.Fatalf("fresh note: %v", got)
	}
	now = now.Add(2 * time.Hour)
	if got := m.Recall("s", "what is my favourite colour turquoise", 3, nil); len(got) != 0 {
		t.Errorf("a note older than the ttl must be gone: %v", got)
	}
	if m.Total() != 0 {
		t.Errorf("expired notes are removed, %d left", m.Total())
	}
	m.Remember("s", "user", "fresh note about turquoise paint")
	m.Remember("t", "user", "other scope note about turquoise paint")
	if n := m.Forget("s"); n != 1 {
		t.Errorf("Forget removed %d, want 1", n)
	}
	if m.Len("s") != 0 || m.Len("t") != 1 {
		t.Errorf("Forget must clear only its scope: s=%d t=%d", m.Len("s"), m.Len("t"))
	}
	if !m.Remember("s", "user", "fresh note about turquoise paint") {
		t.Error("after Forget the same text is new again")
	}
}
