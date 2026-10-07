package virtualcontext

import (
	"testing"
)

func TestVirtualContextManager(t *testing.T) {
	vctx := NewVirtualContextManager(8192, 2000000)

	id1 := vctx.AppendItem(CategorySystemPrompt, "You are Strata.", nil)
	id2 := vctx.AppendItem(CategoryUserMessage, "Calculate something.", nil)
	id3 := vctx.AppendToolResult("mathics", "2384 * 7291", "17381744")

	if id1 <= 0 || id2 <= 0 || id3 <= 0 {
		t.Fatalf("item insertion failed")
	}

	stats := vctx.Stats()
	if stats["total_items_tracked"].(uint64) != 3 {
		t.Errorf("expected 3 items tracked, got %v", stats["total_items_tracked"])
	}

	hits := vctx.Query("17381744", 5)
	if len(hits) == 0 {
		t.Errorf("expected query hit for tool result")
	}

	prompt := vctx.AssemblePrompt("Next question")
	if prompt == "" {
		t.Errorf("prompt assembly returned empty")
	}
}
