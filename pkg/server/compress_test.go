package server

import (
	"strings"
	"testing"

	"strata/pkg/chattemplate"
)

func TestCompressOld(t *testing.T) {
	long := strings.Repeat("é", 5000)
	m := []chattemplate.Message{{Role: "tool", Content: long, ReasoningContent: "r"}, {Role: "user", Content: long}, {Role: "user", Content: long}}
	if !compressOld(m) || len(m[0].Content) >= len(long) || m[0].ReasoningContent != "" || m[1].Content != long || m[2].Content != long {
		t.Fatalf("bad compress: %d", len(m[0].Content))
	}
}
