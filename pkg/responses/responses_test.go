// pkg/responses/responses_test.go - Unit Tests for Responses API in Go
package responses

import (
	"strings"
	"testing"
)

func TestResponsesHandler_ProcessRequest(t *testing.T) {
	h := NewResponsesHandler()

	req := ResponsesRequest{
		Model: "Qwen3.8-Flash-Next",
	}

	reply := "Here is the exact answer."
	reasoning := "Let me think step-by-step."

	res, err := h.ProcessRequest(req, reply, reasoning)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if res.Status != "completed" {
		t.Fatalf("expected status completed, got %s", res.Status)
	}

	if len(res.Output) != 2 {
		t.Fatalf("expected 2 output items (reasoning + message), got %d", len(res.Output))
	}

	if res.Output[0].Type != "reasoning" {
		t.Fatalf("expected first item to be reasoning, got %s", res.Output[0].Type)
	}
	if res.Output[1].Type != "message" {
		t.Fatalf("expected second item to be message, got %s", res.Output[1].Type)
	}

	// Test SSE formatting
	sse := FormatSSEEvent("response.created", res)
	if !strings.HasPrefix(sse, "event: response.created\ndata: {") {
		t.Fatalf("unexpected SSE format: %s", sse)
	}
}
