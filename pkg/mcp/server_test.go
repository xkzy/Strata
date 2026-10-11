package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"
)

func TestMCPServer_InitializeAndListTools(t *testing.T) {
	s := NewServer()

	// 1. Send initialize request
	initReq := Request{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "initialize",
		Params:  json.RawMessage(`{"clientInfo":{"name":"test-client","version":"1.0.0"}}`),
	}
	initBytes, _ := json.Marshal(initReq)

	// 2. Send tools/list request
	listReq := Request{
		JSONRPC: "2.0",
		ID:      2,
		Method:  "tools/list",
	}
	listBytes, _ := json.Marshal(listReq)

	inBuf := bytes.NewBuffer(nil)
	inBuf.Write(initBytes)
	inBuf.WriteString("\n")
	inBuf.Write(listBytes)
	inBuf.WriteString("\n")

	outBuf := bytes.NewBuffer(nil)

	err := s.HandleStream(context.Background(), inBuf, outBuf)
	if err != nil {
		t.Fatalf("HandleStream returned error: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(outBuf.String()), "\n")
	if len(lines) < 2 {
		t.Fatalf("expected at least 2 response lines, got %d: %q", len(lines), outBuf.String())
	}

	// Verify initialize response
	var initResp Response
	if err := json.Unmarshal([]byte(lines[0]), &initResp); err != nil {
		t.Fatalf("failed to parse init resp: %v", err)
	}
	if initResp.Error != nil {
		t.Fatalf("init returned error: %v", initResp.Error)
	}

	// Verify tools/list response
	var listResp Response
	if err := json.Unmarshal([]byte(lines[1]), &listResp); err != nil {
		t.Fatalf("failed to parse list resp: %v", err)
	}
	var toolsResult ListToolsResult
	if err := json.Unmarshal(listResp.Result, &toolsResult); err != nil {
		t.Fatalf("failed to unmarshal toolsResult: %v", err)
	}

	toolNames := make(map[string]bool)
	for _, tool := range toolsResult.Tools {
		toolNames[tool.Name] = true
	}

	for _, expected := range []string{"strata_hardware", "strata_models", "strata_status"} {
		if !toolNames[expected] {
			t.Errorf("expected tool %s to be listed", expected)
		}
	}
}

func TestMCPServer_CallToolHardware(t *testing.T) {
	s := NewServer()

	callReq := Request{
		JSONRPC: "2.0",
		ID:      10,
		Method:  "tools/call",
		Params:  json.RawMessage(`{"name":"strata_hardware","arguments":{}}`),
	}
	callBytes, _ := json.Marshal(callReq)

	inBuf := bytes.NewBuffer(callBytes)
	inBuf.WriteString("\n")
	outBuf := bytes.NewBuffer(nil)

	if err := s.HandleStream(context.Background(), inBuf, outBuf); err != nil {
		t.Fatalf("HandleStream error: %v", err)
	}

	var resp Response
	if err := json.Unmarshal(bytes.TrimSpace(outBuf.Bytes()), &resp); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if resp.Error != nil {
		t.Fatalf("tool call error: %v", resp.Error)
	}

	var callRes CallToolResult
	if err := json.Unmarshal(resp.Result, &callRes); err != nil {
		t.Fatalf("unmarshal CallToolResult: %v", err)
	}
	if len(callRes.Content) == 0 || callRes.Content[0].Type != "text" {
		t.Fatalf("expected text content, got %+v", callRes.Content)
	}
	if !strings.Contains(callRes.Content[0].Text, "RAM") {
		t.Errorf("expected RAM in hardware output, got %s", callRes.Content[0].Text)
	}
}
