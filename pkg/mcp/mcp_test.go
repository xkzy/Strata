// pkg/mcp/mcp_test.go - MCP Hub & Native Tools Tests
package mcp

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"strata/pkg/logicverifier"
	"strata/pkg/mathruntime"
	"strata/pkg/virtualcontext"
)

func TestNativeToolsProvider(t *testing.T) {
	mr := mathruntime.NewMathRuntime(nil)
	lv := logicverifier.NewLogicVerifier()
	vc := virtualcontext.NewVirtualContextManager(8192, 2000000)

	provider := NewNativeToolsProvider(mr, lv, vc)
	tools := provider.GetTools()
	if len(tools) != 3 {
		t.Fatalf("expected exactly 3 native tools (logic_verify, context_query, system_status), got %d", len(tools))
	}

	// Verify math tools are NOT exposed as MCP tools
	for _, tool := range tools {
		if tool.Name == "math_evaluate" || tool.Name == "sage" || tool.Name == "mathics" {
			t.Fatalf("Math engine tool %q should not be exposed via MCP", tool.Name)
		}
	}

	// Test logic_verify tool
	resLogic, err := provider.ExecuteTool("logic_verify", map[string]interface{}{
		"type":          "proposition",
		"premises":      []interface{}{"A -> B", "A"},
		"claimed_value": "B",
	})
	if err != nil {
		t.Fatalf("logic_verify error: %v", err)
	}
	if resLogic.IsError {
		t.Fatalf("expected PASS for Modus Ponens")
	}

	// Test system_status tool
	resStatus, err := provider.ExecuteTool("system_status", map[string]interface{}{})
	if err != nil {
		t.Fatalf("system_status error: %v", err)
	}
	if len(resStatus.Content) == 0 {
		t.Fatalf("expected non-empty system_status content")
	}
}

func TestMcpHub_JSONRPCServer(t *testing.T) {
	mr := mathruntime.NewMathRuntime(nil)
	lv := logicverifier.NewLogicVerifier()
	provider := NewNativeToolsProvider(mr, lv, nil)
	hub := NewMcpHub(provider, 10*time.Second, 20000)

	ctx := context.Background()

	// 1. Initialize
	initReq := `{"jsonrpc":"2.0","id":1,"method":"initialize"}`
	initRespBytes, err := hub.HandleJSONRPC(ctx, []byte(initReq))
	if err != nil {
		t.Fatalf("HandleJSONRPC initialize error: %v", err)
	}
	var initResp JSONRPCResponse
	json.Unmarshal(initRespBytes, &initResp)
	if initResp.Error != nil {
		t.Fatalf("initialize returned error: %v", initResp.Error)
	}

	// 2. tools/list
	listReq := `{"jsonrpc":"2.0","id":2,"method":"tools/list"}`
	listRespBytes, _ := hub.HandleJSONRPC(ctx, []byte(listReq))
	var listResp JSONRPCResponse
	json.Unmarshal(listRespBytes, &listResp)
	var listResult ListToolsResult
	json.Unmarshal(listResp.Result, &listResult)
	if len(listResult.Tools) != 3 {
		t.Fatalf("expected 3 tools in tools/list, got %d", len(listResult.Tools))
	}

	// 3. tools/call
	callReq := `{
		"jsonrpc": "2.0",
		"id": 3,
		"method": "tools/call",
		"params": {
			"name": "logic_verify",
			"arguments": {
				"type": "proposition",
				"premises": ["P -> Q", "P"],
				"claimed_value": "Q"
			}
		}
	}`
	callRespBytes, _ := hub.HandleJSONRPC(ctx, []byte(callReq))
	var callResp JSONRPCResponse
	json.Unmarshal(callRespBytes, &callResp)
	if callResp.Error != nil {
		t.Fatalf("tools/call error: %v", callResp.Error)
	}

	var callResult CallToolResult
	json.Unmarshal(callResp.Result, &callResult)
	if len(callResult.Content) == 0 {
		t.Fatalf("expected non-empty tool call content")
	}
}
