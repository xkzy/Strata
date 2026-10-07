// pkg/mcp/native_tools.go - Built-in Strata Native MCP Tools Provider
package mcp

import (
	"encoding/json"
	"fmt"
	"runtime"
	"time"

	"strata/pkg/logicverifier"
	"strata/pkg/mathruntime"
	"strata/pkg/virtualcontext"
)

type NativeToolsProvider struct {
	mathRuntime    *mathruntime.MathRuntime
	logicVerifier  *logicverifier.LogicVerifier
	virtualContext *virtualcontext.VirtualContextManager
	startTime      time.Time
}

func NewNativeToolsProvider(
	mr *mathruntime.MathRuntime,
	lv *logicverifier.LogicVerifier,
	vc *virtualcontext.VirtualContextManager,
) *NativeToolsProvider {
	if mr == nil {
		mr = mathruntime.NewMathRuntime(nil)
	}
	if lv == nil {
		lv = logicverifier.NewLogicVerifier()
	}
	return &NativeToolsProvider{
		mathRuntime:    mr,
		logicVerifier:  lv,
		virtualContext: vc,
		startTime:      time.Now(),
	}
}

func (p *NativeToolsProvider) GetTools() []ToolInfo {
	return []ToolInfo{
		{
			Name:        "math_evaluate",
			Description: "Authoritative deterministic calculation engine (exact rational fractions, matrix algebra, CAS via Mathics).",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"expression": map[string]interface{}{
						"type":        "string",
						"description": "Mathematical expression (e.g. '1/3 + 1/6', '2384 * 7291', '[[1,2],[3,4]]')",
					},
					"operation": map[string]interface{}{
						"type":        "string",
						"description": "Operation type: 'evaluate', 'simplify', 'expand', 'factor', 'determinant', 'solve'",
					},
					"mode": map[string]interface{}{
						"type":        "string",
						"description": "'exact' or 'numeric'",
					},
				},
				"required": []string{"expression"},
			},
		},
		{
			Name:        "logic_verify",
			Description: "Deterministic logic and constraint verifier (Modus Ponens truth-tables, interval constraints, SI dimensional analysis, JSON schemas).",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"type": map[string]interface{}{
						"type":        "string",
						"description": "'arithmetic', 'proposition', 'constraint', 'unit_dimension', 'schema_type'",
					},
					"expression": map[string]interface{}{
						"type":        "string",
						"description": "Expression or conclusion to verify",
					},
					"claimed_value": map[string]interface{}{
						"type":        "string",
						"description": "Claimed result or value",
					},
					"premises": map[string]interface{}{
						"type": "array",
						"items": map[string]interface{}{
							"type": "string",
						},
						"description": "Logical premises (e.g. ['P -> Q', 'P'])",
					},
					"constraints": map[string]interface{}{
						"type": "array",
						"items": map[string]interface{}{
							"type": "string",
						},
						"description": "Interval constraints (e.g. ['x > 0', 'x < 10'])",
					},
				},
				"required": []string{"type"},
			},
		},
		{
			Name:        "context_query",
			Description: "Search the virtual context external audit and conversation memory.",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type":        "string",
						"description": "Text query string to match",
					},
					"top_k": map[string]interface{}{
						"type":        "integer",
						"description": "Maximum number of results to retrieve (default: 5)",
					},
				},
				"required": []string{"query"},
			},
		},
		{
			Name:        "system_status",
			Description: "Get runtime status, memory stats, and active backend diagnostic metrics.",
			InputSchema: map[string]interface{}{
				"type": "object",
			},
		},
	}
}

func (p *NativeToolsProvider) ExecuteTool(name string, args map[string]interface{}) (CallToolResult, error) {
	switch name {
	case "math_evaluate":
		expr, _ := args["expression"].(string)
		op, _ := args["operation"].(string)
		mode, _ := args["mode"].(string)

		req := mathruntime.MathRequest{
			Expression: expr,
			Operation:  mathruntime.MathOperation(op),
			Mode:       mathruntime.MathMode(mode),
		}
		res := p.mathRuntime.ProcessRequest(req)
		outJSON, _ := json.MarshalIndent(res, "", "  ")
		return CallToolResult{
			Content: []ContentBlock{
				{Type: "text", Text: string(outJSON)},
			},
		}, nil

	case "logic_verify":
		var claim logicverifier.VerificationClaim
		argsBytes, _ := json.Marshal(args)
		json.Unmarshal(argsBytes, &claim)

		res := p.logicVerifier.Verify(claim)
		outJSON, _ := json.MarshalIndent(res, "", "  ")
		return CallToolResult{
			Content: []ContentBlock{
				{Type: "text", Text: string(outJSON)},
			},
			IsError: res.Status == logicverifier.StatusFail,
		}, nil

	case "context_query":
		query, _ := args["query"].(string)
		topK := 5
		if k, ok := args["top_k"].(float64); ok && k > 0 {
			topK = int(k)
		}
		var hits []map[string]interface{}
		if p.virtualContext != nil {
			hits = p.virtualContext.Query(query, topK)
		}
		outJSON, _ := json.MarshalIndent(hits, "", "  ")
		return CallToolResult{
			Content: []ContentBlock{
				{Type: "text", Text: string(outJSON)},
			},
		}, nil

	case "system_status":
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		status := map[string]interface{}{
			"uptime_seconds": time.Since(p.startTime).Seconds(),
			"goroutines":     runtime.NumGoroutine(),
			"alloc_mb":       float64(m.Alloc) / 1024 / 1024,
			"sys_mb":         float64(m.Sys) / 1024 / 1024,
			"math_stats":     p.mathRuntime.GetStats(),
			"verify_metrics": p.logicVerifier.GetMetrics(),
		}
		outJSON, _ := json.MarshalIndent(status, "", "  ")
		return CallToolResult{
			Content: []ContentBlock{
				{Type: "text", Text: string(outJSON)},
			},
		}, nil

	default:
		return CallToolResult{
			Content: []ContentBlock{
				{Type: "text", Text: fmt.Sprintf("unknown native tool: %s", name)},
			},
			IsError: true,
		}, fmt.Errorf("tool not found: %s", name)
	}
}
