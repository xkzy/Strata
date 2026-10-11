package mcp

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"strata/pkg/installer"
	"strata/pkg/modelschema"
)

func (s *Server) registerTools() {
	s.tools = map[string]ToolInfo{
		"strata_hardware": {
			Name:        "strata_hardware",
			Description: "Detect CPU, physical RAM, GPU/VRAM hardware and driver capabilities",
			InputSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			},
		},
		"strata_models": {
			Name:        "strata_models",
			Description: "List all curated, local, and supported model families and quantizations with RAM requirements",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"family": map[string]any{
						"type":        "string",
						"description": "Optional model family filter (e.g. qwen, coder, swift, unsloth, mimo, openai)",
					},
				},
			},
		},
		"strata_status": {
			Name:        "strata_status",
			Description: "Check health, current telemetry, and active models of the local Strata inference engine",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"port": map[string]any{
						"type":        "integer",
						"description": "Server port to check (default: 8080)",
					},
				},
			},
		},
	}
}

func (s *Server) executeTool(name string, args map[string]any) (CallToolResult, error) {
	switch name {
	case "strata_hardware":
		specs := installer.DetectHardware()
		var sb strings.Builder
		sb.WriteString("=== Strata Hardware Report ===\n")
		sb.WriteString(fmt.Sprintf("CPU Cores:    %d\n", specs.CPUCores))
		sb.WriteString(fmt.Sprintf("System RAM:   %.2f GB\n", specs.RAMTotalGB))
		sb.WriteString(fmt.Sprintf("GPU Vendor:   %s\n", specs.GPUVendor))
		sb.WriteString(fmt.Sprintf("GPU Name:     %s\n", specs.GPUName))
		if specs.VRAMGB > 0 {
			sb.WriteString(fmt.Sprintf("VRAM:         %.2f GB\n", specs.VRAMGB))
		}
		return CallToolResult{
			Content: []ContentItem{{Type: "text", Text: sb.String()}},
		}, nil

	case "strata_models":
		reg := modelschema.NewRegistry()
		var sb strings.Builder
		sb.WriteString("=== Strata Supported Models & Families ===\n")

		curated := installer.GetCuratedHuggingFaceModels()
		for _, m := range curated {
			sb.WriteString(fmt.Sprintf("- %s (%s)\n  Size: %.1f GB | Min RAM: %.0f GB | Quant: %s\n  Repo: %s\n",
				m.Name, m.Category, m.SizeGB, m.MinRAMGB, m.Quantization, m.RepoID))
		}

		_ = reg
		return CallToolResult{
			Content: []ContentItem{{Type: "text", Text: sb.String()}},
		}, nil

	case "strata_status":
		port := 8080
		if pVal, ok := args["port"].(float64); ok && pVal > 0 {
			port = int(pVal)
		}

		client := &http.Client{Timeout: 2 * time.Second}
		resp, err := client.Get(fmt.Sprintf("http://127.0.0.1:%d/health", port))
		if err != nil {
			return CallToolResult{
				Content: []ContentItem{{
					Type: "text",
					Text: fmt.Sprintf("Strata server is not running on 127.0.0.1:%d (%v)", port, err),
				}},
			}, nil
		}
		defer resp.Body.Close()

		var healthData map[string]any
		_ = json.NewDecoder(resp.Body).Decode(&healthData)
		hBytes, _ := json.MarshalIndent(healthData, "", "  ")

		return CallToolResult{
			Content: []ContentItem{{
				Type: "text",
				Text: fmt.Sprintf("Strata Server Status (HTTP %d):\n%s", resp.StatusCode, string(hBytes)),
			}},
		}, nil

	default:
		return CallToolResult{
			IsError: true,
			Content: []ContentItem{{Type: "text", Text: fmt.Sprintf("Unknown tool %q", name)}},
		}, nil
	}
}
