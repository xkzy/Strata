package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sync"
)

// Server implements a native Model Context Protocol (MCP) server
type Server struct {
	mu    sync.RWMutex
	tools map[string]ToolInfo
}

// NewServer initializes a new MCP server instance
func NewServer() *Server {
	s := &Server{
		tools: make(map[string]ToolInfo),
	}
	s.registerTools()
	return s
}

// HandleStream reads newline-delimited JSON-RPC requests from in and writes responses to out
func (s *Server) HandleStream(ctx context.Context, in io.Reader, out io.Writer) error {
	scanner := bufio.NewScanner(in)
	// Buffer up to 10MB per line for large tool arguments
	scanner.Buffer(make([]byte, 64*1024), 10*1024*1024)

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		var req Request
		if err := json.Unmarshal(line, &req); err != nil {
			resp := Response{
				JSONRPC: "2.0",
				Error: &RPCError{
					Code:    -32700,
					Message: fmt.Sprintf("Parse error: %v", err),
				},
			}
			writeResponse(out, resp)
			continue
		}

		resp := s.handleRequest(ctx, req)
		if resp != nil {
			writeResponse(out, *resp)
		}
	}
	return scanner.Err()
}

// ServeStdio starts the MCP server over standard input and standard output
func (s *Server) ServeStdio(ctx context.Context) error {
	return s.HandleStream(ctx, os.Stdin, os.Stdout)
}

func (s *Server) handleRequest(ctx context.Context, req Request) *Response {
	// Notifications (without ID) don't need a response
	if req.ID == nil {
		return nil
	}

	switch req.Method {
	case "initialize":
		result := map[string]any{
			"protocolVersion": "2024-11-05",
			"serverInfo": map[string]string{
				"name":    "strata-mcp",
				"version": "1.0.0",
			},
			"capabilities": map[string]any{
				"tools": map[string]any{},
			},
		}
		raw, _ := json.Marshal(result)
		return &Response{
			JSONRPC: "2.0",
			ID:      req.ID,
			Result:  raw,
		}

	case "notifications/initialized":
		return nil

	case "tools/list":
		s.mu.RLock()
		var toolList []ToolInfo
		for _, t := range s.tools {
			toolList = append(toolList, t)
		}
		s.mu.RUnlock()

		res := ListToolsResult{Tools: toolList}
		raw, _ := json.Marshal(res)
		return &Response{
			JSONRPC: "2.0",
			ID:      req.ID,
			Result:  raw,
		}

	case "tools/call":
		var params CallToolParams
		if len(req.Params) > 0 {
			if err := json.Unmarshal(req.Params, &params); err != nil {
				return &Response{
					JSONRPC: "2.0",
					ID:      req.ID,
					Error: &RPCError{
						Code:    -32602,
						Message: fmt.Sprintf("Invalid params: %v", err),
					},
				}
			}
		}

		callRes, err := s.executeTool(params.Name, params.Arguments)
		if err != nil {
			return &Response{
				JSONRPC: "2.0",
				ID:      req.ID,
				Error: &RPCError{
					Code:    -32000,
					Message: err.Error(),
				},
			}
		}

		raw, _ := json.Marshal(callRes)
		return &Response{
			JSONRPC: "2.0",
			ID:      req.ID,
			Result:  raw,
		}

	default:
		return &Response{
			JSONRPC: "2.0",
			ID:      req.ID,
			Error: &RPCError{
				Code:    -32601,
				Message: fmt.Sprintf("Method not found: %s", req.Method),
			},
		}
	}
}

func writeResponse(w io.Writer, resp Response) {
	bytes, _ := json.Marshal(resp)
	_, _ = w.Write(bytes)
	_, _ = w.Write([]byte("\n"))
}
