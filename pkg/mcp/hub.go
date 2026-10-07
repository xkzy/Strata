// pkg/mcp/hub.go - MCP Client Hub & Server Router
package mcp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sort"
	"strings"
	"sync"
	"time"
)

type McpClient interface {
	Name() string
	ListTools(ctx context.Context) ([]ToolInfo, error)
	CallTool(ctx context.Context, name string, args map[string]interface{}) (CallToolResult, error)
	Close() error
}

// McpHub manages all connected MCP clients and provides the central tool router
type McpHub struct {
	mu          sync.RWMutex
	clients     map[string]McpClient
	nativeTools *NativeToolsProvider
	timeout     time.Duration
	maxChars    int
}

func NewMcpHub(native *NativeToolsProvider, timeout time.Duration, maxChars int) *McpHub {
	if timeout <= 0 {
		timeout = 60 * time.Second
	}
	if maxChars <= 0 {
		maxChars = 20000
	}
	return &McpHub{
		clients:     make(map[string]McpClient),
		nativeTools: native,
		timeout:     timeout,
		maxChars:    maxChars,
	}
}

func (h *McpHub) RegisterClient(client McpClient) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.clients[client.Name()] = client
}

// GetAllTools aggregates native tools and all connected external MCP server tools
func (h *McpHub) GetAllTools(ctx context.Context) []ToolInfo {
	h.mu.RLock()
	defer h.mu.RUnlock()

	var all []ToolInfo
	if h.nativeTools != nil {
		all = append(all, h.nativeTools.GetTools()...)
	}

	for srvName, client := range h.clients {
		tools, err := client.ListTools(ctx)
		if err != nil {
			continue
		}
		for _, t := range tools {
			namespaced := t
			namespaced.Name = fmt.Sprintf("%s__%s", srvName, t.Name)
			all = append(all, namespaced)
		}
	}

	return all
}

// ExecuteTool routes tool calls to the appropriate native or external MCP client
func (h *McpHub) ExecuteTool(ctx context.Context, name string, args map[string]interface{}) (CallToolResult, error) {
	// Check native tools first
	if !strings.Contains(name, "__") {
		if h.nativeTools != nil {
			return h.nativeTools.ExecuteTool(name, args)
		}
	}

	// External namespaced tool: <server>__<tool>
	parts := strings.SplitN(name, "__", 2)
	if len(parts) != 2 {
		return CallToolResult{
			Content: []ContentBlock{{Type: "text", Text: fmt.Sprintf("invalid tool name format: %s", name)}},
			IsError: true,
		}, fmt.Errorf("invalid tool name: %s", name)
	}

	srvName, toolName := parts[0], parts[1]

	h.mu.RLock()
	client, exists := h.clients[srvName]
	h.mu.RUnlock()

	if !exists {
		return CallToolResult{
			Content: []ContentBlock{{Type: "text", Text: fmt.Sprintf("mcp server '%s' is not active", srvName)}},
			IsError: true,
		}, fmt.Errorf("mcp server not found: %s", srvName)
	}

	callCtx, cancel := context.WithTimeout(ctx, h.timeout)
	defer cancel()

	res, err := client.CallTool(callCtx, toolName, args)
	if err != nil {
		return CallToolResult{
			Content: []ContentBlock{{Type: "text", Text: fmt.Sprintf("error executing %s: %v", name, err)}},
			IsError: true,
		}, err
	}

	// Truncate output if exceeding max characters
	for i, c := range res.Content {
		if len(c.Text) > h.maxChars {
			res.Content[i].Text = c.Text[:h.maxChars] + fmt.Sprintf("\n... [output truncated at %d characters]", h.maxChars)
		}
	}

	return res, nil
}

// HandleJSONRPC processes incoming JSON-RPC 2.0 requests (for MCP Server mode)
func (h *McpHub) HandleJSONRPC(ctx context.Context, reqBytes []byte) ([]byte, error) {
	var req JSONRPCRequest
	if err := json.Unmarshal(reqBytes, &req); err != nil {
		errResp := JSONRPCResponse{
			JSONRPC: "2.0",
			Error:   &JSONRPCError{Code: -32700, Message: "Parse error"},
		}
		return json.Marshal(errResp)
	}

	switch req.Method {
	case "initialize":
		result := map[string]interface{}{
			"protocolVersion": ProtocolVersion,
			"capabilities": map[string]interface{}{
				"tools": map[string]interface{}{},
			},
			"serverInfo": map[string]interface{}{
				"name":    "Strata-MCP-Server",
				"version": "1.0.0",
			},
		}
		raw, _ := json.Marshal(result)
		resp := JSONRPCResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Result:  raw,
		}
		return json.Marshal(resp)

	case "tools/list":
		tools := h.GetAllTools(ctx)
		res := ListToolsResult{Tools: tools}
		raw, _ := json.Marshal(res)
		resp := JSONRPCResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Result:  raw,
		}
		return json.Marshal(resp)

	case "tools/call":
		var params CallToolParams
		if err := json.Unmarshal(req.Params, &params); err != nil {
			resp := JSONRPCResponse{
				JSONRPC: "2.0",
				ID:      req.ID,
				Error:   &JSONRPCError{Code: -32602, Message: "Invalid params"},
			}
			return json.Marshal(resp)
		}

		res, err := h.ExecuteTool(ctx, params.Name, params.Arguments)
		if err != nil && len(res.Content) == 0 {
			resp := JSONRPCResponse{
				JSONRPC: "2.0",
				ID:      req.ID,
				Error:   &JSONRPCError{Code: -32000, Message: err.Error()},
			}
			return json.Marshal(resp)
		}

		raw, _ := json.Marshal(res)
		resp := JSONRPCResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Result:  raw,
		}
		return json.Marshal(resp)

	case "ping":
		resp := JSONRPCResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Result:  json.RawMessage(`{}`),
		}
		return json.Marshal(resp)

	default:
		resp := JSONRPCResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Error:   &JSONRPCError{Code: -32601, Message: fmt.Sprintf("Method '%s' not found", req.Method)},
		}
		return json.Marshal(resp)
	}
}

// ---------------------------------------------------------------------------
// Stdio Transport for external subprocess MCP servers
// ---------------------------------------------------------------------------

type StdioMcpClient struct {
	name    string
	cfg     ServerConfig
	cmd     *exec.Cmd
	stdin   io.WriteCloser
	stdout  *bufio.Reader
	mu      sync.Mutex
	nextID  int64
	pending map[int64]chan JSONRPCResponse
}

func NewStdioMcpClient(name string, cfg ServerConfig) (*StdioMcpClient, error) {
	cmd := exec.Command(cfg.Command, cfg.Args...)
	if cfg.Cwd != "" {
		cmd.Dir = cfg.Cwd
	}
	cmd.Env = os.Environ()
	for k, v := range cfg.Env {
		cmd.Env = append(cmd.Env, fmt.Sprintf("%s=%s", k, v))
	}

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}

	if err := cmd.Start(); err != nil {
		return nil, err
	}

	client := &StdioMcpClient{
		name:    name,
		cfg:     cfg,
		cmd:     cmd,
		stdin:   stdin,
		stdout:  bufio.NewReader(stdout),
		pending: make(map[int64]chan JSONRPCResponse),
	}

	go client.readLoop()
	return client, nil
}

func (c *StdioMcpClient) Name() string {
	return c.name
}

func (c *StdioMcpClient) readLoop() {
	for {
		line, err := c.stdout.ReadBytes('\n')
		if err != nil {
			break
		}
		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}

		var resp JSONRPCResponse
		if err := json.Unmarshal(line, &resp); err == nil {
			if idFloat, ok := resp.ID.(float64); ok {
				id := int64(idFloat)
				c.mu.Lock()
				ch, found := c.pending[id]
				if found {
					delete(c.pending, id)
					ch <- resp
				}
				c.mu.Unlock()
			}
		}
	}
}

func (c *StdioMcpClient) ListTools(ctx context.Context) ([]ToolInfo, error) {
	c.mu.Lock()
	id := c.nextID
	c.nextID++
	respChan := make(chan JSONRPCResponse, 1)
	c.pending[id] = respChan
	c.mu.Unlock()

	req := JSONRPCRequest{
		JSONRPC: "2.0",
		ID:      id,
		Method:  "tools/list",
	}
	reqBytes, _ := json.Marshal(req)
	reqBytes = append(reqBytes, '\n')

	c.mu.Lock()
	_, err := c.stdin.Write(reqBytes)
	c.mu.Unlock()
	if err != nil {
		return nil, err
	}

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case resp := <-respChan:
		if resp.Error != nil {
			return nil, fmt.Errorf("MCP error %d: %s", resp.Error.Code, resp.Error.Message)
		}
		var listRes ListToolsResult
		if err := json.Unmarshal(resp.Result, &listRes); err != nil {
			return nil, err
		}
		return listRes.Tools, nil
	}
}

func (c *StdioMcpClient) CallTool(ctx context.Context, name string, args map[string]interface{}) (CallToolResult, error) {
	c.mu.Lock()
	id := c.nextID
	c.nextID++
	respChan := make(chan JSONRPCResponse, 1)
	c.pending[id] = respChan
	c.mu.Unlock()

	params := CallToolParams{Name: name, Arguments: args}
	paramsBytes, _ := json.Marshal(params)

	req := JSONRPCRequest{
		JSONRPC: "2.0",
		ID:      id,
		Method:  "tools/call",
		Params:  paramsBytes,
	}
	reqBytes, _ := json.Marshal(req)
	reqBytes = append(reqBytes, '\n')

	c.mu.Lock()
	_, err := c.stdin.Write(reqBytes)
	c.mu.Unlock()
	if err != nil {
		return CallToolResult{}, err
	}

	select {
	case <-ctx.Done():
		return CallToolResult{}, ctx.Err()
	case resp := <-respChan:
		if resp.Error != nil {
			return CallToolResult{IsError: true}, fmt.Errorf("MCP error %d: %s", resp.Error.Code, resp.Error.Message)
		}
		var callRes CallToolResult
		if err := json.Unmarshal(resp.Result, &callRes); err != nil {
			return CallToolResult{}, err
		}
		return callRes, nil
	}
}

func (c *StdioMcpClient) Close() error {
	if c.stdin != nil {
		c.stdin.Close()
	}
	if c.cmd != nil && c.cmd.Process != nil {
		return c.cmd.Process.Kill()
	}
	return nil
}

// Status is what the web app's MCP card shows: the connected external servers and the tools each one offers. The
// built-in tools are not listed as a server (they are always there and need no switch).
func (h *McpHub) Status(ctx context.Context) map[string]interface{} {
	h.mu.RLock()
	defer h.mu.RUnlock()

	servers := []map[string]interface{}{}
	total := 0
	for name, client := range h.clients {
		entry := map[string]interface{}{"name": name, "transport": "stdio", "status": "ready", "tools": []map[string]interface{}{}}
		tools, err := client.ListTools(ctx)
		if err != nil {
			entry["status"], entry["error"] = "failed", err.Error()
		}
		list := make([]map[string]interface{}, 0, len(tools))
		for _, t := range tools {
			list = append(list, map[string]interface{}{"tool": t.Name, "description": t.Description})
		}
		entry["tools"] = list
		total += len(list)
		servers = append(servers, entry)
	}
	sort.Slice(servers, func(i, j int) bool { return servers[i]["name"].(string) < servers[j]["name"].(string) })
	return map[string]interface{}{"servers": servers, "tools": total}
}
