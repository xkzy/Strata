// pkg/responses/responses.go - OpenAI Responses API (POST /v1/responses) in Go
package responses

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"time"
)

type ResponsesRequest struct {
	Model           string          `json:"model"`
	Input           json.RawMessage `json:"input"`
	Instructions    string          `json:"instructions,omitempty"`
	Tools           []interface{}   `json:"tools,omitempty"`
	Temperature     *float64        `json:"temperature,omitempty"`
	MaxOutputTokens *int            `json:"max_output_tokens,omitempty"`
	Stream          bool            `json:"stream,omitempty"`
	Store           bool            `json:"store,omitempty"`
	ReasoningEffort string          `json:"reasoning_effort,omitempty"`
}

type OutputItem struct {
	ID        string      `json:"id"`
	Type      string      `json:"type"` // "message", "function_call", "reasoning"
	Role      string      `json:"role,omitempty"`
	Content   interface{} `json:"content,omitempty"`
	Name      string      `json:"name,omitempty"`
	Arguments string      `json:"arguments,omitempty"`
	CallID    string      `json:"call_id,omitempty"`
	Status    string      `json:"status,omitempty"`
}

type UsageStats struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

type ResponsesObject struct {
	ID        string       `json:"id"`
	Object    string       `json:"object"`
	CreatedAt int64        `json:"created_at"`
	Model     string       `json:"model"`
	Status    string       `json:"status"` // "completed", "in_progress", "incomplete", "failed"
	Output    []OutputItem `json:"output"`
	Usage     UsageStats   `json:"usage"`
	Error     interface{}  `json:"error,omitempty"`
}

type SSEEvent struct {
	Type     string      `json:"type"`
	Response interface{} `json:"response,omitempty"`
	Item     interface{} `json:"item,omitempty"`
	Delta    string      `json:"delta,omitempty"`
	ItemID   string      `json:"item_id,omitempty"`
}

func NewID(prefix string) string {
	b := make([]byte, 12)
	rand.Read(b)
	return fmt.Sprintf("%s_%s", prefix, hex.EncodeToString(b))
}

type ResponsesHandler struct{}

func NewResponsesHandler() *ResponsesHandler {
	return &ResponsesHandler{}
}

// ProcessRequest parses request and generates standard ResponsesObject
func (h *ResponsesHandler) ProcessRequest(req ResponsesRequest, replyText, reasoningText string) (ResponsesObject, error) {
	respID := NewID("resp")
	now := time.Now().Unix()

	var outputItems []OutputItem

	// Add reasoning item if present
	if reasoningText != "" {
		outputItems = append(outputItems, OutputItem{
			ID:     NewID("rsn"),
			Type:   "reasoning",
			Status: "completed",
			Content: []map[string]interface{}{
				{"type": "reasoning_text", "text": reasoningText},
			},
		})
	}

	// Add main message item
	outputItems = append(outputItems, OutputItem{
		ID:     NewID("msg"),
		Type:   "message",
		Role:   "assistant",
		Status: "completed",
		Content: []map[string]interface{}{
			{"type": "text", "text": replyText},
		},
	})

	inTokens := 20
	outTokens := len(replyText)/4 + len(reasoningText)/4
	if outTokens <= 0 {
		outTokens = 1
	}

	respObj := ResponsesObject{
		ID:        respID,
		Object:    "response",
		CreatedAt: now,
		Model:     req.Model,
		Status:    "completed",
		Output:    outputItems,
		Usage: UsageStats{
			InputTokens:  inTokens,
			OutputTokens: outTokens,
			TotalTokens:  inTokens + outTokens,
		},
	}

	return respObj, nil
}

// FormatSSEEvent serializes a server-sent event line
func FormatSSEEvent(event string, data interface{}) string {
	b, _ := json.Marshal(data)
	return fmt.Sprintf("event: %s\ndata: %s\n\n", event, string(b))
}
