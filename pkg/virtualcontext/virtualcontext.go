package virtualcontext

import (
	"fmt"
	"math"
	"strings"
	"sync"
	"time"
)

type ItemCategory string

const (
	CategorySystemPrompt      ItemCategory = "system_prompt"
	CategoryUserMessage       ItemCategory = "user_message"
	CategoryAssistantMessage  ItemCategory = "assistant_message"
	CategoryToolResult        ItemCategory = "tool_result"
	CategoryFileContent       ItemCategory = "file_content"
	CategoryVerificationAudit ItemCategory = "verification_audit"
)

type ContextItem struct {
	ID        int64             `json:"id"`
	Category  ItemCategory      `json:"category"`
	Content   string            `json:"content"`
	Tokens    int               `json:"tokens"`
	Timestamp float64           `json:"timestamp"`
	Metadata  map[string]string `json:"metadata,omitempty"`
}

type VirtualContextStats struct {
	VirtualContextTotalTokens uint64 `json:"virtual_context_total_tokens"`
	PhysicalActiveTokens      uint64 `json:"physical_active_tokens"`
	CompressedTokens          uint64 `json:"compressed_tokens"`
	IndexedTokens             uint64 `json:"indexed_tokens"`
	TotalItemsTracked         uint64 `json:"total_items_tracked"`
	CompactionEvents          uint64 `json:"compaction_events"`
	RetrievalQueries          uint64 `json:"retrieval_queries"`
}

type VirtualContextManager struct {
	PhysicalLimit int
	VirtualLimit  int
	nextID        int64
	items         []ContextItem
	stats         VirtualContextStats
	mu            sync.RWMutex
}

func NewVirtualContextManager(physicalLimit, virtualLimit int) *VirtualContextManager {
	if physicalLimit <= 0 {
		physicalLimit = 8192
	}
	if virtualLimit <= 0 {
		virtualLimit = 2000000
	}
	return &VirtualContextManager{
		PhysicalLimit: physicalLimit,
		VirtualLimit:  virtualLimit,
		nextID:        1,
		items:         make([]ContextItem, 0),
	}
}

func estimateTokens(text string) int {
	if text == "" {
		return 0
	}
	return int(math.Max(1, float64(len(text)+3)/4.0))
}

func (v *VirtualContextManager) AppendItem(category ItemCategory, content string, metadata map[string]string) int64 {
	v.mu.Lock()
	defer v.mu.Unlock()

	id := v.nextID
	v.nextID++
	tokens := estimateTokens(content)

	item := ContextItem{
		ID:        id,
		Category:  category,
		Content:   content,
		Tokens:    tokens,
		Timestamp: float64(time.Now().UnixNano()) / 1e9,
		Metadata:  metadata,
	}

	v.items = append(v.items, item)
	v.stats.VirtualContextTotalTokens += uint64(tokens)
	v.stats.PhysicalActiveTokens += uint64(tokens)
	v.stats.TotalItemsTracked++

	return id
}

func (v *VirtualContextManager) AppendToolResult(toolName, command, output string) int64 {
	meta := map[string]string{
		"tool_name": toolName,
		"command":   command,
	}
	return v.AppendItem(CategoryToolResult, output, meta)
}

func (v *VirtualContextManager) Query(query string, topK int) []map[string]interface{} {
	v.mu.Lock()
	v.stats.RetrievalQueries++
	v.mu.Unlock()

	v.mu.RLock()
	defer v.mu.RUnlock()

	var hits []map[string]interface{}
	qLower := strings.ToLower(query)
	for _, item := range v.items {
		if strings.Contains(strings.ToLower(item.Content), qLower) {
			hits = append(hits, map[string]interface{}{
				"item_id": item.ID,
				"score":   0.95,
				"content": item.Content,
			})
			if len(hits) >= topK {
				break
			}
		}
	}
	return hits
}

func (v *VirtualContextManager) Stats() map[string]interface{} {
	v.mu.RLock()
	defer v.mu.RUnlock()

	return map[string]interface{}{
		"virtual_context_total_tokens": v.stats.VirtualContextTotalTokens,
		"physical_active_tokens":       v.stats.PhysicalActiveTokens,
		"compressed_tokens":            v.stats.CompressedTokens,
		"indexed_tokens":               v.stats.IndexedTokens,
		"total_items_tracked":          v.stats.TotalItemsTracked,
		"compaction_events":            v.stats.CompactionEvents,
		"retrieval_queries":            v.stats.RetrievalQueries,
		"physical_limit":               v.PhysicalLimit,
		"virtual_limit":                v.VirtualLimit,
	}
}

func (v *VirtualContextManager) AssemblePrompt(userQuery string) string {
	v.mu.RLock()
	defer v.mu.RUnlock()

	var sb strings.Builder
	for _, it := range v.items {
		if it.Category == CategorySystemPrompt {
			sb.WriteString(fmt.Sprintf("<|im_start|>system\n%s<|im_end|>\n", it.Content))
		} else if it.Category == CategoryUserMessage {
			sb.WriteString(fmt.Sprintf("<|im_start|>user\n%s<|im_end|>\n", it.Content))
		} else if it.Category == CategoryAssistantMessage {
			sb.WriteString(fmt.Sprintf("<|im_start|>assistant\n%s<|im_end|>\n", it.Content))
		}
	}
	sb.WriteString(fmt.Sprintf("<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", userQuery))
	return sb.String()
}
