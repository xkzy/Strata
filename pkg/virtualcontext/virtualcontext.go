package virtualcontext

import (
	"fmt"
	"math"
	"sort"
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
	EvictedItems              uint64 `json:"evicted_items"`
}

// entry is a stored item with what the index needs.
type entry struct {
	ContextItem
	scope string
	words []string
}

// VirtualContextManager is the store behind the verification audit and the context endpoints. Two limits bound it:
//
//   - PhysicalLimit: the newest items that fit it are the physical window (what AssemblePrompt uses); older ones are
//     compressed - still stored and searchable, no longer in the prompt.
//   - VirtualLimit: all stored items together; the oldest are evicted past it.
//
// Search uses a word index (postings per word), so a query costs the matching items, not the whole store.
type VirtualContextManager struct {
	PhysicalLimit int
	VirtualLimit  int
	nextID        int64
	items         []*entry // oldest first; items[head:] are live
	head          int      // first live item
	physHead      int      // first physical item (head <= physHead <= len(items))
	byID          map[int64]*entry
	postings      map[string][]int64 // word -> item ids, ascending; ids of evicted items are skipped and swept lazily
	deadPostings  int
	livePostings  int
	stored        uint64 // tokens of live items
	physical      uint64 // tokens of the physical window
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
		byID:          map[int64]*entry{},
		postings:      map[string][]int64{},
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
	e := &entry{
		ContextItem: ContextItem{
			ID: id, Category: category, Content: content, Tokens: tokens,
			Timestamp: float64(time.Now().UnixNano()) / 1e9, Metadata: metadata,
		},
		scope: metadata["scope"],
	}
	for w := range contentWords(content) {
		e.words = append(e.words, w)
		v.postings[w] = append(v.postings[w], id)
	}
	v.livePostings += len(e.words)

	v.items = append(v.items, e)
	v.byID[id] = e
	v.stored += uint64(tokens)
	v.physical += uint64(tokens)
	v.stats.TotalItemsTracked++

	// the window: older items leave it, the newest always stays
	for v.physical > uint64(v.PhysicalLimit) && v.physHead < len(v.items)-1 {
		v.physical -= uint64(v.items[v.physHead].Tokens)
		v.physHead++
		v.stats.CompactionEvents++
	}

	// the store: the oldest items go, the newest always stays
	for v.stored > uint64(v.VirtualLimit) && v.head < len(v.items)-1 {
		old := v.items[v.head]
		v.stored -= uint64(old.Tokens)
		if v.physHead <= v.head { // it was still in the window
			v.physical -= uint64(old.Tokens)
			v.physHead = v.head + 1
		}
		delete(v.byID, old.ID)
		v.deadPostings += len(old.words)
		v.livePostings -= len(old.words)
		v.items[v.head] = nil
		v.head++
		v.stats.EvictedItems++
	}
	if v.head > 1024 && v.head > len(v.items)/2 {
		v.items = append([]*entry(nil), v.items[v.head:]...)
		v.physHead -= v.head
		v.head = 0
	}
	if v.deadPostings > 10000 && v.deadPostings > v.livePostings {
		v.rebuildIndex()
	}
	return id
}

func (v *VirtualContextManager) rebuildIndex() {
	v.postings = make(map[string][]int64, len(v.postings))
	for _, e := range v.items[v.head:] {
		for _, w := range e.words {
			v.postings[w] = append(v.postings[w], e.ID)
		}
	}
	v.deadPostings = 0
}

func (v *VirtualContextManager) AppendToolResult(toolName, command, output string) int64 {
	meta := map[string]string{
		"tool_name": toolName,
		"command":   command,
	}
	return v.AppendItem(CategoryToolResult, output, meta)
}

// Query searches every item, whoever it belongs to: for in-process callers only. A caller acting for a user goes
// through QueryScoped.
func (v *VirtualContextManager) Query(query string, topK int) []map[string]interface{} {
	return v.search(query, topK, "", false)
}

// QueryScoped searches only the items stored with metadata "scope" equal to scope. An empty scope matches nothing.
func (v *VirtualContextManager) QueryScoped(scope, query string, topK int) []map[string]interface{} {
	if scope == "" {
		return nil
	}
	return v.search(query, topK, scope, true)
}

const (
	maxCandidates   = 64   // items scored in full per query
	commonPostings  = 2000 // a word in more items than this says little: it is left out of the lookup
	fallbackScanMax = 2000 // newest items scanned for a query with no usable word (a short id like "42")
)

func (v *VirtualContextManager) search(query string, topK int, scope string, scoped bool) []map[string]interface{} {
	v.mu.Lock()
	v.stats.RetrievalQueries++
	v.mu.Unlock()
	if topK <= 0 {
		return nil
	}

	v.mu.RLock()
	defer v.mu.RUnlock()
	visible := func(e *entry) bool { return !scoped || e.scope == scope }
	qw := contentWords(query)
	phrase := strings.ToLower(strings.TrimSpace(query))

	type cand struct {
		e       *entry
		weight  float64
		overlap int
	}
	var cands []cand
	if len(qw) == 0 {
		if len(phrase) < 2 {
			return nil
		}
		scanned := 0
		for i := len(v.items) - 1; i >= v.head && scanned < fallbackScanMax; i-- {
			scanned++
			if e := v.items[i]; visible(e) && strings.Contains(strings.ToLower(e.Content), phrase) {
				cands = append(cands, cand{e: e, weight: 1, overlap: 1})
			}
		}
	} else {
		live := float64(len(v.items) - v.head)
		acc := map[int64]*cand{}
		var total float64
		for w := range qw {
			pl := v.postings[w]
			df := float64(len(pl)) // an upper bound: evicted ids are still listed until the sweep
			idf := math.Log(1 + live/math.Max(1, df))
			total += idf
			if len(pl) > commonPostings {
				continue
			}
			for _, id := range pl {
				e, ok := v.byID[id]
				if !ok || !visible(e) {
					continue
				}
				c := acc[id]
				if c == nil {
					c = &cand{e: e}
					acc[id] = c
				}
				c.weight += idf
				c.overlap++
			}
		}
		need := 2
		if len(qw) < 2 {
			need = len(qw)
		}
		for _, c := range acc {
			if c.overlap >= need {
				c.weight /= total
				cands = append(cands, *c)
			}
		}
	}
	sort.Slice(cands, func(a, b int) bool {
		if cands[a].weight != cands[b].weight {
			return cands[a].weight > cands[b].weight
		}
		return cands[a].e.ID > cands[b].e.ID
	})
	if len(cands) > maxCandidates {
		cands = cands[:maxCandidates]
	}
	// the whole query as a phrase in the content is the strongest evidence
	if len(phrase) >= 3 {
		for i := range cands {
			if strings.Contains(strings.ToLower(cands[i].e.Content), phrase) {
				cands[i].weight = 1
			}
		}
		sort.SliceStable(cands, func(a, b int) bool { return cands[a].weight > cands[b].weight })
	}
	if len(cands) > topK {
		cands = cands[:topK]
	}
	hits := make([]map[string]interface{}, 0, len(cands))
	for _, c := range cands {
		hits = append(hits, map[string]interface{}{
			"item_id":  c.e.ID,
			"score":    math.Round(c.weight*1000) / 1000,
			"content":  c.e.Content,
			"category": c.e.Category,
		})
	}
	return hits
}

func (v *VirtualContextManager) Stats() map[string]interface{} {
	v.mu.RLock()
	defer v.mu.RUnlock()

	return map[string]interface{}{
		"virtual_context_total_tokens": v.stored,
		"physical_active_tokens":       v.physical,
		"compressed_tokens":            v.stored - v.physical,
		"indexed_tokens":               v.stored,
		"total_items_tracked":          v.stats.TotalItemsTracked,
		"compaction_events":            v.stats.CompactionEvents,
		"retrieval_queries":            v.stats.RetrievalQueries,
		"items_stored":                 len(v.items) - v.head,
		"evicted_items":                v.stats.EvictedItems,
		"physical_limit":               v.PhysicalLimit,
		"virtual_limit":                v.VirtualLimit,
	}
}

// AssemblePrompt builds a prompt from the physical window, the newest items that fit PhysicalLimit.
func (v *VirtualContextManager) AssemblePrompt(userQuery string) string {
	v.mu.RLock()
	defer v.mu.RUnlock()

	var sb strings.Builder
	for _, it := range v.items[v.physHead:] {
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
