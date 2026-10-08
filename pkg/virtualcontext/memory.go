package virtualcontext

import (
	"hash/fnv"
	"sort"
	"strings"
	"sync"
	"unicode"
)

const (
	// MaxNoteChars is the longest note text Recall returns; MaxStoredChars the longest one Remember keeps.
	MaxNoteChars   = 600
	MaxStoredChars = 4000
)

// ConversationMemory is the per-conversation recall behind the server's automatic context lookup: the turns a
// caller's requests carried, kept per scope (tenant / user / session) so one caller never recalls another's.
// It is bounded: a cap per scope and one over all scopes, the oldest note going first.
type ConversationMemory struct {
	mu       sync.Mutex
	maxTotal int
	perScope int
	notes    []note // oldest first
	count    map[string]int
	seen     map[string]map[uint64]struct{}
}

type note struct {
	scope, role, text string
	hash              uint64
	words             map[string]struct{}
}

// Recalled is one note found for a query.
type Recalled struct {
	Role  string
	Text  string
	Score float64
}

func NewConversationMemory(maxTotal, perScope int) *ConversationMemory {
	if maxTotal <= 0 {
		maxTotal = 4096
	}
	if perScope <= 0 {
		perScope = 256
	}
	return &ConversationMemory{maxTotal: maxTotal, perScope: perScope,
		count: map[string]int{}, seen: map[string]map[uint64]struct{}{}}
}

var stopwords = map[string]struct{}{}

func init() {
	for _, w := range strings.Fields(`the and for that this with have from your what when where which who whom whose why how
		are was were been being can could would should will shall may might must you your yours not but about into over
		then than them they their there here also just very some any all each other more most such only own same too its
		our out off has had did does done get got let say said tell please give make made like want need know`) {
		stopwords[w] = struct{}{}
	}
}

func contentWords(text string) map[string]struct{} {
	out := map[string]struct{}{}
	for _, w := range strings.FieldsFunc(strings.ToLower(text), func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsDigit(r) }) {
		if len([]rune(w)) < 3 {
			continue
		}
		if _, stop := stopwords[w]; stop {
			continue
		}
		out[w] = struct{}{}
	}
	return out
}

func clip(s string, n int) string {
	if r := []rune(s); len(r) > n {
		return string(r[:n]) + "…"
	}
	return s
}

func hashOf(s string) uint64 {
	h := fnv.New64a()
	h.Write([]byte(s))
	return h.Sum64()
}

// Remember keeps a turn under a scope. The same text twice in a scope is one note. It reports whether the
// note was new.
func (m *ConversationMemory) Remember(scope, role, text string) bool {
	text = strings.TrimSpace(text)
	if text == "" {
		return false
	}
	text = clip(text, MaxStoredChars)
	h := hashOf(text)
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, dup := m.seen[scope][h]; dup {
		return false
	}
	if m.seen[scope] == nil {
		m.seen[scope] = map[uint64]struct{}{}
	}
	m.seen[scope][h] = struct{}{}
	m.count[scope]++
	m.notes = append(m.notes, note{scope: scope, role: role, text: text, hash: h, words: contentWords(text)})
	for m.count[scope] > m.perScope {
		m.dropOldest(func(n note) bool { return n.scope == scope })
	}
	for len(m.notes) > m.maxTotal {
		m.dropOldest(func(note) bool { return true })
	}
	return true
}

func (m *ConversationMemory) dropOldest(match func(note) bool) {
	for i, n := range m.notes {
		if !match(n) {
			continue
		}
		m.notes = append(m.notes[:i], m.notes[i+1:]...)
		delete(m.seen[n.scope], n.hash)
		if m.count[n.scope]--; m.count[n.scope] <= 0 {
			delete(m.count, n.scope)
			delete(m.seen, n.scope)
		}
		return
	}
}

// Recall returns up to topK notes of the scope that share enough words with the query, best first (newer first
// among equals). skip drops notes the caller already has (for example turns that are in the prompt).
func (m *ConversationMemory) Recall(scope, query string, topK int, skip func(text string) bool) []Recalled {
	q := contentWords(query)
	if len(q) == 0 || topK <= 0 {
		return nil
	}
	need := 2
	if len(q) < 2 {
		need = len(q)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	type scored struct {
		Recalled
		order int
	}
	var found []scored
	for i, n := range m.notes {
		if n.scope != scope {
			continue
		}
		overlap := 0
		for w := range q {
			if _, ok := n.words[w]; ok {
				overlap++
			}
		}
		if overlap < need || (skip != nil && skip(n.text)) {
			continue
		}
		found = append(found, scored{Recalled{Role: n.role, Text: clip(n.text, MaxNoteChars), Score: float64(overlap) / float64(len(q))}, i})
	}
	sort.SliceStable(found, func(a, b int) bool {
		if found[a].Score != found[b].Score {
			return found[a].Score > found[b].Score
		}
		return found[a].order > found[b].order
	})
	if len(found) > topK {
		found = found[:topK]
	}
	out := make([]Recalled, len(found))
	for i, f := range found {
		out[i] = f.Recalled
	}
	return out
}

// Len is how many notes a scope holds; Total how many all scopes hold.
func (m *ConversationMemory) Len(scope string) int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.count[scope]
}

func (m *ConversationMemory) Total() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.notes)
}
