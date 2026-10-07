// Package tokenizer is the model's byte-level BPE tokenizer (Qwen3.5 family, GGUF `gpt2` model with the `qwen35`
// pre-tokenizer), read from the pack's tokenizer/ directory. It is a port of tools/strata_tokenizer.py, the repository's
// reference implementation, and is checked against it (see tokenizer_test.go): token ids are what the engine
// consumes, so a tokenizer that is almost right produces a model that is almost right.
package tokenizer

import (
	"container/heap"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"unicode/utf8"
)

// byte <-> printable-unicode mapping of GPT-2 style byte-level BPE
var (
	byteToRune [256]rune
	runeToByte = map[rune]byte{}
)

func init() {
	var bs []int
	for b := 0x21; b < 0x7F; b++ {
		bs = append(bs, b)
	}
	for b := 0xA1; b < 0xAD; b++ {
		bs = append(bs, b)
	}
	for b := 0xAE; b < 0x100; b++ {
		bs = append(bs, b)
	}
	have := map[int]bool{}
	cs := append([]int(nil), bs...)
	for _, b := range bs {
		have[b] = true
	}
	n := 0
	for b := 0; b < 256; b++ {
		if !have[b] {
			bs = append(bs, b)
			cs = append(cs, 256+n)
			n++
		}
	}
	for i, b := range bs {
		byteToRune[b] = rune(cs[i])
		runeToByte[rune(cs[i])] = byte(b)
	}
}

type Tokenizer struct {
	tokens     []string
	ids        map[string]int32
	ranks      map[string]int32 // "left right" -> rank (a space never occurs inside a token: byte 0x20 maps to U+0120)
	tokenBytes [][]byte
	special    map[string]int32 // GGUF token types 3 (control) and 4 (user defined)
	control    map[string]bool  // type 3: matched only when parse_special
	always     map[byte][]string
	allSpecial map[byte][]string
	EOS, BOS   int

	mu    sync.RWMutex
	cache map[string][]int32
}

type packConfig struct {
	SpecialIDs map[string]int `json:"special_ids"`
}

// Load reads vocab.json, merges.txt, token_type.json and tokenizer.json from a pack's tokenizer directory.
func Load(dir string) (*Tokenizer, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "vocab.json"))
	if err != nil {
		return nil, fmt.Errorf("tokenizer: %w", err)
	}
	vocab := map[string]int{}
	if err := json.Unmarshal(raw, &vocab); err != nil {
		return nil, fmt.Errorf("tokenizer: vocab.json: %w", err)
	}
	t := &Tokenizer{
		tokens: make([]string, len(vocab)), ids: make(map[string]int32, len(vocab)), ranks: map[string]int32{},
		special: map[string]int32{}, control: map[string]bool{}, always: map[byte][]string{}, allSpecial: map[byte][]string{},
		cache: map[string][]int32{}, EOS: -1, BOS: -1,
	}
	for tok, id := range vocab {
		if id < 0 || id >= len(t.tokens) {
			return nil, fmt.Errorf("tokenizer: vocab.json id %d out of range", id)
		}
		t.tokens[id] = tok
		t.ids[tok] = int32(id)
	}
	if len(t.ids) != len(t.tokens) {
		return nil, fmt.Errorf("tokenizer: vocabulary ids are not a dense range")
	}
	mraw, err := os.ReadFile(filepath.Join(dir, "merges.txt"))
	if err != nil {
		return nil, fmt.Errorf("tokenizer: %w", err)
	}
	for i, line := range strings.Split(string(mraw), "\n") {
		if line == "" {
			continue
		}
		parts := strings.Split(line, " ")
		if len(parts) != 2 {
			return nil, fmt.Errorf("tokenizer: merge %d is not a pair", i)
		}
		if _, ok := t.ids[parts[0]]; !ok {
			return nil, fmt.Errorf("tokenizer: merge %d names a token outside the vocabulary", i)
		}
		if _, ok := t.ids[parts[1]]; !ok {
			return nil, fmt.Errorf("tokenizer: merge %d names a token outside the vocabulary", i)
		}
		t.ranks[line] = int32(i)
	}
	var types []int
	if traw, err := os.ReadFile(filepath.Join(dir, "token_type.json")); err == nil {
		_ = json.Unmarshal(traw, &types)
	}
	for i, ty := range types {
		if (ty == 3 || ty == 4) && i < len(t.tokens) && t.tokens[i] != "" {
			lit := t.tokens[i]
			t.special[lit] = int32(i)
			t.allSpecial[lit[0]] = append(t.allSpecial[lit[0]], lit)
			if ty == 3 {
				t.control[lit] = true
			} else {
				t.always[lit[0]] = append(t.always[lit[0]], lit)
			}
		}
	}
	for _, m := range []map[byte][]string{t.always, t.allSpecial} {
		for k := range m {
			sort.Slice(m[k], func(a, b int) bool { return len(m[k][a]) > len(m[k][b]) }) // longest literal first
		}
	}
	if craw, err := os.ReadFile(filepath.Join(dir, "tokenizer.json")); err == nil {
		var pc packConfig
		if json.Unmarshal(craw, &pc) == nil {
			if v, ok := pc.SpecialIDs["tokenizer.ggml.eos_token_id"]; ok {
				t.EOS = v
			}
			if v, ok := pc.SpecialIDs["tokenizer.ggml.bos_token_id"]; ok {
				t.BOS = v
			}
		}
	}
	t.tokenBytes = make([][]byte, len(t.tokens))
	return t, nil
}

func (t *Tokenizer) VocabSize() int { return len(t.tokens) }

// ID returns the id of an exact token string (a special token such as "<|im_end|>").
func (t *Tokenizer) ID(literal string) (int, bool) {
	i, ok := t.ids[literal]
	return int(i), ok
}

// Encode tokenizes text. parseSpecial controls the type-3 control literals (<|im_start|>, <|im_end|>, ...); the type-4
// user-defined ones (<think>, <tool_call>, ...) are always matched, exactly like llama.cpp and the Python reference.
func (t *Tokenizer) Encode(text string, parseSpecial bool) []int {
	return t.EncodeWithPlain(text, parseSpecial, nil)
}

// EncodeWithPlain is Encode where a special literal that starts inside one of the plain byte ranges stays ordinary text
// (a "</think>" or "<|im_end|>" quoted in a message). The literal is skipped as a whole, like the reference does.
func (t *Tokenizer) EncodeWithPlain(text string, parseSpecial bool, plain [][2]int) []int {
	inPlain := func(pos int) bool {
		for _, sp := range plain {
			if sp[0] <= pos && pos < sp[1] {
				return true
			}
		}
		return false
	}
	table := t.always
	if parseSpecial {
		table = t.allSpecial
	}
	var out []int
	start, i := 0, 0
	for i < len(text) {
		matched := ""
		for _, lit := range table[text[i]] {
			if strings.HasPrefix(text[i:], lit) {
				matched = lit
				break
			}
		}
		if matched == "" {
			i++
			continue
		}
		if len(plain) > 0 && inPlain(i) {
			i += len(matched)
			continue
		}
		if i > start {
			out = t.encodePlain(text[start:i], out)
		}
		out = append(out, int(t.special[matched]))
		i += len(matched)
		start = i
	}
	if start < len(text) {
		out = t.encodePlain(text[start:], out)
	}
	return out
}

func (t *Tokenizer) encodePlain(text string, out []int) []int {
	rs := []rune(text)
	// invalid UTF-8 would become U+FFFD under []rune; keep the original bytes instead
	if !utf8.ValidString(text) {
		return t.encodeBytesPiece([]byte(text), out)
	}
	for _, span := range pretokenize(rs) {
		out = t.encodeBytesPiece([]byte(string(rs[span[0]:span[1]])), out)
	}
	return out
}

func (t *Tokenizer) encodeBytesPiece(piece []byte, out []int) []int {
	var sb strings.Builder
	for _, b := range piece {
		sb.WriteRune(byteToRune[b])
	}
	word := sb.String()
	t.mu.RLock()
	ids, ok := t.cache[word]
	t.mu.RUnlock()
	if !ok {
		parts := t.bpe(word)
		ids = make([]int32, len(parts))
		for i, p := range parts {
			id, found := t.ids[p]
			if !found {
				// cannot happen with a consistent pack (every merge result is a vocabulary token); fall back to bytes
				for _, r := range p {
					ids = append(ids, t.ids[string(r)])
				}
				continue
			}
			ids[i] = id
		}
		t.mu.Lock()
		if len(t.cache) > 200000 {
			t.cache = map[string][]int32{}
		}
		t.cache[word] = ids
		t.mu.Unlock()
	}
	for _, id := range ids {
		out = append(out, int(id))
	}
	return out
}

const heapMin = 64

// bpe merges by lowest rank first (leftmost among equals), like the reference.
func (t *Tokenizer) bpe(word string) []string {
	rs := []rune(word)
	parts := make([]string, len(rs))
	for i, r := range rs {
		parts[i] = string(r)
	}
	if len(parts) > heapMin {
		return t.bpeHeap(parts)
	}
	for len(parts) > 1 {
		best, bestRank := -1, int32(0)
		for i := 0; i+1 < len(parts); i++ {
			if r, ok := t.ranks[parts[i]+" "+parts[i+1]]; ok && (best < 0 || r < bestRank) {
				best, bestRank = i, r
			}
		}
		if best < 0 {
			break
		}
		parts[best] += parts[best+1]
		parts = append(parts[:best+1], parts[best+2:]...)
	}
	return parts
}

type pairItem struct {
	rank        int32
	pos         int
	left, right string
}
type pairHeap []pairItem

func (h pairHeap) Len() int { return len(h) }
func (h pairHeap) Less(a, b int) bool {
	if h[a].rank != h[b].rank {
		return h[a].rank < h[b].rank
	}
	return h[a].pos < h[b].pos
}
func (h pairHeap) Swap(a, b int) { h[a], h[b] = h[b], h[a] }
func (h *pairHeap) Push(x any)   { *h = append(*h, x.(pairItem)) }
func (h *pairHeap) Pop() any {
	old := *h
	x := old[len(old)-1]
	*h = old[:len(old)-1]
	return x
}

func (t *Tokenizer) bpeHeap(parts []string) []string {
	n := len(parts)
	nxt := make([]int, n)
	prv := make([]int, n)
	live := make([]bool, n)
	for i := range parts {
		nxt[i] = i + 1
		prv[i] = i - 1
		live[i] = true
	}
	nxt[n-1] = -1
	h := &pairHeap{}
	for i := 0; i+1 < n; i++ {
		if r, ok := t.ranks[parts[i]+" "+parts[i+1]]; ok {
			*h = append(*h, pairItem{r, i, parts[i], parts[i+1]})
		}
	}
	heap.Init(h)
	for h.Len() > 0 {
		it := heap.Pop(h).(pairItem)
		i := it.pos
		j := nxt[i]
		if !live[i] || parts[i] != it.left || j < 0 || parts[j] != it.right {
			continue // stale: one of its symbols has merged since
		}
		parts[i] = it.left + it.right
		live[j] = false
		k := nxt[j]
		nxt[i] = k
		if k >= 0 {
			prv[k] = i
			if r, ok := t.ranks[parts[i]+" "+parts[k]]; ok {
				heap.Push(h, pairItem{r, i, parts[i], parts[k]})
			}
		}
		if p := prv[i]; p >= 0 {
			if r, ok := t.ranks[parts[p]+" "+parts[i]]; ok {
				heap.Push(h, pairItem{r, p, parts[p], parts[i]})
			}
		}
	}
	var out []string
	for i := range parts {
		if live[i] {
			out = append(out, parts[i])
		}
	}
	return out
}

// TokenBytes is the raw bytes of one token (a multi-byte character can be split across tokens).
func (t *Tokenizer) TokenBytes(id int) []byte {
	if id < 0 || id >= len(t.tokens) {
		return nil
	}
	t.mu.RLock()
	b := t.tokenBytes[id]
	t.mu.RUnlock()
	if b != nil {
		return b
	}
	tok := t.tokens[id]
	raw := make([]byte, 0, len(tok))
	for _, r := range tok {
		if v, ok := runeToByte[r]; ok {
			raw = append(raw, v)
		} else {
			raw = utf8.AppendRune(raw, r)
		}
	}
	t.mu.Lock()
	t.tokenBytes[id] = raw
	t.mu.Unlock()
	return raw
}

func (t *Tokenizer) Decode(ids []int) string {
	var b []byte
	for _, id := range ids {
		b = append(b, t.TokenBytes(id)...)
	}
	return strings.ToValidUTF8(string(b), "�")
}

// Detokenizer turns a stream of token ids into text without ever emitting half of a multi-byte character.
type Detokenizer struct {
	t       ByteSource
	pending []byte
}

// ByteSource gives the raw bytes of a token id (the BPE tokenizer, or a byte-per-id test tokenizer).
type ByteSource interface{ TokenBytes(id int) []byte }

func NewDetokenizer(t ByteSource) *Detokenizer { return &Detokenizer{t: t} }

// Push adds one token and returns the text that is complete so far ("" if the token ended inside a character).
func (d *Detokenizer) Push(id int) string {
	d.pending = append(d.pending, d.t.TokenBytes(id)...)
	cut := len(d.pending)
	// hold back an incomplete UTF-8 tail (at most 3 bytes)
	for back := 1; back <= 3 && back <= len(d.pending); back++ {
		c := d.pending[len(d.pending)-back]
		if c&0xC0 == 0x80 {
			continue // continuation byte: keep looking for the lead byte
		}
		need := 1
		switch {
		case c&0xE0 == 0xC0:
			need = 2
		case c&0xF0 == 0xE0:
			need = 3
		case c&0xF8 == 0xF0:
			need = 4
		}
		if need > back {
			cut = len(d.pending) - back
		}
		break
	}
	out := strings.ToValidUTF8(string(d.pending[:cut]), "�")
	d.pending = append(d.pending[:0], d.pending[cut:]...)
	return out
}

// Flush returns whatever is left (an incomplete character becomes U+FFFD).
func (d *Detokenizer) Flush() string {
	out := strings.ToValidUTF8(string(d.pending), "�")
	d.pending = d.pending[:0]
	return out
}

func (d *Detokenizer) Pending() bool { return len(d.pending) > 0 }

// ByteToken is the vocabulary spelling of one byte in the byte-level alphabet (used to build small test vocabularies).
func ByteToken(b byte) string { return string(byteToRune[b]) }
