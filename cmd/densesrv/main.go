// densesrv serves the dense Qwen3.8-27B (pkg/densecpu, CUDA build) behind a minimal OpenAI-compatible
// /v1/chat/completions, for agent harnesses. One request at a time; the session is reused when a new prompt extends
// the previous one token for token (the delta-net state cannot be rolled back, so anything else re-prefills).
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"strata/pkg/chattemplate"
	"strata/pkg/densecpu"
	"strata/pkg/tokenizer"
)

type server struct {
	mu      sync.Mutex
	q       *densecpu.Qwen
	tok     *tokenizer.Tokenizer
	st      *densecpu.QwenState
	seen    []int // tokens the session has consumed
	ctx     int
	imEnd   int
	nextBuf []float32 // logits after the last consumed token
}

type chatReq struct {
	Messages []struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	} `json:"messages"`
	Tools       []json.RawMessage `json:"tools"`
	Temperature *float64          `json:"temperature"`
	TopP        *float64          `json:"top_p"`
	MaxTokens   int               `json:"max_tokens"`
	Stop        json.RawMessage   `json:"stop"`
	Think       *bool             `json:"enable_thinking"`
}

func flatten(raw json.RawMessage) string {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var parts []struct{ Text string }
	if json.Unmarshal(raw, &parts) == nil {
		var b strings.Builder
		for _, p := range parts {
			b.WriteString(p.Text)
		}
		return b.String()
	}
	return ""
}

func sample(logits []float32, temp, topP float64, rng *rand.Rand) int {
	if temp <= 0 {
		b := 0
		for i, v := range logits {
			if v > logits[b] {
				b = i
			}
		}
		return b
	}
	idx := make([]int, len(logits))
	for i := range idx {
		idx[i] = i
	}
	const topK = 40
	sort.Slice(idx, func(a, b int) bool { return logits[idx[a]] > logits[idx[b]] })
	idx = idx[:topK]
	mx := float64(logits[idx[0]])
	p := make([]float64, topK)
	var sum float64
	for i, id := range idx {
		p[i] = math.Exp((float64(logits[id]) - mx) / temp)
		sum += p[i]
	}
	var acc float64
	cut := topK
	for i := range p {
		acc += p[i] / sum
		if topP > 0 && acc >= topP {
			cut = i + 1
			break
		}
	}
	var s2 float64
	for _, v := range p[:cut] {
		s2 += v
	}
	r := rng.Float64() * s2
	for i := 0; i < cut; i++ {
		if r -= p[i]; r <= 0 {
			return idx[i]
		}
	}
	return idx[cut-1]
}

func (s *server) chat(w http.ResponseWriter, r *http.Request) {
	var req chatReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	var msgs []chattemplate.Message
	for _, m := range req.Messages {
		msgs = append(msgs, chattemplate.Message{Role: m.Role, Content: flatten(m.Content)})
	}
	text, plain, err := chattemplate.RenderSpans(msgs, chattemplate.Options{Tools: req.Tools, AddGenerationPrompt: true, EnableThinking: req.Think})
	if err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	ids := s.tok.EncodeWithPlain(text, true, plain)
	var stops []string
	var one string
	if json.Unmarshal(req.Stop, &one) == nil && one != "" {
		stops = []string{one}
	} else {
		_ = json.Unmarshal(req.Stop, &stops)
	}
	temp, topP := 0.0, 1.0
	if req.Temperature != nil {
		temp = *req.Temperature
	}
	if req.TopP != nil {
		topP = *req.TopP
	}
	maxTok := req.MaxTokens
	if maxTok <= 0 || maxTok > 8192 {
		maxTok = 8192
	}
	if len(ids)+maxTok > s.ctx {
		http.Error(w, fmt.Sprintf("prompt %d + max_tokens %d exceeds context %d", len(ids), maxTok, s.ctx), 400)
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	t0 := time.Now()
	common := 0
	for common < len(s.seen) && common < len(ids) && s.seen[common] == ids[common] {
		common++
	}
	if s.st == nil || common < len(s.seen) || common == len(ids) { // not an extension (or nothing new to feed): start over
		if s.st != nil {
			s.st.Free()
		}
		s.st, s.seen, common = s.q.NewState(), nil, 0
	}
	feed := func(tk []int) []float32 {
		var lg []float32
		for len(tk) > 0 {
			n := min(512, len(tk))
			lg = s.q.Step(s.st, tk[:n])
			s.seen = append(s.seen, tk[:n]...)
			tk = tk[n:]
		}
		return lg
	}
	logits := feed(ids[common:])
	prefill := time.Since(t0)
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	det := tokenizer.NewDetokenizer(s.tok)
	var out strings.Builder
	n, finish := 0, "length"
gen:
	for ; n < maxTok; n++ {
		id := sample(logits, temp, topP, rng)
		if id == s.imEnd || id == s.tok.EOS {
			finish = "stop"
			break
		}
		out.WriteString(det.Push(id))
		for _, st := range stops {
			if i := strings.Index(out.String(), st); i >= 0 {
				o := out.String()[:i]
				out.Reset()
				out.WriteString(o)
				finish = "stop"
				break gen
			}
		}
		logits = feed([]int{id})
	}
	out.WriteString(det.Flush())
	log.Printf("prompt %d tokens (%d new, prefill %.1fs), %d generated in %.1fs, %s", len(ids), len(ids)-common, prefill.Seconds(), n, time.Since(t0).Seconds()-prefill.Seconds(), finish)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"id": "chatcmpl-dense", "object": "chat.completion", "created": time.Now().Unix(), "model": "qwen3.8-27b",
		"choices": []any{map[string]any{"index": 0, "finish_reason": finish, "message": map[string]any{"role": "assistant", "content": out.String()}}},
		"usage":   map[string]int{"prompt_tokens": len(ids), "completion_tokens": n, "total_tokens": len(ids) + n},
	})
}

func main() {
	model := flag.String("model", "", "Qwen3.8-27B GGUF")
	tokDir := flag.String("tokenizer", "", "pack tokenizer directory (vocab.json, merges.txt, ...)")
	addr := flag.String("addr", "127.0.0.1:8081", "listen address")
	ctx := flag.Int("ctx", 32768, "context limit")
	flag.Parse()
	q, err := densecpu.OpenQwen(*model)
	if err != nil {
		log.Fatal(err)
	}
	tok, err := tokenizer.Load(*tokDir)
	if err != nil {
		log.Fatal(err)
	}
	im, ok := tok.ID("<|im_end|>")
	if !ok || q.Vocab[im] != "<|im_end|>" {
		log.Fatalf("tokenizer does not match the model vocabulary (<|im_end|> = %d)", im)
	}
	if _, err := q.EnableGPU(); err != nil {
		log.Fatal(err)
	}
	s := &server{q: q, tok: tok, ctx: *ctx, imEnd: im}
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", s.chat)
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"object":"list","data":[{"id":"qwen3.8-27b","object":"model"}]}`)
	})
	log.Printf("listening on %s", *addr)
	log.Fatal(http.ListenAndServe(*addr, mux))
}
