package tokenizer

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"strata/pkg/gguf"
)

func asInt(v any) (int, bool) {
	switch x := v.(type) {
	case uint8:
		return int(x), true
	case int8:
		return int(x), true
	case uint16:
		return int(x), true
	case int16:
		return int(x), true
	case uint32:
		return int(x), true
	case int32:
		return int(x), true
	case uint64:
		return int(x), true
	case int64:
		return int(x), true
	}
	return 0, false
}

// ExportGGUF writes the tokenizer pack Load reads (vocab.json, merges.txt, token_type.json, tokenizer.json and, when the
// file has one, chat_template.jinja) from the gpt2-style tokenizer inside a GGUF.
func ExportGGUF(ggufPath, dir string) error {
	g, err := gguf.Open(ggufPath)
	if err != nil {
		return err
	}
	defer g.Close()
	md := g.Metadata
	if m, _ := md["tokenizer.ggml.model"].(string); m != "gpt2" {
		return fmt.Errorf("tokenizer: %s has tokenizer model %q, not gpt2", ggufPath, m)
	}
	toks, _ := md["tokenizer.ggml.tokens"].([]any)
	merges, _ := md["tokenizer.ggml.merges"].([]any)
	types, _ := md["tokenizer.ggml.token_type"].([]any)
	if len(toks) == 0 || len(types) != len(toks) {
		return fmt.Errorf("tokenizer: %s has no usable token list", ggufPath)
	}
	// vocab.json is an object, so two ids sharing a spelling would silently lose one
	var vb bytes.Buffer
	vb.WriteByte('{')
	seen := make(map[string]int, len(toks))
	ty := make([]int, len(toks))
	for i, t := range toks {
		s, _ := t.(string)
		if j, dup := seen[s]; dup {
			return fmt.Errorf("tokenizer: tokens %d and %d are both %q", j, i, s)
		}
		seen[s] = i
		k, _ := json.Marshal(s)
		if i > 0 {
			vb.WriteByte(',')
		}
		fmt.Fprintf(&vb, "%s:%d", k, i)
		ty[i], _ = asInt(types[i])
	}
	vb.WriteByte('}')
	var mb strings.Builder
	for _, m := range merges {
		s, _ := m.(string)
		mb.WriteString(s + "\n")
	}
	cfg := map[string]any{"pre": md["tokenizer.ggml.pre"]}
	ids := map[string]int{}
	for _, k := range []string{"tokenizer.ggml.eos_token_id", "tokenizer.ggml.bos_token_id"} {
		if v, ok := asInt(md[k]); ok {
			ids[k] = v
		}
	}
	cfg["special_ids"] = ids
	var stops []int
	eog, _ := md["tokenizer.ggml.eog_token_ids"].([]any)
	for _, v := range eog {
		if n, ok := asInt(v); ok {
			stops = append(stops, n)
		}
	}
	cfg["stop_ids"] = stops
	tj, _ := json.Marshal(ty)
	cj, _ := json.Marshal(cfg)
	files := map[string][]byte{"vocab.json": vb.Bytes(), "merges.txt": []byte(mb.String()), "token_type.json": tj, "tokenizer.json": cj}
	if ct, _ := md["tokenizer.chat_template"].(string); ct != "" {
		files["chat_template.jinja"] = []byte(ct)
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	for name, b := range files {
		if err := os.WriteFile(filepath.Join(dir, name), b, 0o644); err != nil {
			return err
		}
	}
	return nil
}
