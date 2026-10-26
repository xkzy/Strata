package densecpu

import (
	"strings"

	"strata/pkg/gguf"
)

func vocabOf(g *gguf.GGUFFile) []string {
	toks, _ := g.Metadata["tokenizer.ggml.tokens"].([]any)
	out := make([]string, len(toks))
	for i, t := range toks {
		out[i], _ = t.(string)
	}
	return out
}

// greedyEncode is a test helper, not the real tokenizer (no merges): spaces become space (the vocab's marker for
// one) and the longest vocab entry is taken at each step. It is enough for plain English reference prompts.
func greedyEncode(vocab []string, text, space string) []int {
	index := make(map[string]int, len(vocab))
	for i, s := range vocab {
		if _, ok := index[s]; !ok {
			index[s] = i
		}
	}
	text = strings.ReplaceAll(text, " ", space)
	var out []int
	for len(text) > 0 {
		best, bestLen := -1, 0
		for l := len(text); l > 0; l-- {
			if id, ok := index[text[:l]]; ok {
				best, bestLen = id, l
				break
			}
		}
		if best < 0 {
			panic("densecpu: cannot encode " + text)
		}
		out = append(out, best)
		text = text[bestLen:]
	}
	return out
}

// GreedyEncode encodes text with Gemma's U+2581 space marker.
func (m *Gemma) GreedyEncode(text string) []int { return greedyEncode(m.Vocab, text, "▁") }
