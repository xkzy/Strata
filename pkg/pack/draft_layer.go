package pack

import (
	"fmt"
	"strings"
	"unicode"
)

// FilterDraftVocab filters token indices matching a specific language/script subset
func FilterDraftVocab(tokens []string, script string) ([]int32, error) {
	script = strings.ToLower(strings.TrimSpace(script))
	var indices []int32

	for i, tok := range tokens {
		if matchesScript(tok, script) {
			indices = append(indices, int32(i))
		}
	}

	if len(indices) == 0 {
		return nil, fmt.Errorf("no tokens matched script %q", script)
	}
	return indices, nil
}

func matchesScript(token, script string) bool {
	switch script {
	case "cjk":
		for _, r := range token {
			if unicode.Is(unicode.Han, r) || unicode.Is(unicode.Hiragana, r) || unicode.Is(unicode.Katakana, r) || unicode.Is(unicode.Hangul, r) {
				return true
			}
		}
		return false
	case "cyrillic":
		for _, r := range token {
			if unicode.Is(unicode.Cyrillic, r) {
				return true
			}
		}
		return false
	case "fr":
		lower := strings.ToLower(token)
		for _, r := range lower {
			if r == 'é' || r == 'à' || r == 'è' || r == 'ù' || r == 'ç' || r == 'â' || r == 'ê' || r == 'î' || r == 'ô' || r == 'û' {
				return true
			}
		}
		return false
	case "en", "latin":
		for _, r := range token {
			if unicode.Is(unicode.Latin, r) {
				return true
			}
		}
		return false
	default:
		return true
	}
}
