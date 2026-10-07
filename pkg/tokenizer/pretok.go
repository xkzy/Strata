package tokenizer

import "unicode"

// pretokenize splits text into the pieces the Qwen3.5 byte-level BPE merges inside of. It is a hand-written equivalent
// of llama.cpp's LLAMA_VOCAB_PRE_TYPE_QWEN35 pattern (Go's regexp has no look-ahead):
//
//	(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
//	|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
//	|\p{N}
//	| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
//	|\s*[\r\n]+
//	|\s+(?!\S)
//	|\s+
//
// Alternatives are tried in that order at each position (leftmost, first alternative that matches), with the same
// backtracking a regex engine would do. The result is checked against the Python `regex` implementation in the tests.
func pretokenize(rs []rune) [][2]int {
	var out [][2]int
	n := len(rs)
	isL := func(r rune) bool { return unicode.IsLetter(r) }
	isLM := func(r rune) bool { return unicode.IsLetter(r) || unicode.IsMark(r) }
	isN := func(r rune) bool { return unicode.IsNumber(r) }
	isNL := func(r rune) bool { return r == '\r' || r == '\n' }
	// class of alternative 4's body: not whitespace, letter, mark or number
	isPunct := func(r rune) bool {
		return !unicode.IsSpace(r) && !unicode.IsLetter(r) && !unicode.IsMark(r) && !unicode.IsNumber(r)
	}

	i := 0
	for i < n {
		c := rs[i]
		// 1. contractions
		if c == '\'' && i+1 < n {
			d := rs[i+1]
			if d == 's' || d == 'S' || d == 't' || d == 'T' || d == 'm' || d == 'M' || d == 'd' || d == 'D' {
				out = append(out, [2]int{i, i + 2})
				i += 2
				continue
			}
			if i+2 < n {
				e := rs[i+2]
				if ((d == 'r' || d == 'R') && (e == 'e' || e == 'E')) || ((d == 'v' || d == 'V') && (e == 'e' || e == 'E')) ||
					((d == 'l' || d == 'L') && (e == 'l' || e == 'L')) {
					out = append(out, [2]int{i, i + 3})
					i += 3
					continue
				}
			}
		}
		// 2. optional single non-letter/number/CRLF character, then letters/marks
		if !isNL(c) && !isL(c) && !isN(c) && i+1 < n && isLM(rs[i+1]) {
			j := i + 1
			for j < n && isLM(rs[j]) {
				j++
			}
			out = append(out, [2]int{i, j})
			i = j
			continue
		}
		if isLM(c) {
			j := i + 1
			for j < n && isLM(rs[j]) {
				j++
			}
			out = append(out, [2]int{i, j})
			i = j
			continue
		}
		// 3. one number character
		if isN(c) {
			out = append(out, [2]int{i, i + 1})
			i++
			continue
		}
		// 4. optional space, run of punctuation/symbols, trailing newlines
		{
			j := i
			if c == ' ' && i+1 < n && isPunct(rs[i+1]) {
				j = i + 1
			}
			if j < n && isPunct(rs[j]) {
				for j < n && isPunct(rs[j]) {
					j++
				}
				for j < n && isNL(rs[j]) {
					j++
				}
				out = append(out, [2]int{i, j})
				i = j
				continue
			}
		}
		// c is whitespace from here on (every other class was handled above)
		j := i
		for j < n && unicode.IsSpace(rs[j]) {
			j++
		}
		// 5. whitespace ending in a newline: up to the last CR/LF of the run
		last := -1
		for k := i; k < j; k++ {
			if isNL(rs[k]) {
				last = k
			}
		}
		if last >= 0 {
			out = append(out, [2]int{i, last + 1})
			i = last + 1
			continue
		}
		// 6. whitespace not followed by a non-space: all of it at the end of the text, else all but the last character
		if j == n {
			out = append(out, [2]int{i, j})
			i = j
			continue
		}
		if j-i >= 2 {
			out = append(out, [2]int{i, j - 1})
			i = j - 1
			continue
		}
		// 7. a single whitespace character before text
		out = append(out, [2]int{i, j})
		i = j
	}
	return out
}
