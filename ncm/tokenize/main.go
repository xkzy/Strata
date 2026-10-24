// Tokenize a JSON array of prompts with Strata's own tokenizer into OUT/NN.txt (comma-separated ids for --tokens-file).
// usage: go run ./ncm/tokenize TOKENIZER_DIR PROMPTS.json OUT_DIR
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"

	"strata/pkg/tokenizer"
)

func main() {
	if len(os.Args) != 4 {
		fmt.Fprintln(os.Stderr, "usage: tokenize TOKENIZER_DIR PROMPTS.json OUT_DIR")
		os.Exit(2)
	}
	tk, err := tokenizer.Load(os.Args[1])
	check(err)
	raw, err := os.ReadFile(os.Args[2])
	check(err)
	// accepts ["text", ...] or [{"text": ...}, ...]
	var prompts []string
	if json.Unmarshal(raw, &prompts) != nil {
		prompts = nil // a failed parse can leave zero-value elements behind
		var objs []struct{ Text string }
		check(json.Unmarshal(raw, &objs))
		for _, o := range objs {
			prompts = append(prompts, o.Text)
		}
	}
	check(os.MkdirAll(os.Args[3], 0o755))
	for i, p := range prompts {
		ids := tk.Encode(p, true)
		if back := tk.Decode(ids); back != p { // the mapping must be recoverable: skip, never feed a lossy prompt
			fmt.Printf("%02d SKIPPED: does not round-trip\n", i)
			continue
		}
		s := make([]string, len(ids))
		for j, id := range ids {
			s[j] = strconv.Itoa(id)
		}
		check(os.WriteFile(fmt.Sprintf("%s/%03d.txt", os.Args[3], i), []byte(strings.Join(s, ",")), 0o644))
		fmt.Printf("%02d %3d tokens\n", i, len(ids))
	}
}

func check(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
