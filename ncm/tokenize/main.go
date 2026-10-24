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
	var prompts []string
	check(json.Unmarshal(raw, &prompts))
	check(os.MkdirAll(os.Args[3], 0o755))
	for i, p := range prompts {
		ids := tk.Encode(p, true)
		if back := tk.Decode(ids); back != p { // the mapping must be recoverable
			check(fmt.Errorf("prompt %d does not round-trip: %q", i, back))
		}
		s := make([]string, len(ids))
		for j, id := range ids {
			s[j] = strconv.Itoa(id)
		}
		check(os.WriteFile(fmt.Sprintf("%s/%02d.txt", os.Args[3], i), []byte(strings.Join(s, ",")), 0o644))
		fmt.Printf("%02d %3d tokens\n", i, len(ids))
	}
}

func check(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
