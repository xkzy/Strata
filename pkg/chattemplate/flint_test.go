package chattemplate

import (
	"encoding/json"
	"os"
	"testing"
)

// testdata/flint_golden.json holds the flint GGUF's own chat template rendered by Jinja (tojson = json.dumps, as the
// Hugging Face and llama.cpp runtimes do).
func TestFlintGolden(t *testing.T) {
	raw, err := os.ReadFile("testdata/flint_golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []fxCase
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	for _, c := range cases {
		msgs, opt := convert(c)
		opt.Format = "flint"
		got, err := Render(msgs, opt)
		if err != nil {
			t.Errorf("%s: %v", c.Name, err)
		} else if got != *c.Rendered {
			t.Errorf("%s:\n got %q\nwant %q", c.Name, got, *c.Rendered)
		}
	}
}
