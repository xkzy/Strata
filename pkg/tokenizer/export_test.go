package tokenizer

import (
	"os"
	"reflect"
	"testing"
)

const flintGGUF = "/home/khing/Downloads/Strata-data/models/flint-UD-Q4_K_XL.gguf"

func TestExportFlint(t *testing.T) {
	if _, err := os.Stat(flintGGUF); err != nil {
		t.Skip("flint GGUF not present")
	}
	dir := t.TempDir()
	if err := ExportGGUF(flintGGUF, dir); err != nil {
		t.Fatal(err)
	}
	tk, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !tk.GLM || tk.EOS != 150001 || !reflect.DeepEqual(tk.StopIDs, []int{150001, 150003, 150005, 150018}) {
		t.Fatalf("config: glm=%v eos=%d stops=%v", tk.GLM, tk.EOS, tk.StopIDs)
	}
	for _, s := range []string{"The capital of France is", "x = 1234567 + 89;\n\n  def f(a):\n\treturn 'é' # 你好 🙂", "<|user|>hi<|assistant|>"} {
		ids := tk.Encode(s, true)
		if got := tk.Decode(ids); got != s {
			t.Errorf("round trip %q -> %q", s, got)
		}
	}
	if ids := tk.Encode("<|user|>hi<|assistant|>", true); ids[0] != 150003 || ids[len(ids)-1] != 150004 {
		t.Errorf("special ids: %v", ids)
	}
}

func TestPretokGLM(t *testing.T) {
	split := func(s string) []string {
		rs := []rune(s)
		var out []string
		for _, sp := range pretokenize(rs, true) {
			out = append(out, string(rs[sp[0]:sp[1]]))
		}
		return out
	}
	for in, want := range map[string][]string{
		"abc 1234567":   {"abc", " ", "123", "456", "7"}, // " 1" is not a letter-prefixed word; the space stays alone before digits
		"it's é\u0301x": {"it", "'s", " é", "\u0301x"},
		"a\n\nb":        {"a", "\n\n", "b"},
		"x  y":          {"x", " ", " y"},
	} {
		if got := split(in); !reflect.DeepEqual(got, want) {
			t.Errorf("%q: got %q want %q", in, got, want)
		}
	}
}
