package tokenizer

import (
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"
)

// The pack's tokenizer directory is large and lives outside the repository: tests that need it are skipped without it.
func loadPack(t *testing.T) *Tokenizer {
	t.Helper()
	dir := os.Getenv("STRATA_TOKENIZER_DIR")
	if dir == "" {
		dir = "/home/khing/Downloads/Strata-data/packs/coder-iq1_m/tokenizer"
	}
	if _, err := os.Stat(dir + "/vocab.json"); err != nil {
		t.Skipf("no pack tokenizer at %s (set STRATA_TOKENIZER_DIR)", dir)
	}
	tk, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	return tk
}

type oracleCase struct {
	Text       string `json:"text"`
	IDs        []int  `json:"ids"`
	IDsSpecial []int  `json:"ids_special"`
}

func TestMatchesPythonReference(t *testing.T) {
	tk := loadPack(t)
	raw, err := os.ReadFile("testdata/oracle.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []oracleCase
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	bad := 0
	for _, c := range cases {
		got := tk.Encode(c.Text, false)
		gotS := tk.Encode(c.Text, true)
		if !reflect.DeepEqual(append([]int{}, got...), append([]int{}, c.IDs...)) || !reflect.DeepEqual(append([]int{}, gotS...), append([]int{}, c.IDsSpecial...)) {
			bad++
			if bad <= 5 {
				short := c.Text
				if len(short) > 60 {
					short = short[:60]
				}
				t.Errorf("mismatch for %q: got %v want %v (special: got %v want %v)", short, head(got), head(c.IDs), head(gotS), head(c.IDsSpecial))
			}
		}
	}
	if bad > 0 {
		t.Fatalf("%d of %d strings differ from the Python reference", bad, len(cases))
	}
}

func head(a []int) []int {
	if len(a) > 12 {
		return a[:12]
	}
	return a
}

func TestRoundTrip(t *testing.T) {
	tk := loadPack(t)
	for _, s := range []string{"", "Hello, world!", "  spaces  ", "a\n\n\nb", "你好，世界", "\U0001f600\U0001f680", "é combining", "\x00\x01\x7f", "tab\there"} {
		if got := tk.Decode(tk.Encode(s, false)); got != s {
			t.Errorf("round trip %q -> %q", s, got)
		}
	}
}

func TestSpecialTokens(t *testing.T) {
	tk := loadPack(t)
	end, ok := tk.ID("<|im_end|>")
	if !ok {
		t.Fatal("no <|im_end|> in the vocabulary")
	}
	if ids := tk.Encode("<|im_end|>", true); len(ids) != 1 || ids[0] != end {
		t.Errorf("<|im_end|> with parse_special = %v, want [%d]", ids, end)
	}
	if ids := tk.Encode("<|im_end|>", false); len(ids) == 1 {
		t.Errorf("<|im_end|> must be ordinary text without parse_special, got %v", ids)
	}
	if think, _ := tk.ID("<think>"); true {
		if ids := tk.Encode("<think>", false); len(ids) != 1 || ids[0] != think {
			t.Errorf("<think> is a user-defined token and is always matched, got %v", ids)
		}
	}
}

func TestDetokenizerNeverSplitsACharacter(t *testing.T) {
	tk := loadPack(t)
	text := "Hello 你好 \U0001f600 café สวัสดี"
	d := NewDetokenizer(tk)
	var sb strings.Builder
	for _, id := range tk.Encode(text, false) {
		piece := d.Push(id)
		if strings.Contains(piece, "�") {
			t.Fatalf("emitted a broken character: %q", piece)
		}
		sb.WriteString(piece)
	}
	sb.WriteString(d.Flush())
	if sb.String() != text {
		t.Fatalf("streamed %q, want %q", sb.String(), text)
	}
	// a token boundary inside a character: feed the bytes of an emoji one id at a time via the byte tokens
	d2 := NewDetokenizer(tk)
	emoji := []byte("\U0001f600")
	var out strings.Builder
	for _, b := range emoji {
		id, ok := tk.ID(string(byteToRune[b]))
		if !ok {
			t.Skip("single-byte token missing")
		}
		out.WriteString(d2.Push(id))
	}
	if out.String() != "\U0001f600" {
		t.Fatalf("byte-split emoji decoded as %q", out.String())
	}
}

func BenchmarkEncode(b *testing.B) {
	tk := loadPack(&testing.T{})
	text := strings.Repeat("The quick brown fox jumps over the lazy dog. def f(x): return x\n", 200)
	b.SetBytes(int64(len(text)))
	for i := 0; i < b.N; i++ {
		tk.Encode(text, false)
	}
}

func TestControlTokensInsideCallerTextStayText(t *testing.T) {
	tk := loadPack(t)
	end, _ := tk.ID("<|im_end|>")
	text := "<|im_start|>user\nhi <|im_end|> there<|im_end|>\n"
	start := strings.Index(text, "hi")
	plain := [][2]int{{start, start + len("hi <|im_end|> there")}}
	ids := tk.EncodeWithPlain(text, true, plain)
	n := 0
	for _, id := range ids {
		if id == end {
			n++
		}
	}
	if n != 1 {
		t.Fatalf("exactly the real <|im_end|> may be a control token, found %d in %v", n, ids)
	}
	if tk.Decode(ids) != text {
		t.Fatalf("round trip %q", tk.Decode(ids))
	}
}
