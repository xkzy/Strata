package tune

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func base() map[string]interface{} {
	return map[string]interface{}{
		"args": []interface{}{"--pack", "p", "--layer-split", "auto", "--pcie-frac", "0.5", "--abliterate"},
		"env":  map[string]interface{}{"STRATA_CVEC_FUSE": "1"},
		"port": 8080,
	}
}

func TestParseServedTakesTheLastRequest(t *testing.T) {
	out := "x\nstrata serve: prompt 100 tokens = 0 reused + 100 read in 50 ms (2000.0 tok/s), 10 generated in 500 ms (20.0 tok/s), drafts accepted 1 of 2, 1 checkpoints\n" +
		"strata serve: prompt 2113 tokens = 4 reused + 2109 read in 1204 ms (1751.7 tok/s), 256 generated in 6100 ms (42.0 tok/s), drafts accepted 1 of 2, 1 checkpoints\n"
	r, ok := ParseServed(out)
	if !ok || r.DecodeTokS != 42.0 || r.PrefillTokS != 1751.7 || r.DecodeTokens != 256 || r.PrefillTokens != 2109 {
		t.Fatalf("got %+v ok=%v", r, ok)
	}
	if _, ok := ParseServed("nothing here"); ok {
		t.Fatal("no line should not parse")
	}
}

func TestCandidates(t *testing.T) {
	if Candidates(base(), 0) != nil {
		t.Fatal("no cards, no candidates")
	}
	if n := len(Candidates(base(), 1)); n != 1 {
		t.Fatalf("one card: %d candidates", n)
	}
	cs := Candidates(base(), 2)
	var names []string
	for _, c := range cs {
		names = append(names, c.Name)
	}
	if strings.Join(names, ",") != "gpu0,gpu1,split,split-probe,peer" {
		t.Fatalf("names = %v", names)
	}
	flat := func(c Candidate) string { return strings.Join(argsOf(c.Config), " ") }
	if strings.Contains(flat(cs[0]), "--layer-split") || !strings.Contains(flat(cs[0]), "--abliterate") {
		t.Errorf("single card: %s", flat(cs[0]))
	}
	if !strings.Contains(flat(cs[2]), "--layer-split auto") || !strings.Contains(flat(cs[2]), "--pcie-frac 0.5") {
		t.Errorf("split: %s", flat(cs[2]))
	}
	if strings.Contains(flat(cs[3]), "--pcie-frac") {
		t.Errorf("split-probe keeps --pcie-frac: %s", flat(cs[3]))
	}
	if !strings.Contains(flat(cs[4]), "--peer-device 1") || strings.Contains(flat(cs[4]), "--layer-split") {
		t.Errorf("peer: %s", flat(cs[4]))
	}
	withPeer := base()
	withPeer["args"] = append(argsOf(withPeer), "--peer-device", "1")
	for _, c := range Candidates(withPeer, 2) {
		if c.Name != "peer" && strings.Contains(strings.Join(argsOf(c.Config), " "), "--peer-device") {
			t.Errorf("%s keeps the user's --peer-device", c.Name)
		}
	}
	env := cs[1].Config["env"].(map[string]interface{})
	if env["CUDA_VISIBLE_DEVICES"] != "1" || env["CUDA_DEVICE_ORDER"] != "PCI_BUS_ID" || env["STRATA_CVEC_FUSE"] != "1" {
		t.Errorf("env = %v", env)
	}
	if _, in := base()["env"].(map[string]interface{})["CUDA_VISIBLE_DEVICES"]; in {
		t.Error("base was changed")
	}
}

func TestReserveMore(t *testing.T) {
	c := Candidates(base(), 1)[0]
	flat := func(c Candidate) string { return strings.Join(argsOf(c.Config), " ") }
	v, ok := ReserveMore(c)
	if !ok || !strings.Contains(flat(v), "--vram-reserve-mib 2200") {
		t.Fatalf("first step from the default 700: %v %s", ok, flat(v))
	}
	v, ok = ReserveMore(v)
	if !ok || !strings.Contains(flat(v), "--vram-reserve-mib 3700") || strings.Count(flat(v), "--vram-reserve-mib") != 1 {
		t.Fatalf("second step: %v %s", ok, flat(v))
	}
	if strings.Contains(flat(c), "--vram-reserve-mib") {
		t.Error("the original candidate was changed")
	}
	s := "6000"
	withArgs(c.Config, SetArg(argsOf(c.Config), "--vram-reserve-mib", &s))
	if _, ok := ReserveMore(c); ok {
		t.Error("no step past the maximum")
	}
}

func TestChoose(t *testing.T) {
	if Choose([]Result{{Name: "a", Err: "x"}}) != -1 {
		t.Fatal("all failed")
	}
	rs := []Result{{Name: "a", DecodeTokS: 20, PrefillTokS: 900}, {Name: "b", DecodeTokS: 30, PrefillTokS: 800},
		{Name: "c", DecodeTokS: 29.5, PrefillTokS: 1200}, {Name: "d", Err: "boom"}}
	if i := Choose(rs); rs[i].Name != "c" { // c is within 3% of b and reads prompts faster
		t.Fatalf("chose %s", rs[i].Name)
	}
}

func TestApplyChangesOnlyThePlacement(t *testing.T) {
	path := filepath.Join(t.TempDir(), "strata-m.json")
	orig, _ := json.Marshal(map[string]interface{}{
		"args": []interface{}{"--pack", "p", "--layer-split", "auto", "--pcie-frac", "0.5", "--abliterate"},
		"env":  map[string]interface{}{"STRATA_CVEC_FUSE": "1"}, "port": 8080})
	if err := os.WriteFile(path, orig, 0o644); err != nil {
		t.Fatal(err)
	}
	cs := Candidates(base(), 2)
	if err := Apply(path, cs[1], Record{Chosen: "gpu1", Hardware: []string{"GPU 0: a", "GPU 1: b"}}); err != nil {
		t.Fatal(err)
	}
	var got map[string]interface{}
	raw, _ := os.ReadFile(path)
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatal(err)
	}
	args := strings.Join(argsOf(got), " ")
	if strings.Contains(args, "--layer-split") || !strings.Contains(args, "--pcie-frac 0.5") || !strings.Contains(args, "--abliterate") {
		t.Errorf("args = %s", args)
	}
	env := got["env"].(map[string]interface{})
	if env["CUDA_VISIBLE_DEVICES"] != "1" || env["STRATA_CVEC_FUSE"] != "1" || got["port"] != float64(8080) {
		t.Errorf("env/port = %v / %v", env, got["port"])
	}
	if got["tuned"].(map[string]interface{})["chosen"] != "gpu1" {
		t.Errorf("tuned = %v", got["tuned"])
	}
	if bak, _ := os.ReadFile(path + ".bak"); string(bak) != string(orig) {
		t.Error("backup is not the original")
	}
}

func TestResidentVariantAndApply(t *testing.T) {
	b := base()
	b["args"] = append(b["args"].([]interface{}), "--mmap-experts")
	c := Candidates(b, 2)[2]
	v, ok := ResidentVariant(c)
	if !ok || v.Name != "split+resident" {
		t.Fatalf("variant = %v %v", v.Name, ok)
	}
	if a := strings.Join(argsOf(v.Config), " "); strings.Contains(a, "--mmap-experts") || !strings.Contains(a, "--resident-experts") {
		t.Errorf("args = %s", a)
	}
	if _, again := ResidentVariant(v); again {
		t.Error("a resident run has no variant")
	}
	if _, none := ResidentVariant(Candidates(base(), 2)[0]); none {
		t.Error("no memory flag, no variant")
	}
	path := filepath.Join(t.TempDir(), "strata-m.json")
	orig, _ := json.Marshal(b)
	_ = os.WriteFile(path, orig, 0o644)
	if err := Apply(path, v, Record{Chosen: v.Name}); err != nil {
		t.Fatal(err)
	}
	var got map[string]interface{}
	raw, _ := os.ReadFile(path)
	_ = json.Unmarshal(raw, &got)
	if a := strings.Join(argsOf(got), " "); strings.Contains(a, "--mmap-experts") || !strings.Contains(a, "--resident-experts") || !strings.Contains(a, "--layer-split auto") {
		t.Errorf("applied args = %s", a)
	}
}
