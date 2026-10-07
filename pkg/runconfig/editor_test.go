package runconfig

import (
	"os"
	"path/filepath"
	"testing"
)

func TestViewListsEditableKeys(t *testing.T) {
	cfg := map[string]interface{}{
		"sampling": map[string]interface{}{"temperature": 0.7},
		"args":     []interface{}{"--vram-reserve-mib", "900"},
		"aliases":  []interface{}{"a", "b"},
	}
	v := View(cfg, "/x/strata-m.json")
	if v["file"] != "strata-m.json" {
		t.Fatalf("file = %v", v["file"])
	}
	byKey := map[string]map[string]interface{}{}
	for _, k := range v["keys"].([]map[string]interface{}) {
		byKey[k["key"].(string)] = k
	}
	if byKey["sampling.temperature"]["value"] != 0.7 || byKey["sampling.temperature"]["kind"] != "number" {
		t.Errorf("temperature: %v", byKey["sampling.temperature"])
	}
	if byKey["vram_reserve_mib"]["value"] != 900 {
		t.Errorf("vram: %v", byKey["vram_reserve_mib"])
	}
	if c, _ := byKey["power_policy"]["choices"].([]string); len(c) != 4 || byKey["power_policy"]["kind"] != "enum" {
		t.Errorf("power_policy: %v", byKey["power_policy"])
	}
	if byKey["fit_max_tokens"]["value"] != nil || byKey["fit_max_tokens"]["kind"] != "bool" {
		t.Errorf("fit_max_tokens: %v", byKey["fit_max_tokens"])
	}
}

func TestApplyChangesOnlyNamedKeys(t *testing.T) {
	cfg := map[string]interface{}{"exe": "engine", "args": []interface{}{"--x", "1"}, "api_key": "secret"}
	nw, changed, err := Apply(cfg, map[string]interface{}{
		"sampling.top_k": float64(20), "power_policy": "LOW_LATENCY", "vram_reserve_mib": float64(512), "aliases": "m1, m2",
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(changed) != 4 {
		t.Errorf("changed = %v", changed)
	}
	if nw["api_key"] != "secret" || nw["exe"] != "engine" {
		t.Errorf("other keys must stay: %v", nw)
	}
	if nw["sampling"].(map[string]interface{})["top_k"] != 20 || nw["power_policy"] != "LOW_LATENCY" {
		t.Errorf("not applied: %v", nw)
	}
	args := nw["args"].([]interface{})
	if len(args) != 4 || args[2] != "--vram-reserve-mib" || args[3] != "512" {
		t.Errorf("args = %v", args)
	}
	if _, ok := cfg["sampling"]; ok {
		t.Error("the input config must be left alone")
	}
	// null removes a key
	nw2, _, err := Apply(nw, map[string]interface{}{"power_policy": nil, "vram_reserve_mib": nil})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := nw2["power_policy"]; ok || len(nw2["args"].([]interface{})) != 2 {
		t.Errorf("null must remove: %v", nw2)
	}
}

func TestApplyRejectsBadValuesWholesale(t *testing.T) {
	cfg := map[string]interface{}{}
	for _, bad := range []map[string]interface{}{
		{"api_key": "x"},
		{"sampling.top_p": float64(0)},
		{"sampling.top_k": float64(65)},
		{"power_policy": "TURBO"},
		{"fit_max_tokens": "yes"},
		{},
	} {
		if _, _, err := Apply(cfg, bad); err == nil {
			t.Errorf("%v must be refused", bad)
		}
	}
	if _, _, err := Apply(cfg, map[string]interface{}{"fit_max_tokens": true, "power_policy": "TURBO"}); err == nil {
		t.Error("one bad value must refuse the whole request")
	}
}

func TestSaveKeepsBackup(t *testing.T) {
	p := filepath.Join(t.TempDir(), "strata-m.json")
	if err := os.WriteFile(p, []byte(`{"exe":"e"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	bak, err := SaveRaw(p, map[string]interface{}{"exe": "e", "fit_max_tokens": true})
	if err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(bak); string(got) != `{"exe":"e"}` {
		t.Errorf("backup = %q", got)
	}
	cfg, err := LoadRaw(p)
	if err != nil || cfg["fit_max_tokens"] != true {
		t.Errorf("reload = %v, %v", cfg, err)
	}
}

func TestSaveKeepsOwnerOnlyMode(t *testing.T) {
	p := filepath.Join(t.TempDir(), "strata-m.json")
	if err := os.WriteFile(p, []byte(`{"api_key":"k"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	bak, err := SaveRaw(p, map[string]interface{}{"api_key": "k", "fit_max_tokens": true})
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range []string{p, bak} {
		if st, _ := os.Stat(f); st.Mode().Perm() != 0o600 {
			t.Errorf("%s mode = %v, want 0600", f, st.Mode().Perm())
		}
	}
}
