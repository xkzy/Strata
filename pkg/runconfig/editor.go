// pkg/runconfig/editor.go - the web page's Settings view of the run config (strata-<model>.json), #564.
//
// A short list of documented keys can be read and changed from Strata's own page (GET / POST /config). Everything else
// in the file (the engine command line, the keys setup writes, the network and security keys, the MCP servers) is kept
// as it is: a change touches only the keys it names. The earlier file is kept as <name>.bak. The server reads the file
// when it starts, so a change is used from the next start on.
package runconfig

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

type editableKey struct {
	key, kind, help string
	choices         []string // kind "enum"
	rule            string   // kind "sampling": the value's rule
	flag            string   // kind "arg": the engine option kept in "args"
}

var editable = []editableKey{
	{key: "sampling.temperature", kind: "sampling", rule: "num>=0",
		help: "Default temperature for requests that send none (0 = greedy, the default without a sampling block)"},
	{key: "sampling.top_p", kind: "sampling", rule: "0<x<=1", help: "Default top_p for requests that send none"},
	{key: "sampling.top_k", kind: "sampling", rule: "1..64", help: "Default top_k for requests that send none (1-64)"},
	{key: "sampling.min_p", kind: "sampling", rule: "0<=x<=1", help: "Default min_p for requests that send none"},
	{key: "reasoning_budget_tokens", kind: "int>=0", help: "Cap the thinking of every request at this many tokens (0 or empty: no cap)"},
	{key: "fit_max_tokens", kind: "bool", help: "Shorten a max_tokens that does not fit the context instead of answering 400"},
	{key: "anthropic_thinking", kind: "enum", choices: []string{"model", "on_request"},
		help: "Anthropic requests that do not ask for thinking: think as the model does (model) or not (on_request)"},
	{key: "effort_position", kind: "enum", choices: []string{"start", "end"},
		help: "Where a non-default reasoning effort goes: start (the default) or end (keeps the cache when it changes)"},
	{key: "aliases", kind: "names", help: "Other model names the server lists and answers to (comma-separated)"},
	{key: "idle_unload_s", kind: "num>=0", help: "Unload the model after this many seconds without requests (0 or empty: never)"},
	{key: "lazy_load", kind: "bool", help: "Start without loading the model; the first request loads it (text only)"},
	{key: "engine_silence_s", kind: "num>=0", help: "End a request when the engine says nothing for this long (default 300 s, 0 = wait)"},
	{key: "api_monitor", kind: "bool", help: "Keep the last 100 requests' prompts and answers in memory for /api-monitor"},
	{key: "open_browser", kind: "bool", help: "Open the chat page in the browser when the model is ready"},
	{key: "vram_reserve_mib", kind: "arg", flag: "--vram-reserve-mib",
		help: "VRAM in MiB the engine leaves free for other programs (engine default 700)"},
	{key: "power_policy", kind: "enum", choices: []string{"MAX_THROUGHPUT", "LOW_LATENCY", "BALANCED", "ENERGY_SAVING"},
		help: "What the adaptive resource controller aims for (#18): throughput, latency, a balance, or saved energy"},
	{key: "resource_adapt", kind: "bool",
		help: "Adapt queue limits and admission to measured load (#18); off keeps one fixed limit whatever the load"},
}

func spec(key string) (editableKey, bool) {
	for _, e := range editable {
		if e.key == key {
			return e, true
		}
	}
	return editableKey{}, false
}

func argValue(cfg map[string]interface{}, flag string) interface{} {
	args, _ := cfg["args"].([]interface{})
	for i := 0; i+1 < len(args); i++ {
		if s, _ := args[i].(string); s == flag {
			v := fmt.Sprint(args[i+1])
			if n, err := strconv.Atoi(v); err == nil {
				return n
			}
			return v
		}
	}
	return nil
}

func valueOf(cfg map[string]interface{}, e editableKey) interface{} {
	switch e.kind {
	case "sampling":
		s, _ := cfg["sampling"].(map[string]interface{})
		return s[strings.TrimPrefix(e.key, "sampling.")]
	case "arg":
		return argValue(cfg, e.flag)
	}
	return cfg[e.key]
}

// View is GET /config: the editable keys with their values (nil: not set, the default applies).
func View(cfg map[string]interface{}, path string) map[string]interface{} {
	keys := make([]map[string]interface{}, 0, len(editable))
	for _, e := range editable {
		kind := e.kind
		switch kind {
		case "sampling", "arg", "int>=0", "num>=0":
			kind = "number"
		}
		k := map[string]interface{}{"key": e.key, "value": valueOf(cfg, e), "help": e.help, "kind": kind}
		if e.kind == "enum" {
			k["choices"] = e.choices
		}
		keys = append(keys, k)
	}
	return map[string]interface{}{"file": filepath.Base(path), "keys": keys,
		"note": "Saved to the run config; used from the next start of the model."}
}

func number(key string, v interface{}, whole bool, lo float64, hi *float64, loOpen bool) (float64, error) {
	f, ok := v.(float64)
	if !ok {
		if n, isInt := v.(int); isInt {
			f, ok = float64(n), true
		}
	}
	bad := !ok || math.IsNaN(f) || math.IsInf(f, 0) || (whole && f != math.Trunc(f)) ||
		(loOpen && f <= lo) || (!loOpen && f < lo) || (hi != nil && f > *hi)
	if bad {
		rng := "at least " + strconv.FormatFloat(lo, 'g', -1, 64)
		if loOpen {
			rng = "more than " + strconv.FormatFloat(lo, 'g', -1, 64)
		}
		if hi != nil {
			rng += " and at most " + strconv.FormatFloat(*hi, 'g', -1, 64)
		}
		w := ""
		if whole {
			w = "whole "
		}
		return 0, fmt.Errorf("%s: expected a %snumber, %s, not %v", key, w, rng, v)
	}
	return f, nil
}

func num(v float64, whole bool) interface{} {
	if whole {
		return int(v)
	}
	return v
}

// check returns the value to store for key, or an error naming what is expected. nil removes the key.
func check(key string, v interface{}, cfg map[string]interface{}) (interface{}, error) {
	e, ok := spec(key)
	if !ok {
		return nil, fmt.Errorf("%q cannot be changed here (only the keys the Settings view lists)", key)
	}
	if v == nil {
		return nil, nil
	}
	one := 1.0
	sixtyFour := 64.0
	switch e.kind {
	case "bool":
		b, ok := v.(bool)
		if !ok {
			return nil, fmt.Errorf("%s: expected true or false, not %v", key, v)
		}
		if key == "lazy_load" && b && cfg["vision"] != nil && cfg["vision"] != false {
			return nil, fmt.Errorf("lazy_load: lazy loading is text-only, and this model reads images (\"vision\")")
		}
		return b, nil
	case "enum":
		s, _ := v.(string)
		for _, c := range e.choices {
			if s == c {
				return s, nil
			}
		}
		return nil, fmt.Errorf("%s: expected one of %s, not %v", key, strings.Join(e.choices, ", "), v)
	case "names":
		var names []interface{}
		switch x := v.(type) {
		case string:
			for _, p := range strings.Split(x, ",") {
				if p = strings.TrimSpace(p); p != "" {
					names = append(names, p)
				}
			}
		case []interface{}:
			for _, p := range x {
				s, ok := p.(string)
				if !ok || strings.TrimSpace(s) == "" || len(s) > 128 {
					return nil, fmt.Errorf("%s: expected a list of model names, not %v", key, v)
				}
				names = append(names, s)
			}
		default:
			return nil, fmt.Errorf("%s: expected a list of model names, not %v", key, v)
		}
		if len(names) == 0 {
			return nil, nil
		}
		return names, nil
	case "int>=0", "arg":
		f, err := number(key, v, true, 0, nil, false)
		return num(f, true), err
	case "num>=0":
		f, err := number(key, v, false, 0, nil, false)
		return num(f, false), err
	}
	switch e.rule { // a sampling key
	case "num>=0":
		return number(key, v, false, 0, nil, false)
	case "0<x<=1":
		return number(key, v, false, 0, &one, true)
	case "0<=x<=1":
		return number(key, v, false, 0, &one, false)
	}
	f, err := number(key, v, true, 1, &sixtyFour, false)
	return num(f, true), err
}

func deepCopy(m map[string]interface{}) map[string]interface{} {
	b, _ := json.Marshal(m)
	out := map[string]interface{}{}
	_ = json.Unmarshal(b, &out)
	return out
}

func equalValue(a, b interface{}) bool {
	x, _ := json.Marshal(a)
	y, _ := json.Marshal(b)
	return bytes.Equal(x, y)
}

// Apply returns the config with changes ({key: value or nil}) made, and the keys that changed. Every value is checked
// first, so a request with one bad value changes nothing.
func Apply(cfg map[string]interface{}, changes map[string]interface{}) (map[string]interface{}, []string, error) {
	if len(changes) == 0 {
		return nil, nil, fmt.Errorf(`send {"set": {"<key>": <value or null>, ...}}`)
	}
	checked := map[string]interface{}{}
	for k, v := range changes {
		c, err := check(k, v, cfg)
		if err != nil {
			return nil, nil, err
		}
		checked[k] = c
	}
	nw := deepCopy(cfg)
	var changed []string
	for _, e := range editable { // the list's order, so the result does not depend on map order
		v, ok := checked[e.key]
		if !ok || equalValue(valueOf(nw, e), v) {
			continue
		}
		switch e.kind {
		case "sampling":
			s, _ := nw["sampling"].(map[string]interface{})
			if s == nil {
				s = map[string]interface{}{}
			}
			sub := strings.TrimPrefix(e.key, "sampling.")
			if v == nil {
				delete(s, sub)
			} else {
				s[sub] = v
			}
			if len(s) > 0 {
				nw["sampling"] = s
			} else {
				delete(nw, "sampling")
			}
		case "arg":
			args, _ := nw["args"].([]interface{})
			at := -1
			for i := 0; i+1 < len(args); i++ {
				if s, _ := args[i].(string); s == e.flag {
					at = i
				}
			}
			switch {
			case at >= 0 && v == nil:
				args = append(args[:at], args[at+2:]...)
			case at >= 0:
				args[at+1] = fmt.Sprint(v)
			case v != nil:
				args = append(args, e.flag, fmt.Sprint(v))
			}
			nw["args"] = args
		default:
			if v == nil {
				delete(nw, e.key)
			} else {
				nw[e.key] = v
			}
		}
		changed = append(changed, e.key)
	}
	return nw, changed, nil
}

// LoadRaw reads a run config as a generic JSON object.
func LoadRaw(path string) (map[string]interface{}, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	raw = bytes.TrimPrefix(raw, []byte("\xef\xbb\xbf"))
	var cfg map[string]interface{}
	if err := json.Unmarshal(raw, &cfg); err != nil || cfg == nil {
		return nil, fmt.Errorf("%s is not a JSON object", filepath.Base(path))
	}
	return cfg, nil
}

// SaveRaw writes the config whole (a temporary file moved over the old one) and keeps the earlier one as <name>.bak.
func SaveRaw(path string, cfg map[string]interface{}) (string, error) {
	bak := path + ".bak"
	old, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	// the file may hold an api_key: the backup and the new file keep its mode, never wider than owner-only read/write
	mode := os.FileMode(0o600)
	if st, err := os.Stat(path); err == nil {
		mode = st.Mode().Perm() & 0o600
	}
	if err := os.WriteFile(bak, old, mode); err != nil {
		return "", err
	}
	data, err := json.MarshalIndent(cfg, "", " ")
	if err != nil {
		return "", err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, mode); err != nil {
		return "", err
	}
	if err := os.Chmod(bak, mode); err != nil { // WriteFile keeps the mode of a backup left by an earlier save
		return "", err
	}
	return bak, os.Rename(tmp, path)
}
