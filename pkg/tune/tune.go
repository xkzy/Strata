// Package tune picks the device placement (which cards, a layer split or not, the PCIe share) from measurements: it
// builds the candidate run configs, reads the engine's own speed line from a run, chooses the fastest and writes it
// back into the run config. Starting the server and timing a request is the caller's job (cmd/strata/tune.go), so
// everything here runs without a GPU.
package tune

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// Candidate is one placement to measure.
type Candidate struct {
	Name   string                 `json:"name"`
	Config map[string]interface{} `json:"-"` // the full run config with this placement
}

// Result is what one candidate measured. Err is set when it could not run (its other fields are then zero).
type Result struct {
	Name          string  `json:"name"`
	DecodeTokS    float64 `json:"decode_tok_s"`
	PrefillTokS   float64 `json:"prefill_tok_s"`
	DecodeTokens  int     `json:"decode_tokens"`
	PrefillTokens int     `json:"prefill_tokens"`
	Err           string  `json:"error,omitempty"`
}

// Record is what is saved in the run config under "tuned": the numbers behind the choice and what they were measured on.
type Record struct {
	At       string   `json:"at"`
	Hardware []string `json:"hardware"`
	Chosen   string   `json:"chosen"`
	Results  []Result `json:"results"`
}

// servedLine is the engine's per-request line in serve mode (src/program/generate.cpp), e.g.
// "strata serve: prompt 2113 tokens = 0 reused + 2113 read in 1204 ms (1755.1 tok/s), 256 generated in 6100 ms (42.0 tok/s), ..."
var servedLine = regexp.MustCompile(`strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in [\d.]+ ms \(([\d.]+) tok/s\), (\d+) generated in [\d.]+ ms \(([\d.]+) tok/s\)`)

// ParseServed reads the last request line in the engine's output. ok is false when there is none.
func ParseServed(output string) (r Result, ok bool) {
	for _, line := range strings.Split(output, "\n") {
		m := servedLine.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		read, _ := strconv.Atoi(m[3])
		gen, _ := strconv.Atoi(m[5])
		pf, _ := strconv.ParseFloat(m[4], 64)
		dc, _ := strconv.ParseFloat(m[6], 64)
		r, ok = Result{DecodeTokS: dc, PrefillTokS: pf, DecodeTokens: gen, PrefillTokens: read}, true
	}
	return r, ok
}

func copyMap(m map[string]interface{}) map[string]interface{} {
	b, _ := json.Marshal(m)
	out := map[string]interface{}{}
	_ = json.Unmarshal(b, &out)
	return out
}

func argsOf(cfg map[string]interface{}) []string {
	var out []string
	l, _ := cfg["args"].([]interface{})
	for _, a := range l {
		s, _ := a.(string)
		out = append(out, s)
	}
	return out
}

// SetArg returns args with flag set to value (value nil: the flag and its value removed).
func SetArg(args []string, flag string, value *string) []string {
	var out []string
	done := false
	for i := 0; i < len(args); i++ {
		if args[i] != flag {
			out = append(out, args[i])
			continue
		}
		i++ // its value
		if value != nil && !done {
			out = append(out, flag, *value)
			done = true
		}
	}
	if value != nil && !done {
		out = append(out, flag, *value)
	}
	return out
}

func withArgs(cfg map[string]interface{}, args []string) {
	l := make([]interface{}, len(args))
	for i, a := range args {
		l[i] = a
	}
	cfg["args"] = l
}

func withEnv(cfg map[string]interface{}, k, v string) {
	env, _ := cfg["env"].(map[string]interface{})
	if env == nil {
		env = map[string]interface{}{}
	}
	env[k] = v
	cfg["env"] = env
}

// Candidates builds the placements to measure for `cards` NVIDIA GPUs (numbered as nvidia-smi does) from the user's own
// run config, so every other setting (control vector, speculation, context) is what they run. Each card alone, then all
// cards in a layer split, then the split with --pcie-frac left to the engine's per-card link probe, then the first card
// with the second as an expert-cache peer (--peer-device).
func Candidates(base map[string]interface{}, cards int) []Candidate {
	if cards < 1 {
		return nil
	}
	var out []Candidate
	mk := func(name string, edit func(cfg map[string]interface{}, args []string) []string) {
		cfg := copyMap(base)
		args := SetArg(SetArg(argsOf(cfg), "--layer-split", nil), "--peer-device", nil)
		args = edit(cfg, args)
		withArgs(cfg, args)
		withEnv(cfg, "CUDA_DEVICE_ORDER", "PCI_BUS_ID") // so the numbers below are nvidia-smi's, not CUDA's fastest-first
		out = append(out, Candidate{Name: name, Config: cfg})
	}
	for i := 0; i < cards; i++ {
		i := i
		mk(fmt.Sprintf("gpu%d", i), func(cfg map[string]interface{}, args []string) []string {
			withEnv(cfg, "CUDA_VISIBLE_DEVICES", strconv.Itoa(i))
			return args
		})
	}
	if cards > 1 {
		ids := make([]string, cards)
		for i := range ids {
			ids[i] = strconv.Itoa(i)
		}
		list := strings.Join(ids, ",")
		auto := "auto"
		mk("split", func(cfg map[string]interface{}, args []string) []string {
			withEnv(cfg, "CUDA_VISIBLE_DEVICES", list)
			return SetArg(args, "--layer-split", &auto)
		})
		mk("split-probe", func(cfg map[string]interface{}, args []string) []string {
			withEnv(cfg, "CUDA_VISIBLE_DEVICES", list)
			return SetArg(SetArg(args, "--layer-split", &auto), "--pcie-frac", nil)
		})
		// The first card runs the model; the second holds more of its experts and computes their rows (no P2P needed).
		peer := "1"
		mk("peer", func(cfg map[string]interface{}, args []string) []string {
			withEnv(cfg, "CUDA_VISIBLE_DEVICES", list)
			return SetArg(args, "--peer-device", &peer)
		})
	}
	return out
}

// Choose returns the index of the fastest candidate by decode speed. Results within 3% of the best are a tie, and the
// tie goes to the faster prompt reading. -1 when none ran.
func Choose(rs []Result) int {
	best := -1
	for i, r := range rs {
		if r.Err == "" && r.DecodeTokS > 0 && (best < 0 || r.DecodeTokS > rs[best].DecodeTokS) {
			best = i
		}
	}
	if best < 0 {
		return -1
	}
	top := rs[best]
	for i, r := range rs {
		if r.Err == "" && r.DecodeTokS >= top.DecodeTokS*0.97 && r.PrefillTokS > rs[best].PrefillTokS {
			best = i
		}
	}
	return best
}

// ResidentVariant is c with the experts held in a locked RAM copy (--resident-experts) instead of mapped from the pack
// (--mmap-experts, which depends on the OS file cache holding the model). Worth measuring when the model fits in RAM.
// ok is false when c already runs resident or uses neither mode.
func ResidentVariant(c Candidate) (v Candidate, ok bool) {
	args := argsOf(c.Config)
	if !hasFlag(args, "--mmap-experts") || hasFlag(args, "--resident-experts") {
		return Candidate{}, false
	}
	var out []string
	for _, a := range args {
		if a == "--mmap-experts" {
			a = "--resident-experts"
		}
		out = append(out, a)
	}
	cfg := copyMap(c.Config)
	withArgs(cfg, out)
	return Candidate{Name: c.Name + "+resident", Config: cfg}, true
}

// Defaults of the engine's --vram-reserve-mib, the step ReserveMore adds and the most it asks for.
const (
	defaultReserveMiB = 700
	reserveStepMiB    = 1500
	maxReserveMiB     = 6000
)

// ReserveMore is c with a bigger --vram-reserve-mib (the VRAM each card keeps free of the expert cache), for a card that
// ran out of room for the prompt buffers. ok is false when c already asks for the most.
func ReserveMore(c Candidate) (v Candidate, ok bool) {
	args := argsOf(c.Config)
	cur := defaultReserveMiB
	if s := lookup(args, "--vram-reserve-mib"); s != "" {
		if n, err := strconv.Atoi(s); err == nil {
			cur = n
		}
	}
	if cur >= maxReserveMiB {
		return Candidate{}, false
	}
	next := strconv.Itoa(min(cur+reserveStepMiB, maxReserveMiB))
	cfg := copyMap(c.Config)
	withArgs(cfg, SetArg(args, "--vram-reserve-mib", &next))
	return Candidate{Name: c.Name, Config: cfg}, true
}

func hasFlag(args []string, flag string) bool {
	for _, a := range args {
		if a == flag {
			return true
		}
	}
	return false
}

func lookup(args []string, flag string) string {
	for i := 0; i+1 < len(args); i++ {
		if args[i] == flag {
			return args[i+1]
		}
	}
	return ""
}

// Apply writes the chosen candidate's placement into the run config file (the file as it was goes to path+".bak") and
// records the measurements next to it. Only the placement changes: --layer-split, --pcie-frac and the two CUDA
// variables; the rest of the file is left as the user has it.
func Apply(path string, chosen Candidate, rec Record) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var cur map[string]interface{}
	if err := json.Unmarshal(raw, &cur); err != nil {
		return err
	}
	if err := os.WriteFile(path+".bak", raw, 0o644); err != nil {
		return err
	}
	args := argsOf(cur)
	for _, flag := range []string{"--layer-split", "--peer-device", "--pcie-frac"} {
		var v *string
		if x := lookup(argsOf(chosen.Config), flag); x != "" {
			v = &x
		}
		args = SetArg(args, flag, v)
	}
	// the memory mode: only when the chosen run used one of the two, so a config with neither keeps neither
	if cargs := argsOf(chosen.Config); hasFlag(cargs, "--mmap-experts") || hasFlag(cargs, "--resident-experts") {
		var kept []string
		for _, a := range args {
			if a != "--mmap-experts" && a != "--resident-experts" {
				kept = append(kept, a)
			}
		}
		for _, f := range []string{"--mmap-experts", "--resident-experts"} {
			if hasFlag(cargs, f) {
				kept = append(kept, f)
			}
		}
		args = kept
	}
	withArgs(cur, args)
	env, _ := cur["env"].(map[string]interface{})
	if env == nil {
		env = map[string]interface{}{}
	}
	cenv, _ := chosen.Config["env"].(map[string]interface{})
	for _, k := range []string{"CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES"} {
		if v, ok := cenv[k]; ok {
			env[k] = v
		}
	}
	cur["env"] = env
	if rec.At == "" {
		rec.At = time.Now().Format(time.RFC3339)
	}
	cur["tuned"] = rec
	out, err := json.MarshalIndent(cur, "", " ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(path, out, 0o644); err != nil {
		return err
	}
	return syncRuntime(filepath.Join(filepath.Dir(path), "strata-config.json"), args, env)
}

// syncRuntime makes the settings in strata-config.json (the runtime settings the server and its web page show) say what
// the tuned run config runs: the cards in use, --pcie-frac and --pool-workers. A missing file is left missing; a key
// the run config has no flag for is removed, as the settings page does for an empty field. Other keys stay.
func syncRuntime(path string, args []string, env map[string]interface{}) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var rt map[string]interface{}
	if json.Unmarshal(raw, &rt) != nil {
		return nil
	}
	if err := os.WriteFile(path+".bak", raw, 0o644); err != nil {
		return err
	}
	cards, _ := env["CUDA_VISIBLE_DEVICES"].(string)
	switch {
	case hasFlag(args, "--peer-device"):
		rt["gpus"] = "peer"
	case hasFlag(args, "--layer-split"):
		rt["gpus"] = "split"
	case cards != "":
		rt["gpus"] = cards
	}
	if v, err := strconv.ParseFloat(lookup(args, "--pcie-frac"), 64); err == nil {
		rt["pcie_frac"] = v
	} else {
		delete(rt, "pcie_frac")
	}
	if v, err := strconv.Atoi(lookup(args, "--pool-workers")); err == nil {
		rt["pool_workers"] = v
	} else {
		delete(rt, "pool_workers")
	}
	out, err := json.MarshalIndent(rt, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, out, 0o644)
}
