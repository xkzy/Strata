package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"strata/pkg/tune"
)

// runTune measures the run config on this PC's devices - each NVIDIA card alone and all of them in a layer split -
// with the real server and the real model, and writes the fastest placement into the run config.
func runTune(args []string) {
	fs := flag.NewFlagSet("tune", flag.ExitOnError)
	configFile := fs.String("config", "", "the strata-<model>.json to measure and update (default: the only one here)")
	decode := fs.Int("tokens", 256, "tokens to generate in the measured request")
	words := fs.Int("prompt-words", 1500, "length of the measured prompt in words (about 2K tokens)")
	only := fs.String("only", "", "comma-separated candidates to measure (gpu0, gpu1, split, split-probe, peer)")
	noResident := fs.Bool("no-resident", false, "skip measuring the winner with --resident-experts (a locked RAM copy of the experts)")
	dry := fs.Bool("dry-run", false, "measure and print, but leave the run config as it is")
	loadWait := fs.Duration("load-timeout", 25*time.Minute, "how long one candidate may take to load the model")
	fs.Parse(args)

	path := *configFile
	if path == "" {
		matches, _ := filepath.Glob("strata-*.json")
		sort.Strings(matches)
		if len(matches) != 1 {
			fmt.Fprintf(os.Stderr, "strata tune: pass --config strata-<model>.json (found %d in %s)\n", len(matches), mustCwd())
			os.Exit(2)
		}
		path = matches[0]
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, "strata tune:", err)
		os.Exit(2)
	}
	var base map[string]interface{}
	if err := json.Unmarshal(raw, &base); err != nil {
		fmt.Fprintf(os.Stderr, "strata tune: %s: %v\n", path, err)
		os.Exit(2)
	}
	if busy := engineRunning(); busy != "" {
		fmt.Fprintf(os.Stderr, "strata tune: an engine is running (%s); stop it first, a benchmark needs the cards to itself\n", busy)
		os.Exit(2)
	}
	cards := nvidiaCards()
	if len(cards) == 0 {
		fmt.Fprintln(os.Stderr, "strata tune: no NVIDIA card found (nvidia-smi lists none); there is nothing to choose between")
		os.Exit(2)
	}
	fmt.Println("Devices:")
	for _, c := range cards {
		fmt.Println("  " + c)
	}
	cands := tune.Candidates(base, len(cards))
	if *only != "" {
		want := map[string]bool{}
		for _, n := range strings.Split(*only, ",") {
			want[strings.TrimSpace(n)] = true
		}
		kept := cands[:0]
		for _, c := range cands {
			if want[c.Name] {
				kept = append(kept, c)
			}
		}
		cands = kept
	}
	exe, _ := os.Executable()
	var results []tune.Result
	run := func(c tune.Candidate) {
		fmt.Printf("\n== %s ==\n", c.Name)
		r := measure(exe, c, *decode, *words, *loadWait)
		// A card with no room left for the prompt buffers: keep more VRAM free of the expert cache and measure again.
		for tries := 0; tries < 2 && r.Err != "" && strings.Contains(r.Err, "free VRAM"); tries++ {
			more, ok := tune.ReserveMore(c)
			if !ok {
				break
			}
			fmt.Printf("   not enough VRAM for the prompt buffers; again with --vram-reserve-mib %s\n", reserveOf(more))
			c = more
			cands[indexOf(cands, c.Name)] = c
			r = measure(exe, c, *decode, *words, *loadWait)
		}
		r.Name = c.Name
		if r.Err != "" {
			fmt.Printf("   failed: %s\n", r.Err)
		} else {
			fmt.Printf("   decode %.1f tok/s (%d tokens), prompt %.1f tok/s (%d tokens)\n", r.DecodeTokS, r.DecodeTokens, r.PrefillTokS, r.PrefillTokens)
		}
		results = append(results, r)
	}
	for _, c := range cands {
		run(c)
	}
	// Second stage: the model may fit in RAM, and then a locked RAM copy of the experts (--resident-experts) does not
	// depend on the OS file cache the way --mmap-experts does. Measure the winning placement that way too.
	if best := tune.Choose(results); best >= 0 && !*noResident {
		if v, ok := tune.ResidentVariant(cands[indexOf(cands, results[best].Name)]); ok {
			cands = append(cands, v)
			run(v)
		}
	}
	best := tune.Choose(results)
	if best < 0 {
		fmt.Fprintln(os.Stderr, "\nstrata tune: no candidate ran; the run config is unchanged")
		os.Exit(1)
	}
	fmt.Printf("\nFastest: %s (%.1f tok/s decode)\n", results[best].Name, results[best].DecodeTokS)
	if *dry {
		fmt.Println("--dry-run: the run config is unchanged")
		return
	}
	rec := tune.Record{Hardware: cards, Chosen: results[best].Name, Results: results}
	if err := tune.Apply(path, cands[indexOf(cands, results[best].Name)], rec); err != nil {
		fmt.Fprintln(os.Stderr, "strata tune:", err)
		os.Exit(1)
	}
	fmt.Printf("Wrote the placement to %s (the old file is %s.bak). It is used from the next start of the model.\n", path, path)
}

// reserveOf is the --vram-reserve-mib a candidate asks for.
func reserveOf(c tune.Candidate) string {
	args, _ := c.Config["args"].([]interface{})
	for i := 0; i+1 < len(args); i++ {
		if a, _ := args[i].(string); a == "--vram-reserve-mib" {
			v, _ := args[i+1].(string)
			return v
		}
	}
	return "?"
}

func indexOf(cs []tune.Candidate, name string) int {
	for i, c := range cs {
		if c.Name == name {
			return i
		}
	}
	return 0
}

func mustCwd() string {
	d, _ := os.Getwd()
	return d
}

// nvidiaCards lists the cards as nvidia-smi numbers them (PCI bus order, the numbering CUDA_DEVICE_ORDER=PCI_BUS_ID uses).
func nvidiaCards() []string {
	out, err := exec.Command("nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader").Output()
	if err != nil {
		return nil
	}
	var cards []string
	for _, l := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if p := strings.SplitN(l, ",", 3); len(p) == 3 {
			cards = append(cards, fmt.Sprintf("GPU %s: %s, %s", strings.TrimSpace(p[0]), strings.TrimSpace(p[1]), strings.TrimSpace(p[2])))
		}
	}
	return cards
}

func engineRunning() string {
	out, err := exec.Command("pgrep", "-af", "strata").Output()
	if err != nil {
		return ""
	}
	return findEngine(string(out), os.Getpid())
}

// findEngine picks the first `pid command line` row of pgrep -af output that is a Strata engine or server: its
// executable (not just some word in its arguments, like a shell that quotes the command) is strata / strata-server /
// strata_rt_server, started to serve. self is skipped.
func findEngine(pgrep string, self int) string {
	for _, line := range strings.Split(pgrep, "\n") {
		f := strings.Fields(line)
		if len(f) < 2 || f[0] == fmt.Sprint(self) {
			continue
		}
		switch filepath.Base(f[1]) {
		case "strata_rt_server", "strata-server":
			return strings.TrimSpace(line)
		case "strata":
			for _, a := range f[2:] {
				if a == "--serve" {
					return strings.TrimSpace(line)
				}
			}
			if len(f) > 2 && f[2] == "serve" {
				return strings.TrimSpace(line)
			}
		}
	}
	return ""
}

type lockedBuf struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (l *lockedBuf) Write(p []byte) (int, error) { l.mu.Lock(); defer l.mu.Unlock(); return l.b.Write(p) }
func (l *lockedBuf) String() string              { l.mu.Lock(); defer l.mu.Unlock(); return l.b.String() }

// measure starts the server on this candidate, warms it with one request (--mmap-experts depends on the page cache, so
// a cold first request says little), times a second, different request and stops the server and its engine.
func measure(exe string, c tune.Candidate, decode, words int, loadWait time.Duration) (res tune.Result) {
	dir, err := os.MkdirTemp("", "strata-tune-")
	if err != nil {
		return tune.Result{Err: err.Error()}
	}
	defer os.RemoveAll(dir)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return tune.Result{Err: err.Error()}
	}
	port := ln.Addr().(*net.TCPAddr).Port
	ln.Close()
	cfg := c.Config
	cfg["port"] = port
	cfg["host"] = "127.0.0.1"
	cfg["log"] = filepath.Join(dir, "engine.log")
	b, _ := json.Marshal(cfg)
	cfgPath := filepath.Join(dir, "strata-tune.json")
	if err := os.WriteFile(cfgPath, b, 0o600); err != nil {
		return tune.Result{Err: err.Error()}
	}

	out := &lockedBuf{}
	cmd := exec.Command(exe, "serve", "--config", cfgPath, "--port", fmt.Sprint(port), "--rt", "off")
	cmd.Stdout, cmd.Stderr = io.Discard, out
	setProcessGroup(cmd)
	if err := cmd.Start(); err != nil {
		return tune.Result{Err: err.Error()}
	}
	exited := make(chan struct{})
	go func() { cmd.Wait(); close(exited) }()
	defer func() {
		stopProcessGroup(cmd, exited)
	}()

	base := fmt.Sprintf("http://127.0.0.1:%d", port)
	deadline := time.Now().Add(loadWait)
	for {
		select {
		case <-exited:
			return tune.Result{Err: "the server exited while loading: " + lastLines(out.String(), 3)}
		default:
		}
		if time.Now().After(deadline) {
			return tune.Result{Err: "the model did not load in " + loadWait.String()}
		}
		ok, failed := loaded(base)
		if failed != "" {
			return tune.Result{Err: "the engine failed to start: " + failed}
		}
		if ok {
			break
		}
		time.Sleep(2 * time.Second)
	}
	if _, err := ask(base, prompt("warm-up", 400), 128); err != nil {
		return tune.Result{Err: "warm-up request: " + err.Error()}
	}
	before := len(out.String())
	if _, err := ask(base, prompt("measured", words), decode); err != nil {
		return tune.Result{Err: "measured request: " + err.Error()}
	}
	time.Sleep(500 * time.Millisecond) // the engine prints its line just after the reply
	r, ok := tune.ParseServed(out.String()[before:])
	if !ok {
		return tune.Result{Err: "the engine printed no speed line for the request: " + lastLines(out.String(), 3)}
	}
	if r.PrefillTokens < words/2 {
		return tune.Result{Err: fmt.Sprintf("only %d prompt tokens were read (the rest came from the prompt cache), so the prompt speed is not measured", r.PrefillTokens)}
	}
	return r
}

// loaded reports whether the model is loaded; failed is the engine's reason (its last log lines) when it gave up.
func loaded(base string) (ok bool, failed string) {
	resp, err := http.Get(base + "/health")
	if err != nil {
		return false, ""
	}
	defer resp.Body.Close()
	var h struct {
		Loaded bool   `json:"loaded"`
		Engine string `json:"engine"`
		Error  string `json:"error"`
	}
	json.NewDecoder(resp.Body).Decode(&h)
	if h.Engine == "failed" {
		return false, lastSentence(h.Error)
	}
	return h.Loaded, ""
}

// lastSentence keeps the end of the engine's message, which is where it says what went wrong.
func lastSentence(s string) string {
	parts := strings.Split(s, " | ")
	if len(parts) > 2 {
		parts = parts[len(parts)-2:]
	}
	return strings.Join(parts, " | ")
}

// prompt is text that differs per tag, so the measured request never matches the warm-up's cached prompt.
func prompt(tag string, words int) string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "Notes (%s). ", tag)
	for i := 0; sb.Len() < words*6; i++ {
		fmt.Fprintf(&sb, "Item %d of the %s list: the %s valve on line %d reads %d units. ", i, tag, []string{"red", "blue", "green", "grey"}[i%4], i%17, (i*37)%1000)
	}
	sb.WriteString("\n\nWrite a long, detailed essay about these notes, and do not stop early.")
	return sb.String()
}

func ask(base, content string, maxTokens int) (string, error) {
	body, _ := json.Marshal(map[string]interface{}{
		"messages":    []map[string]string{{"role": "user", "content": content}},
		"max_tokens":  maxTokens,
		"temperature": 0,
	})
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "POST", base+"/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return "", fmt.Errorf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(data)))
	}
	return string(data), nil
}

func lastLines(s string, n int) string {
	l := strings.Split(strings.TrimSpace(s), "\n")
	if len(l) > n {
		l = l[len(l)-n:]
	}
	return strings.Join(l, " | ")
}
