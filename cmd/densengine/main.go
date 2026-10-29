// densengine is the dense Qwen3.8-27B (pkg/densecpu, CUDA build) as a Strata engine: it speaks the same stdin/stdout
// line protocol as the C++ engine (see pkg/engineipc), so `strata serve` runs it unchanged, with its tokenizer, chat
// template, tool calls, streaming, web app and API key.
//
//	--serve --native <model.gguf> [--max-context N]
//
// stdin: GEN <max_new> [key=value ...] <id,id,...> | STOP | QUIT.  stdout: INFO .., READY <ctx> stop, then PP .., T <id>,
// DONE <gen> <prompt> <prefill_ms> <total_ms> <reason> or ERR <msg>. One session is kept: a prompt that extends the last
// one token for token only feeds the new tokens (the delta-net state cannot be rolled back, so anything else starts over).
package main

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"strata/pkg/densecpu"
	"strata/pkg/gguf"
)

type sampling struct {
	temp, topP, minP, rep, freq, pres float64
	topK                              int
	seed                              int64
}

func parseGen(line string) (maxNew int, sp sampling, ids []int, err error) {
	f := strings.Fields(line)
	if len(f) < 3 {
		return 0, sp, nil, fmt.Errorf("bad GEN line")
	}
	if maxNew, err = strconv.Atoi(f[1]); err != nil {
		return
	}
	sp.topP, sp.rep = 1, 1
	for _, kv := range f[2 : len(f)-1] {
		p := strings.SplitN(kv, "=", 2)
		if len(p) != 2 {
			continue
		}
		v, _ := strconv.ParseFloat(p[1], 64)
		switch p[0] {
		case "temperature":
			sp.temp = v
		case "top_p":
			sp.topP = v
		case "top_k":
			sp.topK = int(v)
		case "min_p":
			sp.minP = v
		case "penalty_repeat":
			sp.rep = v
		case "penalty_freq":
			sp.freq = v
		case "penalty_present":
			sp.pres = v
		case "seed":
			sp.seed = int64(v)
		}
	}
	for _, s := range strings.Split(f[len(f)-1], ",") {
		id, e := strconv.Atoi(s)
		if e != nil {
			return 0, sp, nil, fmt.Errorf("bad token id %q", s)
		}
		ids = append(ids, id)
	}
	return
}

// pick applies the penalties to logits (in place) and returns the next token.
func pick(logits []float32, sp sampling, hist map[int]int, rng *rand.Rand) int {
	for id, n := range hist {
		l := float64(logits[id])
		if sp.rep != 1 && sp.rep > 0 {
			if l > 0 {
				l /= sp.rep
			} else {
				l *= sp.rep
			}
		}
		l -= sp.freq*float64(n) + sp.pres
		logits[id] = float32(l)
	}
	if sp.temp <= 0 {
		b := 0
		for i, v := range logits {
			if v > logits[b] {
				b = i
			}
		}
		return b
	}
	k := sp.topK
	if k < 1 || k > 64 {
		k = 64
	}
	idx := make([]int, len(logits))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return logits[idx[a]] > logits[idx[b]] })
	idx = idx[:k]
	mx := float64(logits[idx[0]])
	p := make([]float64, k)
	var sum float64
	for i, id := range idx {
		p[i] = math.Exp((float64(logits[id]) - mx) / sp.temp)
		sum += p[i]
	}
	cut, acc := k, 0.0
	for i := range p {
		if sp.minP > 0 && p[i]/p[0] < sp.minP {
			cut = i
			break
		}
		acc += p[i] / sum
		if sp.topP > 0 && sp.topP < 1 && acc >= sp.topP {
			cut = i + 1
			break
		}
	}
	var s2 float64
	for _, v := range p[:cut] {
		s2 += v
	}
	r := rng.Float64() * s2
	for i := 0; i < cut; i++ {
		if r -= p[i]; r <= 0 {
			return idx[i]
		}
	}
	return idx[cut-1]
}

func main() {
	var model string
	ctx := 32768
	args := os.Args[1:]
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--native":
			i++
			model = args[i]
		case "--max-context":
			i++
			ctx, _ = strconv.Atoi(args[i])
		}
	}
	if model == "" {
		execEngine("STRATA_MAIN_ENGINE", "engine/strata") // a pack launch
	}
	switch ggufArch(model) { // the dense Qwen and Gemma run here; everything else has an engine of its own
	case "qwen35", "gemma4":
	case "flint":
		execEngine("STRATA_FLINT_ENGINE", "build/flint_engine")
	default:
		execEngine("STRATA_MAIN_ENGINE", "engine/strata")
	}
	q, err := densecpu.OpenQwen(model)
	if err != nil {
		fmt.Fprintln(os.Stderr, "densengine:", err)
		os.Exit(1)
	}
	gpu, err := q.EnableGPU()
	if err != nil {
		fmt.Fprintln(os.Stderr, "densengine:", err)
		os.Exit(1)
	}
	out := bufio.NewWriter(os.Stdout)
	say := func(format string, a ...any) { fmt.Fprintf(out, format+"\n", a...); out.Flush() }
	say("INFO engine=densecpu model=%s context=%d", "qwen3.8-27b", ctx)
	fmt.Fprintf(os.Stderr, "densengine: %v\n", gpu)
	say("READY %d stop", ctx)

	var stop atomic.Bool
	cmds := make(chan string, 4)
	go func() {
		sc := bufio.NewScanner(os.Stdin)
		sc.Buffer(make([]byte, 0, 1<<20), 64<<20) // a 200k-token prompt is ~1.3 MB of ids
		for sc.Scan() {
			switch l := sc.Text(); {
			case l == "STOP":
				stop.Store(true)
			case l == "QUIT":
				os.Exit(0)
			default:
				cmds <- l
			}
		}
		os.Exit(0) // the server went away
	}()

	var st *densecpu.QwenState
	var seen []int
	for line := range cmds {
		stop.Store(false)
		if !strings.HasPrefix(line, "GEN ") {
			continue
		}
		maxNew, sp, ids, err := parseGen(line)
		if err != nil || len(ids) == 0 || len(ids)+1 > ctx {
			say("ERR bad request (prompt %d tokens, context %d): %v", len(ids), ctx, err)
			continue
		}
		t0 := time.Now()
		common := 0
		for common < len(seen) && common < len(ids) && seen[common] == ids[common] {
			common++
		}
		if st == nil || common < len(seen) || common == len(ids) {
			if st != nil {
				st.Free()
			}
			st, seen, common = q.NewState(), nil, 0
		} else {
			say("REUSED %d", common)
		}
		var logits []float32
		for p := common; p < len(ids) && !stop.Load(); {
			n := min(512, len(ids)-p)
			logits = q.Step(st, ids[p:p+n])
			seen = append(seen, ids[p:p+n]...)
			p += n
			el := time.Since(t0).Seconds()
			say("PP %d %d %.3f %.1f", p, len(ids), float64(p)/float64(len(ids)), float64(p-common)/el)
		}
		prefill := time.Since(t0)
		seed := sp.seed
		if seed == 0 {
			seed = time.Now().UnixNano()
		}
		rng := rand.New(rand.NewSource(seed))
		hist := map[int]int{}
		n, reason := 0, "length"
		for ; n < maxNew; n++ {
			if stop.Load() {
				reason = "stop"
				break
			}
			if len(seen)+1 >= ctx {
				break
			}
			id := pick(logits, sp, hist, rng)
			hist[id]++
			say("T %d", id)
			logits = q.Step(st, []int{id})
			seen = append(seen, id)
		}
		say("DONE %d %d %d %d %s", n, len(ids), prefill.Milliseconds(), time.Since(t0).Milliseconds()-prefill.Milliseconds(), reason)
	}
}

func ggufArch(path string) string {
	g, err := gguf.Open(path)
	if err != nil {
		return ""
	}
	defer g.Close()
	a, _ := g.Metadata["general.architecture"].(string)
	return a
}

// execEngine replaces this process with another engine (same arguments, same line protocol) found at $env or one of
// cands (relative paths: working directory first, then next to this binary's parent), so the server launches every model
// through bin/densengine.
func execEngine(env string, cands ...string) {
	self, _ := os.Executable()
	list := []string{os.Getenv(env)}
	for _, c := range cands {
		list = append(list, c, filepath.Join(filepath.Dir(self), "..", c))
	}
	for _, c := range list {
		if c == "" {
			continue
		}
		if p, err := filepath.Abs(c); err == nil {
			if st, err := os.Stat(p); err == nil && !st.IsDir() {
				fmt.Fprintln(os.Stderr, "densengine: running", p)
				err = syscall.Exec(p, append([]string{p}, os.Args[1:]...), os.Environ())
				fmt.Fprintln(os.Stderr, "densengine:", err)
				os.Exit(1)
			}
		}
	}
	fmt.Fprintf(os.Stderr, "densengine: engine not found (set %s or build it): %v\n", env, cands)
	os.Exit(1)
}
