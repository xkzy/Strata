// pkg/engineipc/orchestrator.go - the C++ engine process and its line protocol (`strata --serve`)
//
// The engine loads the model once and then serves one request at a time over stdin/stdout:
//
//	INFO k=v ...            (startup, any number)
//	READY <max_context> [stop]
//	GEN <max_new> [key=value ...] <id,id,...>   ->   T <id> ... (PP/RESUME/REUSED progress lines) ... DONE <n> <prompt_tokens> <prompt_ms> <decode_ms> <stop|length|cancel> ...
//	                                                 or ERR <message>
//	STOP (end the running request at its next step) / QUIT
//
// Nothing here pretends: when the engine is not running, requests fail with the reason (a missing model, a crashed
// process, the last lines of its stderr) instead of being answered with invented text.
package engineipc

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"strata/pkg/tokenizer"
)

// ByteTokenizer is a stand-in tokenizer for tests without a pack: one id per UTF-8 byte, specials as ids >= 256.
type ByteTokenizer struct {
	Specials []string
}

func NewByteTokenizer() *ByteTokenizer {
	return &ByteTokenizer{Specials: []string{"<|im_start|>", "<|im_end|>", "<|endoftext|>"}}
}

func (bt *ByteTokenizer) Encode(text string, parseSpecial bool) []int {
	var out []int
	for i := 0; i < len(text); {
		matched := false
		if parseSpecial {
			for si, sp := range bt.Specials {
				if strings.HasPrefix(text[i:], sp) {
					out = append(out, 256+si)
					i += len(sp)
					matched = true
					break
				}
			}
		}
		if !matched {
			out = append(out, int(text[i]))
			i++
		}
	}
	return out
}

func (bt *ByteTokenizer) TokenBytes(id int) []byte {
	if id >= 256 && id-256 < len(bt.Specials) {
		return []byte(bt.Specials[id-256])
	}
	if id >= 0 && id < 256 {
		return []byte{byte(id)}
	}
	return nil
}

func (bt *ByteTokenizer) Decode(ids []int) string {
	var b []byte
	for _, id := range ids {
		b = append(b, bt.TokenBytes(id)...)
	}
	return strings.ToValidUTF8(string(b), "�")
}

// TextCodec is what the orchestrator needs from a tokenizer.
type TextCodec interface {
	Encode(text string, parseSpecial bool) []int
	Decode(ids []int) string
	TokenBytes(id int) []byte
}

// StopMatcher holds back text that could still become a stop string.
type StopMatcher struct {
	stops []string
	held  string
	hit   string
}

func NewStopMatcher(stops []string) *StopMatcher {
	var clean []string
	for _, s := range stops {
		if s != "" {
			clean = append(clean, s)
		}
	}
	return &StopMatcher{stops: clean}
}

func (sm *StopMatcher) Push(text string) string {
	if sm.hit != "" {
		return ""
	}
	buf := sm.held + text
	firstIdx := -1
	var firstStop string
	for _, s := range sm.stops {
		idx := strings.Index(buf, s)
		if idx >= 0 && (firstIdx == -1 || idx < firstIdx) {
			firstIdx = idx
			firstStop = s
		}
	}
	if firstIdx >= 0 {
		sm.held = ""
		sm.hit = firstStop
		return buf[:firstIdx]
	}
	keep := 0
	for _, s := range sm.stops {
		maxLen := len(s) - 1
		if len(buf) < maxLen {
			maxLen = len(buf)
		}
		for k := maxLen; k > keep; k-- {
			if strings.HasSuffix(buf, s[:k]) {
				keep = k
				break
			}
		}
	}
	if keep > 0 {
		sm.held = buf[len(buf)-keep:]
		return buf[:len(buf)-keep]
	}
	sm.held = ""
	return buf
}

func (sm *StopMatcher) Flush() string {
	held := sm.held
	sm.held = ""
	return held
}

func (sm *StopMatcher) Hit() string { return sm.hit }

// EngineConfig says how to start the engine.
type EngineConfig struct {
	BinaryPath   string        `json:"binary_path"`
	Args         []string      `json:"args"` // everything after --serve (the pack, model shards, cache sizes ...)
	Cwd          string        `json:"cwd"`
	Env          []string      `json:"env"`        // extra KEY=VALUE entries
	ModelPath    string        `json:"model_path"` // legacy: used as --native when Args is empty
	ContextSize  int           `json:"context_size"`
	ExtraArgs    []string      `json:"extra_args"`
	StartTimeout time.Duration `json:"start_timeout"` // model loading can take minutes (default 30 min)
	// Compatibility fields from the earlier configuration; the engine takes its sizes from Args.
	Threads      int           `json:"threads"`
	GPULayers    int           `json:"gpu_layers"`
	BatchSize    int           `json:"batch_size"`
	PrefillChunk int           `json:"prefill_chunk"`
	RestartDelay time.Duration `json:"restart_delay"`
}

type SamplingParams struct {
	Temperature       float64 `json:"temperature"`
	TopP              float64 `json:"top_p"`
	TopK              int     `json:"top_k"`
	MinP              float64 `json:"min_p"`
	RepetitionPenalty float64 `json:"repetition_penalty"`
	FrequencyPenalty  float64 `json:"frequency_penalty"`
	PresencePenalty   float64 `json:"presence_penalty"`
	Seed              int64   `json:"seed"`
}

type TokenEvent struct {
	TokenID      int     `json:"token_id,omitempty"`
	Text         string  `json:"text"`
	IsEnd        bool    `json:"is_end"`
	FinishReason string  `json:"finish_reason,omitempty"`
	PromptTokens int     `json:"prompt_tokens,omitempty"`
	GenTokens    int     `json:"gen_tokens,omitempty"`
	TokPerSec    float64 `json:"tok_per_sec,omitempty"`
	Error        error   `json:"error,omitempty"`
}

// State of the engine process.
type State int

const (
	StateStopped State = iota
	StateStarting
	StateReady
	StateFailed
)

func (s State) String() string {
	switch s {
	case StateStarting:
		return "starting"
	case StateReady:
		return "ready"
	case StateFailed:
		return "failed"
	}
	return "stopped"
}

// ErrNotReady wraps the reason the engine cannot take requests.
type ErrNotReady struct {
	State  State
	Reason string
}

func (e *ErrNotReady) Error() string {
	switch e.State {
	case StateStarting:
		return "the engine is still loading the model" + suffix(e.Reason)
	case StateFailed:
		return "the engine is not running" + suffix(e.Reason)
	}
	return "the engine has not been started" + suffix(e.Reason)
}

func suffix(r string) string {
	if r == "" {
		return ""
	}
	return ": " + r
}

type EngineOrchestrator struct {
	mu         sync.RWMutex
	cfg        EngineConfig
	cmd        *exec.Cmd
	stdin      io.WriteCloser
	lines      chan string // engine output after READY; closed when the process ends
	state      State
	reason     string
	canStop    bool
	maxContext int
	info       map[string]string
	tokenizer  TextCodec
	cancelFunc context.CancelFunc
	logTail    []string
	readyCh    chan struct{}
	reqSem     chan struct{} // one request at a time: the engine has one session
}

func NewEngineOrchestrator(cfg EngineConfig) *EngineOrchestrator {
	if cfg.BinaryPath == "" {
		for _, c := range []string{"./engine/strata", "./build/strata"} {
			if _, err := os.Stat(c); err == nil {
				cfg.BinaryPath = c
				break
			}
		}
		if cfg.BinaryPath == "" {
			cfg.BinaryPath = "strata"
		}
	}
	if cfg.ContextSize <= 0 {
		cfg.ContextSize = 32768
	}
	if cfg.StartTimeout <= 0 {
		cfg.StartTimeout = 30 * time.Minute
	}
	return &EngineOrchestrator{
		cfg: cfg, info: map[string]string{}, tokenizer: NewByteTokenizer(), maxContext: cfg.ContextSize,
		readyCh: make(chan struct{}), reqSem: make(chan struct{}, 1),
	}
}

func (eo *EngineOrchestrator) SetTokenizer(t TextCodec) {
	eo.mu.Lock()
	eo.tokenizer = t
	eo.mu.Unlock()
}

func (eo *EngineOrchestrator) Tokenizer() TextCodec {
	eo.mu.RLock()
	defer eo.mu.RUnlock()
	return eo.tokenizer
}

func (eo *EngineOrchestrator) fail(reason string) {
	eo.mu.Lock()
	eo.state = StateFailed
	eo.reason = reason
	eo.mu.Unlock()
}

// Start launches the engine and returns immediately; the model loads in the background. Use WaitReady to block.
func (eo *EngineOrchestrator) Start(ctx context.Context) error {
	eo.mu.Lock()
	if eo.state == StateStarting || eo.state == StateReady {
		eo.mu.Unlock()
		return nil
	}
	if _, err := os.Stat(eo.cfg.BinaryPath); err != nil && !strings.ContainsRune(eo.cfg.BinaryPath, os.PathSeparator) {
		if p, lerr := exec.LookPath(eo.cfg.BinaryPath); lerr == nil {
			eo.cfg.BinaryPath = p
			err = nil
		}
	}
	if _, err := os.Stat(eo.cfg.BinaryPath); err != nil {
		eo.state, eo.reason = StateFailed, fmt.Sprintf("engine binary %q not found", eo.cfg.BinaryPath)
		eo.mu.Unlock()
		return errors.New(eo.reason)
	}
	args := []string{"--serve"}
	switch {
	case len(eo.cfg.Args) > 0:
		args = append(args, eo.cfg.Args...)
	case eo.cfg.ModelPath != "":
		args = append(args, "--native", eo.cfg.ModelPath, "--max-context", strconv.Itoa(eo.cfg.ContextSize))
	default:
		eo.state, eo.reason = StateFailed, "no model is configured (start with --config strata-<model>.json or --model-path)"
		eo.mu.Unlock()
		return errors.New(eo.reason)
	}
	args = append(args, eo.cfg.ExtraArgs...)

	procCtx, cancel := context.WithCancel(context.Background())
	cmd := exec.CommandContext(procCtx, eo.cfg.BinaryPath, args...)
	cmd.Dir = eo.cfg.Cwd
	cmd.Env = append(os.Environ(), eo.cfg.Env...)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		cancel()
		eo.state, eo.reason = StateFailed, err.Error()
		eo.mu.Unlock()
		return err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		cancel()
		eo.state, eo.reason = StateFailed, err.Error()
		eo.mu.Unlock()
		return err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		cancel()
		eo.state, eo.reason = StateFailed, err.Error()
		eo.mu.Unlock()
		return err
	}
	if err := cmd.Start(); err != nil {
		cancel()
		eo.state, eo.reason = StateFailed, "cannot start the engine: "+err.Error()
		eo.mu.Unlock()
		return err
	}
	eo.cmd, eo.stdin, eo.cancelFunc = cmd, stdin, cancel
	eo.state, eo.reason = StateStarting, ""
	eo.lines = make(chan string, 8192)
	eo.readyCh = make(chan struct{})
	lines, readyCh := eo.lines, eo.readyCh
	eo.mu.Unlock()

	go eo.pumpStderr(stderr)
	go eo.readLoop(bufio.NewReaderSize(stdout, 1<<20), lines, readyCh)
	go func() {
		select {
		case <-readyCh:
		case <-time.After(eo.cfg.StartTimeout):
			eo.mu.Lock()
			if eo.state == StateStarting {
				eo.state, eo.reason = StateFailed, fmt.Sprintf("the engine did not become ready within %s", eo.cfg.StartTimeout)
				cancel()
			}
			eo.mu.Unlock()
		}
	}()
	go eo.watchProcess(cmd)
	return nil
}

func (eo *EngineOrchestrator) pumpStderr(r io.Reader) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		fmt.Fprintln(os.Stderr, line)
		eo.mu.Lock()
		eo.logTail = append(eo.logTail, line)
		if len(eo.logTail) > 30 {
			eo.logTail = eo.logTail[len(eo.logTail)-30:]
		}
		eo.mu.Unlock()
	}
}

func (eo *EngineOrchestrator) readLoop(r *bufio.Reader, lines chan string, readyCh chan struct{}) {
	defer close(lines)
	ready := false
	for {
		line, err := r.ReadString('\n')
		if err != nil {
			return
		}
		line = strings.TrimSpace(line)
		if !ready {
			if strings.HasPrefix(line, "INFO ") {
				eo.mu.Lock()
				if eo.readyCh == readyCh { // not an earlier process's output after a restart
					for _, kv := range strings.Fields(line[5:]) {
						if p := strings.SplitN(kv, "=", 2); len(p) == 2 {
							eo.info[p[0]] = p[1]
						}
					}
				}
				eo.mu.Unlock()
				continue
			}
			if strings.HasPrefix(line, "READY") {
				f := strings.Fields(line)
				eo.mu.Lock()
				if eo.readyCh != readyCh { // an earlier process's READY after a restart: the new one is not ready
					eo.mu.Unlock()
					return
				}
				if len(f) >= 2 {
					if n, err := strconv.Atoi(f[1]); err == nil && n > 0 {
						eo.maxContext = n
					}
				}
				for _, x := range f[min(2, len(f)):] {
					if x == "stop" {
						eo.canStop = true
					}
				}
				eo.state, eo.reason = StateReady, ""
				eo.mu.Unlock()
				ready = true
				close(readyCh)
				continue
			}
			continue // anything else before READY is the engine's own chatter (stderr normally)
		}
		lines <- line
	}
}

func (eo *EngineOrchestrator) watchProcess(cmd *exec.Cmd) {
	err := cmd.Wait()
	eo.mu.Lock()
	defer eo.mu.Unlock()
	if eo.state == StateStopped || eo.cmd != cmd { // stopped, or an earlier process's exit after a restart
		return
	}
	why := "the engine process exited"
	if err != nil {
		why = "the engine process exited: " + err.Error()
	}
	if n := len(eo.logTail); n > 0 {
		tail := eo.logTail
		if n > 5 {
			tail = tail[n-5:]
		}
		why += " (" + strings.Join(tail, " | ") + ")"
	}
	eo.state, eo.reason = StateFailed, why
}

// WaitReady blocks until the engine is ready, failed, or ctx ends.
func (eo *EngineOrchestrator) WaitReady(ctx context.Context) error {
	for {
		eo.mu.RLock()
		st, reason, ch := eo.state, eo.reason, eo.readyCh
		eo.mu.RUnlock()
		switch st {
		case StateReady:
			return nil
		case StateFailed, StateStopped:
			return &ErrNotReady{State: st, Reason: reason}
		}
		select {
		case <-ch:
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(200 * time.Millisecond):
		}
	}
}

// Status returns the state and the reason for a failed one.
func (eo *EngineOrchestrator) Status() (State, string) {
	eo.mu.RLock()
	defer eo.mu.RUnlock()
	return eo.state, eo.reason
}

func (eo *EngineOrchestrator) IsAlive() bool {
	st, _ := eo.Status()
	return st == StateReady
}

func (eo *EngineOrchestrator) MaxContext() int {
	eo.mu.RLock()
	defer eo.mu.RUnlock()
	return eo.maxContext
}

func (eo *EngineOrchestrator) Info() map[string]string {
	eo.mu.RLock()
	defer eo.mu.RUnlock()
	out := make(map[string]string, len(eo.info))
	for k, v := range eo.info {
		out[k] = v
	}
	return out
}

// samplingKeys mirrors the keys the engine understands; absent keys mean greedy decoding without penalties.
func samplingKeys(s SamplingParams) string {
	var b strings.Builder
	f := func(v float64) string { return strconv.FormatFloat(v, 'g', -1, 64) }
	if s.Temperature > 0 {
		b.WriteString(" temperature=" + f(s.Temperature))
	}
	if s.TopP > 0 && s.TopP < 1 {
		b.WriteString(" top_p=" + f(s.TopP))
	}
	if s.TopK >= 0 && s.Temperature > 0 {
		k := s.TopK
		if k < 1 || k > 64 { // the engine's sampled path keeps at most 64 candidates
			k = 64
		}
		b.WriteString(" top_k=" + strconv.Itoa(k))
	}
	if s.MinP > 0 && s.MinP <= 1 {
		b.WriteString(" min_p=" + f(s.MinP))
	}
	if s.RepetitionPenalty > 0 && s.RepetitionPenalty != 1 {
		b.WriteString(" penalty_repeat=" + f(s.RepetitionPenalty))
	}
	if s.FrequencyPenalty != 0 {
		b.WriteString(" penalty_freq=" + f(s.FrequencyPenalty))
	}
	if s.PresencePenalty != 0 {
		b.WriteString(" penalty_present=" + f(s.PresencePenalty))
	}
	if s.Seed != 0 {
		b.WriteString(" seed=" + strconv.FormatInt(s.Seed, 10))
	}
	return b.String()
}

// GenerateStream encodes prompt (control tokens parsed) and streams the reply. Prefer GenerateIDs when the caller has
// already tokenized (it can protect user text from being read as control tokens).
func (eo *EngineOrchestrator) GenerateStream(ctx context.Context, prompt string, sampling SamplingParams, maxNew int, stops []string, out chan<- TokenEvent) {
	eo.GenerateIDs(ctx, eo.Tokenizer().Encode(prompt, true), sampling, maxNew, nil, stops, out)
}

// GenerateIDs sends one request and streams text events until an event with IsEnd. stopIDs end the reply (the end-of-turn
// tokens); stops are text stop sequences. The channel is closed at the end.
func (eo *EngineOrchestrator) GenerateIDs(ctx context.Context, ids []int, sampling SamplingParams, maxNew int, stopIDs map[int]bool, stops []string, out chan<- TokenEvent) {
	defer close(out)
	eo.mu.RLock()
	st, reason := eo.state, eo.reason
	lines, stdin, canStop, codec := eo.lines, eo.stdin, eo.canStop, eo.tokenizer
	eo.mu.RUnlock()
	if st != StateReady {
		out <- TokenEvent{Error: &ErrNotReady{State: st, Reason: reason}, IsEnd: true, FinishReason: "error"}
		return
	}
	// one request at a time
	select {
	case eo.reqSem <- struct{}{}:
		defer func() { <-eo.reqSem }()
	case <-ctx.Done():
		out <- TokenEvent{IsEnd: true, FinishReason: "cancel"}
		return
	}
	if maxNew <= 0 {
		maxNew = 2048
	}
	idStrs := make([]string, len(ids))
	for i, id := range ids {
		idStrs[i] = strconv.Itoa(id)
	}
	if _, err := io.WriteString(stdin, fmt.Sprintf("GEN %d%s %s\n", maxNew, samplingKeys(sampling), strings.Join(idStrs, ","))); err != nil {
		out <- TokenEvent{Error: fmt.Errorf("the engine stopped unexpectedly: %w", err), IsEnd: true, FinishReason: "error"}
		return
	}

	matcher := NewStopMatcher(stops)
	detok := tokenizer.NewDetokenizer(codec)
	genCount := 0
	finish := ""     // set when we decide to stop; the engine is then drained to its DONE line
	stopped := false // STOP sent
	sendStop := func() {
		if !stopped && canStop {
			_, _ = io.WriteString(stdin, "STOP\n")
		}
		stopped = true
	}
	curID := -1
	emit := func(s string) {
		if s != "" {
			out <- TokenEvent{Text: s, TokenID: curID}
		}
	}
	done := ctx.Done()
	for {
		var line string
		var ok bool
		select {
		case <-done:
			done = nil
			if finish == "" {
				finish = "cancel"
			}
			sendStop()
			continue
		case line, ok = <-lines:
		}
		if !ok {
			st, why := eo.Status()
			if why == "" {
				why = "the engine process ended"
			}
			out <- TokenEvent{Error: &ErrNotReady{State: st, Reason: why}, IsEnd: true, FinishReason: "error", GenTokens: genCount}
			return
		}
		switch {
		case strings.HasPrefix(line, "T "):
			id, err := strconv.Atoi(strings.TrimSpace(line[2:]))
			if err != nil || finish != "" {
				continue // draining after a stop decision
			}
			if stopIDs[id] {
				finish = "stop"
				sendStop()
				continue
			}
			genCount++
			curID = id
			safe := matcher.Push(detok.Push(id))
			emit(safe)
			if matcher.Hit() != "" {
				finish = "stop"
				sendStop()
			}
		case strings.HasPrefix(line, "DONE"):
			emit(matcher.Flush()) // text held back as a possible stop string that never completed
			emit(detok.Flush())
			f := strings.Fields(line)
			ev := TokenEvent{IsEnd: true, GenTokens: genCount}
			reasonStr := finish
			if len(f) >= 6 && reasonStr == "" {
				reasonStr = f[5]
			}
			if len(f) >= 3 {
				ev.PromptTokens, _ = strconv.Atoi(f[2])
			}
			if len(f) >= 5 {
				if ms, err := strconv.ParseFloat(f[4], 64); err == nil && ms > 0 {
					n, _ := strconv.Atoi(f[1])
					ev.TokPerSec = float64(n) / (ms / 1000)
				}
			}
			if reasonStr == "" {
				reasonStr = "stop"
			}
			ev.FinishReason = reasonStr
			out <- ev
			return
		case strings.HasPrefix(line, "ERR"):
			out <- TokenEvent{Error: errors.New(strings.TrimSpace(strings.TrimPrefix(line, "ERR"))), IsEnd: true, FinishReason: "error", GenTokens: genCount}
			return
		}
		// PP / RESUME / REUSED / INFO lines are progress; ignored here
	}
}

// Stop ends the engine process.
func (eo *EngineOrchestrator) Stop() error {
	eo.mu.Lock()
	eo.state = StateStopped
	cancel, stdin, cmd := eo.cancelFunc, eo.stdin, eo.cmd
	eo.mu.Unlock()
	if stdin != nil {
		_, _ = io.WriteString(stdin, "QUIT\n")
		stdin.Close()
	}
	if cancel != nil {
		cancel()
	}
	if cmd != nil && cmd.Process != nil {
		_ = cmd.Process.Kill()
	}
	return nil
}
