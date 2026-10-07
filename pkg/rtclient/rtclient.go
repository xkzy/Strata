// Package rtclient runs the C++ transparent runtime (strata_rt_server) as a sidecar process and speaks its JSON-lines
// protocol. The runtime decides what the model sees (virtual context, retrieval, memory) and checks what the model says
// (claim verification, loop protection, recovery); this side owns what it cannot: the tokenizer, the chat template and
// the engine. The rt calls back into the Backend to build prompts and to run the model.
package rtclient

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sync"
	"time"
)

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	Name    string `json:"name,omitempty"`
}

type Scope struct {
	Tenant    string `json:"tenant,omitempty"`
	User      string `json:"user,omitempty"`
	Workspace string `json:"workspace,omitempty"`
	Agent     string `json:"agent,omitempty"`
	Session   string `json:"session,omitempty"`
}

type Sampling struct {
	Temperature       float64  `json:"temperature"`
	TopP              float64  `json:"top_p"`
	TopK              int      `json:"top_k"`
	MinP              float64  `json:"min_p,omitempty"`
	RepetitionPenalty float64  `json:"repetition_penalty,omitempty"`
	FrequencyPenalty  float64  `json:"frequency_penalty,omitempty"`
	PresencePenalty   float64  `json:"presence_penalty,omitempty"`
	Seed              int64    `json:"seed,omitempty"`
	Stop              []string `json:"stop,omitempty"`
}

type Request struct {
	Scope           Scope
	Messages        []Message
	Sampling        Sampling
	MaxTokens       int
	ReasoningPrefix bool // the reply stream starts inside a <think> block
	Ephemeral       bool // no session identity: a private context for this request only
	Debug           bool
}

type TraceEvent struct {
	Ms     float64 `json:"ms"`
	Kind   string  `json:"kind"`
	Detail string  `json:"detail"`
}

type Done struct {
	Finish           string
	PromptTokens     int
	CompletionTokens int
	Trace            []TraceEvent
}

// APIError is an error the runtime reports for one request (a status the HTTP layer can use).
type APIError struct {
	Status int
	Type   string
	Msg    string
}

func (e *APIError) Error() string { return e.Msg }

// GenToken is one generated token of a model run.
type GenToken struct {
	ID   int
	Text string
	End  bool
	// set on End
	Finish string
	Err    error
}

// Backend is what the host provides to the runtime.
type Backend interface {
	// BuildPrompt renders the messages with the chat template, appends prefix (text already delivered) after the
	// assistant turn start, and tokenizes. An *APIError carries a status.
	BuildPrompt(ctx context.Context, messages []Message, prefix string) ([]int, error)
	// Generate runs the model on ids and streams tokens; the channel ends with a token that has End set. Cancelling ctx
	// stops the run. The channel must be drained or closed by the implementation (no leaks after cancellation).
	Generate(ctx context.Context, ids []int, s Sampling, maxTokens int) <-chan GenToken
}

type Config struct {
	Binary         string
	PhysicalTokens int
	VirtualTokens  int
	StorageDir     string // "" = the runtime's default
	Persist        bool
	Verification   bool
	Identity       string // model + tokenizer + template identity (KV compatibility)
	StartTimeout   time.Duration
}

type Client struct {
	cfg Config

	mu      sync.Mutex // serializes requests: one runtime, one engine
	procMu  sync.Mutex
	cmd     *exec.Cmd
	stdin   io.WriteCloser
	writeMu sync.Mutex
	in      chan map[string]any
	closed  chan struct{}
	ready   bool
	lastErr string
	nextID  int64
}

func New(cfg Config) *Client {
	if cfg.StartTimeout <= 0 {
		cfg.StartTimeout = 30 * time.Second
	}
	return &Client{cfg: cfg}
}

// FindBinary looks for strata_rt_server in the usual places.
func FindBinary(explicit string) string {
	if explicit != "" {
		return explicit
	}
	exe, _ := os.Executable()
	cands := []string{"./engine/strata_rt_server", "./build/strata_rt_server", "./strata_rt_server"}
	if exe != "" {
		dir := exe[:max(0, lastSlash(exe))]
		cands = append(cands, dir+"/strata_rt_server")
	}
	for _, c := range cands {
		if st, err := os.Stat(c); err == nil && !st.IsDir() {
			return c
		}
	}
	if p, err := exec.LookPath("strata_rt_server"); err == nil {
		return p
	}
	return ""
}

func lastSlash(s string) int {
	for i := len(s) - 1; i >= 0; i-- {
		if s[i] == '/' || s[i] == '\\' {
			return i
		}
	}
	return -1
}

func (c *Client) send(v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	c.writeMu.Lock()
	defer c.writeMu.Unlock()
	if c.stdin == nil {
		return errors.New("the runtime is not running")
	}
	_, err = c.stdin.Write(append(b, '\n'))
	return err
}

// ensure starts the sidecar if it is not running and waits for its ready message.
func (c *Client) ensure() error {
	c.procMu.Lock()
	defer c.procMu.Unlock()
	if c.cmd != nil {
		select {
		case <-c.closed:
			c.cmd, c.ready = nil, false
		default:
			if c.ready {
				return nil
			}
		}
	}
	if c.cfg.Binary == "" {
		return errors.New("no runtime binary (strata_rt_server)")
	}
	cmd := exec.Command(c.cfg.Binary)
	cmd.Stderr = os.Stderr
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	if err := cmd.Start(); err != nil {
		c.lastErr = err.Error()
		return err
	}
	c.cmd, c.stdin, c.ready = cmd, stdin, false
	c.in = make(chan map[string]any, 256)
	c.closed = make(chan struct{})
	in, closed := c.in, c.closed
	go func() {
		defer close(closed)
		defer close(in)
		r := bufio.NewReaderSize(stdout, 1<<20)
		for {
			line, err := r.ReadBytes('\n')
			if len(line) > 1 {
				var m map[string]any
				if json.Unmarshal(line, &m) == nil {
					in <- m
				}
			}
			if err != nil {
				return
			}
		}
	}()
	go func() { _ = cmd.Wait() }()

	conf := map[string]any{"op": "config", "physical_tokens": c.cfg.PhysicalTokens, "persist": c.cfg.Persist,
		"storage_dir": c.cfg.StorageDir, "verification": c.cfg.Verification, "identity": c.cfg.Identity}
	if c.cfg.VirtualTokens > 0 {
		conf["virtual_tokens"] = c.cfg.VirtualTokens
	}
	if err := c.send(conf); err != nil {
		return err
	}
	timer := time.After(c.cfg.StartTimeout)
	for {
		select {
		case m, ok := <-in:
			if !ok {
				c.lastErr = "the runtime exited during start-up"
				return errors.New(c.lastErr)
			}
			if m["op"] == "ready" {
				c.ready = true
				return nil
			}
		case <-timer:
			_ = cmd.Process.Kill()
			c.lastErr = "the runtime did not become ready"
			return errors.New(c.lastErr)
		}
	}
}

// Available reports whether the runtime can be started (a binary is configured).
func (c *Client) Available() bool { return c.cfg.Binary != "" }

// Healthy reports whether the sidecar is up right now.
func (c *Client) Healthy() bool {
	c.procMu.Lock()
	defer c.procMu.Unlock()
	if c.cmd == nil || !c.ready {
		return false
	}
	select {
	case <-c.closed:
		return false
	default:
		return true
	}
}

func (c *Client) LastError() string { return c.lastErr }

func asInt(v any) int {
	f, _ := v.(float64)
	return int(f)
}

// Run sends one request and returns when the runtime is done. Released text is passed to sink as it arrives.
func (c *Client) Run(ctx context.Context, req Request, be Backend, sink func(text string)) (*Done, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.ensure(); err != nil {
		return nil, &APIError{Status: 503, Type: "runtime_unavailable", Msg: "the runtime is not available: " + err.Error()}
	}
	c.procMu.Lock()
	in, closed := c.in, c.closed
	c.procMu.Unlock()

	c.nextID++
	id := c.nextID
	sampling := map[string]any{}
	b, _ := json.Marshal(req.Sampling)
	_ = json.Unmarshal(b, &sampling)
	msg := map[string]any{"op": "request", "id": id, "scope": req.Scope, "messages": req.Messages, "sampling": sampling,
		"max_tokens": req.MaxTokens, "reasoning_prefix": req.ReasoningPrefix, "debug": req.Debug, "ephemeral": req.Ephemeral}
	if err := c.send(msg); err != nil {
		return nil, &APIError{Status: 503, Type: "runtime_unavailable", Msg: "the runtime is not available: " + err.Error()}
	}

	type gen struct{ cancel context.CancelFunc }
	var gmu sync.Mutex
	gens := map[int]*gen{}
	var wg sync.WaitGroup
	defer wg.Wait()
	cancelAll := func() {
		gmu.Lock()
		for _, g := range gens {
			g.cancel()
		}
		gmu.Unlock()
	}
	defer cancelAll()

	done := ctx.Done()
	cancelSent := false
	var drainDeadline <-chan time.Time
	for {
		select {
		case <-done:
			done = nil
			if !cancelSent {
				cancelSent = true
				_ = c.send(map[string]any{"op": "cancel", "id": id})
				drainDeadline = time.After(30 * time.Second)
			}
		case <-drainDeadline:
			return nil, &APIError{Status: 504, Type: "runtime_error", Msg: "the runtime did not finish after cancellation"}
		case <-closed:
			return nil, &APIError{Status: 502, Type: "runtime_error", Msg: "the runtime process ended unexpectedly"}
		case m, ok := <-in:
			if !ok {
				return nil, &APIError{Status: 502, Type: "runtime_error", Msg: "the runtime process ended unexpectedly"}
			}
			switch m["op"] {
			case "emit":
				if t, _ := m["text"].(string); t != "" && sink != nil && !cancelSent {
					sink(t)
				}
			case "done":
				d := &Done{Finish: fmt.Sprint(m["finish"]), PromptTokens: asInt(m["prompt_tokens"]), CompletionTokens: asInt(m["completion_tokens"])}
				if tr, ok := m["trace"].([]any); ok {
					for _, t := range tr {
						if o, ok := t.(map[string]any); ok {
							ms, _ := o["ms"].(float64)
							k, _ := o["kind"].(string)
							dt, _ := o["detail"].(string)
							d.Trace = append(d.Trace, TraceEvent{ms, k, dt})
						}
					}
				}
				return d, nil
			case "error":
				msgText, _ := m["message"].(string)
				typ, _ := m["type"].(string)
				return nil, &APIError{Status: asInt(m["status"]), Type: typ, Msg: msgText}
			case "prompt":
				rid := m["rid"]
				var msgs []Message
				if raw, err := json.Marshal(m["messages"]); err == nil {
					_ = json.Unmarshal(raw, &msgs)
				}
				prefix, _ := m["prefix"].(string)
				ids, err := be.BuildPrompt(ctx, msgs, prefix)
				if err != nil {
					status, typ := 500, "server_error"
					var ae *APIError
					if errors.As(err, &ae) {
						status, typ = ae.Status, ae.Type
					}
					_ = c.send(map[string]any{"op": "prompt_error", "rid": rid, "status": status, "type": typ, "message": err.Error()})
					continue
				}
				_ = c.send(map[string]any{"op": "prompt_result", "rid": rid, "ids": ids})
			case "gen":
				gid := asInt(m["gid"])
				var ids []int
				if raw, err := json.Marshal(m["ids"]); err == nil {
					_ = json.Unmarshal(raw, &ids)
				}
				var s Sampling
				if raw, err := json.Marshal(m["sampling"]); err == nil {
					_ = json.Unmarshal(raw, &s)
				}
				s.Stop = req.Sampling.Stop
				maxTok := asInt(m["max_tokens"])
				gctx, gcancel := context.WithCancel(context.Background())
				gmu.Lock()
				gens[gid] = &gen{cancel: gcancel}
				gmu.Unlock()
				events := be.Generate(gctx, ids, s, maxTok)
				wg.Add(1)
				go func() {
					defer wg.Done()
					defer gcancel()
					end := false
					for ev := range events {
						if ev.End {
							end = true
							if ev.Err != nil {
								_ = c.send(map[string]any{"op": "gen_error", "gid": gid, "message": ev.Err.Error()})
							} else {
								_ = c.send(map[string]any{"op": "gen_end", "gid": gid, "finish": ev.Finish})
							}
							continue
						}
						_ = c.send(map[string]any{"op": "gen_token", "gid": gid, "id": ev.ID, "text": ev.Text})
					}
					if !end { // the backend closed without an end event
						_ = c.send(map[string]any{"op": "gen_end", "gid": gid, "finish": "stop"})
					}
					gmu.Lock()
					delete(gens, gid)
					gmu.Unlock()
				}()
			case "gen_cancel":
				gid := asInt(m["gid"])
				gmu.Lock()
				if g := gens[gid]; g != nil {
					g.cancel()
				}
				gmu.Unlock()
			}
		}
	}
}

// Metrics asks the runtime for its counters.
func (c *Client) Metrics(ctx context.Context) (map[string]any, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.ensure(); err != nil {
		return nil, err
	}
	c.procMu.Lock()
	in := c.in
	c.procMu.Unlock()
	c.nextID++
	rid := c.nextID
	if err := c.send(map[string]any{"op": "metrics", "rid": rid}); err != nil {
		return nil, err
	}
	timer := time.After(5 * time.Second)
	for {
		select {
		case m, ok := <-in:
			if !ok {
				return nil, errors.New("the runtime ended")
			}
			if m["op"] == "metrics_result" {
				out, _ := m["metrics"].(map[string]any)
				return out, nil
			}
		case <-timer:
			return nil, errors.New("timeout")
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
}

// Close ends the sidecar.
func (c *Client) Close() {
	c.procMu.Lock()
	defer c.procMu.Unlock()
	if c.cmd != nil {
		_ = c.send(map[string]any{"op": "quit"})
		time.AfterFunc(2*time.Second, func() { _ = c.cmd.Process.Kill() })
	}
}
