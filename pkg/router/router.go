// Package router sends chat requests to the fast MoE engine first and escalates to the dense engine when the answer
// looks like a failure. The two engines do not fit on the GPUs together, so a switch stops one process and starts the
// other; the dense engine then keeps serving until it has handled DenseRequests requests, then the MoE comes back.
package router

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os/exec"
	"sync"
	"syscall"
	"time"
)

// Backend is one engine: its address and how to start and stop it.
type Backend struct {
	Name, URL   string
	Start, Stop func() error
}

// Router serves one request at a time (the engines do too).
type Router struct {
	Fast, Dense   *Backend
	DenseRequests int // requests served by Dense after an escalation before switching back
	Wait          func(b *Backend) error

	mu     sync.Mutex
	cur    *Backend
	left   int
	recent [][32]byte // hashes of the last replies: a repeat means the agent is stuck
}

const escalateHeader = "X-Strata-Escalate" // a client that knows the task failed (tests red) sets this to 1

func (r *Router) switchTo(b *Backend) error {
	if r.cur == b {
		return nil
	}
	if r.cur != nil {
		if err := r.cur.Stop(); err != nil {
			return err
		}
	}
	r.cur = nil
	if err := b.Start(); err != nil {
		return err
	}
	if err := r.Wait(b); err != nil {
		return err
	}
	r.cur = b
	return nil
}

func post(url string, body []byte) (int, []byte, error) {
	resp, err := http.Post(url+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		return 0, nil, err
	}
	defer resp.Body.Close()
	out, err := io.ReadAll(resp.Body)
	return resp.StatusCode, out, err
}

// failed says why a reply counts as a failure ("" if it does not).
func (r *Router) failed(code int, out []byte) string {
	if code != http.StatusOK {
		return fmt.Sprintf("http %d", code)
	}
	var v struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
			Message      struct{ Content string }
		}
	}
	if json.Unmarshal(out, &v) != nil || len(v.Choices) == 0 || v.Choices[0].Message.Content == "" {
		return "empty reply"
	}
	if v.Choices[0].FinishReason == "length" {
		return "cut off at max_tokens"
	}
	h := sha256.Sum256([]byte(v.Choices[0].Message.Content))
	for _, p := range r.recent {
		if p == h {
			return "repeated reply"
		}
	}
	r.recent = append(r.recent, h)
	if len(r.recent) > 3 {
		r.recent = r.recent[1:]
	}
	return ""
}

// Chat handles /v1/chat/completions. Streaming requests go to the current engine unchecked.
func (r *Router) Chat(w http.ResponseWriter, req *http.Request) {
	body, _ := io.ReadAll(req.Body)
	var probe struct{ Stream bool }
	_ = json.Unmarshal(body, &probe)
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.cur == nil || (r.cur == r.Dense && r.left <= 0) {
		if err := r.switchTo(r.Fast); err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
	}
	if req.Header.Get(escalateHeader) == "1" && r.cur != r.Dense {
		log.Printf("router: client asked to escalate")
		r.escalate()
	}
	if probe.Stream {
		code, out, err := post(r.cur.URL, body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		w.WriteHeader(code)
		w.Write(out)
		return
	}
	code, out, err := post(r.cur.URL, body)
	if err != nil {
		code, out = 0, nil
	}
	if why := r.failed(code, out); why != "" && r.cur != r.Dense {
		log.Printf("router: %s gave %s; escalating to %s", r.cur.Name, why, r.Dense.Name)
		r.escalate()
		if r.cur == r.Dense {
			code, out, err = post(r.Dense.URL, body)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadGateway)
				return
			}
		}
	}
	if r.cur == r.Dense {
		r.left--
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	w.Write(out)
}

func (r *Router) escalate() {
	if err := r.switchTo(r.Dense); err != nil {
		log.Printf("router: cannot start %s: %v (staying with the old reply)", r.Dense.Name, err)
		return
	}
	r.left, r.recent = r.DenseRequests, nil
}

// ProcBackend runs shell command cmd as the engine and stops it by signalling its process group.
func ProcBackend(name, url, cmd string) *Backend {
	var c *exec.Cmd
	return &Backend{Name: name, URL: url,
		Start: func() error {
			c = exec.Command("sh", "-c", cmd)
			c.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
			return c.Start()
		},
		Stop: func() error {
			if c == nil || c.Process == nil {
				return nil
			}
			_ = syscall.Kill(-c.Process.Pid, syscall.SIGTERM)
			done := make(chan struct{})
			go func() { c.Wait(); close(done) }()
			select {
			case <-done:
			case <-time.After(60 * time.Second):
				_ = syscall.Kill(-c.Process.Pid, syscall.SIGKILL)
				<-done
			}
			time.Sleep(3 * time.Second) // let the driver release the cards
			return nil
		}}
}

// WaitReady polls GET /v1/models until the engine answers.
func WaitReady(b *Backend) error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	for ctx.Err() == nil {
		if resp, err := http.Get(b.URL + "/v1/models"); err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				return nil
			}
		}
		time.Sleep(2 * time.Second)
	}
	return fmt.Errorf("%s did not become ready", b.Name)
}
