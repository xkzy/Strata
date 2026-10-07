// pkg/processjob/processjob.go - OS Process Tree Containment & Lifecycle Supervisor in Go
package processjob

import (
	"context"
	"os/exec"
	"runtime"
	"sync"
	"syscall"
	"time"
)

type ProcessJob struct {
	mu        sync.Mutex
	processes map[int]*exec.Cmd
}

var (
	defaultJob *ProcessJob
	once       sync.Once
)

func GlobalJob() *ProcessJob {
	once.Do(func() {
		defaultJob = &ProcessJob{
			processes: make(map[int]*exec.Cmd),
		}
	})
	return defaultJob
}

// PrepareCommand sets OS-level process group flags so child processes can be terminated together
func PrepareCommand(cmd *exec.Cmd) {
	if runtime.GOOS != "windows" {
		if cmd.SysProcAttr == nil {
			cmd.SysProcAttr = &syscall.SysProcAttr{}
		}
		cmd.SysProcAttr.Setpgid = true
	}
}

// Track registers a running command with the supervisor
func (pj *ProcessJob) Track(cmd *exec.Cmd) {
	pj.mu.Lock()
	defer pj.mu.Unlock()
	if cmd != nil && cmd.Process != nil {
		pj.processes[cmd.Process.Pid] = cmd
	}
}

// Untrack removes a process from the supervisor
func (pj *ProcessJob) Untrack(cmd *exec.Cmd) {
	pj.mu.Lock()
	defer pj.mu.Unlock()
	if cmd != nil && cmd.Process != nil {
		delete(pj.processes, cmd.Process.Pid)
	}
}

// TerminateProcessGroup cleanly stops a command and its entire process tree
func TerminateProcessGroup(cmd *exec.Cmd, timeout time.Duration) error {
	if cmd == nil || cmd.Process == nil {
		return nil
	}

	pid := cmd.Process.Pid

	if runtime.GOOS != "windows" {
		// Send SIGTERM to the process group (-pid)
		_ = syscall.Kill(-pid, syscall.SIGTERM)

		done := make(chan struct{})
		go func() {
			_, _ = cmd.Process.Wait()
			close(done)
		}()

		select {
		case <-done:
			return nil
		case <-time.After(timeout):
			// Escalate to SIGKILL
			_ = syscall.Kill(-pid, syscall.SIGKILL)
			return nil
		}
	} else {
		_ = cmd.Process.Kill()
		return nil
	}
}

// TerminateAll stops all tracked child processes
func (pj *ProcessJob) TerminateAll(timeout time.Duration) {
	pj.mu.Lock()
	cmds := make([]*exec.Cmd, 0, len(pj.processes))
	for _, cmd := range pj.processes {
		cmds = append(cmds, cmd)
	}
	pj.mu.Unlock()

	var wg sync.WaitGroup
	for _, cmd := range cmds {
		wg.Add(1)
		go func(c *exec.Cmd) {
			defer wg.Done()
			_ = TerminateProcessGroup(c, timeout)
		}(cmd)
	}

	c := make(chan struct{})
	go func() {
		wg.Wait()
		close(c)
	}()

	select {
	case <-c:
	case <-time.After(timeout + 500*time.Millisecond):
	}
}

// RunWithTimeout runs a command with process group containment and context timeout
func RunWithTimeout(ctx context.Context, cmd *exec.Cmd) ([]byte, error) {
	PrepareCommand(cmd)
	GlobalJob().Track(cmd)
	defer GlobalJob().Untrack(cmd)

	outChan := make(chan struct {
		out []byte
		err error
	}, 1)

	go func() {
		out, err := cmd.CombinedOutput()
		outChan <- struct {
			out []byte
			err error
		}{out, err}
	}()

	select {
	case <-ctx.Done():
		_ = TerminateProcessGroup(cmd, 1*time.Second)
		return nil, ctx.Err()
	case res := <-outChan:
		return res.out, res.err
	}
}
