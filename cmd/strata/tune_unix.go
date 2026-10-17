//go:build !windows

package main

import (
	"os/exec"
	"syscall"
	"time"
)

// setProcessGroup puts the server and its engine in one group, so stopping the group stops both (the engine does not
// exit with the server).
func setProcessGroup(cmd *exec.Cmd) { cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true} }

func stopProcessGroup(cmd *exec.Cmd, exited <-chan struct{}) {
	if cmd.Process == nil {
		return
	}
	_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGTERM)
	select {
	case <-exited:
	case <-time.After(15 * time.Second):
	}
	_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	time.Sleep(3 * time.Second) // the cards' memory is given back before the next candidate loads
}
