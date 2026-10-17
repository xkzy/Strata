//go:build windows

package main

import (
	"os/exec"
	"time"
)

func setProcessGroup(cmd *exec.Cmd) {}

// stopProcessGroup ends the server; Windows has no process group to signal, so the engine is stopped with taskkill's
// tree option.
func stopProcessGroup(cmd *exec.Cmd, exited <-chan struct{}) {
	if cmd.Process == nil {
		return
	}
	_ = exec.Command("taskkill", "/T", "/F", "/PID", itoa(cmd.Process.Pid)).Run()
	select {
	case <-exited:
	case <-time.After(15 * time.Second):
	}
	time.Sleep(3 * time.Second)
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for ; n > 0; n /= 10 {
		b = append([]byte{byte('0' + n%10)}, b...)
	}
	return string(b)
}
