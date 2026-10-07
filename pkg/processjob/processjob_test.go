// pkg/processjob/processjob_test.go - Unit Tests for Process Job Containment
package processjob

import (
	"context"
	"os/exec"
	"testing"
	"time"
)

func TestProcessJob_RunWithTimeout(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	cmd := exec.Command("echo", "hello process group")
	out, err := RunWithTimeout(ctx, cmd)
	if err != nil {
		t.Fatalf("unexpected error running command: %v", err)
	}

	if len(out) == 0 {
		t.Fatalf("expected output from command")
	}
}

func TestProcessJob_TerminationOnTimeout(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	cmd := exec.Command("sleep", "10")
	_, err := RunWithTimeout(ctx, cmd)
	if err == nil {
		t.Fatalf("expected context deadline error, got nil")
	}
}
