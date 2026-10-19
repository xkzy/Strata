package engineipc

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestXidLinesKeepsFaultsNotRPCSpam(t *testing.T) {
	in := "NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus.\n" +
		strings.Repeat("NVRM: _issueRpcAndWait: rpcSendMessage failed\n", 50) +
		"NVRM: Xid (PCI:0000:01:00): 154, GPU recovery action changed from 0x0 (None) to 0x2 (Node Reboot Required)\n"
	got := xidLines(in)
	if len(got) != 2 {
		t.Fatalf("got %d lines, want the 2 fault lines:\n%v", len(got), got)
	}
	for _, l := range got {
		if strings.Contains(l, "rpcSendMessage") {
			t.Errorf("RPC spam kept: %s", l)
		}
	}
}

func TestGPUSnapshotIsLoggedBeforeRestart(t *testing.T) {
	old := gpuSnapshot
	gpuSnapshot = func() []string { return []string{"gpu: 0, test card", "kernel: Xid 79"} }
	defer func() { gpuSnapshot = old }()
	dir := t.TempDir()
	script, counter := crashingEngine(t, dir, "echo 'boom' >&2\nexit 1\n")
	eo := NewEngineOrchestrator(EngineConfig{
		BinaryPath: script, Args: []string{"--pack", "x"}, Cwd: dir, RestartDelay: 20 * time.Millisecond,
	})
	if err := eo.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	defer eo.Stop()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) && starts(counter) < 2 {
		time.Sleep(50 * time.Millisecond)
	}
	b, _ := os.ReadFile(filepath.Join(dir, "strata-engine-failures.log"))
	for _, want := range []string{"GPU state before restart 1/8", "gpu: 0, test card", "kernel: Xid 79"} {
		if !strings.Contains(string(b), want) {
			t.Errorf("failure log lacks %q:\n%s", want, b)
		}
	}
}

// Stop must not return while a background snapshot is still writing the failure log: a caller that then removes the
// logs directory (a test's TempDir, or a cleanup) must not race the write.
func TestStopWaitsForBackgroundSnapshot(t *testing.T) {
	release := make(chan struct{})
	started := make(chan struct{})
	old := gpuSnapshot
	gpuSnapshot = func() []string {
		close(started)
		<-release
		return []string{"gpu: 0, test card"}
	}
	defer func() { gpuSnapshot = old }()

	dir := t.TempDir()
	script, _ := crashingEngine(t, dir, "echo 'CUDA error: unspecified launch failure' >&2\nexit 1\n")
	eo := NewEngineOrchestrator(EngineConfig{BinaryPath: script, Args: []string{"x"}, Cwd: dir, NoAutoRestart: true})
	if err := eo.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	select {
	case <-started:
	case <-time.After(10 * time.Second):
		t.Fatal("the background snapshot did not start")
	}

	stopped := make(chan struct{})
	go func() { _ = eo.Stop(); close(stopped) }()
	select {
	case <-stopped:
		t.Fatal("Stop returned while the snapshot was still writing the failure log")
	case <-time.After(200 * time.Millisecond):
	}
	close(release)
	select {
	case <-stopped:
	case <-time.After(10 * time.Second):
		t.Fatal("Stop did not return after the snapshot finished")
	}
}

// The failure reason must name the card that fell off the bus and the Xid, not only the engine's own text
// ("cannot pin 322 MiB"): the owner maps bus id to card with lspci, and the Xid says what the driver saw.
func TestGPUFaultSummaryNamesTheCardAndXid(t *testing.T) {
	lines := []string{
		"gpu: 0, NVIDIA GeForce RTX 4060, 0000:03:00.0, 41, [N/A], 512 MiB, 8188 MiB, 4, 8",
		"kernel: NVRM: Xid (PCI:0000:01:00.0): 79, GPU has fallen off the bus at PCI:0000:01:00.0.",
		"kernel: NVRM: Xid (PCI:0000:01:00.0): 154, GPU recovery action changed from 0x0 (None) to 0x2 (Node Reboot Required)",
	}
	got := gpuFaultSummary(lines)
	for _, want := range []string{"0000:01:00.0", "Xid 79", "fallen off the bus"} {
		if !strings.Contains(got, want) {
			t.Errorf("summary lacks %q: %q", want, got)
		}
	}
	if strings.Contains(got, "Xid 154") {
		t.Errorf("the second, consequential Xid must not mask the first: %q", got)
	}
	if strings.Contains(got, "RTX 4060") {
		t.Errorf("the healthy card must not be named as the one that died: %q", got)
	}
}

// When nvidia-smi still lists the card that faulted, its name goes into the summary.
func TestGPUFaultSummaryUsesTheListedName(t *testing.T) {
	lines := []string{
		"gpu: 0, NVIDIA GeForce RTX 5060 Ti, 0000:01:00.0, 39, 3.84 W, 447 MiB, 16311 MiB, 4, 8",
		"kernel: NVRM: Xid (PCI:0000:01:00.0): 79, GPU has fallen off the bus.",
	}
	got := gpuFaultSummary(lines)
	if !strings.Contains(got, "NVIDIA GeForce RTX 5060 Ti") {
		t.Errorf("the listed card name is missing: %q", got)
	}
}

// A lost device handle without an Xid (the state on 2026-10-19) is named too.
func TestGPUFaultSummaryNamesADriverLostHandle(t *testing.T) {
	got := gpuFaultSummary([]string{
		"gpu: driver lost device handle - try: sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm",
		"gpu: NVIDIA-SMI has failed: Unable to determine the device handle for GPU0: 0000:01:00.0: Unknown Error",
	})
	if !strings.Contains(got, "0000:01:00.0") || !strings.Contains(got, "device handle") {
		t.Errorf("a lost device handle must name the card: %q", got)
	}
}

func TestGPUFaultSummaryOfACleanSnapshot(t *testing.T) {
	if got := gpuFaultSummary([]string{"gpu: 0, RTX 4060, 0000:03:00.0", "kernel: no Xid / fallen-off-the-bus lines this boot"}); got != "" {
		t.Errorf("a clean snapshot must summarize to nothing, got %q", got)
	}
}

// The reason a client sees (429, /health, /status) must name the card that fell, not only the engine's own
// "unspecified launch failure".
func TestFailureReasonNamesTheLostCard(t *testing.T) {
	old := gpuSnapshot
	gpuSnapshot = func() []string {
		return []string{
			"gpu: NVIDIA-SMI has failed: Unable to determine the device handle for GPU0: 0000:01:00.0: Unknown Error",
			"kernel: NVRM: Xid (PCI:0000:01:00.0): 79, GPU has fallen off the bus at PCI:0000:01:00.0.",
		}
	}
	defer func() { gpuSnapshot = old }()
	dir := t.TempDir()
	script, _ := crashingEngine(t, dir, "echo 'prefill copy_i32: CUDA error: unspecified launch failure' >&2\nexit 1\n")
	eo := NewEngineOrchestrator(EngineConfig{BinaryPath: script, Args: []string{"x"}, Cwd: dir, NoAutoRestart: true})
	if err := eo.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	defer eo.Stop()
	deadline := time.Now().Add(10 * time.Second)
	var reason string
	for time.Now().Before(deadline) {
		_, reason = eo.Status()
		if strings.Contains(reason, "0000:01:00.0") && strings.Contains(reason, "Xid 79") {
			return
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("the failure reason never named the lost card: %q", reason)
}

func TestDeviceLost(t *testing.T) {
	if _, ok := deviceLost([]string{"gpu: 0, RTX", "kernel: no Xid / fallen-off-the-bus lines this boot"}); ok {
		t.Fatal("a clean snapshot read as device loss")
	}
	if _, ok := deviceLost([]string{"kernel: NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus."}); !ok {
		t.Fatal("Xid 79 not read as device loss")
	}
}

func TestNoRestartIntoLostGPU(t *testing.T) {
	old := gpuSnapshot
	gpuSnapshot = func() []string {
		return []string{"kernel: NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus."}
	}
	defer func() { gpuSnapshot = old }()
	dir := t.TempDir()
	script, counter := crashingEngine(t, dir, "echo 'CUDA error: unspecified launch failure' >&2\nexit 1\n")
	eo := NewEngineOrchestrator(EngineConfig{
		BinaryPath: script, Args: []string{"--pack", "x"}, Cwd: dir, RestartDelay: 20 * time.Millisecond,
	})
	if err := eo.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	defer eo.Stop()
	time.Sleep(700 * time.Millisecond)
	if n := starts(counter); n != 1 {
		t.Fatalf("engine started %d times; a lost GPU must not be restarted into", n)
	}
	b, _ := os.ReadFile(filepath.Join(dir, "strata-engine-failures.log"))
	if !strings.Contains(string(b), "not restarting: the driver reports a lost or faulted GPU") {
		t.Errorf("failure log lacks the reason:\n%s", b)
	}
}
