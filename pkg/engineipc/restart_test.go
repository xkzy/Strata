package engineipc

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestRestartDelayDoublesAndCaps(t *testing.T) {
	if got := restartDelay(0, 1); got != defaultRestartDelay {
		t.Fatalf("first delay = %v", got)
	}
	if got := restartDelay(time.Second, 4); got != 8*time.Second {
		t.Fatalf("fourth delay = %v", got)
	}
	if got := restartDelay(time.Second, 30); got != maxRestartDelay {
		t.Fatalf("capped delay = %v", got)
	}
}

func crashingEngine(t *testing.T, dir, body string) (script, counter string) {
	t.Helper()
	script = filepath.Join(dir, "engine.sh")
	counter = filepath.Join(dir, "starts")
	if err := os.WriteFile(script, []byte("#!/bin/sh\necho start >> "+counter+"\n"+body), 0o755); err != nil {
		t.Fatal(err)
	}
	return script, counter
}

func starts(counter string) int {
	b, _ := os.ReadFile(counter)
	return strings.Count(string(b), "start")
}

// A fake engine that dies at once: every death is logged and the engine is started again.
func TestCrashedEngineIsLoggedAndRestarted(t *testing.T) {
	dir := t.TempDir()
	script, counter := crashingEngine(t, dir, "echo 'boom: out of memory' >&2\nexit 1\n")
	eo := NewEngineOrchestrator(EngineConfig{
		BinaryPath: script, Args: []string{"--pack", "x"}, Cwd: dir, RestartDelay: 20 * time.Millisecond,
	})
	if err := eo.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	defer eo.Stop()

	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) && starts(counter) < 3 {
		time.Sleep(50 * time.Millisecond)
	}
	if n := starts(counter); n < 3 {
		t.Fatalf("the engine was started %d times, want at least 3 (restarts)", n)
	}
	logText, err := os.ReadFile(filepath.Join(dir, "strata-engine-failures.log"))
	if err != nil {
		t.Fatalf("no failure log: %v", err)
	}
	for _, want := range []string{"engine failed (restart 1/8 in 20ms)", "exit status 1", "boom: out of memory"} {
		if !strings.Contains(string(logText), want) {
			t.Errorf("failure log lacks %q:\n%s", want, logText)
		}
	}
}

// A process stuck in the driver after SIGKILL still holds the GPU; Start must not load a second engine onto that card.
func TestStartRefusesWhileTheOldEngineIsStillAlive(t *testing.T) {
	old := reapEngine
	reapEngine = func(*exec.Cmd, context.CancelFunc, <-chan struct{}) bool { return false }
	defer func() { reapEngine = old }()
	dir := t.TempDir()
	script, counter := crashingEngine(t, dir, "exit 0\n")
	eo := NewEngineOrchestrator(EngineConfig{BinaryPath: script, Args: []string{"x"}, Cwd: dir})
	err := eo.Start(context.Background())
	if err == nil {
		t.Fatal("Start launched an engine while the previous process was still alive")
	}
	if !strings.Contains(err.Error(), "still alive after SIGKILL") {
		t.Fatalf("the error does not say why: %v", err)
	}
	if n := starts(counter); n != 0 {
		t.Fatalf("the engine binary was started %d times, want 0", n)
	}
	if st, why := eo.Status(); st != StateFailed || !strings.Contains(why, "still alive after SIGKILL") {
		t.Fatalf("state %v, reason %q", st, why)
	}
}

func TestStopPreventsRestart(t *testing.T) {
	dir := t.TempDir()
	script, counter := crashingEngine(t, dir, "exit 1\n")
	eo := NewEngineOrchestrator(EngineConfig{BinaryPath: script, Args: []string{"x"}, Cwd: dir, RestartDelay: 300 * time.Millisecond})
	if err := eo.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	time.Sleep(100 * time.Millisecond) // it died; a restart is pending
	_ = eo.Stop()
	time.Sleep(700 * time.Millisecond)
	if n := starts(counter); n != 1 {
		t.Fatalf("started %d times after Stop, want 1", n)
	}
}

func TestNoAutoRestartStaysDown(t *testing.T) {
	dir := t.TempDir()
	script, counter := crashingEngine(t, dir, "exit 1\n")
	eo := NewEngineOrchestrator(EngineConfig{BinaryPath: script, Args: []string{"x"}, Cwd: dir,
		RestartDelay: 10 * time.Millisecond, NoAutoRestart: true})
	if err := eo.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	defer eo.Stop()
	time.Sleep(500 * time.Millisecond)
	if n := starts(counter); n != 1 {
		t.Fatalf("started %d times with NoAutoRestart, want 1", n)
	}
	if st, _ := eo.Status(); st != StateFailed {
		t.Fatalf("state = %v, want failed", st)
	}
}

func TestGPUFaultMarkers(t *testing.T) {
	if m, ok := gpuFault([]string{"fine", "prefill copy_i32: unspecified launch failure", "later line"}); !ok || m != "unspecified launch failure" {
		t.Fatalf("launch failure not recognised: %q %v", m, ok)
	}
	if _, ok := gpuFault([]string{"boom: out of memory"}); ok {
		t.Fatal("an ordinary failure was taken for a GPU fault")
	}
}

func TestPlanFaultMarkers(t *testing.T) {
	// the line the 2026-10-19 19:45 run ended with
	if m, ok := planFault([]string{"strata generate: expert cache auto", "strata serve: the GPU has too little free VRAM for the prompt path: turn images off"}); !ok || m != "the GPU has too little free VRAM for the prompt path" {
		t.Fatalf("the prompt-path failure was not recognised: %q %v", m, ok)
	}
	if _, ok := planFault([]string{"strata serve: prefill: device buffers for a chunk of 512 tokens do not fit (42 of 7807 MiB free)"}); ok {
		// a request's own chunk can fail to fit while the engine keeps serving: not evidence the plan is broken
		t.Fatal("a per-request prefill line was taken for the plan failing")
	}
	if _, ok := planFault([]string{"strata serve: the prompt path's own buffers do not fit on CUDA0: 162 MiB free"}); !ok {
		t.Fatal("the new prompt-path line was not recognised")
	}
	if _, ok := planFault([]string{"boom: out of memory"}); ok {
		t.Fatal("an ordinary failure was taken for a plan failure")
	}
	if _, ok := planFault([]string{"prefill copy_i32: unspecified launch failure"}); ok {
		t.Fatal("a GPU fault was taken for a plan failure")
	}
}

// A VRAM plan that cannot fit is arithmetic, not a bad run: the same failure twice stops the restarts (the 19:45 run
// reloaded the model four times for it) and the reason says which flags free the VRAM.
func TestRepeatedPlanFailureStopsRestarts(t *testing.T) {
	dir := t.TempDir()
	script, counter := crashingEngine(t, dir, "echo 'strata serve: the GPU has too little free VRAM for the prompt path: turn images off' >&2\nexit 1\n")
	eo := NewEngineOrchestrator(EngineConfig{BinaryPath: script, Args: []string{"x"}, Cwd: dir, RestartDelay: 10 * time.Millisecond})
	if err := eo.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	defer eo.Stop()
	time.Sleep(1500 * time.Millisecond)
	if n := starts(counter); n != 2 {
		t.Fatalf("started %d times, want 2 (the first run and one retry)", n)
	}
	if st, why := eo.Status(); st != StateFailed || !strings.Contains(why, "VRAM plan") {
		t.Fatalf("state %v, reason %q", st, why)
	}
	logText, _ := os.ReadFile(filepath.Join(dir, "strata-engine-failures.log"))
	if !strings.Contains(string(logText), "--kv-resident") || !strings.Contains(string(logText), "not restarting") {
		t.Fatalf("failure log does not say what to change:\n%s", logText)
	}
}

// The same GPU fault twice in a row stops the restarts after the one retry, instead of reloading the model eight times.
func TestRepeatedGPUFaultStopsRestarts(t *testing.T) {
	dir := t.TempDir()
	script, counter := crashingEngine(t, dir, "echo 'strata generate: native embedding: cannot pin 322 MiB, nor place it in VRAM' >&2\nexit 1\n")
	eo := NewEngineOrchestrator(EngineConfig{BinaryPath: script, Args: []string{"x"}, Cwd: dir, RestartDelay: 10 * time.Millisecond})
	if err := eo.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	defer eo.Stop()
	time.Sleep(1500 * time.Millisecond)
	if n := starts(counter); n != 2 {
		t.Fatalf("started %d times, want 2 (the first run and one retry)", n)
	}
	if st, why := eo.Status(); st != StateFailed || !strings.Contains(why, "GPU is unusable") {
		t.Fatalf("state %v, reason %q", st, why)
	}
	logText, _ := os.ReadFile(filepath.Join(dir, "strata-engine-failures.log"))
	if !strings.Contains(string(logText), "not restarting") {
		t.Fatalf("failure log does not say why it stopped:\n%s", logText)
	}
}

// A GPU fault on crash 1 must not count against crash 2 if crash 2 is unrelated: logTail is cleared on each Start.
func TestGPUFaultFromPreviousCrashDoesNotTriggerUnusable(t *testing.T) {
	dir := t.TempDir()
	run := filepath.Join(dir, "run")
	script := filepath.Join(dir, "engine.sh")
	body := "#!/bin/sh\n" +
		"n=$(cat " + run + " 2>/dev/null || echo 0)\n" +
		"echo $((n+1)) > " + run + "\n" +
		`if [ "$n" = "0" ]; then` + "\n" +
		"  echo 'prefill: unspecified launch failure' >&2\n" +
		"else\n" +
		"  echo 'plain startup failure' >&2\n" +
		"fi\n" +
		"exit 1\n"
	if err := os.WriteFile(script, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	eo := NewEngineOrchestrator(EngineConfig{BinaryPath: script, Args: []string{"x"}, Cwd: dir, RestartDelay: 10 * time.Millisecond})
	if err := eo.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	defer eo.Stop()
	time.Sleep(500 * time.Millisecond)
	if _, why := eo.Status(); strings.Contains(why, "GPU is unusable") {
		t.Fatalf("a plain second failure was incorrectly treated as a repeated GPU fault: %q", why)
	}
}

// fakeServingEngine starts an orchestrator on a shell script that prints READY, then runs `afterGen` once a GEN line
// arrives. Returns the events of one request.
func fakeServingEngine(t *testing.T, preamble, afterGen string) []TokenEvent {
	t.Helper()
	dir := t.TempDir()
	script := filepath.Join(dir, "engine.sh")
	body := "#!/bin/sh\n" + preamble + "echo 'READY 4096 stop'\nread gen\n" + afterGen
	if err := os.WriteFile(script, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	eo := NewEngineOrchestrator(EngineConfig{BinaryPath: script, Args: []string{"x"}, Cwd: dir, NoAutoRestart: true, StartTimeout: time.Minute})
	eo.SetTokenizer(NewByteTokenizer())
	if err := eo.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	defer eo.Stop()
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	if err := eo.WaitReady(ctx); err != nil {
		t.Fatal(err)
	}
	out := make(chan TokenEvent, 64)
	go eo.GenerateIDs(ctx, []int{1, 2}, SamplingParams{}, 8, nil, nil, out)
	var evs []TokenEvent
	for ev := range out {
		evs = append(evs, ev)
	}
	return evs
}

// An engine that dies during a request: the error must carry why (the GPU fault in its output), not "ended".
func TestEngineDeathMidRequestReportsTheReason(t *testing.T) {
	evs := fakeServingEngine(t, "", "echo 'prefill copy_i32: unspecified launch failure' >&2\nexit 1\n")
	last := evs[len(evs)-1]
	if last.Error == nil || !strings.Contains(last.Error.Error(), "unspecified launch failure") {
		t.Fatalf("the error does not carry the engine's reason: %+v", last)
	}
	var nr *ErrNotReady
	if !errors.As(last.Error, &nr) || nr.State != StateFailed {
		t.Fatalf("state in the error = %+v, want failed", last.Error)
	}
}

// A token line that cannot be read is a hole in the reply: an error, not a clean stop.
func TestUnreadableTokenLineIsAnError(t *testing.T) {
	evs := fakeServingEngine(t, "", "echo 'T 65'\necho 'T xyz'\necho 'T 66'\necho 'DONE 3 2 1 1 stop'\n")
	last := evs[len(evs)-1]
	if last.Error == nil || !strings.Contains(last.Error.Error(), "unreadable token line") {
		t.Fatalf("an unreadable token line was not reported: %+v", last)
	}
}

// A stderr line over the scanner's limit must not stop the draining: the engine would block on its next write.
func TestOversizedStderrLineDoesNotWedgeTheEngine(t *testing.T) {
	preamble := "head -c 2000000 /dev/zero | tr '\\0' x >&2\necho >&2\nfor i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do head -c 65536 /dev/zero | tr '\\0' y >&2; echo >&2; done\n"
	evs := fakeServingEngine(t, preamble, "echo 'T 65'\necho 'DONE 1 2 1 1 stop'\n")
	last := evs[len(evs)-1]
	if last.Error != nil || !last.IsEnd {
		t.Fatalf("request after a huge stderr line: %+v", last)
	}
}

// The failure log holds the engine's last output lines: owner-only (an older file too), and it does not grow forever.
func TestFailureLogIsOwnerOnlyAndRotates(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "strata-engine-failures.log")
	if err := os.WriteFile(path, []byte("old\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	eo := NewEngineOrchestrator(EngineConfig{Cwd: dir})
	eo.appendFailureLog("first\n")
	if st, _ := os.Stat(path); st.Mode().Perm() != 0o600 {
		t.Fatalf("mode = %v, want 0600", st.Mode().Perm())
	}
	if err := os.WriteFile(path, make([]byte, maxFailureLogBytes), 0o600); err != nil {
		t.Fatal(err)
	}
	eo.appendFailureLog("after the cap\n")
	if st, err := os.Stat(path + ".1"); err != nil || st.Size() != maxFailureLogBytes {
		t.Fatalf("the full log was not rotated to .1: %v", err)
	}
	if b, _ := os.ReadFile(path); string(b) != "after the cap\n" {
		t.Fatalf("the new log = %q", b)
	}
}
