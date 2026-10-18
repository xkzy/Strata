// pkg/engineipc/restart.go - what happens after the engine process dies: the failure is written to a log file and
// the engine is started again, with a back-off, so a crash (an out-of-memory kill, a driver reset) does not leave
// the server answering "not ready" until someone restarts it by hand.
package engineipc

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

const (
	defaultRestartDelay = 3 * time.Second
	maxRestartDelay     = 2 * time.Minute
	// maxRestarts failures in a row without a stable run stop the restarts (a bad config or a missing GPU would
	// otherwise reload a 25 GB model forever); a request still starts the engine, as before.
	maxRestarts = 8
	// stableRun: an engine that served this long since READY is healthy again, the back-off starts over.
	stableRun = time.Minute
	// maxFailureLogBytes: the failure log is rotated to <name>.1 once it reaches this size.
	maxFailureLogBytes = 1 << 20
)

func (eo *EngineOrchestrator) failureLogPath() string {
	if eo.cfg.FailureLog != "" {
		return eo.cfg.FailureLog
	}
	return filepath.Join(eo.cfg.Cwd, "strata-engine-failures.log")
}

// restartDelay is the wait before restart number `attempt` (1-based): RestartDelay, doubled for each earlier one.
func restartDelay(base time.Duration, attempt int) time.Duration {
	if base <= 0 {
		base = defaultRestartDelay
	}
	d := base
	for i := 1; i < attempt && d < maxRestartDelay; i++ {
		d *= 2
	}
	if d > maxRestartDelay {
		d = maxRestartDelay
	}
	return d
}

// gpuFaultMarkers are engine/driver messages that say the GPU itself is unusable (it fell off the bus, a driver
// reset), not that the engine is misconfigured. Reloading the model into a dead GPU cannot work.
var gpuFaultMarkers = []string{
	"unspecified launch failure",
	"nor place it in VRAM",
	"no CUDA-capable device",
	"fallen off the bus",
	"uncorrectable ECC",
}

// gpuFault reports whether the engine's last output shows a dead GPU, and which message it was.
func gpuFault(tail []string) (string, bool) {
	for i := len(tail) - 1; i >= 0; i-- {
		for _, m := range gpuFaultMarkers {
			if strings.Contains(tail[i], m) {
				return m, true
			}
		}
	}
	return "", false
}

// afterFailure runs from watchProcess with eo.mu held, once the process `cmd` has ended and the state is Failed.
// It records the failure and, unless auto restart is off, schedules the restart.
func (eo *EngineOrchestrator) afterFailure(cmd *exec.Cmd, why string) {
	if !eo.readyAt.IsZero() && time.Since(eo.readyAt) >= stableRun {
		eo.failsInRow = 0
		eo.gpuFaultsInRow = 0
	}
	eo.failsInRow++
	marker, faulted := gpuFault(eo.logTail)
	if faulted {
		eo.gpuFaultsInRow++
	} else {
		eo.gpuFaultsInRow = 0
	}
	attempt := eo.failsInRow
	note := "auto restart is off"
	restart := !eo.cfg.NoAutoRestart && os.Getenv("STRATA_NO_AUTO_RESTART") != "1"
	var delay time.Duration
	if restart && eo.gpuFaultsInRow >= 2 {
		// One restart is worth trying (a single launch failure can leave only that process's context dead); the same
		// fault again in a fresh process means the GPU is gone, and reloading a 25 GB model into it just repeats it.
		restart = false
		note = fmt.Sprintf("the GPU is unusable (%q twice in a row): not restarting; reset or reboot, or run on a healthy card (CUDA_VISIBLE_DEVICES); the next request starts it", marker)
		eo.reason = "the GPU is unusable (" + marker + "): " + why
	} else if restart && attempt > maxRestarts {
		restart = false
		note = fmt.Sprintf("giving up after %d failures in a row; the next request starts it", maxRestarts)
	} else if restart {
		delay = restartDelay(eo.cfg.RestartDelay, attempt)
		note = fmt.Sprintf("restart %d/%d in %s", attempt, maxRestarts, delay)
	}
	eo.writeFailureLog(why, note)
	if !restart {
		return
	}
	time.AfterFunc(delay, func() {
		eo.mu.RLock()
		// only if nothing else happened meanwhile: still this dead process, not stopped, not started by a request
		again := eo.state == StateFailed && eo.cmd == cmd
		eo.mu.RUnlock()
		if !again {
			return
		}
		if err := eo.Start(context.Background()); err != nil {
			log.Printf("strata: engine restart failed: %v", err)
			eo.writeFailureLine("engine restart failed: " + err.Error())
		} else {
			log.Printf("strata: engine restarted (attempt %d)", attempt)
		}
	})
}

// writeFailureLog appends one entry; the caller holds eo.mu (it reads logTail).
func (eo *EngineOrchestrator) writeFailureLog(why, note string) {
	var b strings.Builder
	fmt.Fprintf(&b, "%s engine failed (%s): %s\n", time.Now().Format(time.RFC3339), note, why)
	tail := eo.logTail
	if len(tail) > 20 {
		tail = tail[len(tail)-20:]
	}
	for _, l := range tail {
		b.WriteString("    | " + l + "\n")
	}
	eo.appendFailureLog(b.String())
}

func (eo *EngineOrchestrator) writeFailureLine(msg string) {
	eo.appendFailureLog(fmt.Sprintf("%s %s\n", time.Now().Format(time.RFC3339), msg))
}

func (eo *EngineOrchestrator) appendFailureLog(text string) {
	log.Printf("strata: %s", strings.TrimRight(strings.SplitN(text, "\n", 2)[0], "\n"))
	path := eo.failureLogPath()
	// One older generation is kept: a crash loop must not grow this file without end.
	if st, err := os.Stat(path); err == nil && st.Size() >= maxFailureLogBytes {
		if err := os.Rename(path, path+".1"); err != nil {
			log.Printf("strata: cannot rotate %s: %v", path, err)
		}
	}
	// It holds the engine's last output lines (file paths, GPU messages): owner-only, also for a file made earlier.
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		log.Printf("strata: cannot write %s: %v", path, err)
		return
	}
	defer f.Close()
	_ = f.Chmod(0o600)
	if _, err := f.WriteString(text); err != nil {
		log.Printf("strata: cannot write %s: %v", path, err)
	}
}
