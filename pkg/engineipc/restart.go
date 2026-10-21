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
// reset, a UVM fatal error), not that the engine is misconfigured. Reloading the model into a dead GPU cannot work.
var gpuFaultMarkers = []string{
	"unspecified launch failure",
	"illegal memory access",          // UVM fatal error 0x60 surfaces as CUDA error 700 in the engine's stderr
	"Unable to determine the device", // nvidia-smi / driver lost the device handle (a fall-off needs a power-off, see gpuRemedy)
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

// planFailMarkers are engine messages that say the run's own VRAM plan does not fit this card. Not a GPU fault (no
// snapshot, no "unusable" reason) and not something a restart can change: the plan is arithmetic over the card's size
// and the flags, so a restart reloads a 25 GB model to compute the same numbers and fail in the same place. On
// 2026-10-19 19:45 the engine failed this way four times in 1 m 45 s (restarts 1/8 to 4/8, finding 21) and the 09:53
// log before it shows five more; the message a user needs at that point is which flag frees the VRAM, not another
// 503 loop.
// Both lines are printed only by the startup path that gives up on the prompt path; `prefill: device buffers for a
// chunk of ... do not fit` is NOT one of them, because a request whose own chunk does not fit prints it and the
// engine keeps serving afterwards (prefill.cpp `relayout`).
var planFailMarkers = []string{
	"the GPU has too little free VRAM for the prompt path",
	"the prompt path's own buffers do not fit",
}

// planFault reports whether the engine's last output shows its VRAM plan does not fit, and which message it was.
func planFault(tail []string) (string, bool) {
	for i := len(tail) - 1; i >= 0; i-- {
		for _, m := range planFailMarkers {
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
		eo.planFailsInRow = 0
	}
	eo.failsInRow++
	marker, faulted := gpuFault(eo.logTail)
	if faulted {
		eo.gpuFaultsInRow++
	} else {
		eo.gpuFaultsInRow = 0
	}
	planMarker, planBroken := planFault(eo.logTail)
	if planBroken {
		eo.planFailsInRow++
	} else {
		eo.planFailsInRow = 0
	}
	attempt := eo.failsInRow
	note := "auto restart is off"
	restart := !eo.cfg.NoAutoRestart && os.Getenv("STRATA_NO_AUTO_RESTART") != "1"
	var delay time.Duration
	if restart && eo.gpuFaultsInRow >= 2 {
		// One restart is worth trying (a single launch failure can leave only that process's context dead); the same
		// fault again in a fresh process means the GPU is gone, and reloading a 25 GB model into it just repeats it.
		restart = false
		note = fmt.Sprintf("the GPU is unusable (%q twice in a row): not restarting; the failure log names the cause (a card that stopped answering needs the PC powered fully off, not a driver reload); then the next request starts it automatically", marker)
		eo.reason = "the GPU is unusable (" + marker + "): " + why
	} else if restart && eo.planFailsInRow >= 2 {
		// One restart is worth trying (a process can die before it prints the fuller picture).  The same plan failure in
		// a fresh process is the same arithmetic over the same card: say what to change instead of reloading the model.
		restart = false
		note = fmt.Sprintf("the VRAM plan does not fit this card (%q twice in a row): not restarting - free VRAM (close "+
			"other GPU users, --vision no) or ask the plan for less (--kv-resident, --max-context, --vram-reserve-mib); "+
			"the next request starts it", planMarker)
		eo.reason = "the engine's VRAM plan does not fit this card: " + why
	} else if restart && attempt > maxRestarts {
		restart = false
		note = fmt.Sprintf("giving up after %d failures in a row; the next request starts it", maxRestarts)
	} else if restart {
		delay = restartDelay(eo.cfg.RestartDelay, attempt)
		note = fmt.Sprintf("restart %d/%d in %s", attempt, maxRestarts, delay)
	}
	eo.writeFailureLog(why, note)
	if !restart {
		if faulted {
			eo.startGPUSnapshot("after the engine died, not restarting") // off the lock: nvidia-smi may hang
		}
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
		snap := eo.writeGPUSnapshot(fmt.Sprintf("before restart %d/%d", attempt, maxRestarts))
		// The engine died on a GPU fault and the driver says it lost a card: loading a model into a card in that
		// state repeats the fault, so the restart waits for a request (which starts the engine, as before).
		if line, lost := deviceLost(snap); faulted && lost {
			msg := "not restarting: the driver reports a lost or faulted GPU (" + line + "); the next request starts the engine"
			if r := gpuRemedy(snap); r != "" {
				msg = "not restarting: the driver reports a lost or faulted GPU (" + line + "); " + r + "; then the next request starts the engine"
			}
			reason := "the GPU is unusable: " + why
			if s := gpuFaultSummary(snap); s != "" { // name the card and the Xid, not only the engine's own text
				reason += "; " + s
			}
			eo.mu.Lock()
			eo.reason = reason
			eo.mu.Unlock()
			eo.writeFailureLine(msg)
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
