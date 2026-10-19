// pkg/server/runtime.go - the server half of the runtime controller.
//
// It samples the telemetry collector once a second, turns the sample into a resourcemanager.Sample, and lets the
// controller run its fast/medium/slow loops. The loop never blocks a request: admission is a short lock around
// counters, and retuning is a background goroutine that is single-flighted.
package server

import (
	"context"
	"fmt"
	"strconv"
	"sync"
	"time"

	"strata/pkg/autotune"
	"strata/pkg/engineipc"
	"strata/pkg/resourcemanager"
)

type runtimeTuneInfo struct {
	mu     sync.Mutex
	record string
	args   []string
	geom   autotune.Geometry
	maxCtx int
	have   bool
}

func (s *StrataServer) setRuntimeTuneInfo(record string, args []string, geom autotune.Geometry, maxCtx int) {
	s.tuneInfo.mu.Lock()
	s.tuneInfo.record = record
	s.tuneInfo.args = append([]string(nil), args...)
	s.tuneInfo.geom = geom
	s.tuneInfo.maxCtx = maxCtx
	s.tuneInfo.have = true
	s.tuneInfo.mu.Unlock()
}

func (s *StrataServer) runtimeTuneSnapshot() (string, []string, autotune.Geometry, int, bool) {
	s.tuneInfo.mu.Lock()
	defer s.tuneInfo.mu.Unlock()
	return s.tuneInfo.record, append([]string(nil), s.tuneInfo.args...), s.tuneInfo.geom, s.tuneInfo.maxCtx, s.tuneInfo.have
}

// StartRuntime starts the 1 Hz adaptation loop. It is a no-op when resource_adapt is off, and it is safe to call
// more than once. The loop stops when ctx ends or StopRuntime is called.
func (s *StrataServer) StartRuntime(ctx context.Context) {
	if s.ResourceManager == nil || s.ResourceManager.Adapt == nil || !s.ResourceManager.Adapt.Enabled() {
		return
	}
	s.runtimeOnce.Do(func() {
		rctx, cancel := context.WithCancel(ctx)
		s.cancelMu.Lock()
		s.runtimeCancel = cancel
		s.cancelMu.Unlock()
		if s.Telemetry != nil {
			s.Telemetry.Start()
		}
		go s.runtimeLoop(rctx)
	})
}

// StopRuntime stops the adaptation loop and the 1 Hz telemetry sampler.
func (s *StrataServer) StopRuntime() {
	s.cancelMu.Lock()
	cancel := s.runtimeCancel
	s.runtimeCancel = nil
	s.cancelMu.Unlock()
	if cancel != nil {
		cancel()
	}
	if s.lifecycleCancel != nil {
		s.lifecycleCancel() // stop a startup/refinement measurement that is still running
	}
	if s.Telemetry != nil {
		s.Telemetry.Stop()
	}
}

func (s *StrataServer) runtimeLoop(ctx context.Context) {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case now := <-ticker.C:
			if s.ResourceManager == nil || s.ResourceManager.Adapt == nil {
				return
			}
			s.ResourceManager.Adapt.Observe(s.runtimeSample(now))
			if s.ResourceManager.Adapt.RetuneDue() {
				s.maybeRetune(ctx) // only clears the request when the pass actually starts
			}
		}
	}
}

// runtimeSample copies the telemetry collector's current values. A missing value stays unknown; nothing here
// invents a zero that would look like a safe measurement.
func (s *StrataServer) runtimeSample(now time.Time) resourcemanager.Sample {
	st := resourcemanager.Sample{At: now}
	if s.Telemetry != nil {
		applyTelemetry(&st, s.Telemetry.Snapshot())
	}
	ws := s.mon.recentStats(10 * time.Second)
	st.P95TTFTms = ws.TTFTmsP95
	st.P95LatencyMS = ws.LatencySP95 * 1000
	st.DecodeTokS = ws.DecodeTokSMean
	st.PromptP95, st.OutputP95 = ws.PromptP95, ws.OutputP95
	live := s.mon.live(true)
	if v, ok := live["running"].(int); ok {
		st.InFlight = v
	}
	if s.EngineIPC != nil {
		state, reason := s.EngineIPC.Status()
		st.EngineState = state.String()
		if state == engineipc.StateFailed {
			st.DeviceFailed = true
			st.HealthReason = reason
		}
		if state == engineipc.StateReady {
			applyEnginePlan(&st, s.EngineIPC.Info())
		}
	}
	return st
}

// applyEnginePlan records what the engine's ready-time plan counted on. Its INFO line's vram_free_mib is VRAM left
// free once every tier, the KV cache and the prompt-path buffers were allocated, so a met plan says the used
// percentage is the plan rather than pressure. Only a ready engine is read: after a restart the previous plan's
// permission must not survive into the new one, and an engine that has not printed INFO yet has no plan to trust.
func applyEnginePlan(st *resourcemanager.Sample, info map[string]string) {
	mib, err := strconv.Atoi(info["vram_free_mib"])
	if err != nil || mib < 0 {
		return
	}
	st.KnownVRAMPlan = true
	st.VRAMPlanFreeByte = uint64(mib) << 20
}

// applyTelemetry copies the collector's current values into the sample. Every value is optional and stays
// unknown when the collector does not have it: a GPU whose driver stopped answering publishes no used/util/power,
// and reading that absence as "0% VRAM" would be a false permission to grow. RAM only counts when the collector
// measured system memory (on a host without a system-wide reading it reports the Go heap, which is not host RAM).
func applyTelemetry(st *resourcemanager.Sample, snap map[string]interface{}) {
	cur, ok := snap["now"].(map[string]interface{})
	if !ok {
		return
	}
	if v, ok := cur["cpu_util"].(float64); ok {
		st.KnownCPU, st.CPUPercent = true, v
	}
	ramTotal, _ := cur["ram_total"].(uint64)
	ramUsed, _ := cur["ram_used"].(uint64)
	ramSystem, _ := cur["ram_system"].(bool)
	if ramSystem && ramTotal > 0 {
		st.KnownRAM = true
		st.RAMUsedPercent = float64(ramUsed) / float64(ramTotal) * 100
		if ramUsed <= ramTotal {
			st.RAMAvailableByte = ramTotal - ramUsed
		}
	}
	vramTotal, hasTotal := cur["gpu_mem_total"].(uint64)
	vramUsed, hasUsed := cur["gpu_mem_used"].(uint64)
	if hasTotal && hasUsed && vramTotal > 0 {
		st.KnownVRAM = true
		st.VRAMUsedPercent = float64(vramUsed) / float64(vramTotal) * 100
		if vramUsed <= vramTotal {
			st.VRAMFreeBytes = vramTotal - vramUsed
		}
	}
	if v, ok := cur["gpu_temp"].(float64); ok {
		st.GPUTempC = v
	}
}

// admitRequest asks the runtime controller for a slot. The request waits in the controller's bounded queue; a
// closed admission, a full queue or a device fault returns an explicit 429 instead of starting more work.
func (s *StrataServer) admitRequest(ctx context.Context, promptTokens int) (uint64, *apiError) {
	if s.ResourceManager == nil || s.ResourceManager.Adapt == nil {
		return 0, nil
	}
	a := s.ResourceManager.Adapt.Admit(ctx, promptTokens)
	if a.Admitted {
		return a.Ticket, nil
	}
	msg := "the server is at capacity: " + a.Reason
	if a.RetryAfter > 0 {
		msg = fmt.Sprintf("%s (retry after %d s)", msg, a.RetryAfter)
	}
	return 0, &apiError{Status: 429, Type: "server_overloaded", Msg: msg, RetryAfter: a.RetryAfter}
}

func (s *StrataServer) releaseRequest(ticket uint64) {
	if s.ResourceManager == nil || s.ResourceManager.Adapt == nil {
		return
	}
	s.ResourceManager.Adapt.Release(ticket, time.Now())
}

// maybeRetune runs one bounded engine measurement after a slow-loop workload shift. It never runs during active
// work, never overlaps another refinement, and never changes the running engine's arguments. The pending request
// survives a busy evaluation; it is cleared only when a pass is actually launched, or when tuning is unavailable.
func (s *StrataServer) maybeRetune(ctx context.Context) {
	if s.ResourceManager == nil || s.ResourceManager.Adapt == nil || s.EngineIPC == nil {
		return
	}
	if !s.Config.AutoTune {
		s.ResourceManager.Adapt.TakeRetuneDue()
		return
	}
	if !s.ResourceManager.Adapt.Knobs().Background {
		return // busy: try again when the medium loop sees the server idle
	}
	if state, _ := s.EngineIPC.Status(); state != engineipc.StateReady {
		return // starting, restarted or failed: a later healthy sample decides
	}
	if live := s.mon.live(true); live["state"] != "idle" {
		return
	}
	s.tuning.mu.RLock()
	refining := s.tuning.refining
	s.tuning.mu.RUnlock()
	if refining {
		return // another bounded pass owns the engine measurement
	}
	record, args, geom, maxCtx, ok := s.runtimeTuneSnapshot()
	if !ok {
		// No startup pass ran in this mode, so there is nothing to refine; do not retry forever.
		s.ResourceManager.Adapt.TakeRetuneDue()
		return
	}
	if !s.ResourceManager.Adapt.TakeRetuneDue() {
		return
	}
	go s.refineTuning(ctx, record, args, geom, maxCtx)
}
