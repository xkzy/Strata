// pkg/server/tuning.go - the startup tuner's state, /metrics view and idle-time refinement.
package server

import (
	"context"
	"fmt"
	"sync"
	"time"

	"strata/pkg/autotune"
	"strata/pkg/engineipc"
)

type tuningState struct {
	mu      sync.RWMutex
	have    bool
	caps    autotune.Capabilities
	wl      autotune.Workload
	plan    autotune.Plan
	meas    autotune.Measurements
	applied []string
	saved   bool
	notes   []string
	at      time.Time
	engine  bool // the last pass measured the running engine
	// refining keeps an engine restart from starting a second background pass while the previous one is still
	// waiting for an idle server.
	refining bool
}

func (s *StrataServer) setTuning(res autotune.StartupResult, enginePass bool) {
	s.tuning.mu.Lock()
	defer s.tuning.mu.Unlock()
	s.tuning.have = true
	s.tuning.caps, s.tuning.wl = res.Capabilities, res.Workload
	s.tuning.plan, s.tuning.meas = res.Plan, res.Measurements
	s.tuning.applied, s.tuning.saved = res.AppliedArgs, res.Saved
	s.tuning.notes, s.tuning.at = res.Notes, time.Now()
	s.tuning.engine = enginePass
}

// tuningView is the /metrics "tuning" section. It is empty ({"available": false}) until a pass has run, so a
// failure of the tuner cannot break /metrics.
func (s *StrataServer) tuningView() map[string]interface{} {
	s.tuning.mu.RLock()
	defer s.tuning.mu.RUnlock()
	if !s.tuning.have {
		return map[string]interface{}{"available": false}
	}
	sel := s.tuning.plan.Selected
	eng := "host"
	if s.tuning.engine {
		eng = "engine"
	}
	out := map[string]interface{}{
		"available":      true,
		"pass":           eng,
		"fingerprint":    s.tuning.plan.Fingerprint,
		"objective":      s.tuning.plan.Objective,
		"classification": s.tuning.plan.Classification,
		"coverage":       s.tuning.plan.Coverage,
		"gap":            s.tuning.plan.Gap,
		"evaluated":      s.tuning.plan.Evaluated,
		"feasible":       s.tuning.plan.Feasible,
		"noise":          s.tuning.plan.Noise,
		"selected": map[string]interface{}{
			"placement": sel.Placement, "devices": sel.Devices, "pool_workers": sel.PoolWorkers,
			"pcie_frac": sel.PCIeFrac, "spec_min_p": sel.SpecMinP, "batch": sel.Batch,
		},
		"prediction": map[string]interface{}{
			"decode_tok_s": s.tuning.plan.Prediction.DecodeTokS, "prefill_tok_s": s.tuning.plan.Prediction.PrefillTokS,
			"ttft_ms": s.tuning.plan.Prediction.TTFTms, "basis": s.tuning.plan.Prediction.Basis,
			"bound": s.tuning.plan.Prediction.Bound,
		},
		"evidence":     s.tuning.plan.Evidence,
		"record_saved": s.tuning.saved,
		"at":           float64(s.tuning.at.UnixNano()) / 1e9,
		"notes":        s.tuning.notes,
		"applied_args": s.tuning.applied,
	}
	if s.tuning.caps.Memory.TotalBytes > 0 {
		out["memory_pressure"] = s.tuning.caps.Memory.Pressure
	}
	return out
}

// refineTuning waits for the engine to become ready, then runs one small, bounded engine measurement while the
// server is idle. It never changes the running engine's arguments; its record is used at the next start.
func (s *StrataServer) refineTuning(ctx context.Context, recordPath string, args []string, geom autotune.Geometry, maxContext int) {
	if s.EngineIPC == nil || s.Config.Port <= 0 {
		return
	}
	s.tuning.mu.Lock()
	if s.tuning.refining {
		s.tuning.mu.Unlock()
		return
	}
	s.tuning.refining = true
	s.tuning.mu.Unlock()
	defer func() {
		s.tuning.mu.Lock()
		s.tuning.refining = false
		s.tuning.mu.Unlock()
	}()
	tick := time.NewTicker(time.Second)
	defer tick.Stop()
	deadline := time.After(10 * time.Minute)
ready:
	for {
		select {
		case <-ctx.Done():
			return
		case <-deadline:
			return
		case <-tick.C:
			st, _ := s.EngineIPC.Status()
			if st == engineipc.StateReady {
				break ready
			}
			if st == engineipc.StateFailed || st == engineipc.StateStopped {
				return
			}
		}
	}
	// Only measure an idle server: a real request arriving first wins. If the server stays busy for 30 s the pass
	// is skipped; it must never add load to a serving machine.
	idle := false
	for i := 0; i < 30 && !idle; i++ {
		live := s.mon.live(true)
		idle = live["state"] == "idle" && live["queued"] == 0
		if idle {
			break
		}
		select {
		case <-ctx.Done():
			return
		case <-tick.C:
		}
	}
	if !idle {
		return
	}
	pctx, cancel := context.WithTimeout(ctx, 45*time.Second)
	defer cancel()
	res := autotune.Startup(pctx, autotune.StartupOptions{
		Args: args, RecordPath: recordPath, MaxContext: maxContext, Geometry: geom, ModelName: s.Config.ModelName,
		Envelope: autotune.Envelope{PromptTokens: 64, OutputTokens: 8, Concurrency: 1, Source: "startup-refinement"}, Engine: true, BaseURL: fmt.Sprintf("http://127.0.0.1:%d", s.Config.Port), APIKey: s.Config.APIKey,
		Model: s.Config.ModelName, Measure: true,
	})
	s.setTuning(res, true)
}
