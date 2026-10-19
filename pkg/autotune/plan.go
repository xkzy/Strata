// pkg/autotune/plan.go - the startup workflow.
//
//	discover (caller) -> load hints -> bounded probe -> solve -> validate -> persist -> apply
//
// Any step can be skipped without failing startup. A record from a previous run is a hint: it is re-checked
// against the current fingerprint and constraints, and a measured CPU curve always replaces its CPU numbers.
package autotune

import (
	"context"
	"fmt"
	"strings"
	"time"
)

// PlanOptions configures PlanStartup.
type PlanOptions struct {
	Geometry   Geometry
	Envelope   Envelope
	MaxContext int
	RecordPath string
	Objective  string
	Now        time.Time

	Probe         Probe // nil: solve from hints/model only
	Limits        Limits
	MaxCandidates int  // cap on candidates the probe runs (0: the whole domain)
	Save          bool // persist the result
	Measure       bool // run the probe even when a record is usable
}

// PlanResult is the whole workflow's outcome. Err is informational: the caller always gets a usable Plan.
type PlanResult struct {
	Plan         Plan         `json:"plan"`
	Measurements Measurements `json:"measurements"`
	Bench        Result       `json:"bench"`
	Record       Record       `json:"record"`
	Saved        bool         `json:"saved"`
	Notes        []string     `json:"notes,omitempty"`
	Err          error        `json:"-"`
}

// PlanStartup runs the workflow. It never returns an error that should stop the server: when nothing can be
// measured, the conservative default candidate is still returned.
func PlanStartup(ctx context.Context, caps Capabilities, opts PlanOptions) PlanResult {
	now := opts.Now
	if now.IsZero() {
		now = time.Now()
	}
	wl := NewWorkload(opts.Geometry, opts.Envelope, opts.MaxContext)
	if wl.Envelope.Source == "" {
		wl.Envelope = DefaultEnvelope(wl.MaxContext)
	}
	cons := DefaultConstraints(caps, wl)
	res := PlanResult{}

	var hints Measurements
	var prev *Candidate
	var prevRec *Record
	if prevPathUsable(opts.RecordPath, caps, wl, now, &hints, &prev, &prevRec) {
		res.Notes = append(res.Notes, "a previous tuning record matched this hardware and model; its measurements are hints")
	}

	m := hints
	if opts.Probe != nil && (opts.Measure || !m.Valid()) {
		domain := BuildDomain(caps, wl, m)
		candidates, truncated := domain.EnumerateCapped(opts.MaxCandidates)
		// A scoped probe represents exactly one configuration. It must be in the list even when the domain does not
		// contain it yet (the engine's --pcie-frac or --pool-workers had no measured counterpart before this run).
		if sc, ok := opts.Probe.(ScopedCandidates); ok {
			for _, covered := range sc.CoveredCandidates() {
				found := false
				for _, c := range candidates {
					if c.Key() == covered.Key() {
						found = true
						break
					}
				}
				if !found {
					candidates = append(candidates, covered)
				}
			}
		}
		if truncated {
			res.Notes = append(res.Notes, fmt.Sprintf("the probe ran %d of %d candidates", len(candidates), len(domain.Enumerate())))
		}
		lim := opts.Limits
		if lim.Total == 0 {
			lim = DefaultLimits()
		}
		bench := (&Runner{Limits: lim, Probe: opts.Probe, Now: func() time.Time { return now }}).Run(ctx, caps, wl, candidates)
		res.Bench = bench
		m = mergeMeasurements(hints, bench.Measurements)
		if bench.Aborted != "" {
			res.Notes = append(res.Notes, "the benchmark stopped in phase "+bench.Aborted+": "+errString(bench.Err))
		}
		for _, p := range bench.Phases {
			if p.Skipped {
				res.Notes = append(res.Notes, "phase "+p.Phase+" was skipped: "+p.Detail)
			}
		}
	}

	plan := Solve(wl, caps, m, cons, Options{Objective: opts.Objective, Previous: prev, Now: now})
	res.Plan, res.Measurements = plan, m

	if opts.Save && plan.Selected.Placement != "" {
		rec := NewRecord(caps, wl, m, plan, cons, Methodology{
			Tool: "strata/autotune", Probe: probeName(opts.Probe), Limits: opts.Limits,
			Started: res.Bench.Started, Ended: res.Bench.Ended, Phases: res.Bench.Phases,
			Samples: len(res.Bench.Samples), Notes: res.Notes,
		})
		if res.Bench.Aborted != "" {
			rec.Failures = append(rec.Failures, Failure{At: now, Phase: res.Bench.Aborted, Error: errString(res.Bench.Err)})
		}
		// Validation is a smoke test of the selected configuration through the same probe. It is recorded; it is
		// not a numerical comparison, which needs a trusted reference this package does not have.
		if v, detail := validate(ctx, opts.Probe, plan, wl); v {
			rec.Validation = Validation{Validated: true, Correct: true, Detail: detail, At: now}
		} else if detail != "" {
			rec.Validation = Validation{Validated: !strings.HasPrefix(detail, "not run"), Correct: false, Detail: detail, At: now}
		}
		if prevRec != nil && prevRec.Fingerprint == rec.Fingerprint && prevRec.WorkloadHash == rec.WorkloadHash &&
			prevRec.Selected.Key() == rec.Selected.Key() {
			// A cached start does not re-run the smoke test; it must not erase a validation the same configuration
			// already earned. A validation that actually ran and failed is kept as a failure.
			if !rec.Validation.Validated && strings.HasPrefix(rec.Validation.Detail, "not run") && prevRec.Validation.Validated {
				rec.Validation = prevRec.Validation
			}
			// Failure history is diagnostic evidence: keep the first causal errors across runs, newest last.
			rec.Failures = append(append([]Failure(nil), prevRec.Failures...), rec.Failures...)
			if len(rec.Failures) > 10 {
				rec.Failures = rec.Failures[len(rec.Failures)-10:]
			}
		}
		path := opts.RecordPath
		if path == "" {
			path = DefaultRecordPath()
		}
		if err := Save(path, rec); err != nil {
			res.Err = err
			res.Notes = append(res.Notes, "the tuning record was not saved: "+err.Error())
		} else {
			res.Saved = true
			res.Record = rec
		}
	}
	return res
}

// validate runs the smallest possible request through the selected plan. A failure here means the plan is not
// published as validated; it does not change the plan the server already started with.
func validate(ctx context.Context, probe Probe, plan Plan, wl Workload) (bool, string) {
	if probe == nil {
		return false, "not run: no probe"
	}
	if cp, ok := probe.(CorrectnessProbe); ok {
		if err := cp.Correctness(ctx); err != nil {
			return false, "correctness check failed: " + err.Error()
		}
	}
	h, reason, err := probe.Health(ctx)
	if err != nil || h != HealthOK {
		return false, "device health check failed: " + reason + errString(err)
	}
	pctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	prefill, decode, err := probe.Serving(pctx, plan.Selected, Envelope{PromptTokens: 32, OutputTokens: 2})
	if err != nil {
		if err == ErrUnsupported {
			return false, "not run: the probe cannot serve"
		}
		return false, "validation request failed: " + err.Error()
	}
	if prefill <= 0 || decode <= 0 {
		return false, "validation request produced no usable rate"
	}
	return true, fmt.Sprintf("served a 32-token prompt and 2 tokens (%.0f prefill tok/s, %.0f decode tok/s)",
		prefill, decode)
}

// CorrectnessProbe is optional: a probe that can compare outputs against a trusted reference.
type CorrectnessProbe interface {
	Correctness(ctx context.Context) error
}

func prevPathUsable(path string, caps Capabilities, wl Workload, now time.Time, hints *Measurements, prev **Candidate, recOut **Record) bool {
	if path == "" {
		path = DefaultRecordPath()
	}
	rec, err := Load(path)
	if err != nil {
		return false
	}
	if ok, _ := rec.Usable(caps, wl, now, RecordMaxAge); !ok {
		return false
	}
	*hints = rec.Measurements
	(*hints).Source = "record"
	c := rec.Selected
	*prev = &c
	*recOut = &rec
	return true
}

func mergeMeasurements(base, add Measurements) Measurements {
	out := base
	if out.DecodeTokS == nil {
		out.DecodeTokS = map[string]float64{}
	}
	if out.PrefillTokS == nil {
		out.PrefillTokS = map[string]float64{}
	}
	if out.Samples == nil {
		out.Samples = map[string]int{}
	}
	if out.CV == nil {
		out.CV = map[string]float64{}
	}
	for k, v := range add.DecodeTokS {
		out.DecodeTokS[k] = v
	}
	for k, v := range add.PrefillTokS {
		out.PrefillTokS[k] = v
	}
	for k, v := range add.Samples {
		out.Samples[k] = v
	}
	for k, v := range add.CV {
		out.CV[k] = v
	}
	for k, v := range add.GPUMemBandwidthGBs {
		if out.GPUMemBandwidthGBs == nil {
			out.GPUMemBandwidthGBs = map[string]float64{}
		}
		out.GPUMemBandwidthGBs[k] = v
	}
	for k, v := range add.GPUComputeTFLOPS {
		if out.GPUComputeTFLOPS == nil {
			out.GPUComputeTFLOPS = map[string]float64{}
		}
		out.GPUComputeTFLOPS[k] = v
	}
	for k, v := range add.GPULinkGBs {
		if out.GPULinkGBs == nil {
			out.GPULinkGBs = map[string]float64{}
		}
		out.GPULinkGBs[k] = v
	}
	if add.CPUMemBandwidthGBs > 0 {
		out.CPUMemBandwidthGBs = add.CPUMemBandwidthGBs
	}
	if add.CPUComputeGFLOPS > 0 {
		out.CPUComputeGFLOPS = add.CPUComputeGFLOPS
	}
	if len(add.CPUScaling) > 0 {
		out.CPUScaling = add.CPUScaling
	}
	if add.LaunchOverheadMS > 0 {
		out.LaunchOverheadMS = add.LaunchOverheadMS
	}
	if add.SyncOverheadMS > 0 {
		out.SyncOverheadMS = add.SyncOverheadMS
	}
	out.Tested = append(out.Tested, add.Tested...)
	out.MeasuredAt = add.MeasuredAt
	if add.Source != "" {
		if base.Valid() && add.Valid() {
			out.Source = "mixed"
		} else {
			out.Source = add.Source
		}
	}
	return out
}

func probeName(p Probe) string {
	switch p.(type) {
	case *HostProbe:
		return "host"
	case *EngineProbe:
		return "engine"
	case nil:
		return "none"
	}
	return "custom"
}
