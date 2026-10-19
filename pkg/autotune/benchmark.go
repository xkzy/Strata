// pkg/autotune/benchmark.go - bounded startup measurement.
//
// Rules enforced here, not by the probes:
//   - every phase has a time limit and the whole run has one too;
//   - a device whose health is unknown or failed gets no work;
//   - the first fatal error stops all further work and is preserved (there is no retry);
//   - Cleanup runs even after a timeout, with its own short deadline;
//   - reported memory use is checked against explicit limits;
//   - unsupported phases are skipped and listed, never approximated by a synthetic number.
package autotune

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"
)

// ErrUnsupported means a probe cannot measure that phase on this machine. It is not a failure of the run.
var ErrUnsupported = errors.New("measurement unsupported")

// Phase names, stable for /metrics and records.
const (
	PhaseHealth   = "health"
	PhaseWarmup   = "warmup"
	PhaseMemory   = "memory"
	PhaseCompute  = "compute"
	PhaseScaling  = "scaling"
	PhaseTransfer = "transfer"
	PhaseServing  = "serving"
	PhaseCleanup  = "cleanup"
)

// Limits are the safety envelope of a run. Zero time limits mean "no limit" and are only acceptable for an
// explicit `strata tune --unbounded` diagnostic; Start() always sets them.
type Limits struct {
	Total          time.Duration
	Phase          time.Duration
	MaxHostBytes   uint64
	MaxDeviceBytes map[string]uint64
	Repeats        int // measured repeats per candidate (default 1)
}

// DefaultLimits is the startup policy: 45 s total, 8 s per phase, at most 2 GiB of host memory and one repeat.
// It is documented in docs/optimization/startup-benchmark.md.
func DefaultLimits() Limits {
	return Limits{Total: 45 * time.Second, Phase: 8 * time.Second, MaxHostBytes: 2 << 30, Repeats: 1}
}

// Sample is one measurement of one candidate in one phase.
type Sample struct {
	Phase     string             `json:"phase"`
	Candidate Candidate          `json:"candidate"`
	Metrics   map[string]float64 `json:"metrics"`
	Duration  time.Duration      `json:"duration_ns"`
	At        time.Time          `json:"at"`
	Err       string             `json:"error,omitempty"`
}

// PhaseRecord is the timing and outcome of one phase.
type PhaseRecord struct {
	Phase    string        `json:"phase"`
	Duration time.Duration `json:"duration_ns"`
	Skipped  bool          `json:"skipped,omitempty"`
	Detail   string        `json:"detail,omitempty"`
}

// Usage is what a probe says it is holding, so the runner can enforce memory limits.
type Usage struct {
	HostBytes   uint64            `json:"host_bytes,omitempty"`
	DeviceBytes map[string]uint64 `json:"device_bytes,omitempty"`
}

// Probe is the only interface the runner needs. A probe that cannot measure a phase returns ErrUnsupported.
type Probe interface {
	Health(ctx context.Context) (Health, string, error)
	Warmup(ctx context.Context, c Candidate) error
	MemoryBandwidth(ctx context.Context) (float64, error)
	Compute(ctx context.Context, c Candidate) (float64, error)
	Transfer(ctx context.Context, from, to Accelerator) (float64, error)
	Serving(ctx context.Context, c Candidate, e Envelope) (prefillTokS, decodeTokS float64, err error)
	Usage() Usage
	Cleanup(ctx context.Context) error
}

// ScalingProbe measures a CPU thread-scaling curve. Optional.
type ScalingProbe interface {
	Scaling(ctx context.Context, workers []int, perWorker time.Duration) ([]CPUScalePoint, error)
}

// CandidateScoper is implemented by a probe whose serving measurement represents one specific configuration (a
// running engine). The runner then measures no other candidate, because a rate measured on the running engine
// cannot be attributed to a configuration the engine is not executing.
type CandidateScoper interface {
	Covers(c Candidate) bool
}

// ScopedCandidates lets a scoped probe add its own configuration to the probe list when the solver's domain does
// not happen to contain it (for example a PCIe share that only the engine command line records).
type ScopedCandidates interface {
	CoveredCandidates() []Candidate
}

// Result is the outcome of a run. Measurements is valid even when the run aborted: whatever completed is kept.
type Result struct {
	Measurements Measurements  `json:"measurements"`
	Samples      []Sample      `json:"samples,omitempty"`
	Phases       []PhaseRecord `json:"phases,omitempty"`
	Aborted      string        `json:"aborted,omitempty"`
	Err          error         `json:"-"`
	Started      time.Time     `json:"started"`
	Ended        time.Time     `json:"ended"`
	Notes        []string      `json:"notes,omitempty"`
}

// Runner runs probes under Limits. Now is injectable for tests.
type Runner struct {
	Limits Limits
	Probe  Probe
	Now    func() time.Time
}

// Run executes the phases in order. It never retries and always calls Cleanup.
func (r *Runner) Run(ctx context.Context, caps Capabilities, wl Workload, candidates []Candidate) Result {
	now := r.Now
	if now == nil {
		now = time.Now
	}
	res := Result{Started: now(), Measurements: Measurements{Source: "probe", MeasuredAt: now()}}
	if sc, ok := r.Probe.(CandidateScoper); ok {
		kept := make([]Candidate, 0, len(candidates))
		for _, c := range candidates {
			if sc.Covers(c) {
				kept = append(kept, c)
			}
		}
		if skipped := len(candidates) - len(kept); skipped > 0 {
			res.Notes = append(res.Notes, fmt.Sprintf("%d candidate(s) were not measured: the probe represents only one running configuration", skipped))
		}
		candidates = kept
	}
	res.Measurements.DecodeTokS = map[string]float64{}
	res.Measurements.PrefillTokS = map[string]float64{}
	res.Measurements.Samples = map[string]int{}
	res.Measurements.CV = map[string]float64{}
	if r.Probe == nil {
		res.Aborted = PhaseHealth
		res.Err = errors.New("no probe")
		res.Notes = append(res.Notes, "no measurement probe was provided")
		res.Ended = now()
		return res
	}
	runCtx := ctx
	var cancel context.CancelFunc
	if r.Limits.Total > 0 {
		runCtx, cancel = context.WithTimeout(ctx, r.Limits.Total)
		defer cancel()
	}
	repeats := r.Limits.Repeats
	if repeats < 1 {
		repeats = 1
	}
	abort := func(phase string, err error) {
		res.Aborted = phase
		res.Err = err
	}
	// Cleanup always runs, on a fresh deadline: a canceled run still has to release its own resources.
	defer func() {
		cctx, ccancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer ccancel()
		start := now()
		err := r.Probe.Cleanup(cctx)
		res.Phases = append(res.Phases, PhaseRecord{Phase: PhaseCleanup, Duration: now().Sub(start), Detail: errString(err)})
		res.Ended = now()
	}()

	// 1. Health gate.
	start := now()
	h, reason, err := r.Probe.Health(runCtx)
	res.Phases = append(res.Phases, PhaseRecord{Phase: PhaseHealth, Duration: now().Sub(start), Detail: reason})
	if err != nil {
		abort(PhaseHealth, err)
		return res
	}
	if h != HealthOK {
		abort(PhaseHealth, fmt.Errorf("device health is %s: %s", h, reason))
		return res
	}

	// 2. Warm up only the candidates whose devices are healthy (the caller filtered, this re-checks).
	start = now()
	warmed := 0
	for _, c := range candidates {
		if err := runCtx.Err(); err != nil {
			abort(PhaseWarmup, err)
			return res
		}
		if !candidateHealthy(c, caps) {
			continue
		}
		if err := r.phase(runCtx, func(pctx context.Context) error { return r.Probe.Warmup(pctx, c) }); err != nil {
			abort(PhaseWarmup, err)
			return res
		}
		warmed++
	}
	res.Phases = append(res.Phases, PhaseRecord{Phase: PhaseWarmup, Duration: now().Sub(start), Detail: fmt.Sprintf("%d candidate(s)", warmed)})
	if err := r.checkUsage(); err != nil {
		abort(PhaseWarmup, err)
		return res
	}

	// 3. Memory and compute of the host / selected candidates.
	start = now()
	bw, err := measurePhase(runCtx, r, func(pctx context.Context) (float64, error) { return r.Probe.MemoryBandwidth(pctx) })
	if errors.Is(err, ErrUnsupported) {
		res.Phases = append(res.Phases, PhaseRecord{Phase: PhaseMemory, Skipped: true, Detail: "unsupported"})
	} else if err != nil {
		abort(PhaseMemory, err)
		return res
	} else {
		res.Measurements.CPUMemBandwidthGBs = bw
		res.Phases = append(res.Phases, PhaseRecord{Phase: PhaseMemory, Duration: now().Sub(start), Detail: fmt.Sprintf("%.1f GB/s", bw)})
	}

	start = now()
	computed := 0
	computeUnsupported := false
	for _, c := range candidates {
		if !candidateHealthy(c, caps) {
			continue
		}
		gf, err := measurePhase(runCtx, r, func(pctx context.Context) (float64, error) { return r.Probe.Compute(pctx, c) })
		if errors.Is(err, ErrUnsupported) {
			// Unsupported for this candidate, not for the whole phase: the host probe measures CPU but not a
			// device, and the CPU number must still be kept.
			computeUnsupported = true
			continue
		}
		if err != nil {
			abort(PhaseCompute, err)
			return res
		}
		if c.Placement == "cpu" {
			res.Measurements.CPUComputeGFLOPS = gf
			computed++
		}
	}
	if computed > 0 {
		res.Phases = append(res.Phases, PhaseRecord{Phase: PhaseCompute, Duration: now().Sub(start), Detail: fmt.Sprintf("%d CPU candidate(s)", computed)})
	} else if computeUnsupported {
		res.Phases = append(res.Phases, PhaseRecord{Phase: PhaseCompute, Skipped: true, Detail: "unsupported"})
	} else {
		res.Phases = append(res.Phases, PhaseRecord{Phase: PhaseCompute, Duration: now().Sub(start)})
	}

	// 4. CPU scaling curve (optional, bounded by the phase limit).
	if sp, ok := r.Probe.(ScalingProbe); ok {
		start = now()
		perWorker := 100 * time.Millisecond
		if r.Limits.Phase > 0 && perWorker > r.Limits.Phase/4 {
			perWorker = r.Limits.Phase / 4
		}
		if perWorker < 10*time.Millisecond {
			perWorker = 10 * time.Millisecond
		}
		pts, err := sp.Scaling(runCtx, cpuWorkerCandidates(caps), perWorker)
		if errors.Is(err, ErrUnsupported) {
			res.Phases = append(res.Phases, PhaseRecord{Phase: PhaseScaling, Skipped: true, Detail: "unsupported"})
		} else if err != nil {
			abort(PhaseScaling, err)
			return res
		} else {
			res.Measurements.CPUScaling = pts
			res.Phases = append(res.Phases, PhaseRecord{Phase: PhaseScaling, Duration: now().Sub(start), Detail: fmt.Sprintf("%d point(s)", len(pts))})
		}
	}

	// 5. Pair transfers among healthy devices only.
	var devices []Accelerator
	for _, a := range caps.Accelerators {
		if a.Health == HealthOK {
			devices = append(devices, a)
		}
	}
	if len(devices) >= 2 {
		start = now()
		if res.Measurements.GPULinkGBs == nil {
			res.Measurements.GPULinkGBs = map[string]float64{}
		}
	transfers:
		for i := 0; i < len(devices); i++ {
			for j := 0; j < len(devices); j++ {
				if i == j {
					continue
				}
				gbps, err := measurePhase(runCtx, r, func(pctx context.Context) (float64, error) {
					return r.Probe.Transfer(pctx, devices[i], devices[j])
				})
				if errors.Is(err, ErrUnsupported) {
					res.Phases = append(res.Phases, PhaseRecord{Phase: PhaseTransfer, Skipped: true, Detail: "unsupported"})
					break transfers
				}
				if err != nil {
					abort(PhaseTransfer, err)
					return res
				}
				// Keep the direction: a peer link is not symmetric and a single per-device entry would overwrite it.
				res.Measurements.GPULinkGBs[devices[i].ID+"->"+devices[j].ID] = gbps
			}
		}
		if len(res.Measurements.GPULinkGBs) > 0 {
			res.Phases = append(res.Phases, PhaseRecord{Phase: PhaseTransfer, Duration: now().Sub(start)})
		}
	}

	// 6. Serving: the smallest representative workload, repeated only as often as the limits allow.
	start = now()
	ran := 0
	for _, c := range candidates {
		if !candidateHealthy(c, caps) {
			continue
		}
		var prefill, decode float64
		var lastErr error
		vals := make([]float64, 0, repeats)
		for rep := 0; rep < repeats; rep++ {
			if err := runCtx.Err(); err != nil {
				abort(PhaseServing, err)
				return res
			}
			p, d, err := measurePair(runCtx, r, func(pctx context.Context) (float64, float64, error) {
				return r.Probe.Serving(pctx, c, wl.Envelope)
			})
			if errors.Is(err, ErrUnsupported) {
				res.Phases = append(res.Phases, PhaseRecord{Phase: PhaseServing, Skipped: true, Detail: "unsupported"})
				ran = -1
				break
			}
			if err != nil {
				lastErr = err
				break
			}
			prefill, decode = p, d
			vals = append(vals, d)
			res.Samples = append(res.Samples, Sample{Phase: PhaseServing, Candidate: c,
				Metrics: map[string]float64{"prefill_tok_s": p, "decode_tok_s": d}, At: now()})
		}
		if ran == -1 {
			break
		}
		if lastErr != nil {
			abort(PhaseServing, lastErr)
			return res
		}
		if prefill > 0 && decode > 0 && len(vals) > 0 {
			key := c.Key()
			res.Measurements.PrefillTokS[key] = prefill
			res.Measurements.DecodeTokS[key] = mean(vals)
			res.Measurements.Samples[key] = len(vals)
			if cv, ok := coefficientOfVariation(vals); ok {
				res.Measurements.CV[key] = cv
			}
			res.Measurements.Tested = append(res.Measurements.Tested, c)
			ran++
		}
		if err := r.checkUsage(); err != nil {
			abort(PhaseServing, err)
			return res
		}
	}
	if ran >= 0 {
		res.Phases = append(res.Phases, PhaseRecord{Phase: PhaseServing, Duration: now().Sub(start), Detail: fmt.Sprintf("%d candidate(s)", ran)})
	}
	res.Ended = now()
	return res
}

func (r *Runner) phase(ctx context.Context, f func(context.Context) error) error {
	if r.Limits.Phase <= 0 {
		return f(ctx)
	}
	pctx, cancel := context.WithTimeout(ctx, r.Limits.Phase)
	defer cancel()
	return f(pctx)
}

func measurePhase(ctx context.Context, r *Runner, f func(context.Context) (float64, error)) (float64, error) {
	var v float64
	err := r.phase(ctx, func(pctx context.Context) error {
		var err error
		v, err = f(pctx)
		return err
	})
	return v, err
}

func measurePair(ctx context.Context, r *Runner, f func(context.Context) (float64, float64, error)) (float64, float64, error) {
	var a, b float64
	err := r.phase(ctx, func(pctx context.Context) error {
		var err error
		a, b, err = f(pctx)
		return err
	})
	return a, b, err
}

func (r *Runner) checkUsage() error {
	u := r.Probe.Usage()
	if r.Limits.MaxHostBytes > 0 && u.HostBytes > r.Limits.MaxHostBytes {
		return fmt.Errorf("the probe holds %.1f MiB host memory, limit %.1f MiB", mib(u.HostBytes), mib(r.Limits.MaxHostBytes))
	}
	for id, v := range r.Limits.MaxDeviceBytes {
		if used := u.DeviceBytes[id]; used > v {
			return fmt.Errorf("the probe holds %.1f MiB on %s, limit %.1f MiB", mib(used), id, mib(v))
		}
	}
	return nil
}

func candidateHealthy(c Candidate, caps Capabilities) bool {
	if len(c.Devices) == 0 {
		return c.Placement == "cpu"
	}
	for _, d := range c.Devices {
		if d < 0 || d >= len(caps.Accelerators) || caps.Accelerators[d].Health != HealthOK {
			return false
		}
	}
	return true
}

func cpuWorkerCandidates(caps Capabilities) []int {
	avail := caps.CPU.Available
	if avail <= 0 {
		avail = caps.CPU.Logical
	}
	if avail <= 1 {
		return []int{1}
	}
	pts := []int{1, maxInt(1, avail/2), maxInt(1, avail*3/4), avail}
	seen := map[int]bool{}
	var out []int
	for _, p := range pts {
		if p <= avail && !seen[p] {
			seen[p] = true
			out = append(out, p)
		}
	}
	return out
}

func mean(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	s := 0.0
	for _, x := range v {
		s += x
	}
	return s / float64(len(v))
}

func coefficientOfVariation(v []float64) (float64, bool) {
	if len(v) < 2 {
		return 0, false
	}
	m := mean(v)
	if m <= 0 {
		return 0, true
	}
	s := 0.0
	for _, x := range v {
		d := x - m
		s += d * d
	}
	sd := sqrt(s / float64(len(v)-1))
	return sd / m, true
}

func sqrt(x float64) float64 {
	if x <= 0 {
		return 0
	}
	z := x
	for i := 0; i < 40; i++ {
		z = (z + x/z) / 2
	}
	return z
}

func errString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

// ---------------------------------------------------------------------------------------------------------------
// HostProbe: CPU-only, safe, no device work.

// HostProbe measures the host with ordinary Go loops. It allocates at most BufferBytes and never runs a GPU
// operation, so it is safe on production hardware and in CI.
type HostProbe struct {
	BufferBytes int
	Duration    time.Duration
	mu          sync.Mutex
	held        uint64
}

// NewHostProbe returns a probe sized from available memory: 64 MiB or a sixteenth of it, whichever is smaller.
func NewHostProbe(availBytes uint64) *HostProbe {
	buf := uint64(64) << 20
	if availBytes > 0 && availBytes/16 < buf {
		buf = availBytes / 16
	}
	if buf < 1<<20 {
		buf = 1 << 20
	}
	return &HostProbe{BufferBytes: int(buf), Duration: 200 * time.Millisecond}
}

func (p *HostProbe) Health(context.Context) (Health, string, error) {
	return HealthOK, "host probe", nil
}

func (p *HostProbe) Warmup(context.Context, Candidate) error { return nil }

func (p *HostProbe) MemoryBandwidth(ctx context.Context) (float64, error) {
	n := p.BufferBytes / 8
	if n < 1024 {
		n = 1024
	}
	src := make([]byte, n*8)
	dst := make([]byte, n*8)
	p.mu.Lock()
	p.held = uint64(len(src) + len(dst))
	p.mu.Unlock()
	defer func() {
		p.mu.Lock()
		p.held = 0
		p.mu.Unlock()
	}()
	dur := p.Duration
	if dur <= 0 {
		dur = 200 * time.Millisecond
	}
	started := time.Now()
	deadline := started.Add(dur)
	var bytesCopied uint64
	for time.Now().Before(deadline) {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		copy(dst, src)
		bytesCopied += uint64(len(src))
	}
	elapsed := time.Since(started).Seconds()
	if elapsed <= 0 {
		return 0, errors.New("clock did not advance")
	}
	return float64(bytesCopied) / elapsed / 1e9, nil
}

func (p *HostProbe) Compute(ctx context.Context, c Candidate) (float64, error) {
	if c.Placement != "cpu" {
		return 0, ErrUnsupported // device compute is not measured by the host probe
	}
	dur := p.Duration
	if dur <= 0 {
		dur = 200 * time.Millisecond
	}
	deadline := time.Now().Add(dur)
	var ops float64
	x, y := 1.000001, 1.000002
	for time.Now().Before(deadline) {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		for i := 0; i < 100000; i++ {
			x = x*y + 1e-9
		}
		ops += 200000 // one multiply and one add per iteration
	}
	_ = x
	elapsed := dur.Seconds()
	return ops / elapsed / 1e9, nil
}

func (p *HostProbe) Transfer(context.Context, Accelerator, Accelerator) (float64, error) {
	return 0, ErrUnsupported
}

func (p *HostProbe) Serving(context.Context, Candidate, Envelope) (float64, float64, error) {
	return 0, 0, ErrUnsupported
}

func (p *HostProbe) Usage() Usage {
	p.mu.Lock()
	defer p.mu.Unlock()
	return Usage{HostBytes: p.held}
}

func (p *HostProbe) Cleanup(context.Context) error { return nil }

// Scaling measures the copy bandwidth at several worker counts. It is bounded by ctx and by perWorker per point.
func (p *HostProbe) Scaling(ctx context.Context, workers []int, perWorker time.Duration) ([]CPUScalePoint, error) {
	if len(workers) == 0 {
		return nil, ErrUnsupported
	}
	var out []CPUScalePoint
	for _, w := range workers {
		if err := ctx.Err(); err != nil {
			return out, err
		}
		if w <= 0 {
			continue
		}
		chunk := p.BufferBytes / w
		if chunk < 1<<20 {
			chunk = 1 << 20
		}
		src := make([]byte, chunk)
		// Each worker copies into its own region: sharing one destination would be a data race, not a bandwidth
		// measurement (the copies would also serialise on the same cache lines).
		dst := make([]byte, chunk*w)
		p.mu.Lock()
		p.held = uint64(len(src) + len(dst))
		p.mu.Unlock()
		deadline := time.Now().Add(perWorker)
		var mu sync.Mutex
		var total uint64
		var wg sync.WaitGroup
		for i := 0; i < w; i++ {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				seg := dst[i*chunk : (i+1)*chunk]
				var n uint64
				for time.Now().Before(deadline) {
					if ctx.Err() != nil {
						break
					}
					copy(seg, src)
					n += uint64(len(src))
				}
				mu.Lock()
				total += n
				mu.Unlock()
			}(i)
		}
		wg.Wait()
		p.mu.Lock()
		p.held = 0
		p.mu.Unlock()
		gbs := float64(total) / perWorker.Seconds() / 1e9
		out = append(out, CPUScalePoint{Workers: w, GBs: gbs})
	}
	return out, nil
}

// ---------------------------------------------------------------------------------------------------------------
// EngineProbe: a tiny, bounded measurement through the server's own OpenAI-compatible endpoint.

// EngineProbe sends at most one short streaming request per call. It is used after the engine is ready, never
// before: a request against a loading engine would measure the queue, not the model.
type EngineProbe struct {
	BaseURL string
	APIKey  string
	Model   string
	Client  *http.Client
	// Only is the configuration the running engine executes. When nil the probe measures nothing: a rate from
	// the engine would otherwise be attributed to every candidate in the domain.
	Only *Candidate
}

// Covers implements CandidateScoper.
func (p *EngineProbe) Covers(c Candidate) bool {
	return p.Only != nil && p.Only.Key() == c.Key()
}

// CoveredCandidates implements ScopedCandidates.
func (p *EngineProbe) CoveredCandidates() []Candidate {
	if p.Only == nil {
		return nil
	}
	return []Candidate{*p.Only}
}

func (p *EngineProbe) httpClient() *http.Client {
	if p.Client != nil {
		return p.Client
	}
	return &http.Client{Timeout: 30 * time.Second}
}

func (p *EngineProbe) Health(ctx context.Context) (Health, string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(p.BaseURL, "/")+"/health", nil)
	if err != nil {
		return HealthUnknown, "", err
	}
	if p.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.APIKey)
	}
	resp, err := p.httpClient().Do(req)
	if err != nil {
		return HealthUnknown, "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return HealthUnknown, fmt.Sprintf("health answered %d", resp.StatusCode), nil
	}
	var body struct {
		Loaded bool   `json:"loaded"`
		Status string `json:"status"`
		Error  any    `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return HealthUnknown, "health is not JSON", nil
	}
	if !body.Loaded {
		return HealthUnknown, "the engine is not loaded yet", nil
	}
	return HealthOK, body.Status, nil
}

func (p *EngineProbe) Warmup(ctx context.Context, c Candidate) error {
	_, _, err := p.Serving(ctx, c, Envelope{PromptTokens: 16, OutputTokens: 1})
	return err
}

func (p *EngineProbe) MemoryBandwidth(context.Context) (float64, error) { return 0, ErrUnsupported }
func (p *EngineProbe) Compute(context.Context, Candidate) (float64, error) {
	return 0, ErrUnsupported
}
func (p *EngineProbe) Transfer(context.Context, Accelerator, Accelerator) (float64, error) {
	return 0, ErrUnsupported
}
func (p *EngineProbe) Usage() Usage                  { return Usage{} }
func (p *EngineProbe) Cleanup(context.Context) error { return nil }

// Serving sends one streaming completion of the envelope's prompt size and measures time to first chunk and the
// chunks per second after it. The server emits one SSE event per token, so the chunk count is the token count.
func (p *EngineProbe) Serving(ctx context.Context, _ Candidate, e Envelope) (float64, float64, error) {
	prompt := strings.Repeat("word ", maxInt(e.PromptTokens, 1))
	out := maxInt(e.OutputTokens, 1)
	if out > 8 {
		out = 8 // a startup measurement is not a load test
	}
	body := map[string]any{
		"model":      p.Model,
		"stream":     true,
		"max_tokens": out,
		"messages":   []map[string]string{{"role": "user", "content": prompt}},
	}
	raw, _ := json.Marshal(body)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(p.BaseURL, "/")+"/v1/chat/completions", bytes.NewReader(raw))
	if err != nil {
		return 0, 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	if p.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.APIKey)
	}
	start := time.Now()
	resp, err := p.httpClient().Do(req)
	if err != nil {
		return 0, 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0, 0, fmt.Errorf("the engine answered %d", resp.StatusCode)
	}
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 1<<20), 1<<22)
	first := time.Time{}
	chunks := 0
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}
		var ev struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
			Usage *struct {
				PromptTokens int `json:"prompt_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			continue
		}
		if len(ev.Choices) > 0 && ev.Choices[0].Delta.Content != "" {
			if first.IsZero() {
				first = time.Now()
			}
			chunks++
		}
	}
	if err := sc.Err(); err != nil {
		return 0, 0, err
	}
	if first.IsZero() || chunks == 0 {
		return 0, 0, errors.New("no content arrived")
	}
	ttft := first.Sub(start).Seconds()
	decode := time.Since(first).Seconds()
	if ttft <= 0 || decode <= 0 {
		return 0, 0, errors.New("stream timing is not usable")
	}
	prefill := float64(maxInt(e.PromptTokens, 1)) / ttft
	return prefill, float64(chunks) / decode, nil
}
