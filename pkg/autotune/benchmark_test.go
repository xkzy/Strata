package autotune

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

type mockProbe struct {
	health       Health
	healthReason string
	healthErr    error
	bw           float64
	gflops       float64
	computeErr   error
	serving      func(ctx context.Context, c Candidate) (float64, float64, error)
	transferErr  error
	usage        Usage
	cleaned      bool
	warmups      int
	servings     int
}

func (p *mockProbe) Health(context.Context) (Health, string, error) {
	return p.health, p.healthReason, p.healthErr
}
func (p *mockProbe) Warmup(context.Context, Candidate) error { p.warmups++; return nil }
func (p *mockProbe) MemoryBandwidth(context.Context) (float64, error) {
	return p.bw, nil
}
func (p *mockProbe) Compute(context.Context, Candidate) (float64, error) {
	if p.computeErr != nil {
		return 0, p.computeErr
	}
	return p.gflops, nil
}
func (p *mockProbe) Transfer(context.Context, Accelerator, Accelerator) (float64, error) {
	if p.transferErr != nil {
		return 0, p.transferErr
	}
	return 10, nil
}
func (p *mockProbe) Serving(ctx context.Context, c Candidate, _ Envelope) (float64, float64, error) {
	p.servings++
	if p.serving != nil {
		return p.serving(ctx, c)
	}
	return 1000, 100, nil
}
func (p *mockProbe) Usage() Usage                  { return p.usage }
func (p *mockProbe) Cleanup(context.Context) error { p.cleaned = true; return nil }

func okProbe() *mockProbe { return &mockProbe{health: HealthOK, bw: 50, gflops: 400} }

func TestRunnerAbortsOnFailedHealth(t *testing.T) {
	p := okProbe()
	p.health = HealthFailed
	r := Runner{Limits: DefaultLimits(), Probe: p}
	res := r.Run(context.Background(), testCaps(), testWorkload(), []Candidate{DefaultCandidate(testCaps())})
	if res.Aborted != PhaseHealth || res.Err == nil {
		t.Fatalf("aborted=%q err=%v", res.Aborted, res.Err)
	}
	if !p.cleaned {
		t.Error("cleanup must run after an abort")
	}
	if p.servings != 0 || p.warmups != 0 {
		t.Errorf("no work may run on an unhealthy device (warmups=%d servings=%d)", p.warmups, p.servings)
	}
}

func TestRunnerPreservesFirstErrorAndKeepsCompletedSamples(t *testing.T) {
	caps := testCaps()
	w := testWorkload()
	cands := []Candidate{
		{Placement: "gpu", Devices: []int{0}, PCIeFrac: -1, Batch: 1},
		{Placement: "gpu", Devices: []int{1}, PCIeFrac: -1, Batch: 1},
	}
	p := okProbe()
	p.serving = func(_ context.Context, c Candidate) (float64, float64, error) {
		if c.Devices[0] == 1 {
			return 0, 0, errors.New("device 1 fault")
		}
		return 900, 90, nil
	}
	res := (&Runner{Limits: Limits{Phase: time.Second, Repeats: 2}, Probe: p}).Run(context.Background(), caps, w, cands)
	if res.Aborted != PhaseServing || !strings.Contains(res.Err.Error(), "device 1 fault") {
		t.Fatalf("aborted=%q err=%v", res.Aborted, res.Err)
	}
	if len(res.Measurements.DecodeTokS) != 1 {
		t.Errorf("the completed candidate must stay measured: %+v", res.Measurements.DecodeTokS)
	}
	if !p.cleaned {
		t.Error("cleanup must run")
	}
	if p.servings != 3 { // two repeats of device 0, one failing attempt on device 1
		t.Errorf("servings=%d; a failed attempt must not be retried", p.servings)
	}
}

func TestRunnerTimeoutStopsAndCleansUp(t *testing.T) {
	p := okProbe()
	p.serving = func(ctx context.Context, _ Candidate) (float64, float64, error) {
		<-ctx.Done()
		return 0, 0, ctx.Err()
	}
	start := time.Now()
	res := (&Runner{Limits: Limits{Total: 200 * time.Millisecond, Phase: 40 * time.Millisecond}, Probe: p}).
		Run(context.Background(), testCaps(), testWorkload(), []Candidate{DefaultCandidate(testCaps())})
	if res.Aborted != PhaseServing || !errors.Is(res.Err, context.DeadlineExceeded) {
		t.Fatalf("aborted=%q err=%v", res.Aborted, res.Err)
	}
	if time.Since(start) > 2*time.Second {
		t.Errorf("a bounded run took %v", time.Since(start))
	}
	if !p.cleaned {
		t.Error("cleanup must run after a timeout")
	}
}

func TestRunnerSkipsUnsupportedTransfer(t *testing.T) {
	p := okProbe()
	p.transferErr = ErrUnsupported
	res := (&Runner{Limits: Limits{Phase: time.Second}, Probe: p}).Run(context.Background(), testCaps(), testWorkload(),
		[]Candidate{{Placement: "cpu", PCIeFrac: -1, Batch: 1}})
	found := false
	for _, ph := range res.Phases {
		if ph.Phase == PhaseTransfer && ph.Skipped {
			found = true
		}
	}
	if !found {
		t.Errorf("unsupported transfer must be recorded as skipped: %+v", res.Phases)
	}
	if res.Aborted != "" {
		t.Errorf("unsupported is not a failure: %+v", res)
	}
}

func TestRunnerEnforcesMemoryLimit(t *testing.T) {
	p := okProbe()
	p.usage = Usage{HostBytes: 4 << 30}
	res := (&Runner{Limits: Limits{Phase: time.Second, MaxHostBytes: 2 << 30}, Probe: p}).
		Run(context.Background(), testCaps(), testWorkload(), []Candidate{{Placement: "cpu", PCIeFrac: -1, Batch: 1}})
	// The check runs after every phase, so it stops as soon as the probe admits it holds too much.
	if res.Aborted != PhaseWarmup || !strings.Contains(res.Err.Error(), "limit") {
		t.Fatalf("aborted=%q err=%v", res.Aborted, res.Err)
	}
	if !p.cleaned {
		t.Error("cleanup must run")
	}
}

func TestHostProbeIsBoundedAndMeasures(t *testing.T) {
	p := NewHostProbe(1 << 30)
	p.BufferBytes = 1 << 20
	p.Duration = 20 * time.Millisecond
	bw, err := p.MemoryBandwidth(context.Background())
	if err != nil || bw <= 0 {
		t.Fatalf("bandwidth=%v err=%v", bw, err)
	}
	pts, err := p.Scaling(context.Background(), []int{1, 2}, 10*time.Millisecond)
	if err != nil || len(pts) != 2 {
		t.Fatalf("scaling: %+v %v", pts, err)
	}
	if _, _, err := p.Serving(context.Background(), Candidate{}, Envelope{}); !errors.Is(err, ErrUnsupported) {
		t.Errorf("host probe must not claim to serve: %v", err)
	}
	if u := p.Usage(); u.HostBytes != 0 {
		t.Errorf("after cleanup the probe must hold nothing: %+v", u)
	}
}

func TestEngineProbeMeasuresAFakeServer(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"status":"ok","loaded":true}`)
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for i := 0; i < 4; i++ {
			fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n")
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			time.Sleep(2 * time.Millisecond)
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	})
	ts := httptest.NewServer(mux)
	defer ts.Close()
	p := &EngineProbe{BaseURL: ts.URL, Model: "test"}
	h, _, err := p.Health(context.Background())
	if err != nil || h != HealthOK {
		t.Fatalf("health=%v err=%v", h, err)
	}
	prefill, decode, err := p.Serving(context.Background(), Candidate{}, Envelope{PromptTokens: 32, OutputTokens: 4})
	if err != nil {
		t.Fatal(err)
	}
	if prefill <= 0 || decode <= 0 {
		t.Errorf("rates: %v %v", prefill, decode)
	}
}

func TestEngineProbeNotLoadedIsNotReady(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"status":"degraded","loaded":false}`)
	}))
	defer ts.Close()
	h, reason, err := (&EngineProbe{BaseURL: ts.URL}).Health(context.Background())
	if err != nil || h != HealthUnknown || !strings.Contains(reason, "not loaded") {
		t.Fatalf("h=%v reason=%q err=%v", h, reason, err)
	}
}

type scopedProbe struct {
	*mockProbe
	only Candidate
}

func (p *scopedProbe) Covers(c Candidate) bool { return c.Key() == p.only.Key() }

func TestRunnerMeasuresOnlyTheCoveredCandidate(t *testing.T) {
	caps := testCaps()
	w := testWorkload()
	covered := Candidate{Placement: "gpu", Devices: []int{0}, PCIeFrac: -1, Batch: 1}
	other := Candidate{Placement: "gpu", Devices: []int{1}, PCIeFrac: -1, Batch: 1}
	p := &scopedProbe{mockProbe: okProbe(), only: covered}
	res := (&Runner{Limits: Limits{Phase: time.Second}, Probe: p}).Run(context.Background(), caps, w, []Candidate{covered, other})
	if res.Aborted != "" {
		t.Fatalf("aborted=%q err=%v", res.Aborted, res.Err)
	}
	if len(res.Measurements.DecodeTokS) != 1 {
		t.Fatalf("a running-engine probe must not attribute its rate to every candidate: %+v", res.Measurements.DecodeTokS)
	}
	if _, ok := res.Measurements.DecodeTokS[covered.Key()]; !ok {
		t.Errorf("the covered candidate is missing: %+v", res.Measurements.DecodeTokS)
	}
	if _, ok := res.Measurements.DecodeTokS[other.Key()]; ok {
		t.Errorf("an uncovered candidate was measured: %+v", res.Measurements.DecodeTokS)
	}
	if p.warmups != 1 || p.servings != 1 {
		t.Errorf("warmups=%d servings=%d; only the covered candidate may run", p.warmups, p.servings)
	}
}

func TestEngineProbeWithoutScopeMeasuresNothing(t *testing.T) {
	p := &EngineProbe{BaseURL: "http://127.0.0.1:1"}
	if p.Covers(Candidate{Placement: "cpu"}) {
		t.Error("an unscoped engine probe must not claim any candidate")
	}
	c := Candidate{Placement: "cpu", PCIeFrac: -1, Batch: 1}
	p.Only = &c
	if !p.Covers(c) {
		t.Error("the declared running configuration must be covered")
	}
}

func TestScopedProbeConfigurationIsAddedToThePlan(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"status":"ok","loaded":true}`)
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for i := 0; i < 3; i++ {
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n")
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	})
	ts := httptest.NewServer(mux)
	defer ts.Close()

	// This candidate is not in the default domain: only the running engine's arguments name its PCIe share and
	// worker count. It must still be measured, not silently skipped by the scope filter.
	covered := Candidate{Placement: "gpu", Devices: []int{0}, PoolWorkers: 16, PCIeFrac: 0.42, Batch: 1}
	probe := &EngineProbe{BaseURL: ts.URL, Model: "test", Only: &covered}
	w := testWorkload()
	pr := PlanStartup(context.Background(), testCaps(), PlanOptions{
		Geometry: w.Geometry, Envelope: w.Envelope, MaxContext: w.MaxContext,
		Probe: probe, Limits: Limits{Phase: 2 * time.Second}, MaxCandidates: 2, Measure: true,
	})
	if _, ok := pr.Measurements.DecodeTokS[covered.Key()]; !ok {
		t.Fatalf("the running configuration was not measured: %+v", pr.Measurements.DecodeTokS)
	}
	if len(pr.Measurements.DecodeTokS) != 1 {
		t.Errorf("only the covered configuration may be measured: %+v", pr.Measurements.DecodeTokS)
	}
	if pr.Plan.Selected.Key() != covered.Key() {
		t.Errorf("a measured running configuration should be selected, got %s", pr.Plan.Selected.Key())
	}
}

func TestRunnerAbortsOnComputeFailure(t *testing.T) {
	p := okProbe()
	p.computeErr = errors.New("kernel launch failed")
	res := (&Runner{Limits: Limits{Phase: time.Second}, Probe: p}).
		Run(context.Background(), testCaps(), testWorkload(), []Candidate{{Placement: "cpu", PCIeFrac: -1, Batch: 1}})
	if res.Aborted != PhaseCompute || !strings.Contains(res.Err.Error(), "kernel launch failed") {
		t.Fatalf("aborted=%q err=%v", res.Aborted, res.Err)
	}
	if !p.cleaned {
		t.Error("cleanup must run after a kernel failure")
	}
}

func TestRunnerAbortsOnTransferFailure(t *testing.T) {
	p := okProbe()
	p.transferErr = errors.New("peer link fault")
	res := (&Runner{Limits: Limits{Phase: time.Second}, Probe: p}).
		Run(context.Background(), testCaps(), testWorkload(), []Candidate{{Placement: "cpu", PCIeFrac: -1, Batch: 1}})
	if res.Aborted != PhaseTransfer || !strings.Contains(res.Err.Error(), "peer link fault") {
		t.Fatalf("aborted=%q err=%v", res.Aborted, res.Err)
	}
	if !p.cleaned {
		t.Error("cleanup must run after a transfer failure")
	}
	if p.servings != 0 {
		t.Errorf("no serving work may run after a transfer failure: %d", p.servings)
	}
}
