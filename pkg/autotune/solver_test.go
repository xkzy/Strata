package autotune

import (
	"math"
	"sort"
	"strings"
	"testing"
	"time"
)

func testGeometry() Geometry {
	return Geometry{
		Architecture: "test", Layers: 24, Hidden: 2048, Heads: 16, KVHeads: 8, HeadDim: 128,
		Experts: 64, ExpertsPerToken: 4, ExpertInterDim: 1024, DenseInterDim: 8192,
		Vocab: 32000, Context: 32768, Quant: "q4_k", WeightBytes: 10 << 30, NativePack: true, KVBytesPerElem: 2,
	}
}

func testWorkload() Workload {
	return NewWorkload(testGeometry(), Envelope{PromptTokens: 2048, OutputTokens: 256, Concurrency: 1, Source: "default"}, 32768)
}

func testCaps() Capabilities {
	return Capabilities{
		Schema: 1, OS: "linux", Arch: "amd64",
		CPU:    CPUInfo{Model: "Test CPU", Logical: 16, Physical: 8, Available: 8, SIMD: []string{"avx2", "avx512f", "fma"}},
		Memory: MemoryInfo{TotalBytes: 64 << 30, AvailableBytes: 48 << 30, SpreadBytes: 64 << 20, Pressure: "low"},
		Accelerators: []Accelerator{
			{ID: "GPU-0", Vendor: "nvidia", Backend: "cuda", Name: "Card A", MemoryTotalBytes: 16 << 30, MemoryFreeBytes: 15 << 30,
				Health: HealthOK, DataTypes: []string{"fp16", "int4"}},
			{ID: "GPU-1", Vendor: "nvidia", Backend: "cuda", Name: "Card B", MemoryTotalBytes: 16 << 30, MemoryFreeBytes: 15 << 30,
				Health: HealthOK, DataTypes: []string{"fp16", "int4"}},
		},
		Backends: []string{"cpu", "cuda"},
	}
}

func testMeasurements() Measurements {
	return Measurements{
		CPUMemBandwidthGBs: 50, CPUComputeGFLOPS: 400,
		GPUComputeTFLOPS:   map[string]float64{"GPU-0": 50, "GPU-1": 50},
		GPUMemBandwidthGBs: map[string]float64{"GPU-0": 400, "GPU-1": 400},
		CV:                 map[string]float64{},
		Source:             "probe", MeasuredAt: time.Now(),
	}
}

func bruteForce(w Workload, caps Capabilities, m Measurements, cons Constraints, objective string) (Candidate, float64, bool) {
	model := Model{M: m}
	domain := BuildDomain(caps, w, m)
	var best Candidate
	bestScore := 0.0
	found := false
	for _, c := range domain.Enumerate() {
		if !Check(c, w, caps, cons).OK {
			continue
		}
		p := model.Predict(c, w, caps)
		if !p.Usable() {
			continue
		}
		s := Score(objective, p, w)
		if !found || s > bestScore || (s == bestScore && tieBreak(c, best)) {
			best, bestScore, found = c, s, true
		}
	}
	return best, bestScore, found
}

func TestSolveAgreesWithBruteForce(t *testing.T) {
	w, caps := testWorkload(), testCaps()
	m := testMeasurements()
	cons := DefaultConstraints(caps, w)
	plan := Solve(w, caps, m, cons, Options{})
	best, _, ok := bruteForce(w, caps, m, cons, ChooseObjective(w))
	if !ok {
		t.Fatal("brute force found nothing")
	}
	if plan.Selected.Key() != best.Key() {
		t.Fatalf("solver %s != brute force %s", plan.Selected.Key(), best.Key())
	}
	if plan.Classification != ClassModel {
		t.Errorf("unmeasured roofline search must be labelled model_optimum, got %s", plan.Classification)
	}
	if plan.Coverage != "exhaustive" || plan.Evaluated != len(BuildDomain(caps, w, m).Enumerate()) {
		t.Errorf("coverage=%s evaluated=%d", plan.Coverage, plan.Evaluated)
	}
	if plan.Selected.Placement != "gpu" {
		t.Errorf("with a measured GPU bandwidth the GPU placement must win: %+v", plan.Selected)
	}
}

func TestHardConstraintsReject(t *testing.T) {
	w, caps := testWorkload(), testCaps()
	cons := DefaultConstraints(caps, w)
	gpu := Candidate{Placement: "gpu", Devices: []int{0}, PoolWorkers: 0, PCIeFrac: -1, Batch: 1, PrefillChunk: 0}

	// Device memory: 3 GiB free cannot hold the KV cache plus reserve.
	tight := caps
	tight.Accelerators = append([]Accelerator(nil), caps.Accelerators...)
	tight.Accelerators[0].MemoryFreeBytes = 3 << 30
	if f := Check(gpu, w, tight, cons); f.OK || !hasViolation(f, ViolDeviceMemory) {
		t.Errorf("tight VRAM must be rejected: %+v", f)
	}
	// Health.
	sick := caps
	sick.Accelerators = append([]Accelerator(nil), caps.Accelerators...)
	sick.Accelerators[0].Health = HealthFailed
	sick.Accelerators[0].HealthReason = "thermal brake"
	if f := Check(gpu, w, sick, cons); f.OK || !hasViolation(f, ViolHealth) {
		t.Errorf("failed device must be rejected: %+v", f)
	}
	// Backend.
	noCUDA := caps
	noCUDA.Backends = []string{"cpu"}
	if f := Check(gpu, w, noCUDA, cons); f.OK || !hasViolation(f, ViolBackend) {
		t.Errorf("missing backend must be rejected: %+v", f)
	}
	// Context.
	long := w
	long.Envelope.PromptTokens = 32768
	long.Envelope.OutputTokens = 512
	if f := Check(gpu, long, caps, cons); f.OK || !hasViolation(f, ViolContext) {
		t.Errorf("prompt+output beyond context must be rejected: %+v", f)
	}
	// CPU kernel support: the canonical pack needs AVX-512.
	noAVX := caps
	noAVX.CPU.SIMD = []string{"avx2"}
	canonical := testGeometry()
	canonical.NativePack = false
	wc := NewWorkload(canonical, w.Envelope, 32768)
	if f := Check(Candidate{Placement: "cpu", PCIeFrac: -1, Batch: 1}, wc, noAVX, cons); f.OK || !hasViolation(f, ViolCPUSIMD) {
		t.Errorf("canonical pack without AVX-512 must be rejected: %+v", f)
	}
	// Mixed backends in a split.
	mixed := caps
	mixed.Accelerators = append([]Accelerator(nil), caps.Accelerators...)
	mixed.Accelerators[1].Backend = "rocm"
	mixed.Backends = append(mixed.Backends, "rocm")
	if f := Check(Candidate{Placement: "split", Devices: []int{0, 1}, PCIeFrac: -1, Batch: 1}, w, mixed, cons); f.OK || !hasViolation(f, ViolMixedBackend) {
		t.Errorf("mixed-backend split must be rejected: %+v", f)
	}
}

func hasViolation(f Feasibility, code string) bool {
	for _, v := range f.Reasons {
		if v.Code == code {
			return true
		}
	}
	return false
}

func TestCPUFallbackWhenNoDeviceIsHealthy(t *testing.T) {
	w, caps := testWorkload(), testCaps()
	caps.Accelerators[0].Health = HealthFailed
	caps.Accelerators[1].Health = HealthFailed
	m := testMeasurements()
	// CPU-only rates: 50 GB/s over ~2.2 GB of active weights is ~23 tok/s.
	cons := DefaultConstraints(caps, w)
	plan := Solve(w, caps, m, cons, Options{})
	if plan.Selected.Placement != "cpu" {
		t.Fatalf("no healthy device: CPU must be selected, got %+v", plan.Selected)
	}
	if plan.Selected.PoolWorkers != 0 {
		t.Errorf("without a measured scaling curve the engine default must be kept: %+v", plan.Selected)
	}
}

func TestBranchAndBoundMatchesAndCertifies(t *testing.T) {
	w, caps := testWorkload(), testCaps()
	m := testMeasurements()
	cons := DefaultConstraints(caps, w)
	exhaustive := Solve(w, caps, m, cons, Options{})
	bound := BranchAndBound(w, caps, m, cons, Options{})
	if bound.Selected.Key() != exhaustive.Selected.Key() {
		t.Fatalf("B&B %s != exhaustive %s", bound.Selected.Key(), exhaustive.Selected.Key())
	}
	if bound.Pruned == 0 {
		t.Fatal("two GPUs produce split candidates whose model has no usable prediction; they must be pruned")
	}
	if bound.Gap < 0 || bound.Gap > 1 {
		t.Errorf("gap must be a fraction: %v", bound.Gap)
	}
	if bound.Classification != ClassBound {
		t.Errorf("modeled B&B must be labelled certified_bound, got %s", bound.Classification)
	}
}

func TestNoiseKeepsTheDefault(t *testing.T) {
	w, caps := testWorkload(), testCaps()
	cons := DefaultConstraints(caps, w)
	def := Candidate{Placement: "gpu", Devices: []int{0}, PCIeFrac: -1, Batch: 1, PrefillChunk: 0}
	alt := Candidate{Placement: "gpu", Devices: []int{1}, PCIeFrac: -1, Batch: 1, PrefillChunk: 0}
	m := testMeasurements()
	m.DecodeTokS = map[string]float64{def.Key(): 100, alt.Key(): 101}
	m.PrefillTokS = map[string]float64{def.Key(): 1000, alt.Key(): 1000}
	m.CV = map[string]float64{def.Key(): 0.1, alt.Key(): 0.1} // 10% noise -> a 1% win is not a win
	domain := Domain{Placements: []Placement{{Kind: "gpu", Devices: []int{0}}, {Kind: "gpu", Devices: []int{1}}},
		Workers: []int{0}, PCIeFracs: []float64{-1}, SpecMinPs: []float64{0}, Batches: []int{1}, PrefillChunk: []int{0}}
	plan := Solve(w, caps, m, cons, Options{Domain: &domain, Objective: ObjDecodeTokS})
	if plan.Selected.Key() != def.Key() {
		t.Errorf("a 1%% difference under 10%% noise must keep the default: %+v", plan.Selected)
	}
	// The kept default is not the fastest tested candidate, so it cannot be called an empirical optimum.
	if plan.Classification != ClassHeuristic {
		t.Errorf("a noise-aware fallback to the default must be labelled heuristic, got %s", plan.Classification)
	}

	// The same space with a 30% win does switch.
	m.DecodeTokS[alt.Key()] = 130
	plan = Solve(w, caps, m, cons, Options{Domain: &domain, Objective: ObjDecodeTokS})
	if plan.Selected.Key() != alt.Key() {
		t.Errorf("a 30%% win under 10%% noise must switch: %+v", plan.Selected)
	}
	if plan.Classification != ClassEmpirical {
		t.Errorf("the fastest measured candidate must be labelled empirical, got %s", plan.Classification)
	}
}

func TestNoiseFromSamples(t *testing.T) {
	cv, ok := coefficientOfVariation([]float64{100, 120, 80})
	if !ok || cv <= 0 || cv > 1 {
		t.Fatalf("cv=%v ok=%v", cv, ok)
	}
	m := Measurements{CV: map[string]float64{"a": 0.4, "b": 0.05}}
	if got := m.Noise(); math.Abs(got-0.25) > 1e-9 {
		t.Errorf("noise is capped at 25%%, got %v", got)
	}
	m.CV = map[string]float64{"a": 0.01}
	if got := m.Noise(); got != 0.02 {
		t.Errorf("noise floors at 2%%, got %v", got)
	}
}

func TestBestPoolWorkersIsSmallestNearPeak(t *testing.T) {
	m := Measurements{CPUScaling: []CPUScalePoint{{1, 10, 0}, {4, 40, 0}, {8, 41, 0}}}
	w, ok := m.BestPoolWorkers()
	if !ok || w != 4 {
		t.Errorf("workers=%d ok=%v, want 4 (smallest within 3%% of 41)", w, ok)
	}
	d := BuildDomain(testCaps(), testWorkload(), m)
	if len(d.Workers) != 4 || d.Workers[0] != 0 { // 0 (engine default) + 1,4,8
		t.Errorf("domain workers: %v", d.Workers)
	}
}

func TestSplitIsNotSelectedWithoutMeasurement(t *testing.T) {
	w, caps := testWorkload(), testCaps()
	// P2P claims are irrelevant to this decision: a split is only used with a measured rate, because the model
	// does not invent a communication cost.
	caps.Peers = []PeerLink{{From: "GPU-0", To: "GPU-1", Kind: "nvlink", P2P: true}}
	m := testMeasurements()
	cons := DefaultConstraints(caps, w)
	plan := Solve(w, caps, m, cons, Options{})
	if plan.Selected.Placement == "split" {
		t.Fatalf("an unmeasured split must not be selected: %+v", plan.Selected)
	}
	for _, s := range plan.Ranked {
		if s.Candidate.Placement == "split" && s.Prediction.Basis != "unavailable" {
			t.Errorf("split prediction was invented: %+v", s.Prediction)
		}
	}
}

func TestEnumerateIsDeterministicAndCapped(t *testing.T) {
	d := BuildDomain(testCaps(), testWorkload(), testMeasurements())
	a, b := d.Enumerate(), d.Enumerate()
	if len(a) != len(b) || len(a) == 0 {
		t.Fatalf("enumeration: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i].Key() != b[i].Key() {
			t.Fatalf("order changed at %d", i)
		}
	}
	capped, truncated := d.EnumerateCapped(3)
	if !truncated || len(capped) != 3 {
		t.Errorf("cap: %d truncated=%v", len(capped), truncated)
	}
}

func TestTieBreakPrefersFewerDevices(t *testing.T) {
	if !tieBreak(Candidate{Placement: "gpu", Devices: []int{0}}, Candidate{Placement: "split", Devices: []int{0, 1}}) {
		t.Error("fewer devices must win a tie")
	}
	list := []ScoredCandidate{{Candidate: Candidate{Devices: []int{2}}, Score: 1}, {Candidate: Candidate{Devices: []int{1}}, Score: 1}}
	sort.SliceStable(list, func(i, j int) bool { return tieBreak(list[i].Candidate, list[j].Candidate) })
	if list[0].Candidate.Devices[0] != 1 {
		t.Errorf("tie-break order: %+v", list)
	}
}

func TestHostOnlyModelKeepsTheDefaultPlacement(t *testing.T) {
	w, caps := testWorkload(), testCaps()
	cons := DefaultConstraints(caps, w)
	// No accelerator rate at all: the host probe ran, the engine has not.
	m := Measurements{CPUMemBandwidthGBs: 50, CPUComputeGFLOPS: 400, CV: map[string]float64{}}
	plan := Solve(w, caps, m, cons, Options{Objective: ObjDecodeTokS})
	if plan.Selected.Placement != "gpu" {
		t.Fatalf("a host-only model must not override the GPU default: %+v", plan.Selected)
	}
	if plan.Classification != ClassHeuristic {
		t.Errorf("an unmeasured placement choice must be heuristic, got %s", plan.Classification)
	}
	if !strings.Contains(strings.Join(plan.Notes, " "), "no accelerator throughput") {
		t.Errorf("the kept default needs its reason: %v", plan.Notes)
	}
}

func TestNoFeasibleCandidateSelectsNothing(t *testing.T) {
	w, caps := testWorkload(), testCaps()
	caps.CPU.SIMD = nil // the native pack needs AVX2; this host cannot run it
	for i := range caps.Accelerators {
		caps.Accelerators[i].MemoryFreeBytes = 1 << 20 // far too little for KV + reserve
	}
	plan := Solve(w, caps, testMeasurements(), DefaultConstraints(caps, w), Options{})
	if plan.Selected.Placement != "" {
		t.Fatalf("a plan was selected even though no candidate is feasible: %+v", plan.Selected)
	}
	if len(plan.Rejected) == 0 || plan.Feasible != 0 {
		t.Errorf("the rejection reasons must be reported: %+v", plan)
	}
	if !strings.Contains(strings.Join(plan.Notes, " "), "no configuration passes") {
		t.Errorf("missing note: %v", plan.Notes)
	}
}

func TestFallbackFromAnInfeasibleDefaultStaysInsideTheConstraints(t *testing.T) {
	w, caps := testWorkload(), testCaps()
	for i := range caps.Accelerators {
		caps.Accelerators[i].MemoryFreeBytes = 1 << 20
	}
	cons := DefaultConstraints(caps, w)
	plan := Solve(w, caps, Measurements{}, cons, Options{})
	if plan.Selected.Placement != "cpu" {
		t.Fatalf("the infeasible GPU default must fall back to the feasible CPU: %+v", plan.Selected)
	}
	if f := Check(plan.Selected, w, caps, cons); !f.OK {
		t.Fatalf("the selected fallback violates a hard constraint: %+v", f)
	}
}

func TestHeterogeneousSplitSharesFollowFreeMemory(t *testing.T) {
	caps := testCaps()
	caps.Accelerators[1].MemoryFreeBytes = 4 << 30 // a smaller card gets a smaller layer share
	c := Candidate{Placement: "split", Devices: []int{0, 1}, PCIeFrac: -1, Batch: 1}
	s0 := splitShare(caps.Accelerators[0], c, caps)
	s1 := splitShare(caps.Accelerators[1], c, caps)
	if !(s0 > s1) || math.Abs(s0+s1-1) > 1e-9 {
		t.Errorf("shares must be proportional and sum to one: %.4f %.4f", s0, s1)
	}
	// The same pair with different backends must not be a split at all.
	caps.Accelerators[1].Backend = "rocm"
	if f := Check(c, testWorkload(), caps, DefaultConstraints(caps, testWorkload())); f.OK || !hasViolation(f, ViolMixedBackend) {
		t.Errorf("a mixed-backend split must be rejected: %+v", f)
	}
}

func TestUnknownGeometryCannotBeCertified(t *testing.T) {
	w := NewWorkload(Geometry{}, DefaultEnvelope(32768), 32768)
	caps := testCaps()
	plan := Solve(w, caps, testMeasurements(), DefaultConstraints(caps, w), Options{})
	if plan.Selected.Placement != "" || plan.Feasible != 0 {
		t.Fatalf("a plan was selected from an incomplete model shape: %+v", plan)
	}
	if plan.Rejected[ViolMemoryUncertain] == 0 {
		t.Errorf("the missing shape must be reported as memory uncertainty: %+v", plan.Rejected)
	}
}
