// pkg/autotune/solver.go - pick a configuration from a finite, explicitly defined search space.
//
// The objective is one of ObjDecodeTokS, ObjTTFT, ObjLatency or ObjThroughput, chosen from the workload unless
// the caller overrides it. Correctness, memory and device safety are hard constraints: a candidate that violates
// one is rejected before it is scored, never traded against speed.
//
// Search is exhaustive over the domain the caller defines. BranchAndBound prunes with a valid upper bound from
// the same roofline model (throughput without any launch/sync/transfer overhead can only overestimate; latency
// without them can only underestimate), and reports the certified gap. Every result states which kind it is:
// empirical optimum, model optimum, certified bound, or heuristic selection.
package autotune

import (
	"fmt"
	"math"
	"sort"
	"time"
)

// Violation is one reason a candidate cannot be used. Codes are stable for tests and /metrics.
type Violation struct {
	Code   string `json:"code"`
	Detail string `json:"detail"`
}

const (
	ViolNoDevice        = "no_healthy_device"
	ViolUnknownDevice   = "unknown_device"
	ViolBackend         = "backend_unsupported"
	ViolHealth          = "device_health"
	ViolCPUSIMD         = "cpu_simd_unsupported"
	ViolContext         = "context_invalid"
	ViolHostMemory      = "host_memory"
	ViolDeviceMemory    = "device_memory"
	ViolPinnedMemory    = "pinned_memory"
	ViolMemoryUncertain = "memory_uncertain"
	ViolBatch           = "batch_unsupported"
	ViolMixedBackend    = "mixed_backend_split"
	ViolPressure        = "host_memory_pressure"
	ViolNumerical       = "numerical_format_unsupported"
)

// Feasibility is the result of the hard-constraint check.
type Feasibility struct {
	OK       bool              `json:"ok"`
	Reasons  []Violation       `json:"reasons,omitempty"`
	PeakHost uint64            `json:"peak_host_bytes,omitempty"`
	PeakGPU  map[string]uint64 `json:"peak_gpu_bytes,omitempty"`
}

// Constraints carries the reserves and format facts the solver must respect. They are policies of this
// implementation, listed in docs/optimization/solver-and-optimality.md, never claimed to be optimal.
type Constraints struct {
	HostReserveBytes   uint64  `json:"host_reserve_bytes"`
	DeviceReserveBytes uint64  `json:"device_reserve_bytes"`
	MemoryUncertainty  float64 `json:"memory_uncertainty"` // fraction of free memory added to the reserve
	WorkspacePerStream uint64  `json:"workspace_per_stream"`
	PinnedStagingBytes uint64  `json:"pinned_staging_bytes,omitempty"`
	NumericalTolerance float64 `json:"numerical_tolerance,omitempty"`
}

// DefaultConstraints derives the reserves from this machine's own numbers: a fixed OS/allocator floor plus the
// measured free-memory spread, and per-device workspace that grows with concurrency. The percentage floors are
// policy constants, not claims of optimality.
func DefaultConstraints(caps Capabilities, w Workload) Constraints {
	hostFloor := caps.Memory.TotalBytes / 10
	if hostFloor < 1<<30 {
		hostFloor = 1 << 30
	}
	spread := caps.Memory.SpreadBytes
	if caps.Memory.AvailableBytes > 0 && spread < caps.Memory.AvailableBytes/100 {
		spread = caps.Memory.AvailableBytes / 100 // at least 1%: one reading cannot show its own error
	}
	return Constraints{
		HostReserveBytes:   reserveBytes(hostFloor, spread, 0),
		DeviceReserveBytes: reserveBytes(512<<20, 0, 0),
		MemoryUncertainty:  0.03,
		WorkspacePerStream: 128 << 20,
	}
}

// Options configures Solve.
type Options struct {
	Objective string
	Domain    *Domain
	Previous  *Candidate // last known-good configuration, if it still validates
	Now       time.Time
}

// Placement is one device layout.
type Placement struct {
	Kind    string `json:"kind"` // cpu | gpu | split
	Devices []int  `json:"devices,omitempty"`
}

// Domain is the finite search space.
type Domain struct {
	Placements   []Placement
	Workers      []int
	PCIeFracs    []float64
	SpecMinPs    []float64
	Batches      []int
	PrefillChunk []int
}

// BuildDomain is the default finite domain for this machine and these measurements.
//
//   - Placements: CPU, every healthy GPU, and every same-backend pair for a layer split. A split stays in the
//     domain so its constraints can reject it; without a measured rate the cost model marks it unavailable
//     rather than inventing a communication cost.
//   - Workers: the engine's own default (0) plus the counts the CPU probe actually ran.
//   - PCIe/spec/batch: only values an engine measurement actually covered are added; otherwise the engine's own
//     defaults are the only candidates. "No blind override" is enforced in the search itself.
func BuildDomain(caps Capabilities, w Workload, m Measurements) Domain {
	d := Domain{
		Placements:   []Placement{{Kind: "cpu"}},
		Workers:      []int{0},
		PCIeFracs:    []float64{-1},
		SpecMinPs:    []float64{0},
		Batches:      []int{1},
		PrefillChunk: []int{0},
	}
	for i, a := range caps.Accelerators {
		if a.Health != HealthOK || a.MemoryTotalBytes == 0 {
			continue
		}
		d.Placements = append(d.Placements, Placement{Kind: "gpu", Devices: []int{i}})
	}
	for i := 0; i < len(caps.Accelerators); i++ {
		for j := i + 1; j < len(caps.Accelerators); j++ {
			a, b := caps.Accelerators[i], caps.Accelerators[j]
			if a.Health != HealthOK || b.Health != HealthOK || a.Backend != b.Backend || a.MemoryTotalBytes == 0 || b.MemoryTotalBytes == 0 {
				continue
			}
			d.Placements = append(d.Placements, Placement{Kind: "split", Devices: []int{i, j}})
		}
	}
	if best, ok := m.BestPoolWorkers(); ok {
		for _, p := range m.CPUScaling {
			if p.Workers > 0 {
				d.Workers = append(d.Workers, p.Workers)
			}
		}
		if best > 0 {
			d.Workers = append(d.Workers, best)
		}
		sort.Ints(d.Workers)
		d.Workers = dedupInts(d.Workers)
	}
	seenPCIe := map[float64]bool{-1: true}
	seenSpec := map[float64]bool{0: true}
	seenBatch := map[int]bool{1: true}
	seenWorkers := map[int]bool{0: true}
	seenChunk := map[int]bool{0: true}
	for _, c := range m.Tested {
		if !seenPCIe[c.PCIeFrac] {
			seenPCIe[c.PCIeFrac] = true
			d.PCIeFracs = append(d.PCIeFracs, c.PCIeFrac)
		}
		if !seenSpec[c.SpecMinP] {
			seenSpec[c.SpecMinP] = true
			d.SpecMinPs = append(d.SpecMinPs, c.SpecMinP)
		}
		if c.Batch > 1 && c.Batch <= maxInt(1, w.Envelope.Concurrency) && !seenBatch[c.Batch] {
			seenBatch[c.Batch] = true
			d.Batches = append(d.Batches, c.Batch)
		}
		// A measured candidate is part of the search space even when the host probe did not sweep its knobs;
		// otherwise the solver could not select the configuration whose rate it just measured.
		if c.PoolWorkers > 0 && !seenWorkers[c.PoolWorkers] {
			seenWorkers[c.PoolWorkers] = true
			d.Workers = append(d.Workers, c.PoolWorkers)
		}
		if c.PrefillChunk > 0 && !seenChunk[c.PrefillChunk] {
			seenChunk[c.PrefillChunk] = true
			d.PrefillChunk = append(d.PrefillChunk, c.PrefillChunk)
		}
	}
	sort.Ints(d.Workers)
	sort.Ints(d.PrefillChunk)
	sort.Float64s(d.PCIeFracs)
	sort.Float64s(d.SpecMinPs)
	sort.Ints(d.Batches)
	return d
}

func dedupInts(in []int) []int {
	out := in[:0]
	for i, v := range in {
		if i == 0 || v != in[i-1] {
			out = append(out, v)
		}
	}
	return out
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Enumerate expands the domain deterministically.
func (d Domain) Enumerate() []Candidate {
	var out []Candidate
	for _, pl := range d.Placements {
		for _, workers := range d.Workers {
			for _, pcie := range d.PCIeFracs {
				for _, spec := range d.SpecMinPs {
					for _, batch := range d.Batches {
						for _, chunk := range d.PrefillChunk {
							out = append(out, Candidate{
								Placement: pl.Kind, Devices: append([]int(nil), pl.Devices...),
								PoolWorkers: workers, PCIeFrac: pcie, SpecMinP: spec, Batch: batch, PrefillChunk: chunk,
							})
						}
					}
				}
			}
		}
	}
	return out
}

// EnumerateCapped enumerates at most max candidates (0: no cap) and reports whether it truncated the space. A
// truncated space can never yield a proven optimum.
func (d Domain) EnumerateCapped(max int) ([]Candidate, bool) {
	all := d.Enumerate()
	if max > 0 && len(all) > max {
		return all[:max], true
	}
	return all, false
}

// Check enforces the hard constraints. It never looks at performance.
func Check(c Candidate, w Workload, caps Capabilities, cons Constraints) Feasibility {
	f := Feasibility{OK: true, PeakGPU: map[string]uint64{}}
	reject := func(code, detail string) {
		f.OK = false
		f.Reasons = append(f.Reasons, Violation{Code: code, Detail: detail})
	}
	g := w.Geometry
	ctx := w.MaxContext
	if ctx <= 0 {
		ctx = 32768
	}
	if w.Envelope.PromptTokens+w.Envelope.OutputTokens > ctx {
		reject(ViolContext, fmt.Sprintf("prompt+output %d exceeds the physical context %d",
			w.Envelope.PromptTokens+w.Envelope.OutputTokens, ctx))
	}
	if g.Context > 0 && ctx > g.Context {
		reject(ViolContext, fmt.Sprintf("context %d exceeds the model's trained %d", ctx, g.Context))
	}
	conc := w.Envelope.Concurrency
	if conc < 1 {
		conc = 1
	}
	if c.Batch < 1 {
		reject(ViolBatch, "batch must be at least 1")
	}
	if c.Batch > conc && c.Batch > 1 {
		reject(ViolBatch, fmt.Sprintf("batch %d exceeds the %d concurrent request(s)", c.Batch, conc))
	}
	batch := maxInt(c.Batch, 1)
	// A memory budget cannot be proven from a number that is not known. These are rejections, not warnings: with
	// an incomplete shape no placement can be certified to fit.
	if g.WeightBytes == 0 {
		reject(ViolMemoryUncertain, "on-disk weight bytes are unknown")
	}
	if g.KVBytesPerToken() == 0 {
		reject(ViolMemoryUncertain, "KV bytes per token are unknown (incomplete attention geometry)")
	}

	switch c.Placement {
	case "cpu":
		if !cpuCanRun(g, caps) {
			reject(ViolCPUSIMD, cpuSIMDReason(g, caps))
		}
		required := g.WeightBytes + g.KVBytesPerToken()*uint64(ctx)*uint64(batch) +
			uint64(conc)*cons.WorkspacePerStream + cons.HostReserveBytes
		f.PeakHost = required
		if required > caps.Memory.UsableBytes() {
			reject(ViolHostMemory, fmt.Sprintf("needs %.1f GiB host, %.1f GiB usable",
				gib(required), gib(caps.Memory.UsableBytes())))
		}
		if caps.Memory.Pressure == "critical" {
			reject(ViolPressure, "host memory is critically low at discovery")
		}
	case "gpu", "split":
		if len(c.Devices) == 0 {
			reject(ViolNoDevice, "no device selected")
			break
		}
		if c.Placement == "gpu" && len(c.Devices) != 1 {
			reject(ViolNoDevice, "a single-device plan must name one device")
		}
		if c.Placement == "split" && len(c.Devices) < 2 {
			reject(ViolNoDevice, "a split needs at least two devices")
		}
		backend := ""
		for _, idx := range c.Devices {
			if idx < 0 || idx >= len(caps.Accelerators) {
				reject(ViolUnknownDevice, fmt.Sprintf("device index %d is not in the discovered list", idx))
				continue
			}
			a := caps.Accelerators[idx]
			if backend == "" {
				backend = a.Backend
			} else if a.Backend != backend {
				reject(ViolMixedBackend, "a split across different backends cannot share pointers, streams or events")
			}
			if a.Health != HealthOK {
				reject(ViolHealth, fmt.Sprintf("%s is %s: %s", a.ID, a.Health, a.HealthReason))
			}
			if !caps.HasBackend(a.Backend) {
				reject(ViolBackend, fmt.Sprintf("the %s backend was not established", a.Backend))
			}
			if !numericalOK(a, g) {
				reject(ViolNumerical, fmt.Sprintf("%s does not report support for this model's format", a.ID))
			}
			if a.MemoryFreeBytes == 0 {
				reject(ViolMemoryUncertain, fmt.Sprintf("%s did not report free memory", a.ID))
				continue
			}
			share := 1.0
			if c.Placement == "split" {
				share = splitShare(a, c, caps)
			}
			need := uint64(float64(capsKV(g, ctx, batch))*share) +
				uint64(float64(conc)*float64(cons.WorkspacePerStream)) +
				uint64(float64(a.MemoryFreeBytes)*cons.MemoryUncertainty) +
				cons.DeviceReserveBytes
			if c.Placement == "split" {
				need += uint64(float64(g.WeightBytes) * share)
			}
			f.PeakGPU[a.ID] = need
			if need > a.MemoryFreeBytes {
				reject(ViolDeviceMemory, fmt.Sprintf("%s needs %.1f GiB free, has %.1f GiB",
					a.ID, gib(need), gib(a.MemoryFreeBytes)))
			}
		}
		hostPaint := g.WeightBytes/8 + 256<<20 + cons.HostReserveBytes
		f.PeakHost = hostPaint
		if hostPaint > caps.Memory.UsableBytes() {
			reject(ViolHostMemory, fmt.Sprintf("the host needs %.1f GiB for staging, %.1f GiB usable",
				gib(hostPaint), gib(caps.Memory.UsableBytes())))
		}
		if cons.PinnedStagingBytes > 0 && caps.Memory.PinnedLimitBytes > 0 &&
			cons.PinnedStagingBytes > caps.Memory.PinnedLimitBytes {
			reject(ViolPinnedMemory, fmt.Sprintf("pinned staging %.1f MiB exceeds RLIMIT_MEMLOCK %.1f MiB",
				mib(cons.PinnedStagingBytes), mib(caps.Memory.PinnedLimitBytes)))
		}
	default:
		reject(ViolNoDevice, "unknown placement "+c.Placement)
	}
	return f
}

func capsKV(g Geometry, ctx, batch int) uint64 {
	return g.KVBytesPerToken() * uint64(ctx) * uint64(batch)
}

// splitShare is this stage's fraction of the model, proportional to free memory. Unequal cards get unequal
// layer counts; the imbalance then bounds the split's speed.
func splitShare(a Accelerator, c Candidate, caps Capabilities) float64 {
	var total float64
	for _, idx := range c.Devices {
		if idx >= 0 && idx < len(caps.Accelerators) {
			total += float64(caps.Accelerators[idx].MemoryFreeBytes)
		}
	}
	if total <= 0 {
		return 1 / float64(len(c.Devices))
	}
	return float64(a.MemoryFreeBytes) / total
}

// cpuCanRun encodes the engine's kernel constraints, not a guess: the canonical Q2_0 pack needs AVX-512, a
// GGUF-native pack needs AVX2 (or NEON/SVE on ARM).
func cpuCanRun(g Geometry, caps Capabilities) bool {
	if g.NativePack {
		return hasSIMD(caps.CPU.SIMD, "avx2") || hasSIMD(caps.CPU.SIMD, "neon") || hasSIMD(caps.CPU.SIMD, "sve")
	}
	return hasSIMD(caps.CPU.SIMD, "avx512f")
}

func cpuSIMDReason(g Geometry, caps Capabilities) string {
	if g.NativePack {
		return "the native pack needs AVX2 (or NEON/SVE), this CPU reports " + joinOrNone(caps.CPU.SIMD)
	}
	return "the canonical Q2_0 pack needs AVX-512, this CPU reports " + joinOrNone(caps.CPU.SIMD)
}

func hasSIMD(list []string, name string) bool {
	for _, s := range list {
		if s == name {
			return true
		}
	}
	return false
}

func joinOrNone(list []string) string {
	if len(list) == 0 {
		return "nothing"
	}
	s := ""
	for i, v := range list {
		if i > 0 {
			s += ","
		}
		s += v
	}
	return s
}

// numericalOK enforces only what is known: a device that reported data types must list the model's format family.
// A device that reported none stays Unknown; the engine's own start-up validates kernels, and the uncertainty is
// recorded in the plan notes rather than assumed either way.
func numericalOK(a Accelerator, g Geometry) bool {
	if len(a.DataTypes) == 0 || g.Quant == "" {
		return true
	}
	switch g.Quant {
	case "f32", "f16", "bf16":
		return hasString(a.DataTypes, "fp16") || hasString(a.DataTypes, "bf16") || hasString(a.DataTypes, "fp32")
	case "q8_0", "q8_k", "int8":
		return hasString(a.DataTypes, "int8")
	default:
		return hasString(a.DataTypes, "int4") || hasString(a.DataTypes, "fp16")
	}
}

func hasString(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}

// ScoredCandidate is one evaluated point.
type ScoredCandidate struct {
	Candidate  Candidate  `json:"candidate"`
	Score      float64    `json:"score"`
	Prediction Prediction `json:"prediction"`
	Feasible   bool       `json:"feasible"`
}

// Evidence says which parts of the selected plan are backed by a measurement.
type Evidence struct {
	PoolWorkersMeasured bool     `json:"pool_workers_measured"`
	PCIeMeasured        bool     `json:"pcie_measured"`
	SpecMeasured        bool     `json:"spec_measured"`
	DecodeMeasured      bool     `json:"decode_measured"`
	PrefillMeasured     bool     `json:"prefill_measured"`
	DeviceMeasured      []string `json:"device_measured,omitempty"`
}

// Plan is the solver's answer.
type Plan struct {
	Selected       Candidate         `json:"selected"`
	Prediction     Prediction        `json:"prediction"`
	Objective      string            `json:"objective"`
	Classification string            `json:"classification"`
	Coverage       string            `json:"coverage"` // exhaustive | bounded
	Gap            float64           `json:"gap,omitempty"`
	Evaluated      int               `json:"evaluated"`
	Feasible       int               `json:"feasible"`
	Pruned         int               `json:"pruned"`
	Rejected       map[string]int    `json:"rejected,omitempty"`
	Noise          float64           `json:"noise"`
	Evidence       Evidence          `json:"evidence"`
	Notes          []string          `json:"notes,omitempty"`
	Ranked         []ScoredCandidate `json:"ranked,omitempty"`
	Fingerprint    string            `json:"fingerprint"`
	WorkloadHash   string            `json:"workload_hash"`
	CreatedAt      time.Time         `json:"created_at"`
}

const (
	ClassEmpirical = "empirical_optimum"
	ClassModel     = "model_optimum"
	ClassBound     = "certified_bound"
	ClassHeuristic = "heuristic_selection"
)

// Solve evaluates every candidate in the domain.
func Solve(w Workload, caps Capabilities, m Measurements, cons Constraints, opts Options) Plan {
	domain := BuildDomain(caps, w, m)
	if opts.Domain != nil {
		domain = *opts.Domain
	}
	objective := opts.Objective
	if objective == "" {
		objective = ChooseObjective(w)
	}
	feasible, evaluated, rejected := evaluate(objective, w, caps, m, cons, domain)
	return finish(w, caps, m, cons, opts, feasible, evaluated, 0, 0, rejected, false)
}

// BranchAndBound evaluates in order of an upper bound and stops once no remaining candidate can beat the best.
//
// Valid bound: throughput from the roofline without launch/sync/transfer overhead is at least the model's
// throughput (those terms only add time), and latency without them is at most the model's latency. Measured
// candidates get an infinite bound because a measurement is not bounded by the model. The gap reported is the
// best bound among pruned candidates relative to the selected score, so it is a certificate for the model, not
// for the measured world.
func BranchAndBound(w Workload, caps Capabilities, m Measurements, cons Constraints, opts Options) Plan {
	domain := BuildDomain(caps, w, m)
	if opts.Domain != nil {
		domain = *opts.Domain
	}
	objective := opts.Objective
	if objective == "" {
		objective = ChooseObjective(w)
	}
	type bounded struct {
		sc    ScoredCandidate
		bound float64
	}
	evaluated := 0
	rejected := map[string]int{}
	model := Model{M: m}
	var ranked []bounded
	for _, c := range domain.Enumerate() {
		evaluated++
		f := Check(c, w, caps, cons)
		if !f.OK {
			for _, v := range f.Reasons {
				rejected[v.Code]++
			}
			continue
		}
		p := model.Predict(c, w, caps)
		s := ScoredCandidate{Candidate: c, Score: Score(objective, p, w), Prediction: p, Feasible: true}
		ranked = append(ranked, bounded{sc: s, bound: upperBound(objective, c, p, w, caps, m)})
	}
	sort.SliceStable(ranked, func(i, j int) bool { return ranked[i].bound > ranked[j].bound })

	best := 0.0
	kept := make([]ScoredCandidate, 0, len(ranked))
	pruned := 0
	worstPrunedBound := 0.0
	for _, b := range ranked {
		if best > 0 && b.bound <= best*(1-1e-9) {
			pruned++
			if b.bound > worstPrunedBound {
				worstPrunedBound = b.bound
			}
			continue
		}
		kept = append(kept, b.sc)
		if b.sc.Score > best {
			best = b.sc.Score
		}
	}
	gap := 0.0
	if best > 0 && pruned > 0 {
		gap = (best - worstPrunedBound) / best
		if gap < 0 {
			gap = 0
		}
	}
	plan := finish(w, caps, m, cons, opts, kept, evaluated, pruned, gap, rejected, pruned > 0)
	if pruned > 0 && plan.Classification == ClassModel {
		plan.Classification = ClassBound
	}
	return plan
}

func evaluate(objective string, w Workload, caps Capabilities, m Measurements, cons Constraints, d Domain) ([]ScoredCandidate, int, map[string]int) {
	model := Model{M: m}
	var feasible []ScoredCandidate
	evaluated := 0
	rejected := map[string]int{}
	for _, c := range d.Enumerate() {
		evaluated++
		f := Check(c, w, caps, cons)
		if !f.OK {
			for _, v := range f.Reasons {
				rejected[v.Code]++
			}
			continue
		}
		p := model.Predict(c, w, caps)
		feasible = append(feasible, ScoredCandidate{Candidate: c, Score: Score(objective, p, w), Prediction: p, Feasible: true})
	}
	return feasible, evaluated, rejected
}

// finish applies the noise-aware choice, the previous known-good candidate and the classification.
func finish(w Workload, caps Capabilities, m Measurements, cons Constraints, opts Options, feasible []ScoredCandidate,
	evaluated, pruned int, gap float64, rejected map[string]int, bounded bool) Plan {
	now := opts.Now
	if now.IsZero() {
		now = time.Now()
	}
	objective := opts.Objective
	if objective == "" {
		objective = ChooseObjective(w)
	}
	noise := m.Noise()
	sort.SliceStable(feasible, func(i, j int) bool {
		if feasible[i].Score != feasible[j].Score {
			return feasible[i].Score > feasible[j].Score
		}
		return tieBreak(feasible[i].Candidate, feasible[j].Candidate)
	})
	plan := Plan{
		Objective: objective, Evaluated: evaluated, Feasible: len(feasible), Pruned: pruned, Gap: gap,
		Rejected: rejected, Noise: noise, Fingerprint: caps.Fingerprint(), WorkloadHash: w.Fingerprint(), CreatedAt: now,
	}
	plan.Ranked = topN(feasible, 8)
	model := Model{M: m}
	defaultCand := DefaultCandidate(caps)

	var best ScoredCandidate
	for _, s := range feasible {
		if s.Prediction.Usable() {
			best = s
			break
		}
	}
	if !best.Prediction.Usable() {
		if len(feasible) == 0 {
			// Nothing passes the hard constraints. Selecting the default anyway would present an unsafe plan as a
			// result, so no candidate is selected and the caller keeps its own configuration.
			plan.Selected = Candidate{}
			plan.Prediction = Prediction{Basis: "unavailable", Bound: "none"}
			plan.Classification = ClassHeuristic
			plan.Coverage = "exhaustive"
			if bounded {
				plan.Coverage = "bounded"
			}
			plan.Notes = append(plan.Notes, "no configuration passes the hard constraints; no plan was selected")
			return plan
		}
		plan.Selected = feasible[0].Candidate
		if def, ok := findCandidate(feasible, defaultCand); ok {
			plan.Selected = def.Candidate
		}
		if workers, ok := m.BestPoolWorkers(); ok {
			// The worker count is measured independently of the model cost model: a saturation curve is valid even
			// when the geometry (and therefore any throughput prediction) is unknown.
			plan.Selected.PoolWorkers = workers
		}
		plan.Prediction = model.Predict(plan.Selected, w, caps)
		plan.Classification = ClassHeuristic
		plan.Coverage = "exhaustive"
		if bounded {
			plan.Coverage = "bounded"
		}
		plan.Evidence = evidenceFor(plan.Selected, m)
		plan.Notes = append(plan.Notes, "no candidate had a usable prediction; the conservative default was kept only because it passes every hard constraint")
		return plan
	}

	threshold := math.Max(0.03, 2*noise)
	keep := best
	deviatedReason := ""
	if prev := opts.Previous; prev != nil && prev.Key() != best.Candidate.Key() {
		if f := Check(*prev, w, caps, cons); f.OK {
			pp := model.Predict(*prev, w, caps)
			ps := Score(objective, pp, w)
			if ps > 0 && best.Score <= ps*(1+threshold) {
				keep = ScoredCandidate{Candidate: *prev, Score: ps, Prediction: pp, Feasible: true}
				deviatedReason = fmt.Sprintf("the previous configuration is within the noise-aware threshold (%.1f%%) of the model's best candidate", 100*threshold)
			}
		}
	}
	if def, ok := findCandidate(feasible, defaultCand); ok && keep.Candidate.Key() != defaultCand.Key() &&
		def.Prediction.Usable() && keep.Score <= def.Score*(1+threshold) {
		keep = def
		deviatedReason = fmt.Sprintf("the engine's default configuration is within the noise-aware threshold (%.1f%%) of the model's best candidate", 100*threshold)
	}
	// A host-only model must not override the engine's choice between CPU and an accelerator: without a single
	// measured accelerator rate the comparison would be between a measured host and a theoretical device.
	if keep.Candidate.Placement != defaultCand.Placement && !anyMeasuredAccelerator(feasible) {
		if def, ok := findCandidate(feasible, defaultCand); ok {
			keep = def
			deviatedReason = "no accelerator throughput has been measured; the engine default placement is kept instead of overriding it from a host-only model"
		}
	}
	// A worker count is chosen from its own measured saturation curve; it does not change the model's throughput,
	// so it is applied after the placement decision instead of being folded into the cost comparison.
	deviated := !sameShape(keep.Candidate, best.Candidate)
	plan.Selected = keep.Candidate
	if workers, ok := m.BestPoolWorkers(); ok {
		plan.Selected.PoolWorkers = workers
	}
	plan.Prediction = keep.Prediction
	plan.Evidence = evidenceFor(plan.Selected, m)
	if bounded {
		plan.Coverage = "bounded"
	} else {
		plan.Coverage = "exhaustive"
	}
	if pruned == 0 {
		plan.Coverage = "exhaustive"
	}

	allMeasured := true
	for _, s := range feasible {
		if s.Prediction.Basis != "measured" {
			allMeasured = false
			break
		}
	}
	switch {
	case allMeasured && !deviated:
		plan.Classification = ClassEmpirical
		if len(feasible) > 1 && runnerUpGap(feasible, keep) <= threshold {
			plan.Notes = append(plan.Notes, fmt.Sprintf(
				"the best measured candidates are within the noise floor (%.1f%%); the selection is not a reliable win", 100*noise))
		}
	case best.Prediction.Basis == "modeled" && !deviated:
		plan.Classification = ClassModel
		plan.Notes = append(plan.Notes, "optimal over the defined finite space under the roofline model; this is not a measured guarantee")
		plan.Notes = append(plan.Notes, "the device throughput this model used was not measured for the selected candidate")
	default:
		plan.Classification = ClassHeuristic
		if deviated {
			if deviatedReason == "" {
				deviatedReason = "the selected configuration differs from the model's best candidate, so this is not claimed as optimal"
			}
			plan.Notes = append(plan.Notes, deviatedReason)
		}
	}
	if keep.Candidate.Placement == "split" && !keep.Prediction.Usable() {
		plan.Notes = append(plan.Notes, "multi-device use was not selected: no measured rate exists for the pair")
	}
	return plan
}

// upperBound is the score implied by the roofline with every overhead removed, using the best bandwidth and
// compute any device offers, never the candidate's own. Measured candidates have no finite model bound.
func upperBound(objective string, c Candidate, p Prediction, w Workload, caps Capabilities, m Measurements) float64 {
	if p.Basis == "measured" {
		return math.Inf(1)
	}
	if p.Basis == "unavailable" {
		return 0
	}
	g := w.Geometry
	params := g.ActiveParams()
	if params == 0 {
		return 0
	}
	bytes := 0.0
	if g.WeightBytes > 0 && g.TotalParams() > 0 {
		bytes = float64(params) * float64(g.WeightBytes) / float64(g.TotalParams())
	}
	if bytes <= 0 {
		return 0
	}
	bytes += float64(g.KVBytesPerToken())
	bestBW := m.CPUMemBandwidthGBs
	bestCompute := (Model{M: m}).cpuCompute(caps)
	for _, a := range caps.Accelerators {
		if v := m.GPUMemBandwidthGBs[a.ID]; v > bestBW {
			bestBW = v
		}
		if v := m.GPUComputeTFLOPS[a.ID] * 1e12; v > bestCompute {
			bestCompute = v
		}
	}
	if bestBW <= 0 {
		return math.Inf(1) // cannot bound without a bandwidth
	}
	var tokS float64
	switch objective {
	case ObjTTFT:
		lb := bytes / (bestBW * 1e9)
		if bestCompute > 0 {
			lb = math.Max(lb, 2*float64(params)*float64(w.Envelope.PromptTokens)/bestCompute)
		}
		if lb <= 0 {
			return math.Inf(1)
		}
		return 1 / lb
	case ObjLatency:
		lb := bytes / (bestBW * 1e9)
		if bestCompute > 0 {
			lb = math.Max(lb, 2*float64(params)*float64(w.Envelope.PromptTokens+w.Envelope.OutputTokens)/bestCompute)
		}
		if lb <= 0 {
			return math.Inf(1)
		}
		return 1 / lb
	default:
		tokS = (bestBW * 1e9) / bytes
		if objective == ObjThroughput {
			tokS *= float64(maxInt(w.Envelope.Concurrency, 1))
		}
		return tokS
	}
}

func runnerUpGap(sorted []ScoredCandidate, keep ScoredCandidate) float64 {
	if keep.Score <= 0 {
		return 0
	}
	for _, s := range sorted {
		if s.Candidate.Key() == keep.Candidate.Key() {
			continue
		}
		if s.Score <= 0 {
			return 1
		}
		return (keep.Score - s.Score) / keep.Score
	}
	return 1
}

func findCandidate(list []ScoredCandidate, c Candidate) (ScoredCandidate, bool) {
	for _, s := range list {
		if s.Candidate.Key() == c.Key() {
			return s, true
		}
	}
	return ScoredCandidate{}, false
}

// DefaultCandidate is the safe single-device configuration the tuner keeps unless a measurement beats it.
func DefaultCandidate(caps Capabilities) Candidate {
	c := Candidate{Placement: "cpu", PoolWorkers: 0, PCIeFrac: -1, SpecMinP: 0, Batch: 1, PrefillChunk: 0}
	for i, a := range caps.Accelerators {
		if a.Health == HealthOK && a.MemoryTotalBytes > 0 {
			return Candidate{Placement: "gpu", Devices: []int{i}, PoolWorkers: 0, PCIeFrac: -1, SpecMinP: 0, Batch: 1, PrefillChunk: 0}
		}
	}
	return c
}

func evidenceFor(c Candidate, m Measurements) Evidence {
	key := c.Key()
	e := Evidence{}
	if _, ok := m.DecodeTokS[key]; ok {
		e.DecodeMeasured = true
	}
	if _, ok := m.PrefillTokS[key]; ok {
		e.PrefillMeasured = true
	}
	if _, ok := m.BestPoolWorkers(); ok && c.PoolWorkers > 0 {
		e.PoolWorkersMeasured = true
	}
	if c.PCIeFrac >= 0 {
		_, ok := m.DecodeTokS[key]
		e.PCIeMeasured = ok
	}
	if c.SpecMinP > 0 {
		_, ok := m.DecodeTokS[key]
		e.SpecMeasured = ok
	}
	for id := range m.GPUMemBandwidthGBs {
		e.DeviceMeasured = append(e.DeviceMeasured, id)
	}
	sort.Strings(e.DeviceMeasured)
	return e
}

// anyMeasuredAccelerator reports whether any accelerator candidate has a measured rate, which is what allows a
// modeled comparison between host and device placements.
func anyMeasuredAccelerator(list []ScoredCandidate) bool {
	for _, s := range list {
		if s.Candidate.Placement != "cpu" && s.Prediction.Basis == "measured" {
			return true
		}
	}
	return false
}

// sameShape compares the controls the cost model ranks, ignoring the independently measured worker count.
func sameShape(a, b Candidate) bool {
	if a.Placement != b.Placement || a.Batch != b.Batch || a.PCIeFrac != b.PCIeFrac ||
		a.SpecMinP != b.SpecMinP || a.PrefillChunk != b.PrefillChunk || len(a.Devices) != len(b.Devices) {
		return false
	}
	for i := range a.Devices {
		if a.Devices[i] != b.Devices[i] {
			return false
		}
	}
	return true
}

func tieBreak(a, b Candidate) bool {
	if len(a.Devices) != len(b.Devices) {
		return len(a.Devices) < len(b.Devices)
	}
	for i := range a.Devices {
		if a.Devices[i] != b.Devices[i] {
			return a.Devices[i] < b.Devices[i]
		}
	}
	if a.PoolWorkers != b.PoolWorkers {
		return a.PoolWorkers < b.PoolWorkers
	}
	return a.PCIeFrac < b.PCIeFrac
}

func topN(list []ScoredCandidate, n int) []ScoredCandidate {
	if len(list) <= n {
		return list
	}
	return list[:n]
}

func gib(v uint64) float64 { return float64(v) / (1 << 30) }
func mib(v uint64) float64 { return float64(v) / (1 << 20) }
