// pkg/autotune/cost.go - a lower-bound performance model with measured calibration.
//
// The model is the roofline:
//
//	T >= max(F / P_effective, B / BW_effective + transfer + launch + synchronization)
//
// It is a bound, not a prediction of milliseconds. Measured rates, when they exist for the exact candidate,
// always replace the model (Basis "measured"). A model result is labelled "modeled" and never presented as a
// measurement.
package autotune

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"time"
)

// Candidate is one point in the finite search space. Its fields map to real Strata controls:
//
//	Placement    cpu (no device), gpu (one device), split (layer split across devices)
//	Devices      indexes into Capabilities.Accelerators
//	PoolWorkers  --pool-workers (0: the engine's own physical-core default)
//	PCIeFrac     --pcie-frac (-1: the engine's own measured probe decides)
//	SpecMinP     --spec-min-p
//	Batch        the server's parallel slot count
//	PrefillChunk --prefill chunk size (0: engine auto)
type Candidate struct {
	Placement    string  `json:"placement"`
	Devices      []int   `json:"devices,omitempty"`
	PoolWorkers  int     `json:"pool_workers"`
	PCIeFrac     float64 `json:"pcie_frac"`
	SpecMinP     float64 `json:"spec_min_p"`
	Batch        int     `json:"batch"`
	PrefillChunk int     `json:"prefill_chunk"`
}

// Key is stable across processes and is what measurements and records are keyed by.
func (c Candidate) Key() string {
	devs := append([]int(nil), c.Devices...)
	sort.Ints(devs)
	s := ""
	for i, d := range devs {
		if i > 0 {
			s += ","
		}
		s += fmt.Sprint(d)
	}
	pcie := "auto"
	if c.PCIeFrac >= 0 {
		pcie = fmt.Sprintf("%.2f", c.PCIeFrac)
	}
	return fmt.Sprintf("%s|dev=[%s]|workers=%d|pcie=%s|spec=%.2f|batch=%d|chunk=%d",
		c.Placement, s, c.PoolWorkers, pcie, c.SpecMinP, c.Batch, c.PrefillChunk)
}

// CPUScalePoint is one measured worker count and the bandwidth/compute it sustained.
type CPUScalePoint struct {
	Workers int     `json:"workers"`
	GBs     float64 `json:"gbs"`
	GFLOPS  float64 `json:"gflops,omitempty"`
}

// Measurements are the calibrated inputs. Every map is keyed by Candidate.Key() or by device ID; a missing key
// means "not measured", never zero performance.
type Measurements struct {
	CPUComputeGFLOPS   float64            `json:"cpu_compute_gflops,omitempty"`
	CPUMemBandwidthGBs float64            `json:"cpu_mem_bandwidth_gbs,omitempty"`
	CPUScaling         []CPUScalePoint    `json:"cpu_scaling,omitempty"`
	GPUComputeTFLOPS   map[string]float64 `json:"gpu_compute_tflops,omitempty"`
	GPUMemBandwidthGBs map[string]float64 `json:"gpu_mem_bandwidth_gbs,omitempty"`
	GPULinkGBs         map[string]float64 `json:"gpu_link_gbs,omitempty"`
	PrefillTokS        map[string]float64 `json:"prefill_tok_s,omitempty"`
	DecodeTokS         map[string]float64 `json:"decode_tok_s,omitempty"`
	Tested             []Candidate        `json:"tested,omitempty"` // every candidate a probe actually ran
	Samples            map[string]int     `json:"samples,omitempty"`
	CV                 map[string]float64 `json:"cv,omitempty"` // coefficient of variation of repeated runs
	LaunchOverheadMS   float64            `json:"launch_overhead_ms,omitempty"`
	SyncOverheadMS     float64            `json:"sync_overhead_ms,omitempty"`
	Source             string             `json:"source,omitempty"` // probe | record | mixed | none
	MeasuredAt         time.Time          `json:"measured_at,omitempty"`
}

// Valid reports whether anything was actually measured.
func (m Measurements) Valid() bool {
	return m.CPUMemBandwidthGBs > 0 || m.CPUComputeGFLOPS > 0 || len(m.DecodeTokS) > 0 || len(m.PrefillTokS) > 0
}

// BestPoolWorkers returns the smallest worker count that reaches 97% of the measured peak. That is an empirical
// optimum over the points actually run, not a claim about untested counts.
func (m Measurements) BestPoolWorkers() (int, bool) {
	if len(m.CPUScaling) < 2 {
		return 0, false
	}
	pts := append([]CPUScalePoint(nil), m.CPUScaling...)
	sort.Slice(pts, func(i, j int) bool { return pts[i].Workers < pts[j].Workers })
	peak := 0.0
	for _, p := range pts {
		if p.GBs > peak {
			peak = p.GBs
		}
	}
	if peak <= 0 {
		return 0, false
	}
	for _, p := range pts {
		if p.GBs >= 0.97*peak {
			return p.Workers, true
		}
	}
	return pts[len(pts)-1].Workers, true
}

// Noise is the largest coefficient of variation among the keys a decision depends on, floored at 2% (the
// repeatability floor of a single short run) and capped at 25% (beyond that the measurement is not usable for a
// narrow decision).
func (m Measurements) Noise() float64 {
	noise := 0.02
	for _, cv := range m.CV {
		if cv > noise {
			noise = cv
		}
	}
	if noise > 0.25 {
		noise = 0.25
	}
	return noise
}

// Prediction is what the model can say about one candidate. Basis says how much of it is measured.
type Prediction struct {
	DecodeTokS    float64           `json:"decode_tok_s,omitempty"`
	PrefillTokS   float64           `json:"prefill_tok_s,omitempty"`
	TTFTms        float64           `json:"ttft_ms,omitempty"`
	LatencyMS     float64           `json:"latency_ms,omitempty"`
	PeakHostBytes uint64            `json:"peak_host_bytes,omitempty"`
	PeakGPU       map[string]uint64 `json:"peak_gpu_bytes,omitempty"`
	Basis         string            `json:"basis"` // measured | modeled | unavailable
	Bound         string            `json:"bound"` // lower_bound | none
	Notes         []string          `json:"notes,omitempty"`
}

// Usable reports whether the prediction can rank a candidate.
func (p Prediction) Usable() bool {
	return p.Basis != "unavailable" && (p.DecodeTokS > 0 || p.TTFTms > 0)
}

// objectives; one is chosen automatically unless Options.Objective says otherwise.
const (
	ObjDecodeTokS = "max_decode_tok_s"
	ObjTTFT       = "min_ttft"
	ObjLatency    = "min_latency"
	ObjThroughput = "max_throughput"
)

// ChooseObjective derives the serving objective from the workload: a long prompt is a TTFT problem, several
// users are an aggregate-throughput problem, and otherwise decode speed is what the user feels.
func ChooseObjective(w Workload) string {
	if w.Envelope.Concurrency > 1 {
		return ObjThroughput
	}
	if w.Envelope.PromptTokens > 4*maxInt(w.Envelope.OutputTokens, 64) {
		return ObjTTFT
	}
	return ObjDecodeTokS
}

// Score turns a prediction into a value where higher is better, and whether the objective is per-request.
func Score(objective string, p Prediction, w Workload) float64 {
	switch objective {
	case ObjTTFT:
		if p.TTFTms <= 0 {
			return 0
		}
		return 1000 / p.TTFTms
	case ObjLatency:
		if p.LatencyMS <= 0 {
			return 0
		}
		return 1000 / p.LatencyMS
	case ObjThroughput:
		if p.DecodeTokS <= 0 {
			return 0
		}
		c := w.Envelope.Concurrency
		if c < 1 {
			c = 1
		}
		return p.DecodeTokS * float64(c)
	default:
		return p.DecodeTokS
	}
}

// Model is the calibrated roofline. Zero Measurements produce "unavailable" predictions rather than invented
// numbers.
type Model struct{ M Measurements }

// Predict evaluates one candidate. The memory figures are computed from the workload, not measured.
func (m Model) Predict(c Candidate, w Workload, caps Capabilities) Prediction {
	p := Prediction{PeakGPU: map[string]uint64{}, Basis: "modeled", Bound: "lower_bound"}
	g := w.Geometry
	conc := w.Envelope.Concurrency
	if conc < 1 {
		conc = 1
	}
	if c.Batch > conc {
		conc = c.Batch
	}
	kv := g.KVBytesPerToken() * uint64(w.MaxContext) * uint64(conc)
	if g.KVBytesPerToken() == 0 {
		p.Notes = append(p.Notes, "KV size unknown (geometry incomplete): memory constraint cannot be proven")
	}
	if c.Placement == "cpu" {
		p.PeakHostBytes = g.WeightBytes + kv + 256<<20
	} else {
		for _, d := range c.Devices {
			p.PeakGPU[fmt.Sprint(d)] = kv + uint64(conc)*128<<20
		}
		p.PeakHostBytes = g.WeightBytes/8 + 256<<20 // the host keeps page tables and staging, not all weights
	}

	key := c.Key()
	prefillMeasured := m.M.PrefillTokS[key]
	decodeMeasured := m.M.DecodeTokS[key]
	if prefillMeasured > 0 && decodeMeasured > 0 {
		p.Basis = "measured"
		p.Bound = "none"
		p.PrefillTokS = prefillMeasured
		p.DecodeTokS = decodeMeasured
		p.TTFTms = 1000 * float64(w.Envelope.PromptTokens) / prefillMeasured
		p.LatencyMS = p.TTFTms + 1000*float64(w.Envelope.OutputTokens)/decodeMeasured
		return p
	}

	activeBytes, compute := m.flopsAndBytes(c, g, caps)
	bandwidth := m.bandwidth(c, caps)
	if activeBytes == 0 || g.ActiveParams() == 0 || (compute == 0 && bandwidth == 0) {
		p.Basis = "unavailable"
		p.Bound = "none"
		p.Notes = append(p.Notes, "no measured rate for this candidate and the geometry or device throughput is unknown")
		return p
	}
	params := float64(g.ActiveParams())
	prefillCompute, prefillMem := 0.0, 0.0
	decodeCompute, decodeMem := 0.0, 0.0
	if compute > 0 {
		prefillCompute = 2 * params * float64(w.Envelope.PromptTokens) / compute
		decodeCompute = 2 * params / compute
	}
	if bandwidth > 0 {
		prefillMem = activeBytes / bandwidth
		decodeMem = (activeBytes + float64(g.KVBytesPerToken())) / bandwidth
	}
	prefill := maxNonZero(prefillCompute, prefillMem)
	decode := maxNonZero(decodeCompute, decodeMem)
	overhead := (m.M.LaunchOverheadMS + m.M.SyncOverheadMS) / 1000
	if c.Placement == "cpu" {
		p.Notes = append(p.Notes, "CPU rates come from the scalar probe scaled by the widest SIMD width discovery reported; a bound, not a measurement of the expert kernels")
	}
	if c.Placement == "split" {
		// Communication, synchronization and imbalance are not predicted from theory: a split is selected only
		// when a measured rate for that exact split exists, so a modeled split is not usable.
		p.Basis = "unavailable"
		p.Bound = "none"
		p.Notes = append(p.Notes, "a multi-device split needs a measured rate for that device pair")
		return p
	}
	if prefill <= 0 || decode <= 0 {
		p.Basis = "unavailable"
		p.Bound = "none"
		return p
	}
	p.PrefillTokS = float64(w.Envelope.PromptTokens) / (prefill + overhead)
	p.DecodeTokS = 1 / (decode + overhead)
	p.TTFTms = 1000 * (prefill + overhead)
	p.LatencyMS = p.TTFTms + 1000*float64(w.Envelope.OutputTokens)/p.DecodeTokS
	p.Notes = append(p.Notes, "roofline lower bound; not a measured rate")
	return p
}

// flopsAndBytes returns the per-token active weight bytes and the effective FLOP/s available to the placement.
func (m Model) flopsAndBytes(c Candidate, g Geometry, caps Capabilities) (float64, float64) {
	params := g.ActiveParams()
	if params == 0 || g.WeightBytes == 0 {
		return 0, 0
	}
	total := g.TotalParams()
	bytesPerParam := 0.5 // q4-family fallback when the on-disk size is unknown; labelled in the prediction
	if total > 0 {
		bytesPerParam = float64(g.WeightBytes) / float64(total)
	}
	activeBytes := float64(params) * bytesPerParam
	switch c.Placement {
	case "cpu":
		// A Go FMA loop measures one scalar pipeline, not the vector expert kernels. Scaling it by the SIMD width
		// the CPU reports keeps prefill from looking bandwidth-free (weights are reused across prompt tokens); the
		// factor is a model, so a modeled CPU result stays a bound and is never reported as measured.
		return activeBytes, m.cpuCompute(caps)
	default:
		var bestCompute float64
		for _, d := range c.Devices {
			if len(caps.Accelerators) <= d {
				continue
			}
			id := caps.Accelerators[d].ID
			if v := m.M.GPUComputeTFLOPS[id]; v > bestCompute {
				bestCompute = v
			}
		}
		return activeBytes, bestCompute * 1e12
	}
}

// maxNonZero is max, treating an unmeasured term (0) as absent rather than as zero time.
func maxNonZero(a, b float64) float64 {
	if a <= 0 {
		return b
	}
	if b <= 0 {
		return a
	}
	return math.Max(a, b)
}

// cpuCompute turns the scalar FMA probe into an order-of-magnitude vector rate. It is deliberately coarse: the
// alternative is to treat a vector kernel as if it had the throughput of one scalar lane.
func (m Model) cpuCompute(caps Capabilities) float64 {
	gflops := m.M.CPUComputeGFLOPS
	if gflops <= 0 {
		return 0
	}
	width := 4.0
	for _, s := range caps.CPU.SIMD {
		switch strings.ToLower(s) {
		case "avx512f", "avx512", "avx512f_bf16":
			width = math.Max(width, 16)
		case "avx2", "avx":
			width = math.Max(width, 8)
		case "sse4_2", "sse4.2", "sse2", "neon", "asimd":
			width = math.Max(width, 4)
		}
	}
	return gflops * width * 1e9
}

func (m Model) bandwidth(c Candidate, caps Capabilities) float64 {
	if c.Placement == "cpu" {
		return m.M.CPUMemBandwidthGBs * 1e9
	}
	bw := 0.0
	for _, d := range c.Devices {
		if len(caps.Accelerators) <= d {
			continue
		}
		id := caps.Accelerators[d].ID
		v := m.M.GPUMemBandwidthGBs[id]
		if c.Placement == "split" {
			if v > 0 && (bw == 0 || v < bw) {
				bw = v
			}
		} else if v > bw {
			bw = v
		}
	}
	return bw * 1e9
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func max(a, b float64) float64 { return math.Max(a, b) }
