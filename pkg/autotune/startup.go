// pkg/autotune/startup.go - the one call a server start needs.
package autotune

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"strata/pkg/modelschema"
)

// StartupOptions describes a start.
type StartupOptions struct {
	Args       []string // the engine arguments the config resolved
	Cwd        string
	RecordPath string
	MaxContext int
	Now        time.Time
	// Engine probes the running server instead of only the host. The server sets this only in its background
	// refinement pass, after READY and while idle.
	Engine  bool
	BaseURL string
	APIKey  string
	Model   string
	Measure bool
	// FirstRunCalibration measures the host only when no usable record exists. The server sets it so that the
	// first start pays for the bounded calibration and later starts read the record instead.
	FirstRunCalibration bool
	NoSave              bool     // do not persist (a diagnostic run)
	ModelName           string   // a manifest name/alias, used to fill a canonical pack's shape
	Geometry            Geometry // overrides the geometry read from Args when non-zero
	Envelope            Envelope // overrides the default workload envelope when non-zero
}

// StartupResult is everything the caller may want to report or persist.
type StartupResult struct {
	Capabilities Capabilities `json:"capabilities"`
	Workload     Workload     `json:"workload"`
	Plan         Plan         `json:"plan"`
	Measurements Measurements `json:"measurements"`
	AppliedArgs  []string     `json:"applied_args,omitempty"`
	Saved        bool         `json:"saved"`
	Notes        []string     `json:"notes,omitempty"`
	Err          error        `json:"-"`
}

// Startup discovers, plans, probes (host only unless opts.Engine), applies measured knobs and persists the
// record. It is bounded by the probe limits and never returns an error that should stop a server: on failure the
// caller keeps its original arguments.
func Startup(ctx context.Context, opts StartupOptions) StartupResult {
	res := StartupResult{}
	caps := Discover(ctx, DefaultHost(), nil)
	res.Capabilities = caps

	geoArgs := opts.Args
	if opts.Cwd != "" {
		// The engine runs in its own Cwd, so relative weight paths are relative to that directory, not to the
		// server's. Only the geometry lookup is rewritten; the arguments passed on stay as the config wrote them.
		geoArgs = absolutePaths(opts.Args, opts.Cwd)
	}
	g, maxContext, err := GeometryFromArgs(geoArgs)
	if g.Layers == 0 && opts.ModelName != "" {
		if mg, ok := GeometryFromManifest(opts.ModelName); ok {
			mg.WeightBytes = g.WeightBytes // the pack's own on-disk size is the measurement; the manifest has only the shape
			g = mg
		}
	}
	if opts.Geometry.Layers > 0 {
		g = opts.Geometry
	}
	if opts.MaxContext > 0 {
		maxContext = opts.MaxContext
	}
	if err != nil {
		// Still return the machine facts; without a model shape the solver can only use the engine defaults.
		res.Notes = append(res.Notes, "model geometry is unknown: "+err.Error())
	}
	g = g.ApplyKVFlag(opts.Args)
	if g.KVBytesPerElem == 0 {
		g.KVBytesPerElem = 2
	}
	env := DefaultEnvelope(maxContext)
	if opts.Envelope.PromptTokens > 0 {
		env = opts.Envelope
	}
	wl := NewWorkload(g, env, maxContext)
	res.Workload = wl

	var probe Probe
	measure := opts.Measure
	if opts.Engine {
		if opts.BaseURL == "" {
			res.Notes = append(res.Notes, "engine refinement needs a base URL; no engine measurement was taken")
		} else {
			ep := &EngineProbe{BaseURL: opts.BaseURL, APIKey: opts.APIKey, Model: opts.Model}
			if c, ok := candidateFromArgs(caps, opts.Args); ok {
				ep.Only = &c
			} else {
				res.Notes = append(res.Notes, "the running engine's placement cannot be mapped to one candidate; requests are used for validation only")
			}
			probe = ep
		}
	}
	if probe == nil && (measure || opts.FirstRunCalibration) {
		probe = NewHostProbe(caps.Memory.AvailableBytes)
	}
	lim := DefaultLimits()
	if opts.Engine {
		lim.Total = 30 * time.Second
		lim.Phase = 8 * time.Second
		lim.Repeats = 2
	}
	pr := PlanStartup(ctx, caps, PlanOptions{
		Geometry: g, Envelope: wl.Envelope, MaxContext: maxContext, RecordPath: opts.RecordPath,
		Now: opts.Now, Probe: probe, Limits: lim, MaxCandidates: 4, Save: !opts.NoSave, Measure: measure,
	})
	res.Plan, res.Measurements, res.Saved = pr.Plan, pr.Measurements, pr.Saved
	res.Notes = append(res.Notes, pr.Notes...)
	res.Err = pr.Err
	ev := EvidenceFromMeasurements(res.Plan.Selected, res.Measurements)
	res.AppliedArgs = ApplyArgs(opts.Args, res.Plan.Selected, ev)
	if !hasEvidence(ev) {
		res.Notes = append(res.Notes, "the engine defaults are kept: no knob was measured well enough to override")
	}
	return res
}

func absolutePaths(args []string, cwd string) []string {
	out := append([]string(nil), args...)
	for i := 0; i+1 < len(out); i++ {
		if out[i] != "--native" && out[i] != "--pack" {
			continue
		}
		if !filepath.IsAbs(out[i+1]) {
			out[i+1] = filepath.Join(cwd, out[i+1])
		}
	}
	return out
}

func hasEvidence(ev ApplyEvidence) bool {
	return ev.PoolWorkersMeasured || ev.PCIeMeasured || ev.SpecMeasured
}

// GeometryFromArgs reads the model shape from the engine arguments: a GGUF-native path gives full geometry, a
// canonical pack gives its on-disk size (the shape is not in the pack, so the solver treats it as unknown) and
// --max-context gives the physical window.
func GeometryFromArgs(args []string) (Geometry, int, error) {
	native := flagValue(args, "--native")
	pack := flagValue(args, "--pack")
	maxContext := 0
	if n, err := strconv.Atoi(flagValue(args, "--max-context")); err == nil && n > 0 {
		maxContext = n
	}
	if native != "" {
		if st, err := os.Stat(native); err == nil && !st.IsDir() {
			if g, err := GeometryFromGGUF(native); err == nil {
				if maxContext == 0 {
					maxContext = g.Context
				}
				return g, maxContext, nil
			}
		}
	}
	if pack != "" {
		st, err := os.Stat(pack)
		if err != nil || !st.IsDir() {
			return Geometry{}, maxContext, fmt.Errorf("pack directory %q is not readable", pack)
		}
		// The canonical pack has no GGUF header and its shape is not recorded next to the weights, so the shape
		// stays unknown rather than being guessed; the pack's CPU kernels are the AVX-512 canonical ones.
		return Geometry{NativePack: false, WeightBytes: packBytes(pack), KVBytesPerElem: 2}, maxContext, nil
	}
	if native != "" {
		return Geometry{}, maxContext, fmt.Errorf("native model %q is not readable", native)
	}
	return Geometry{}, maxContext, errors.New("no --pack or --native path in the engine arguments")
}

// packBytes sums the weight files in a pack. It walks at most three levels (the layout is pack/, pack/full/,
// pack/chunks/) and skips tokenizer files, which are small but would only blur the number.
func packBytes(dir string) uint64 {
	var total uint64
	root := filepath.Clean(dir)
	_ = filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		rel, _ := filepath.Rel(root, path)
		if strings.Contains(rel, "tokenizer") || strings.Count(rel, string(filepath.Separator)) > 2 {
			return nil
		}
		if d.IsDir() {
			return nil
		}
		ext := strings.ToLower(filepath.Ext(path))
		if ext == ".bin" || ext == ".gguf" {
			if info, err := d.Info(); err == nil {
				total += uint64(info.Size())
			}
		}
		return nil
	})
	return total
}

// GeometryFromManifest fills a canonical pack's shape from the model manifest (pkg/modelschema), which records the
// layer count and attention geometry the pack itself does not. Expert width is not in the manifest, so the active
// expert FLOPs stay unknown and the prediction says so.
func GeometryFromManifest(name string) (Geometry, bool) {
	reg := modelschema.NewRegistry()
	m, ok := reg.GetManifest(name)
	if !ok || m == nil || m.Resources.LayersTotal <= 0 || m.Resources.HiddenDim <= 0 {
		return Geometry{}, false
	}
	r := m.Resources
	headDim := r.HeadDimQK
	if headDim == 0 && r.NumHeads > 0 {
		headDim = r.HiddenDim / r.NumHeads
	}
	g := Geometry{
		Architecture: string(m.Architecture), Layers: r.LayersTotal, Hidden: r.HiddenDim,
		Heads: r.NumHeads, KVHeads: r.NumKVHeads, HeadDim: headDim,
		Context: r.MaxContextTokens, KVBytesPerElem: 2, NativePack: false,
	}
	if m.MoE != nil {
		g.Experts, g.ExpertsPerToken = m.MoE.NumExperts, m.MoE.TopK
		g.ExpertInterDim = m.MoE.ExpertIntermediateDim
	}
	return g, true
}

// candidateFromArgs reconstructs the configuration a running engine executes from its command line. It returns
// false when that is not safe: an explicit --gpu list uses the driver's device order, which need not match the
// discovery order, and several visible GPUs without a list are a multi-device configuration the Candidate model
// does not describe. A rate measured under those conditions cannot be attributed to a single candidate.
func candidateFromArgs(caps Capabilities, args []string) (Candidate, bool) {
	if flagValue(args, "--gpu") != "" {
		return Candidate{}, false
	}
	healthy := caps.HealthyGPUs()
	if len(healthy) > 1 {
		return Candidate{}, false
	}
	c := Candidate{Placement: "cpu", PCIeFrac: -1, Batch: 1}
	if len(healthy) == 1 {
		for i, a := range caps.Accelerators {
			if a.ID == healthy[0].ID {
				c.Placement, c.Devices = "gpu", []int{i}
				break
			}
		}
		if c.Placement != "gpu" {
			return Candidate{}, false
		}
	}
	if v := flagValue(args, "--pool-workers"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			return Candidate{}, false
		}
		c.PoolWorkers = n
	}
	if v := flagValue(args, "--pcie-frac"); v != "" {
		f, err := strconv.ParseFloat(v, 64)
		if err != nil {
			return Candidate{}, false
		}
		c.PCIeFrac = f
	}
	if v := flagValue(args, "--spec-min-p"); v != "" {
		f, err := strconv.ParseFloat(v, 64)
		if err != nil {
			return Candidate{}, false
		}
		c.SpecMinP = f
	}
	return c, true
}

func flagValue(args []string, name string) string {
	for i := 0; i+1 < len(args); i++ {
		if args[i] == name {
			return args[i+1]
		}
		if strings.HasPrefix(args[i], name+"=") {
			return strings.TrimPrefix(args[i], name+"=")
		}
	}
	return ""
}
