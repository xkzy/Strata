// pkg/resourcemanager/controller.go - the runtime adaptation controller.
//
// Three cooperating loops:
//
//	fast   (1 s)  admission concurrency, queue limits, large-prompt slots
//	medium (5 s)  memory/cache tier, background activity, queue sizing
//	slow  (30 s) health transitions, persistent workload shifts, retune requests
//
// Every change goes through one transaction: propose -> check -> commit -> monitor (and roll back when the
// measured result is worse and the previous configuration is still safe). All decisions are observable; no
// knob is changed because a raw utilisation number "looks high".
//
// The controller only changes server-owned policy. Engine settings such as --pool-workers, the expert cache and
// KV precision have no runtime protocol in this engine and are only planned for the next start; those transitions
// are reported as unsupported instead of being faked.
package resourcemanager

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"
)

// Controller policy names, matching runconfig power_policy.
const (
	PolicyBalanced      = "BALANCED"
	PolicyMaxThroughput = "MAX_THROUGHPUT"
	PolicyLowLatency    = "LOW_LATENCY"
	PolicyEnergySaving  = "ENERGY_SAVING"
)

// vramPlanFloorByte is the VRAM a met engine plan must leave free before its used percentage stops being a
// pressure signal.  It is the engine's own line (generate.cpp: below ~256 MiB a later allocation can make the driver
// page GPU memory and a verify graph spinning on a host flag then never finishes), the same number the engine
// prints as "LOW: requests may stall".
const vramPlanFloorByte = 256 << 20

// Admission tiers, from least to most constrained.
const (
	TierRetain     = "retain"
	TierBalanced   = "balanced"
	TierTight      = "tight"
	TierAggressive = "aggressive"
)

// Request stats from the monitor are p90/p95 over a short window; they are never required to be present.
type Sample struct {
	At time.Time

	KnownCPU   bool
	CPUPercent float64

	KnownRAM         bool
	RAMUsedPercent   float64
	RAMAvailableByte uint64

	KnownVRAM       bool
	VRAMUsedPercent float64
	VRAMFreeBytes   uint64
	// KnownVRAMPlan says the engine's own ready-time plan ran to completion and reported what it left free
	// (its INFO line's vram_free_mib).  The plan accounts for every tier, the KV cache and the prompt path's
	// buffers, so the used percentage it produces is the plan, not pressure: see vramPlanFloorByte.
	KnownVRAMPlan    bool
	VRAMPlanFreeByte uint64
	GPUTempC         float64

	InFlight          int
	Queued            int
	ArrivalsPerSec    float64
	CompletionsPerSec float64

	P95TTFTms    float64
	P95LatencyMS float64
	DecodeTokS   float64
	PromptP95    int
	OutputP95    int
	KVUsedBytes  int64

	EngineState  string
	DeviceFailed bool
	HealthReason string
}

// Knobs is the whole runtime write surface. It is replaced atomically as one value.
type Knobs struct {
	Concurrency      int    `json:"concurrency"`
	QueueLimit       int    `json:"queue_limit"`
	AdmissionOpen    bool   `json:"admission_open"`
	Tier             string `json:"tier"`
	Background       bool   `json:"background_optimization"`
	LargePromptSlots int    `json:"large_prompt_slots"`
}

// ControllerConfig is policy, not a hardware fact. Bounds differ by power policy.
type ControllerConfig struct {
	Enabled           bool
	Policy            string
	StartConcurrency  int
	MinConcurrency    int
	MaxConcurrency    int
	StartQueue        int
	MinQueue          int
	MaxQueue          int
	LargePromptTokens int
	FastInterval      time.Duration
	MediumInterval    time.Duration
	SlowInterval      time.Duration
	MinDwell          time.Duration
	MonitorWindow     time.Duration
	MemoryPerRequest  uint64
	RAMReserve        uint64
	RAMTight          float64
	RAMCritical       float64
	VRAMTight         float64
	VRAMCritical      float64
	CPUSaturated      float64
	GPUTempCritical   float64
	RiskWeight        float64
	MinRetuneGap      time.Duration
}

// DefaultControllerConfig is the balanced starting point. Values are policy and are documented in
// docs/ADAPTIVE.md; none is claimed to be universally optimal.
func DefaultControllerConfig() ControllerConfig {
	return ControllerConfig{
		Enabled:           true,
		Policy:            PolicyBalanced,
		StartConcurrency:  4,
		MinConcurrency:    1,
		MaxConcurrency:    16,
		StartQueue:        32,
		MinQueue:          4,
		MaxQueue:          128,
		LargePromptTokens: 4096,
		FastInterval:      time.Second,
		MediumInterval:    5 * time.Second,
		SlowInterval:      30 * time.Second,
		MinDwell:          10 * time.Second,
		MonitorWindow:     15 * time.Second,
		MemoryPerRequest:  256 << 20,
		RAMReserve:        512 << 20,
		RAMTight:          0.85,
		RAMCritical:       0.93,
		VRAMTight:         0.90,
		VRAMCritical:      0.95,
		CPUSaturated:      0.90,
		GPUTempCritical:   95,
		RiskWeight:        0.5,
		MinRetuneGap:      10 * time.Minute,
	}
}

// ConfigForPolicy returns the bounds of one policy without changing the measurement policy.
func ConfigForPolicy(policy string) ControllerConfig {
	c := DefaultControllerConfig()
	switch normalizePolicy(policy) {
	case PolicyMaxThroughput:
		c.Policy, c.StartConcurrency, c.MaxConcurrency = PolicyMaxThroughput, 8, 64
		c.StartQueue, c.MaxQueue = 128, 512
	case PolicyLowLatency:
		c.Policy, c.StartConcurrency, c.MaxConcurrency = PolicyLowLatency, 2, 8
		c.StartQueue, c.MaxQueue = 8, 32
	case PolicyEnergySaving:
		c.Policy, c.StartConcurrency, c.MaxConcurrency = PolicyEnergySaving, 2, 6
		c.StartQueue, c.MaxQueue = 8, 32
	default:
		c.Policy = PolicyBalanced
	}
	return c
}

func normalizePolicy(p string) string {
	switch p {
	case PolicyMaxThroughput, PolicyLowLatency, PolicyBalanced, PolicyEnergySaving:
		return p
	}
	return PolicyBalanced
}

// Decision is one committed (or refused) reconfiguration.
type Decision struct {
	At       time.Time `json:"at"`
	From     Knobs     `json:"from"`
	To       Knobs     `json:"to"`
	Reason   string    `json:"reason"`
	CostMS   float64   `json:"transition_cost_ms"`
	Risk     float64   `json:"transition_risk"`
	Rollback bool      `json:"rollback,omitempty"`
	Applied  bool      `json:"applied"`
}

type pendingChange struct {
	from           Knobs
	to             Knobs
	until          time.Time
	baselineP95    float64
	baselineDecode float64
	baselinePrompt int
	baselineRej    int
}

// Controller is safe for concurrent use. The zero value is not usable; use NewController.
type Controller struct {
	mu  sync.Mutex
	cfg ControllerConfig

	knobs     Knobs
	prev      Knobs
	lastGood  Knobs
	last      Sample
	have      bool
	active    int
	waiting   int
	activeBig int
	tickets   map[uint64]bool // ticket -> large prompt
	nextID    uint64
	notify    chan struct{}

	lastFast, lastMedium, lastSlow time.Time
	lastChange                     time.Time
	lastShrink                     time.Time
	closedReason                   string
	closedMemory                   bool // admission was closed (at least partly) by a memory/temperature tier
	lastDeviceFailed               bool
	pending                        *pendingChange
	decisions                      []Decision
	maxDecisions                   int
	retuneDue                      bool
	lastRetune                     time.Time

	admitted, rejected, canceled int64
	lastReject                   string
	lastRejectAt                 time.Time
	lastRejectCount              int64
	completions                  []time.Time
	arrivals                     []time.Time

	ewmaTTFT    float64
	ewmaDecode  float64
	ewmaCount   int
	baselineP95 int

	validator func(Knobs) error
	onKnobs   func(Knobs)
	now       func() time.Time
}

// NewController creates a controller in the disabled state only when cfg.Enabled is false.
func NewController(cfg ControllerConfig) *Controller {
	if cfg.FastInterval <= 0 {
		cfg = ConfigForPolicy(cfg.Policy)
	}
	if cfg.StartConcurrency < cfg.MinConcurrency {
		cfg.StartConcurrency = cfg.MinConcurrency
	}
	if cfg.MaxConcurrency < cfg.MinConcurrency {
		cfg.MaxConcurrency = cfg.MinConcurrency
	}
	if cfg.StartQueue < cfg.MinQueue {
		cfg.StartQueue = cfg.MinQueue
	}
	if cfg.MemoryPerRequest == 0 {
		cfg.MemoryPerRequest = 256 << 20
	}
	if cfg.MonitorWindow <= 0 {
		cfg.MonitorWindow = 15 * time.Second
	}
	c := &Controller{
		cfg:          cfg,
		notify:       make(chan struct{}),
		tickets:      map[uint64]bool{},
		maxDecisions: 64,
		now:          time.Now,
	}
	c.knobs = Knobs{
		Concurrency: cfg.StartConcurrency, QueueLimit: cfg.StartQueue, AdmissionOpen: true,
		Tier: TierRetain, Background: true, LargePromptSlots: cfg.StartConcurrency,
	}
	c.prev, c.lastGood = c.knobs, c.knobs
	return c
}

// Configure replaces policy bounds and resets the operable knobs. It is a startup operation: it resets the live
// ticket set, so it must not run while requests are in flight. A disabled controller admits unconditionally.
func (c *Controller) Configure(policy string, enabled bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.cfg = ConfigForPolicy(policy)
	c.cfg.Enabled = enabled
	c.knobs = Knobs{
		Concurrency: c.cfg.StartConcurrency, QueueLimit: c.cfg.StartQueue, AdmissionOpen: true,
		Tier: TierRetain, Background: true, LargePromptSlots: c.cfg.StartConcurrency,
	}
	c.prev, c.lastGood = c.knobs, c.knobs
	c.pending = nil
	c.have = false
	c.active, c.waiting, c.activeBig = 0, 0, 0
	c.tickets = map[uint64]bool{}
	// Wake anyone waiting in the old configuration; it is replaced, not invalidated (the waiters re-check the
	// new knobs and either proceed or wait on the new channel).
	close(c.notify)
	c.notify = make(chan struct{})
}

// SetValidator installs an external check run before commit. It must not call back into the controller.
func (c *Controller) SetValidator(f func(Knobs) error) {
	c.mu.Lock()
	c.validator = f
	c.mu.Unlock()
}

// SetOnKnobs is notified after a successful commit so the server can mirror non-admission policy. It is called
// without the controller lock.
func (c *Controller) SetOnKnobs(f func(Knobs)) {
	c.mu.Lock()
	c.onKnobs = f
	c.mu.Unlock()
}

// Enabled reports whether the controller is allowed to change policy.
func (c *Controller) Enabled() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.cfg.Enabled
}

// Knobs returns a copy of the active policy.
func (c *Controller) Knobs() Knobs {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.knobs
}

// Observe feeds one sample and runs whichever loops are due. It returns the decisions committed this call.
func (c *Controller) Observe(s Sample) []Decision {
	c.mu.Lock()
	defer c.mu.Unlock()
	if s.At.IsZero() {
		s.At = c.now()
	}
	if !c.cfg.Enabled {
		return nil
	}
	c.last, c.have = s, true
	s.InFlight, s.Queued = c.active, c.waiting
	c.updateEWMA(s)

	var out []Decision
	if c.lastFast.IsZero() || s.At.Sub(c.lastFast) >= c.cfg.FastInterval {
		c.lastFast = s.At
		out = append(out, c.fastLocked(s)...)
	}
	if c.lastMedium.IsZero() || s.At.Sub(c.lastMedium) >= c.cfg.MediumInterval {
		c.lastMedium = s.At
		out = append(out, c.mediumLocked(s)...)
	}
	if c.lastSlow.IsZero() || s.At.Sub(c.lastSlow) >= c.cfg.SlowInterval {
		c.lastSlow = s.At
		out = append(out, c.slowLocked(s)...)
	}
	out = append(out, c.monitorLocked(s)...)
	return out
}

func (c *Controller) updateEWMA(s Sample) {
	if s.P95TTFTms > 0 {
		if c.ewmaTTFT == 0 {
			c.ewmaTTFT = s.P95TTFTms
		} else {
			c.ewmaTTFT = 0.7*c.ewmaTTFT + 0.3*s.P95TTFTms
		}
	}
	if s.DecodeTokS > 0 {
		if c.ewmaDecode == 0 {
			c.ewmaDecode = s.DecodeTokS
		} else {
			c.ewmaDecode = 0.7*c.ewmaDecode + 0.3*s.DecodeTokS
		}
	}
	c.ewmaCount++
	if c.baselineP95 == 0 && s.PromptP95 > 0 {
		c.baselineP95 = s.PromptP95
	}
}

func (c *Controller) tier(s Sample) string {
	tier := TierRetain
	if s.KnownRAM {
		switch {
		case s.RAMUsedPercent >= c.cfg.RAMCritical*100:
			return TierAggressive
		case s.RAMUsedPercent >= c.cfg.RAMTight*100:
			tier = TierTight
		case s.RAMUsedPercent >= 70:
			tier = TierBalanced
		}
	}
	// The engine fills the card to its own ready-time plan, so on a card that only just fits the model the used
	// percentage says what the plan decided, not that anything is under pressure: an RTX 5060 Ti alone reads 97%
	// (15.5 of 16 GiB) at rest and serves.  Read as pressure it closed admission on every request of the 2026-10-19
	// run (429 memory_critical) with 334 MiB still free.  The percentage only counts when the plan left less than
	// the engine's own 256 MiB stall floor, which is where the engine also warns the card only just fits.
	vramPlanMet := s.KnownVRAMPlan && s.VRAMPlanFreeByte >= vramPlanFloorByte
	if s.KnownVRAM && !vramPlanMet {
		switch {
		case s.VRAMUsedPercent >= c.cfg.VRAMCritical*100:
			return TierAggressive
		case s.VRAMUsedPercent >= c.cfg.VRAMTight*100:
			if tier == TierRetain {
				tier = TierTight
			}
		}
	}
	if s.GPUTempC >= c.cfg.GPUTempCritical {
		return TierAggressive
	}
	return tier
}

func (c *Controller) fastLocked(s Sample) []Decision {
	var out []Decision
	c.observeHealthLocked(s, &out)
	tier := c.tier(s)
	if s.DeviceFailed || s.EngineState == "failed" || tier == TierAggressive {
		// The reason is recorded even when admission is already closed, so a request queued by an earlier
		// memory close reports the device fault that now dominates.
		c.closedReason = "memory_critical"
		c.closedMemory = tier == TierAggressive
		if s.DeviceFailed || s.EngineState == "failed" {
			c.closedReason = "device_failure"
		}
		if c.knobs.AdmissionOpen {
			to := c.knobs
			to.AdmissionOpen = false
			to.Tier = tier
			reason := "critical resource pressure: admission closed"
			if s.DeviceFailed || s.EngineState == "failed" {
				reason = "engine/device failure: admission closed (" + s.HealthReason + ")"
				to.Tier = TierAggressive
			}
			if d, ok := c.commitLocked(to, reason, 0, 0.2, s.At); ok {
				out = append(out, d)
			}
		}
		return out
	}
	// Reopening uses a stricter (lower) memory line than closing. Coming back at the critical threshold would
	// flap between open and closed on a sample that crosses it by a fraction of a percent.
	clearOfDegraded := s.GPUTempC < c.cfg.GPUTempCritical
	if s.KnownRAM {
		clearOfDegraded = clearOfDegraded && s.RAMUsedPercent < c.cfg.RAMTight*100
	} else if c.closedMemory {
		clearOfDegraded = false // lost telemetry is not evidence that memory recovered
	}
	if s.KnownVRAM {
		clearOfDegraded = clearOfDegraded && s.VRAMUsedPercent < c.cfg.VRAMTight*100
	}
	if !c.knobs.AdmissionOpen && clearOfDegraded {
		to := c.knobs
		to.AdmissionOpen = true
		to.Tier = tier
		if d, ok := c.commitLocked(to, "device healthy and pressure below the degraded line: admission reopened", 0, 0.05, s.At); ok {
			c.closedReason = ""
			c.closedMemory = false
			out = append(out, d)
		}
	}
	saturated := (s.KnownCPU && s.CPUPercent >= c.cfg.CPUSaturated*100) ||
		(s.KnownRAM && s.RAMUsedPercent >= c.cfg.RAMTight*100) ||
		(s.KnownVRAM && s.VRAMUsedPercent >= c.cfg.VRAMTight*100)
	latencyRising := c.ewmaTTFT > 0 && s.P95TTFTms > c.ewmaTTFT*1.5 && c.active > 0
	shrink := saturated || latencyRising || tier == TierTight
	if shrink && c.knobs.Concurrency > c.cfg.MinConcurrency {
		to := c.knobs
		to.Concurrency = maxInt(c.cfg.MinConcurrency, c.knobs.Concurrency/2)
		to.LargePromptSlots = minInt(to.QueueLimit, maxInt(1, to.Concurrency))
		reason := "resource pressure or rising p95 latency: concurrency reduced"
		if d, ok := c.commitLocked(to, reason, 0, 0.05, s.At); ok {
			out = append(out, d)
			c.lastShrink = s.At
		}
		return out
	}
	// Growth needs demand (a waiting request) and a measured absence of pressure, and waits out the dwell time.
	if c.knobs.Concurrency < c.cfg.MaxConcurrency && c.waiting > 0 && !saturated &&
		s.At.Sub(c.lastShrink) >= c.cfg.MinDwell && s.At.Sub(c.lastChange) >= c.cfg.MinDwell &&
		c.enoughMemoryLocked(c.knobs.Concurrency+1, s) {
		to := c.knobs
		step := 1
		if c.cfg.Policy == PolicyMaxThroughput {
			step = maxInt(1, c.knobs.Concurrency/4)
		}
		to.Concurrency = minInt(c.cfg.MaxConcurrency, c.knobs.Concurrency+step)
		to.LargePromptSlots = to.Concurrency
		if d, ok := c.commitLocked(to, fmt.Sprintf("demand: %d request(s) waiting with no measured pressure", c.waiting), 0, 0.03, s.At); ok {
			out = append(out, d)
		}
	}
	return out
}

func (c *Controller) enoughMemoryLocked(concurrency int, s Sample) bool {
	if !s.KnownRAM {
		// Unknown memory is not evidence that there is room. Growth waits for a reading; safety is never
		// inferred from a missing metric.
		return false
	}
	need := uint64(concurrency) * c.cfg.MemoryPerRequest
	avail := s.RAMAvailableByte
	if avail < need+c.cfg.RAMReserve {
		return false
	}
	return true
}

func (c *Controller) mediumLocked(s Sample) []Decision {
	var out []Decision
	tier := c.tier(s)
	to := c.knobs
	to.Tier = tier
	to.QueueLimit = c.queueForPolicyLocked(tier)
	if to.QueueLimit < to.Concurrency {
		to.QueueLimit = to.Concurrency
	}
	slots := to.Concurrency
	if s.PromptP95 > c.cfg.LargePromptTokens && to.Concurrency > 1 {
		slots = maxInt(1, to.Concurrency/2)
	}
	to.LargePromptSlots = slots
	to.Background = c.active == 0 && (tier == TierRetain || tier == TierBalanced)
	if to != c.knobs {
		reason := fmt.Sprintf("medium loop: tier=%s queue=%d large_prompt_slots=%d background=%t", tier, to.QueueLimit, to.LargePromptSlots, to.Background)
		if d, ok := c.commitLocked(to, reason, 0, 0.02, s.At); ok {
			out = append(out, d)
		}
	}
	return out
}

func (c *Controller) queueForPolicyLocked(tier string) int {
	q := c.cfg.StartQueue
	switch c.cfg.Policy {
	case PolicyMaxThroughput:
		q = c.cfg.MaxQueue
	case PolicyLowLatency:
		q = c.cfg.MinQueue * 2
	case PolicyEnergySaving:
		q = c.cfg.MinQueue * 2
	}
	switch tier {
	case TierTight:
		q /= 2
	case TierAggressive:
		q /= 4
	}
	return clampInt(q, c.cfg.MinQueue, c.cfg.MaxQueue)
}

// observeHealthLocked tracks failed -> healthy transitions. A recovery is the one moment a slow-loop
// re-measurement is worth its cost: the placement on disk may have been chosen for a device that is no longer
// the same. It never probes the failed device itself; it only consumes the orchestrator's ready state.
func (c *Controller) observeHealthLocked(s Sample, out *[]Decision) {
	failed := s.DeviceFailed || s.EngineState == "failed"
	recovered := c.lastDeviceFailed && !failed && s.EngineState == "ready"
	if recovered && s.At.Sub(c.lastRetune) >= c.cfg.MinRetuneGap {
		c.retuneDue = true
		c.lastRetune = s.At
		*out = append(*out, c.recordLocked(Decision{At: s.At, From: c.knobs, To: c.knobs,
			Reason: "engine recovered to ready: bounded re-measurement requested for the next start", Applied: false}))
	}
	c.lastDeviceFailed = failed
}

func (c *Controller) slowLocked(s Sample) []Decision {
	var out []Decision
	if c.baselineP95 > 0 && s.PromptP95 > 2*c.baselineP95 && s.PromptP95 >= 1024 &&
		s.At.Sub(c.lastRetune) >= c.cfg.MinRetuneGap {
		c.retuneDue = true
		c.lastRetune = s.At
		c.baselineP95 = s.PromptP95
		out = append(out, c.recordLocked(Decision{At: s.At, From: c.knobs, To: c.knobs,
			Reason: "workload shifted to longer prompts: bounded retune requested", Applied: false}))
	}
	return out
}

func (c *Controller) monitorLocked(s Sample) []Decision {
	p := c.pending
	if p == nil || s.At.Before(p.until) {
		return nil
	}
	c.pending = nil
	// Attribution: a prompt mix that moved by more than half can change latency and decode speed by itself. In
	// that case the window cannot separate the reconfiguration's effect from the workload's, so it is recorded
	// as inconclusive and the configuration is kept (the next window can measure it again).
	if p.baselinePrompt > 0 && s.PromptP95 > 0 &&
		(float64(s.PromptP95) > 1.5*float64(p.baselinePrompt) || float64(s.PromptP95) < float64(p.baselinePrompt)/1.5) {
		return []Decision{c.recordLocked(Decision{At: s.At, From: c.knobs, To: c.knobs,
			Reason: "monitor window inconclusive: prompt length changed by more than half, not attributed to the reconfiguration", Applied: false})}
	}
	worseLatency := p.baselineP95 > 0 && s.P95TTFTms > p.baselineP95*1.2
	worsePressure := s.KnownRAM && s.RAMUsedPercent >= c.cfg.RAMCritical*100
	worseRejects := c.rejected > int64(p.baselineRej) && p.to.Concurrency > p.from.Concurrency
	// A growth that made decode slower was not useful even if its TTFT looked acceptable: the measured work
	// per second is the objective, and keeping the larger setting would only add memory pressure.
	worseDecode := p.to.Concurrency > p.from.Concurrency && p.baselineDecode > 0 && s.DecodeTokS > 0 &&
		s.DecodeTokS < p.baselineDecode*0.9
	if !(worseLatency || worsePressure || worseRejects || worseDecode) {
		c.lastGood = c.knobs
		return nil
	}
	// Roll back only to a configuration that is still safe; never to a closed/failed state.
	if !p.from.AdmissionOpen || c.tier(s) == TierAggressive || s.DeviceFailed {
		return nil
	}
	c.prev = c.knobs
	c.knobs = p.from
	c.lastChange = s.At
	d := c.recordLocked(Decision{At: s.At, From: c.prev, To: c.knobs, Rollback: true, Applied: true,
		Reason: "monitor window showed worse latency, decode throughput, pressure or rejections: reverted"})
	c.notifyLocked()
	return []Decision{d}
}

// commitLocked applies a target if it passes bounds, external validation and the transition cost test.
func (c *Controller) commitLocked(to Knobs, reason string, costMS, risk float64, at time.Time) (Decision, bool) {
	if !c.cfg.Enabled {
		return Decision{}, false
	}
	// C_transition is measured here, not assumed: the time spent checking, validating and swapping the whole
	// Knobs value under the lock. Server-owned knobs allocate and migrate nothing, so this is the whole cost.
	started := c.now()
	if to == c.knobs {
		return Decision{}, false
	}
	if to.Concurrency < c.cfg.MinConcurrency || to.Concurrency > c.cfg.MaxConcurrency {
		return c.recordLocked(Decision{At: at, From: c.knobs, To: to, Reason: "rejected: concurrency outside policy bounds", Risk: risk}), false
	}
	if to.QueueLimit < c.cfg.MinQueue || to.QueueLimit > c.cfg.MaxQueue {
		return c.recordLocked(Decision{At: at, From: c.knobs, To: to, Reason: "rejected: queue outside policy bounds", Risk: risk}), false
	}
	if to.Concurrency > c.knobs.Concurrency && !c.enoughMemoryLocked(to.Concurrency, c.last) {
		return c.recordLocked(Decision{At: at, From: c.knobs, To: to, Reason: "rejected: host memory cannot hold the proposed concurrency"}), false
	}
	if c.validator != nil {
		if err := c.validator(to); err != nil {
			return c.recordLocked(Decision{At: at, From: c.knobs, To: to, Reason: "rejected by validator: " + err.Error()}), false
		}
	}
	gain := 0.0
	if !to.AdmissionOpen && c.knobs.AdmissionOpen {
		// Closing admission is a safety action; it must never lose the cost test to a sub-millisecond
		// measurement. Its risk term still records what was given up (capacity), not whether to act.
		gain = 1.0
	} else if to.Concurrency > c.knobs.Concurrency {
		gain = float64(to.Concurrency-c.knobs.Concurrency) / float64(maxInt(1, to.Concurrency))
	} else if to.Concurrency < c.knobs.Concurrency {
		gain = 0.5 // a pressure response is worth more than the throughput it costs
	} else {
		gain = 0.1 // tier/queue/background adjustments: no throughput claim, but the safety policy changed
	}
	// The measured commit work is added to any caller-supplied estimate. Nothing is allocated or migrated, so it
	// is the whole transition cost of the server-owned knobs.
	measuredMS := float64(c.now().Sub(started)) / float64(time.Millisecond)
	if measuredMS > costMS {
		costMS = measuredMS
	}
	// J = -gain + riskWeight*risk + cost; only commit when the expected gain pays for the transition.
	if gain <= 0 || gain < c.cfg.RiskWeight*risk+costMS/1000 {
		return c.recordLocked(Decision{At: at, From: c.knobs, To: to, Reason: "rejected: transition cost/risk exceeds expected gain", CostMS: costMS, Risk: risk}), false
	}
	c.prev = c.knobs
	changedConcurrency := to.Concurrency != c.prev.Concurrency
	c.knobs = to
	c.lastChange = at
	if changedConcurrency {
		c.pending = &pendingChange{from: c.prev, to: to, until: at.Add(c.cfg.MonitorWindow),
			baselineP95: c.last.P95TTFTms, baselineDecode: c.last.DecodeTokS,
			baselinePrompt: c.last.PromptP95, baselineRej: int(c.rejected)}
	}
	d := c.recordLocked(Decision{At: at, From: c.prev, To: to, Reason: reason, CostMS: costMS, Risk: risk, Applied: true})
	c.notifyLocked()
	if c.onKnobs != nil {
		f := c.onKnobs
		knobs := c.knobs
		go f(knobs)
	}
	return d, true
}

func (c *Controller) recordLocked(d Decision) Decision {
	if len(c.decisions) >= c.maxDecisions {
		c.decisions = c.decisions[len(c.decisions)-c.maxDecisions+1:]
	}
	c.decisions = append(c.decisions, d)
	return d
}

func (c *Controller) notifyLocked() {
	close(c.notify)
	c.notify = make(chan struct{})
}

// Admission is the answer to one request.
type Admission struct {
	Admitted   bool
	Ticket     uint64
	RetryAfter int
	Reason     string
	Wait       time.Duration
}

// Admit reserves a slot or queues the request within QueueLimit. It returns as soon as a slot is free, the
// queue is full, the controller closes admission, or ctx ends. The caller must Release the ticket.
func (c *Controller) Admit(ctx context.Context, promptTokens int) Admission {
	start := time.Now()
	c.mu.Lock()
	c.arrivals = append(c.arrivals, start)
	if len(c.arrivals) > 128 {
		c.arrivals = c.arrivals[len(c.arrivals)-128:]
	}
	if !c.cfg.Enabled {
		c.active++
		c.nextID++
		id := c.nextID
		c.tickets[id] = promptTokens >= c.cfg.LargePromptTokens
		c.mu.Unlock()
		return Admission{Admitted: true, Ticket: id, Reason: "controller disabled"}
	}
	for {
		reject, retry, reason := c.admissionBlockLocked(promptTokens)
		if !reject {
			if c.active < c.knobs.Concurrency {
				return c.grantLocked(promptTokens, start)
			}
			if c.waiting >= c.knobs.QueueLimit {
				return c.rejectLocked("server_overloaded", start)
			}
		} else {
			c.rejected++
			c.lastReject = reason
			c.lastRejectAt = time.Now()
			c.mu.Unlock()
			return Admission{Admitted: false, RetryAfter: retry, Reason: reason}
		}
		c.waiting++
		ch := c.notify
		c.mu.Unlock()
		select {
		case <-ctx.Done():
			c.mu.Lock()
			c.waiting--
			c.canceled++
			c.mu.Unlock()
			return Admission{Admitted: false, RetryAfter: 1, Reason: "request canceled while queued"}
		case <-ch:
		}
		c.mu.Lock()
		c.waiting--
	}
}

func (c *Controller) admissionBlockLocked(promptTokens int) (bool, int, string) {
	if !c.knobs.AdmissionOpen {
		reason := c.closedReason
		if reason == "" {
			reason = "admission_closed"
		}
		return true, c.retryAfterLocked(), reason
	}
	if c.have && c.tier(c.last) == TierAggressive {
		return true, c.retryAfterLocked(), "memory_critical"
	}
	if c.have && c.last.DeviceFailed {
		return true, c.retryAfterLocked(), "device_failure"
	}
	if promptTokens >= c.cfg.LargePromptTokens && c.active > 0 && c.activeBig >= c.knobs.LargePromptSlots {
		return true, c.retryAfterLocked(), "prefill_slot_limit"
	}
	return false, 0, ""
}

func (c *Controller) grantLocked(promptTokens int, start time.Time) Admission {
	c.active++
	c.nextID++
	id := c.nextID
	c.tickets[id] = promptTokens >= c.cfg.LargePromptTokens
	if c.tickets[id] {
		c.activeBig++
	}
	c.admitted++
	knobs := c.knobs
	c.mu.Unlock()
	return Admission{Admitted: true, Ticket: id, Wait: time.Since(start), Reason: fmt.Sprintf("concurrency=%d queue=%d", knobs.Concurrency, knobs.QueueLimit)}
}

func (c *Controller) rejectLocked(reason string, start time.Time) Admission {
	_ = start
	c.rejected++
	c.lastReject = reason
	c.lastRejectAt = time.Now()
	retry := c.retryAfterLocked()
	c.mu.Unlock()
	return Admission{Admitted: false, RetryAfter: retry, Reason: reason}
}

// MarkLarge upgrades an admitted ticket to a large-prompt slot once the prompt has been tokenized. The
// runtime path admits before it knows the prompt length; the upgrade makes later large prefills respect the
// slot limit without rejecting or restarting this request. It is idempotent.
func (c *Controller) MarkLarge(ticket uint64, promptTokens int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.cfg.Enabled || promptTokens < c.cfg.LargePromptTokens {
		return
	}
	big, ok := c.tickets[ticket]
	if !ok || big {
		return
	}
	c.tickets[ticket] = true
	c.activeBig++
}

// Release frees an admitted ticket and records a completion time for retry estimates.
func (c *Controller) Release(ticket uint64, now time.Time) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.cfg.Enabled {
		if c.active > 0 {
			c.active--
		}
		delete(c.tickets, ticket)
		return
	}
	if _, ok := c.tickets[ticket]; !ok {
		return
	}
	delete(c.tickets, ticket)
	if c.active > 0 {
		c.active--
	}
	c.completions = append(c.completions, now)
	if len(c.completions) > 128 {
		c.completions = c.completions[len(c.completions)-128:]
	}
	// activeBig is recomputed from live tickets to stay correct across rollbacks and restarts.
	c.activeBig = 0
	for _, big := range c.tickets {
		if big {
			c.activeBig++
		}
	}
	c.notifyLocked()
}

func (c *Controller) retryAfterLocked() int {
	now := time.Now()
	recent := 0
	for i := len(c.completions) - 1; i >= 0; i-- {
		if now.Sub(c.completions[i]) > 10*time.Second {
			break
		}
		recent++
	}
	if recent == 0 {
		return 2
	}
	rate := float64(recent) / 10
	if rate <= 0 {
		return 2
	}
	sec := int(math.Ceil(float64(c.active+c.waiting) / rate))
	return clampInt(sec, 1, 60)
}

// RetuneDue reports a slow-loop request for a bounded engine retune without consuming it. The server keeps the
// request pending while the engine is busy or still starting; TakeRetuneDue clears it only once the measurement
// actually starts, so a request made during a busy period is not lost.
func (c *Controller) RetuneDue() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.retuneDue
}

// TakeRetuneDue clears a pending retune request and reports whether there was one.
func (c *Controller) TakeRetuneDue() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	due := c.retuneDue
	c.retuneDue = false
	return due
}

// Stats is the controller's observable state.
type Stats struct {
	Enabled       bool           `json:"enabled"`
	Policy        string         `json:"policy"`
	Knobs         Knobs          `json:"knobs"`
	InFlight      int            `json:"in_flight"`
	Queued        int            `json:"queued"`
	Admitted      int64          `json:"admitted"`
	Rejected      int64          `json:"rejected"`
	Canceled      int64          `json:"canceled"`
	LastReject    string         `json:"last_reject,omitempty"`
	LastRejectAt  float64        `json:"last_reject_at,omitempty"`
	Decisions     []Decision     `json:"decisions"`
	Pending       map[string]any `json:"pending,omitempty"`
	RetunePending bool           `json:"retune_pending,omitempty"`
	EWMATTFTms    float64        `json:"ewma_ttft_ms,omitempty"`
	EWMADecode    float64        `json:"ewma_decode_tok_s,omitempty"`
	Sample        map[string]any `json:"sample,omitempty"`
}

// Stats returns a deep-enough copy for JSON.
func (c *Controller) Stats() Stats {
	c.mu.Lock()
	defer c.mu.Unlock()
	st := Stats{
		Enabled: c.cfg.Enabled, Policy: c.cfg.Policy, Knobs: c.knobs,
		InFlight: c.active, Queued: c.waiting, Admitted: c.admitted, Rejected: c.rejected, Canceled: c.canceled,
		LastReject: c.lastReject, EWMATTFTms: round2(c.ewmaTTFT), EWMADecode: round2(c.ewmaDecode),
		RetunePending: c.retuneDue,
	}
	if !c.lastRejectAt.IsZero() {
		st.LastRejectAt = float64(c.lastRejectAt.UnixNano()) / 1e9
	}
	st.Decisions = append([]Decision{}, c.decisions...)
	if c.pending != nil {
		st.Pending = map[string]any{"from": c.pending.from, "to": c.pending.to, "until": c.pending.until.Unix()}
	}
	if c.have {
		s := c.last
		// A missing reading is omitted, not published as a zero that a reader could mistake for a measurement.
		engineState := s.EngineState
		if engineState == "" {
			engineState = "unknown"
		}
		st.Sample = map[string]any{
			"p95_ttft_ms": s.P95TTFTms, "prompt_p95": s.PromptP95,
			"engine_state": engineState, "device_failed": s.DeviceFailed,
		}
		if s.KnownCPU {
			st.Sample["cpu_percent"] = round2(s.CPUPercent)
		}
		if s.KnownRAM {
			st.Sample["ram_percent"] = round2(s.RAMUsedPercent)
			st.Sample["ram_available_bytes"] = s.RAMAvailableByte
		}
		if s.KnownVRAM {
			st.Sample["vram_percent"] = round2(s.VRAMUsedPercent)
			st.Sample["vram_free_bytes"] = s.VRAMFreeBytes
		}
		if s.GPUTempC > 0 {
			st.Sample["gpu_temp_c"] = round2(s.GPUTempC)
		}
	}
	return st
}

func round2(v float64) float64 { return math.Round(v*100) / 100 }

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// sortDecisions is used by tests to make ordering explicit.
func sortDecisions(d []Decision) {
	sort.SliceStable(d, func(i, j int) bool { return d[i].At.Before(d[j].At) })
}
