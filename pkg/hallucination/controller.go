// pkg/hallucination/controller.go - Generation Safety Controller & Circuit Breaker
package hallucination

import (
	"sync"
)

// ControllerAction dictates the runtime action for the generation engine
type ControllerAction string

const (
	ActionContinue   ControllerAction = "CONTINUE"
	ActionWarn       ControllerAction = "WARN"
	ActionRegenerate ControllerAction = "REGENERATE"
	ActionBlock      ControllerAction = "BLOCK"
)

// SafetyControllerOptions configures the controller
type SafetyControllerOptions struct {
	DefaultPolicy       PolicyMode `json:"default_policy"`
	MaxRegenAttempts    int        `json:"max_regen_attempts"`
	StrictCoverageFloor float64    `json:"strict_coverage_floor"`
}

// DefaultSafetyControllerOptions returns sensible defaults
func DefaultSafetyControllerOptions() SafetyControllerOptions {
	return SafetyControllerOptions{
		DefaultPolicy:       PolicyMonitor,
		MaxRegenAttempts:    2,
		StrictCoverageFloor: 0.5,
	}
}

// GenerationSafetyController manages generation safety, policies, dynamic temperature and circuit breakers
type GenerationSafetyController struct {
	mu           sync.RWMutex
	opts         SafetyControllerOptions
	regenCounts  map[string]int                   // sessionID -> attempts
	safetyStates map[string]GenerationSafetyState // sessionID -> state
}

// NewGenerationSafetyController creates a new safety controller
func NewGenerationSafetyController(opts SafetyControllerOptions) *GenerationSafetyController {
	if opts.MaxRegenAttempts <= 0 {
		opts.MaxRegenAttempts = 2
	}
	return &GenerationSafetyController{
		opts:         opts,
		regenCounts:  make(map[string]int),
		safetyStates: make(map[string]GenerationSafetyState),
	}
}

// EvaluateGeneration determines the controller action, temperature adjustment, and state transition
func (gsc *GenerationSafetyController) EvaluateGeneration(sessionID string, report *HallucinationReport) (ControllerAction, float64, string) {
	gsc.mu.Lock()
	defer gsc.mu.Unlock()

	policy := report.Policy
	if policy == "" || policy == PolicyOff {
		return ActionContinue, 0.0, "Policy OFF: bypass verification"
	}

	cov := report.Coverage
	currentState := gsc.safetyStates[sessionID]
	if currentState == "" {
		currentState = StateNormal
	}

	// If no claims extracted or all verified/supported -> clean
	if cov.ContradictedCount == 0 && (cov.UnsupportedCount == 0 || cov.CoverageRatio >= gsc.opts.StrictCoverageFloor) {
		gsc.safetyStates[sessionID] = StateCompleted
		return ActionContinue, 0.0, "Generation clean and verified"
	}

	// If contradiction exists
	if cov.ContradictedCount > 0 {
		switch policy {
		case PolicyMonitor:
			gsc.safetyStates[sessionID] = StateSuspicious
			return ActionContinue, 0.0, "Monitor mode: contradiction noted in audit report"

		case PolicyWarn:
			gsc.safetyStates[sessionID] = StateSuspicious
			return ActionWarn, 0.0, "Warning: output contains factual contradictions"

		case PolicyStrict:
			gsc.safetyStates[sessionID] = StateBlocked
			return ActionBlock, 0.0, "Strict policy: output rejected due to contradictory claims"

		case PolicyRegenerate:
			attempts := gsc.regenCounts[sessionID]
			if attempts >= gsc.opts.MaxRegenAttempts {
				// Circuit breaker tripped: switch from RECOVERY to BLOCK or WARN
				gsc.safetyStates[sessionID] = StateBlocked
				return ActionBlock, 0.0, "Circuit breaker tripped: max regeneration attempts reached"
			}
			gsc.regenCounts[sessionID] = attempts + 1
			gsc.safetyStates[sessionID] = StateRegenerating
			// Lower temperature towards deterministic output for regeneration
			tempDelta := -0.3
			return ActionRegenerate, tempDelta, "Contradiction detected: triggering bounded regeneration"

		case PolicyBlock:
			gsc.safetyStates[sessionID] = StateBlocked
			return ActionBlock, 0.0, "Block policy: output contains contradictions"
		}
	}

	// If unsupported claims exist without contradiction
	if cov.UnsupportedCount > 0 {
		if policy == PolicyStrict && cov.CoverageRatio < gsc.opts.StrictCoverageFloor {
			return ActionWarn, 0.0, "Coverage below threshold for strict policy"
		}
	}

	gsc.safetyStates[sessionID] = StateNormal
	return ActionContinue, 0.0, "Accepted with standard confidence"
}

// ResetSession resets attempt counters for a session
func (gsc *GenerationSafetyController) ResetSession(sessionID string) {
	gsc.mu.Lock()
	defer gsc.mu.Unlock()
	delete(gsc.regenCounts, sessionID)
	delete(gsc.safetyStates, sessionID)
}

// GetSafetyState returns the current state of a session
func (gsc *GenerationSafetyController) GetSafetyState(sessionID string) GenerationSafetyState {
	gsc.mu.RLock()
	defer gsc.mu.RUnlock()
	st, ok := gsc.safetyStates[sessionID]
	if !ok {
		return StateNormal
	}
	return st
}
