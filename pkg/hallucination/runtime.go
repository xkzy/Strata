// pkg/hallucination/runtime.go - Top-Level Deterministic Hallucination Runtime
package hallucination

import (
	"context"
	"sync"
	"sync/atomic"
	"time"
)

// HallucinationMetrics tracks system-wide verification counts
type HallucinationMetrics struct {
	TotalEvaluations    uint64 `json:"total_evaluations"`
	TotalClaims         uint64 `json:"total_claims"`
	TotalVerified       uint64 `json:"total_verified"`
	TotalSupported      uint64 `json:"total_supported"`
	TotalContradicted   uint64 `json:"total_contradicted"`
	TotalUnsupported    uint64 `json:"total_unsupported"`
	TotalUnknown        uint64 `json:"total_unknown"`
	RegenerationsForced uint64 `json:"regenerations_forced"`
	BlocksEnacted       uint64 `json:"blocks_enacted"`
}

// HallucinationRuntime orchestrates deterministic hallucination detection and verification
type HallucinationRuntime struct {
	mu         sync.RWMutex
	graph      *EvidenceGraph
	extractor  *ClaimExtractor
	planner    *VerificationPlanner
	controller *GenerationSafetyController
	metrics    HallucinationMetrics
}

// NewRuntime initializes a fully configured HallucinationRuntime
func NewRuntime(workspaceRoot string, opts SafetyControllerOptions) *HallucinationRuntime {
	return &HallucinationRuntime{
		graph:      NewEvidenceGraph(),
		extractor:  NewClaimExtractor(),
		planner:    NewVerificationPlanner(workspaceRoot),
		controller: NewGenerationSafetyController(opts),
	}
}

// AddEvidence registers ground truth evidence into the runtime graph
func (hr *HallucinationRuntime) AddEvidence(ev Evidence) {
	hr.graph.AddEvidence(ev)
}

// EvidenceGraph returns the underlying evidence graph
func (hr *HallucinationRuntime) EvidenceGraph() *EvidenceGraph {
	return hr.graph
}

// Controller returns the safety controller
func (hr *HallucinationRuntime) Controller() *GenerationSafetyController {
	return hr.controller
}

// EvaluateText parses, verifies, audits and evaluates claims in generated text
func (hr *HallucinationRuntime) EvaluateText(ctx context.Context, text, sessionID, requestID, tenantID string, policy PolicyMode) (*HallucinationReport, error) {
	start := time.Now()
	atomic.AddUint64(&hr.metrics.TotalEvaluations, 1)

	if policy == "" {
		policy = hr.controller.opts.DefaultPolicy
	}

	claims := hr.extractor.ExtractClaims(text, sessionID, requestID)
	var explanations []ClaimExplanation

	var vCount, sCount, pCount, cCount, uCount, unkCount int

	for i := range claims {
		c := &claims[i]
		atomic.AddUint64(&hr.metrics.TotalClaims, 1)

		exp := hr.planner.VerifyClaim(c, hr.graph)
		hr.graph.AddClaim(*c, tenantID)
		explanations = append(explanations, exp)

		switch c.Status {
		case StatusVerified:
			vCount++
			atomic.AddUint64(&hr.metrics.TotalVerified, 1)
		case StatusSupported:
			sCount++
			atomic.AddUint64(&hr.metrics.TotalSupported, 1)
		case StatusPartiallySupported:
			pCount++
		case StatusContradicted:
			cCount++
			atomic.AddUint64(&hr.metrics.TotalContradicted, 1)
		case StatusUnsupported:
			uCount++
			atomic.AddUint64(&hr.metrics.TotalUnsupported, 1)
		case StatusUnknown:
			unkCount++
			atomic.AddUint64(&hr.metrics.TotalUnknown, 1)
		}
	}

	total := len(claims)
	coverageRatio := 1.0
	if total > 0 {
		coverageRatio = float64(vCount+sCount+pCount) / float64(total)
	}

	overallRisk := RiskLow
	recommendedAction := "Accept response"
	if cCount > 0 {
		overallRisk = RiskCritical
		recommendedAction = "Contradictions detected; review or regenerate"
	} else if uCount > 0 && coverageRatio < 0.5 {
		overallRisk = RiskMedium
		recommendedAction = "Low evidence coverage"
	}

	nodes, _ := hr.graph.Size()

	coverage := EvidenceCoverage{
		TotalClaims:             total,
		VerifiedCount:           vCount,
		SupportedCount:          sCount,
		PartiallySupportedCount: pCount,
		ContradictedCount:       cCount,
		UnsupportedCount:        uCount,
		UnknownCount:            unkCount,
		CoverageRatio:           coverageRatio,
		OverallRisk:             overallRisk,
		RecommendedAction:       recommendedAction,
		Explanations:            explanations,
	}

	report := &HallucinationReport{
		RequestID:        requestID,
		SessionID:        sessionID,
		TenantID:         tenantID,
		Policy:           policy,
		Coverage:         coverage,
		Claims:           claims,
		EvidenceGraphLen: nodes,
		ExecutionTimeMs:  float64(time.Since(start).Microseconds()) / 1000.0,
		IsClean:          (cCount == 0),
	}

	// Consult safety controller
	action, _, _ := hr.controller.EvaluateGeneration(sessionID, report)
	if action == ActionRegenerate {
		atomic.AddUint64(&hr.metrics.RegenerationsForced, 1)
	} else if action == ActionBlock {
		atomic.AddUint64(&hr.metrics.BlocksEnacted, 1)
	}

	return report, nil
}

// ExplainClaim returns human/machine readable explanation for a specific claim
func (hr *HallucinationRuntime) ExplainClaim(claimID string) ClaimExplanation {
	return hr.graph.ExplainClaim(claimID)
}

// GetMetrics returns aggregate runtime metrics
func (hr *HallucinationRuntime) GetMetrics() HallucinationMetrics {
	return HallucinationMetrics{
		TotalEvaluations:    atomic.LoadUint64(&hr.metrics.TotalEvaluations),
		TotalClaims:         atomic.LoadUint64(&hr.metrics.TotalClaims),
		TotalVerified:       atomic.LoadUint64(&hr.metrics.TotalVerified),
		TotalSupported:      atomic.LoadUint64(&hr.metrics.TotalSupported),
		TotalContradicted:   atomic.LoadUint64(&hr.metrics.TotalContradicted),
		TotalUnsupported:    atomic.LoadUint64(&hr.metrics.TotalUnsupported),
		TotalUnknown:        atomic.LoadUint64(&hr.metrics.TotalUnknown),
		RegenerationsForced: atomic.LoadUint64(&hr.metrics.RegenerationsForced),
		BlocksEnacted:       atomic.LoadUint64(&hr.metrics.BlocksEnacted),
	}
}
