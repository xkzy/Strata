// pkg/logicverifier/logic_verifier.go - Top-Level Deterministic Logic Verification Runtime
package logicverifier

import (
	"strings"
	"time"
)

type LogicVerifier struct {
	extractor *ClaimExtractor
	planner   *VerificationPlanner
	cache     *VerificationCache
	metrics   *MetricsCollector
}

func NewLogicVerifier() *LogicVerifier {
	cache := NewVerificationCache(10000, 1*time.Hour)
	metrics := NewMetricsCollector()
	planner := NewVerificationPlanner(cache, metrics)
	extractor := NewClaimExtractor()

	return &LogicVerifier{
		extractor: extractor,
		planner:   planner,
		cache:     cache,
		metrics:   metrics,
	}
}

// Verify evaluates a single structured claim deterministically
func (lv *LogicVerifier) Verify(claim VerificationClaim) VerificationResult {
	return lv.planner.Verify(claim)
}

// ExtractClaims inspects text or JSON payload and extracts candidate claims
func (lv *LogicVerifier) ExtractClaims(text string) []VerificationClaim {
	return lv.extractor.ExtractClaims(text, "", "")
}

// VerifyText extracts all deterministic claims from text and verifies them
func (lv *LogicVerifier) VerifyText(text string, tenantID, sessionID string) []VerificationResult {
	claims := lv.extractor.ExtractClaims(text, tenantID, sessionID)
	results := make([]VerificationResult, 0, len(claims))


	for _, claim := range claims {
		if tenantID != "" {
			claim.TenantID = tenantID
		}
		if sessionID != "" {
			claim.SessionID = sessionID
		}
		res := lv.planner.Verify(claim)
		results = append(results, res)
	}

	return results
}

// FormatObservations creates a compact observation block for virtual context injection
func (lv *LogicVerifier) FormatObservations(results []VerificationResult) string {
	if len(results) == 0 {
		return ""
	}

	var sb strings.Builder
	for _, res := range results {
		if res.CompactObservation != "" {
			sb.WriteString(res.CompactObservation)
			sb.WriteString("\n")
		}
	}
	return strings.TrimSpace(sb.String())
}

// InterceptReasoning detects logical deductions, constraints, units, and fallacies in reasoning text
// and produces verified observations to intercept flawed reasoning and prevent hallucinations.
func (lv *LogicVerifier) InterceptReasoning(text, tenantID, sessionID string) (transformed string, results []VerificationResult, hasViolations bool) {
	results = lv.VerifyText(text, tenantID, sessionID)
	if len(results) == 0 {
		return text, nil, false
	}

	transformed = text
	for _, res := range results {
		if res.Status == StatusFail {
			hasViolations = true
		}
		if res.RawStatement != "" && res.CompactObservation != "" {
			obs := " " + res.CompactObservation
			transformed = strings.Replace(transformed, res.RawStatement, res.RawStatement+obs, 1)
		}
	}

	return transformed, results, hasViolations
}

// GetMetrics returns aggregated verification runtime performance metrics
func (lv *LogicVerifier) GetMetrics() VerificationMetrics {
	return lv.metrics.Get()
}

// ResetCache clears the internal verification result cache
func (lv *LogicVerifier) ResetCache() {
	lv.cache.Clear()
}
