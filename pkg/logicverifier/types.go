// pkg/logicverifier/types.go - Logic Verification Type System in Go
package logicverifier

import (
	"sync"
	"time"
)

type VerificationStatus string

const (
	StatusPass    VerificationStatus = "PASS"
	StatusFail    VerificationStatus = "FAIL"
	StatusUnknown VerificationStatus = "UNKNOWN"
	StatusPartial VerificationStatus = "PARTIAL"
)

type ClaimType string

const (
	ClaimArithmetic    ClaimType = "arithmetic"
	ClaimSymbolic      ClaimType = "symbolic"
	ClaimEquation      ClaimType = "equation"
	ClaimProposition   ClaimType = "proposition"
	ClaimConstraint    ClaimType = "constraint"
	ClaimConsistency   ClaimType = "consistency"
	ClaimUnitDimension ClaimType = "unit_dimension"
	ClaimSchemaType    ClaimType = "schema_type"
	ClaimCodeSyntax    ClaimType = "code_syntax"
	ClaimSQL           ClaimType = "sql"
	ClaimComposite     ClaimType = "composite"
	ClaimUnknown       ClaimType = "unknown"
)

type NumericalTolerance struct {
	AbsTol    float64 `json:"abs_tol"`
	RelTol    float64 `json:"rel_tol"`
	ExactOnly bool    `json:"exact_only"`
}

type VerificationClaim struct {
	ClaimID        string              `json:"claim_id"`
	Type           ClaimType           `json:"type"`
	RawStatement   string              `json:"raw_statement,omitempty"`
	Expression     string              `json:"expression"`
	ClaimedValue   string              `json:"claimed_value"`
	Premises       []string            `json:"premises,omitempty"`
	Constraints    []string            `json:"constraints,omitempty"`
	SchemaJSON     string              `json:"schema_json,omitempty"`
	CodeSnippet    string              `json:"code_snippet,omitempty"`
	Language       string              `json:"language,omitempty"`
	SQLQuery       string              `json:"sql_query,omitempty"`
	UnitExpression string              `json:"unit_expression,omitempty"`
	ClaimedUnit    string              `json:"claimed_unit,omitempty"`
	SubClaims      []VerificationClaim `json:"sub_claims,omitempty"`
	TenantID       string              `json:"tenant_id,omitempty"`
	SessionID      string              `json:"session_id,omitempty"`
	UserID         string              `json:"user_id,omitempty"`
	Tolerance      NumericalTolerance  `json:"tolerance"`
	Metadata       map[string]string   `json:"metadata,omitempty"`
}

type VerificationResult struct {
	ClaimID            string             `json:"claim_id"`
	Status             VerificationStatus `json:"status"`
	Type               ClaimType          `json:"type"`
	RawStatement       string             `json:"raw_statement,omitempty"`
	BackendUsed        string             `json:"backend_used"`
	Evidence           string             `json:"evidence,omitempty"`
	FailureReason      string             `json:"failure_reason,omitempty"`
	ExpectedValue      string             `json:"expected_value,omitempty"`
	ActualValue        string             `json:"actual_value,omitempty"`
	VerifiedComponents []string           `json:"verified_components,omitempty"`
	FailedComponents   []string           `json:"failed_components,omitempty"`
	UnknownComponents  []string           `json:"unknown_components,omitempty"`
	CompactObservation string             `json:"compact_observation"`
	ObservationTokens  int64              `json:"observation_tokens"`
	ExecutionTimeMs    float64            `json:"execution_time_ms"`
	CacheHit           bool               `json:"cache_hit"`
	CanonicalHash      string             `json:"canonical_hash,omitempty"`
	Timestamp          time.Time          `json:"timestamp"`
}

type VerificationMetrics struct {
	TotalVerifications   uint64            `json:"total_verifications"`
	PassCount            uint64            `json:"pass_count"`
	FailCount            uint64            `json:"fail_count"`
	UnknownCount         uint64            `json:"unknown_count"`
	PartialCount         uint64            `json:"partial_count"`
	CacheHits            uint64            `json:"cache_hits"`
	CacheMisses          uint64            `json:"cache_misses"`
	TotalExecutionTimeMs float64           `json:"total_execution_time_ms"`
	BackendDistribution  map[string]uint64 `json:"backend_distribution"`
}

type MetricsCollector struct {
	mu      sync.Mutex
	metrics VerificationMetrics
}

func NewMetricsCollector() *MetricsCollector {
	return &MetricsCollector{
		metrics: VerificationMetrics{
			BackendDistribution: make(map[string]uint64),
		},
	}
}

func (mc *MetricsCollector) Record(res VerificationResult) {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	mc.metrics.TotalVerifications++
	switch res.Status {
	case StatusPass:
		mc.metrics.PassCount++
	case StatusFail:
		mc.metrics.FailCount++
	case StatusPartial:
		mc.metrics.PartialCount++
	default:
		mc.metrics.UnknownCount++
	}

	if res.CacheHit {
		mc.metrics.CacheHits++
	} else {
		mc.metrics.CacheMisses++
	}

	mc.metrics.TotalExecutionTimeMs += res.ExecutionTimeMs
	if res.BackendUsed != "" {
		mc.metrics.BackendDistribution[res.BackendUsed]++
	}
}

func (mc *MetricsCollector) Get() VerificationMetrics {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	m := mc.metrics
	m.BackendDistribution = make(map[string]uint64)
	for k, v := range mc.metrics.BackendDistribution {
		m.BackendDistribution[k] = v
	}
	return m
}
