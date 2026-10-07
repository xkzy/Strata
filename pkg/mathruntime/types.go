// Package mathruntime provides the deterministic mathematical backing runtime,
// calculator router, Mathics3 CAS integration, and exact rational arithmetic for Strata.
package mathruntime

import (
	"time"
)

type MathOperation string

const (
	OpEvaluate        MathOperation = "evaluate"
	OpSimplify        MathOperation = "simplify"
	OpExpand          MathOperation = "expand"
	OpFactor          MathOperation = "factor"
	OpSolve           MathOperation = "solve"
	OpDifferentiate   MathOperation = "differentiate"
	OpIntegrate       MathOperation = "integrate"
	OpLimit           MathOperation = "limit"
	OpSeries          MathOperation = "series"
	OpNumericEvaluate MathOperation = "numeric_evaluate"
	OpMatrixOp        MathOperation = "matrix_op"
	OpDeterminant     MathOperation = "determinant"
	OpProbability     MathOperation = "probability"
)

type MathMode string

const (
	ModeExact    MathMode = "exact"
	ModeNumeric  MathMode = "numeric"
	ModeSymbolic MathMode = "symbolic"
)

type MathStatus string

const (
	StatusSuccess              MathStatus = "success"
	StatusTimeout              MathStatus = "timeout"
	StatusInvalidExpression    MathStatus = "invalid_expression"
	StatusExecutionError       MathStatus = "execution_error"
	StatusCancelled            MathStatus = "cancelled"
	StatusResourceLimitExceed MathStatus = "resource_limit_exceeded"
	StatusUnsupportedOp        MathStatus = "unsupported_operation"
)

type MathSecurityLimits struct {
	MaxExpressionLength      int
	MaxComplexityScore       int
	Timeout                  time.Duration
	MaxRecursionDepth        int
	MaxMemoryMB              int
	MaxConcurrentEvaluations int
}

func DefaultSecurityLimits() MathSecurityLimits {
	return MathSecurityLimits{
		MaxExpressionLength:      4096,
		MaxComplexityScore:       10000,
		Timeout:                  5 * time.Second,
		MaxRecursionDepth:        50,
		MaxMemoryMB:              512,
		MaxConcurrentEvaluations: 8,
	}
}

type MathRequest struct {
	RequestID       string        `json:"request_id"`
	TenantID        string        `json:"tenant_id"`
	UserID          string        `json:"user_id"`
	ProjectID       string        `json:"project_id"`
	SessionID       string        `json:"session_id"`
	AgentID         string        `json:"agent_id"`
	Operation       MathOperation `json:"operation"`
	Expression      string        `json:"expression"`
	Variable        string        `json:"variable"`
	Point           string        `json:"point"`
	Order           int           `json:"order"`
	Assumptions     string        `json:"assumptions"`
	Mode            MathMode      `json:"mode"`
	PrecisionDigits int           `json:"precision_digits"`
	Timeout         time.Duration `json:"timeout"`
}

type MathResult struct {
	ResultID            string                 `json:"result_id"`
	RequestID           string                 `json:"request_id"`
	Status              MathStatus             `json:"status"`
	BackendName         string                 `json:"backend_name"`
	BackendVersion      string                 `json:"backend_version"`
	CanonicalExpression string                 `json:"canonical_expression"`
	ExactResult         string                 `json:"exact_result"`
	NumericResult       string                 `json:"numeric_result"`
	RawResult           string                 `json:"raw_result"`
	CompactObservation  string                 `json:"compact_observation"`
	ExecutionTimeMs     float64                `json:"execution_time_ms"`
	ComplexityScore     int                    `json:"complexity_score"`
	CacheHit            bool                   `json:"cache_hit"`
	ObservationTokens   int                    `json:"observation_tokens"`
	ErrorMessage        string                 `json:"error_message,omitempty"`
	Metadata            map[string]interface{} `json:"metadata,omitempty"`
}

type MathVerificationResult struct {
	Matches            bool    `json:"matches"`
	LLMClaimedResult   string  `json:"llm_claimed_result"`
	GroundTruthResult  string  `json:"ground_truth_result"`
	DiscrepancyDetails string  `json:"discrepancy_details,omitempty"`
	RelativeError      float64 `json:"relative_error"`
	Confidence         float64 `json:"confidence"`
}

type MathRuntimeStats struct {
	TotalCalculations    uint64  `json:"total_calculations"`
	CacheHits            uint64  `json:"cache_hits"`
	CacheMisses          uint64  `json:"cache_misses"`
	CacheHitRate         float64 `json:"cache_hit_rate"`
	FastPathCount        uint64  `json:"fast_path_count"`
	MathicsCount         uint64  `json:"mathics_count"`
	VerificationCount    uint64  `json:"verification_count"`
	VerificationFailures uint64  `json:"verification_failures"`
	Timeouts             uint64  `json:"timeouts"`
	SecurityRejections   uint64  `json:"security_rejections"`
	AvgExecutionTimeMs   float64 `json:"avg_execution_time_ms"`
	TotalExecutionTimeMs float64 `json:"total_execution_time_ms"`
}
