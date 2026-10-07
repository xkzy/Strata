// pkg/hallucination/types.go - Deterministic Hallucination Detection & Verification Types
package hallucination

import (
	"time"
)

// ClaimStatus represents the deterministic verification status of a claim
type ClaimStatus string

const (
	StatusVerified           ClaimStatus = "VERIFIED"
	StatusSupported          ClaimStatus = "SUPPORTED"
	StatusPartiallySupported ClaimStatus = "PARTIALLY_SUPPORTED"
	StatusContradicted       ClaimStatus = "CONTRADICTED"
	StatusUnsupported        ClaimStatus = "UNSUPPORTED"
	StatusUnknown            ClaimStatus = "UNKNOWN"
)

// ClaimType categorizes the nature of an extracted claim
type ClaimType string

const (
	TypeFact          ClaimType = "FACT"
	TypeEntity        ClaimType = "ENTITY"
	TypeRelation      ClaimType = "RELATION"
	TypeNumber        ClaimType = "NUMBER"
	TypeDate          ClaimType = "DATE"
	TypeTime          ClaimType = "TIME"
	TypeLocation      ClaimType = "LOCATION"
	TypeCausal        ClaimType = "CAUSAL"
	TypeComparison    ClaimType = "COMPARISON"
	TypeQuantitative  ClaimType = "QUANTITATIVE"
	TypeLogical       ClaimType = "LOGICAL"
	TypeMathematical  ClaimType = "MATHEMATICAL"
	TypeCode          ClaimType = "CODE"
	TypeAPI           ClaimType = "API"
	TypeFile          ClaimType = "FILE"
	TypeFunction      ClaimType = "FUNCTION"
	TypeSymbol        ClaimType = "SYMBOL"
	TypeConfiguration ClaimType = "CONFIGURATION"
	TypeCitation      ClaimType = "CITATION"
	TypeToolResult    ClaimType = "TOOL_RESULT"
	TypeProcedure     ClaimType = "PROCEDURE"
	TypePrediction    ClaimType = "PREDICTION"
	TypeOpinion       ClaimType = "OPINION"
	TypeInference     ClaimType = "INFERENCE"
)

// Claim represents a normalized factual or verifiable claim extracted from LLM generation
type Claim struct {
	ID               string      `json:"id"`
	SessionID        string      `json:"session_id"`
	RequestID        string      `json:"request_id"`
	SourceSpan       string      `json:"source_span"`
	TokenRange       [2]int      `json:"token_range"`
	Type             ClaimType   `json:"claim_type"`
	Subject          string      `json:"subject"`
	Predicate        string      `json:"predicate"`
	Object           string      `json:"object"`
	Qualifiers       []string    `json:"qualifiers,omitempty"`
	Conditions       []string    `json:"conditions,omitempty"`
	Assumptions      []string    `json:"assumptions,omitempty"`
	TemporalScope    string      `json:"temporal_scope,omitempty"`
	SpatialScope     string      `json:"spatial_scope,omitempty"`
	Confidence       float64     `json:"confidence"`
	EvidenceRefs     []string    `json:"evidence_refs,omitempty"`
	Status           ClaimStatus `json:"verification_state"`
	ContradictionRefs []string   `json:"contradiction_refs,omitempty"`
	Provenance       string      `json:"provenance,omitempty"`
	CreatedAt        time.Time   `json:"created_at"`
}

// EvidenceSourceType denotes the origin and format of verification evidence
type EvidenceSourceType string

const (
	SourceRAGDocument            EvidenceSourceType = "rag_document"
	SourceCode                   EvidenceSourceType = "source_code"
	SourceDecompiledCode         EvidenceSourceType = "decompiled_code"
	SourceToolResult             EvidenceSourceType = "tool_result"
	SourceDatabaseRecord         EvidenceSourceType = "database_record"
	SourceAPIResponse            EvidenceSourceType = "api_response"
	SourceConfiguration          EvidenceSourceType = "configuration"
	SourceLogs                   EvidenceSourceType = "logs"
	SourceFilesystem             EvidenceSourceType = "filesystem"
	SourceCompilerOutput         EvidenceSourceType = "compiler_output"
	SourceTestResult             EvidenceSourceType = "test_result"
	SourceMathicsResult          EvidenceSourceType = "mathics_result"
	SourceSymbolTable            EvidenceSourceType = "symbol_table"
	SourceSchema                 EvidenceSourceType = "schema"
	SourcePreviousVerifiedState  EvidenceSourceType = "previous_verified_state"
	SourceUserFact               EvidenceSourceType = "user_provided_fact"
	SourceSystemState            EvidenceSourceType = "system_state"
)

// EvidenceAuthority indicates the trust level of an evidence source
type EvidenceAuthority int

const (
	AuthorityModelClaim EvidenceAuthority = 10 // Untrusted
	AuthorityRetrievedSource EvidenceAuthority = 30 // RAG Documents
	AuthorityVerifiedDB     EvidenceAuthority = 50 // Ground Truth DB
	AuthorityAuthoritativeRepo EvidenceAuthority = 70 // Source Repository / Filesystem
	AuthorityAuthorizedTool EvidenceAuthority = 90 // Authorized Tool Results
	AuthorityTrustedSystem  EvidenceAuthority = 100 // System State & Exact Config
)

// Evidence represents authoritative ground truth used to prove, support, or contradict claims
type Evidence struct {
	ID            string             `json:"id"`
	SourceType    EvidenceSourceType `json:"source_type"`
	SourceID      string             `json:"source_id"`
	ContentHash   string             `json:"content_hash"`
	Location      string             `json:"location"`
	SourceVersion string             `json:"source_version"`
	Authority     EvidenceAuthority  `json:"authority"`
	Timestamp     time.Time          `json:"timestamp"`
	AccessScope   string             `json:"access_scope"`
	ExtractedFact string             `json:"extracted_fact"`
	RawContent    string             `json:"raw_content,omitempty"`
	Provenance    string             `json:"provenance"`
	Confidence    float64            `json:"confidence"`
	Immutable     bool               `json:"immutable"`
}

// PolicyMode dictates how hallucination detections govern model responses
type PolicyMode string

const (
	PolicyOff        PolicyMode = "OFF"
	PolicyMonitor    PolicyMode = "MONITOR"
	PolicyWarn       PolicyMode = "WARN"
	PolicyStrict     PolicyMode = "STRICT"
	PolicyRegenerate PolicyMode = "REGENERATE"
	PolicyBlock      PolicyMode = "BLOCK"
)

// GenerationSafetyState represents the active lifecycle state of the safety controller
type GenerationSafetyState string

const (
	StateNormal               GenerationSafetyState = "NORMAL"
	StateEvidenceRequired     GenerationSafetyState = "EVIDENCE_REQUIRED"
	StateSuspicious           GenerationSafetyState = "SUSPICIOUS"
	StateVerificationRequired GenerationSafetyState = "VERIFICATION_REQUIRED"
	StateRecovery             GenerationSafetyState = "RECOVERY"
	StateRegenerating         GenerationSafetyState = "REGENERATING"
	StateBlocked              GenerationSafetyState = "BLOCKED"
	StateCompleted            GenerationSafetyState = "COMPLETED"
)

// RiskLevel categorizes verification urgency
type RiskLevel string

const (
	RiskLow      RiskLevel = "LOW"
	RiskMedium   RiskLevel = "MEDIUM"
	RiskHigh     RiskLevel = "HIGH"
	RiskCritical RiskLevel = "CRITICAL"
)

// ClaimExplanation provides clear, explainable diagnostic evidence for a claim verdict
type ClaimExplanation struct {
	ClaimID           string      `json:"claim_id"`
	ClaimText         string      `json:"claim_text"`
	Status            ClaimStatus `json:"status"`
	RiskLevel         RiskLevel   `json:"risk_level"`
	VerifierUsed      string      `json:"verifier_used"`
	EvidenceDetails   []string    `json:"evidence_details,omitempty"`
	Contradictions    []string    `json:"contradictions,omitempty"`
	FailureReason     string      `json:"failure_reason,omitempty"`
	RecommendedAction string      `json:"recommended_action"`
}

// EvidenceCoverage aggregates verification statistics across all claims in a generation
type EvidenceCoverage struct {
	TotalClaims              int                `json:"total_claims"`
	VerifiedCount            int                `json:"verified_count"`
	SupportedCount           int                `json:"supported_count"`
	PartiallySupportedCount  int                `json:"partially_supported_count"`
	ContradictedCount        int                `json:"contradicted_count"`
	UnsupportedCount         int                `json:"unsupported_count"`
	UnknownCount             int                `json:"unknown_count"`
	CoverageRatio            float64            `json:"coverage_ratio"`
	OverallRisk              RiskLevel          `json:"overall_risk"`
	RecommendedAction        string             `json:"recommended_action"`
	Explanations             []ClaimExplanation `json:"explanations,omitempty"`
}

// HallucinationReport represents the complete diagnostic audit of a generation
type HallucinationReport struct {
	RequestID        string           `json:"request_id"`
	SessionID        string           `json:"session_id"`
	TenantID         string           `json:"tenant_id"`
	Policy           PolicyMode       `json:"policy"`
	Coverage         EvidenceCoverage `json:"coverage"`
	Claims           []Claim          `json:"claims"`
	EvidenceGraphLen int              `json:"evidence_graph_nodes"`
	ExecutionTimeMs  float64          `json:"execution_time_ms"`
	IsClean          bool             `json:"is_clean"`
}
