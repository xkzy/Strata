// pkg/hallucination/contradiction_detector.go - Deterministic Contradiction & Inconsistency Detection
package hallucination

import (
	"fmt"
	"strings"

	"strata/pkg/logicverifier"
	"strata/pkg/mathruntime"
)

// ContradictionDetector evaluates whether a claim directly conflicts with authoritative evidence
type ContradictionDetector struct {
	mathRuntime  *mathruntime.MathRuntime
	unitVerifier *logicverifier.UnitVerifier
}

// NewContradictionDetector initializes the contradiction detector
func NewContradictionDetector(mr *mathruntime.MathRuntime, uv *logicverifier.UnitVerifier) *ContradictionDetector {
	if mr == nil {
		mr = mathruntime.NewMathRuntime(nil)
	}
	if uv == nil {
		uv = logicverifier.NewUnitVerifier()
	}
	return &ContradictionDetector{
		mathRuntime:  mr,
		unitVerifier: uv,
	}
}

// DetectContradiction inspects a claim against available evidence and symbolic solvers
func (cd *ContradictionDetector) DetectContradiction(claim *Claim, graph *EvidenceGraph) (bool, Evidence, string) {
	if claim == nil {
		return false, Evidence{}, ""
	}

	// 1. Math equations verification
	if claim.Type == TypeMathematical {
		expr := claim.SourceSpan
		parts := strings.Split(expr, "=")
		if len(parts) == 2 {
			lhs := strings.TrimSpace(parts[0])
			rhs := strings.TrimSpace(parts[1])

			resLHS := cd.mathRuntime.Evaluate(lhs, mathruntime.ModeExact, "", "")
			resRHS := cd.mathRuntime.Evaluate(rhs, mathruntime.ModeExact, "", "")

			if resLHS.Status == mathruntime.StatusSuccess && resRHS.Status == mathruntime.StatusSuccess {
				expected := resLHS.ExactResult
				if expected == "" {
					expected = resLHS.NumericResult
				}
				actual := resRHS.ExactResult
				if actual == "" {
					actual = resRHS.NumericResult
				}

				if expected != "" && actual != "" && expected != actual {
					ev := Evidence{
						ID:            fmt.Sprintf("ev-math-mismatch-%s", claim.ID),
						SourceType:    SourceMathicsResult,
						SourceID:      "fast_numeric_cas",
						Authority:     AuthorityTrustedSystem,
						ExtractedFact: fmt.Sprintf("Expression %s evaluates to %s, but claimed %s (evaluates to %s)", lhs, expected, rhs, actual),
						Provenance:    "MathCASVerification",
						Confidence:    1.0,
					}
					return true, ev, fmt.Sprintf("Calculated %s != claimed %s", expected, actual)
				}
			}
		}
	}

	// 2. Unit & Dimensional verification
	if claim.Type == TypeQuantitative {
		parts := strings.Split(claim.SourceSpan, "=")
		if len(parts) == 2 {
			uRes := cd.unitVerifier.Verify(logicverifier.VerificationClaim{
				ClaimID:      claim.ID,
				Expression:   strings.TrimSpace(parts[0]),
				ClaimedValue: strings.TrimSpace(parts[1]),
			})
			if uRes.Status == logicverifier.StatusFail {
				ev := Evidence{
					ID:            fmt.Sprintf("ev-unit-mismatch-%s", claim.ID),
					SourceType:    SourceMathicsResult,
					SourceID:      "unit_verifier",
					Authority:     AuthorityTrustedSystem,
					ExtractedFact: fmt.Sprintf("Dimensional or numeric discrepancy: %s (%s)", claim.SourceSpan, uRes.FailureReason),
					Provenance:    "UnitDimensionalVerifier",
					Confidence:    1.0,
				}
				return true, ev, uRes.FailureReason
			}
		}
	}

	// 3. Configuration & State Contradictions in EvidenceGraph
	if graph != nil {
		allEvidence := graph.GetAllEvidence()
		for _, ev := range allEvidence {
			// Check port contradictions
			if claim.Type == TypeConfiguration && claim.Subject == "server_port" && ev.SourceType == SourceConfiguration && strings.Contains(ev.SourceID, "port") {
				if ev.ExtractedFact != "" && !strings.Contains(ev.ExtractedFact, claim.Object) {
					return true, ev, fmt.Sprintf("Claimed port %s conflicts with configured %s", claim.Object, ev.ExtractedFact)
				}
			}

			// Check tool result contradictions
			if claim.Type == TypeToolResult && ev.SourceType == SourceToolResult {
				if ev.ExtractedFact != "" && !strings.EqualFold(ev.ExtractedFact, claim.Object) && strings.Contains(ev.ExtractedFact, "error") {
					return true, ev, fmt.Sprintf("Claimed success but tool execution returned: %s", ev.ExtractedFact)
				}
			}
		}
	}

	return false, Evidence{}, ""
}
