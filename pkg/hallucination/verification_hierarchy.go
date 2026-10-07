// pkg/hallucination/verification_hierarchy.go - Deterministic Verification Hierarchy & Planner
package hallucination

import (
	"fmt"
	"strings"

	"strata/pkg/logicverifier"
	"strata/pkg/mathruntime"
)

// VerificationPlanner coordinates deterministic solvers across the 6-stage hierarchy
type VerificationPlanner struct {
	codeVerifier *CodeVerifier
	contraDetect *ContradictionDetector
	mathRuntime  *mathruntime.MathRuntime
	unitVerifier *logicverifier.UnitVerifier
	ruleEngine   *logicverifier.RuleEngine
	cache        *VerificationCache
}

// NewVerificationPlanner initializes the verification hierarchy
func NewVerificationPlanner(workspaceRoot string) *VerificationPlanner {
	mr := mathruntime.NewMathRuntime(nil)
	uv := logicverifier.NewUnitVerifier()
	re := logicverifier.NewRuleEngine()
	cv := NewCodeVerifier(workspaceRoot)
	cd := NewContradictionDetector(mr, uv)
	cache := NewVerificationCache(0, 10000)

	return &VerificationPlanner{
		codeVerifier: cv,
		contraDetect: cd,
		mathRuntime:  mr,
		unitVerifier: uv,
		ruleEngine:   re,
		cache:        cache,
	}
}

// VerifyClaim executes the deterministic hierarchy on a claim
func (vp *VerificationPlanner) VerifyClaim(claim *Claim, graph *EvidenceGraph) ClaimExplanation {
	if claim == nil {
		return ClaimExplanation{Status: StatusUnknown, RiskLevel: RiskLow}
	}

	// 0. Cache check
	if entry, ok := vp.cache.Get(claim); ok {
		claim.Status = entry.Status
		claim.EvidenceRefs = entry.EvidenceRefs
		return entry.Explanation
	}

	// 1. Contradiction Detection check across all claims
	hasContra, contraEv, contraReason := vp.contraDetect.DetectContradiction(claim, graph)
	if hasContra {
		claim.Status = StatusContradicted
		claim.Provenance = contraEv.Provenance
		if graph != nil {
			graph.AddEvidence(contraEv)
			_ = graph.Link(claim.ID, contraEv.ID, EdgeContradictedBy, 1.0, contraReason)
		}
		exp := ClaimExplanation{
			ClaimID:           claim.ID,
			ClaimText:         claim.SourceSpan,
			Status:            StatusContradicted,
			RiskLevel:         RiskCritical,
			VerifierUsed:      contraEv.Provenance,
			Contradictions:    []string{fmt.Sprintf("[%s] %s (Reason: %s)", contraEv.SourceType, contraEv.ExtractedFact, contraReason)},
			FailureReason:     contraReason,
			RecommendedAction: "Regenerate span or alert user to factual contradiction",
		}
		vp.cache.Put(claim, StatusContradicted, []string{contraEv.ID}, exp)
		return exp
	}

	// 2. Code, File & Symbol Verification
	switch claim.Type {
	case TypeFile:
		st, ev, _ := vp.codeVerifier.VerifyFileClaim(claim)
		claim.Status = st
		claim.Provenance = ev.Provenance
		if graph != nil {
			graph.AddEvidence(ev)
			if st == StatusVerified {
				_ = graph.Link(claim.ID, ev.ID, EdgeVerifiedBy, 1.0, "File exists on disk")
			} else {
				_ = graph.Link(claim.ID, ev.ID, EdgeContradictedBy, 1.0, "File does not exist")
			}
		}
		exp := vp.buildExplanation(claim, st, ev.Provenance, []string{ev.ExtractedFact}, nil, "")
		vp.cache.Put(claim, st, []string{ev.ID}, exp)
		return exp

	case TypeFunction, TypeSymbol:
		st, ev, _ := vp.codeVerifier.VerifySymbolClaim(claim)
		claim.Status = st
		claim.Provenance = ev.Provenance
		if graph != nil {
			graph.AddEvidence(ev)
			if st == StatusVerified {
				_ = graph.Link(claim.ID, ev.ID, EdgeVerifiedBy, 1.0, "Symbol exists in AST")
			} else {
				_ = graph.Link(claim.ID, ev.ID, EdgeContradictedBy, 1.0, "Symbol not found in AST")
			}
		}
		exp := vp.buildExplanation(claim, st, ev.Provenance, []string{ev.ExtractedFact}, nil, "")
		vp.cache.Put(claim, st, []string{ev.ID}, exp)
		return exp

	case TypeAPI:
		st, ev, _ := vp.codeVerifier.VerifyAPIClaim(claim)
		claim.Status = st
		claim.Provenance = ev.Provenance
		if graph != nil {
			graph.AddEvidence(ev)
			if st == StatusVerified {
				_ = graph.Link(claim.ID, ev.ID, EdgeVerifiedBy, 1.0, "API Route registered")
			} else {
				_ = graph.Link(claim.ID, ev.ID, EdgeContradictedBy, 1.0, "API Route not registered")
			}
		}
		exp := vp.buildExplanation(claim, st, ev.Provenance, []string{ev.ExtractedFact}, nil, "")
		vp.cache.Put(claim, st, []string{ev.ID}, exp)
		return exp

	case TypeMathematical:
		expr := claim.SourceSpan
		parts := strings.Split(expr, "=")
		if len(parts) == 2 {
			lhs := strings.TrimSpace(parts[0])
			rhs := strings.TrimSpace(parts[1])

			resL := vp.mathRuntime.Evaluate(lhs, mathruntime.ModeExact, "", "")
			resR := vp.mathRuntime.Evaluate(rhs, mathruntime.ModeExact, "", "")

			if resL.Status == mathruntime.StatusSuccess && resR.Status == mathruntime.StatusSuccess {
				expected := resL.ExactResult
				if expected == "" {
					expected = resL.NumericResult
				}
				actual := resR.ExactResult
				if actual == "" {
					actual = resR.NumericResult
				}

				if expected != "" && actual != "" && expected == actual {
					ev := Evidence{
						ID:            fmt.Sprintf("ev-math-%s", claim.ID),
						SourceType:    SourceMathicsResult,
						SourceID:      "fast_numeric_cas",
						Authority:     AuthorityTrustedSystem,
						ExtractedFact: fmt.Sprintf("%s = %s evaluated correctly to %s", lhs, rhs, expected),
						Provenance:    "MathCASVerification",
						Confidence:    1.0,
					}
					claim.Status = StatusVerified
					claim.Provenance = ev.Provenance
					if graph != nil {
						graph.AddEvidence(ev)
						_ = graph.Link(claim.ID, ev.ID, EdgeVerifiedBy, 1.0, "Exact math match")
					}
					exp := vp.buildExplanation(claim, StatusVerified, ev.Provenance, []string{ev.ExtractedFact}, nil, "")
					vp.cache.Put(claim, StatusVerified, []string{ev.ID}, exp)
					return exp
				}
			}
		}

	case TypeQuantitative:
		parts := strings.Split(claim.SourceSpan, "=")
		if len(parts) == 2 {
			uRes := vp.unitVerifier.Verify(logicverifier.VerificationClaim{
				ClaimID:      claim.ID,
				Expression:   strings.TrimSpace(parts[0]),
				ClaimedValue: strings.TrimSpace(parts[1]),
			})
			if uRes.Status == logicverifier.StatusPass {
				ev := Evidence{
					ID:            fmt.Sprintf("ev-unit-%s", claim.ID),
					SourceType:    SourceMathicsResult,
					SourceID:      "unit_verifier",
					Authority:     AuthorityTrustedSystem,
					ExtractedFact: fmt.Sprintf("Dimensional and numeric match: %s", claim.SourceSpan),
					Provenance:    "UnitDimensionalVerifier",
					Confidence:    1.0,
				}
				claim.Status = StatusVerified
				claim.Provenance = ev.Provenance
				if graph != nil {
					graph.AddEvidence(ev)
					_ = graph.Link(claim.ID, ev.ID, EdgeVerifiedBy, 1.0, "Unit match")
				}
				exp := vp.buildExplanation(claim, StatusVerified, ev.Provenance, []string{ev.ExtractedFact}, nil, "")
				vp.cache.Put(claim, StatusVerified, []string{ev.ID}, exp)
				return exp
			}
		}

	case TypeLogical:
		// Attempt logical derivation with RuleEngine
		lRes := vp.ruleEngine.VerifyDeduction([]string{claim.Subject}, claim.Object, claim.ID)
		if lRes.Status == logicverifier.StatusPass {
			ev := Evidence{
				ID:            fmt.Sprintf("ev-logic-%s", claim.ID),
				SourceType:    SourcePreviousVerifiedState,
				SourceID:      "rule_engine",
				Authority:     AuthorityTrustedSystem,
				ExtractedFact: fmt.Sprintf("Derived %s from premise %s via Modus Ponens", claim.Object, claim.Subject),
				Provenance:    "PropositionalRuleEngine",
				Confidence:    1.0,
			}
			claim.Status = StatusVerified
			claim.Provenance = ev.Provenance
			if graph != nil {
				graph.AddEvidence(ev)
				_ = graph.Link(claim.ID, ev.ID, EdgeVerifiedBy, 1.0, "Logic derivation")
			}
			exp := vp.buildExplanation(claim, StatusVerified, ev.Provenance, []string{ev.ExtractedFact}, nil, "")
			vp.cache.Put(claim, StatusVerified, []string{ev.ID}, exp)
			return exp
		} else if lRes.Status == logicverifier.StatusFail {
			ev := Evidence{
				ID:            fmt.Sprintf("ev-logic-invalid-%s", claim.ID),
				SourceType:    SourcePreviousVerifiedState,
				SourceID:      "rule_engine",
				Authority:     AuthorityTrustedSystem,
				ExtractedFact: fmt.Sprintf("Deduction %s from %s is logically invalid: %s", claim.Object, claim.Subject, lRes.FailureReason),
				Provenance:    "PropositionalRuleEngine",
				Confidence:    1.0,
			}
			claim.Status = StatusContradicted
			claim.Provenance = ev.Provenance
			if graph != nil {
				graph.AddEvidence(ev)
				_ = graph.Link(claim.ID, ev.ID, EdgeContradictedBy, 1.0, lRes.FailureReason)
			}
			exp := vp.buildExplanation(claim, StatusContradicted, ev.Provenance, nil, []string{ev.ExtractedFact}, lRes.FailureReason)
			vp.cache.Put(claim, StatusContradicted, []string{ev.ID}, exp)
			return exp
		}
		// If cannot derive logically, remains UNKNOWN (Never convert to FALSE/CONTRADICTED!)
		claim.Status = StatusUnknown
		claim.Provenance = "PropositionalRuleEngine"
		exp := vp.buildExplanation(claim, StatusUnknown, "PropositionalRuleEngine", nil, nil, "Cannot formalize or derive logical proof")
		return exp

	case TypeOpinion, TypeInference, TypePrediction:
		claim.Status = StatusUnknown
		claim.Provenance = "OpinionFilter"
		exp := vp.buildExplanation(claim, StatusUnknown, "OpinionFilter", nil, nil, "Statement is non-verifiable opinion or hypothetical")
		return exp
	}

	// 3. Evidence Graph Search for Facts & Config
	if graph != nil {
		allEvidence := graph.GetAllEvidence()
		for _, ev := range allEvidence {
			if strings.Contains(strings.ToLower(ev.ExtractedFact), strings.ToLower(claim.Subject)) ||
				strings.Contains(strings.ToLower(ev.RawContent), strings.ToLower(claim.SourceSpan)) {
				st := StatusSupported
				if ev.Authority >= AuthorityAuthorizedTool {
					st = StatusVerified
				}
				claim.Status = st
				claim.Provenance = "EvidenceGraphMatch"
				_ = graph.Link(claim.ID, ev.ID, EdgeSupportedBy, 0.90, "Matched authoritative evidence")
				exp := vp.buildExplanation(claim, st, "EvidenceGraphMatch", []string{ev.ExtractedFact}, nil, "")
				vp.cache.Put(claim, st, []string{ev.ID}, exp)
				return exp
			}
		}
	}

	// 4. Fallback: If no contradiction and no evidence -> UNSUPPORTED (or UNKNOWN)
	// (Never convert UNSUPPORTED to CONTRADICTED!)
	claim.Status = StatusUnsupported
	claim.Provenance = "HierarchicalVerifierFallback"
	exp := vp.buildExplanation(claim, StatusUnsupported, "HierarchicalVerifierFallback", nil, nil, "No supporting evidence found in authoritative sources")
	return exp
}

func (vp *VerificationPlanner) buildExplanation(c *Claim, st ClaimStatus, verifier string, details, contras []string, failReason string) ClaimExplanation {
	exp := ClaimExplanation{
		ClaimID:         c.ID,
		ClaimText:       c.SourceSpan,
		Status:          st,
		VerifierUsed:    verifier,
		EvidenceDetails: details,
		Contradictions:  contras,
		FailureReason:   failReason,
	}

	switch st {
	case StatusVerified:
		exp.RiskLevel = RiskLow
		exp.RecommendedAction = "Accept claim as verified"
	case StatusSupported:
		exp.RiskLevel = RiskLow
		exp.RecommendedAction = "Accept claim as supported"
	case StatusPartiallySupported:
		exp.RiskLevel = RiskMedium
		exp.RecommendedAction = "Review claim qualifications"
	case StatusContradicted:
		exp.RiskLevel = RiskCritical
		exp.RecommendedAction = "Regenerate span or notify user of contradiction"
	case StatusUnsupported:
		exp.RiskLevel = RiskMedium
		exp.RecommendedAction = "Unverified; request evidence if critical"
	case StatusUnknown:
		exp.RiskLevel = RiskLow
		exp.RecommendedAction = "Non-formalizable or subjective assertion"
	}
	return exp
}
