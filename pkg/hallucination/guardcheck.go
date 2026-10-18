// pkg/hallucination/guardcheck.go - the stateless check the server runs on every finished reply.
//
// HallucinationRuntime keeps every claim it sees in a graph that only grows; a guard that runs on each reply needs a
// check with no memory: claims in, findings out. Only arithmetic and unit claims are Actionable (a regenerate is
// justified): they are decided by the CAS, not by a guess. File, function and API claims are checked against the
// server's own working directory, which is not the caller's project, so they are advisory.
package hallucination

import "sync"

// Finding is one claim that failed its check.
type Finding struct {
	ClaimID    string `json:"claim_id"`
	Type       string `json:"type"`
	Span       string `json:"span"`
	Reason     string `json:"reason"`
	Actionable bool   `json:"actionable"`
}

// Checker extracts claims from a reply and checks them. It is safe for concurrent use.
type Checker struct {
	mu        sync.Mutex
	extractor *ClaimExtractor
	detector  *ContradictionDetector
	planner   *VerificationPlanner
}

// NewChecker builds a checker whose code checks look at workspaceRoot.
func NewChecker(workspaceRoot string) *Checker {
	return &Checker{
		extractor: NewClaimExtractor(),
		detector:  NewContradictionDetector(nil, nil),
		planner:   NewVerificationPlanner(workspaceRoot),
	}
}

// Check returns the claims in text that were contradicted, actionable ones first.
func (c *Checker) Check(text, sessionID, requestID string) []Finding {
	c.mu.Lock()
	defer c.mu.Unlock()
	var actionable, advisory []Finding
	for _, claim := range c.extractor.ExtractClaims(text, sessionID, requestID) {
		cl := claim
		switch cl.Type {
		case TypeMathematical, TypeQuantitative:
			if bad, _, why := c.detector.DetectContradiction(&cl, nil); bad {
				actionable = append(actionable, Finding{ClaimID: cl.ID, Type: string(cl.Type), Span: cl.SourceSpan, Reason: why, Actionable: true})
			}
		default:
			exp := c.planner.VerifyClaim(&cl, NewEvidenceGraph()) // a throwaway graph: nothing is kept between replies
			if cl.Status == StatusContradicted {
				advisory = append(advisory, Finding{ClaimID: cl.ID, Type: string(cl.Type), Span: cl.SourceSpan, Reason: exp.FailureReason})
			}
		}
	}
	return append(actionable, advisory...)
}
