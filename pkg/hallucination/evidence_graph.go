// pkg/hallucination/evidence_graph.go - Directed Thread-Safe Evidence & Claim Provenance Graph
package hallucination

import (
	"fmt"
	"sync"
	"time"
)

// EdgeType defines the relationship between claims and evidence or other claims
type EdgeType string

const (
	EdgeSupportedBy    EdgeType = "SUPPORTED_BY"
	EdgeContradictedBy EdgeType = "CONTRADICTED_BY"
	EdgeVerifiedBy     EdgeType = "VERIFIED_BY"
	EdgeDerivedFrom    EdgeType = "DERIVED_FROM"
)

// GraphEdge represents a directed, typed link in the evidence graph
type GraphEdge struct {
	FromID     string    `json:"from_id"` // Claim ID or Child Claim ID
	ToID       string    `json:"to_id"`   // Evidence ID or Parent Claim ID
	Type       EdgeType  `json:"type"`    // SUPPORTED_BY, CONTRADICTED_BY, VERIFIED_BY, DERIVED_FROM
	Confidence float64   `json:"confidence"`
	CreatedAt  time.Time `json:"created_at"`
	Reason     string    `json:"reason,omitempty"`
}

// EvidenceGraph manages directed relationships between Claims and Evidence
type EvidenceGraph struct {
	mu           sync.RWMutex
	claims       map[string]*Claim
	evidence     map[string]*Evidence
	outEdges     map[string][]GraphEdge // fromID -> edges
	inEdges      map[string][]GraphEdge // toID -> edges
	sessionIndex map[string][]string    // sessionID -> claimIDs
	tenantIndex  map[string][]string    // tenantID -> claimIDs
}

// NewEvidenceGraph initializes an in-memory thread-safe evidence graph
func NewEvidenceGraph() *EvidenceGraph {
	return &EvidenceGraph{
		claims:       make(map[string]*Claim),
		evidence:     make(map[string]*Evidence),
		outEdges:     make(map[string][]GraphEdge),
		inEdges:      make(map[string][]GraphEdge),
		sessionIndex: make(map[string][]string),
		tenantIndex:  make(map[string][]string),
	}
}

// AddEvidence stores an authoritative evidence item in the graph
func (g *EvidenceGraph) AddEvidence(ev Evidence) {
	g.mu.Lock()
	defer g.mu.Unlock()

	if ev.Timestamp.IsZero() {
		ev.Timestamp = time.Now()
	}
	g.evidence[ev.ID] = &ev
}

// AddClaim registers a claim node in the graph
func (g *EvidenceGraph) AddClaim(claim Claim, tenantID string) {
	g.mu.Lock()
	defer g.mu.Unlock()

	if claim.CreatedAt.IsZero() {
		claim.CreatedAt = time.Now()
	}
	g.claims[claim.ID] = &claim

	if claim.SessionID != "" {
		g.sessionIndex[claim.SessionID] = append(g.sessionIndex[claim.SessionID], claim.ID)
	}
	if tenantID != "" {
		g.tenantIndex[tenantID] = append(g.tenantIndex[tenantID], claim.ID)
	}
}

// Link connects a claim to an evidence node or another claim
func (g *EvidenceGraph) Link(fromID, toID string, edgeType EdgeType, confidence float64, reason string) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	edge := GraphEdge{
		FromID:     fromID,
		ToID:       toID,
		Type:       edgeType,
		Confidence: confidence,
		CreatedAt:  time.Now(),
		Reason:     reason,
	}

	g.outEdges[fromID] = append(g.outEdges[fromID], edge)
	g.inEdges[toID] = append(g.inEdges[toID], edge)

	// Update claim reference fields if applicable
	if c, ok := g.claims[fromID]; ok {
		switch edgeType {
		case EdgeSupportedBy, EdgeVerifiedBy:
			c.EvidenceRefs = append(c.EvidenceRefs, toID)
		case EdgeContradictedBy:
			c.ContradictionRefs = append(c.ContradictionRefs, toID)
		}
	}
	return nil
}

// GetClaim retrieves a claim by ID
func (g *EvidenceGraph) GetClaim(claimID string) (*Claim, bool) {
	g.mu.RLock()
	defer g.mu.RUnlock()

	c, ok := g.claims[claimID]
	if !ok {
		return nil, false
	}
	// Return shallow copy
	res := *c
	return &res, true
}

// GetEvidence retrieves an evidence record by ID
func (g *EvidenceGraph) GetEvidence(evID string) (*Evidence, bool) {
	g.mu.RLock()
	defer g.mu.RUnlock()

	ev, ok := g.evidence[evID]
	if !ok {
		return nil, false
	}
	res := *ev
	return &res, true
}

// GetAllEvidence returns all evidence currently in the graph
func (g *EvidenceGraph) GetAllEvidence() []Evidence {
	g.mu.RLock()
	defer g.mu.RUnlock()

	res := make([]Evidence, 0, len(g.evidence))
	for _, ev := range g.evidence {
		res = append(res, *ev)
	}
	return res
}

// GetSupportingEvidence finds all evidence supporting a given claim
func (g *EvidenceGraph) GetSupportingEvidence(claimID string) []Evidence {
	g.mu.RLock()
	defer g.mu.RUnlock()

	var result []Evidence
	edges := g.outEdges[claimID]
	for _, edge := range edges {
		if edge.Type == EdgeSupportedBy || edge.Type == EdgeVerifiedBy {
			if ev, ok := g.evidence[edge.ToID]; ok {
				result = append(result, *ev)
			}
		}
	}
	return result
}

// GetContradictingEvidence finds all evidence contradicting a given claim
func (g *EvidenceGraph) GetContradictingEvidence(claimID string) []Evidence {
	g.mu.RLock()
	defer g.mu.RUnlock()

	var result []Evidence
	edges := g.outEdges[claimID]
	for _, edge := range edges {
		if edge.Type == EdgeContradictedBy {
			if ev, ok := g.evidence[edge.ToID]; ok {
				result = append(result, *ev)
			}
		}
	}
	return result
}

// ExplainClaim produces a human and machine-readable explanation of why a claim received its status
func (g *EvidenceGraph) ExplainClaim(claimID string) ClaimExplanation {
	g.mu.RLock()
	defer g.mu.RUnlock()

	claim, ok := g.claims[claimID]
	if !ok {
		return ClaimExplanation{
			ClaimID:           claimID,
			Status:            StatusUnknown,
			RiskLevel:         RiskLow,
			FailureReason:     "Claim not found in evidence graph",
			RecommendedAction: "None",
		}
	}

	exp := ClaimExplanation{
		ClaimID:      claim.ID,
		ClaimText:    claim.SourceSpan,
		Status:       claim.Status,
		VerifierUsed: claim.Provenance,
	}

	// Calculate risk level
	switch claim.Status {
	case StatusContradicted:
		exp.RiskLevel = RiskCritical
		exp.RecommendedAction = "Regenerate span or warn user: claim directly contradicts authoritative evidence"
	case StatusUnsupported:
		exp.RiskLevel = RiskMedium
		exp.RecommendedAction = "Evidence not found; consider requesting retrieval or verifying"
	case StatusUnknown:
		exp.RiskLevel = RiskLow
		exp.RecommendedAction = "Claim cannot be formalized or deterministically verified"
	case StatusVerified, StatusSupported:
		exp.RiskLevel = RiskLow
		exp.RecommendedAction = "Accept claim as verified/supported"
	case StatusPartiallySupported:
		exp.RiskLevel = RiskLow
		exp.RecommendedAction = "Accept with qualification"
	}

	// Trace edges
	edges := g.outEdges[claimID]
	for _, edge := range edges {
		if edge.Type == EdgeSupportedBy || edge.Type == EdgeVerifiedBy {
			if ev, exists := g.evidence[edge.ToID]; exists {
				exp.EvidenceDetails = append(exp.EvidenceDetails,
					fmt.Sprintf("[%s] %s (Authority: %d, Provenance: %s)", ev.SourceType, ev.ExtractedFact, ev.Authority, ev.Provenance))
			}
		} else if edge.Type == EdgeContradictedBy {
			if ev, exists := g.evidence[edge.ToID]; exists {
				exp.Contradictions = append(exp.Contradictions,
					fmt.Sprintf("[%s] CONTRADICTS with: %s (Source: %s, Reason: %s)", ev.SourceType, ev.ExtractedFact, ev.Location, edge.Reason))
			}
		}
	}

	return exp
}

// Size returns total counts of claims and evidence in the graph
func (g *EvidenceGraph) Size() (int, int) {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return len(g.claims), len(g.evidence)
}
