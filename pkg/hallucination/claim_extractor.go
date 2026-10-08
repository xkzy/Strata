// pkg/hallucination/claim_extractor.go - Deterministic Claim Extraction Layer
package hallucination

import (
	"crypto/sha256"
	"fmt"
	"regexp"
	"strings"
	"sync/atomic"
	"time"
)

var (
	// Regex patterns for deterministic extraction
	mathEqRegex      = regexp.MustCompile(`(?i)(?:^|[\s(])([0-9\.\+\-\*\/\^\(\)\s]+=\s*[0-9\.\+\-\*\/\^\(\)\s]+)`)
	unitEqRegex      = regexp.MustCompile(`(?i)([0-9\.]+\s*(?:m|s|kg|g|km|h|min|ms|hz|b|kb|mb|gb|v|a|w|c|k)(?:\/[a-z]+|\s*\*\s*[a-z]+)?\s*[\+\-\*\/]\s*[0-9\.]+\s*(?:m|s|kg|g|km|h|min|ms|hz|b|kb|mb|gb|v|a|w|c|k)(?:\/[a-z]+|\s*\*\s*[a-z]+)?\s*=\s*[0-9\.]+\s*(?:m|s|kg|g|km|h|min|ms|hz|b|kb|mb|gb|v|a|w|c|k)(?:\/[a-z]+|\s*\*\s*[a-z]+)?)`)
	filePathRegex    = regexp.MustCompile(`(?i)(?:file|path|filepath|in|directory|file:)\s+([a-zA-Z0-9_\-\.\/]+\.[a-zA-Z0-9_]+)`)
	standAlonePath   = regexp.MustCompile(`(?i)(?:^|\s)([\.\/]?[a-zA-Z0-9_\-]+(?:\/[a-zA-Z0-9_\-\.]+)+\.[a-zA-Z0-9]+)`)
	funcSymbolRegex  = regexp.MustCompile(`(?i)(?:function|func|method|procedure|symbol)\s+([a-zA-Z_][a-zA-Z0-9_]*)(?:\(\))?`)
	apiEndpointRegex = regexp.MustCompile(`(?i)(?:(?:GET|POST|PUT|DELETE|PATCH|HEAD)\s+(\/[a-zA-Z0-9_\-\.\/]+)|(?:route|endpoint|url|path)\s+(\/[a-zA-Z0-9_\-\.\/]+))`)
	configPortRegex  = regexp.MustCompile(`(?i)(?:port|listen on|host|binding)\s+(?:(?:127\.0\.0\.1|0\.0\.0\.0|localhost):)?([0-9]{2,5})`)
	toolResultRegex  = regexp.MustCompile(`(?i)(?:tool\s+(?:result|output|call)|tool_result|execution result)[:\s]+(.+)`)
	logicalImpRegex  = regexp.MustCompile(`(?i)if\s+(.+?)\s+then\s+(.+)`)
	opinionMarker    = regexp.MustCompile(`(?i)\b(i think|in my opinion|it seems|probably|perhaps|i believe|suppose that|if we were to|hypothetically)\b`)
)

var claimSeq uint64

// ClaimExtractor deterministically extracts factual and verifiable claims from raw text
type ClaimExtractor struct{}

// NewClaimExtractor initializes the claim extractor
func NewClaimExtractor() *ClaimExtractor {
	return &ClaimExtractor{}
}

// ExtractClaims parses generation text and returns structured claims
func (ce *ClaimExtractor) ExtractClaims(text, sessionID, requestID string) []Claim {
	var claims []Claim

	lines := strings.Split(text, "\n")
	for lineIdx, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}

		// Filter out opinions / hypothetical speculation
		if opinionMarker.MatchString(trimmed) {
			claimID := fmt.Sprintf("claim-%d-%d", time.Now().UnixNano(), atomic.AddUint64(&claimSeq, 1))
			claims = append(claims, Claim{
				ID:         claimID,
				SessionID:  sessionID,
				RequestID:  requestID,
				SourceSpan: trimmed,
				TokenRange: [2]int{lineIdx, lineIdx + 1},
				Type:       TypeOpinion,
				Status:     StatusUnknown,
				Subject:    "speaker",
				Predicate:  "expresses_opinion",
				Object:     trimmed,
				Confidence: 1.0,
				Provenance: "OpinionMarkerFilter",
				CreatedAt:  time.Now(),
			})
			continue
		}

		// 1. Math equations
		if matches := mathEqRegex.FindAllStringSubmatch(trimmed, -1); len(matches) > 0 {
			for _, m := range matches {
				expr := strings.TrimSpace(m[1])
				// Ensure it has numbers and operators
				if strings.ContainsAny(expr, "+-*/^") && strings.Contains(expr, "=") {
					claimID := fmt.Sprintf("claim-%d-%d", time.Now().UnixNano(), atomic.AddUint64(&claimSeq, 1))
					claims = append(claims, Claim{
						ID:         claimID,
						SessionID:  sessionID,
						RequestID:  requestID,
						SourceSpan: expr,
						TokenRange: [2]int{lineIdx, lineIdx + 1},
						Type:       TypeMathematical,
						Subject:    expr,
						Predicate:  "evaluates_to",
						Object:     expr,
						Confidence: 1.0,
						Status:     StatusUnknown, // To be verified by Math CAS
						Provenance: "MathEquationExtractor",
						CreatedAt:  time.Now(),
					})
				}
			}
		}

		// 2. Unit / Dimensional Equations
		if matches := unitEqRegex.FindAllStringSubmatch(trimmed, -1); len(matches) > 0 {
			for _, m := range matches {
				expr := strings.TrimSpace(m[1])
				claimID := fmt.Sprintf("claim-%d-%d", time.Now().UnixNano(), atomic.AddUint64(&claimSeq, 1))
				claims = append(claims, Claim{
					ID:         claimID,
					SessionID:  sessionID,
					RequestID:  requestID,
					SourceSpan: expr,
					TokenRange: [2]int{lineIdx, lineIdx + 1},
					Type:       TypeQuantitative,
					Subject:    expr,
					Predicate:  "unit_dimensional_relation",
					Object:     expr,
					Confidence: 1.0,
					Status:     StatusUnknown, // To be verified by UnitVerifier
					Provenance: "UnitEquationExtractor",
					CreatedAt:  time.Now(),
				})
			}
		}

		// 3. File path assertions
		if matches := filePathRegex.FindAllStringSubmatch(trimmed, -1); len(matches) > 0 {
			for _, m := range matches {
				path := strings.TrimSpace(m[1])
				claimID := fmt.Sprintf("claim-%d-%d", time.Now().UnixNano(), atomic.AddUint64(&claimSeq, 1))
				claims = append(claims, Claim{
					ID:         claimID,
					SessionID:  sessionID,
					RequestID:  requestID,
					SourceSpan: trimmed,
					TokenRange: [2]int{lineIdx, lineIdx + 1},
					Type:       TypeFile,
					Subject:    path,
					Predicate:  "exists_in_workspace",
					Object:     path,
					Confidence: 0.95,
					Status:     StatusUnknown,
					Provenance: "FilePathExtractor",
					CreatedAt:  time.Now(),
				})
			}
		} else if standMatches := standAlonePath.FindAllStringSubmatch(trimmed, -1); len(standMatches) > 0 {
			for _, m := range standMatches {
				path := strings.TrimSpace(m[1])
				// Ensure it looks like a valid repo file (has extension, not url)
				if (strings.HasSuffix(path, ".go") || strings.HasSuffix(path, ".h") || strings.HasSuffix(path, ".cpp") ||
					strings.HasSuffix(path, ".md") || strings.HasSuffix(path, ".json") || strings.HasSuffix(path, ".sh") ||
					strings.HasSuffix(path, ".py") || strings.HasSuffix(path, ".bat")) && !strings.HasPrefix(path, "http") {
					claimID := fmt.Sprintf("claim-%d-%d", time.Now().UnixNano(), atomic.AddUint64(&claimSeq, 1))
					claims = append(claims, Claim{
						ID:         claimID,
						SessionID:  sessionID,
						RequestID:  requestID,
						SourceSpan: trimmed,
						TokenRange: [2]int{lineIdx, lineIdx + 1},
						Type:       TypeFile,
						Subject:    path,
						Predicate:  "exists_in_workspace",
						Object:     path,
						Confidence: 0.90,
						Status:     StatusUnknown,
						Provenance: "StandAlonePathExtractor",
						CreatedAt:  time.Now(),
					})
				}
			}
		}

		// 4. Function / Symbol assertions
		if matches := funcSymbolRegex.FindAllStringSubmatch(trimmed, -1); len(matches) > 0 {
			for _, m := range matches {
				sym := strings.TrimSpace(m[1])
				claimID := fmt.Sprintf("claim-%d-%d", time.Now().UnixNano(), atomic.AddUint64(&claimSeq, 1))
				claims = append(claims, Claim{
					ID:         claimID,
					SessionID:  sessionID,
					RequestID:  requestID,
					SourceSpan: trimmed,
					TokenRange: [2]int{lineIdx, lineIdx + 1},
					Type:       TypeFunction,
					Subject:    sym,
					Predicate:  "is_defined_in_codebase",
					Object:     sym,
					Confidence: 0.95,
					Status:     StatusUnknown,
					Provenance: "FunctionSymbolExtractor",
					CreatedAt:  time.Now(),
				})
			}
		}

		// 5. API endpoint assertions
		if matches := apiEndpointRegex.FindAllStringSubmatch(trimmed, -1); len(matches) > 0 {
			for _, m := range matches {
				route := m[1]
				if route == "" {
					route = m[2]
				}
				route = strings.TrimSpace(route)
				claimID := fmt.Sprintf("claim-%d-%d", time.Now().UnixNano(), atomic.AddUint64(&claimSeq, 1))
				claims = append(claims, Claim{
					ID:         claimID,
					SessionID:  sessionID,
					RequestID:  requestID,
					SourceSpan: trimmed,
					TokenRange: [2]int{lineIdx, lineIdx + 1},
					Type:       TypeAPI,
					Subject:    route,
					Predicate:  "is_supported_api_route",
					Object:     route,
					Confidence: 0.95,
					Status:     StatusUnknown,
					Provenance: "APIRouteExtractor",
					CreatedAt:  time.Now(),
				})
			}
		}

		// 6. Configuration / Port assertions
		if matches := configPortRegex.FindAllStringSubmatch(trimmed, -1); len(matches) > 0 {
			for _, m := range matches {
				port := strings.TrimSpace(m[1])
				claimID := fmt.Sprintf("claim-%d-%d", time.Now().UnixNano(), atomic.AddUint64(&claimSeq, 1))
				claims = append(claims, Claim{
					ID:         claimID,
					SessionID:  sessionID,
					RequestID:  requestID,
					SourceSpan: trimmed,
					TokenRange: [2]int{lineIdx, lineIdx + 1},
					Type:       TypeConfiguration,
					Subject:    "server_port",
					Predicate:  "equals",
					Object:     port,
					Confidence: 0.90,
					Status:     StatusUnknown,
					Provenance: "ConfigPortExtractor",
					CreatedAt:  time.Now(),
				})
			}
		}

		// 7. Tool result assertions
		if matches := toolResultRegex.FindAllStringSubmatch(trimmed, -1); len(matches) > 0 {
			for _, m := range matches {
				resultContent := strings.TrimSpace(m[1])
				claimID := fmt.Sprintf("claim-%d-%d", time.Now().UnixNano(), atomic.AddUint64(&claimSeq, 1))
				claims = append(claims, Claim{
					ID:         claimID,
					SessionID:  sessionID,
					RequestID:  requestID,
					SourceSpan: trimmed,
					TokenRange: [2]int{lineIdx, lineIdx + 1},
					Type:       TypeToolResult,
					Subject:    "tool_execution",
					Predicate:  "produced_result",
					Object:     resultContent,
					Confidence: 1.0,
					Status:     StatusUnknown,
					Provenance: "ToolResultExtractor",
					CreatedAt:  time.Now(),
				})
			}
		}

		// 8. Logical implications
		if matches := logicalImpRegex.FindAllStringSubmatch(trimmed, -1); len(matches) > 0 {
			for _, m := range matches {
				antecedent := strings.TrimSpace(m[1])
				consequent := strings.TrimSpace(m[2])
				claimID := fmt.Sprintf("claim-%d-%d", time.Now().UnixNano(), atomic.AddUint64(&claimSeq, 1))
				claims = append(claims, Claim{
					ID:         claimID,
					SessionID:  sessionID,
					RequestID:  requestID,
					SourceSpan: trimmed,
					TokenRange: [2]int{lineIdx, lineIdx + 1},
					Type:       TypeLogical,
					Subject:    antecedent,
					Predicate:  "implies",
					Object:     consequent,
					Confidence: 0.90,
					Status:     StatusUnknown,
					Provenance: "LogicalImplicationExtractor",
					CreatedAt:  time.Now(),
				})
			}
		}
	}

	return claims
}

// HashClaim computes a deterministic SHA256 identifier for a claim
func HashClaim(c *Claim) string {
	raw := fmt.Sprintf("%s|%s|%s|%s|%s", c.Type, c.Subject, c.Predicate, c.Object, c.SourceSpan)
	h := sha256.Sum256([]byte(raw))
	return fmt.Sprintf("%x", h[:16])
}
