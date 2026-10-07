// pkg/logicverifier/claim_extractor.go - Claim Extractor in Go
package logicverifier

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

type ClaimExtractor struct{}

func NewClaimExtractor() *ClaimExtractor {
	return &ClaimExtractor{}
}

func (ce *ClaimExtractor) HasVerifiableContent(text string) bool {
	if len(text) == 0 {
		return false
	}
	return strings.Contains(text, "=") ||
		strings.Contains(text, "==") ||
		strings.Contains(text, "->") ||
		strings.Contains(text, "=>") ||
		strings.Contains(text, "```") ||
		strings.Contains(text, "{") ||
		strings.Contains(strings.ToUpper(text), "SELECT") ||
		strings.Contains(strings.ToLower(text), "premise") ||
		strings.Contains(strings.ToLower(text), "constraint")
}

func (ce *ClaimExtractor) ExtractClaims(text, tenantID, sessionID string) []VerificationClaim {
	var claims []VerificationClaim
	if !ce.HasVerifiableContent(text) {
		return claims
	}

	claims = append(claims, ce.extractJSONAndCode(text, tenantID, sessionID)...)
	claims = append(claims, ce.extractArithmetic(text, tenantID, sessionID)...)
	claims = append(claims, ce.extractEquations(text, tenantID, sessionID)...)
	claims = append(claims, ce.extractPropositions(text, tenantID, sessionID)...)
	claims = append(claims, ce.extractConstraints(text, tenantID, sessionID)...)
	claims = append(claims, ce.extractUnits(text, tenantID, sessionID)...)

	return claims
}

func (ce *ClaimExtractor) extractArithmetic(text, tenantID, sessionID string) []VerificationClaim {
	var claims []VerificationClaim
	re := regexp.MustCompile(`([\d\.\s\+\-\*\/\^\(\)]+?)\s*(?:=|==|is equal to)\s*([\-\+]?\d+(?:\.\d+)?(?:[eE][\-\+]?\d+)?|\d+\/\d+)`)
	matches := re.FindAllStringSubmatch(text, -1)

	for i, m := range matches {
		if len(m) < 3 {
			continue
		}
		expr := strings.TrimSpace(m[1])
		claimed := strings.TrimSpace(m[2])

		hasOp := strings.ContainsAny(expr, "+-*/^")
		if hasOp && len(expr) >= 3 {
			claims = append(claims, VerificationClaim{
				ClaimID:      fmt.Sprintf("claim_arith_%d", i+1),
				Type:         ClaimArithmetic,
				RawStatement: m[0],
				Expression:   expr,
				ClaimedValue: claimed,
				TenantID:     tenantID,
				SessionID:    sessionID,
			})
		}
	}
	return claims
}

func (ce *ClaimExtractor) extractEquations(text, tenantID, sessionID string) []VerificationClaim {
	var claims []VerificationClaim
	re := regexp.MustCompile(`(?i)(?:solve\s+|equation:\s*)?([a-zA-Z0-9\s\+\-\*\/\^\(\)]+?)\s*(?:==|=)\s*([a-zA-Z0-9\s\+\-\*\/\^\(\)]+?)\s*(?:=>|->|therefore|,)\s*([a-zA-Z])\s*(?:=|==)\s*([\-\+]?\d+(?:\.\d+)?|\d+\/\d+)`)
	matches := re.FindAllStringSubmatch(text, -1)

	for i, m := range matches {
		if len(m) < 5 {
			continue
		}
		lhs := strings.TrimSpace(m[1])
		rhs := strings.TrimSpace(m[2])
		varName := strings.TrimSpace(m[3])
		val := strings.TrimSpace(m[4])

		claims = append(claims, VerificationClaim{
			ClaimID:      fmt.Sprintf("claim_eq_%d", i+1),
			Type:         ClaimEquation,
			RawStatement: m[0],
			Expression:   fmt.Sprintf("%s == %s", lhs, rhs),
			ClaimedValue: fmt.Sprintf("%s == %s", varName, val),
			TenantID:     tenantID,
			SessionID:    sessionID,
		})
	}
	return claims
}

func (ce *ClaimExtractor) extractPropositions(text, tenantID, sessionID string) []VerificationClaim {
	var claims []VerificationClaim
	re := regexp.MustCompile(`(?i)(?:Premises?:\s*\[?([^\]\n]+)\]?|Given\s+([^;\n]+?))\s*,?\s*(?:Conclusion:|therefore|so)\s*([a-zA-Z0-9\s\-\>\=\<\!\&\|\~]+)`)
	matches := re.FindAllStringSubmatch(text, -1)

	for i, m := range matches {
		if len(m) < 4 {
			continue
		}
		premiseStr := m[1]
		if premiseStr == "" {
			premiseStr = m[2]
		}
		concl := strings.TrimSpace(m[3])

		// Replace " and " with "," to split multiple premises
		normalized := regexp.MustCompile(`(?i)\s+and\s+`).ReplaceAllString(premiseStr, ",")
		normalized = strings.ReplaceAll(normalized, ";", ",")

		var premises []string
		for _, part := range strings.Split(normalized, ",") {
			p := strings.TrimSpace(part)
			// Strip prefix "premise:" or "premises:"
			p = regexp.MustCompile(`(?i)^premises?:\s*`).ReplaceAllString(p, "")
			p = regexp.MustCompile(`(?i)^given:\s*`).ReplaceAllString(p, "")
			p = strings.TrimSpace(p)
			if p != "" {
				premises = append(premises, p)
			}
		}

		claims = append(claims, VerificationClaim{
			ClaimID:      fmt.Sprintf("claim_prop_%d", i+1),
			Type:         ClaimProposition,
			RawStatement: m[0],
			Premises:     premises,
			Expression:   concl,
			ClaimedValue: concl,
			TenantID:     tenantID,
			SessionID:    sessionID,
		})
	}
	return claims
}

func (ce *ClaimExtractor) extractConstraints(text, tenantID, sessionID string) []VerificationClaim {
	var claims []VerificationClaim
	re1 := regexp.MustCompile(`(?i)constraints?:\s*\[?([^\]\n]+)\]?\s*(?:claim|value|assignment):\s*([a-zA-Z0-9\s\=\<\>\!]+)`)
	matches1 := re1.FindAllStringSubmatch(text, -1)

	for _, m := range matches1 {
		if len(m) < 3 {
			continue
		}
		constrStr := m[1]
		claimStr := strings.TrimSpace(m[2])

		var constrs []string
		for _, part := range strings.Split(constrStr, ",") {
			c := strings.TrimSpace(part)
			if c != "" {
				constrs = append(constrs, c)
			}
		}

		claims = append(claims, VerificationClaim{
			ClaimID:      fmt.Sprintf("claim_constr_%d", len(claims)+1),
			Type:         ClaimConstraint,
			RawStatement: m[0],
			Constraints:  constrs,
			Expression:   claimStr,
			ClaimedValue: claimStr,
			TenantID:     tenantID,
			SessionID:    sessionID,
		})
	}

	re2 := regexp.MustCompile(`(?i)constraints?\s+([a-zA-Z_]\w*\s*[<>!=]=?\s*[-+]?\d+(?:\.\d+)?(?:\s+(?:and|,)\s+[a-zA-Z_]\w*\s*[<>!=]=?\s*[-+]?\d+(?:\.\d+)?)*)[,\s]+(?:we find|yields|results in|gives|claim)\s+([a-zA-Z_]\w*\s*=\s*[-+]?\d+(?:\.\d+)?)`)
	matches2 := re2.FindAllStringSubmatch(text, -1)
	for _, m := range matches2 {
		if len(m) < 3 {
			continue
		}
		constrBlock := m[1]
		claimStr := strings.TrimSpace(m[2])

		rawConstrs := strings.FieldsFunc(constrBlock, func(r rune) bool {
			return r == ',' || r == ';'
		})
		var constrs []string
		for _, rc := range rawConstrs {
			parts := strings.Split(rc, " and ")
			for _, p := range parts {
				p = strings.TrimSpace(p)
				if p != "" {
					constrs = append(constrs, p)
				}
			}
		}

		claims = append(claims, VerificationClaim{
			ClaimID:      fmt.Sprintf("claim_constr_%d", len(claims)+1),
			Type:         ClaimConstraint,
			RawStatement: m[0],
			Constraints:  constrs,
			Expression:   claimStr,
			ClaimedValue: claimStr,
			TenantID:     tenantID,
			SessionID:    sessionID,
		})
	}

	return claims
}

func (ce *ClaimExtractor) extractUnits(text, tenantID, sessionID string) []VerificationClaim {
	var claims []VerificationClaim
	re := regexp.MustCompile(`([\d\.\s\+\-\*\/\(\)\w]+?\b(?:m|s|kg|km|cm|mm|g|N|J|W|Pa|Hz|degC|degF|A|K|mol|cd)\b[\d\.\s\+\-\*\/\(\)\w]*)\s*(?:=|==|results in)\s*([\d\.\s\+\-\*\/\(\)\w]+?\b(?:m|s|kg|km|cm|mm|g|N|J|W|Pa|Hz|degC|degF|A|K|mol|cd)\b[\w\s\/\*\^]*)`)
	matches := re.FindAllStringSubmatch(text, -1)

	for i, m := range matches {
		if len(m) < 3 {
			continue
		}
		left := strings.TrimSpace(m[1])
		right := strings.TrimSpace(m[2])

		claims = append(claims, VerificationClaim{
			ClaimID:        fmt.Sprintf("claim_unit_%d", i+1),
			Type:           ClaimUnitDimension,
			RawStatement:   m[0],
			UnitExpression: left,
			ClaimedUnit:    right,
			Expression:     left,
			ClaimedValue:   right,
			TenantID:       tenantID,
			SessionID:      sessionID,
		})
	}
	return claims
}

func (ce *ClaimExtractor) extractJSONAndCode(text, tenantID, sessionID string) []VerificationClaim {
	var claims []VerificationClaim
	re := regexp.MustCompile("```(json|python|cpp|c\\+\\+|javascript|sql)?\\s*\\n([\\s\\S]*?)\\n```")
	matches := re.FindAllStringSubmatch(text, -1)

	for i, m := range matches {
		if len(m) < 3 {
			continue
		}
		lang := strings.ToLower(strings.TrimSpace(m[1]))
		content := m[2]

		claim := VerificationClaim{
			ClaimID:      fmt.Sprintf("claim_code_%d", i+1),
			RawStatement: m[0],
			Expression:   content,
			TenantID:     tenantID,
			SessionID:    sessionID,
		}

		if lang == "json" || (lang == "" && strings.Contains(content, "{")) {
			claim.Type = ClaimSchemaType
			claim.SchemaJSON = content
		} else if lang == "sql" {
			claim.Type = ClaimSQL
			claim.SQLQuery = content
		} else {
			claim.Type = ClaimCodeSyntax
			claim.CodeSnippet = content
			claim.Language = lang
		}
		claims = append(claims, claim)
	}
	return claims
}

func (ce *ClaimExtractor) ParseJSONClaim(jsonStr string) (VerificationClaim, error) {
	var claim VerificationClaim
	if err := json.Unmarshal([]byte(jsonStr), &claim); err != nil {
		return claim, err
	}
	if claim.ClaimID == "" {
		claim.ClaimID = "claim_json_1"
	}
	if claim.TenantID == "" {
		claim.TenantID = "default"
	}
	if claim.SessionID == "" {
		claim.SessionID = "default"
	}
	return claim, nil
}
