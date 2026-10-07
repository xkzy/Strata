// pkg/logicverifier/rule_engine.go - Propositional Logic & Rule Engine in Go
package logicverifier

import (
	"fmt"
	"strings"
	"time"
	"unicode"
)

type PropNodeType int

const (
	NodeVariable PropNodeType = iota
	NodeNot
	NodeAnd
	NodeOr
	NodeImplies
	NodeEquivalence
	NodeConstant
)

type PropNode struct {
	Type     PropNodeType
	VarName  string
	ConstVal bool
	Left     *PropNode
	Right    *PropNode
}

func (n *PropNode) Evaluate(val map[string]bool) bool {
	if n == nil {
		return false
	}
	switch n.Type {
	case NodeConstant:
		return n.ConstVal
	case NodeVariable:
		return val[n.VarName]
	case NodeNot:
		return !n.Left.Evaluate(val)
	case NodeAnd:
		return n.Left.Evaluate(val) && n.Right.Evaluate(val)
	case NodeOr:
		return n.Left.Evaluate(val) || n.Right.Evaluate(val)
	case NodeImplies:
		return !n.Left.Evaluate(val) || n.Right.Evaluate(val)
	case NodeEquivalence:
		return n.Left.Evaluate(val) == n.Right.Evaluate(val)
	}
	return false
}

func (n *PropNode) CollectVars(vars map[string]bool) {
	if n == nil {
		return
	}
	if n.Type == NodeVariable && n.VarName != "" {
		vars[n.VarName] = true
	}
	if n.Left != nil {
		n.Left.CollectVars(vars)
	}
	if n.Right != nil {
		n.Right.CollectVars(vars)
	}
}

type RuleEngine struct{}

func NewRuleEngine() *RuleEngine {
	return &RuleEngine{}
}

func (re *RuleEngine) skipWhitespace(s string, pos *int) {
	for *pos < len(s) && unicode.IsSpace(rune(s[*pos])) {
		*pos++
	}
}

func (re *RuleEngine) parsePrimary(s string, pos *int) (*PropNode, error) {
	re.skipWhitespace(s, pos)
	if *pos >= len(s) {
		return nil, fmt.Errorf("unexpected end of expression")
	}

	if s[*pos] == '(' {
		*pos++
		node, err := re.parseImplication(s, pos)
		if err != nil {
			return nil, err
		}
		re.skipWhitespace(s, pos)
		if *pos >= len(s) || s[*pos] != ')' {
			return nil, fmt.Errorf("expected closing parenthesis ')'")
		}
		*pos++
		return node, nil
	}

	if strings.HasPrefix(strings.ToLower(s[*pos:]), "true") {
		*pos += 4
		return &PropNode{Type: NodeConstant, ConstVal: true}, nil
	}
	if strings.HasPrefix(strings.ToLower(s[*pos:]), "false") {
		*pos += 5
		return &PropNode{Type: NodeConstant, ConstVal: false}, nil
	}

	if unicode.IsLetter(rune(s[*pos])) || s[*pos] == '_' {
		start := *pos
		for *pos < len(s) && (unicode.IsLetter(rune(s[*pos])) || unicode.IsDigit(rune(s[*pos])) || s[*pos] == '_') {
			*pos++
		}
		return &PropNode{Type: NodeVariable, VarName: s[start:*pos]}, nil
	}

	return nil, fmt.Errorf("unexpected character '%c'", s[*pos])
}

func (re *RuleEngine) parseUnary(s string, pos *int) (*PropNode, error) {
	re.skipWhitespace(s, pos)
	if *pos >= len(s) {
		return nil, fmt.Errorf("unexpected end of expression")
	}

	if s[*pos] == '!' || s[*pos] == '~' {
		*pos++
		sub, err := re.parseUnary(s, pos)
		if err != nil {
			return nil, err
		}
		return &PropNode{Type: NodeNot, Left: sub}, nil
	}

	if strings.HasPrefix(strings.ToLower(s[*pos:]), "not") {
		*pos += 3
		sub, err := re.parseUnary(s, pos)
		if err != nil {
			return nil, err
		}
		return &PropNode{Type: NodeNot, Left: sub}, nil
	}

	return re.parsePrimary(s, pos)
}

func (re *RuleEngine) parseAnd(s string, pos *int) (*PropNode, error) {
	left, err := re.parseUnary(s, pos)
	if err != nil {
		return nil, err
	}

	for {
		re.skipWhitespace(s, pos)
		if *pos >= len(s) {
			break
		}

		isAnd := false
		if strings.HasPrefix(s[*pos:], "&&") {
			*pos += 2
			isAnd = true
		} else if s[*pos] == '&' || s[*pos] == '*' {
			*pos++
			isAnd = true
		} else if strings.HasPrefix(strings.ToLower(s[*pos:]), "and") {
			*pos += 3
			isAnd = true
		}

		if isAnd {
			right, err := re.parseUnary(s, pos)
			if err != nil {
				return nil, err
			}
			left = &PropNode{Type: NodeAnd, Left: left, Right: right}
		} else {
			break
		}
	}
	return left, nil
}

func (re *RuleEngine) parseOr(s string, pos *int) (*PropNode, error) {
	left, err := re.parseAnd(s, pos)
	if err != nil {
		return nil, err
	}

	for {
		re.skipWhitespace(s, pos)
		if *pos >= len(s) {
			break
		}

		isOr := false
		if strings.HasPrefix(s[*pos:], "||") {
			*pos += 2
			isOr = true
		} else if s[*pos] == '|' || s[*pos] == '+' {
			*pos++
			isOr = true
		} else if strings.HasPrefix(strings.ToLower(s[*pos:]), "or") {
			*pos += 2
			isOr = true
		}

		if isOr {
			right, err := re.parseAnd(s, pos)
			if err != nil {
				return nil, err
			}
			left = &PropNode{Type: NodeOr, Left: left, Right: right}
		} else {
			break
		}
	}
	return left, nil
}

func (re *RuleEngine) parseImplication(s string, pos *int) (*PropNode, error) {
	left, err := re.parseOr(s, pos)
	if err != nil {
		return nil, err
	}

	re.skipWhitespace(s, pos)
	if *pos >= len(s) {
		return left, nil
	}

	if strings.HasPrefix(s[*pos:], "->") || strings.HasPrefix(s[*pos:], "=>") {
		*pos += 2
		right, err := re.parseImplication(s, pos)
		if err != nil {
			return nil, err
		}
		return &PropNode{Type: NodeImplies, Left: left, Right: right}, nil
	}

	if strings.HasPrefix(s[*pos:], "<->") || strings.HasPrefix(s[*pos:], "<=>") {
		*pos += 3
		right, err := re.parseImplication(s, pos)
		if err != nil {
			return nil, err
		}
		return &PropNode{Type: NodeEquivalence, Left: left, Right: right}, nil
	}

	if strings.HasPrefix(strings.ToLower(s[*pos:]), "implies") {
		*pos += 7
		right, err := re.parseImplication(s, pos)
		if err != nil {
			return nil, err
		}
		return &PropNode{Type: NodeImplies, Left: left, Right: right}, nil
	}

	return left, nil
}

func (re *RuleEngine) ParseFormula(formula string) (*PropNode, error) {
	pos := 0
	node, err := re.parseImplication(formula, &pos)
	if err != nil {
		return nil, err
	}
	re.skipWhitespace(formula, &pos)
	if pos < len(formula) {
		return nil, fmt.Errorf("trailing characters at index %d: %s", pos, formula[pos:])
	}
	return node, nil
}

func (re *RuleEngine) VerifyDeduction(premises []string, conclusion, claimID string) VerificationResult {
	start := time.Now()
	res := VerificationResult{
		ClaimID:     claimID,
		Type:        ClaimProposition,
		BackendUsed: "rule_engine",
		Timestamp:   time.Now(),
	}

	conclNode, err := re.ParseFormula(conclusion)
	if err != nil {
		res.Status = StatusUnknown
		res.FailureReason = fmt.Sprintf("Unparseable conclusion: %v", err)
		res.CompactObservation = "Verification: UNKNOWN | Unparseable conclusion"
		res.ExecutionTimeMs = float64(time.Since(start).Microseconds()) / 1000.0
		return res
	}

	var premiseNodes []*PropNode
	for _, p := range premises {
		pNode, err := re.ParseFormula(p)
		if err != nil {
			res.Status = StatusUnknown
			res.FailureReason = fmt.Sprintf("Unparseable premise '%s': %v", p, err)
			res.CompactObservation = "Verification: UNKNOWN | Unparseable premise"
			res.ExecutionTimeMs = float64(time.Since(start).Microseconds()) / 1000.0
			return res
		}
		premiseNodes = append(premiseNodes, pNode)
	}

	varSet := make(map[string]bool)
	conclNode.CollectVars(varSet)
	for _, pNode := range premiseNodes {
		pNode.CollectVars(varSet)
	}

	var vars []string
	for v := range varSet {
		vars = append(vars, v)
	}

	if len(vars) > 16 {
		res.Status = StatusUnknown
		res.FailureReason = "Proposition contains > 16 variables (exceeds deterministic table bound)"
		res.CompactObservation = "Verification: UNKNOWN | Variable limit exceeded"
		res.ExecutionTimeMs = float64(time.Since(start).Microseconds()) / 1000.0
		return res
	}

	totalModels := 1 << len(vars)
	premiseSatisfiable := false

	for i := 0; i < totalModels; i++ {
		val := make(map[string]bool)
		for bit, vName := range vars {
			val[vName] = (i & (1 << bit)) != 0
		}

		premisesHold := true
		for _, pNode := range premiseNodes {
			if !pNode.Evaluate(val) {
				premisesHold = false
				break
			}
		}

		if premisesHold {
			premiseSatisfiable = true
			if !conclNode.Evaluate(val) {
				res.Status = StatusFail
				var ev strings.Builder
				ev.WriteString("Countermodel found: ")
				for _, v := range vars {
					ev.WriteString(fmt.Sprintf("%s=%v ", v, val[v]))
				}
				ev.WriteString("makes premises TRUE but conclusion FALSE.")
				res.Evidence = ev.String()
				res.FailureReason = "Logical deduction is invalid; countermodel exists"
				res.CompactObservation = "Verification: FAIL | Deduction invalid (countermodel found)"
				res.ExecutionTimeMs = float64(time.Since(start).Microseconds()) / 1000.0
				return res
			}
		}
	}

	if !premiseSatisfiable && len(premiseNodes) > 0 {
		res.Status = StatusFail
		res.FailureReason = "Premises are inherently contradictory"
		res.Evidence = "Contradiction detected in premise set; principle of explosion prevented."
		res.CompactObservation = "Verification: FAIL | Contradictory premises"
		res.ExecutionTimeMs = float64(time.Since(start).Microseconds()) / 1000.0
		return res
	}

	res.Status = StatusPass
	res.Evidence = fmt.Sprintf("Logical deduction formally proven across all %d truth valuations.", totalModels)
	res.ExpectedValue = conclusion
	res.ActualValue = conclusion
	res.CompactObservation = "Verification: PASS | Deduction formally proven"
	res.ExecutionTimeMs = float64(time.Since(start).Microseconds()) / 1000.0
	return res
}

func (re *RuleEngine) Verify(claim VerificationClaim) VerificationResult {
	concl := claim.ClaimedValue
	if concl == "" {
		concl = claim.Expression
	}
	return re.VerifyDeduction(claim.Premises, concl, claim.ClaimID)
}

