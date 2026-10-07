package mathruntime

import (
	"fmt"
	"strings"
	"unicode"
)

var forbiddenTokens = []string{
	"import", "exec", "eval", "__", "system", "os.", "subprocess", "open(", "file(",
	"globals", "locals", "compile", "builtins", "getattr", "setattr", "delattr",
	"lambda", "while", "for", "def", "class", "socket", "http", "curl", "wget",
	"rm ", "sh", "bash", "powershell", "cmd.exe",
}

type ExpressionValidator struct {
	Limits MathSecurityLimits
}

func NewExpressionValidator(limits MathSecurityLimits) *ExpressionValidator {
	return &ExpressionValidator{Limits: limits}
}

func (v *ExpressionValidator) Validate(expression string) (bool, error) {
	expr := strings.TrimSpace(expression)
	if expr == "" {
		return false, fmt.Errorf("expression is empty")
	}

	if len(expr) > v.Limits.MaxExpressionLength {
		return false, fmt.Errorf("expression length (%d) exceeds limit (%d)", len(expr), v.Limits.MaxExpressionLength)
	}

	// Check bracket nesting and balance
	depth := 0
	maxDepth := 0
	for _, c := range expr {
		if c == '(' || c == '[' || c == '{' {
			depth++
			if depth > maxDepth {
				maxDepth = depth
			}
		} else if c == ')' || c == ']' || c == '}' {
			depth--
			if depth < 0 {
				return false, fmt.Errorf("mismatched closing bracket in expression")
			}
		}
	}

	if depth != 0 {
		return false, fmt.Errorf("unclosed opening bracket in expression")
	}

	if maxDepth > v.Limits.MaxRecursionDepth {
		return false, fmt.Errorf("nesting depth (%d) exceeds recursion limit (%d)", maxDepth, v.Limits.MaxRecursionDepth)
	}

	// Security blacklist check
	lower := strings.ToLower(expr)
	for _, tok := range forbiddenTokens {
		if strings.Contains(lower, tok) {
			return false, fmt.Errorf("expression contains forbidden token: '%s'", tok)
		}
	}

	// Complexity check
	complexity := v.ComputeComplexity(expr)
	if complexity > v.Limits.MaxComplexityScore {
		return false, fmt.Errorf("expression complexity (%d) exceeds budget (%d)", complexity, v.Limits.MaxComplexityScore)
	}

	return true, nil
}

func (v *ExpressionValidator) ComputeComplexity(expr string) int {
	score := 1
	nesting := 0
	for _, c := range expr {
		switch c {
		case '(', '[', '{':
			nesting++
			score += 2 * (nesting + 1)
		case ')', ']', '}':
			if nesting > 0 {
				nesting--
			}
		case '+', '-':
			score += 1
		case '*', '/', '%':
			score += 2
		case '^':
			score += 5
		default:
			if unicode.IsLetter(c) {
				score += 3
			}
		}
	}
	return score
}

func (v *ExpressionValidator) IsPureArithmetic(expr string) bool {
	for _, c := range expr {
		if unicode.IsLetter(c) {
			return false
		}
		if !strings.ContainsRune("0123456789+-*/%^()., \t\r\n", c) {
			return false
		}
	}
	return true
}
