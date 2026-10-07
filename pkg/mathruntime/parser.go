package mathruntime

import (
	"fmt"
	"regexp"
	"strings"
	"unicode"
)

type ParsedIntent struct {
	Operation  MathOperation
	Expression string
	Variable   string
	Point      string
	Order      int
	Mode       MathMode
	RawMatch   string
	StartPos   int
	Length     int
}

type ExpressionParser struct{}

func NewExpressionParser() *ExpressionParser {
	return &ExpressionParser{}
}

// Canonicalize cleans up mathematical expressions: normalizes Unicode operators,
// power representations, and strips thousands separator commas from numbers without affecting commas in matrices.
func (p *ExpressionParser) Canonicalize(expr string) string {
	if expr == "" {
		return ""
	}

	clean := strings.ReplaceAll(expr, "×", "*")
	clean = strings.ReplaceAll(clean, "·", "*")
	clean = strings.ReplaceAll(clean, "÷", "/")
	clean = strings.ReplaceAll(clean, "**", "^")

	var sb strings.Builder
	sb.Grow(len(clean))
	runes := []rune(clean)

	for i := 0; i < len(runes); i++ {
		c := runes[i]
		// Remove thousands separator commas (e.g. 2,384 -> 2384, but keep [[1,2],[3,4]] and C(10, 3))
		if c == ',' && i > 0 && unicode.IsDigit(runes[i-1]) &&
			(i+3 < len(runes) && unicode.IsDigit(runes[i+1]) && unicode.IsDigit(runes[i+2]) && unicode.IsDigit(runes[i+3]) &&
				(i+4 >= len(runes) || !unicode.IsDigit(runes[i+4]))) {
			continue
		}
		if !unicode.IsSpace(c) {
			sb.WriteRune(c)
		}
	}
	return sb.String()
}

func (p *ExpressionParser) FormatCompactObservation(expr, exactRes, numericRes string, op MathOperation) string {
	resStr := exactRes
	if resStr == "" {
		resStr = numericRes
	}
	if exactRes != "" && numericRes != "" && exactRes != numericRes {
		resStr = fmt.Sprintf("%s (approx %s)", exactRes, numericRes)
	}
	return fmt.Sprintf("[MathResult: %s(%s) = %s]", op, expr, resStr)
}

var (
	diffRegex     = regexp.MustCompile(`(?i)(?:differentiate|derivative(?:\s+of)?|d/dx)\s*([a-zA-Z0-9_\+\-\*\/\^\(\)\s]+?)(?:\s+with\s+respect\s+to\s+([a-zA-Z]))?$`)
	intRegex      = regexp.MustCompile(`(?i)(?:integrate|integral(?:\s+of)?)\s*([a-zA-Z0-9_\+\-\*\/\^\(\)\s]+?)(?:\s+d([a-zA-Z]))?$`)
	solveRegex    = regexp.MustCompile(`(?i)(?:solve|find\s+roots\s+of)\s*([a-zA-Z0-9_\+\-\*\/\^\(\)\s\=\<\>]+?)(?:\s+for\s+([a-zA-Z]))?$`)
	simplifyRegex = regexp.MustCompile(`(?i)(?:simplify)\s*([a-zA-Z0-9_\+\-\*\/\^\(\)\s]+)`)
	factorRegex   = regexp.MustCompile(`(?i)(?:factor)\s*([a-zA-Z0-9_\+\-\*\/\^\(\)\s]+)`)
	expandRegex   = regexp.MustCompile(`(?i)(?:expand)\s*([a-zA-Z0-9_\+\-\*\/\^\(\)\s]+)`)
	sqrtRegex     = regexp.MustCompile(`(?i)sqrt\(([0-9\.]+)\)`)
	arithRegex    = regexp.MustCompile(`(?i)(?:calculate|eval|compute)?\s*([0-9,]+(?:\.[0-9]+)?\s*[\+\-\*\/\^×÷]\s*[0-9,]+(?:\.[0-9]+)?(?:\s*[\+\-\*\/\^×÷]\s*[0-9,]+(?:\.[0-9]+)?)*)`)
)

func (p *ExpressionParser) DetectCalculationIntents(text string) []ParsedIntent {
	var intents []ParsedIntent
	if text == "" {
		return intents
	}

	if m := diffRegex.FindStringSubmatch(text); len(m) > 1 {
		v := "x"
		if len(m) > 2 && m[2] != "" {
			v = m[2]
		}
		intents = append(intents, ParsedIntent{
			Operation:  OpDifferentiate,
			Expression: strings.TrimSpace(m[1]),
			Variable:   v,
			RawMatch:   m[0],
		})
		return intents
	}

	if m := intRegex.FindStringSubmatch(text); len(m) > 1 {
		v := "x"
		if len(m) > 2 && m[2] != "" {
			v = m[2]
		}
		intents = append(intents, ParsedIntent{
			Operation:  OpIntegrate,
			Expression: strings.TrimSpace(m[1]),
			Variable:   v,
			RawMatch:   m[0],
		})
		return intents
	}

	if m := solveRegex.FindStringSubmatch(text); len(m) > 1 {
		v := "x"
		if len(m) > 2 && m[2] != "" {
			v = m[2]
		}
		intents = append(intents, ParsedIntent{
			Operation:  OpSolve,
			Expression: strings.TrimSpace(m[1]),
			Variable:   v,
			RawMatch:   m[0],
		})
		return intents
	}

	if m := simplifyRegex.FindStringSubmatch(text); len(m) > 1 {
		intents = append(intents, ParsedIntent{
			Operation:  OpSimplify,
			Expression: strings.TrimSpace(m[1]),
			RawMatch:   m[0],
		})
		return intents
	}

	if m := factorRegex.FindStringSubmatch(text); len(m) > 1 {
		intents = append(intents, ParsedIntent{
			Operation:  OpFactor,
			Expression: strings.TrimSpace(m[1]),
			RawMatch:   m[0],
		})
		return intents
	}

	if m := expandRegex.FindStringSubmatch(text); len(m) > 1 {
		intents = append(intents, ParsedIntent{
			Operation:  OpExpand,
			Expression: strings.TrimSpace(m[1]),
			RawMatch:   m[0],
		})
		return intents
	}

	if m := arithRegex.FindStringSubmatch(text); len(m) > 1 && strings.ContainsAny(m[1], "+-*/^×÷") {
		intents = append(intents, ParsedIntent{
			Operation:  OpEvaluate,
			Expression: strings.TrimSpace(m[1]),
			RawMatch:   strings.TrimSpace(m[0]),
		})
	}

	return intents
}
