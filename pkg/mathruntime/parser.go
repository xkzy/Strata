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
	diffRegex     = regexp.MustCompile(`(?i)(?:differentiate|derivative(?:\s+of)?|d/dx)\s*([a-zA-Z0-9_\+\-\*\/\^\(\)\s]+?)(?:\s+with\s+respect\s+to\s+([a-zA-Z]))?(?:\s+(?:give|state|in|to|as)|\?|$|\.|\n)`)
	intRegex      = regexp.MustCompile(`(?i)(?:integrate|integral(?:\s+of)?)\s*([a-zA-Z0-9_\+\-\*\/\^\(\)\s]+?)(?:\s+d([a-zA-Z]))?(?:\s+(?:give|state|in|to|as)|\?|$|\.|\n)`)
	limitRegex    = regexp.MustCompile(`(?i)(?:limit|lim)\s*(?:of)?\s*([a-zA-Z0-9_\+\-\*\/\^\(\)\s]+?)\s+(?:as|when)\s+([a-zA-Z])\s*(?:->|to)\s*([-+]?[0-9\.]+)`)
	solveRegex    = regexp.MustCompile(`(?i)(?:solve|find\s+roots\s+of)\s*([a-zA-Z0-9_\+\-\*\/\^\(\)\s\=\<\>]+?)(?:\s+for\s+([a-zA-Z]))?$`)
	simplifyRegex = regexp.MustCompile(`(?i)(?:simplify)\s*([a-zA-Z0-9_\+\-\*\/\^\(\)\s]+?)(?:\s+(?:to|as|in|for|give|state)|\?|$|\.|\n)`)
	factorRegex   = regexp.MustCompile(`(?i)(?:factor)\s*([a-zA-Z0-9_\+\-\*\/\^\(\)\s]+)`)
	expandRegex   = regexp.MustCompile(`(?i)(?:expand)\s*([a-zA-Z0-9_\+\-\*\/\^\(\)\s]+)`)
	detRegex      = regexp.MustCompile(`(?i)\b(?:det|determinant)\b.*?(\[\[[0-9\s\,\-\+\.\/\[\]]+\]\])`)
	invRegex      = regexp.MustCompile(`(?i)\b(?:inv|inverse\s+of)\b.*?(\[\[[0-9\s\,\-\+\.\/\[\]]+\]\])`)
	matMulRegex   = regexp.MustCompile(`(?i)(?:multiply|product\s+of)\s*(?:matrices\s+)?(\[\[.+?\]\])\s*(?:and|\*|by)\s*(\[\[.+?\]\])`)
	traceRegex    = regexp.MustCompile(`(?i)\b(?:trace|tr)\b.*?(\[\[[0-9\s\,\-\+\.\/\[\]]+\]\])`)
	eigenRegex    = regexp.MustCompile(`(?i)\b(?:eigenvalues?\s+of)\b.*?(\[\[[0-9\s\,\-\+\.\/\[\]]+\]\])`)
	linSysRegex   = regexp.MustCompile(`(?i)(?:solve\s+system|linear\s+system)\s*(\[\[.+?\]\])\s*(?:with|=)\s*(\[[0-9\s\,\-\+\.\/]+\])`)
	statsRegex    = regexp.MustCompile(`(?i)\b(mean|median|stddev|standard\s+deviation|variance)\s+(?:of\s+)?(\[[0-9\s\,\-\+\.]+\]|[0-9\.]+(?:\s*,\s*[0-9\.]+)+)`)
	modInvRegex   = regexp.MustCompile(`(?i)(?:modular\s+inverse|modinv)\s+(?:of\s+)?([0-9]+)\s+mod(?:ulo)?\s+([0-9]+)`)
	modPowRegex   = regexp.MustCompile(`(?i)(?:modpow|power\s+mod)\s+([0-9]+)\s*\^\s*([0-9]+)\s+mod(?:ulo)?\s+([0-9]+)`)
	combRegex     = regexp.MustCompile(`(?i)\b([CP])\s*\(\s*([0-9]+)\s*,\s*([0-9]+)\s*\)`)
	gcdRegex      = regexp.MustCompile(`(?i)\b(GCD|LCM)\s*\(\s*([0-9]+)\s*,\s*([0-9]+)\s*\)`)
	questionArith = regexp.MustCompile(`(?i)(?:what\s+is|calculate|compute|eval|value\s+of|evaluate|simplify)\s+([0-9\.\,\s\+\-\*\/\^\(\)×÷]+?)(?:\s+(?:in|to|as|for|give|state)|\?|$|\.|\n)`)
	rawArithRegex = regexp.MustCompile(`(?i)\b([0-9,]+(?:\.[0-9]+)?(?:\s*[\+\-\*\/\^×÷]\s*[0-9,]+(?:\.[0-9]+)?)+)\b`)
)

func (p *ExpressionParser) DetectCalculationIntents(text string) []ParsedIntent {
	var intents []ParsedIntent
	if text == "" {
		return intents
	}

	trimmed := strings.TrimSpace(text)

	if m := diffRegex.FindStringSubmatch(trimmed); len(m) > 1 {
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

	if m := intRegex.FindStringSubmatch(trimmed); len(m) > 1 {
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

	if m := limitRegex.FindStringSubmatch(trimmed); len(m) == 4 {
		intents = append(intents, ParsedIntent{
			Operation:  OpLimit,
			Expression: strings.TrimSpace(m[1]),
			Variable:   strings.TrimSpace(m[2]),
			Point:      strings.TrimSpace(m[3]),
			RawMatch:   m[0],
		})
		return intents
	}

	if m := invRegex.FindStringSubmatch(trimmed); len(m) > 1 {
		intents = append(intents, ParsedIntent{
			Operation:  OpMatrixInverse,
			Expression: strings.TrimSpace(m[1]),
			RawMatch:   m[0],
		})
		return intents
	}

	if m := matMulRegex.FindStringSubmatch(trimmed); len(m) == 3 {
		intents = append(intents, ParsedIntent{
			Operation:  OpMatrixMultiply,
			Expression: fmt.Sprintf("%s * %s", strings.TrimSpace(m[1]), strings.TrimSpace(m[2])),
			RawMatch:   m[0],
		})
		return intents
	}

	if m := traceRegex.FindStringSubmatch(trimmed); len(m) > 1 {
		intents = append(intents, ParsedIntent{
			Operation:  OpMatrixTrace,
			Expression: strings.TrimSpace(m[1]),
			RawMatch:   m[0],
		})
		return intents
	}

	if m := eigenRegex.FindStringSubmatch(trimmed); len(m) > 1 {
		intents = append(intents, ParsedIntent{
			Operation:  OpEigenvalues,
			Expression: strings.TrimSpace(m[1]),
			RawMatch:   m[0],
		})
		return intents
	}

	if m := linSysRegex.FindStringSubmatch(trimmed); len(m) == 3 {
		intents = append(intents, ParsedIntent{
			Operation:  OpLinearSystem,
			Expression: fmt.Sprintf("%s = %s", strings.TrimSpace(m[1]), strings.TrimSpace(m[2])),
			RawMatch:   m[0],
		})
		return intents
	}

	if m := statsRegex.FindStringSubmatch(trimmed); len(m) == 3 {
		intents = append(intents, ParsedIntent{
			Operation:  OpStatistics,
			Expression: strings.TrimSpace(m[2]),
			Variable:   strings.ToLower(strings.TrimSpace(m[1])),
			RawMatch:   m[0],
		})
		return intents
	}

	if m := modInvRegex.FindStringSubmatch(trimmed); len(m) == 3 {
		intents = append(intents, ParsedIntent{
			Operation:  OpModInverse,
			Expression: fmt.Sprintf("%s mod %s", m[1], m[2]),
			RawMatch:   m[0],
		})
		return intents
	}

	if m := modPowRegex.FindStringSubmatch(trimmed); len(m) == 4 {
		intents = append(intents, ParsedIntent{
			Operation:  OpModPow,
			Expression: fmt.Sprintf("%s^%s mod %s", m[1], m[2], m[3]),
			RawMatch:   m[0],
		})
		return intents
	}

	if m := solveRegex.FindStringSubmatch(trimmed); len(m) > 1 {
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

	if m := simplifyRegex.FindStringSubmatch(trimmed); len(m) > 1 {
		intents = append(intents, ParsedIntent{
			Operation:  OpSimplify,
			Expression: strings.TrimSpace(m[1]),
			RawMatch:   m[0],
		})
		return intents
	}

	if m := factorRegex.FindStringSubmatch(trimmed); len(m) > 1 {
		intents = append(intents, ParsedIntent{
			Operation:  OpFactor,
			Expression: strings.TrimSpace(m[1]),
			RawMatch:   m[0],
		})
		return intents
	}

	if m := expandRegex.FindStringSubmatch(trimmed); len(m) > 1 {
		intents = append(intents, ParsedIntent{
			Operation:  OpExpand,
			Expression: strings.TrimSpace(m[1]),
			RawMatch:   m[0],
		})
		return intents
	}

	if m := detRegex.FindStringSubmatch(trimmed); len(m) > 1 {
		intents = append(intents, ParsedIntent{
			Operation:  OpDeterminant,
			Expression: strings.TrimSpace(m[1]),
			RawMatch:   m[0],
		})
		return intents
	}

	if m := combRegex.FindStringSubmatch(trimmed); len(m) == 4 {
		intents = append(intents, ParsedIntent{
			Operation:  OpProbability,
			Expression: fmt.Sprintf("%s(%s, %s)", strings.ToUpper(m[1]), m[2], m[3]),
			RawMatch:   m[0],
		})
		return intents
	}

	if m := gcdRegex.FindStringSubmatch(trimmed); len(m) == 4 {
		intents = append(intents, ParsedIntent{
			Operation:  OpEvaluate,
			Expression: fmt.Sprintf("%s(%s, %s)", strings.ToUpper(m[1]), m[2], m[3]),
			RawMatch:   m[0],
		})
		return intents
	}

	if m := questionArith.FindStringSubmatch(trimmed); len(m) > 1 {
		expr := strings.TrimRight(strings.TrimSpace(m[1]), ".")
		if strings.ContainsAny(expr, "+-*/^×÷") {
			intents = append(intents, ParsedIntent{
				Operation:  OpEvaluate,
				Expression: expr,
				RawMatch:   strings.TrimSpace(m[0]),
			})
			return intents
		}
	}

	if m := rawArithRegex.FindStringSubmatch(trimmed); len(m) > 1 && strings.ContainsAny(m[1], "+-*/^×÷") {
		expr := strings.TrimRight(strings.TrimSpace(m[1]), ".")
		intents = append(intents, ParsedIntent{
			Operation:  OpEvaluate,
			Expression: expr,
			RawMatch:   strings.TrimSpace(m[0]),
		})
	}

	return intents
}
