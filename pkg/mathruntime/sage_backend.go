// Package mathruntime provides the deterministic mathematical backing runtime for Strata.
package mathruntime

import (
	"fmt"
	"math/big"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

type SageBackend struct {
	Name      string
	Version   string
	Available bool
	mu        sync.Mutex
}

func NewSageBackend() *SageBackend {
	return &SageBackend{
		Name:      "SageMath",
		Version:   "10.4",
		Available: true,
	}
}

func (b *SageBackend) SupportsOperation(op MathOperation, mode MathMode) bool {
	return true
}

func (b *SageBackend) TranslateSageSyntax(sageExpr string) string {
	if sageExpr == "" {
		return ""
	}

	text := sageExpr

	// 1. Strip var declarations: var('x'), var('x y z'), x, y = var('x y')
	varDeclRe := regexp.MustCompile(`(?:(?:[a-zA-Z_][a-zA-Z0-9_,\s]*=)?\s*var\s*\(\s*['"][^'"]*['"]\s*\)\s*(?:;|\n)?)`)
	text = varDeclRe.ReplaceAllString(text, "")

	// 2. Strip PolynomialRing declarations: R.<x> = PolynomialRing(QQ)
	ringDeclRe := regexp.MustCompile(`(?:[a-zA-Z_][a-zA-Z0-9_]*\.<[^>]+>\s*=\s*(?:PolynomialRing\([^)]+\)|[a-zA-Z0-9_]+\[[^\]]+\])\s*(?:;|\n)?)`)
	text = ringDeclRe.ReplaceAllString(text, "")

	text = strings.TrimSpace(text)

	// 3. Dot-method transformations:
	// expr.diff(x) -> diff(expr, x)
	// expr.factor() -> factor(expr)
	// expr.roots(x) -> solve(expr == 0, x)
	dotMethodRe := regexp.MustCompile(`\(?([a-zA-Z0-9_\+\-\*\/\^\s\(\)]+?)\)?\.(diff|derivative|integrate|integral|factor|expand|simplify|roots|det|inverse|transpose)\s*\(([^)]*)\)`)
	for {
		m := dotMethodRe.FindStringSubmatch(text)
		if len(m) == 0 {
			break
		}
		target := m[1]
		method := m[2]
		args := m[3]

		var replacement string
		if method == "roots" {
			v := "x"
			if strings.TrimSpace(args) != "" {
				v = strings.TrimSpace(args)
			}
			replacement = fmt.Sprintf("solve(%s == 0, %s)", target, v)
		} else if strings.TrimSpace(args) == "" {
			replacement = fmt.Sprintf("%s(%s)", method, target)
		} else {
			replacement = fmt.Sprintf("%s(%s, %s)", method, target, args)
		}
		text = strings.Replace(text, m[0], replacement, 1)
	}

	// 4. Translate matrix([[...]]) to [[...]]
	matrixRe := regexp.MustCompile(`\bmatrix\s*\(\s*(\[\[[\s\S]*?\]\])\s*\)`)
	text = matrixRe.ReplaceAllString(text, "$1")

	return strings.TrimSpace(text)
}

func (b *SageBackend) Execute(req MathRequest) MathResult {
	startTime := time.Now()
	res := MathResult{
		RequestID:           req.RequestID,
		BackendName:         b.Name,
		BackendVersion:      b.Version,
		CanonicalExpression: req.Expression,
	}

	translated := b.TranslateSageSyntax(req.Expression)
	if translated == "" {
		translated = req.Expression
	}

	expr := strings.TrimSpace(translated)
	varName := req.Variable
	if varName == "" {
		varName = "x"
	}

	parser := NewExpressionParser()

	// Number theory handlers
	if strings.HasPrefix(expr, "is_prime(") || strings.HasPrefix(expr, "isprime(") {
		numStr := strings.TrimSuffix(strings.TrimPrefix(strings.TrimPrefix(expr, "is_prime("), "isprime("), ")")
		numStr = strings.TrimSpace(numStr)
		if n, ok := new(big.Int).SetString(numStr, 10); ok {
			if n.ProbablyPrime(20) {
				res.ExactResult = "True"
			} else {
				res.ExactResult = "False"
			}
			res.RawResult = res.ExactResult
			res.Status = StatusSuccess
		}
	} else if strings.HasPrefix(expr, "euler_phi(") || strings.HasPrefix(expr, "phi(") {
		numStr := strings.TrimSuffix(strings.TrimPrefix(strings.TrimPrefix(expr, "euler_phi("), "phi("), ")")
		numStr = strings.TrimSpace(numStr)
		if n, err := strconv.ParseInt(numStr, 10, 64); err == nil && n > 0 {
			res.ExactResult = strconv.FormatInt(eulerPhi(n), 10)
			res.RawResult = res.ExactResult
			res.Status = StatusSuccess
		}
	} else if strings.HasPrefix(expr, "fibonacci(") || strings.HasPrefix(expr, "fib(") {
		numStr := strings.TrimSuffix(strings.TrimPrefix(strings.TrimPrefix(expr, "fibonacci("), "fib("), ")")
		numStr = strings.TrimSpace(numStr)
		if n, err := strconv.ParseInt(numStr, 10, 64); err == nil && n >= 0 {
			res.ExactResult = fibonacciBig(n).String()
			res.RawResult = res.ExactResult
			res.Status = StatusSuccess
		}
	} else if strings.HasPrefix(expr, "power_mod(") || strings.HasPrefix(expr, "powermod(") {
		argsStr := strings.TrimSuffix(strings.TrimPrefix(strings.TrimPrefix(expr, "power_mod("), "powermod("), ")")
		parts := strings.Split(argsStr, ",")
		if len(parts) == 3 {
			base, ok1 := new(big.Int).SetString(strings.TrimSpace(parts[0]), 10)
			exp, ok2 := new(big.Int).SetString(strings.TrimSpace(parts[1]), 10)
			mod, ok3 := new(big.Int).SetString(strings.TrimSpace(parts[2]), 10)
			if ok1 && ok2 && ok3 && mod.Sign() > 0 {
				ans := new(big.Int).Exp(base, exp, mod)
				res.ExactResult = ans.String()
				res.RawResult = res.ExactResult
				res.Status = StatusSuccess
			}
		}
	} else if strings.HasPrefix(expr, "gcd(") {
		argsStr := strings.TrimSuffix(strings.TrimPrefix(expr, "gcd("), ")")
		parts := strings.Split(argsStr, ",")
		if len(parts) == 2 {
			a, ok1 := new(big.Int).SetString(strings.TrimSpace(parts[0]), 10)
			c, ok2 := new(big.Int).SetString(strings.TrimSpace(parts[1]), 10)
			if ok1 && ok2 {
				g := new(big.Int).GCD(nil, nil, a, c)
				res.ExactResult = g.String()
				res.RawResult = res.ExactResult
				res.Status = StatusSuccess
			}
		}
	}

	if res.Status != StatusSuccess {
		// Delegate algebra and calculus to Mathics/CAS semantics with Sage formatting
		mathics := NewMathicsBackend()
		reqCopy := req
		reqCopy.Expression = translated
		mathResult := mathics.Execute(reqCopy)
		res.ExactResult = formatSageString(mathResult.ExactResult, req.Operation)
		res.RawResult = res.ExactResult
		res.NumericResult = mathResult.NumericResult
		res.Status = mathResult.Status
		res.ErrorMessage = mathResult.ErrorMessage
	}

	res.ExecutionTimeMs = float64(time.Since(startTime).Microseconds()) / 1000.0
	res.CompactObservation = parser.FormatCompactObservation(req.Expression, res.ExactResult, res.NumericResult, req.Operation)
	return res
}

func eulerPhi(n int64) int64 {
	result := n
	p := int64(2)
	for p*p <= n {
		if n%p == 0 {
			for n%p == 0 {
				n /= p
			}
			result -= result / p
		}
		p++
	}
	if n > 1 {
		result -= result / n
	}
	return result
}

func fibonacciBig(n int64) *big.Int {
	if n == 0 {
		return big.NewInt(0)
	}
	if n == 1 {
		return big.NewInt(1)
	}
	a := big.NewInt(0)
	b := big.NewInt(1)
	for i := int64(2); i <= n; i++ {
		c := new(big.Int).Add(a, b)
		a = b
		b = c
	}
	return b
}

func formatSageString(s string, op MathOperation) string {
	if op == OpSolve {
		// Convert {x -> -3, x -> -2} to [x == -3, x == -2]
		s = strings.ReplaceAll(s, "{", "[")
		s = strings.ReplaceAll(s, "}", "]")
		s = strings.ReplaceAll(s, "->", "==")
		return s
	}
	if strings.HasPrefix(s, "{") && strings.HasSuffix(s, "}") {
		return "[" + s[1:len(s)-1] + "]"
	}
	return s
}
