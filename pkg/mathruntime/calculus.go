package mathruntime

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

// DifferentiateSymbolic performs analytical symbolic differentiation with respect to variable v.
func DifferentiateSymbolic(expr, v string) string {
	if v == "" {
		v = "x"
	}
	expr = strings.TrimSpace(expr)

	// Polynomial term: a*x^n or a*x or x^n or c
	// Match single term: ^([-+]?[0-9\.]*)?\s*\*?\s*([a-zA-Z])(?:\^([-+]?[0-9\.]+))?$
	polyRe := regexp.MustCompile(`^([-+]?[0-9\.]*)\s*\*?\s*([a-zA-Z])(?:\^([-+]?[0-9\.]+))?$`)
	if m := polyRe.FindStringSubmatch(expr); len(m) > 0 && m[2] == v {
		coefStr := m[1]
		powStr := m[3]

		coef := 1.0
		if coefStr == "-" {
			coef = -1.0
		} else if coefStr != "" && coefStr != "+" {
			coef, _ = strconv.ParseFloat(coefStr, 64)
		}

		pow := 1.0
		if powStr != "" {
			pow, _ = strconv.ParseFloat(powStr, 64)
		}

		newCoef := coef * pow
		newPow := pow - 1.0

		if newPow == 0 {
			return fmt.Sprintf("%g", newCoef)
		} else if newPow == 1 {
			if newCoef == 1 {
				return v
			} else if newCoef == -1 {
				return "-" + v
			}
			return fmt.Sprintf("%g*%s", newCoef, v)
		} else {
			if newCoef == 1 {
				return fmt.Sprintf("%s^%g", v, newPow)
			} else if newCoef == -1 {
				return fmt.Sprintf("-%s^%g", v, newPow)
			}
			return fmt.Sprintf("%g*%s^%g", newCoef, v, newPow)
		}
	}

	// Trig functions: sin(k*x), cos(k*x), tan(k*x), exp(k*x), ln(x)
	sinRe := regexp.MustCompile(`^sin\((?:([-+]?[0-9\.]*)\s*\*?\s*)?` + regexp.QuoteMeta(v) + `\)$`)
	if m := sinRe.FindStringSubmatch(expr); len(m) > 0 {
		k := 1.0
		if m[1] == "-" {
			k = -1.0
		} else if m[1] != "" {
			k, _ = strconv.ParseFloat(m[1], 64)
		}
		if k == 1 {
			return fmt.Sprintf("cos(%s)", v)
		}
		return fmt.Sprintf("%g*cos(%g*%s)", k, k, v)
	}

	cosRe := regexp.MustCompile(`^cos\((?:([-+]?[0-9\.]*)\s*\*?\s*)?` + regexp.QuoteMeta(v) + `\)$`)
	if m := cosRe.FindStringSubmatch(expr); len(m) > 0 {
		k := 1.0
		if m[1] == "-" {
			k = -1.0
		} else if m[1] != "" {
			k, _ = strconv.ParseFloat(m[1], 64)
		}
		if k == 1 {
			return fmt.Sprintf("-sin(%s)", v)
		}
		return fmt.Sprintf("-%g*sin(%g*%s)", k, k, v)
	}

	expRe := regexp.MustCompile(`^(?:e\^|exp\()((?:[-+]?[0-9\.]*\s*\*?\s*)?` + regexp.QuoteMeta(v) + `)\)?$`)
	if m := expRe.FindStringSubmatch(expr); len(m) > 0 {
		arg := m[1]
		k := 1.0
		parts := strings.Split(arg, "*")
		if len(parts) == 2 {
			k, _ = strconv.ParseFloat(strings.TrimSpace(parts[0]), 64)
		}
		if k == 1 {
			return fmt.Sprintf("e^%s", v)
		}
		return fmt.Sprintf("%g*e^(%g*%s)", k, k, v)
	}

	lnRe := regexp.MustCompile(`^(?:ln|log)\(` + regexp.QuoteMeta(v) + `\)$`)
	if lnRe.MatchString(expr) {
		return fmt.Sprintf("1/%s", v)
	}

	// Constant check
	if _, err := strconv.ParseFloat(expr, 64); err == nil {
		return "0"
	}

	// Multi-term polynomial: sum of terms (e.g. 3*x^2 + 5*x - 7)
	if strings.ContainsAny(expr, "+-") {
		terms := splitPolynomialTerms(expr)
		if len(terms) > 1 {
			var diffTerms []string
			for _, term := range terms {
				d := DifferentiateSymbolic(term, v)
				if d != "0" {
					if strings.HasPrefix(d, "-") || len(diffTerms) == 0 {
						diffTerms = append(diffTerms, d)
					} else {
						diffTerms = append(diffTerms, "+"+d)
					}
				}
			}
			if len(diffTerms) == 0 {
				return "0"
			}
			return strings.Join(diffTerms, " ")
		}
	}

	return fmt.Sprintf("d/d%s(%s)", v, expr)
}

// IntegrateSymbolic performs analytical symbolic integration with respect to variable v.
func IntegrateSymbolic(expr, v string) string {
	if v == "" {
		v = "x"
	}
	expr = strings.TrimSpace(expr)

	// Constant
	if c, err := strconv.ParseFloat(expr, 64); err == nil {
		if c == 0 {
			return "C"
		}
		return fmt.Sprintf("%g*%s + C", c, v)
	}

	// Polynomial term: a*x^n or x^n or a*x or x
	polyRe := regexp.MustCompile(`^([-+]?[0-9\.]*)\s*\*?\s*([a-zA-Z])(?:\^([-+]?[0-9\.]+))?$`)
	if m := polyRe.FindStringSubmatch(expr); len(m) > 0 && m[2] == v {
		coefStr := m[1]
		powStr := m[3]

		coef := 1.0
		if coefStr == "-" {
			coef = -1.0
		} else if coefStr != "" && coefStr != "+" {
			coef, _ = strconv.ParseFloat(coefStr, 64)
		}

		pow := 1.0
		if powStr != "" {
			pow, _ = strconv.ParseFloat(powStr, 64)
		}

		if pow == -1 {
			if coef == 1 {
				return fmt.Sprintf("ln(|%s|) + C", v)
			}
			return fmt.Sprintf("%g*ln(|%s|) + C", coef, v)
		}

		newPow := pow + 1.0
		newCoef := coef / newPow

		if newPow == 1 {
			return fmt.Sprintf("%g*%s + C", newCoef, v)
		}
		return fmt.Sprintf("%g*%s^%g + C", newCoef, v, newPow)
	}

	// Trig: cos(x), sin(x), e^x
	if expr == fmt.Sprintf("cos(%s)", v) {
		return fmt.Sprintf("sin(%s) + C", v)
	}
	if expr == fmt.Sprintf("sin(%s)", v) {
		return fmt.Sprintf("-cos(%s) + C", v)
	}
	if expr == fmt.Sprintf("e^%s", v) || expr == fmt.Sprintf("exp(%s)", v) {
		return fmt.Sprintf("e^%s + C", v)
	}

	// Multi-term polynomial integration
	if strings.ContainsAny(expr, "+-") {
		terms := splitPolynomialTerms(expr)
		if len(terms) > 1 {
			var intTerms []string
			for _, term := range terms {
				it := IntegrateSymbolic(term, v)
				it = strings.TrimSuffix(it, " + C")
				if strings.HasPrefix(it, "-") || len(intTerms) == 0 {
					intTerms = append(intTerms, it)
				} else {
					intTerms = append(intTerms, "+"+it)
				}
			}
			return strings.Join(intTerms, " ") + " + C"
		}
	}

	return fmt.Sprintf("∫(%s) d%s + C", expr, v)
}

// ComputeLimit evaluates basic analytical limits: lim_{x -> a} f(x)
func ComputeLimit(expr, v, target string) string {
	if v == "" {
		v = "x"
	}
	expr = strings.TrimSpace(expr)
	target = strings.TrimSpace(target)

	// Check sin(x)/x as x->0
	if (expr == "sin(x)/x" || expr == "sin(x) / x") && (target == "0" || target == "0.0") {
		return "1"
	}

	// Polynomial evaluation at finite limit
	if val, err := strconv.ParseFloat(target, 64); err == nil {
		// Substitute x = val
		subExpr := strings.ReplaceAll(expr, v, fmt.Sprintf("(%g)", val))
		if res, err := evaluateArithmeticSimple(subExpr); err == nil {
			return fmt.Sprintf("%g", res)
		}
	}

	return fmt.Sprintf("lim_{%s -> %s} %s", v, target, expr)
}

func evaluateArithmeticSimple(expr string) (float64, error) {
	// Simple evaluation
	be := NewFastNumericBackend()
	res := be.Execute(MathRequest{Operation: OpEvaluate, Expression: expr})
	if res.Status != StatusSuccess {
		return 0, fmt.Errorf("evaluation error: %s", res.ErrorMessage)
	}
	return strconv.ParseFloat(res.NumericResult, 64)
}

func splitPolynomialTerms(expr string) []string {
	var terms []string
	var cur strings.Builder
	depth := 0

	for i, r := range expr {
		if r == '(' {
			depth++
		} else if r == ')' {
			depth--
		}

		if (r == '+' || r == '-') && depth == 0 && i > 0 {
			t := strings.TrimSpace(cur.String())
			if t != "" {
				terms = append(terms, t)
			}
			cur.Reset()
			if r == '-' {
				cur.WriteRune('-')
			}
			continue
		}
		if r != '+' || depth > 0 || i == 0 {
			cur.WriteRune(r)
		}
	}
	if cur.Len() > 0 {
		t := strings.TrimSpace(cur.String())
		if t != "" {
			terms = append(terms, t)
		}
	}
	return terms
}
