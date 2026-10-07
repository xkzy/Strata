package mathruntime

import (
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
)

// QuadraticSolution represents the analytical solutions of a x^2 + b x + c = 0
type QuadraticSolution struct {
	A        float64
	B        float64
	C        float64
	Disc     float64
	Root1    float64
	Root2    float64
	IsReal   bool
	ExactStr string
}

// SolveQuadratic solves a*x^2 + b*x + c = 0
func SolveQuadratic(a, b, c float64) QuadraticSolution {
	if a == 0 {
		if b == 0 {
			return QuadraticSolution{IsReal: false, ExactStr: "No solution or infinite solutions"}
		}
		r := -c / b
		return QuadraticSolution{A: a, B: b, C: c, Root1: r, Root2: r, IsReal: true, ExactStr: fmt.Sprintf("x = %g", r)}
	}

	disc := b*b - 4*a*c
	if disc >= 0 {
		r1 := (-b + math.Sqrt(disc)) / (2 * a)
		r2 := (-b - math.Sqrt(disc)) / (2 * a)
		if disc == 0 {
			return QuadraticSolution{
				A: a, B: b, C: c, Disc: disc,
				Root1: r1, Root2: r2, IsReal: true,
				ExactStr: fmt.Sprintf("x = %g", r1),
			}
		}
		return QuadraticSolution{
			A: a, B: b, C: c, Disc: disc,
			Root1: r1, Root2: r2, IsReal: true,
			ExactStr: fmt.Sprintf("x = %g, x = %g", r1, r2),
		}
	}

	// Complex roots
	realPart := -b / (2 * a)
	imagPart := math.Sqrt(-disc) / (2 * a)
	return QuadraticSolution{
		A: a, B: b, C: c, Disc: disc,
		IsReal:   false,
		ExactStr: fmt.Sprintf("x = %g + %gi, x = %g - %gi", realPart, imagPart, realPart, imagPart),
	}
}

// ParseAndSolveQuadratic parses expressions like "x^2 - 5*x + 6 = 0" or "2*x^2 + 4*x - 6"
func ParseAndSolveQuadratic(expr string) (QuadraticSolution, error) {
	expr = strings.ReplaceAll(expr, " ", "")
	expr = strings.TrimSuffix(expr, "==0")
	expr = strings.TrimSuffix(expr, "=0")

	// Match: a*x^2 + b*x + c
	quadRe := regexp.MustCompile(`^([-+]?[0-9\.]*)?\*?[a-zA-Z]\^2(?:([-+]?[0-9\.]*)\*?[a-zA-Z])?([-+]?[0-9\.]+)?$`)
	m := quadRe.FindStringSubmatch(expr)
	if len(m) == 0 {
		return QuadraticSolution{}, fmt.Errorf("could not parse quadratic expression: %s", expr)
	}

	aStr, bStr, cStr := m[1], m[2], m[3]

	a := 1.0
	if aStr == "-" {
		a = -1.0
	} else if aStr != "" && aStr != "+" {
		a, _ = strconv.ParseFloat(aStr, 64)
	}

	b := 0.0
	if bStr == "-" {
		b = -1.0
	} else if bStr == "+" {
		b = 1.0
	} else if bStr != "" {
		b, _ = strconv.ParseFloat(bStr, 64)
	}

	c := 0.0
	if cStr != "" {
		c, _ = strconv.ParseFloat(cStr, 64)
	}

	return SolveQuadratic(a, b, c), nil
}

// NewtonRaphson finds a root of f(x) = 0 given f, fPrime, initial guess x0, tolerance tol, and maxIter.
func NewtonRaphson(f func(float64) float64, fPrime func(float64) float64, x0 float64, tol float64, maxIter int) (float64, error) {
	x := x0
	for i := 0; i < maxIter; i++ {
		y := f(x)
		if math.Abs(y) < tol {
			return x, nil
		}
		dy := fPrime(x)
		if math.Abs(dy) < 1e-14 {
			return x, fmt.Errorf("derivative near zero, method failed at iteration %d", i)
		}
		xNext := x - y/dy
		if math.Abs(xNext-x) < tol {
			return xNext, nil
		}
		x = xNext
	}
	return x, fmt.Errorf("failed to converge within %d iterations", maxIter)
}
