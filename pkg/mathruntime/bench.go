package mathruntime

import (
	"strings"
	"time"
)

// MathBenchmarkItem defines a test expression and its expected simplification/evaluation.
type MathBenchmarkItem struct {
	Category string
	Expr     string
	Expected string
}

// MathBenchmarkReport summarizes benchmark runs on the mathematical CAS engine.
type MathBenchmarkReport struct {
	TotalProblems  int           `json:"total_problems"`
	PassedProblems int           `json:"passed_problems"`
	Duration       time.Duration `json:"duration_ms"`
	Accuracy       float64       `json:"accuracy"`
}

// DefaultMathBenchmarkSuite provides mathematical problems across arithmetic, algebra, and calculus.
var DefaultMathBenchmarkSuite = []MathBenchmarkItem{
	{Category: "Arithmetic", Expr: "123456789 * 987654321", Expected: "121932631112635269"},
	{Category: "Rational", Expr: "1/3 + 1/6 + 1/12", Expected: "7/12"},
	{Category: "Rational", Expr: "(7/13) * (26/21)", Expected: "2/3"},
	{Category: "Algebra", Expr: "Expand[(x + 1)^2]", Expected: "x^2 + 2*x + 1"},
	{Category: "Calculus", Expr: "D[x^3 + 2*x, x]", Expected: "3*x^2 + 2"},
}

// RunMathBenchmark executes standard benchmark suite against the math runtime.
func (mr *MathRuntime) RunMathBenchmark(items []MathBenchmarkItem) *MathBenchmarkReport {
	if len(items) == 0 {
		items = DefaultMathBenchmarkSuite
	}

	start := time.Now()
	passed := 0

	for _, item := range items {
		req := MathRequest{
			Expression: item.Expr,
			Operation:  OpEvaluate,
			Mode:       ModeExact,
		}
		res := mr.ProcessRequest(req)
		if res.Status == StatusSuccess && (strings.Contains(res.ExactResult, item.Expected) || res.ExactResult == item.Expected) {
			passed++
		}
	}

	elapsed := time.Since(start)
	acc := 0.0
	if len(items) > 0 {
		acc = float64(passed) / float64(len(items)) * 100.0
	}

	return &MathBenchmarkReport{
		TotalProblems:  len(items),
		PassedProblems: passed,
		Duration:       elapsed,
		Accuracy:       acc,
	}
}
