package mathruntime

import (
	"strings"
	"testing"
)

func TestExpressionValidator(t *testing.T) {
	limits := DefaultSecurityLimits()
	v := NewExpressionValidator(limits)

	validCases := []string{
		"2384 * 7291",
		"1/3 + 1/6",
		"differentiate sin(x^2)",
		"solve x^2 + 5x + 6 == 0",
	}
	for _, tc := range validCases {
		ok, err := v.Validate(tc)
		if !ok || err != nil {
			t.Errorf("expected '%s' to be valid, got err: %v", tc, err)
		}
	}

	invalidCases := []string{
		"__import__('os').system('ls')",
		"exec('print(1)')",
		"((1 + 2)",
		"(((((((((((((((((((((((((((((((((((((((((((((((((((1)))))))))))))))))))))))))))))))))))))))))))))))))))",
	}
	for _, tc := range invalidCases {
		ok, _ := v.Validate(tc)
		if ok {
			t.Errorf("expected '%s' to be rejected", tc)
		}
	}
}

func TestFastNumericExactRational(t *testing.T) {
	b := NewFastNumericBackend()

	// 1/3 + 1/6 -> 1/2
	req := MathRequest{
		Expression: "1/3 + 1/6",
		Mode:       ModeExact,
	}
	res := b.Execute(req)
	if res.Status != StatusSuccess {
		t.Fatalf("expected success, got %s: %s", res.Status, res.ErrorMessage)
	}
	if res.ExactResult != "1/2" {
		t.Errorf("expected '1/2', got '%s'", res.ExactResult)
	}

	// 2384 * 7291 -> 17381744
	req2 := MathRequest{
		Expression: "2384 * 7291",
		Mode:       ModeExact,
	}
	res2 := b.Execute(req2)
	if res2.ExactResult != "17381744" {
		t.Errorf("expected '17381744', got '%s'", res2.ExactResult)
	}

	// Determinant
	req3 := MathRequest{
		Operation:  OpDeterminant,
		Expression: "[[1,2],[3,4]]",
	}
	res3 := b.Execute(req3)
	if res3.ExactResult != "-2" {
		t.Errorf("expected '-2', got '%s'", res3.ExactResult)
	}

	// Combinatorics
	req4 := MathRequest{
		Operation:  OpProbability,
		Expression: "C(10, 3)",
	}
	res4 := b.Execute(req4)
	if res4.ExactResult != "120" {
		t.Errorf("expected '120', got '%s'", res4.ExactResult)
	}
}

func TestSymbolicMathics(t *testing.T) {
	runtime := NewMathRuntime(nil)

	// Differentiate
	res1 := runtime.Differentiate("sin(x^2)", "x", 1, "", "")
	if res1.Status != StatusSuccess || !strings.Contains(res1.ExactResult, "cos") {
		t.Errorf("differentiation failed: %v", res1)
	}

	// Integrate
	res2 := runtime.Integrate("x^2", "x", "", "")
	if res2.Status != StatusSuccess || !strings.Contains(res2.ExactResult, "x^3") {
		t.Errorf("integration failed: %v", res2)
	}

	// Solve
	res3 := runtime.Solve("x^2 + 5*x + 6 == 0", "x", "", "")
	if res3.Status != StatusSuccess || !strings.Contains(res3.ExactResult, "-3") {
		t.Errorf("solve failed: %v", res3)
	}

	// Simplify
	res4 := runtime.Simplify("sin(x)^2 + cos(x)^2", "", "")
	if res4.Status != StatusSuccess || res4.ExactResult != "1" {
		t.Errorf("simplify failed: %v", res4)
	}
}

func TestCalculatorRouter(t *testing.T) {
	runtime := NewMathRuntime(nil)

	res1 := runtime.Evaluate("1024 * 4096", ModeExact, "", "")
	if res1.BackendName != "FastNumericNative" {
		t.Errorf("expected FastNumericNative, got %s", res1.BackendName)
	}

	res2 := runtime.Differentiate("sin(x^2)", "x", 1, "", "")
	if res2.BackendName != "Mathics3-Core" {
		t.Errorf("expected Mathics3-Core, got %s", res2.BackendName)
	}

	stats := runtime.GetStats()
	if stats.FastPathCount == 0 || stats.MathicsCount == 0 {
		t.Errorf("expected both fast path and mathics calls in stats: %+v", stats)
	}
}

func TestVerificationLoop(t *testing.T) {
	runtime := NewMathRuntime(nil)

	// Correct claim
	ver1 := runtime.VerifyCalculation("The answer is 17381744.", "2384 * 7291")
	if !ver1.Matches {
		t.Errorf("expected match for accurate calculation")
	}

	// Hallucinated claim
	ver2 := runtime.VerifyCalculation("The answer is 17382094.", "2384 * 7291")
	if ver2.Matches {
		t.Errorf("expected mismatch for hallucinated calculation")
	}
	if ver2.DiscrepancyDetails == "" {
		t.Errorf("expected discrepancy details")
	}
}

func TestIntentInterception(t *testing.T) {
	runtime := NewMathRuntime(nil)

	text := "Please calculate 2,384 * 7,291 for me."
	transformed, results := runtime.InterceptAndEvaluate(text)

	if len(results) == 0 {
		t.Fatalf("expected at least 1 calculation intercepted")
	}
	if results[0].ExactResult != "17381744" {
		t.Errorf("expected '17381744', got '%s'", results[0].ExactResult)
	}
	if !strings.Contains(transformed, "[MathResult:") {
		t.Errorf("expected compact observation in transformed string: %s", transformed)
	}
}

func TestMultiTenantCache(t *testing.T) {
	runtime := NewMathRuntime(nil)

	res := runtime.Evaluate("50 + 50", ModeExact, "tenant_a", "sess_1")
	if res.Status != StatusSuccess {
		t.Fatalf("eval failed")
	}
	if runtime.Cache.Size() == 0 {
		t.Errorf("cache should not be empty")
	}

	runtime.Cache.InvalidateTenant("tenant_a")
	if runtime.Cache.Size() != 0 {
		t.Errorf("cache should be empty after tenant invalidation")
	}
}

func TestBenchmarkQuestionsInterception(t *testing.T) {
	runtime := NewMathRuntime(nil)

	cases := []struct {
		prompt   string
		expected string
	}{
		{"Calculate 123456789 * 987654321. Give only the exact final numerical answer.", "121932631112635269"},
		{"What is 987654321987654321 - 123456789123456789? State the exact difference.", "864197532864197532"},
		{"Compute 847291 * 392817. State the exact product.", "332830308747"},
		{"Evaluate 1/3 + 1/6 + 1/12 as an exact simplified fraction.", "7/12"},
		{"Calculate (7/13) * (26/21) in lowest terms.", "2/3"},
		{"Simplify (5/8 - 1/4) / (3/16) to an exact integer or fraction.", "2"},
		{"Compute the determinant of the 2x2 matrix [[17, 23], [41, 59]]. State the exact integer value.", "60"},
		{"Find the determinant of [[2, 0, 1], [3, 0, 0], [5, 1, 1]].", "3"},
		{"What is the combination C(12, 5)? Give the exact integer result.", "792"},
		{"What is the Greatest Common Divisor GCD(4620, 3696)?", "924"},
		{"Find the inverse of [[4, 7], [2, 6]]. Give only the final matrix directly.", "[[0.6, -0.7], [-0.2, 0.4]]"},
	}

	for _, tc := range cases {
		res := runtime.InterceptAndVerifyIntent(tc.prompt)
		if res == nil {
			t.Errorf("failed to intercept intent for %q", tc.prompt)
			continue
		}
		if res.ExactResult != tc.expected && res.NumericResult != tc.expected {
			t.Errorf("for prompt %q: expected %q, got exact=%q, num=%q", tc.prompt, tc.expected, res.ExactResult, res.NumericResult)
		}
	}
}

