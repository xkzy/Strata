package mathruntime

import (
	"testing"
)

func TestSageBackend_SyntaxTranslation(t *testing.T) {
	backend := NewSageBackend()

	s1 := backend.TranslateSageSyntax("var('x'); x^2 + 1")
	if s1 != "x^2 + 1" {
		t.Errorf("expected 'x^2 + 1', got '%s'", s1)
	}

	s2 := backend.TranslateSageSyntax("(x^3 - 8).factor()")
	if s2 != "factor(x^3 - 8)" {
		t.Errorf("expected 'factor(x^3 - 8)', got '%s'", s2)
	}

	s3 := backend.TranslateSageSyntax("(x^2 + 2*x + 1).roots(x)")
	if s3 != "solve(x^2 + 2*x + 1 == 0, x)" {
		t.Errorf("expected 'solve(x^2 + 2*x + 1 == 0, x)', got '%s'", s3)
	}

	s4 := backend.TranslateSageSyntax("matrix([[1, 2], [3, 4]])")
	if s4 != "[[1, 2], [3, 4]]" {
		t.Errorf("expected '[[1, 2], [3, 4]]', got '%s'", s4)
	}
}

func TestSageBackend_NumberTheory(t *testing.T) {
	backend := NewSageBackend()

	// is_prime(1000000007)
	res1 := backend.Execute(MathRequest{Expression: "is_prime(1000000007)", Operation: OpEvaluate})
	if res1.Status != StatusSuccess || res1.ExactResult != "True" {
		t.Errorf("expected True for 1000000007 prime test, got %v (%s)", res1.Status, res1.ExactResult)
	}

	// is_prime(1000000005)
	res2 := backend.Execute(MathRequest{Expression: "is_prime(1000000005)", Operation: OpEvaluate})
	if res2.Status != StatusSuccess || res2.ExactResult != "False" {
		t.Errorf("expected False for 1000000005 prime test, got %v (%s)", res2.Status, res2.ExactResult)
	}

	// euler_phi(100) -> 40
	res3 := backend.Execute(MathRequest{Expression: "euler_phi(100)", Operation: OpEvaluate})
	if res3.Status != StatusSuccess || res3.ExactResult != "40" {
		t.Errorf("expected 40 for euler_phi(100), got %v (%s)", res3.Status, res3.ExactResult)
	}

	// fibonacci(20) -> 6765
	res4 := backend.Execute(MathRequest{Expression: "fibonacci(20)", Operation: OpEvaluate})
	if res4.Status != StatusSuccess || res4.ExactResult != "6765" {
		t.Errorf("expected 6765 for fibonacci(20), got %v (%s)", res4.Status, res4.ExactResult)
	}

	// power_mod(7, 100, 101) -> 1
	res5 := backend.Execute(MathRequest{Expression: "power_mod(7, 100, 101)", Operation: OpEvaluate})
	if res5.Status != StatusSuccess || res5.ExactResult != "1" {
		t.Errorf("expected 1 for power_mod(7, 100, 101), got %v (%s)", res5.Status, res5.ExactResult)
	}

	// gcd(240, 46) -> 2
	res6 := backend.Execute(MathRequest{Expression: "gcd(240, 46)", Operation: OpEvaluate})
	if res6.Status != StatusSuccess || res6.ExactResult != "2" {
		t.Errorf("expected 2 for gcd(240, 46), got %v (%s)", res6.Status, res6.ExactResult)
	}
}

func TestSageBackend_RuntimeIntegration(t *testing.T) {
	runtime := NewMathRuntime(nil)

	req := MathRequest{
		Expression: "var('x'); (x^2 + 5*x + 6).factor()",
		Operation:  OpEvaluate,
	}
	res := runtime.ProcessRequest(req)
	if res.Status != StatusSuccess {
		t.Fatalf("expected success, got error %s", res.ErrorMessage)
	}
	if res.BackendName != "SageMath" {
		t.Errorf("expected SageMath backend routing, got %s", res.BackendName)
	}
}
