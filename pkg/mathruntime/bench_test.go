package mathruntime

import (
	"testing"
)

func TestMathBenchmark(t *testing.T) {
	mr := NewMathRuntime(nil)
	report := mr.RunMathBenchmark(nil)
	if report.TotalProblems == 0 {
		t.Fatalf("expected non-zero total problems")
	}
	if report.PassedProblems == 0 {
		t.Errorf("expected at least some problems to pass")
	}
}
