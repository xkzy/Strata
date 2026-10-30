package mathruntime

import (
	"strings"
	"testing"
	"time"
)

func TestOpProveThroughProcessRequest(t *testing.T) {
	rt := NewMathRuntime(nil)
	for claim, want := range map[string]string{
		"2^10 = 1024": "proved", "2 + 2 = 5": "refuted", "4 = 4": "proved", // "4 = 4" must not become an assignment
		"3 ∣ 12 ∧ 5 < 7": "proved", "¬ (3 ≤ 2)": "proved", "10 % 4 = 3": "refuted",
		"-3 < 2 ∧ 3 - 5 = -2": "proved", "(-2)^2 = -4": "refuted", // a negative numeral selects the Int fragment
		"-12 / 7 = -2 ∧ -12 % 7 = 2":        "proved", // E-rounding, Lean's Int.ediv/Int.emod
		"-2 ∣ 4":                            "proved", // Int.dvd, Lean's Exists (fun c => b = a * c)
		"gcd(-6, -9) = 3 ∧ min(-3, 5) = -3": "proved", // Int.gcd is Nat-valued; Int.min is the Min Int instance
	} {
		res := rt.ProcessRequest(MathRequest{Operation: OpProve, Expression: claim})
		if res.Status != StatusSuccess || res.ExactResult != want {
			t.Errorf("%q: status %s result %q (%s), want %s", claim, res.Status, res.ExactResult, res.ErrorMessage, want)
		}
		if res.BackendName != "Lean4-Kernel-Go" || res.Metadata["proof_term"] == nil {
			t.Errorf("%q: missing kernel certificate: %+v", claim, res)
		}
	}
	if res := rt.ProveWithLean("x + 1 = 2"); res.Status != StatusUnsupportedOp {
		t.Errorf("variables are outside the fragment: %s", res.Status)
	}
	if res := rt.ProveWithLean("fact(-3) = 1"); res.Status != StatusUnsupportedOp || !strings.Contains(res.ErrorMessage, "Int") {
		t.Errorf("factorial is outside the Int fragment: %s (%s)", res.Status, res.ErrorMessage)
	}
	if res := rt.ProveWithLean(strings.Repeat("1+", 3000) + "1 = 1"); res.Status != StatusInvalidExpression {
		t.Errorf("the runtime's length limit must apply to claims: %s", res.Status)
	}
	if st := rt.GetStats(); st.LeanCount < 13 || st.LeanRefuted != 3 {
		t.Errorf("stats: %+v", st)
	}
}

func TestProveHonoursStepBudget(t *testing.T) {
	rt := NewMathRuntime(nil)
	rt.LeanBackend.MaxSteps = 5
	if res := rt.ProveWithLean("2 + 2 = 4"); res.Status != StatusTimeout {
		t.Errorf("want a budget stop, got %s (%s)", res.Status, res.ErrorMessage)
	}
}

func TestKernelCrossChecksTheCAS(t *testing.T) {
	rt := NewMathRuntime(nil)
	ver := rt.VerifyCalculation("The answer is = 1024", "2^10")
	if !ver.Matches || !ver.KernelCertified || ver.Confidence != 1.0 {
		t.Errorf("2^10 = 1024 should be kernel-certified: %+v", ver)
	}
	// subtraction means something else on Nat, so the kernel is not asked
	ver = rt.VerifyCalculation("= 7", "10 - 3")
	if !ver.Matches || ver.KernelCertified {
		t.Errorf("10 - 3 is outside the shared fragment: %+v", ver)
	}
	if checked, agrees := rt.certify("2^10", "1025"); !checked || agrees {
		t.Errorf("kernel must refute 2^10 = 1025: %v %v", checked, agrees)
	}
	if checked, _ := rt.certify("10 - 3", "7"); checked {
		t.Error("10 - 3 must not be certified through Nat")
	}
	// a CAS answer the kernel refutes turns a match into a failure
	var v MathVerificationResult
	v.Matches = true
	rt.crossCheckWithKernel(&v, "6*7", "43")
	if v.Matches || !strings.Contains(v.DiscrepancyDetails, "kernel refutes") {
		t.Errorf("disagreement not reported: %+v", v)
	}
}

func TestProveHonoursDeadline(t *testing.T) {
	rt := NewMathRuntime(nil)
	rt.LeanBackend.MaxSteps = 1 << 40
	rt.LeanBackend.Timeout = 20 * time.Millisecond
	// fifteen ~280k-bit powers multiplied together: about 200 ms of big-integer work, each step under the kernel's size cap
	claim := strings.TrimSuffix(strings.Repeat("(9999^20000)*", 15), "*") + " = 1"
	start := time.Now()
	res := rt.ProveWithLean(claim)
	if res.Status != StatusTimeout {
		t.Fatalf("want a timeout, got %s %q", res.Status, res.ErrorMessage)
	}
	if elapsed := time.Since(start); elapsed > 250*time.Millisecond {
		t.Errorf("stopped after %v with a 20ms deadline", elapsed)
	}
}

func TestInterceptsLeanDeclarations(t *testing.T) {
	rt := NewMathRuntime(nil)
	res := rt.InterceptAndVerifyIntent("Here it is:\nexample : 2^10 = 1024 := by decide\nDone.")
	if res == nil || res.ExactResult != "proved" || res.BackendName != "Lean4-Kernel-Go" {
		t.Fatalf("got %+v", res)
	}
	res = rt.InterceptAndVerifyIntent("theorem t : 7 * 6 = 41 := by norm_num")
	if res == nil || res.ExactResult != "refuted" {
		t.Fatalf("a wrong theorem must come back refuted: %+v", res)
	}
}

func TestProveIsSafeConcurrently(t *testing.T) {
	rt := NewMathRuntime(nil)
	done := make(chan string, 64)
	for i := 0; i < 64; i++ {
		go func(i int) {
			claim, want := "2^10 = 1024", "proved"
			if i%2 == 1 {
				claim, want = "2^10 = 1025", "refuted"
			}
			if res := rt.ProveWithLean(claim); res.ExactResult != want {
				done <- claim + " gave " + res.ExactResult
				return
			}
			done <- ""
		}(i)
	}
	for i := 0; i < 64; i++ {
		if msg := <-done; msg != "" {
			t.Error(msg)
		}
	}
}

func TestCASPowerIsRightAssociative(t *testing.T) {
	rt := NewMathRuntime(nil)
	for expr, want := range map[string]string{"2^3^2": "512", "2^2^3": "256", "(2^3)^2": "64", "2*3^2": "18", "10-2-3": "5", "2^3*2": "16"} {
		if got := rt.Evaluate(expr, ModeExact, "", "").ExactResult; got != want {
			t.Errorf("%s = %s, want %s", expr, got, want)
		}
	}
}
