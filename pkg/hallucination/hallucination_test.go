// pkg/hallucination/hallucination_test.go - Comprehensive Deterministic Hallucination Subsystem Tests
package hallucination

import (
	"context"
	"testing"
	"time"
)

func TestMathVerification(t *testing.T) {
	runtime := NewRuntime("../..", DefaultSafetyControllerOptions())

	// 1. Correct arithmetic
	rep1, err := runtime.EvaluateText(context.Background(), "The calculation is 37 * 19 = 703", "sess-1", "req-1", "tenant-1", PolicyStrict)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if rep1.Coverage.VerifiedCount == 0 {
		t.Errorf("expected 37 * 19 = 703 to be verified, got: %+v", rep1.Coverage)
	}
	if !rep1.IsClean {
		t.Errorf("expected report to be clean")
	}

	// 2. Contradictory arithmetic
	rep2, err := runtime.EvaluateText(context.Background(), "The calculation is 37 * 19 = 713", "sess-2", "req-2", "tenant-1", PolicyStrict)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if rep2.Coverage.ContradictedCount == 0 {
		t.Errorf("expected 37 * 19 = 713 to be CONTRADICTED, got: %+v", rep2.Coverage)
	}
	if rep2.IsClean {
		t.Errorf("expected report to NOT be clean")
	}
}

func TestUnitDimensionalVerification(t *testing.T) {
	runtime := NewRuntime("../..", DefaultSafetyControllerOptions())

	// 1. Correct physical dimension
	rep1, err := runtime.EvaluateText(context.Background(), "Speed is 10 m / 2 s = 5 m/s", "sess-3", "req-3", "tenant-1", PolicyStrict)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if rep1.Coverage.VerifiedCount == 0 {
		t.Errorf("expected unit equation to be verified, got: %+v", rep1.Coverage)
	}

	// 2. Impossible physical dimension
	rep2, err := runtime.EvaluateText(context.Background(), "Speed is 10 m / 2 s = 5 kg", "sess-4", "req-4", "tenant-1", PolicyStrict)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if rep2.Coverage.ContradictedCount == 0 {
		t.Errorf("expected 10 m / 2 s = 5 kg to be CONTRADICTED, got: %+v", rep2.Coverage)
	}
}

func TestCodebaseFilesystemVerification(t *testing.T) {
	runtime := NewRuntime("../..", DefaultSafetyControllerOptions())

	// 1. Real existing file
	rep1, err := runtime.EvaluateText(context.Background(), "Look in file pkg/hallucination/types.go for the definitions.", "sess-5", "req-5", "tenant-1", PolicyStrict)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if rep1.Coverage.VerifiedCount == 0 {
		t.Errorf("expected types.go to be verified, got: %+v", rep1.Coverage)
	}

	// 2. Non-existent file
	rep2, err := runtime.EvaluateText(context.Background(), "Check file pkg/server/non_existent_fake_file.go", "sess-6", "req-6", "tenant-1", PolicyStrict)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if rep2.Coverage.ContradictedCount == 0 {
		t.Errorf("expected non-existent file to be contradicted, got: %+v", rep2.Coverage)
	}
}

func TestAPIRouteVerification(t *testing.T) {
	runtime := NewRuntime("../..", DefaultSafetyControllerOptions())

	// 1. Real API route
	rep1, err := runtime.EvaluateText(context.Background(), "Send request to POST /v1/chat/completions", "sess-7", "req-7", "tenant-1", PolicyStrict)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if rep1.Coverage.VerifiedCount == 0 {
		t.Errorf("expected /v1/chat/completions to be verified, got: %+v", rep1.Coverage)
	}

	// 2. Fake API route
	rep2, err := runtime.EvaluateText(context.Background(), "Send request to POST /v1/telepathic_quantum_route", "sess-8", "req-8", "tenant-1", PolicyStrict)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if rep2.Coverage.ContradictedCount == 0 {
		t.Errorf("expected fake API route to be contradicted, got: %+v", rep2.Coverage)
	}
}

func TestConservativeOpinionFiltering(t *testing.T) {
	runtime := NewRuntime("../..", DefaultSafetyControllerOptions())

	// Opinion / Hypothetical statement should be UNKNOWN and NOT falsely flagged as CONTRADICTED
	rep, err := runtime.EvaluateText(context.Background(), "In my opinion Linux is the best operating system", "sess-9", "req-9", "tenant-1", PolicyStrict)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if rep.Coverage.ContradictedCount > 0 {
		t.Errorf("opinion should not be flagged as contradiction: %+v", rep.Coverage)
	}
	if rep.Coverage.UnknownCount == 0 {
		t.Errorf("expected opinion to have status UNKNOWN, got: %+v", rep.Coverage)
	}
	if !rep.IsClean {
		t.Errorf("opinion should not violate clean response requirement")
	}
}

func TestCircuitBreaker(t *testing.T) {
	opts := SafetyControllerOptions{
		DefaultPolicy:    PolicyRegenerate,
		MaxRegenAttempts: 2,
	}
	controller := NewGenerationSafetyController(opts)

	// Simulated contradiction report
	contraReport := &HallucinationReport{
		Policy: PolicyRegenerate,
		Coverage: EvidenceCoverage{
			ContradictedCount: 1,
		},
	}

	// Attempt 1: Trigger REGENERATE
	act1, temp1, _ := controller.EvaluateGeneration("sess-cb", contraReport)
	if act1 != ActionRegenerate || temp1 >= 0 {
		t.Errorf("expected REGENERATE with lowered temp, got: %s (temp: %f)", act1, temp1)
	}

	// Attempt 2: Trigger REGENERATE
	act2, _, _ := controller.EvaluateGeneration("sess-cb", contraReport)
	if act2 != ActionRegenerate {
		t.Errorf("expected REGENERATE on attempt 2, got: %s", act2)
	}

	// Attempt 3: Circuit breaker trips -> BLOCK
	act3, _, _ := controller.EvaluateGeneration("sess-cb", contraReport)
	if act3 != ActionBlock {
		t.Errorf("expected BLOCK after circuit breaker trips, got: %s", act3)
	}
}

func TestEvidenceGraphExplainability(t *testing.T) {
	runtime := NewRuntime("../..", DefaultSafetyControllerOptions())

	runtime.AddEvidence(Evidence{
		ID:            "ev-arch-1",
		SourceType:    SourceConfiguration,
		SourceID:      "config.json",
		Authority:     AuthorityTrustedSystem,
		ExtractedFact: "Strata inference engine is written in C++ and CUDA",
		Provenance:    "SystemConfiguration",
		Timestamp:     time.Now(),
	})

	rep, err := runtime.EvaluateText(context.Background(), "The calculation 10 + 5 = 15 is valid", "sess-exp", "req-exp", "tenant-1", PolicyMonitor)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(rep.Claims) == 0 {
		t.Fatalf("expected claims to be extracted")
	}

	claimID := rep.Claims[0].ID
	exp := runtime.ExplainClaim(claimID)

	if exp.ClaimID != claimID {
		t.Errorf("expected claim ID %s, got %s", claimID, exp.ClaimID)
	}
	if exp.Status != StatusVerified {
		t.Errorf("expected StatusVerified, got %s", exp.Status)
	}
}
