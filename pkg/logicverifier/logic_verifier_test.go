// pkg/logicverifier/logic_verifier_test.go - Comprehensive Test Suite for Go Logic Verifier
package logicverifier

import (
	"testing"
)

func TestLogicVerifier_Arithmetic(t *testing.T) {
	lv := NewLogicVerifier()

	// 1. Exact multiplication
	claim1 := VerificationClaim{
		ClaimID:      "test-arith-1",
		Type:         ClaimArithmetic,
		Expression:   "2384 * 7291",
		ClaimedValue: "17381744",
	}
	res1 := lv.Verify(claim1)
	if res1.Status != StatusPass {
		t.Fatalf("expected PASS for 2384 * 7291 == 17381744, got %s (reason: %s)", res1.Status, res1.FailureReason)
	}

	// 2. Arithmetic failure
	claim2 := VerificationClaim{
		ClaimID:      "test-arith-2",
		Type:         ClaimArithmetic,
		Expression:   "2384 * 7291",
		ClaimedValue: "17366800",
	}
	res2 := lv.Verify(claim2)
	if res2.Status != StatusFail {
		t.Fatalf("expected FAIL for incorrect multiplication, got %s", res2.Status)
	}

	// 3. Exact rational addition 1/3 + 1/6 == 0.5 (or 1/2)
	claim3 := VerificationClaim{
		ClaimID:      "test-arith-3",
		Type:         ClaimArithmetic,
		Expression:   "1/3 + 1/6",
		ClaimedValue: "0.5",
	}
	res3 := lv.Verify(claim3)
	if res3.Status != StatusPass {
		t.Fatalf("expected PASS for 1/3 + 1/6 == 0.5, got %s (reason: %s)", res3.Status, res3.FailureReason)
	}

	// 4. Sqrt calculation
	claim4 := VerificationClaim{
		ClaimID:      "test-arith-4",
		Type:         ClaimArithmetic,
		Expression:   "sqrt(144)",
		ClaimedValue: "12",
	}
	res4 := lv.Verify(claim4)
	if res4.Status != StatusPass {
		t.Fatalf("expected PASS for sqrt(144) == 12, got %s", res4.Status)
	}
}

func TestLogicVerifier_PropositionalRules(t *testing.T) {
	lv := NewLogicVerifier()

	// 1. Modus Ponens: A -> B, A => B (PASS)
	claim1 := VerificationClaim{
		ClaimID:      "test-rule-1",
		Type:         ClaimProposition,
		Premises:     []string{"A -> B", "A"},
		ClaimedValue: "B",
	}
	res1 := lv.Verify(claim1)
	if res1.Status != StatusPass {
		t.Fatalf("expected PASS for Modus Ponens, got %s (reason: %s)", res1.Status, res1.FailureReason)
	}

	// 2. Affirming the Consequent: A -> B, B => A (FAIL - fallacy)
	claim2 := VerificationClaim{
		ClaimID:      "test-rule-2",
		Type:         ClaimProposition,
		Premises:     []string{"A -> B", "B"},
		ClaimedValue: "A",
	}
	res2 := lv.Verify(claim2)
	if res2.Status != StatusFail {
		t.Fatalf("expected FAIL for Affirming Consequent fallacy, got %s", res2.Status)
	}

	// 3. Contradictory Premises: A, ~A (FAIL)
	claim3 := VerificationClaim{
		ClaimID:      "test-rule-3",
		Type:         ClaimProposition,
		Premises:     []string{"A", "!A"},
		ClaimedValue: "B",
	}
	res3 := lv.Verify(claim3)
	if res3.Status != StatusFail {
		t.Fatalf("expected FAIL for contradictory premises, got %s", res3.Status)
	}
}

func TestLogicVerifier_ConstraintEngine(t *testing.T) {
	lv := NewLogicVerifier()

	// 1. In-bound constraint: x > 0, x < 10, x == 5 (PASS)
	claim1 := VerificationClaim{
		ClaimID:      "test-const-1",
		Type:         ClaimConstraint,
		Constraints:  []string{"x > 0", "x < 10"},
		ClaimedValue: "x == 5",
	}
	res1 := lv.Verify(claim1)
	if res1.Status != StatusPass {
		t.Fatalf("expected PASS for x=5 in (0, 10), got %s (reason: %s)", res1.Status, res1.FailureReason)
	}

	// 2. Out-of-bound constraint: x > 0, x < 10, x == 15 (FAIL)
	claim2 := VerificationClaim{
		ClaimID:      "test-const-2",
		Type:         ClaimConstraint,
		Constraints:  []string{"x > 0", "x < 10"},
		ClaimedValue: "x = 15",
	}
	res2 := lv.Verify(claim2)
	if res2.Status != StatusFail {
		t.Fatalf("expected FAIL for x=15 in (0, 10), got %s", res2.Status)
	}

	// 3. Contradictory constraints: x > 10, x < 5 (FAIL)
	claim3 := VerificationClaim{
		ClaimID:      "test-const-3",
		Type:         ClaimConstraint,
		Constraints:  []string{"x > 10", "x < 5"},
		ClaimedValue: "x == 7",
	}
	res3 := lv.Verify(claim3)
	if res3.Status != StatusFail {
		t.Fatalf("expected FAIL for contradictory constraints, got %s", res3.Status)
	}
}

func TestLogicVerifier_UnitDimensions(t *testing.T) {
	lv := NewLogicVerifier()

	// 1. Compatible conversion: 1000 m == 1 km (PASS)
	claim1 := VerificationClaim{
		ClaimID:      "test-unit-1",
		Type:         ClaimUnitDimension,
		Expression:   "1000 m",
		ClaimedValue: "1 km",
	}
	res1 := lv.Verify(claim1)
	if res1.Status != StatusPass {
		t.Fatalf("expected PASS for 1000 m == 1 km, got %s (reason: %s)", res1.Status, res1.FailureReason)
	}

	// 2. Dimensional Mismatch: 50 m/s == 50 kg (FAIL)
	claim2 := VerificationClaim{
		ClaimID:      "test-unit-2",
		Type:         ClaimUnitDimension,
		Expression:   "50 m/s",
		ClaimedValue: "50 kg",
	}
	res2 := lv.Verify(claim2)
	if res2.Status != StatusFail {
		t.Fatalf("expected FAIL for m/s vs kg dimensional mismatch, got %s", res2.Status)
	}

	// 3. Value mismatch with same dimension: 10 m == 20 m (FAIL)
	claim3 := VerificationClaim{
		ClaimID:      "test-unit-3",
		Type:         ClaimUnitDimension,
		Expression:   "10 m",
		ClaimedValue: "20 m",
	}
	res3 := lv.Verify(claim3)
	if res3.Status != StatusFail {
		t.Fatalf("expected FAIL for 10 m == 20 m, got %s", res3.Status)
	}
}

func TestLogicVerifier_SchemaVerification(t *testing.T) {
	lv := NewLogicVerifier()

	schema := `{
		"type": "object",
		"required": ["name", "age"],
		"properties": {
			"name": {"type": "string"},
			"age": {"type": "number"}
		}
	}`

	// 1. Valid JSON matching schema (PASS)
	claim1 := VerificationClaim{
		ClaimID:      "test-schema-1",
		Type:         ClaimSchemaType,
		ClaimedValue: `{"name": "Alice", "age": 30}`,
		SchemaJSON:   schema,
	}
	res1 := lv.Verify(claim1)
	if res1.Status != StatusPass {
		t.Fatalf("expected PASS for schema match, got %s (reason: %s)", res1.Status, res1.FailureReason)
	}

	// 2. Missing required property (FAIL)
	claim2 := VerificationClaim{
		ClaimID:      "test-schema-2",
		Type:         ClaimSchemaType,
		ClaimedValue: `{"name": "Bob"}`,
		SchemaJSON:   schema,
	}
	res2 := lv.Verify(claim2)
	if res2.Status != StatusFail {
		t.Fatalf("expected FAIL for missing required property, got %s", res2.Status)
	}

	// 3. Type mismatch (FAIL)
	claim3 := VerificationClaim{
		ClaimID:      "test-schema-3",
		Type:         ClaimSchemaType,
		ClaimedValue: `{"name": "Charlie", "age": "thirty"}`,
		SchemaJSON:   schema,
	}
	res3 := lv.Verify(claim3)
	if res3.Status != StatusFail {
		t.Fatalf("expected FAIL for type mismatch, got %s", res3.Status)
	}
}

func TestLogicVerifier_CompositeAndTriState(t *testing.T) {
	lv := NewLogicVerifier()

	// 1. Composite: 1 PASS, 1 FAIL => PARTIAL
	compositeClaim := VerificationClaim{
		ClaimID: "test-comp-1",
		Type:    ClaimComposite,
		SubClaims: []VerificationClaim{
			{
				ClaimID:      "sub-1",
				Type:         ClaimArithmetic,
				Expression:   "2 + 2",
				ClaimedValue: "4",
			},
			{
				ClaimID:      "sub-2",
				Type:         ClaimArithmetic,
				Expression:   "5 * 5",
				ClaimedValue: "30", // Wrong
			},
		},
	}
	res := lv.Verify(compositeClaim)
	if res.Status != StatusPartial {
		t.Fatalf("expected PARTIAL for mixed composite, got %s", res.Status)
	}

	// 2. Never convert UNKNOWN to PASS rule
	unknownClaim := VerificationClaim{
		ClaimID:      "test-unknown-1",
		Type:         ClaimUnknown,
		RawStatement: "The stock price will double tomorrow",
	}
	resUnknown := lv.Verify(unknownClaim)
	if resUnknown.Status != StatusUnknown {
		t.Fatalf("fundamental rule violation: expected UNKNOWN, got %s", resUnknown.Status)
	}
}

func TestLogicVerifier_TextExtractionAndVerification(t *testing.T) {
	lv := NewLogicVerifier()

	text := `In summary, the total calculation gives 2384 * 7291 = 17381744.
Furthermore, considering the constraints x > 0 and x < 10, we find x = 5.`

	results := lv.VerifyText(text, "tenant-1", "session-1")
	if len(results) < 2 {
		t.Fatalf("expected at least 2 extracted claims and results, got %d", len(results))
	}

	for _, r := range results {
		if r.Status != StatusPass {
			t.Errorf("expected extracted claim %s to PASS, got %s (%s)", r.ClaimID, r.Status, r.FailureReason)
		}
	}

	obs := lv.FormatObservations(results)
	if obs == "" {
		t.Fatalf("expected non-empty compact observation")
	}

	metrics := lv.GetMetrics()
	if metrics.TotalVerifications == 0 {
		t.Fatalf("expected non-zero metrics recorded")
	}
}
