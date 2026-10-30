package lean

import "testing"

func TestPreludeBuilds(t *testing.T) {
	env, err := BuildPrelude()
	if err != nil {
		t.Fatal(err)
	}
	if got := env.Axioms(); len(got) != len(PreludeAxioms) {
		t.Fatalf("axioms %v", got)
	}
}

// The prelude admits nothing: the Nat and Int constants are definitions or theorems, so every certificate the
// runtime hands out is checked from the kernel's own rules alone.
func TestPreludeAdmitsNothing(t *testing.T) {
	env := mustPrelude(t)
	if ax := env.Axioms(); len(ax) != 0 {
		t.Fatalf("the prelude still admits %v", ax)
	}
	// The order decision is the two proved directions and nothing wider: a `Nat.ble` that computes to false cannot
	// be passed off as a proof of `Nat.le`, and one that computes to true can.
	wantReject(t, env, nil, "Nat.le_of_ble_eq_true 5 2 (Eq.refl.{1} Bool Bool.true)", "false must not prove 5 ≤ 2")
	wantAccept(t, env, nil, "Nat.le_of_ble_eq_true 2 5 (Eq.refl.{1} Bool Bool.true)")
	wantAccept(t, env, nil, "Nat.not_le_of_not_ble_eq_true 5 2 (Bool.false_ne_true)")
	if ty, err := checkTerm(t, env, nil, "Nat.le_of_ble_eq_true"); err != nil || ty == nil {
		t.Fatalf("the theorem is not in the environment: %v", err)
	}

	// A claim about order is decided by these proofs: one certificate per direction, and the false one is refused.
	for claim, holds := range map[string]bool{"3 ≤ 5": true, "¬ (5 ≤ 3)": true, "5 ≤ 3": false, "3 ≤ 3": true} {
		p, err := ProveClaim(claim, Options{MaxSteps: 1_000_000})
		if holds && err != nil {
			t.Errorf("%q should be provable: %v", claim, err)
		}
		if !holds && err == nil {
			t.Errorf("%q is false but the kernel accepted %v", claim, p.Term)
		}
	}
}
