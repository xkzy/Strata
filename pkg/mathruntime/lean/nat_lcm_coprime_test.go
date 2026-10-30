package lean

import "testing"

// The ported lcm / Coprime lemmas have their statements pinned: each is applied at an ascribed type, so a drifted
// statement fails here and not only inside the prelude build.
func TestNatLcmCoprimeStatements(t *testing.T) {
	env := mustPrelude(t)
	ascribe := func(ty, term string) string {
		return "(fun (m : Nat) (n : Nat) (k : Nat) => (fun (x : " + ty + ") => x) " + term + ")"
	}
	for _, src := range []string{
		ascribe("Eq.{1} Nat (Nat.lcm (Nat.mul m n) (Nat.mul m k)) (Nat.mul m (Nat.lcm n k))", "(Nat.lcm_mul_left m n k)"),
		ascribe("Eq.{1} Nat (Nat.lcm (Nat.mul m n) (Nat.mul k n)) (Nat.mul (Nat.lcm m k) n)", "(Nat.lcm_mul_right m n k)"),
		ascribe("Eq.{1} Nat (Nat.lcm (Nat.pow m k) (Nat.pow n k)) (Nat.pow (Nat.lcm m n) k)", "(Nat.pow_lcm_pow m n k)"),
		ascribe("Or (Eq.{1} Nat m 0) (Nat.lt 0 m)", "(Nat.eq_zero_or_pos m)"),
	} {
		wantAccept(t, env, nil, src)
	}
	wantReject(t, env, nil,
		"(fun (m : Nat) (n : Nat) (k : Nat) => (fun (x : Eq.{1} Nat (Nat.lcm (Nat.mul m n) (Nat.mul m k)) (Nat.lcm n k)) => x) (Nat.lcm_mul_left m n k))",
		"lcm_mul_left pulls the common factor out; it does not drop it")
	wantReject(t, env, nil,
		"(fun (m : Nat) (n : Nat) (H : Nat.lt 1 m) => Nat.Coprime.gcd_mul_left_cancel m m n H)", "the hypothesis is Coprime, not 1 < m")
}
