package lean

import "testing"

// Statements of the Div / Mod / Dvd / MinMax lemmas ported after the Gcd family, pinned by ascription.
func TestNatDivModMinMaxStatements(t *testing.T) {
	env := mustPrelude(t)
	ascribe := func(ty, term string) string {
		return "(fun (a : Nat) (b : Nat) (c : Nat) => (fun (x : " + ty + ") => x) " + term + ")"
	}
	for _, src := range []string{
		ascribe("Eq.{1} Nat (Nat.mod a (Nat.mul b c)) (Nat.add (Nat.mod a b) (Nat.mul b (Nat.mod (Nat.div a b) c)))", "(Nat.mod_mul b c a)"),
		ascribe("Eq.{1} Nat (Nat.div (Nat.div a b) c) (Nat.div a (Nat.mul b c))", "(Nat.div_div_eq_div_mul a b c)"),
		ascribe("Eq.{1} Nat (Nat.min a b) (Nat.min b a)", "(Nat.min_comm a b)"),
		ascribe("(Nat.le a b) → Eq.{1} Nat (Nat.max a b) b", "(Nat.max_eq_right a b)"),
		ascribe("Iff (Nat.le (Nat.max a b) c) (And (Nat.le a c) (Nat.le b c))", "(Nat.max_le a b c)"),
	} {
		wantAccept(t, env, nil, src)
	}
	wantReject(t, env, nil, ascribe("Eq.{1} Nat (Nat.min a b) (Nat.max a b)", "(Nat.min_comm a b)"), "min a b is min b a, not max a b")
	wantReject(t, env, nil, ascribe("Eq.{1} Nat (Nat.max a b) b", "(Nat.max_eq_right a b)"), "max_eq_right needs a ≤ b")
}
