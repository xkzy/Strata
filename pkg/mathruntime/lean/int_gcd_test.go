package lean

import "testing"

// The ported Int gcd / ediv lemmas have their statements pinned: each is applied at an ascribed type, so a drifted
// statement fails here and not only inside the prelude build.
func TestIntGcdStatements(t *testing.T) {
	env := mustPrelude(t)
	ascribe := func(ty, term string) string {
		return "(fun (m : Int) (n : Int) (k : Int) => (fun (x : " + ty + ") => x) " + term + ")"
	}
	for _, src := range []string{
		ascribe("Eq.{1} Nat (Int.gcd m (Int.add n (Int.mul k m))) (Int.gcd m n)", "(Int.gcd_add_mul_right_right m n k)"),
		ascribe("Eq.{1} Nat (Int.gcd m (Int.sub n m)) (Int.gcd m n)", "(Int.gcd_sub_self_right m n)"),
		ascribe("(Not (Eq.{1} Int m (Int.ofNat 0))) → Eq.{1} Int (Int.ediv (Int.mul m n) m) n", "(Int.mul_ediv_cancel_left m n)"),
		ascribe("(Int.dvd n m) → Eq.{1} Nat (Int.natAbs (Int.ediv m n)) (Nat.div (Int.natAbs m) (Int.natAbs n))", "(Int.natAbs_ediv_of_dvd m n)"),
		ascribe("Iff (Int.dvd k (Int.mul m n)) (Exists.{1} Int (fun (a : Int) => Exists.{1} Int (fun (b : Int) => And (Int.dvd a m) (And (Int.dvd b n) (Eq.{1} Int (Int.mul a b) k)))))", "(Int.dvd_mul m n k)"),
	} {
		wantAccept(t, env, nil, src)
	}
	wantReject(t, env, nil,
		ascribe("Eq.{1} Nat (Int.gcd m (Int.add n (Int.mul k m))) (Int.gcd n k)", "(Int.gcd_add_mul_right_right m n k)"),
		"adding a multiple of m keeps gcd m n, not gcd n k")
	wantReject(t, env, nil,
		ascribe("Eq.{1} Int (Int.ediv (Int.mul m n) m) n", "(Int.mul_ediv_cancel_left m n)"),
		"the divisor must be nonzero: the hypothesis cannot be dropped")
}
