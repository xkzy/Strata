package lean

import (
	"fmt"
	"math/big"
	"testing"
)

// The order and multiplication lemmas `Nat.gcd_comm`/`Nat.gcd_assoc` rest on, exercised the way a caller uses them: each
// applied to numerals and checked against the type it is supposed to have. The `Nat.le` proofs are the prelude's own
// `Nat.le_of_ble_eq_true` (the Bool is computed), and the `Nat.dvd` proofs its `Nat.dvd_of_mod_eq_zero`.
func TestNatOrderAndMultiplicationLemmas(t *testing.T) {
	env := mustPrelude(t)
	le := func(a, b int64) string {
		return fmt.Sprintf("(Nat.le_of_ble_eq_true %d %d (Eq.refl.{1} Bool Bool.true))", a, b)
	}
	dvd := func(k, v int64) string {
		return fmt.Sprintf("(Nat.dvd_of_mod_eq_zero %d %d (Eq.refl.{1} Nat 0))", k, v)
	}
	pos := func(n int64) string { return fmt.Sprintf("(Nat.zero_lt_succ %d)", n-1) }
	for _, c := range []struct {
		name string
		src  string
	}{
		{"Nat.mul_comm", "(fun (h : Eq.{1} Nat (Nat.mul 3 4) (Nat.mul 4 3)) => h) (Nat.mul_comm 3 4)"},
		{"Nat.add_le_add_right", "(fun (h : Nat.le (Nat.add 1 2) (Nat.add 3 2)) => h) (Nat.add_le_add_right 1 3 2 " + le(1, 3) + ")"},
		{"Nat.add_le_add_left", "(fun (h : Nat.le (Nat.add 5 1) (Nat.add 5 3)) => h) (Nat.add_le_add_left 1 3 5 " + le(1, 3) + ")"},
		{"Nat.add_le_add", "(fun (h : Nat.le (Nat.add 1 2) (Nat.add 3 5)) => h) (Nat.add_le_add 1 3 2 5 " + le(1, 3) + " " + le(2, 5) + ")"},
		{"Nat.le_antisymm", "(fun (h : Eq.{1} Nat 3 3) => h) (Nat.le_antisymm 3 3 " + le(3, 3) + " " + le(3, 3) + ")"},
		{"Nat.eq_zero_of_zero_dvd", "(fun (h : Eq.{1} Nat 0 0) => h) (Nat.eq_zero_of_zero_dvd 0 " + dvd(0, 0) + ")"},
		{"Nat.pos_of_dvd_of_pos", "(fun (h : Nat.lt 0 3) => h) (Nat.pos_of_dvd_of_pos 3 12 " + dvd(3, 12) + " " + pos(12) + ")"},
		{"Nat.dvd_antisymm", "(fun (h : Eq.{1} Nat 3 3) => h) (Nat.dvd_antisymm 3 3 " + dvd(3, 3) + " " + dvd(3, 3) + ")"},
		{"Nat.dvd_trans", "(fun (h : Nat.dvd 2 12) => h) (Nat.dvd_trans 2 4 12 " + dvd(2, 4) + " " + dvd(4, 12) + ")"},
	} {
		wantAccept(t, env, nil, c.src)
	}
	// the order has to be the one proved: the reversed inequality is refused
	wantReject(t, env, nil,
		"(fun (h : Nat.le (Nat.add 5 3) (Nat.add 5 1)) => h) (Nat.add_le_add_left 1 3 5 "+le(1, 3)+")",
		"5 + 3 is 8, not at most 5 + 1")
	wantReject(t, env, nil,
		"(fun (h : Nat.le (Nat.add 5 2) (Nat.add 3 5)) => h) (Nat.add_le_add 1 3 2 5 "+le(1, 3)+" "+le(2, 5)+")",
		"1 + 2 is 3, not 7")
	wantReject(t, env, nil,
		"(fun (h : Nat.dvd 2 13) => h) (Nat.dvd_trans 2 4 13 "+dvd(2, 4)+" "+dvd(4, 12)+")",
		"4 does not divide 13")
}

// Lean's `Nat.gcd_comm`: the gcd is symmetric. Both sides are common divisors of `m` and `n`, so `dvd_gcd` plus
// antisymmetry settle it; every pair up to 20 is checked against `big.Int`, and a wrong permutation is refused.
func TestGcdComm(t *testing.T) {
	env := mustPrelude(t)
	tc := NewTC(env, nil)
	mk := func(n int64) *Expr { return MkNat(big.NewInt(n)) }
	for m := int64(0); m <= 20; m++ {
		for n := int64(0); n <= 20; n++ {
			wantAccept(t, env, nil, fmt.Sprintf(
				"(fun (h : Eq.{1} Nat (Nat.gcd %d %d) (Nat.gcd %d %d)) => h) (Nat.gcd_comm %d %d)",
				m, n, n, m, m, n))
			left := tc.whnf(MkApp(MkApp(MkConst("Nat.gcd"), mk(m)), mk(n)))
			right := tc.whnf(MkApp(MkApp(MkConst("Nat.gcd"), mk(n)), mk(m)))
			want := new(big.Int).GCD(nil, nil, big.NewInt(m), big.NewInt(n))
			if !isNatLitExt(left) || natVal(left).Cmp(want) != 0 {
				t.Fatalf("gcd(%d, %d) = %v, want %v", m, n, left, want)
			}
			if !Equal(left, right) {
				t.Fatalf("gcd(%d, %d) = %v but gcd(%d, %d) = %v", m, n, left, n, m, right)
			}
		}
	}
	wantReject(t, env, nil, "(fun (h : Eq.{1} Nat (Nat.gcd 4 6) (Nat.gcd 6 12)) => h) (Nat.gcd_comm 4 6)",
		"gcd(6, 4) is 2, not gcd(6, 12)")
}

// Lean's `Nat.gcd_assoc`: `gcd (gcd m n) k = gcd m (gcd n k)`. Again nothing unfolds the countdown - each side divides
// the other because both divide `m`, `n` and `k` - so every triple up to 8 is checked against `big.Int`.
func TestGcdAssoc(t *testing.T) {
	env := mustPrelude(t)
	tc := NewTC(env, nil)
	mk := func(n int64) *Expr { return MkNat(big.NewInt(n)) }
	for m := int64(0); m <= 8; m++ {
		for n := int64(0); n <= 8; n++ {
			for k := int64(0); k <= 8; k++ {
				wantAccept(t, env, nil, fmt.Sprintf(
					"(fun (h : Eq.{1} Nat (Nat.gcd (Nat.gcd %d %d) %d) (Nat.gcd %d (Nat.gcd %d %d))) => h) "+
						"(Nat.gcd_assoc %d %d %d)", m, n, k, m, n, k, m, n, k))
				left := tc.whnf(MkApp(MkApp(MkConst("Nat.gcd"),
					MkApp(MkApp(MkConst("Nat.gcd"), mk(m)), mk(n))), mk(k)))
				right := tc.whnf(MkApp(MkApp(MkConst("Nat.gcd"), mk(m)),
					MkApp(MkApp(MkConst("Nat.gcd"), mk(n)), mk(k))))
				want := new(big.Int).GCD(nil, nil, new(big.Int).GCD(nil, nil, big.NewInt(m), big.NewInt(n)), big.NewInt(k))
				if !isNatLitExt(left) || natVal(left).Cmp(want) != 0 {
					t.Fatalf("gcd(gcd(%d, %d), %d) = %v, want %v", m, n, k, left, want)
				}
				if !Equal(left, right) {
					t.Fatalf("assoc: %v left, %v right for (%d, %d, %d)", left, right, m, n, k)
				}
			}
		}
	}
	wantReject(t, env, nil,
		"(fun (h : Eq.{1} Nat (Nat.gcd (Nat.gcd 4 6) 8) (Nat.gcd 4 (Nat.gcd 6 7))) => h) (Nat.gcd_assoc 4 6 8)",
		"gcd(gcd(4, 6), 8) is 2, while gcd(4, gcd(6, 7)) is 1")
}

func TestNatCoprimeCoreLemmas(t *testing.T) {
	env := mustPrelude(t)
	for _, tc := range []struct {
		name string
		src  string
	}{
		{"iff_gcd_eq_one", "(fun (h : Iff (Nat.Coprime 3 5) (Eq.{1} Nat (Nat.gcd 3 5) 1)) => h) (Nat.coprime_iff_gcd_eq_one 3 5)"},
		{"gcd_eq_one", "(fun (h : Eq.{1} Nat (Nat.gcd 3 5) 1) => h) (Nat.Coprime.gcd_eq_one 3 5 (Eq.refl.{1} Nat 1))"},
		{"comm", "(fun (h : Iff (Nat.Coprime 3 2) (Nat.Coprime 2 3)) => h) (Nat.coprime_comm 2 3)"},
		{"one_left", "Nat.coprime_one_left 9"},
		{"one_right", "Nat.coprime_one_right 9"},
		{"self", "(fun (h : Iff (Nat.Coprime 2 2) (Eq.{1} Nat 2 1)) => h) (Nat.coprime_self 2)"},
		{"zero_left", "(fun (h : Iff (Nat.Coprime 0 3) (Eq.{1} Nat 3 1)) => h) (Nat.coprime_zero_left 3)"},
		{"zero_right", "(fun (h : Iff (Nat.Coprime 3 0) (Eq.{1} Nat 3 1)) => h) (Nat.coprime_zero_right 3)"},
	} {
		t.Run(tc.name, func(t *testing.T) { wantAccept(t, env, nil, tc.src) })
	}
	wantReject(t, env, nil,
		"(fun (a : Nat) (b : Nat) (h : Nat.Coprime a b) => (fun (x : Nat.Coprime a b) => x) (Nat.Coprime.symm a b h))",
		"symmetry produces coprime b a, not coprime a b")
}

func TestNatCoprimeProductReductionLemmas(t *testing.T) {
	env := mustPrelude(t)
	for _, tc := range []struct {
		name string
		src  string
	}{
		{"dvd_right", "(fun (m : Nat) (n : Nat) (k : Nat) (hd : Nat.dvd n m) (h : Nat.Coprime k m) => Nat.Coprime.coprime_dvd_right k m n hd h)"},
		{"mul_left", "(fun (m : Nat) (k : Nat) (n : Nat) (h : Nat.Coprime (Nat.mul k m) n) => Nat.Coprime.coprime_mul_left m k n h)"},
		{"mul_right", "(fun (m : Nat) (k : Nat) (n : Nat) (h : Nat.Coprime (Nat.mul m k) n) => Nat.Coprime.coprime_mul_right m k n h)"},
		{"mul_left_right", "(fun (m : Nat) (k : Nat) (n : Nat) (h : Nat.Coprime m (Nat.mul k n)) => Nat.Coprime.coprime_mul_left_right m k n h)"},
		{"mul_right_right", "(fun (m : Nat) (n : Nat) (k : Nat) (h : Nat.Coprime m (Nat.mul n k)) => Nat.Coprime.coprime_mul_right_right m n k h)"},
		{"mul_iff_right", "(fun (k : Nat) (m : Nat) (n : Nat) => Nat.coprime_mul_iff_right k m n)"},
		{"gcd_left", "(fun (m : Nat) (n : Nat) (k : Nat) (h : Nat.Coprime m n) => Nat.Coprime.gcd_left m n k h)"},
		{"gcd_right", "(fun (m : Nat) (n : Nat) (k : Nat) (h : Nat.Coprime m n) => Nat.Coprime.gcd_right m n k h)"},
		{"gcd_both", "(fun (m : Nat) (n : Nat) (k : Nat) (l : Nat) (h : Nat.Coprime m n) => Nat.Coprime.gcd_both m n k l h)"},
	} {
		t.Run(tc.name, func(t *testing.T) { wantAccept(t, env, nil, tc.src) })
	}
	wantReject(t, env, nil,
		"(fun (m : Nat) (k : Nat) (n : Nat) (h : Nat.Coprime (Nat.mul m k) n) => (fun (x : Nat.Coprime k n) => x) (Nat.Coprime.coprime_mul_right m k n h))",
		"removing a factor from m * k proves coprime m n, not coprime k n")
}

func TestNatGcdProductDivisibilityLemmas(t *testing.T) {
	env := mustPrelude(t)
	for _, tc := range []struct {
		name string
		src  string
	}{
		{"gcd_divides_gcd_of_dvd_left", "(fun (m : Nat) (k : Nat) (n : Nat) (h : Nat.dvd m k) => Nat.gcd_dvd_gcd_of_dvd_left n m k h)"},
		{"gcd_divides_gcd_of_dvd_right", "(fun (m : Nat) (k : Nat) (n : Nat) (h : Nat.dvd m k) => Nat.gcd_dvd_gcd_of_dvd_right n m k h)"},
		{"gcd_divides_gcd_mul_right_right", "(fun (m : Nat) (n : Nat) (k : Nat) => Nat.gcd_dvd_gcd_mul_right_right m n k)"},
		{"gcd_divides_gcd_mul_left_right", "(fun (m : Nat) (n : Nat) (k : Nat) => Nat.gcd_dvd_gcd_mul_left_right m n k)"},
		{"dvd_mul_factorization", "(fun (k : Nat) (m : Nat) (n : Nat) (h : Nat.dvd k (Nat.mul m n)) => Iff.mp (Nat.dvd k (Nat.mul m n)) (Exists.{1} Nat (fun (a : Nat) => Exists.{1} Nat (fun (b : Nat) => And (And (Nat.dvd a m) (Nat.dvd b n)) (Eq.{1} Nat (Nat.mul a b) k)))) (Nat.dvd_mul k m n) h)"},
		{"gcd_mul_right_divides_product", "(fun (k : Nat) (m : Nat) (n : Nat) => Nat.gcd_mul_right_dvd_mul_gcd k m n)"},
		{"coprime_gcd_mul", "(fun (k : Nat) (m : Nat) (n : Nat) (h : Nat.Coprime m n) => Nat.Coprime.gcd_mul k m n h)"},
		{"coprime_mul_gcd", "(fun (m : Nat) (n : Nat) (k : Nat) (h : Nat.Coprime m n) => Nat.Coprime.mul_gcd m n k h)"},
	} {
		t.Run(tc.name, func(t *testing.T) { wantAccept(t, env, nil, tc.src) })
	}
	wantReject(t, env, nil,
		"(fun (k : Nat) (m : Nat) (n : Nat) => (fun (e : Eq.{1} Nat (Nat.gcd k (Nat.mul m n)) (Nat.mul (Nat.gcd k m) (Nat.gcd k n))) => e) (Nat.Coprime.gcd_mul k m n))",
		"Nat.Coprime.gcd_mul cannot prove the identity without coprimality")
}

func TestNatCoprimePowerAndDivisibilityLemmas(t *testing.T) {
	env := mustPrelude(t)
	for _, tc := range []struct {
		name string
		src  string
	}{
		{"pow_left", "(fun (m : Nat) (k : Nat) (n : Nat) (h : Nat.Coprime m k) => Nat.Coprime.pow_left m k n h)"},
		{"pow_right", "(fun (k : Nat) (m : Nat) (n : Nat) (h : Nat.Coprime k m) => Nat.Coprime.pow_right k m n h)"},
		{"pow", "(fun (k : Nat) (l : Nat) (m : Nat) (n : Nat) (h : Nat.Coprime k l) => Nat.Coprime.pow k l m n h)"},
		{"eq_one_of_dvd", "(fun (k : Nat) (m : Nat) (h : Nat.Coprime k m) (d : Nat.dvd k m) => Nat.Coprime.eq_one_of_dvd k m h d)"},
	} {
		t.Run(tc.name, func(t *testing.T) { wantAccept(t, env, nil, tc.src) })
	}
	wantReject(t, env, nil,
		"(fun (k : Nat) (m : Nat) (h : Nat.Coprime k m) (d : Nat.dvd k m) => (fun (e : Eq.{1} Nat m 1) => e) (Nat.Coprime.eq_one_of_dvd k m h d))",
		"the theorem proves k = 1, not m = 1")
}

func TestNatLcmDivisibilityLemmas(t *testing.T) {
	env := mustPrelude(t)
	for _, src := range []string{
		"(fun (m : Nat) (n : Nat) (k : Nat) => Nat.lcm_dvd_iff m n k)",
		"(fun (n : Nat) (m : Nat) (k : Nat) (h : Nat.dvd m k) => Nat.lcm_dvd_lcm_of_dvd_left n m k h)",
		"(fun (n : Nat) (m : Nat) (k : Nat) (h : Nat.dvd m k) => Nat.lcm_dvd_lcm_of_dvd_right n m k h)",
	} {
		wantAccept(t, env, nil, src)
	}
	wantReject(t, env, nil,
		"(fun (n : Nat) (m : Nat) (k : Nat) (h : Nat.dvd m k) => (fun (x : Nat.dvd (Nat.lcm n m) (Nat.lcm k n)) => x) (Nat.lcm_dvd_lcm_of_dvd_left n m k h))",
		"the left variant bounds lcm m n by lcm k n, not lcm n m by lcm k n")
}

func TestNatLcmZeroAndPositivity(t *testing.T) {
	env := mustPrelude(t)
	for _, src := range []string{
		"(fun (m : Nat) (n : Nat) (hm : Not (Eq.{1} Nat m 0)) (hn : Not (Eq.{1} Nat n 0)) => Nat.lcm_ne_zero m n hm hn)",
		"(fun (m : Nat) (n : Nat) (hm : Nat.lt 0 m) (hn : Nat.lt 0 n) => Nat.lcm_pos m n hm hn)",
		"(fun (m : Nat) (n : Nat) (h : Eq.{1} Nat (Nat.lcm m n) 0) => Nat.eq_zero_of_lcm_eq_zero m n h)",
	} {
		wantAccept(t, env, nil, src)
	}
	wantReject(t, env, nil,
		"(fun (m : Nat) (n : Nat) (hm : Nat.lt 0 m) => (fun (x : Nat.lt 0 (Nat.lcm m n)) => x) (Nat.lcm_pos m n hm))", "positivity of n is needed too, so a partial application is not a proof")
}
