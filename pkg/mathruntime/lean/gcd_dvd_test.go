package lean

import (
	"math/big"
	"testing"
)

// Nat.mod_lt is the bound `Nat.gcdIter_dvd` needs: the division machine's remainder is always below a non-zero
// divisor. Every proof term here is the one the prelude holds, applied to numerals, and its type is compared with
// `Nat.lt (n % k) k` and with big.Int's remainder.
func TestModLtBoundsTheRemainder(t *testing.T) {
	env := mustPrelude(t)
	tc := NewTC(env, nil)
	mk := func(n int64) *Expr { return MkNat(big.NewInt(n)) }
	for k := int64(1); k <= 9; k++ {
		// `0 < k` as the kernel has it: `k` is a successor
		hk := MkApp(MkConst("Nat.zero_lt_succ"), mk(k-1))
		for n := int64(0); n <= 40; n++ {
			ty, err := tc.Check(MkApp(MkApp(MkApp(MkConst("Nat.mod_lt"), mk(n)), mk(k)), hk))
			if err != nil {
				t.Fatalf("mod_lt %d %d: %v", n, k, err)
			}
			want := MkApp(MkApp(MkConst("Nat.lt"), MkApp(MkApp(MkConst("Nat.mod"), mk(n)), mk(k))), mk(k))
			if ok, err := tc.IsDefEq(ty, want); err != nil || !ok {
				t.Fatalf("mod_lt %d %d has type %v, want %v (%v)", n, k, ty, want, err)
			}
			rem := new(big.Int).Mod(big.NewInt(n), big.NewInt(k))
			if rem.Cmp(big.NewInt(k)) >= 0 {
				t.Fatalf("%d %% %d = %v, which is not below %d", n, k, rem, k)
			}
			// `Nat.lt` is a definition, so the type is a `Nat.le` once reduced
			if fn := GetAppFn(tc.whnf(ty)); fn.Kind != Const || fn.Name != "Nat.le" {
				t.Fatalf("mod_lt %d %d reduced to %v, want a Nat.le", n, k, ty)
			}
		}
	}
	// 8 % 3 is 2, so the kernel must not accept it as a proof of `2 < 2`
	wantReject(t, env, nil, "(fun (h : Nat.lt (Nat.mod 8 3) 2) => h) (Nat.mod_lt 8 3 (Nat.zero_lt_succ 2))",
		"8 % 3 is 2, not a number below 2")
	// and a divisor of 0 leaves nothing to bound the remainder by
	wantReject(t, env, nil, "Nat.mod_lt 5 0 (Nat.zero_lt_succ 0)", "0 < 0 is false")
}

// Lean's `Nat.gcd_dvd_left`/`Nat.gcd_dvd_right` (`Init/Data/Nat/Gcd.lean`): the gcd divides both arguments. The proofs
// here come from the countdown's invariant, so each pair is compared against the value the kernel computes for
// `Nat.gcd`, and the divisibility decision has to agree.
func TestGcdDividesItsArguments(t *testing.T) {
	env := mustPrelude(t)
	tc := NewTC(env, nil)
	mk := func(n int64) *Expr { return MkNat(big.NewInt(n)) }
	for a := int64(0); a <= 20; a++ {
		for b := int64(0); b <= 20; b++ {
			g := tc.whnf(MkApp(MkApp(MkConst("Nat.gcd"), mk(a)), mk(b)))
			if !isNatLitExt(g) {
				t.Fatalf("gcd(%d, %d) did not reduce to a numeral: %v", a, b, g)
			}
			if want := new(big.Int).GCD(nil, nil, big.NewInt(a), big.NewInt(b)); natVal(g).Cmp(want) != 0 {
				t.Fatalf("gcd(%d, %d) is %v, want %v", a, b, natVal(g), want)
			}
			for _, c := range []struct {
				name string
				arg  int64
			}{{"Nat.gcd_dvd_left", a}, {"Nat.gcd_dvd_right", b}} {
				ty, err := tc.Check(MkApp(MkApp(MkConst(c.name), mk(a)), mk(b)))
				if err != nil {
					t.Fatalf("%s %d %d: %v", c.name, a, b, err)
				}
				want := MkApp(MkApp(MkConst("Nat.dvd"), g), mk(c.arg))
				if ok, err := tc.IsDefEq(ty, want); err != nil || !ok {
					t.Fatalf("%s %d %d has type %v, want %v (%v)", c.name, a, b, ty, want, err)
				}
			}
			// and the pair Lean states as `Nat.gcd_dvd`
			ty, err := tc.Check(MkApp(MkApp(MkConst("Nat.gcd_dvd"), mk(a)), mk(b)))
			if err != nil {
				t.Fatalf("gcd_dvd %d %d: %v", a, b, err)
			}
			pair := MkApp(MkApp(MkConst("And"), MkApp(MkApp(MkConst("Nat.dvd"), g), mk(a))),
				MkApp(MkApp(MkConst("Nat.dvd"), g), mk(b)))
			if ok, err := tc.IsDefEq(ty, pair); err != nil || !ok {
				t.Fatalf("gcd_dvd %d %d has type %v, want %v (%v)", a, b, ty, pair, err)
			}
			// the decision procedure has to see the same divisor: gcd divides its argument, and one more than the
			// argument's value by the gcd does not divide it again (2 * gcd is larger than the argument)
			if !decidesTrue(t, tc, MkApp(MkApp(MkConst("Nat.decDvd"), g), mk(a))) {
				t.Fatalf("Nat.decDvd says gcd(%d, %d) does not divide %d", a, b, a)
			}
			if a > 0 && !decidesFalse(t, tc, MkApp(MkApp(MkConst("Nat.decDvd"), mk(a+1)), mk(a))) {
				t.Fatalf("Nat.decDvd says %d divides %d", a+1, a)
			}
		}
	}
}

// The divisibility algebra `Nat.gcdIter_dvd` rest on, on numerals: `3 ∣ 12` and `3 ∣ 15` give `3 ∣ 60` by
// `dvd_mul_right_of_dvd` and `3 ∣ 27` by `dvd_add`, and the same proofs are refused when the target is off by one.
func TestDivisibilityLawsOnNumerals(t *testing.T) {
	env := mustPrelude(t)
	dvd12 := "(Nat.dvd_of_mod_eq_zero 3 12 (Eq.refl.{1} Nat 0))"
	dvd15 := "(Nat.dvd_of_mod_eq_zero 3 15 (Eq.refl.{1} Nat 0))"
	wantAccept(t, env, nil, "(fun (h : Nat.dvd 3 60) => h) (Nat.dvd_mul_right_of_dvd 3 12 5 "+dvd12+")")
	wantAccept(t, env, nil, "(fun (h : Nat.dvd 3 27) => h) (Nat.dvd_add 3 12 15 "+dvd12+" "+dvd15+")")
	wantReject(t, env, nil, "(fun (h : Nat.dvd 3 61) => h) (Nat.dvd_mul_right_of_dvd 3 12 5 "+dvd12+")",
		"12 * 5 is 60, not 61")
	wantReject(t, env, nil, "(fun (h : Nat.dvd 3 28) => h) (Nat.dvd_add 3 12 15 "+dvd12+" "+dvd15+")",
		"12 + 15 is 27, not 28")
	// the witness is a numeral: `dvd_mul_right_of_dvd` and `dvd_add` build `d * c` and `d + e`, not an opaque value
	if ty, err := checkTerm(t, env, nil, "Nat.dvd_add 3 12 15 "+dvd12+" "+dvd15); err != nil || ty == nil {
		t.Fatalf("the sum of two multiples is not a multiple: %v", err)
	}
}

// `Nat.mod_lt` is what makes a reminder usable with the order lemmas: dropping the successor gives `n % k ≤ k`.
func TestModLtComposesWithTheOrderLemmas(t *testing.T) {
	env := mustPrelude(t)
	wantAccept(t, env, nil,
		"(fun (h : Nat.le (Nat.mod 7 3) 3) => h) (Nat.le_of_succ_le (Nat.mod 7 3) 3 (Nat.mod_lt 7 3 (Nat.zero_lt_succ 2)))")
	wantReject(t, env, nil,
		"(fun (h : Nat.le (Nat.mod 8 3) 1) => h) (Nat.le_of_succ_le (Nat.mod 8 3) 3 (Nat.mod_lt 8 3 (Nat.zero_lt_succ 2)))",
		"8 % 3 is 2")
}

// Lean's `Nat.dvd_gcd`: a common divisor divides the gcd, which is what makes the gcd greatest. The hypotheses are the
// prelude's own `Nat.dvd_of_mod_eq_zero` on numerals, so each case runs through the definitions end to end.
func TestCommonDivisorDividesTheGcd(t *testing.T) {
	env := mustPrelude(t)
	tc := NewTC(env, nil)
	mk := func(n int64) *Expr { return MkNat(big.NewInt(n)) }
	refl0 := mustTerm(t, "Eq.refl.{1} Nat 0", nil)
	dvd := func(k, v int64) *Expr {
		return MkApp(MkApp(MkApp(MkConst("Nat.dvd_of_mod_eq_zero"), mk(k)), mk(v)), refl0)
	}
	for k := int64(1); k <= 6; k++ {
		for m := int64(0); m <= 12; m += k {
			for n := int64(0); n <= 12; n += k {
				g := tc.whnf(MkApp(MkApp(MkConst("Nat.gcd"), mk(m)), mk(n)))
				term := MkApp(MkApp(MkApp(MkApp(MkApp(MkConst("Nat.dvd_gcd"), mk(k)), mk(m)), mk(n)),
					dvd(k, m)), dvd(k, n))
				ty, err := tc.Check(term)
				if err != nil {
					t.Fatalf("dvd_gcd %d %d %d: %v", k, m, n, err)
				}
				want := MkApp(MkApp(MkConst("Nat.dvd"), mk(k)), g)
				if ok, err := tc.IsDefEq(ty, want); err != nil || !ok {
					t.Fatalf("dvd_gcd %d %d %d has type %v, want %v (%v)", k, m, n, ty, want, err)
				}
				if !isNatLitExt(g) || new(big.Int).Mod(natVal(g), big.NewInt(k)).Sign() != 0 {
					t.Fatalf("gcd(%d, %d) = %v is not a multiple of %d", m, n, g, k)
				}
			}
		}
	}
}

// Lean's `Nat.gcd_le_left`/`right`: the gcd is at most each argument, since it divides one and a divisor of a non-zero
// number is no larger than it (`Nat.le_of_dvd`, through `Nat.mul_sub_left_distrib`'s `Nat.le_add_left` case).
func TestGcdIsBoundedByItsArguments(t *testing.T) {
	env := mustPrelude(t)
	tc := NewTC(env, nil)
	mk := func(n int64) *Expr { return MkNat(big.NewInt(n)) }
	for m := int64(1); m <= 20; m++ {
		hm := MkApp(MkConst("Nat.zero_lt_succ"), mk(m-1))
		for n := int64(0); n <= 20; n++ {
			g := tc.whnf(MkApp(MkApp(MkConst("Nat.gcd"), mk(m)), mk(n)))
			if !isNatLitExt(g) || natVal(g).Cmp(big.NewInt(m)) > 0 {
				t.Fatalf("gcd(%d, %d) = %v is larger than %d", m, n, g, m)
			}
			ty, err := tc.Check(MkApp(MkApp(MkApp(MkConst("Nat.gcd_le_left"), mk(m)), mk(n)), hm))
			if err != nil {
				t.Fatalf("gcd_le_left %d %d: %v", m, n, err)
			}
			want := MkApp(MkApp(MkConst("Nat.le"), g), mk(m))
			if ok, err := tc.IsDefEq(ty, want); err != nil || !ok {
				t.Fatalf("gcd_le_left %d %d has type %v, want %v (%v)", m, n, ty, want, err)
			}
			if n == 0 {
				continue
			}
			hn := MkApp(MkConst("Nat.zero_lt_succ"), mk(n-1))
			ty, err = tc.Check(MkApp(MkApp(MkApp(MkConst("Nat.gcd_le_right"), mk(m)), mk(n)), hn))
			if err != nil {
				t.Fatalf("gcd_le_right %d %d: %v", m, n, err)
			}
			want = MkApp(MkApp(MkConst("Nat.le"), g), mk(n))
			if ok, err := tc.IsDefEq(ty, want); err != nil || !ok {
				t.Fatalf("gcd_le_right %d %d has type %v, want %v (%v)", m, n, ty, want, err)
			}
		}
	}
}

// Lean's `Int.gcd_dvd_left`/`right` (`Init/Data/Int/Gcd.lean`): `↑(gcd a b)` divides both `Int` arguments. They go
// through `Int.natAbs_dvd_natAbs`, and the sign of the argument is the only thing the constructor cases have to vary,
// so every combination of a positive, a negative and zero is checked, together with `Int.decDvd` agreeing.
func TestIntGcdDividesBothArguments(t *testing.T) {
	env := mustPrelude(t)
	tc := NewTC(env, nil)
	mk := func(n int64) *Expr { return MkNat(big.NewInt(n)) }
	ofNat := func(n int64) *Expr { return MkApp(MkConst("Int.ofNat"), mk(n)) }
	negSucc := func(n int64) *Expr { return MkApp(MkConst("Int.negSucc"), mk(n)) }
	cases := []struct {
		name string
		val  *Expr
		ab   *big.Int
	}{
		{"0", ofNat(0), big.NewInt(0)},
		{"12", ofNat(12), big.NewInt(12)},
		{"-6", negSucc(5), big.NewInt(-6)},
		{"-1", negSucc(0), big.NewInt(-1)},
	}
	for _, a := range cases {
		for _, b := range cases {
			g := tc.whnf(MkApp(MkApp(MkConst("Int.gcd"), a.val), b.val))
			want := new(big.Int).GCD(nil, nil, a.ab, b.ab)
			if !isNatLitExt(g) || natVal(g).Cmp(want) != 0 {
				t.Fatalf("Int.gcd(%s, %s) = %v, want %v", a.name, b.name, g, want)
			}
			divisor := MkApp(MkConst("Int.ofNat"), g)
			for _, side := range []struct {
				name string
				arg  *Expr
			}{{"Int.gcd_dvd_left", a.val}, {"Int.gcd_dvd_right", b.val}} {
				ty, err := tc.Check(MkApp(MkApp(MkConst(side.name), a.val), b.val))
				if err != nil {
					t.Fatalf("%s %s %s: %v", side.name, a.name, b.name, err)
				}
				expected := MkApp(MkApp(MkConst("Int.dvd"), divisor), side.arg)
				if ok, err := tc.IsDefEq(ty, expected); err != nil || !ok {
					t.Fatalf("%s %s %s has type %v, want %v (%v)", side.name, a.name, b.name, ty, expected, err)
				}
			}
			// the pair form, and the decision procedure, on both arguments
			if _, err := tc.Check(MkApp(MkApp(MkConst("Int.gcd_dvd"), a.val), b.val)); err != nil {
				t.Fatalf("Int.gcd_dvd %s %s: %v", a.name, b.name, err)
			}
			if !decidesTrue(t, tc, MkApp(MkApp(MkConst("Int.decDvd"), divisor), a.val)) {
				t.Fatalf("Int.decDvd says %v does not divide %s", divisor, a.name)
			}
			// a larger divisor is refused when it really does not divide `a` (`gcd + 1` may, e.g. `gcd(12, 0) = 12` and
			// `2 ∣ 12` is not `13 ∣ 12`, so let big.Int pick a value that cannot divide)
			if a.ab.Sign() != 0 {
				bigger := new(big.Int).Add(natVal(g), big.NewInt(1))
				for i := 0; i < 4 && new(big.Int).Mod(a.ab, bigger).Sign() == 0; i++ {
					bigger.Add(bigger, big.NewInt(1))
				}
				if new(big.Int).Mod(a.ab, bigger).Sign() == 0 {
					t.Fatalf("no divisor above gcd %v found for %s", natVal(g), a.name)
				}
				if !decidesFalse(t, tc, MkApp(MkApp(MkConst("Int.decDvd"),
					MkApp(MkConst("Int.ofNat"), MkNat(bigger))), a.val)) {
					t.Fatalf("Int.decDvd says %v divides %s", bigger, a.name)
				}
			}
		}
	}
}

func decidesTrue(t *testing.T, tc *TC, e *Expr) bool {
	t.Helper()
	fn := GetAppFn(tc.whnf(e))
	return fn.Kind == Const && fn.Name == "Decidable.isTrue"
}

func decidesFalse(t *testing.T, tc *TC, e *Expr) bool {
	t.Helper()
	fn := GetAppFn(tc.whnf(e))
	return fn.Kind == Const && fn.Name == "Decidable.isFalse"
}
