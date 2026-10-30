package lean

import (
	"fmt"
	"math/big"
	"testing"
)

// Nat.gcd is a definition here (Lean's Euclid recursion with the countdown made explicit), while the kernel's Nat
// literal extension answers `Nat.gcd` on two numerals with big.Int's GCD. If the two disagreed, a claim could be
// proved and refuted by different paths, so the model itself is checked: `Nat.gcdIter` is not in the extension, so
// reducing it exercises the recursion, and `Nat.gcd` on literals must agree with it.
func TestGcdModelAgreesWithBigInt(t *testing.T) {
	tc := NewTC(mustPrelude(t), nil)
	mk := func(n int64) *Expr { return MkNat(big.NewInt(n)) }
	for a := int64(0); a <= 30; a++ {
		for b := int64(0); b <= 30; b++ {
			want := new(big.Int).GCD(nil, nil, big.NewInt(a), big.NewInt(b))
			got := tc.whnf(MkApp(MkApp(MkApp(MkConst("Nat.gcdIter"), mk(a)), mk(a)), mk(b)))
			if !isNatLitExt(got) {
				t.Fatalf("gcdIter %d %d %d did not reduce to a numeral: %v", a, a, b, got)
			}
			if natVal(got).Cmp(want) != 0 {
				t.Fatalf("the countdown from %d says gcd(%d, %d) = %v, want %v", a, a, b, natVal(got), want)
			}
			// The definition of Nat.gcd, and the literal extension, must give that same number.
			direct := tc.whnf(MkApp(MkApp(MkConst("Nat.gcd"), mk(a)), mk(b)))
			if !Equal(direct, got) {
				t.Fatalf("gcd(%d, %d): the literal path says %v, the definition %v", a, b, direct, got)
			}
		}
	}
}

// The countdown starts at the first argument because a step maps `(a, b)` to `(b % a, a)`, whose first argument is
// smaller: extra fuel can never change the answer, and that is what makes `Nat.gcd a b = Nat.gcdIter a a b` right.
func TestGcdFuelDoesNotChangeTheAnswer(t *testing.T) {
	tc := NewTC(mustPrelude(t), nil)
	mk := func(n int64) *Expr { return MkNat(big.NewInt(n)) }
	for a := int64(0); a <= 12; a++ {
		for b := int64(0); b <= 12; b++ {
			want := tc.whnf(MkApp(MkApp(MkApp(MkConst("Nat.gcdIter"), mk(a)), mk(a)), mk(b)))
			for extra := int64(1); extra <= 5; extra++ {
				got := tc.whnf(MkApp(MkApp(MkApp(MkConst("Nat.gcdIter"), mk(a+extra)), mk(a)), mk(b)))
				if !Equal(got, want) {
					t.Fatalf("gcdIter with %d steps of %d fuel says %v, with %d fuel %v", a, a+extra, got, a, want)
				}
			}
		}
	}
}

// The equations Lean's equation compiler gives `Nat.gcd`, exercised through reduction: the zero cases on both sides,
// `gcd 1 n`, and the step, which is Lean's `gcd (succ a) b = gcd (b % succ a) (succ a)` with the countdown explicit.
func TestGcdEquations(t *testing.T) {
	env := mustPrelude(t)
	tc := NewTC(env, nil)
	mk := func(n int64) *Expr { return MkNat(big.NewInt(n)) }
	for _, c := range []struct {
		name string
		term *Expr
		want int64
	}{
		{"gcd_zero_left", MkApp(MkApp(MkConst("Nat.gcd"), mk(0)), mk(5)), 5},
		{"gcd_zero_right", MkApp(MkApp(MkConst("Nat.gcd"), mk(7)), mk(0)), 7},
		{"gcd_one_left", MkApp(MkApp(MkConst("Nat.gcd"), mk(1)), mk(9)), 1},
		{"gcd_self_value", MkApp(MkApp(MkConst("Nat.gcd"), mk(12)), mk(12)), 12},
	} {
		got := tc.whnf(c.term)
		if !isNatLitExt(got) || natVal(got).Int64() != c.want {
			t.Errorf("%s reduced to %v, want %d", c.name, got, c.want)
		}
	}
	// The step: `Nat.gcd (succ a) b` and `Nat.gcdIter a (b % succ a) (succ a)` are the same term up to reduction.
	for a := int64(0); a <= 8; a++ {
		for b := int64(0); b <= 8; b++ {
			lhs := tc.whnf(MkApp(MkApp(MkConst("Nat.gcd"), mk(a+1)), mk(b)))
			rhs := tc.whnf(MkApp(MkApp(MkApp(MkConst("Nat.gcdIter"), mk(a)),
				MkApp(MkApp(MkConst("Nat.mod"), mk(b)), mk(a+1))), mk(a+1)))
			if !Equal(lhs, rhs) {
				t.Fatalf("gcd(%d, %d) reduced to %v but the step gave %v", a+1, b, lhs, rhs)
			}
		}
	}
}

// The zero cases and lcm, which is built on gcd: Lean's `gcd m 0 = m` (not 0) and `lcm m 0 = 0` are the conventions
// a claim gets, and `lcm` divides by the gcd, so a wrong gcd shows up as a wrong lcm too.
func TestGcdAndLcmEdgeCaseClaims(t *testing.T) {
	for claim, holds := range map[string]bool{
		"gcd(0, 5) = 5": true, "gcd(7, 0) = 7": true, "gcd(0, 0) = 0": true, "gcd(4, 6) = 2": true,
		"lcm(4, 6) = 12": true, "lcm(0, 5) = 0": true, "lcm(5, 0) = 0": true, "lcm(7, 7) = 7": true,
		"gcd(4, 6) = 4": false, "lcm(4, 6) = 24": false,
	} {
		_, err := ProveClaim(claim, Options{MaxSteps: 1_000_000})
		if holds && err != nil {
			t.Errorf("%q should be provable: %v", claim, err)
		}
		if !holds && err == nil {
			t.Errorf("%q is false but the kernel accepted it", claim)
		}
	}
}

// A claim's gcd goes through Nat.gcd, so the endpoint's verdict has to match big.Int on large literals too, where the
// number is far past the countdown's reach (the literal extension answers it in one step).
func TestLargeGcdClaimsMatchBigInt(t *testing.T) {
	for _, c := range [][2]int64{{123456789, 987654321}, {1 << 40, 3 << 30}, {999999937, 999999929}} {
		a, b := big.NewInt(c[0]), big.NewInt(c[1])
		g := new(big.Int).GCD(nil, nil, a, b)
		claim := fmt.Sprintf("gcd(%d, %d) = %s", c[0], c[1], g)
		if _, err := ProveClaim(claim, Options{MaxSteps: 1_000_000}); err != nil {
			t.Errorf("%q: %v", claim, err)
		}
		wrong := fmt.Sprintf("gcd(%d, %d) = %s", c[0], c[1], new(big.Int).Add(g, big.NewInt(1)))
		if _, err := ProveClaim(wrong, Options{MaxSteps: 1_000_000}); err == nil {
			t.Errorf("%q is false but the kernel accepted it", wrong)
		}
	}
}
