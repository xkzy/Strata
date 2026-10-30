package lean

import (
	"fmt"
	"math/big"
	"math/rand"
	"strings"
	"testing"
)

// A claim with a negative numeral is checked over Int (`3 - 5 = -2`) and one without stays over Nat
// (`3 - 5 = 0`). These tests pin both readings, and the oracle below compares the kernel with plain
// big-integer arithmetic on the Int fragment.

func TestIntClaims(t *testing.T) {
	for claim, holds := range map[string]bool{
		"3 - 5 = -2": true, "3 - 5 = 0": true, // the first is Int, the second is Nat (truncated)
		"0 - 7 = -7": true, "-0 = 0": true, "-3 = -3": true, "-3 = 3": false,
		"-3 < 2": true, "-3 < -2": true, "-3 < -3": false, "-3 ≤ -3": true, "2 ≥ -3": true, "-3 > 2": false,
		"-5 + 3 = -2": true, "-5 * -3 = 15": true, "(-2)^3 = -8": true, "(-2)^2 = 4": true, "(-2)^2 = -4": false,
		"-2^3^2 = -512":  true, // a negative numeral is an atom, so this is (-2)^(3^2)
		"-1 < 0 ∧ 0 ≤ 0": true, "¬ (-3 = 3)": true, "¬ (-3 = -3)": false, "-3 ≠ 3": true, "-3 ≠ -3": false, "-123456789 * 987654321 = -121932631112635269": true,
		"-1000000 * 1000000 = -1000000000000": true, "-2^64 = 18446744073709551616": true, // a negative numeral is an atom: even exponent, positive
		"(-2)^65 = -36893488147419103232": true,
		// `/` and `%` are Int.ediv/Int.emod (E-rounding): the remainder is never negative
		"-12 / 7 = -2": true, "-12 / 7 = -1": false, "12 / -7 = -1": true, "-12 / -7 = 2": true,
		"-12 % 7 = 2": true, "12 % -7 = 5": true, "-12 % -7 = 2": true, "7 % -7 = 0": true,
		"12 / 7 = 1 ∧ -1 < 0": true, "12 % 7 = 5 ∧ -1 < 0": true, // the non-negative pair, made an Int claim
		"-7 / 0 = 0": true, "-7 % 0 = -7": true, "-12 / 0 = -12": false,
		// `∣` is Int.dvd, Lean's `Exists (fun c => b = a * c)`
		"-3 ∣ 12": true, "3 ∣ -12": true, "-3 ∣ -12": true, "-3 ∣ 13": false, "2 ∣ -3": false,
		"-5 ∣ 0": true, "0 ∣ -5": false, "-4 ∣ 100": true, "-6 ∣ 100": false, "-3 ∣ 12 ∧ -2 ∣ 12": true, "∀ x < 5, (x - 5) ∣ -5 + 5": true, // every integer divides 0, and `-5` keeps this an Int claim
		"∀ x < 5, (x - 5) ∣ -4 + 5": false,
		// Int.gcd/Int.lcm are Nat-valued in Lean; the claim reads them through Int.ofNat
		"gcd(-6, -9) = 3": true, "gcd(10, -15) = 5": true, "gcd(0, -5) = 5": true, "gcd(-7, 0) = 7": true,
		"gcd(-6, -9) = 2": false, "gcd(-6, -9) ≠ 4": true,
		"lcm(-4, 6) = 12": true, "lcm(9, -6) = 18": true, "lcm(-3, 0) = 0": true, "lcm(-4, 6) = 24": false,
		"∀ x < 8, gcd(x - 8, 6) ∣ 6": true,
		// Int.min/Int.max are `ite (a ≤ b) a b` / `ite (a ≤ b) b a`
		"min(-3, 5) = -3": true, "max(-3, 5) = 5": true, "min(-3, -7) = -7": true, "max(-3, -7) = -3": true,
		"min(-3, 5) = 5": false, "min(3, 5) = 3 ∧ -1 < 0": true, "max(3, 5) = 3 ∧ -1 < 0": false,
		"∀ x < 5, ((x - 5) / 3) * 3 + (x - 5) % 3 = x - 5": true, // x = y * (x / y) + x % y
		"∀ x < 3, x - 1 < 3":                               true,
		"∀ x < 3, x - 5 < -2":                              true,
		"∀ x < 3, x - 5 = -5 + x":                          true,
		"∃ x < 5, x - 3 = 0":                               true,
		"∃ x < 3, x - 3 = -1":                              true,
		"∃ x < 2, x - 3 = -1":                              false,
		"∀ x < 5, ∃ y < 5, x - y = x - y":                  true,
	} {
		p, err := ProveClaim(claim, Options{MaxSteps: 1_000_000})
		if holds && err != nil {
			t.Errorf("%q should be provable: %v", claim, err)
		}
		if !holds && err == nil {
			t.Errorf("%q is false but the kernel accepted %v", claim, p.Term)
		}
		if !holds && err != nil {
			if _, ok := err.(ErrNotProved); !ok {
				t.Errorf("%q: false claim should be ErrNotProved, got %T %v", claim, err, err)
			}
		}
	}
}

// Int operations the Nat fragment has but the Int port does not: rejected with a named reason, not read as Nat.
func TestIntUnsupportedOperations(t *testing.T) {
	for claim, want := range map[string]string{
		"fact(-3) = 1":      "fact is a Nat function",
		"factorial(-4) = 1": "factorial is a Nat function",
	} {
		_, err := ProveClaim(claim, Options{MaxSteps: 1_000_000})
		if err == nil {
			t.Errorf("%q should be unsupported", claim)
			continue
		}
		if !strings.Contains(err.Error(), want) {
			t.Errorf("%q: want %q, got %v", claim, want, err)
		}
		if _, ok := err.(ErrUnsupported); !ok {
			t.Errorf("%q: want ErrUnsupported, got %T", claim, err)
		}
	}
}

func TestIntRefutationComesWithACertificate(t *testing.T) {
	for _, claim := range []string{"-3 = 3", "-3 ∣ 13", "0 ∣ -5", "-12 / 7 = -1"} {
		_, err := ProveClaim(claim, Options{MaxSteps: 1_000_000})
		nf, ok := err.(ErrNotProved)
		if !ok || !nf.Refuted || nf.Proof == nil {
			t.Fatalf("%q: want a kernel-checked refutation, got %#v", claim, err)
		}
		tc := NewTC(mustPrelude(t), nil)
		ty, cerr := tc.Check(nf.Proof.Term)
		if cerr != nil || !Equal(ty, nf.Proof.Prop) {
			t.Fatalf("%q: refutation does not check: %v", claim, cerr)
		}
	}
}

// The proposition really is Lean's: `Int.le`/`Int.lt` are `Int.NonNeg (b - a)` / `(a + 1) ≤ b`, `∣` is `Int.dvd`, and
// the named functions are `Int.gcd`/`Int.lcm` (cast with `Int.ofNat`, since Lean's results are `Nat`) and
// `Int.min`/`Int.max`.
func TestIntClaimPropositions(t *testing.T) {
	for claim, want := range map[string]string{
		"-3 < 2":          "(Int.lt (Int.negSucc 2) (Int.ofNat 2))",
		"-3 ≤ 2":          "(Int.le (Int.negSucc 2) (Int.ofNat 2))",
		"3 - 5 = -2":      "(Eq Int (Int.sub (Int.ofNat 3) (Int.ofNat 5)) (Int.negSucc 1))",
		"-3 ∣ 12":         "(Int.dvd (Int.negSucc 2) (Int.ofNat 12))",
		"gcd(-6, -9) = 3": "(Eq Int (Int.ofNat (Int.gcd (Int.negSucc 5) (Int.negSucc 8))) (Int.ofNat 3))",
		"min(-3, 5) = -3": "(Eq Int (Int.min (Int.negSucc 2) (Int.ofNat 5)) (Int.negSucc 2))",
	} {
		p, err := ProveClaim(claim, Options{MaxSteps: 1_000_000})
		if err != nil {
			t.Fatalf("%q: %v", claim, err)
		}
		if got := p.Prop.String(); got != want {
			t.Errorf("%q: proposition %s, want %s", claim, got, want)
		}
	}
}

// Euclidean division, as Lean's Int.ediv/Int.emod: 0 ≤ x % y < |y|, x / 0 = 0, x % 0 = x.
func intEDiv(x, y *big.Int) *big.Int {
	if y.Sign() == 0 {
		return new(big.Int)
	}
	q, r := new(big.Int).QuoRem(x, y, new(big.Int))
	if r.Sign() < 0 {
		if y.Sign() > 0 {
			q.Sub(q, big.NewInt(1))
		} else {
			q.Add(q, big.NewInt(1))
		}
	}
	return q
}

func intEMod(x, y *big.Int) *big.Int {
	if y.Sign() == 0 {
		return new(big.Int).Set(x)
	}
	_, r := new(big.Int).QuoRem(x, y, new(big.Int))
	if r.Sign() < 0 {
		if y.Sign() > 0 {
			r.Add(r, y)
		} else {
			r.Sub(r, y)
		}
	}
	return r
}

// oracle: the kernel agrees with plain big-integer evaluation on random Int claims, in both directions
func TestIntKernelMatchesOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	ops := []string{"+", "-", "*", "/", "%"}
	var expr func(d int) (string, *big.Int, bool)
	expr = func(d int) (string, *big.Int, bool) {
		if d == 0 || rng.Intn(3) == 0 {
			n := int64(rng.Intn(41) - 20) // negative literals force the Int reading
			return fmt.Sprint(n), big.NewInt(n), n < 0
		}
		a, x, an := expr(d - 1)
		b, y, bn := expr(d - 1)
		switch op := ops[rng.Intn(len(ops))]; op {
		case "+":
			return "(" + a + "+" + b + ")", new(big.Int).Add(x, y), an || bn
		case "-":
			return "(" + a + "-" + b + ")", new(big.Int).Sub(x, y), an || bn
		case "*":
			return "(" + a + "*" + b + ")", new(big.Int).Mul(x, y), an || bn
		case "/":
			return "(" + a + "/" + b + ")", intEDiv(x, y), an || bn
		case "%":
			return "(" + a + "%" + b + ")", intEMod(x, y), an || bn
		default:
			e := int64(rng.Intn(4))
			return fmt.Sprintf("(%s^%d)", a, e), new(big.Int).Exp(x, big.NewInt(e), nil), an || bn
		}
	}
	rels := []struct {
		s string
		f func(c int) bool
	}{{"=", func(c int) bool { return c == 0 }}, {"≠", func(c int) bool { return c != 0 }}, {"<", func(c int) bool { return c < 0 }},
		{">", func(c int) bool { return c > 0 }}, {"≤", func(c int) bool { return c <= 0 }}, {"≥", func(c int) bool { return c >= 0 }}}
	accepted, rejected := 0, 0
	for i := 0; i < 400; i++ {
		a, x, an := expr(3)
		b, y, bn := expr(3)
		if rng.Intn(2) == 0 {
			b, y, bn = a, x, an
		}
		// a claim with no negative numeral would be read over Nat; `+ (-1) * 0` keeps the value and forces Int
		guard := func(s string, hasNeg bool) string {
			if hasNeg {
				return s
			}
			return "(" + s + ") + (-1) * 0"
		}
		a = guard(a, an)
		b = guard(b, bn)
		var claim string
		var want bool
		switch rng.Intn(8) {
		case 3: // the named Int functions: gcd/lcm are Nat-valued in Lean and read through Int.ofNat
			fn := []string{"gcd", "lcm", "min", "max"}[rng.Intn(4)]
			v := new(big.Int)
			switch fn {
			case "gcd":
				v.GCD(nil, nil, new(big.Int).Abs(x), new(big.Int).Abs(y))
			case "lcm":
				if x.Sign() != 0 && y.Sign() != 0 {
					g := new(big.Int).GCD(nil, nil, new(big.Int).Abs(x), new(big.Int).Abs(y))
					v.Abs(new(big.Int).Mul(x, y))
					v.Div(v, g)
				}
			case "min":
				v.Set(x)
				if y.Cmp(x) < 0 {
					v.Set(y)
				}
			default: // max
				v.Set(x)
				if y.Cmp(x) > 0 {
					v.Set(y)
				}
			}
			other := new(big.Int).Set(v)
			if rng.Intn(2) == 0 {
				other.Add(other, big.NewInt(1))
			}
			claim = fmt.Sprintf("%s(%s, %s) = %s", fn, a, b, other.String())
			want = v.Cmp(other) == 0
		case 2: // divisibility: a ∣ b is Int.dvd, so b % a = 0 decides it (and b % 0 = b)
			claim = fmt.Sprintf("%s ∣ %s", a, b)
			want = intEMod(y, x).Sign() == 0
		case 0: // connectives
			r1, r2 := rels[rng.Intn(len(rels))], rels[rng.Intn(len(rels))]
			claim = fmt.Sprintf("%s %s %s ∧ ¬ (%s %s %s)", a, r1.s, b, a, r2.s, b)
			want = r1.f(x.Cmp(y)) && !r2.f(x.Cmp(y))
		case 1: // bounded quantifier over the integers, body in Int
			k := int64(rng.Intn(6))
			c := int64(x.Cmp(y))
			claim = fmt.Sprintf("∀ i < %d, i - %d ≤ %d", k, k, c)
			want = true
			for v := int64(0); v < k; v++ {
				if v-k > c {
					want = false
				}
			}
		default:
			r := rels[rng.Intn(len(rels))]
			claim = fmt.Sprintf("%s %s %s", a, r.s, b)
			want = r.f(x.Cmp(y))
		}
		_, err := ProveClaim(claim, Options{MaxSteps: 2_000_000})
		if want && err != nil {
			t.Fatalf("true claim rejected: %s: %v", claim, err)
		}
		if !want && err == nil {
			t.Fatalf("FALSE claim accepted: %s", claim)
		}
		if err == nil {
			accepted++
		} else {
			rejected++
		}
	}
	if accepted < 50 || rejected < 50 {
		t.Fatalf("oracle test is lopsided: %d accepted, %d rejected", accepted, rejected)
	}
}

// The Int literals go through the same kernel, so a big one must not fall back to unary recursion.
func TestIntLiteralsStayCheap(t *testing.T) {
	for _, claim := range []string{
		"-2^64 = 18446744073709551616",
		"-1000000 * 1000000 = -1000000000000",
		"-123456789 * 987654321 = -121932631112635269",
		"-123456789 / 7 = -17636685",
		"-123456789 % 7 = 6",
	} {
		p, err := ProveClaim(claim, Options{MaxSteps: 1000})
		if err != nil {
			t.Fatalf("%q: %v", claim, err)
		}
		t.Logf("%s: %d steps", claim, p.Steps)
	}
}
