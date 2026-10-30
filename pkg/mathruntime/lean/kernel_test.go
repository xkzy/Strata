package lean

import (
	"fmt"
	"math/big"
	"math/rand"
	"strings"
	"testing"
)

func mustPrelude(t *testing.T) *Env {
	t.Helper()
	env, err := Prelude()
	if err != nil {
		t.Fatal(err)
	}
	return env
}

func TestPreludeAxiomSetIsTheDeclaredList(t *testing.T) {
	got := strings.Join(mustPrelude(t).Axioms(), ",")
	want := strings.Join(sortedCopy(PreludeAxioms), ",")
	if got != want {
		t.Fatalf("axioms %s, declared %s", got, want)
	}
}

func TestProveClaim(t *testing.T) {
	for claim, holds := range map[string]bool{
		"2 + 2 = 4": true, "2 + 2 = 5": false, "2^10 = 1024": true, "2^3^2 = 512": true, "(2 + 3) * 4 = 20": true,
		"3 - 5 = 0": true, "7 / 2 = 3": true, "7 / 0 = 0": true, "7 % 0 = 7": true, "10 % 3 = 1": true,
		"3 ∣ 12": true, "3 ∣ 13": false, "0 ∣ 0": true, "0 ∣ 5": false, "¬ 1 = 2": true, "1 ≠ 1": false,
		"1 < 2 ∧ 2 < 3": true, "2 < 2": false, "1 > 2 ∨ 5 ≥ 5": true, "(1 = 2 ∨ 2 = 2) ∧ 3 ≠ 4": true,
		"3 <= 4": true, "5 <= 4": false,
		"∀ x < 10, x * x < 100": true, "∀ x < 10, x * x < 81": false, "∀ x ≤ 9, x * x < 100": true,
		"∃ x < 10, x * x = 49": true, "∃ x < 7, x * x = 49": false, "∃ x ≤ 7, x * x = 49": true,
		"∀ x < 0, 1 = 2": true, "∃ x < 0, 1 = 1": false, "forall a < 5, exists b < 5, a + b = 4": true,
		"∀ x < 20, x % 2 = 0 ∨ x % 2 = 1": true, "∀ x < 5, ∀ y < 5, x + y < 9": true, "∀ x < 5, ∀ y < 5, x + y < 8": false,
		"(∀ x < 4, x < 4) ∧ 1 < 2": true, "∀ x < 30, ¬ (x ∣ 7) ∨ x = 1 ∨ x = 7": true,
		"gcd(12, 18) = 6": true, "gcd(12, 18) = 3": false, "lcm(4, 6) = 12": true, "min(3, 5) = 3": true, "max(3, 5) = 3": false,
		"max(3, 5) = 5": true, "min(5, 5) = 5": true, "fact(5) = 120": true, "factorial(0) = 1": true, "fact(5) = 100": false,
		"∀ x < 8, gcd(x, 6) ∣ 6": true, "∀ x < 6, min(x, 3) ≤ 3 ∧ max(x, 3) ≥ 3": true, "gcd(0, 0) = 0": true,
		"123456789 * 987654321 = 121932631112635269": true, "100 - 1 = 99": true, "2^100 % 7 = 2": true,
		"coprime(8, 15)": true, "coprime(8, 12)": false, "coprime(0, 1)": true, "coprime(0, 2)": false, "¬ coprime(6, 9) ∧ coprime(2 + 3, 7)": true,
		"∀ x < 6, coprime(x, 1)": true,
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
	for _, bad := range []string{"2 +", "x = 1", "1.5 = 1", "1 /\\ 2", "(1 = 1", "gcd(1) = 1", "gcd(1,2,3) = 1", "foo(1) = 1", "gcd 1 2 = 1", "coprime(1)", "coprime(-1, 2)", "∀ x, x = x", "∀ x < y, x = x", "∀ x < 1000, x = x", "∀ x < 256, ∀ y < 256, ∀ z < 256, x = y"} {
		if _, err := ProveClaim(bad, Options{}); err == nil {
			t.Errorf("%q should be unsupported", bad)
		} else if _, ok := err.(ErrUnsupported); !ok {
			t.Errorf("%q: want ErrUnsupported, got %T %v", bad, err, err)
		}
	}
}

// oracle: the kernel agrees with plain big-integer evaluation on random claims, in both directions
func TestKernelMatchesOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	ops := []string{"+", "-", "*", "/", "%", "^"}
	var expr func(d int) (string, *big.Int)
	expr = func(d int) (string, *big.Int) {
		if d == 0 || rng.Intn(3) == 0 {
			n := int64(rng.Intn(30))
			return fmt.Sprint(n), big.NewInt(n)
		}
		a, x := expr(d - 1)
		b, y := expr(d - 1)
		switch op := ops[rng.Intn(len(ops))]; op {
		case "+":
			return "(" + a + "+" + b + ")", new(big.Int).Add(x, y)
		case "-":
			r := new(big.Int).Sub(x, y)
			if r.Sign() < 0 {
				r.SetInt64(0)
			}
			return "(" + a + "-" + b + ")", r
		case "*":
			return "(" + a + "*" + b + ")", new(big.Int).Mul(x, y)
		case "/":
			if y.Sign() == 0 {
				return "(" + a + "/" + b + ")", new(big.Int)
			}
			return "(" + a + "/" + b + ")", new(big.Int).Div(x, y)
		case "^":
			// small exponents only; the exponent is a fresh literal so the oracle never has to raise to a huge power
			e := int64(rng.Intn(5))
			return fmt.Sprintf("(%s^%d)", a, e), new(big.Int).Exp(x, big.NewInt(e), nil)
		default:
			if y.Sign() == 0 {
				return "(" + a + "%" + b + ")", new(big.Int).Set(x)
			}
			return "(" + a + "%" + b + ")", new(big.Int).Mod(x, y)
		}
	}
	rels := []struct {
		s string
		f func(c int) bool
	}{{"=", func(c int) bool { return c == 0 }}, {"≠", func(c int) bool { return c != 0 }}, {"<", func(c int) bool { return c < 0 }},
		{">", func(c int) bool { return c > 0 }}, {"≤", func(c int) bool { return c <= 0 }}, {"≥", func(c int) bool { return c >= 0 }}}
	accepted, rejected := 0, 0
	for i := 0; i < 1500; i++ {
		a, x := expr(3)
		b, y := expr(3)
		if rng.Intn(2) == 0 {
			b, y = a, x // make equalities common
		}
		var claim string
		var want bool
		switch rng.Intn(9) {
		case 0: // divisibility with a small divisor
			k := int64(rng.Intn(6))
			claim = fmt.Sprintf("%d ∣ %s", k, a)
			if k == 0 {
				want = x.Sign() == 0
			} else {
				want = new(big.Int).Mod(x, big.NewInt(k)).Sign() == 0
			}
		case 1: // connectives
			r1, r2 := rels[rng.Intn(len(rels))], rels[rng.Intn(len(rels))]
			claim = fmt.Sprintf("%s %s %s ∧ ¬ (%s %s %s)", a, r1.s, b, a, r2.s, b)
			want = r1.f(x.Cmp(y)) && !r2.f(x.Cmp(y))
		case 2: // disjunction
			r1, r2 := rels[rng.Intn(len(rels))], rels[rng.Intn(len(rels))]
			claim = fmt.Sprintf("(%s %s %s) ∨ (%s %s %s)", a, r1.s, b, b, r2.s, a)
			want = r1.f(x.Cmp(y)) || r2.f(y.Cmp(x))
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
	if accepted < 200 || rejected < 200 {
		t.Fatalf("oracle test is lopsided: %d accepted, %d rejected", accepted, rejected)
	}
}

func TestNatAccelerationAvoidsUnaryRecursion(t *testing.T) {
	// 2^64 by unary Nat.rec would never finish within 1000 steps; the literal extension makes it one big-integer step
	if _, err := ProveClaim("2^64 = 18446744073709551616", Options{MaxSteps: 1000}); err != nil {
		t.Fatal(err)
	}
	if _, err := ProveClaim("1000000 * 1000000 = 1000000000000", Options{MaxSteps: 1000}); err != nil {
		t.Fatal(err)
	}
}

func TestKernelRefusesHugePowers(t *testing.T) {
	_, err := ProveClaim("9^9^9^9 = 1", Options{MaxSteps: 100000})
	if err == nil || !strings.Contains(err.Error(), "refused") {
		t.Fatalf("want the kernel to refuse the power, got %v", err)
	}
}

func TestBudgetIsReportedAsBudget(t *testing.T) {
	_, err := ProveClaim("2 + 2 = 4", Options{MaxSteps: 3})
	if !IsBudgetError(err) {
		t.Fatalf("want a budget error, got %v", err)
	}
}

func TestFalseClaimsComeWithARefutation(t *testing.T) {
	_, err := ProveClaim("2 + 2 = 5", Options{MaxSteps: 100000})
	nf, ok := err.(ErrNotProved)
	if !ok || !nf.Refuted || nf.Proof == nil {
		t.Fatalf("want a kernel-checked refutation, got %#v", err)
	}
	// the certificate really has type Not (2+2=5)
	tc := NewTC(mustPrelude(t), nil)
	ty, cerr := tc.Check(nf.Proof.Term)
	if cerr != nil || !Equal(ty, nf.Proof.Prop) {
		t.Fatalf("refutation does not check: %v", cerr)
	}
}
