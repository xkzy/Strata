package lean

import (
	"math/big"
	"testing"
)

// Nat.div and Nat.mod are definitions here (a state machine over the dividend), while the kernel's literal extension
// answers them with big.Int division. If the two disagreed, a claim could be proved and refuted by different paths, so
// the model itself has to be checked: Nat.divMod is not in the extension, so reducing it exercises the definitions.
func TestDivModModelAgreesWithBigIntDivision(t *testing.T) {
	tc := NewTC(mustPrelude(t), nil)
	mk := func(n int64) *Expr { return MkNat(big.NewInt(n)) }
	for divisor := int64(0); divisor <= 9; divisor++ {
		for dividend := int64(0); dividend <= 40; dividend++ {
			state := tc.whnf(MkApp(MkConst("Nat.divMod"), mk(divisor), mk(dividend)))
			if fn := GetAppFn(state); fn.Kind != Const || fn.Name != "Nat.DivMod.mk" {
				t.Fatalf("%d / %d did not reduce to a state: %v", dividend, divisor, state)
			}
			args := GetAppArgs(state)
			if len(args) != 2 {
				t.Fatalf("%d / %d did not reduce to two components: %v", dividend, divisor, state)
			}
			// the machine builds the remainder with `Nat.succ`, so normalise the components before reading them
			q, r := tc.whnf(args[0]), tc.whnf(args[1])
			if !isNatLitExt(q) || !isNatLitExt(r) {
				t.Fatalf("%d / %d did not reduce to two numerals: %v", dividend, divisor, state)
			}
			wantQ, wantR := new(big.Int), new(big.Int).SetInt64(dividend)
			if divisor != 0 {
				wantQ.QuoRem(wantQ.SetInt64(dividend), big.NewInt(divisor), wantR)
			}
			if got := natVal(q); got.Cmp(wantQ) != 0 || natVal(r).Cmp(wantR) != 0 {
				t.Fatalf("%d / %d reduced to quotient %v remainder %v, want %v and %v",
					dividend, divisor, natVal(q), natVal(r), wantQ, wantR)
			}
		}
	}
}
