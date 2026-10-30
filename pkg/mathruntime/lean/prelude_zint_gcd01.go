package lean

import "strings"

// Int gcd / lcm lemmas (Init/Data/Int/Gcd.lean). `Int.gcd m n` is `Nat.gcd |m| |n|` by delta, so most of these are the Nat
// lemma at the absolute values. Binder order as in prelude_nat_extra.go: explicit Int arguments first, then implicit ones.

func ithm(name, vars string, hyps [][2]string, concl, proof string) valDecl {
	var pi, lam string
	for _, v := range strings.Fields(vars) {
		pi += "(" + v + " : Int) → "
		lam += " (" + v + " : Int)"
	}
	for _, h := range hyps {
		pi += "(" + h[0] + " : " + h[1] + ") → "
		lam += " (" + h[0] + " : " + h[1] + ")"
	}
	return valDecl{"thm", name, nil, pi + concl, "fun" + lam + " => " + proof}
}

func iabs(a string) string       { return "(Int.natAbs " + a + ")" }
func igcd(a, b string) string    { return "(Int.gcd " + a + " " + b + ")" }
func ilcm(a, b string) string    { return "(Int.lcm " + a + " " + b + ")" }
func inz(a string) string        { return notN("(Eq.{1} Int " + a + " (Int.ofNat 0))") }
func nzN(a string) string        { return notN(eqN(a, "0")) }
func ieq(a, b string) string     { return "(Eq.{1} Int " + a + " " + b + ")" }
func imul(a, b string) string    { return "(Int.mul " + a + " " + b + ")" }
func icast(n string) string      { return "(Int.ofNat " + n + ")" }
func iz() string                 { return "(Int.ofNat 0)" }
func io() string                 { return "(Int.ofNat 1)" }
func inatAbsEq0(a string) string { return ap("Int.natAbs_eq_zero", a) }

func init() {
	a, b := "a", "b"
	// 0 < |a| ↔ a ≠ 0
	toNat := func(x, hx string) string { // from x ≠ 0 (Int) to |x| ≠ 0
		return "(fun (e : " + eqN(iabs(x), "0") + ") => " + hx + " " + iffMp(eqN(iabs(x), "0"), ieq(x, iz()), inatAbsEq0(x), "e") + ")"
	}
	pos := func(x, hx string) string { return ap("Nat.pos_of_ne_zero", iabs(x), toNat(x, hx)) }
	preludeExtra = append(preludeExtra,
		ithm("Int.natAbs_pos", "a", nil, "(Iff "+ltN("0", iabs(a))+" "+inz(a)+")",
			iffI(ltN("0", iabs(a)), inz(a), "h", "fun (e : "+ieq(a, iz())+") => "+ap("Nat.lt_irrefl", "0",
				tp(iabs(a), "0", iffMpr(eqN(iabs(a), "0"), ieq(a, iz()), inatAbsEq0(a), "e"), func(z string) string { return ltN("0", z) }, "h"))+"",
				"h", pos(a, "h"))),
	)
	gab := igcd(a, b)
	preludeExtra = append(preludeExtra,
		ithm("Int.gcd_eq_natAbs_gcd_natAbs", "a b", nil, eqN(gab, gcdN(iabs(a), iabs(b))), "Eq.refl.{1} Nat "+gcdN(iabs(a), iabs(b))),
		ithm("Int.gcd_dvd_natAbs_left", "a b", nil, dvdN(gab, iabs(a)), ap("Nat.gcd_dvd_left", iabs(a), iabs(b))),
		ithm("Int.gcd_dvd_natAbs_right", "a b", nil, dvdN(gab, iabs(b)), ap("Nat.gcd_dvd_right", iabs(a), iabs(b))),
		ithm("Int.one_gcd", "a", nil, eqN(igcd(io(), a), "1"), ap("Nat.gcd_one_left", iabs(a))),
		ithm("Int.gcd_one", "a", nil, eqN(igcd(a, io()), "1"), ap("Nat.gcd_one_right", iabs(a))),
		ithm("Int.zero_gcd", "a", nil, eqN(igcd(iz(), a), iabs(a)), ap("Nat.gcd_zero_left", iabs(a))),
		ithm("Int.gcd_zero", "a", nil, eqN(igcd(a, iz()), iabs(a)), ap("Nat.gcd_zero_right", iabs(a))),
		ithm("Int.neg_gcd", "a b", nil, eqN(igcd("(Int.neg a)", b), gab), ap("Int.gcd_neg_left", a, b)),
		ithm("Int.gcd_neg", "a b", nil, eqN(igcd(a, "(Int.neg b)"), gab), ap("Int.gcd_neg_right", a, b)),
		ithm("Int.gcd_assoc", "a b c", nil, eqN(igcd(icast(igcd(a, b)), "c"), igcd(a, icast(igcd("b", "c")))),
			ap("Nat.gcd_assoc", iabs(a), iabs(b), iabs("c"))),
	)
}
