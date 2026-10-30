package lean

// Int powers: |n ^ k| = |n| ^ k, and the gcd / lcm / dvd lemmas about powers.
func init() {
	ab := iabs
	pw := func(a, e string) string { return ap("Nat.pow", a, e) }
	ipw := func(a, e string) string { return ap("Int.pow", a, e) }
	idv := func(a, b string) string { return "(Int.dvd " + a + " " + b + ")" }
	iff := func(l, r string) string { return "(Iff " + l + " " + r + ")" }
	// natAbs_pow (n : Int) (k : Nat)
	motive := func(n string) string { return eqN(ab(ipw(n, "k")), pw(ab(n), "k")) }
	S := "(Nat.succ a)"
	P := pw(ab("(Int.negSucc a)"), "k")
	X := "(Int.neg (Int.ofNat " + P + "))"
	Y := "(Int.ofNat " + P + ")"
	bm := func(b string) string {
		return eqN(ab("(Bool.rec.{1} (fun (_ : Bool) => Int) "+X+" "+Y+" "+b+")"), pw(S, "k"))
	}
	preludeExtra = append(preludeExtra,
		ithm("Int.natAbs_pow", "n", [][2]string{{"k", "Nat"}}, motive("n"),
			"Int.rec.{0} (fun (n : Int) => "+motive("n")+") "+
				"(fun (a : Nat) => Eq.refl.{1} Nat "+pw("a", "k")+") "+
				"(fun (a : Nat) => Bool.rec.{0} (fun (b : Bool) => "+bm("b")+") "+
				tr(ab(X), ab("(Int.ofNat "+P+")"), pw(S, "k"), ap("Int.natAbs_neg", "(Int.ofNat "+P+")"), "(Eq.refl.{1} Nat "+pw(S, "k")+")")+" "+
				"(Eq.refl.{1} Nat "+pw(S, "k")+") "+"(Nat.beq (Nat.mod k 2) 0)) n"),
	)
	pa := func(x string) string { return ap("Int.natAbs_pow", x, "k") } // |x^k| = |x|^k
	h1 := [][2]string{{"h", eqN(igcd("n", "m"), "1")}}
	kk := [][2]string{{"k", "Nat"}}
	withK := func(h [][2]string) [][2]string { return append(append([][2]string{}, kk...), h...) }
	preludeExtra = append(preludeExtra,
		ithm("Int.gcd_pow_left_of_gcd_eq_one", "n m", withK(h1), eqN(igcd(ipw("n", "k"), "m"), "1"),
			tp(pw(ab("n"), "k"), ab(ipw("n", "k")), sym(ab(ipw("n", "k")), pw(ab("n"), "k"), pa("n")), func(z string) string { return eqN(gcdN(z, ab("m")), "1") },
				ap("Nat.gcd_pow_left_of_gcd_eq_one", ab("n"), ab("m"), "k", "h"))),
		ithm("Int.gcd_pow_right_of_gcd_eq_one", "n m", withK(h1), eqN(igcd("n", ipw("m", "k")), "1"),
			tp(pw(ab("m"), "k"), ab(ipw("m", "k")), sym(ab(ipw("m", "k")), pw(ab("m"), "k"), pa("m")), func(z string) string { return eqN(gcdN(ab("n"), z), "1") },
				ap("Nat.gcd_pow_right_of_gcd_eq_one", ab("n"), ab("m"), "k", "h"))),
		ithm("Int.pow_gcd_pow_of_gcd_eq_one", "n m", [][2]string{{"k", "Nat"}, {"l", "Nat"}, {"h", eqN(igcd("n", "m"), "1")}}, eqN(igcd(ipw("n", "k"), ipw("m", "l")), "1"),
			tp(pw(ab("n"), "k"), ab(ipw("n", "k")), sym(ab(ipw("n", "k")), pw(ab("n"), "k"), pa("n")), func(z string) string { return eqN(gcdN(z, ab(ipw("m", "l"))), "1") },
				tp(pw(ab("m"), "l"), ab(ipw("m", "l")), sym(ab(ipw("m", "l")), pw(ab("m"), "l"), ap("Int.natAbs_pow", "m", "l")), func(z string) string { return eqN(gcdN(pw(ab("n"), "k"), z), "1") },
					ap("Nat.pow_gcd_pow_of_gcd_eq_one", ab("n"), ab("m"), "k", "l", "h")))),
	)
	both := func(natEq, P func(x, y string) string, body string) string { // P(|n|^k, |m|^k) carried to P(|n^k|, |m^k|)
		_ = natEq
		return tp(pw(ab("n"), "k"), ab(ipw("n", "k")), sym(ab(ipw("n", "k")), pw(ab("n"), "k"), pa("n")), func(z string) string { return P(z, ab(ipw("m", "k"))) },
			tp(pw(ab("m"), "k"), ab(ipw("m", "k")), sym(ab(ipw("m", "k")), pw(ab("m"), "k"), pa("m")), func(z string) string { return P(pw(ab("n"), "k"), z) }, body))
	}
	gp := func(x, y string) string { return eqN(gcdN(x, y), pw(igcd("n", "m"), "k")) }
	lp := func(x, y string) string { return eqN(lcmN(x, y), pw(ilcm("n", "m"), "k")) }
	preludeExtra = append(preludeExtra,
		ithm("Int.pow_gcd_pow", "n m", kk, eqN(igcd(ipw("n", "k"), ipw("m", "k")), pw(igcd("n", "m"), "k")),
			both(nil, gp, ap("Nat.pow_gcd_pow", ab("n"), ab("m"), "k"))),
		ithm("Int.pow_lcm_pow", "n m", kk, eqN(ilcm(ipw("n", "k"), ipw("m", "k")), pw(ilcm("n", "m"), "k")),
			both(nil, lp, ap("Nat.pow_lcm_pow", ab("n"), ab("m"), "k"))),
	)
	// a^n ∣ b^n ↔ a ∣ b for n ≠ 0
	hn := [][2]string{{"n", "Nat"}, {"hn", notN(eqN("n", "0"))}}
	d1 := dvdN(ab(ipw("a", "n")), ab(ipw("b", "n")))
	d2 := dvdN(pw(ab("a"), "n"), pw(ab("b"), "n"))
	d3 := dvdN(ab("a"), ab("b"))
	nat := ap("Nat.pow_dvd_pow_iff", ab("a"), ab("b"), "n", "hn")
	toD := func(x, y, h string) string {
		return iffMpr(dvdN(ab(x), ab(y)), idv(x, y), ap("Int.natAbs_dvd_natAbs", x, y), h)
	}
	frD := func(x, y, h string) string {
		return iffMp(dvdN(ab(x), ab(y)), idv(x, y), ap("Int.natAbs_dvd_natAbs", x, y), h)
	}
	toPow := func(h string) string { // |a^n| ∣ |b^n| -> |a|^n ∣ |b|^n
		return tp(ab(ipw("a", "n")), pw(ab("a"), "n"), ap("Int.natAbs_pow", "a", "n"), func(z string) string { return dvdN(z, pw(ab("b"), "n")) },
			tp(ab(ipw("b", "n")), pw(ab("b"), "n"), ap("Int.natAbs_pow", "b", "n"), func(z string) string { return dvdN(ab(ipw("a", "n")), z) }, h))
	}
	fromPow := func(h string) string {
		return tp(pw(ab("a"), "n"), ab(ipw("a", "n")), sym(ab(ipw("a", "n")), pw(ab("a"), "n"), ap("Int.natAbs_pow", "a", "n")), func(z string) string { return dvdN(z, ab(ipw("b", "n"))) },
			tp(pw(ab("b"), "n"), ab(ipw("b", "n")), sym(ab(ipw("b", "n")), pw(ab("b"), "n"), ap("Int.natAbs_pow", "b", "n")), func(z string) string { return dvdN(pw(ab("a"), "n"), z) }, h))
	}
	_, _ = d1, d2
	preludeExtra = append(preludeExtra,
		ithm("Int.pow_dvd_pow_iff", "a b", hn, iff(idv(ipw("a", "n"), ipw("b", "n")), idv("a", "b")),
			iffI(idv(ipw("a", "n"), ipw("b", "n")), idv("a", "b"), "h",
				frD("a", "b", iffMp(d2, d3, nat, toPow(toD(ipw("a", "n"), ipw("b", "n"), "h")))),
				"h", frD(ipw("a", "n"), ipw("b", "n"), fromPow(iffMpr(d2, d3, nat, toD("a", "b", "h")))))),
	)
}
