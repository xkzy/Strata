package lean

// gcd of powers: gcd (n^k) (m^k) = (gcd n m)^k (Init/Data/Nat/Gcd.lean `pow_gcd_pow`). Lean splits off n = 0; so does
// this, with the zero case a separate lemma since it needs an induction on k of its own.
func init() {
	pw := func(a, e string) string { return ap("Nat.pow", a, e) }
	g := func(a, b string) string { return gcdN(a, b) }
	zeroEq := func(k string) string { return eqN(g(pw("0", k), pw("m", k)), pw(g("0", "m"), k)) }
	preludeExtra = append(preludeExtra,
		thm("Nat.pow_gcd_pow_zero_left", "m k", nil, zeroEq("k"),
			"Nat.rec.{0} (fun (k : Nat) => "+zeroEq("k")+") (Eq.refl.{1} Nat 1) "+
				"(fun (j : Nat) (_ : "+zeroEq("j")+") => "+
				tr(g(pw("0", "(Nat.succ j)"), pw("m", "(Nat.succ j)")), g("0", pw("m", "(Nat.succ j)")), pw(g("0", "m"), "(Nat.succ j)"),
					cg(func(z string) string { return g(z, pw("m", "(Nat.succ j)")) }, pw("0", "(Nat.succ j)"), "0",
						tr(pw("0", "(Nat.succ j)"), mulN(pw("0", "j"), "0"), "0", ap("Nat.pow_succ", "0", "j"), ap("Nat.mul_zero", pw("0", "j")))),
					tr(g("0", pw("m", "(Nat.succ j)")), pw("m", "(Nat.succ j)"), pw(g("0", "m"), "(Nat.succ j)"),
						ap("Nat.gcd_zero_left", pw("m", "(Nat.succ j)")),
						cg(func(z string) string { return pw(z, "(Nat.succ j)") }, "m", g("0", "m"), sym(g("0", "m"), "m", ap("Nat.gcd_zero_left", "m")))))+") k"),
	)
	// n > 0: n = a * g, m = b * g with gcd a b = 1
	G := g("n", "m")
	a, b := "(Nat.div n "+G+")", "(Nat.div m "+G+")"
	ag, bg := mulN(a, G), mulN(b, G)
	hcop := ap("Nat.gcd_div_gcd_div_gcd_of_pos_left", "n", "m", "hn")
	nEq := sym(ag, "n", ap("Nat.div_mul_cancel", G, "n", ap("Nat.gcd_dvd_left", "n", "m")))  // n = a*g
	mEq := sym(bg, "m", ap("Nat.div_mul_cancel", G, "m", ap("Nat.gcd_dvd_right", "n", "m"))) // m = b*g
	ak, bk, gk := pw(a, "k"), pw(b, "k"), pw(G, "k")
	step1 := g(pw(ag, "k"), pw(bg, "k"))
	step2 := g(mulN(ak, gk), mulN(bk, gk))
	preludeExtra = append(preludeExtra,
		thm("Nat.pow_gcd_pow_pos", "n m k", [][2]string{{"hn", ltN("0", "n")}}, eqN(g(pw("n", "k"), pw("m", "k")), gk),
			tr(g(pw("n", "k"), pw("m", "k")), step1, gk,
				tr(g(pw("n", "k"), pw("m", "k")), g(pw(ag, "k"), pw("m", "k")), step1,
					cg(func(z string) string { return g(pw(z, "k"), pw("m", "k")) }, "n", ag, nEq),
					cg(func(z string) string { return g(pw(ag, "k"), pw(z, "k")) }, "m", bg, mEq)),
				tr(step1, step2, gk,
					tr(step1, g(mulN(ak, gk), pw(bg, "k")), step2,
						cg(func(z string) string { return g(z, pw(bg, "k")) }, pw(ag, "k"), mulN(ak, gk), ap("Nat.mul_pow", a, G, "k")),
						cg(func(z string) string { return g(mulN(ak, gk), z) }, pw(bg, "k"), mulN(bk, gk), ap("Nat.mul_pow", b, G, "k"))),
					tr(step2, mulN(g(ak, bk), gk), gk, ap("Nat.gcd_mul_right", ak, gk, bk),
						tr(mulN(g(ak, bk), gk), mulN("1", gk), gk,
							cg(func(z string) string { return mulN(z, gk) }, g(ak, bk), "1",
								ap("Nat.pow_gcd_pow_of_gcd_eq_one", a, b, "k", "k", hcop)),
							ap("Nat.one_mul", gk)))))),
		thm("Nat.pow_gcd_pow", "n m k", nil, eqN(g(pw("n", "k"), pw("m", "k")), pw(g("n", "m"), "k")),
			"Nat.rec.{0} (fun (n : Nat) => "+eqN(g(pw("n", "k"), pw("m", "k")), pw(g("n", "m"), "k"))+") "+
				ap("Nat.pow_gcd_pow_zero_left", "m", "k")+" "+
				"(fun (i : Nat) (_ : "+eqN(g(pw("i", "k"), pw("m", "k")), pw(g("i", "m"), "k"))+") => "+
				ap("Nat.pow_gcd_pow_pos", "(Nat.succ i)", "m", "k", "(Nat.zero_lt_succ i)")+") n"),
	)
}
