package lean

// exists_coprime / exists_coprime' (Init/Data/Nat/Coprime.lean), with `eq_zero_or_pos` as a helper.
func init() {
	cop := func(a, b string) string { return ap("Nat.Coprime", a, b) }
	g := gcdN("m", "n")
	orZ := "(Or " + eqN("n", "0") + " " + ltN("0", "n") + ")"
	// Exists m' n', Coprime m' n' ∧ m = m' * g ∧ n = n' * g
	body := func(a, b string) string {
		return andN(cop(a, b), andN(eqN("m", mulN(a, g)), eqN("n", mulN(b, g))))
	}
	inner := func(a string) string { return "(fun (b : Nat) => " + body(a, "b") + ")" }
	outer := "(fun (a : Nat) => Exists.{1} Nat " + inner("a") + ")"
	goal := "(Exists.{1} Nat " + outer + ")"
	mk := func(a, b, pc, pm, pn string) string {
		return "(Exists.intro.{1} Nat " + outer + " " + a + " (Exists.intro.{1} Nat " + inner(a) + " " + b + " " +
			ap("And.intro", cop(a, b), andN(eqN("m", mulN(a, g)), eqN("n", mulN(b, g))), pc,
				ap("And.intro", eqN("m", mulN(a, g)), eqN("n", mulN(b, g)), pm, pn)) + "))"
	}
	dm, dn := "(Nat.div m "+g+")", "(Nat.div n "+g+")"
	posCase := mk(dm, dn, ap("Nat.coprime_div_gcd_div_gcd", "m", "n", "hp"),
		sym(mulN(dm, g), "m", ap("Nat.div_mul_cancel", g, "m", ap("Nat.gcd_dvd_left", "m", "n"))),
		sym(mulN(dn, g), "n", ap("Nat.div_mul_cancel", g, "n", ap("Nat.gcd_dvd_right", "m", "n"))))
	z := iffMp(eqN(g, "0"), andN(eqN("m", "0"), eqN("n", "0")), ap("Nat.gcd_eq_zero_iff", "m", "n"), "h0")
	zeroCase := func(x, side string) string { // x = 0 -> x = 1 * g
		return tr(x, "0", mulN("1", g), ap(side, eqN("m", "0"), eqN("n", "0"), z),
			sym(mulN("1", g), "0", tr(mulN("1", g), mulN("1", "0"), "0",
				cg(func(w string) string { return mulN("1", w) }, g, "0", "h0"), ap("Nat.mul_zero", "1"))))
	}
	zeroC := mk("1", "1", ap("Nat.gcd_one_left", "1"), zeroCase("m", "And.left"), zeroCase("n", "And.right"))
	gz := "(Or " + eqN(g, "0") + " " + ltN("0", g) + ")"
	preludeExtra = append(preludeExtra,
		thm("Nat.eq_zero_or_pos", "n", nil, orZ,
			"Nat.rec.{0} (fun (n : Nat) => "+orZ+") "+ap("Or.inl", eqN("0", "0"), ltN("0", "0"), "(Eq.refl.{1} Nat 0)")+
				" (fun (i : Nat) (_ : (Or "+eqN("i", "0")+" "+ltN("0", "i")+")) => "+
				ap("Or.inr", eqN("(Nat.succ i)", "0"), ltN("0", "(Nat.succ i)"), "(Nat.zero_lt_succ i)")+") n"),
		thm("Nat.exists_coprime", "m n", nil, goal,
			"Or.rec "+eqN(g, "0")+" "+ltN("0", g)+" (fun (_ : "+gz+") => "+goal+") "+
				"(fun (h0 : "+eqN(g, "0")+") => "+zeroC+") (fun (hp : "+ltN("0", g)+") => "+posCase+") "+ap("Nat.eq_zero_or_pos", g)),
	)
	// exists_coprime' : 0 < gcd -> ∃ g m' n', 0 < g ∧ Coprime m' n' ∧ m = m' * g ∧ n = n' * g
	b2 := func(gg, a, b string) string {
		return andN(ltN("0", gg), andN(cop(a, b), andN(eqN("m", mulN(a, gg)), eqN("n", mulN(b, gg)))))
	}
	in2 := func(gg, a string) string { return "(fun (b : Nat) => " + b2(gg, a, "b") + ")" }
	mid := func(gg string) string { return "(fun (a : Nat) => Exists.{1} Nat " + in2(gg, "a") + ")" }
	out := "(fun (g : Nat) => Exists.{1} Nat (fun (a : Nat) => Exists.{1} Nat (fun (b : Nat) => " + b2("g", "a", "b") + ")))"
	goal2 := "(Exists.{1} Nat " + out + ")"
	preludeExtra = append(preludeExtra,
		thm("Nat.exists_coprime'", "m n", [][2]string{{"H", ltN("0", g)}}, goal2,
			"Exists.intro.{1} Nat "+out+" "+g+" (Exists.intro.{1} Nat "+mid(g)+" "+dm+" (Exists.intro.{1} Nat "+in2(g, dm)+" "+dn+" "+
				ap("And.intro", ltN("0", g), andN(cop(dm, dn), andN(eqN("m", mulN(dm, g)), eqN("n", mulN(dn, g)))), "H",
					ap("And.intro", cop(dm, dn), andN(eqN("m", mulN(dm, g)), eqN("n", mulN(dn, g))), ap("Nat.coprime_div_gcd_div_gcd", "m", "n", "H"),
						ap("And.intro", eqN("m", mulN(dm, g)), eqN("n", mulN(dn, g)),
							sym(mulN(dm, g), "m", ap("Nat.div_mul_cancel", g, "m", ap("Nat.gcd_dvd_left", "m", "n"))),
							sym(mulN(dn, g), "n", ap("Nat.div_mul_cancel", g, "n", ap("Nat.gcd_dvd_right", "m", "n"))))))+"))"),
	)
}
