package lean

// Powers are injective on Nat for a nonzero exponent, which `pow_dvd_pow_iff` needs.
func init() {
	pw := func(a, e string) string { return ap("Nat.pow", a, e) }
	sj := "(Nat.succ j)"
	// a < b gives a^(j+1) < b^(j+1)
	P := func(j string) string { return ltN(pw("a", "(Nat.succ "+j+")"), pw("b", "(Nat.succ "+j+")")) }
	bpos := ap("Nat.lt_of_le_of_lt", "0", "a", "b", ap("Nat.zero_le", "a"), "h")
	ss := "(Nat.succ (Nat.succ j))"
	xa := mulN(pw("a", sj), "a")
	ya := mulN(pw("b", sj), "a")
	yb := mulN(pw("b", sj), "b")
	preludeExtra = append(preludeExtra,
		thm("Nat.pow_lt_pow_left_succ", "a b j", [][2]string{{"h", ltN("a", "b")}}, P("j"),
			"Nat.rec.{0} (fun (j : Nat) => "+P("j")+") "+
				tp("a", pw("a", "1"), sym(pw("a", "1"), "a", ap("Nat.pow_one", "a")), func(z string) string { return ltN(z, pw("b", "1")) },
					tp("b", pw("b", "1"), sym(pw("b", "1"), "b", ap("Nat.pow_one", "b")), func(z string) string { return ltN("a", z) }, "h"))+" "+
				"(fun (j : Nat) (ih : "+P("j")+") => "+
				tp(xa, pw("a", ss), sym(pw("a", ss), xa, ap("Nat.pow_succ", "a", sj)), func(z string) string { return ltN(z, pw("b", ss)) },
					tp(yb, pw("b", ss), sym(pw("b", ss), yb, ap("Nat.pow_succ", "b", sj)), func(z string) string { return ltN(xa, z) },
						ap("Nat.lt_of_le_of_lt", xa, ya, yb,
							ap("Nat.mul_le_mul_right", "a", pw("a", sj), pw("b", sj), ap("Nat.le_of_lt", pw("a", sj), pw("b", sj), "ih")),
							ap("Nat.mul_lt_mul_of_pos_left", pw("b", sj), "a", "b", "h", ap("Nat.pos_pow_of_pos", "b", sj, bpos)))))+") j"),
	)
	// pow_left_inj
	eqj := eqN(pw("a", sj), pw("b", sj))
	preludeExtra = append(preludeExtra,
		thm("Nat.pow_left_inj", "a b n", [][2]string{{"hn", notN(eqN("n", "0"))}, {"h", eqN(pw("a", "n"), pw("b", "n"))}}, eqN("a", "b"),
			"(Nat.rec.{0} (fun (n : Nat) => "+notN(eqN("n", "0"))+" → "+eqN(pw("a", "n"), pw("b", "n"))+" → "+eqN("a", "b")+") "+
				"(fun (hn : "+notN(eqN("0", "0"))+") (_ : "+eqN(pw("a", "0"), pw("b", "0"))+") => False.elim.{0} "+eqN("a", "b")+" (hn (Eq.refl.{1} Nat 0))) "+
				"(fun (j : Nat) (_ : "+notN(eqN("j", "0"))+" → "+eqN(pw("a", "j"), pw("b", "j"))+" → "+eqN("a", "b")+") (_ : "+notN(eqN(sj, "0"))+") (h : "+eqj+") => "+
				"Or.rec "+ltN("a", "b")+" "+leN("b", "a")+" (fun (_ : Or "+ltN("a", "b")+" "+leN("b", "a")+") => "+eqN("a", "b")+") "+
				"(fun (hlt : "+ltN("a", "b")+") => False.elim.{0} "+eqN("a", "b")+" (Nat.lt_irrefl "+pw("b", sj)+" "+
				tp(pw("a", sj), pw("b", sj), "h", func(z string) string { return ltN(z, pw("b", sj)) }, ap("Nat.pow_lt_pow_left_succ", "a", "b", "j", "hlt"))+")) "+
				"(fun (hba : "+leN("b", "a")+") => Or.rec "+ltN("b", "a")+" "+leN("a", "b")+" (fun (_ : Or "+ltN("b", "a")+" "+leN("a", "b")+") => "+eqN("a", "b")+") "+
				"(fun (hlt : "+ltN("b", "a")+") => False.elim.{0} "+eqN("a", "b")+" (Nat.lt_irrefl "+pw("a", sj)+" "+
				tp(pw("b", sj), pw("a", sj), sym(pw("a", sj), pw("b", sj), "h"), func(z string) string { return ltN(z, pw("a", sj)) }, ap("Nat.pow_lt_pow_left_succ", "b", "a", "j", "hlt"))+")) "+
				"(fun (hab : "+leN("a", "b")+") => Nat.le_antisymm a b hab hba) "+ap("Nat.lt_or_ge", "b", "a")+") "+
				ap("Nat.lt_or_ge", "a", "b")+") n) hn h"),
	)
	// pow_dvd_pow_iff (a b n), n ≠ 0
	gab := gcdN("a", "b")
	L, R := dvdN(pw("a", "n"), pw("b", "n")), dvdN("a", "b")
	geq := func(x, y string) string { return gcdN(x, y) }
	preludeExtra = append(preludeExtra,
		thm("Nat.pow_dvd_pow_iff", "a b n", [][2]string{{"hn", notN(eqN("n", "0"))}}, iffN(L, R),
			iffI(L, R, "hd",
				iffMp(eqN(gab, "a"), R, ap("Nat.gcd_eq_left_iff_dvd", "a", "b"),
					ap("Nat.pow_left_inj", gab, "a", "n", "hn",
						tr(pw(gab, "n"), geq(pw("a", "n"), pw("b", "n")), pw("a", "n"),
							sym(geq(pw("a", "n"), pw("b", "n")), pw(gab, "n"), ap("Nat.pow_gcd_pow", "a", "b", "n")),
							ap("Nat.gcd_eq_left", pw("a", "n"), pw("b", "n"), "hd")))),
				"hd",
				iffMp(eqN(geq(pw("a", "n"), pw("b", "n")), pw("a", "n")), L, ap("Nat.gcd_eq_left_iff_dvd", pw("a", "n"), pw("b", "n")),
					tr(geq(pw("a", "n"), pw("b", "n")), pw(gab, "n"), pw("a", "n"), ap("Nat.pow_gcd_pow", "a", "b", "n"),
						cg(func(z string) string { return pw(z, "n") }, gab, "a", ap("Nat.gcd_eq_left", "a", "b", "hd")))))),
	)
}
