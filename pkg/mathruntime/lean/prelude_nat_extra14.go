package lean

// Coprime cancellation, quotients and the gcd-product lemma (Init/Data/Nat/Coprime.lean).
func init() {
	cop := func(a, b string) string { return ap("Nat.Coprime", a, b) }
	gc := func(a, b string) string { return ap("Nat.gcd_comm", a, b) }
	mc := func(a, b string) string { return ap("Nat.mul_comm", a, b) }
	hk := [][2]string{{"H", cop("k", "n")}}
	km := mulN("k", "m")
	G := gcdN(km, "n")
	// Coprime (gcd (k*m) n) k, i.e. gcd (gcd (k*m) n) k = 1
	h1 := tr(gcdN(G, "k"), gcdN(km, gcdN("n", "k")), "1", ap("Nat.gcd_assoc", km, "n", "k"),
		tr(gcdN(km, gcdN("n", "k")), gcdN(km, "1"), "1",
			cg(func(z string) string { return gcdN(km, z) }, gcdN("n", "k"), "1", ap("Nat.Coprime.symm", "k", "n", "H")),
			ap("Nat.gcd_one_right", km)))
	preludeExtra = append(preludeExtra,
		thm("Nat.Coprime.gcd_mul_left_cancel", "m k n", hk, eqN(G, gcdN("m", "n")),
			ap("Nat.dvd_antisymm", G, gcdN("m", "n"),
				ap("Nat.dvd_gcd", G, "m", "n",
					ap("Nat.Coprime.dvd_of_dvd_mul_left", G, "k", "m", h1, ap("Nat.gcd_dvd_left", km, "n")),
					ap("Nat.gcd_dvd_right", km, "n")),
				ap("Nat.gcd_dvd_gcd_mul_left_left", "m", "n", "k"))),
		thm("Nat.Coprime.gcd_mul_right_cancel", "m k n", hk, eqN(gcdN(mulN("m", "k"), "n"), gcdN("m", "n")),
			tr(gcdN(mulN("m", "k"), "n"), G, gcdN("m", "n"),
				cg(func(z string) string { return gcdN(z, "n") }, mulN("m", "k"), km, mc("m", "k")),
				ap("Nat.Coprime.gcd_mul_left_cancel", "m", "k", "n", "H"))),
		thm("Nat.Coprime.gcd_mul_left_cancel_right", "n k m", [][2]string{{"H", cop("k", "m")}},
			eqN(gcdN("m", mulN("k", "n")), gcdN("m", "n")),
			tr(gcdN("m", mulN("k", "n")), gcdN(mulN("k", "n"), "m"), gcdN("m", "n"), gc("m", mulN("k", "n")),
				tr(gcdN(mulN("k", "n"), "m"), gcdN("n", "m"), gcdN("m", "n"),
					ap("Nat.Coprime.gcd_mul_left_cancel", "n", "k", "m", "H"), gc("n", "m")))),
		thm("Nat.Coprime.gcd_mul_right_cancel_right", "n k m", [][2]string{{"H", cop("k", "m")}},
			eqN(gcdN("m", mulN("n", "k")), gcdN("m", "n")),
			tr(gcdN("m", mulN("n", "k")), gcdN("m", mulN("k", "n")), gcdN("m", "n"),
				cg(func(z string) string { return gcdN("m", z) }, mulN("n", "k"), mulN("k", "n"), mc("n", "k")),
				ap("Nat.Coprime.gcd_mul_left_cancel_right", "n", "k", "m", "H"))),
	)
	// coprime_div_gcd_div_gcd
	g := gcdN("m", "n")
	dm, dn := "(Nat.div m "+g+")", "(Nat.div n "+g+")"
	preludeExtra = append(preludeExtra,
		thm("Nat.coprime_div_gcd_div_gcd", "m n", [][2]string{{"H", ltN("0", g)}}, cop(dm, dn),
			tr(gcdN(dm, dn), "(Nat.div "+g+" "+g+")", "1",
				ap("Nat.gcd_div", "m", "n", g, ap("Nat.gcd_dvd_left", "m", "n"), ap("Nat.gcd_dvd_right", "m", "n")),
				ap("Nat.div_self", g, "H"))),
		thm("Nat.not_coprime_of_dvd_of_dvd", "d m n", [][2]string{{"dgt1", ltN("1", "d")}, {"Hm", dvdN("d", "m")}, {"Hn", dvdN("d", "n")}},
			notN(cop("m", "n")),
			"fun (co : "+cop("m", "n")+") => "+ap("Nat.not_le_of_gt", "1", "d", "dgt1")+" "+
				ap("Nat.le_of_dvd", "d", "1", "(Nat.zero_lt_succ 0)",
					tp(g, "1", "co", func(z string) string { return dvdN("d", z) }, ap("Nat.dvd_gcd", "d", "m", "n", "Hm", "Hn")))),
	)
	// coprime_div_left / right
	a := "a"
	dl := func(x string) string { return cop("(Nat.div m "+x+")", "n") }
	preludeExtra = append(preludeExtra,
		thm("Nat.Coprime.coprime_div_left", "a m n", [][2]string{{"cmn", cop("m", "n")}, {"hd", dvdN(a, "m")}}, dl(a),
			"Nat.rec.{0} (fun (a : Nat) => "+dvdN("a", "m")+" → "+dl("a")+") "+
				"(fun (hd : "+dvdN("0", "m")+") => "+
				tp("0", "m", sym("m", "0", ap("Nat.eq_zero_of_zero_dvd", "m", "hd")), func(z string) string { return cop("(Nat.div "+z+" 0)", "n") },
					tp("0", "(Nat.div 0 0)", sym("(Nat.div 0 0)", "0", ap("Nat.div_zero", "0")), func(z string) string { return cop(z, "n") },
						tp("m", "0", ap("Nat.eq_zero_of_zero_dvd", "m", "hd"), func(z string) string { return cop(z, "n") }, "cmn")))+") "+
				"(fun (i : Nat) (_ : "+dvdN("i", "m")+" → "+dl("i")+") (hd : "+dvdN("(Nat.succ i)", "m")+") => "+
				dvIntro("(Nat.succ i)", "m", "hd", dl("(Nat.succ i)"), func(l, hl string) string {
					S := "(Nat.succ i)"
					return tp(mulN(S, l), "m", sym("m", mulN(S, l), hl), func(z string) string { return cop("(Nat.div "+z+" "+S+")", "n") },
						tp(l, "(Nat.div "+mulN(S, l)+" "+S+")", sym("(Nat.div "+mulN(S, l)+" "+S+")", l, ap("Nat.mul_div_cancel_left", l, S, "(Nat.zero_lt_succ i)")),
							func(z string) string { return cop(z, "n") },
							ap("Nat.Coprime.coprime_mul_left", l, S, "n",
								tp("m", mulN(S, l), hl, func(z string) string { return cop(z, "n") }, "cmn"))))
				})+") a hd"),
		thm("Nat.Coprime.coprime_div_right", "a m n", [][2]string{{"cmn", cop("m", "n")}, {"hd", dvdN(a, "n")}}, cop("m", "(Nat.div n a)"),
			ap("Nat.Coprime.symm", "(Nat.div n a)", "m",
				ap("Nat.Coprime.coprime_div_left", "a", "n", "m", ap("Nat.Coprime.symm", "m", "n", "cmn"), "hd"))),
	)
	// gcd_mul_gcd_of_coprime_of_mul_eq_mul
	ga, gb := gcdN("a", "c"), gcdN("b", "c")
	P := mulN(ga, gb)
	hh := [][2]string{{"cop", cop("c", "d")}, {"h", eqN(mulN("a", "b"), mulN("c", "d"))}}
	pd := ap("Nat.Coprime.mul_left", ga, gb, "d", ap("Nat.Coprime.gcd_left", "c", "d", "a", "cop"), ap("Nat.Coprime.gcd_left", "c", "d", "b", "cop"))
	cgd := gcdN("c", mulN("c", "d"))
	preludeExtra = append(preludeExtra,
		thm("Nat.gcd_mul_gcd_of_coprime_of_mul_eq_mul", "a b c d", hh, eqN(P, "c"),
			ap("Nat.dvd_antisymm", P, "c",
				ap("Nat.Coprime.dvd_of_dvd_mul_right", P, "d", "c", pd,
					tp(mulN("a", "b"), mulN("c", "d"), "h", func(z string) string { return dvdN(P, z) },
						ap("Nat.mul_dvd_mul", ga, "a", gb, "b", ap("Nat.gcd_dvd_left", "a", "c"), ap("Nat.gcd_dvd_left", "b", "c")))),
				tp(gcdN("c", "a"), ga, gc("c", "a"), func(z string) string { return dvdN("c", mulN(z, gb)) },
					tp(gcdN("c", "b"), gb, gc("c", "b"), func(z string) string { return dvdN("c", mulN(gcdN("c", "a"), z)) },
						ap("Nat.dvd_trans", "c", gcdN("c", mulN("a", "b")), mulN(gcdN("c", "a"), gcdN("c", "b")),
							tp("c", gcdN("c", mulN("a", "b")),
								tr("c", cgd, gcdN("c", mulN("a", "b")), sym(cgd, "c", ap("Nat.gcd_mul_right_right", "d", "c")),
									cg(func(z string) string { return gcdN("c", z) }, mulN("c", "d"), mulN("a", "b"), sym(mulN("a", "b"), mulN("c", "d"), "h"))),
								func(z string) string { return dvdN("c", z) }, ap("Nat.dvd_refl", "c")),
							ap("Nat.gcd_mul_right_dvd_mul_gcd", "c", "a", "b")))))),
	)
}
