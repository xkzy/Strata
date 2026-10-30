package lean

// lcm against a common factor, a quotient, and a power (Init/Data/Nat/Lcm.lean).
func init() {
	no := [][2]string{}
	mc := func(a, b string) string { return ap("Nat.mul_comm", a, b) }
	M := "(Nat.succ i)"
	lhs := func(m string) string { return eqN(lcmN(mulN(m, "n"), mulN(m, "k")), mulN(m, lcmN("n", "k"))) }
	Mn, Mk := mulN(M, "n"), mulN(M, "k")
	L := lcmN(Mn, Mk)
	lnk := lcmN("n", "k")
	posM := "(Nat.zero_lt_succ i)"
	preludeExtra = append(preludeExtra,
		thm("Nat.lcm_mul_left", "m n k", no, lhs("m"),
			"Nat.rec.{0} (fun (m : Nat) => "+lhs("m")+") "+
				tr(lcmN(mulN("0", "n"), mulN("0", "k")), lcmN("0", "0"), mulN("0", lnk),
					tr(lcmN(mulN("0", "n"), mulN("0", "k")), lcmN("0", mulN("0", "k")), lcmN("0", "0"),
						cg(func(z string) string { return lcmN(z, mulN("0", "k")) }, mulN("0", "n"), "0", ap("Nat.zero_mul", "n")),
						cg(func(z string) string { return lcmN("0", z) }, mulN("0", "k"), "0", ap("Nat.zero_mul", "k"))),
					tr(lcmN("0", "0"), "0", mulN("0", lnk), ap("Nat.lcm_zero_left", "0"), sym(mulN("0", lnk), "0", ap("Nat.zero_mul", lnk))))+" "+
				"(fun (i : Nat) (_ : "+lhs("i")+") => "+
				ap("Nat.dvd_antisymm", L, mulN(M, lnk),
					ap("Nat.lcm_dvd", Mn, Mk, mulN(M, lnk),
						ap("Nat.mul_dvd_mul_left", M, "n", lnk, ap("Nat.dvd_lcm_left", "n", "k")),
						ap("Nat.mul_dvd_mul_left", M, "k", lnk, ap("Nat.dvd_lcm_right", "n", "k"))),
					dvIntro(M, L, ap("Nat.dvd_trans", M, Mn, L, ap("Nat.dvd_mul_right", M, "n"), ap("Nat.dvd_lcm_left", Mn, Mk)),
						dvdN(mulN(M, lnk), L), func(c, hl string) string {
							return tp(mulN(M, c), L, sym(L, mulN(M, c), hl), func(z string) string { return dvdN(mulN(M, lnk), z) },
								ap("Nat.mul_dvd_mul_left", M, lnk, c,
									ap("Nat.lcm_dvd", "n", "k", c,
										ap("Nat.dvd_of_mul_dvd_mul_left", M, "n", c, posM,
											tp(L, mulN(M, c), hl, func(z string) string { return dvdN(Mn, z) }, ap("Nat.dvd_lcm_left", Mn, Mk))),
										ap("Nat.dvd_of_mul_dvd_mul_left", M, "k", c, posM,
											tp(L, mulN(M, c), hl, func(z string) string { return dvdN(Mk, z) }, ap("Nat.dvd_lcm_right", Mn, Mk))))))
						}))+") m"),
		thm("Nat.lcm_mul_right", "m n k", no, eqN(lcmN(mulN("m", "n"), mulN("k", "n")), mulN(lcmN("m", "k"), "n")),
			tr(lcmN(mulN("m", "n"), mulN("k", "n")), lcmN(mulN("n", "m"), mulN("n", "k")), mulN(lcmN("m", "k"), "n"),
				tr(lcmN(mulN("m", "n"), mulN("k", "n")), lcmN(mulN("n", "m"), mulN("k", "n")), lcmN(mulN("n", "m"), mulN("n", "k")),
					cg(func(z string) string { return lcmN(z, mulN("k", "n")) }, mulN("m", "n"), mulN("n", "m"), mc("m", "n")),
					cg(func(z string) string { return lcmN(mulN("n", "m"), z) }, mulN("k", "n"), mulN("n", "k"), mc("k", "n"))),
				tr(lcmN(mulN("n", "m"), mulN("n", "k")), mulN("n", lcmN("m", "k")), mulN(lcmN("m", "k"), "n"),
					ap("Nat.lcm_mul_left", "n", "m", "k"), mc("n", lcmN("m", "k"))))),
	)
	// lcm_div
	hk := [][2]string{{"H1", dvdN("k", "m")}, {"H2", dvdN("k", "n")}}
	ld := func(k string) string {
		return eqN(lcmN("(Nat.div m "+k+")", "(Nat.div n "+k+")"), "(Nat.div "+lcmN("m", "n")+" "+k+")")
	}
	K := "(Nat.succ k')"
	mq, nq := "(Nat.div m "+K+")", "(Nat.div n "+K+")"
	lmn := lcmN("m", "n")
	mid := lcmN(mulN(mq, K), mulN(nq, K))
	preludeExtra = append(preludeExtra,
		thm("Nat.lcm_div", "m n k", hk, ld("k"),
			"Nat.rec.{0} (fun (k : Nat) => "+imp(dvdN("k", "m"), imp(dvdN("k", "n"), ld("k")))+") "+
				"(fun (_ : "+dvdN("0", "m")+") (_ : "+dvdN("0", "n")+") => "+
				tr(lcmN("(Nat.div m 0)", "(Nat.div n 0)"), lcmN("0", "(Nat.div n 0)"), "(Nat.div "+lmn+" 0)",
					cg(func(z string) string { return lcmN(z, "(Nat.div n 0)") }, "(Nat.div m 0)", "0", ap("Nat.div_zero", "m")),
					tr(lcmN("0", "(Nat.div n 0)"), "0", "(Nat.div "+lmn+" 0)", ap("Nat.lcm_zero_left", "(Nat.div n 0)"),
						sym("(Nat.div "+lmn+" 0)", "0", ap("Nat.div_zero", lmn))))+") "+
				"(fun (k' : Nat) (_ : "+imp(dvdN("k'", "m"), imp(dvdN("k'", "n"), ld("k'")))+") (H1 : "+dvdN(K, "m")+") (H2 : "+dvdN(K, "n")+") => "+
				ap("Nat.mul_right_cancel", lcmN(mq, nq), K, "(Nat.div "+lmn+" "+K+")", "(Nat.zero_lt_succ k')",
					tr(mulN(lcmN(mq, nq), K), mid, mulN("(Nat.div "+lmn+" "+K+")", K),
						sym(mid, mulN(lcmN(mq, nq), K), ap("Nat.lcm_mul_right", mq, K, nq)),
						tr(mid, lmn, mulN("(Nat.div "+lmn+" "+K+")", K),
							tr(mid, lcmN("m", mulN(nq, K)), lmn,
								cg(func(z string) string { return lcmN(z, mulN(nq, K)) }, mulN(mq, K), "m", ap("Nat.div_mul_cancel", K, "m", "H1")),
								cg(func(z string) string { return lcmN("m", z) }, mulN(nq, K), "n", ap("Nat.div_mul_cancel", K, "n", "H2"))),
							sym(mulN("(Nat.div "+lmn+" "+K+")", K), lmn,
								ap("Nat.div_mul_cancel", K, lmn, ap("Nat.dvd_trans", K, "m", lmn, "H1", ap("Nat.dvd_lcm_left", "m", "n")))))))+
				") k H1 H2"),
	)
	// the `_eq_mul_of_lcm_eq_mul` family
	hl := [][2]string{{"h", eqN(lcmN("n", "m"), mulN("n", "m"))}}
	i1 := "(Nat.succ i)"
	j1 := "(Nat.succ j)"
	gcdOne := func() string { // gcd n m = 1 from h, with n = succ i, m = succ j in scope
		orR := "(Or " + eqN(j1, "0") + " " + eqN(gcdN(i1, j1), "1") + ")"
		return "Or.rec " + eqN(i1, "0") + " " + orR + " (fun (_ : Or " + eqN(i1, "0") + " " + orR + ") => " + eqN(gcdN(i1, j1), "1") + ") " +
			"(fun (e : " + eqN(i1, "0") + ") => False.elim.{0} " + eqN(gcdN(i1, j1), "1") + " (" + ap("Nat.succ_ne_zero", "i") + " e)) " +
			"(fun (r : " + orR + ") => Or.rec " + eqN(j1, "0") + " " + eqN(gcdN(i1, j1), "1") + " (fun (_ : " + orR + ") => " + eqN(gcdN(i1, j1), "1") + ") " +
			"(fun (e : " + eqN(j1, "0") + ") => False.elim.{0} " + eqN(gcdN(i1, j1), "1") + " (" + ap("Nat.succ_ne_zero", "j") + " e)) " +
			"(fun (g : " + eqN(gcdN(i1, j1), "1") + ") => g) r) " +
			iffMp(eqN(lcmN(i1, j1), mulN(i1, j1)), "(Or "+eqN(i1, "0")+" "+orR+")", ap("Nat.lcm_eq_mul_iff", i1, j1), "h")
	}
	rr := func(n, m, k string) string { return eqN(lcmN(n, mulN(m, k)), mulN(lcmN(n, k), m)) }
	// succ case, n = succ i, m = succ j, gcd n m = 1
	succ := func() string {
		n, m := i1, j1
		g := gcdN(n, mulN(m, "k"))
		gk := gcdN(n, "k")
		lhsL := lcmN(n, mulN(m, "k"))
		rhsL := mulN(lcmN(n, "k"), m)
		hg := "(" + gcdOne() + ")"
		e1 := ap("Nat.gcd_mul_right_right_of_gcd_eq_one", n, m, "k", hg) // gcd n (m*k) = gcd n k
		// gk * lhsL = n * (m * k)
		a := tr(mulN(gk, lhsL), mulN(g, lhsL), mulN(n, mulN(m, "k")),
			cg(func(z string) string { return mulN(z, lhsL) }, gk, g, sym(g, gk, e1)), ap("Nat.gcd_mul_lcm", n, mulN(m, "k")))
		// gk * rhsL = n * (m * k)
		b := tr(mulN(gk, rhsL), mulN(mulN(gk, lcmN(n, "k")), m), mulN(n, mulN(m, "k")),
			sym(mulN(mulN(gk, lcmN(n, "k")), m), mulN(gk, rhsL), ap("Nat.mul_assoc", gk, lcmN(n, "k"), m)),
			tr(mulN(mulN(gk, lcmN(n, "k")), m), mulN(mulN(n, "k"), m), mulN(n, mulN(m, "k")),
				cg(func(z string) string { return mulN(z, m) }, mulN(gk, lcmN(n, "k")), mulN(n, "k"), ap("Nat.gcd_mul_lcm", n, "k")),
				tr(mulN(mulN(n, "k"), m), mulN(n, mulN("k", m)), mulN(n, mulN(m, "k")), ap("Nat.mul_assoc", n, "k", m),
					cg(func(z string) string { return mulN(n, z) }, mulN("k", m), mulN(m, "k"), mc("k", m)))))
		return ap("Nat.mul_left_cancel", gk, lhsL, rhsL, ap("Nat.gcd_pos_of_pos_left", n, "k", "(Nat.zero_lt_succ i)"),
			tr(mulN(gk, lhsL), mulN(n, mulN(m, "k")), mulN(gk, rhsL), a, sym(mulN(gk, rhsL), mulN(n, mulN(m, "k")), b)))
	}
	motiveN := func(n string) string { return imp(eqN(lcmN(n, "m"), mulN(n, "m")), rr(n, "m", "k")) }
	motiveM := func(m string) string { return imp(eqN(lcmN(i1, m), mulN(i1, m)), rr(i1, m, "k")) }
	preludeExtra = append(preludeExtra,
		thm("Nat.lcm_mul_right_right_eq_mul_of_lcm_eq_mul", "n m k", hl, rr("n", "m", "k"),
			"Nat.rec.{0} (fun (n : Nat) => "+motiveN("n")+") "+
				"(fun (_ : "+eqN(lcmN("0", "m"), mulN("0", "m"))+") => "+
				tr(lcmN("0", mulN("m", "k")), "0", mulN(lcmN("0", "k"), "m"), ap("Nat.lcm_zero_left", mulN("m", "k")),
					sym(mulN(lcmN("0", "k"), "m"), "0",
						tr(mulN(lcmN("0", "k"), "m"), mulN("0", "m"), "0",
							cg(func(z string) string { return mulN(z, "m") }, lcmN("0", "k"), "0", ap("Nat.lcm_zero_left", "k")), ap("Nat.zero_mul", "m"))))+") "+
				"(fun (i : Nat) (_ : "+motiveN("i")+") => Nat.rec.{0} (fun (m : Nat) => "+motiveM("m")+") "+
				"(fun (_ : "+eqN(lcmN(i1, "0"), mulN(i1, "0"))+") => "+
				tr(lcmN(i1, mulN("0", "k")), "0", mulN(lcmN(i1, "k"), "0"),
					tr(lcmN(i1, mulN("0", "k")), lcmN(i1, "0"), "0", cg(func(z string) string { return lcmN(i1, z) }, mulN("0", "k"), "0", ap("Nat.zero_mul", "k")),
						ap("Nat.lcm_zero_right", i1)),
					sym(mulN(lcmN(i1, "k"), "0"), "0", ap("Nat.mul_zero", lcmN(i1, "k"))))+") "+
				"(fun (j : Nat) (_ : "+motiveM("j")+") (h : "+eqN(lcmN(i1, j1), mulN(i1, j1))+") => "+succ()+") m) n h"),
	)
}
