package lean

// lcm and positivity, zero, products and ones (Init/Data/Nat/Lcm.lean).
func init() {
	no := [][2]string{}
	lc := func(a, b string) string { return ap("Nat.lcm_comm", a, b) }
	pos := func(x string) string { return ltN("0", x) }
	preludeExtra = append(preludeExtra,
		thm("Nat.le_lcm_left", "m n", [][2]string{{"hn", pos("n")}}, leN("m", lcmN("m", "n")),
			"Nat.rec.{0} (fun (m : Nat) => "+leN("m", lcmN("m", "n"))+") "+ap("Nat.zero_le", lcmN("0", "n"))+" "+
				"(fun (i : Nat) (_ : "+leN("i", lcmN("i", "n"))+") => "+
				ap("Nat.le_of_dvd", "(Nat.succ i)", lcmN("(Nat.succ i)", "n"), ap("Nat.lcm_pos", "(Nat.succ i)", "n", "(Nat.zero_lt_succ i)", "hn"),
					ap("Nat.dvd_lcm_left", "(Nat.succ i)", "n"))+") m"),
		thm("Nat.le_lcm_right", "m n", [][2]string{{"hm", pos("m")}}, leN("n", lcmN("m", "n")),
			"Nat.rec.{0} (fun (n : Nat) => "+leN("n", lcmN("m", "n"))+") "+ap("Nat.zero_le", lcmN("m", "0"))+" "+
				"(fun (i : Nat) (_ : "+leN("i", lcmN("m", "i"))+") => "+
				ap("Nat.le_of_dvd", "(Nat.succ i)", lcmN("m", "(Nat.succ i)"), ap("Nat.lcm_pos", "m", "(Nat.succ i)", "hm", "(Nat.zero_lt_succ i)"),
					ap("Nat.dvd_lcm_right", "m", "(Nat.succ i)"))+") n"),
		thm("Nat.lcm_eq_zero_iff", "m n", no, iffN(eqN(lcmN("m", "n"), "0"), "(Or "+eqN("m", "0")+" "+eqN("n", "0")+")"),
			iffI(eqN(lcmN("m", "n"), "0"), "(Or "+eqN("m", "0")+" "+eqN("n", "0")+")", "h", ap("Nat.eq_zero_of_lcm_eq_zero", "m", "n", "h"),
				"h",
				"Or.rec "+eqN("m", "0")+" "+eqN("n", "0")+" (fun (_ : Or "+eqN("m", "0")+" "+eqN("n", "0")+") => "+eqN(lcmN("m", "n"), "0")+") "+
					"(fun (hm : "+eqN("m", "0")+") => "+tp("0", "m", sym("m", "0", "hm"), func(z string) string { return eqN(lcmN(z, "n"), "0") }, ap("Nat.lcm_zero_left", "n"))+") "+
					"(fun (hn : "+eqN("n", "0")+") => "+tp("0", "n", sym("n", "0", "hn"), func(z string) string { return eqN(lcmN("m", z), "0") }, ap("Nat.lcm_zero_right", "m"))+") h")),
		thm("Nat.pos_of_lcm_pos_left", "m n", no, imp(pos(lcmN("m", "n")), pos("m")),
			"Nat.rec.{0} (fun (m : Nat) => "+imp(pos(lcmN("m", "n")), pos("m"))+") "+
				"(fun (h : "+pos(lcmN("0", "n"))+") => False.elim.{0} "+pos("0")+" ("+ap("Nat.lt_irrefl", "0")+" "+
				tp(lcmN("0", "n"), "0", ap("Nat.lcm_zero_left", "n"), func(z string) string { return pos(z) }, "h")+")) "+
				"(fun (i : Nat) (_ : "+imp(pos(lcmN("i", "n")), pos("i"))+") (_ : "+pos(lcmN("(Nat.succ i)", "n"))+") => Nat.zero_lt_succ i) m"),
		thm("Nat.pos_of_lcm_pos_right", "m n", no, imp(pos(lcmN("m", "n")), pos("n")),
			"Nat.rec.{0} (fun (n : Nat) => "+imp(pos(lcmN("m", "n")), pos("n"))+") "+
				"(fun (h : "+pos(lcmN("m", "0"))+") => False.elim.{0} "+pos("0")+" ("+ap("Nat.lt_irrefl", "0")+" "+
				tp(lcmN("m", "0"), "0", ap("Nat.lcm_zero_right", "m"), func(z string) string { return pos(z) }, "h")+")) "+
				"(fun (i : Nat) (_ : "+imp(pos(lcmN("m", "i")), pos("i"))+") (_ : "+pos(lcmN("m", "(Nat.succ i)"))+") => Nat.zero_lt_succ i) n"),
		thm("Nat.lcm_pos_iff", "m n", no, iffN(pos(lcmN("m", "n")), andN(pos("m"), pos("n"))),
			iffI(pos(lcmN("m", "n")), andN(pos("m"), pos("n")), "h",
				ap("And.intro", pos("m"), pos("n"), ap("Nat.pos_of_lcm_pos_left", "m", "n", "h"), ap("Nat.pos_of_lcm_pos_right", "m", "n", "h")),
				"h",
				ap("Nat.lcm_pos", "m", "n", ap("And.left", pos("m"), pos("n"), "h"), ap("And.right", pos("m"), pos("n"), "h")))),
	)
	// lcm against a product
	preludeExtra = append(preludeExtra,
		thm("Nat.lcm_mul_right_dvd_mul_lcm", "k m n", no, dvdN(lcmN("k", mulN("m", "n")), mulN(lcmN("k", "m"), lcmN("k", "n"))),
			ap("Nat.lcm_dvd", "k", mulN("m", "n"), mulN(lcmN("k", "m"), lcmN("k", "n")),
				ap("Nat.dvd_trans", "k", lcmN("k", "m"), mulN(lcmN("k", "m"), lcmN("k", "n")), ap("Nat.dvd_lcm_left", "k", "m"),
					ap("Nat.dvd_mul_right", lcmN("k", "m"), lcmN("k", "n"))),
				ap("Nat.mul_dvd_mul", "m", lcmN("k", "m"), "n", lcmN("k", "n"), ap("Nat.dvd_lcm_right", "k", "m"), ap("Nat.dvd_lcm_right", "k", "n")))),
		thm("Nat.lcm_mul_left_dvd_mul_lcm", "k m n", no, dvdN(lcmN(mulN("m", "n"), "k"), mulN(lcmN("m", "k"), lcmN("n", "k"))),
			tp(lcmN("k", mulN("m", "n")), lcmN(mulN("m", "n"), "k"), lc("k", mulN("m", "n")),
				func(z string) string { return dvdN(z, mulN(lcmN("m", "k"), lcmN("n", "k"))) },
				tp(mulN(lcmN("k", "m"), lcmN("k", "n")), mulN(lcmN("m", "k"), lcmN("n", "k")),
					tr(mulN(lcmN("k", "m"), lcmN("k", "n")), mulN(lcmN("m", "k"), lcmN("k", "n")), mulN(lcmN("m", "k"), lcmN("n", "k")),
						cg(func(z string) string { return mulN(z, lcmN("k", "n")) }, lcmN("k", "m"), lcmN("m", "k"), lc("k", "m")),
						cg(func(z string) string { return mulN(lcmN("m", "k"), z) }, lcmN("k", "n"), lcmN("n", "k"), lc("k", "n"))),
					func(z string) string { return dvdN(lcmN("k", mulN("m", "n")), z) },
					ap("Nat.lcm_mul_right_dvd_mul_lcm", "k", "m", "n")))),
		thm("Nat.lcm_dvd_mul_self_left_iff_dvd_mul", "k n m", no,
			iffN(dvdN(lcmN("k", "n"), mulN("k", "m")), dvdN("n", mulN("k", "m"))),
			iffI(dvdN(lcmN("k", "n"), mulN("k", "m")), dvdN("n", mulN("k", "m")), "h",
				ap("Nat.dvd_trans", "n", lcmN("k", "n"), mulN("k", "m"), ap("Nat.dvd_lcm_right", "k", "n"), "h"),
				"h", ap("Nat.lcm_dvd", "k", "n", mulN("k", "m"), ap("Nat.dvd_mul_right", "k", "m"), "h"))),
		thm("Nat.lcm_dvd_mul_self_right_iff_dvd_mul", "k m n", no,
			iffN(dvdN(lcmN("n", "k"), mulN("m", "k")), dvdN("n", mulN("m", "k"))),
			iffI(dvdN(lcmN("n", "k"), mulN("m", "k")), dvdN("n", mulN("m", "k")), "h",
				ap("Nat.dvd_trans", "n", lcmN("n", "k"), mulN("m", "k"), ap("Nat.dvd_lcm_left", "n", "k"), "h"),
				"h", ap("Nat.lcm_dvd", "n", "k", mulN("m", "k"), "h", ap("Nat.dvd_mul_left", "k", "m")))),
	)
	// lcm m n = m * n iff m = 0, n = 0 or gcd m n = 1
	gmn := gcdN("m", "n")
	orR := "(Or " + eqN("n", "0") + " " + eqN(gmn, "1") + ")"
	orA := "(Or " + eqN("m", "0") + " " + orR + ")"
	lmn := eqN(lcmN("m", "n"), mulN("m", "n"))
	succCase := func(i, j string) string { // m = succ i, n = succ j, lcm = m*n: gcd = 1
		m, n := "(Nat.succ "+i+")", "(Nat.succ "+j+")"
		mn := mulN(m, n)
		g := gcdN(m, n)
		return "Or.inr " + eqN(m, "0") + " (Or " + eqN(n, "0") + " " + eqN(g, "1") + ") (Or.inr " + eqN(n, "0") + " " + eqN(g, "1") + " " +
			ap("Nat.mul_right_cancel", g, mn, "1", ap("Nat.mul_pos", m, n, "(Nat.zero_lt_succ "+i+")", "(Nat.zero_lt_succ "+j+")"),
				tr(mulN(g, mn), mulN(g, lcmN(m, n)), mulN("1", mn),
					cg(func(z string) string { return mulN(g, z) }, mn, lcmN(m, n), sym(lcmN(m, n), mn, "h")),
					tr(mulN(g, lcmN(m, n)), mn, mulN("1", mn), ap("Nat.gcd_mul_lcm", m, n), sym(mulN("1", mn), mn, ap("Nat.one_mul", mn))))) + ")"
	}
	_ = succCase
	preludeExtra = append(preludeExtra,
		thm("Nat.lcm_eq_mul_iff", "m n", no, iffN(lmn, orA),
			iffI(lmn, orA, "h",
				"Nat.rec.{0} (fun (m : Nat) => "+imp(eqN(lcmN("m", "n"), mulN("m", "n")), "(Or "+eqN("m", "0")+" (Or "+eqN("n", "0")+" "+eqN(gcdN("m", "n"), "1")+"))")+") "+
					"(fun (_ : "+eqN(lcmN("0", "n"), mulN("0", "n"))+") => Or.inl "+eqN("0", "0")+" (Or "+eqN("n", "0")+" "+eqN(gcdN("0", "n"), "1")+") (Eq.refl.{1} Nat 0)) "+
					"(fun (i : Nat) (_ : "+imp(eqN(lcmN("i", "n"), mulN("i", "n")), "(Or "+eqN("i", "0")+" (Or "+eqN("n", "0")+" "+eqN(gcdN("i", "n"), "1")+"))")+") => "+
					"Nat.rec.{0} (fun (n : Nat) => "+imp(eqN(lcmN("(Nat.succ i)", "n"), mulN("(Nat.succ i)", "n")), "(Or "+eqN("(Nat.succ i)", "0")+" (Or "+eqN("n", "0")+" "+eqN(gcdN("(Nat.succ i)", "n"), "1")+"))")+") "+
					"(fun (_ : "+eqN(lcmN("(Nat.succ i)", "0"), mulN("(Nat.succ i)", "0"))+") => Or.inr "+eqN("(Nat.succ i)", "0")+" (Or "+eqN("0", "0")+" "+eqN(gcdN("(Nat.succ i)", "0"), "1")+") "+
					"(Or.inl "+eqN("0", "0")+" "+eqN(gcdN("(Nat.succ i)", "0"), "1")+" (Eq.refl.{1} Nat 0))) "+
					"(fun (j : Nat) (_ : "+imp(eqN(lcmN("(Nat.succ i)", "j"), mulN("(Nat.succ i)", "j")), "(Or "+eqN("(Nat.succ i)", "0")+" (Or "+eqN("j", "0")+" "+eqN(gcdN("(Nat.succ i)", "j"), "1")+"))")+") (h : "+eqN(lcmN("(Nat.succ i)", "(Nat.succ j)"), mulN("(Nat.succ i)", "(Nat.succ j)"))+") => "+
					succCase("i", "j")+") n) m h",
				"h",
				"Or.rec "+eqN("m", "0")+" "+orR+" (fun (_ : "+orA+") => "+lmn+") "+
					"(fun (hm : "+eqN("m", "0")+") => "+tp("0", "m", sym("m", "0", "hm"), func(z string) string { return eqN(lcmN(z, "n"), mulN(z, "n")) },
					tr(lcmN("0", "n"), "0", mulN("0", "n"), ap("Nat.lcm_zero_left", "n"), sym(mulN("0", "n"), "0", ap("Nat.zero_mul", "n"))))+") "+
					"(fun (h2 : "+orR+") => Or.rec "+eqN("n", "0")+" "+eqN(gmn, "1")+" (fun (_ : "+orR+") => "+lmn+") "+
					"(fun (hn : "+eqN("n", "0")+") => "+tp("0", "n", sym("n", "0", "hn"), func(z string) string { return eqN(lcmN("m", z), mulN("m", z)) },
					tr(lcmN("m", "0"), "0", mulN("m", "0"), ap("Nat.lcm_zero_right", "m"), sym(mulN("m", "0"), "0", ap("Nat.mul_zero", "m"))))+") "+
					"(fun (hg : "+eqN(gmn, "1")+") => "+tr(lcmN("m", "n"), mulN("1", lcmN("m", "n")), mulN("m", "n"),
					sym(mulN("1", lcmN("m", "n")), lcmN("m", "n"), ap("Nat.one_mul", lcmN("m", "n"))),
					tr(mulN("1", lcmN("m", "n")), mulN(gmn, lcmN("m", "n")), mulN("m", "n"),
						cg(func(z string) string { return mulN(z, lcmN("m", "n")) }, "1", gmn, sym(gmn, "1", "hg")), ap("Nat.gcd_mul_lcm", "m", "n")))+") h2) h")),
		thm("Nat.lcm_eq_one_iff", "m n", no, iffN(eqN(lcmN("m", "n"), "1"), andN(eqN("m", "1"), eqN("n", "1"))),
			iffI(eqN(lcmN("m", "n"), "1"), andN(eqN("m", "1"), eqN("n", "1")), "h",
				ap("And.intro", eqN("m", "1"), eqN("n", "1"),
					ap("Nat.eq_one_of_dvd_one", "m", tp(lcmN("m", "n"), "1", "h", func(z string) string { return dvdN("m", z) }, ap("Nat.dvd_lcm_left", "m", "n"))),
					ap("Nat.eq_one_of_dvd_one", "n", tp(lcmN("m", "n"), "1", "h", func(z string) string { return dvdN("n", z) }, ap("Nat.dvd_lcm_right", "m", "n")))),
				"h",
				tp("1", "m", sym("m", "1", ap("And.left", eqN("m", "1"), eqN("n", "1"), "h")), func(z string) string { return eqN(lcmN(z, "n"), "1") },
					tp("1", "n", sym("n", "1", ap("And.right", eqN("m", "1"), eqN("n", "1"), "h")), func(z string) string { return eqN(lcmN("1", z), "1") },
						ap("Nat.lcm_one_left", "1"))))),
	)
}
