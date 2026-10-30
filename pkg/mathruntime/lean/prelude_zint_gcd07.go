package lean

// Int lcm against products, the gcd = 1 cancellations, and lcm zero / positivity.
func init() {
	ab := iabs
	nm := func(x, y string) string { return ap("Int.natAbs_mul", x, y) }
	via := func(x, y string, P func(z string) string, body string) string {
		return tp(mulN(ab(x), ab(y)), ab(imul(x, y)), sym(ab(imul(x, y)), mulN(ab(x), ab(y)), nm(x, y)), P, body)
	}
	idv := func(a, b string) string { return "(Int.dvd " + a + " " + b + ")" }
	toN := func(a, b, h string) string {
		return iffMpr(dvdN(ab(a), ab(b)), idv(a, b), ap("Int.natAbs_dvd_natAbs", a, b), h)
	}
	lc := lcmN
	g := gcdN
	// aliases with explicit arguments
	preludeExtra = append(preludeExtra,
		ithm("Int.gcd_zero_left", "a", nil, eqN(igcd(iz(), "a"), ab("a")), ap("Nat.gcd_zero_left", ab("a"))),
		ithm("Int.gcd_one_left", "a", nil, eqN(igcd(io(), "a"), "1"), ap("Nat.gcd_one_left", ab("a"))),
		ithm("Int.gcd_one_right", "a", nil, eqN(igcd("a", io()), "1"), ap("Nat.gcd_one_right", ab("a"))),
		ithm("Int.lcm_mul_left_left", "a b", nil, eqN(ilcm(imul("a", "b"), "b"), mulN(ab("a"), ab("b"))),
			via("a", "b", func(z string) string { return eqN(lc(z, ab("b")), mulN(ab("a"), ab("b"))) }, ap("Nat.lcm_mul_left_left", ab("a"), ab("b")))),
		ithm("Int.lcm_mul_left_right", "a b", nil, eqN(ilcm("a", imul("b", "a")), mulN(ab("b"), ab("a"))),
			via("b", "a", func(z string) string { return eqN(lc(ab("a"), z), mulN(ab("b"), ab("a"))) }, ap("Nat.lcm_mul_left_right", ab("b"), ab("a")))),
		ithm("Int.lcm_mul_right_left", "a b", nil, eqN(ilcm(imul("b", "a"), "b"), mulN(ab("b"), ab("a"))),
			via("b", "a", func(z string) string { return eqN(lc(z, ab("b")), mulN(ab("b"), ab("a"))) }, ap("Nat.lcm_mul_right_left", ab("a"), ab("b")))),
		ithm("Int.lcm_mul_right_right", "a b", nil, eqN(ilcm("a", imul("a", "b")), mulN(ab("a"), ab("b"))),
			via("a", "b", func(z string) string { return eqN(lc(ab("a"), z), mulN(ab("a"), ab("b"))) }, ap("Nat.lcm_mul_right_right", ab("b"), ab("a")))),
		ithm("Int.lcm_lcm_self_right_left", "m n", nil, eqN(ilcm("m", icast(ilcm("m", "n"))), ilcm("m", "n")), ap("Nat.lcm_lcm_self_right_left", ab("m"), ab("n"))),
		ithm("Int.lcm_lcm_self_right_right", "m n", nil, eqN(ilcm("m", icast(ilcm("n", "m"))), ilcm("m", "n")), ap("Nat.lcm_lcm_self_right_right", ab("m"), ab("n"))),
		ithm("Int.lcm_lcm_self_left_right", "m n", nil, eqN(ilcm(icast(ilcm("n", "m")), "m"), ilcm("n", "m")), ap("Nat.lcm_lcm_self_left_right", ab("m"), ab("n"))),
		ithm("Int.lcm_lcm_self_left_left", "m n", nil, eqN(ilcm(icast(ilcm("m", "n")), "m"), ilcm("n", "m")), ap("Nat.lcm_lcm_self_left_left", ab("m"), ab("n"))),
		ithm("Int.lcm_dvd_lcm_of_dvd_left", "c a b", [][2]string{{"h", idv("a", "b")}}, dvdN(ilcm("a", "c"), ilcm("b", "c")),
			ap("Nat.lcm_dvd_lcm_of_dvd_left", ab("c"), ab("a"), ab("b"), toN("a", "b", "h"))),
		ithm("Int.lcm_dvd_lcm_of_dvd_right", "c a b", [][2]string{{"h", idv("a", "b")}}, dvdN(ilcm("c", "a"), ilcm("c", "b")),
			ap("Nat.lcm_dvd_lcm_of_dvd_right", ab("c"), ab("a"), ab("b"), toN("a", "b", "h"))),
		ithm("Int.lcm_dvd_lcm_mul_left_left", "a b c", nil, dvdN(ilcm("a", "b"), ilcm(imul("c", "a"), "b")),
			via("c", "a", func(z string) string { return dvdN(ilcm("a", "b"), lc(z, ab("b"))) }, ap("Nat.lcm_dvd_lcm_mul_left_left", ab("a"), ab("b"), ab("c")))),
		ithm("Int.lcm_dvd_lcm_mul_right_left", "a b c", nil, dvdN(ilcm("a", "b"), ilcm(imul("a", "c"), "b")),
			via("a", "c", func(z string) string { return dvdN(ilcm("a", "b"), lc(z, ab("b"))) }, ap("Nat.lcm_dvd_lcm_mul_right_left", ab("a"), ab("b"), ab("c")))),
		ithm("Int.lcm_dvd_lcm_mul_left_right", "a b c", nil, dvdN(ilcm("a", "b"), ilcm("a", imul("c", "b"))),
			via("c", "b", func(z string) string { return dvdN(ilcm("a", "b"), lc(ab("a"), z)) }, ap("Nat.lcm_dvd_lcm_mul_left_right", ab("a"), ab("b"), ab("c")))),
		ithm("Int.lcm_dvd_lcm_mul_right_right", "a b c", nil, dvdN(ilcm("a", "b"), ilcm("a", imul("b", "c"))),
			via("b", "c", func(z string) string { return dvdN(ilcm("a", "b"), lc(ab("a"), z)) }, ap("Nat.lcm_dvd_lcm_mul_right_right", ab("a"), ab("b"), ab("c")))),
	)
	// gcd n m = 1 cancellations
	h1 := [][2]string{{"h", eqN(igcd("n", "m"), "1")}}
	preludeExtra = append(preludeExtra,
		ithm("Int.gcd_mul_right_right_of_gcd_eq_one", "n m k", h1, eqN(igcd("n", imul("m", "k")), igcd("n", "k")),
			via("m", "k", func(z string) string { return eqN(g(ab("n"), z), igcd("n", "k")) }, ap("Nat.gcd_mul_right_right_of_gcd_eq_one", ab("n"), ab("m"), ab("k"), "h"))),
		ithm("Int.gcd_mul_left_right_of_gcd_eq_one", "n m k", h1, eqN(igcd("n", imul("k", "m")), igcd("n", "k")),
			via("k", "m", func(z string) string { return eqN(g(ab("n"), z), igcd("n", "k")) }, ap("Nat.gcd_mul_left_right_of_gcd_eq_one", ab("n"), ab("m"), ab("k"), "h"))),
		ithm("Int.gcd_mul_right_left_of_gcd_eq_one", "n m k", h1, eqN(igcd(imul("n", "k"), "m"), igcd("k", "m")),
			via("n", "k", func(z string) string { return eqN(g(z, ab("m")), igcd("k", "m")) }, ap("Nat.gcd_mul_right_left_of_gcd_eq_one", ab("n"), ab("m"), ab("k"), "h"))),
		ithm("Int.gcd_mul_left_left_of_gcd_eq_one", "n m k", h1, eqN(igcd(imul("k", "n"), "m"), igcd("k", "m")),
			via("k", "n", func(z string) string { return eqN(g(z, ab("m")), igcd("k", "m")) }, ap("Nat.gcd_mul_left_left_of_gcd_eq_one", ab("n"), ab("m"), ab("k"), "h"))),
	)
	// lcm zero / positivity
	z := func(x string) string { return ieq(x, iz()) }
	orZ := "(Or " + z("m") + " " + z("n") + ")"
	orZN := "(Or " + eqN(ab("m"), "0") + " " + eqN(ab("n"), "0") + ")"
	fromN := func(x, h string) string { return iffMp(eqN(ab(x), "0"), z(x), ap("Int.natAbs_eq_zero", x), h) }
	toN0 := func(x, h string) string { return iffMpr(eqN(ab(x), "0"), z(x), ap("Int.natAbs_eq_zero", x), h) }
	lz := eqN(ilcm("m", "n"), "0")
	preludeExtra = append(preludeExtra,
		ithm("Int.eq_zero_of_lcm_eq_zero", "m n", [][2]string{{"h", lz}}, orZ,
			"(Or.rec "+eqN(ab("m"), "0")+" "+eqN(ab("n"), "0")+" (fun (_ : "+orZN+") => "+orZ+") "+
				"(fun (x : "+eqN(ab("m"), "0")+") => "+ap("Or.inl", z("m"), z("n"), fromN("m", "x"))+") "+
				"(fun (x : "+eqN(ab("n"), "0")+") => "+ap("Or.inr", z("m"), z("n"), fromN("n", "x"))+") "+
				ap("Nat.eq_zero_of_lcm_eq_zero", ab("m"), ab("n"), "h")+")"),
		ithm("Int.lcm_pos", "m n", [][2]string{{"hm", inz("m")}, {"hn", inz("n")}}, ltN("0", ilcm("m", "n")),
			ap("Nat.lcm_pos", ab("m"), ab("n"), absPos("m", "hm"), absPos("n", "hn"))),
		ithm("Int.lcm_eq_zero_iff", "m n", nil, "(Iff "+lz+" "+orZ+")",
			iffI(lz, orZ, "h", ap("Int.eq_zero_of_lcm_eq_zero", "m", "n", "h"),
				"h", "(Or.rec "+z("m")+" "+z("n")+" (fun (_ : "+orZ+") => "+lz+") "+
					"(fun (x : "+z("m")+") => "+ap("Eq.trans.{1} Nat", ilcm("m", "n"), lc(ab("m"), ab("n")), "0",
					"(Eq.refl.{1} Nat "+ilcm("m", "n")+")", ap("Eq.trans.{1} Nat", lc(ab("m"), ab("n")), lc("0", ab("n")), "0",
						"(congrArg.{1, 1} Nat Nat (fun (w : Nat) => "+lc("w", ab("n"))+") "+ab("m")+" 0 "+toN0("m", "x")+")", ap("Nat.lcm_zero_left", ab("n"))))+") "+
					"(fun (x : "+z("n")+") => "+ap("Eq.trans.{1} Nat", ilcm("m", "n"), lc(ab("m"), ab("n")), "0",
					"(Eq.refl.{1} Nat "+ilcm("m", "n")+")", ap("Eq.trans.{1} Nat", lc(ab("m"), ab("n")), lc(ab("m"), "0"), "0",
						"(congrArg.{1, 1} Nat Nat (fun (w : Nat) => "+lc(ab("m"), "w")+") "+ab("n")+" 0 "+toN0("n", "x")+")", ap("Nat.lcm_zero_right", ab("m"))))+") h)")),
		ithm("Int.lcm_pos_iff", "m n", nil, "(Iff "+ltN("0", ilcm("m", "n"))+" "+andN(inz("m"), inz("n"))+")",
			iffI(ltN("0", ilcm("m", "n")), andN(inz("m"), inz("n")), "h",
				ap("And.intro", inz("m"), inz("n"),
					"(fun (e : "+z("m")+") => "+ap("Nat.lt_irrefl", "0", tp(ilcm("m", "n"), "0", tpI(iz(), "m", "(Eq.symm.{1} Int m (Int.ofNat 0) e)", func(w string) string { return eqN(ilcm(w, "n"), "0") }, ap("Int.lcm_zero_left", "n")), func(w string) string { return ltN("0", w) }, "h"))+")",
					"(fun (e : "+z("n")+") => "+ap("Nat.lt_irrefl", "0", tp(ilcm("m", "n"), "0", tpI(iz(), "n", "(Eq.symm.{1} Int n (Int.ofNat 0) e)", func(w string) string { return eqN(ilcm("m", w), "0") }, ap("Int.lcm_zero_right", "m")), func(w string) string { return ltN("0", w) }, "h"))+")"),
				"h", ap("Int.lcm_pos", "m", "n", ap("And.left", inz("m"), inz("n"), "h"), ap("And.right", inz("m"), inz("n"), "h")))),
	)
}

// tpI is tp over Int: it carries `body : P a` to `P b` along `p : a = b`.
func tpI(a, b, p string, P func(z string) string, body string) string {
	return "(Eq.rec.{0, 1} Int " + a + " (fun (z : Int) (_ : Eq.{1} Int " + a + " z) => " + P("z") + ") " + body + " " + b + " " + p + ")"
}
