package lean

// Int gcd / lcm: products, bounds, and the lcm_eq_mul_of_lcm_eq_mul family.
func init() {
	ab := iabs
	oN := icast
	nm := func(x, y string) string { return ap("Int.natAbs_mul", x, y) }
	via := func(x, y string, P func(z string) string, body string) string {
		return tp(mulN(ab(x), ab(y)), ab(imul(x, y)), sym(ab(imul(x, y)), mulN(ab(x), ab(y)), nm(x, y)), P, body)
	}
	idv := func(a, b string) string { return "(Int.dvd " + a + " " + b + ")" }
	toI := func(a, b, h string) string {
		return iffMp(dvdN(ab(a), ab(b)), idv(a, b), ap("Int.natAbs_dvd_natAbs", a, b), h)
	}
	iff := func(l, r string) string { return "(Iff " + l + " " + r + ")" }
	preludeExtra = append(preludeExtra,
		ithm("Int.gcd_dvd_mul", "m n", nil, idv(oN(igcd("m", "n")), imul("m", "n")),
			toI(oN(igcd("m", "n")), imul("m", "n"), via("m", "n", func(z string) string { return dvdN(igcd("m", "n"), z) }, ap("Nat.gcd_dvd_mul", ab("m"), ab("n"))))),
		ithm("Int.lcm_dvd_mul", "m n", nil, idv(oN(ilcm("m", "n")), imul("m", "n")),
			toI(oN(ilcm("m", "n")), imul("m", "n"), via("m", "n", func(z string) string { return dvdN(ilcm("m", "n"), z) }, ap("Nat.lcm_dvd_mul", ab("m"), ab("n"))))),
		ithm("Int.lcm_eq_mul_div", "m n", nil, eqN(ilcm("m", "n"), "(Nat.div "+mulN(ab("m"), ab("n"))+" "+igcd("m", "n")+")"), ap("Nat.lcm_eq_mul_div", ab("m"), ab("n"))),
		ithm("Int.lcm_eq_left", "a b", [][2]string{{"ha", "(Int.le (Int.ofNat 0) a)"}, {"h", idv("b", "a")}}, ieq(oN(ilcm("a", "b")), "a"),
			iffMpr(ieq(oN(ilcm("a", "b")), "a"), idv("b", "a"), ap("Int.lcm_eq_left_iff_dvd", "a", "b", "ha"), "h")),
		ithm("Int.lcm_eq_right", "a b", [][2]string{{"hb", "(Int.le (Int.ofNat 0) b)"}, {"h", idv("a", "b")}}, ieq(oN(ilcm("a", "b")), "b"),
			iffMpr(ieq(oN(ilcm("a", "b")), "b"), idv("a", "b"), ap("Int.lcm_eq_right_iff_dvd", "a", "b", "hb"), "h")),
	)
	// Nat-argument statements: the casts sit on the leaves, as Lean's `binop%` elaborates them
	k, m, n := oN("k"), oN("m"), oN("n")
	preludeExtra = append(preludeExtra,
		nat3("Int.lcm_mul_right_dvd_mul_lcm", "k m n", dvdN(ilcm(k, imul(m, n)), mulN(ilcm(k, m), ilcm(k, n))),
			via(m, n, func(z string) string { return dvdN(lcmN("k", z), mulN(ilcm(k, m), ilcm(k, n))) }, ap("Nat.lcm_mul_right_dvd_mul_lcm", "k", "m", "n"))),
		nat3("Int.lcm_mul_left_dvd_mul_lcm", "k m n", dvdN(ilcm(imul(m, n), k), mulN(ilcm(m, k), ilcm(n, k))),
			via(m, n, func(z string) string { return dvdN(lcmN(z, "k"), mulN(ilcm(m, k), ilcm(n, k))) }, ap("Nat.lcm_mul_left_dvd_mul_lcm", "k", "m", "n"))),
		nat3("Int.lcm_dvd_mul_self_left_iff_dvd_mul", "k n m", iff(dvdN(ilcm(oN("k"), oN("n")), mulN("k", "m")), dvdN("n", mulN("k", "m"))),
			ap("Nat.lcm_dvd_mul_self_left_iff_dvd_mul", "k", "n", "m")),
		nat3("Int.lcm_dvd_mul_self_right_iff_dvd_mul", "k m n", iff(dvdN(ilcm(oN("n"), oN("k")), mulN("m", "k")), dvdN("n", mulN("m", "k"))),
			ap("Nat.lcm_dvd_mul_self_right_iff_dvd_mul", "k", "m", "n")),
	)
	h := [][2]string{{"h", eqN(ilcm("n", "m"), mulN(ab("n"), ab("m")))}}
	preludeExtra = append(preludeExtra,
		ithm("Int.lcm_mul_right_right_eq_mul_of_lcm_eq_mul", "n m k", h, eqN(ilcm("n", imul("m", "k")), mulN(ilcm("n", "k"), ab("m"))),
			via("m", "k", func(z string) string { return eqN(lcmN(ab("n"), z), mulN(ilcm("n", "k"), ab("m"))) }, ap("Nat.lcm_mul_right_right_eq_mul_of_lcm_eq_mul", ab("n"), ab("m"), ab("k"), "h"))),
		ithm("Int.lcm_mul_left_right_eq_mul_of_lcm_eq_mul", "n m k", h, eqN(ilcm("n", imul("k", "m")), mulN(ilcm("n", "k"), ab("m"))),
			via("k", "m", func(z string) string { return eqN(lcmN(ab("n"), z), mulN(ilcm("n", "k"), ab("m"))) }, ap("Nat.lcm_mul_left_right_eq_mul_of_lcm_eq_mul", ab("n"), ab("m"), ab("k"), "h"))),
		ithm("Int.lcm_mul_right_left_eq_mul_of_lcm_eq_mul", "n m k", h, eqN(ilcm(imul("n", "k"), "m"), mulN(ab("n"), ilcm("k", "m"))),
			via("n", "k", func(z string) string { return eqN(lcmN(z, ab("m")), mulN(ab("n"), ilcm("k", "m"))) }, ap("Nat.lcm_mul_right_left_eq_mul_of_lcm_eq_mul", ab("n"), ab("m"), ab("k"), "h"))),
		ithm("Int.lcm_mul_left_left_eq_mul_of_lcm_eq_mul", "n m k", h, eqN(ilcm(imul("k", "n"), "m"), mulN(ab("n"), ilcm("k", "m"))),
			via("k", "n", func(z string) string { return eqN(lcmN(z, ab("m")), mulN(ab("n"), ilcm("k", "m"))) }, ap("Nat.lcm_mul_left_left_eq_mul_of_lcm_eq_mul", ab("n"), ab("m"), ab("k"), "h"))),
	)
	// le_lcm: a ≤ ↑(lcm a b) via a ≤ ↑|a| ≤ ↑(lcm a b)
	le := func(x, y string) string { return "(Int.le " + x + " " + y + ")" }
	leLcm := func(x, other string, natLe string) string {
		return ap("Int.le_trans", x, oN(ab(x)), oN(ilcm("a", "b")), ap("Int.le_natAbs", x),
			iffMpr(le(oN(ab(x)), oN(ilcm("a", "b"))), leN(ab(x), ilcm("a", "b")), ap("Int.ofNat_le", ab(x), ilcm("a", "b")), natLe))
	}
	preludeExtra = append(preludeExtra,
		ithm("Int.le_lcm_left", "a b", [][2]string{{"hb", inz("b")}}, le("a", oN(ilcm("a", "b"))), leLcm("a", "b", ap("Int.natAbs_le_lcm_left", "a", "b", "hb"))),
		ithm("Int.le_lcm_right", "b a", [][2]string{{"ha", inz("a")}}, le("b", oN(ilcm("a", "b"))), leLcm("b", "a", ap("Int.natAbs_le_lcm_right", "b", "a", "ha"))),
	)
}

// nat3 builds a theorem over Nat variables, stated about their casts.
func nat3(name, vars, concl, proof string) valDecl { return thm(name, vars, nil, concl, proof) }
