package lean

// gcd / lcm of quotients by a common divisor, and a / gcd a b (Init/Data/Int/Gcd.lean). All go through
// natAbs_ediv_of_dvd to the Nat lemmas Nat.gcd_div / Nat.lcm_div / Nat.div_self.
func init() {
	ab := iabs
	ed := func(x, c string) string { return "(Int.ediv " + x + " " + c + ")" }
	dv := func(x, c string) string { return "(Nat.div " + ab(x) + " " + ab(c) + ")" }
	toN := func(c, x, h string) string {
		return iffMpr(dvdN(ab(c), ab(x)), idvd(c, x), ap("Int.natAbs_dvd_natAbs", c, x), h)
	}
	hs := [][2]string{{"ha", idvd("c", "a")}, {"hb", idvd("c", "b")}}
	nea := ap("Int.natAbs_ediv_of_dvd", "a", "c", "ha")
	neb := ap("Int.natAbs_ediv_of_dvd", "b", "c", "hb")
	viaNat := func(op func(a, b string) string, nat string) string {
		x0, x1, x2 := op(ab(ed("a", "c")), ab(ed("b", "c"))), op(dv("a", "c"), ab(ed("b", "c"))), op(dv("a", "c"), dv("b", "c"))
		x3 := "(Nat.div " + op(ab("a"), ab("b")) + " " + ab("c") + ")"
		s1 := cg(func(z string) string { return op(z, ab(ed("b", "c"))) }, ab(ed("a", "c")), dv("a", "c"), nea)
		s2 := cg(func(z string) string { return op(dv("a", "c"), z) }, ab(ed("b", "c")), dv("b", "c"), neb)
		return tr(x0, x2, x3, tr(x0, x1, x2, s1, s2), ap(nat, ab("a"), ab("b"), ab("c"), toN("c", "a", "ha"), toN("c", "b", "hb")))
	}
	// og = (gcd a b : Int); a / og ≠ 0 from 0 < |a| / gcd a b
	og := func(a, b string) string { return icast(igcd(a, b)) }
	neZero := func(name, vars, x, a, b, dvdLemma, posLemma string, posArgs ...string) valDecl {
		e := ed(x, og(a, b))
		e1 := ap("Int.natAbs_ediv_of_dvd", x, og(a, b), ap(dvdLemma, a, b))
		e2 := icgN(ab, e, iz(), "hz")
		d := "(Nat.div " + ab(x) + " " + igcd(a, b) + ")"
		eq := tr(d, ab(e), "0", sym(ab(e), d, e1), e2)
		return ithm(name, vars, [][2]string{{"h", inz(x)}}, notN(ieq(e, iz())),
			"(fun (hz : "+ieq(e, iz())+") => Nat.lt_irrefl 0 "+tp(d, "0", eq, func(z string) string { return ltN("0", z) }, ap(posLemma, append(posArgs, "h")...))+")")
	}
	preludeExtra = append(preludeExtra,
		ithm("Int.gcd_ediv", "a b c", hs, eqN(igcd(ed("a", "c"), ed("b", "c")), "(Nat.div "+igcd("a", "b")+" "+ab("c")+")"),
			viaNat(gcdN, "Nat.gcd_div")),
		ithm("Int.lcm_ediv", "a b c", hs, eqN(ilcm(ed("a", "c"), ed("b", "c")), "(Nat.div "+ilcm("a", "b")+" "+ab("c")+")"),
			viaNat(lcmN, "Nat.lcm_div")),
		neZero("Int.ediv_gcd_ne_zero_of_ne_zero_left", "b a", "a", "a", "b", "Int.gcd_dvd_left", "Int.natAbs_div_gcd_pos_of_ne_zero_left", "b", "a"),
		neZero("Int.ediv_gcd_ne_zero_of_ne_zero_right", "a b", "b", "a", "b", "Int.gcd_dvd_right", "Int.natAbs_div_gcd_pos_of_ne_zero_right", "a", "b"),
		ithm("Int.ediv_gcd_ne_zero_if_ne_zero_right", "a b", [][2]string{{"h", inz("b")}}, notN(ieq(ed("b", og("a", "b")), iz())),
			ap("Int.ediv_gcd_ne_zero_of_ne_zero_right", "a", "b", "h")),
	)
	// gcd (n / g) (m / g) = 1
	g := igcd("n", "m")
	one := func(pos string) string {
		return tr(igcd(ed("n", og("n", "m")), ed("m", og("n", "m"))), "(Nat.div "+g+" "+g+")", "1",
			ap("Int.gcd_ediv", "n", "m", og("n", "m"), ap("Int.gcd_dvd_left", "n", "m"), ap("Int.gcd_dvd_right", "n", "m")),
			ap("Nat.div_self", g, pos))
	}
	concl := eqN(igcd(ed("n", og("n", "m")), ed("m", og("n", "m"))), "1")
	preludeExtra = append(preludeExtra,
		ithm("Int.gcd_ediv_gcd_ediv_gcd_of_ne_zero_left", "n m", [][2]string{{"h", inz("n")}}, concl, one(ap("Int.gcd_pos_of_ne_zero_left", "m", "n", "h"))),
		ithm("Int.gcd_ediv_gcd_ediv_gcd_of_ne_zero_right", "n m", [][2]string{{"h", inz("m")}}, concl, one(ap("Int.gcd_pos_of_ne_zero_right", "n", "m", "h"))),
	)
	orI := "(Or " + inz("i") + " " + inz("j") + ")"
	preludeExtra = append(preludeExtra,
		ithm("Int.gcd_ediv_gcd_ediv_gcd", "i j", [][2]string{{"h", ltN("0", igcd("i", "j"))}},
			eqN(igcd(ed("i", og("i", "j")), ed("j", og("i", "j"))), "1"),
			"Or.rec "+inz("i")+" "+inz("j")+" (fun (_ : "+orI+") => "+eqN(igcd(ed("i", og("i", "j")), ed("j", og("i", "j"))), "1")+") "+
				"(fun (x : "+inz("i")+") => "+ap("Int.gcd_ediv_gcd_ediv_gcd_of_ne_zero_left", "i", "j", "x")+") "+
				"(fun (x : "+inz("j")+") => "+ap("Int.gcd_ediv_gcd_ediv_gcd_of_ne_zero_right", "i", "j", "x")+") "+
				iffMp(ltN("0", igcd("i", "j")), orI, ap("Int.gcd_pos_iff", "i", "j"), "h")),
	)
}
