package lean

func lcmN(a, b string) string { return "(Nat.lcm " + a + " " + b + ")" }

// lcm characterised by what it divides, its equalities with an argument, and the usual lcm-with-multiple forms
// (Init/Data/Nat/Lcm.lean).
func init() {
	no := [][2]string{}
	lc := func(a, b string) string { return ap("Nat.lcm_comm", a, b) }
	dl := func(a, b string) string { return ap("Nat.dvd_lcm_left", a, b) }
	dr := func(a, b string) string { return ap("Nat.dvd_lcm_right", a, b) }
	// lcm_eq_iff (n m l)
	lnm := lcmN("n", "m")
	allL := fa("c", imp(dvdN("n", "c"), imp(dvdN("m", "c"), dvdN("l", "c"))))
	allC := fa("c", imp(dvdN("n", "c"), imp(dvdN("m", "c"), dvdN(lnm, "c"))))
	tail := func(l string) string {
		return andN(dvdN("m", l), fa("c", imp(dvdN("n", "c"), imp(dvdN("m", "c"), dvdN(l, "c")))))
	}
	rhsL := func(l string) string { return andN(dvdN("n", l), tail(l)) }
	hN := ap("And.left", dvdN("n", "l"), tail("l"), "H")
	hT := ap("And.right", dvdN("n", "l"), tail("l"), "H")
	hM := ap("And.left", dvdN("m", "l"), allL, hT)
	hA := ap("And.right", dvdN("m", "l"), allL, hT)
	preludeExtra = append(preludeExtra,
		thm("Nat.lcm_eq_iff", "n m l", no, iffN(eqN(lnm, "l"), rhsL("l")),
			iffI(eqN(lnm, "l"), rhsL("l"), "h",
				tp(lnm, "l", "h", rhsL,
					ap("And.intro", dvdN("n", lnm), tail(lnm), dl("n", "m"),
						ap("And.intro", dvdN("m", lnm), allC, dr("n", "m"),
							"(fun (c : Nat) (hn : "+dvdN("n", "c")+") (hm : "+dvdN("m", "c")+") => "+ap("Nat.lcm_dvd", "n", "m", "c", "hn", "hm")+")"))),
				"H",
				ap("Nat.dvd_antisymm", lnm, "l", ap("Nat.lcm_dvd", "n", "m", "l", hN, hM),
					"("+hA+" "+lnm+" "+dl("n", "m")+" "+dr("n", "m")+")"))),
	)
	// equalities with an argument
	preludeExtra = append(preludeExtra,
		thm("Nat.lcm_eq_left_iff_dvd", "m n", no, iffN(eqN(lcmN("m", "n"), "m"), dvdN("n", "m")),
			iffI(eqN(lcmN("m", "n"), "m"), dvdN("n", "m"), "h",
				tp(lcmN("m", "n"), "m", "h", func(z string) string { return dvdN("n", z) }, dr("m", "n")),
				"h",
				ap("Nat.dvd_antisymm", lcmN("m", "n"), "m", ap("Nat.lcm_dvd", "m", "n", "m", ap("Nat.dvd_refl", "m"), "h"), dl("m", "n")))),
		thm("Nat.lcm_eq_right_iff_dvd", "m n", no, iffN(eqN(lcmN("m", "n"), "n"), dvdN("m", "n")),
			iffI(eqN(lcmN("m", "n"), "n"), dvdN("m", "n"), "h",
				tp(lcmN("m", "n"), "n", "h", func(z string) string { return dvdN("m", z) }, dl("m", "n")),
				"h",
				ap("Nat.dvd_antisymm", lcmN("m", "n"), "n", ap("Nat.lcm_dvd", "m", "n", "n", "h", ap("Nat.dvd_refl", "n")), dr("m", "n")))),
		thm("Nat.lcm_eq_left", "m n", [][2]string{{"h", dvdN("n", "m")}}, eqN(lcmN("m", "n"), "m"),
			iffMpr(eqN(lcmN("m", "n"), "m"), dvdN("n", "m"), ap("Nat.lcm_eq_left_iff_dvd", "m", "n"), "h")),
		thm("Nat.lcm_eq_right", "m n", [][2]string{{"h", dvdN("m", "n")}}, eqN(lcmN("m", "n"), "n"),
			iffMpr(eqN(lcmN("m", "n"), "n"), dvdN("m", "n"), ap("Nat.lcm_eq_right_iff_dvd", "m", "n"), "h")),
		thm("Nat.lcm_mul_left_left", "m n", no, eqN(lcmN(mulN("m", "n"), "n"), mulN("m", "n")),
			ap("Nat.lcm_eq_left", mulN("m", "n"), "n", ap("Nat.dvd_mul_left", "n", "m"))),
		thm("Nat.lcm_mul_left_right", "m n", no, eqN(lcmN("n", mulN("m", "n")), mulN("m", "n")),
			ap("Nat.lcm_eq_right", "n", mulN("m", "n"), ap("Nat.dvd_mul_left", "n", "m"))),
		thm("Nat.lcm_mul_right_left", "m n", no, eqN(lcmN(mulN("n", "m"), "n"), mulN("n", "m")),
			ap("Nat.lcm_eq_left", mulN("n", "m"), "n", ap("Nat.dvd_mul_right", "n", "m"))),
		thm("Nat.lcm_mul_right_right", "m n", no, eqN(lcmN("n", mulN("n", "m")), mulN("n", "m")),
			ap("Nat.lcm_eq_right", "n", mulN("n", "m"), ap("Nat.dvd_mul_right", "n", "m"))),
		thm("Nat.lcm_lcm_self_right_left", "m n", no, eqN(lcmN("m", lcmN("m", "n")), lcmN("m", "n")),
			ap("Nat.lcm_eq_right", "m", lcmN("m", "n"), dl("m", "n"))),
		thm("Nat.lcm_lcm_self_right_right", "m n", no, eqN(lcmN("m", lcmN("n", "m")), lcmN("m", "n")),
			tr(lcmN("m", lcmN("n", "m")), lcmN("m", lcmN("m", "n")), lcmN("m", "n"),
				cg(func(z string) string { return lcmN("m", z) }, lcmN("n", "m"), lcmN("m", "n"), lc("n", "m")),
				ap("Nat.lcm_lcm_self_right_left", "m", "n"))),
		thm("Nat.lcm_lcm_self_left_left", "m n", no, eqN(lcmN(lcmN("m", "n"), "m"), lcmN("n", "m")),
			tr(lcmN(lcmN("m", "n"), "m"), lcmN("m", "n"), lcmN("n", "m"),
				ap("Nat.lcm_eq_left", lcmN("m", "n"), "m", dl("m", "n")), lc("m", "n"))),
		thm("Nat.lcm_lcm_self_left_right", "m n", no, eqN(lcmN(lcmN("n", "m"), "m"), lcmN("n", "m")),
			ap("Nat.lcm_eq_left", lcmN("n", "m"), "m", dr("n", "m"))),
	)
	// lcm with a multiple on one side
	preludeExtra = append(preludeExtra,
		thm("Nat.lcm_dvd_lcm_mul_left_left", "m n k", no, dvdN(lcmN("m", "n"), lcmN(mulN("k", "m"), "n")),
			ap("Nat.lcm_dvd_lcm_of_dvd_left", "n", "m", mulN("k", "m"), ap("Nat.dvd_mul_left", "m", "k"))),
		thm("Nat.lcm_dvd_lcm_mul_right_left", "m n k", no, dvdN(lcmN("m", "n"), lcmN(mulN("m", "k"), "n")),
			ap("Nat.lcm_dvd_lcm_of_dvd_left", "n", "m", mulN("m", "k"), ap("Nat.dvd_mul_right", "m", "k"))),
		thm("Nat.lcm_dvd_lcm_mul_left_right", "m n k", no, dvdN(lcmN("m", "n"), lcmN("m", mulN("k", "n"))),
			ap("Nat.lcm_dvd_lcm_of_dvd_right", "m", "n", mulN("k", "n"), ap("Nat.dvd_mul_left", "n", "k"))),
		thm("Nat.lcm_dvd_lcm_mul_right_right", "m n k", no, dvdN(lcmN("m", "n"), lcmN("m", mulN("n", "k"))),
			ap("Nat.lcm_dvd_lcm_of_dvd_right", "m", "n", mulN("n", "k"), ap("Nat.dvd_mul_right", "n", "k"))),
		thm("Nat.lcm_le_mul", "m n", [][2]string{{"hm", ltN("0", "m")}, {"hn", ltN("0", "n")}}, leN(lcmN("m", "n"), mulN("m", "n")),
			ap("Nat.le_of_dvd", lcmN("m", "n"), mulN("m", "n"), ap("Nat.mul_pos", "m", "n", "hm", "hn"), ap("Nat.lcm_dvd_mul", "m", "n"))),
		thm("Nat.gcd_le_mul", "m n", [][2]string{{"hm", ltN("0", "m")}, {"hn", ltN("0", "n")}}, leN(gcdN("m", "n"), mulN("m", "n")),
			ap("Nat.le_of_dvd", gcdN("m", "n"), mulN("m", "n"), ap("Nat.mul_pos", "m", "n", "hm", "hn"), ap("Nat.gcd_dvd_mul", "m", "n"))),
		thm("Nat.lcm_one_left", "n", no, eqN(lcmN("1", "n"), "n"), ap("Nat.lcm_eq_right", "1", "n", ap("Nat.one_dvd", "n"))),
		thm("Nat.lcm_one_right", "n", no, eqN(lcmN("n", "1"), "n"), ap("Nat.lcm_eq_left", "n", "1", ap("Nat.one_dvd", "n"))),
		thm("Nat.lcm_self", "n", no, eqN(lcmN("n", "n"), "n"), ap("Nat.lcm_eq_left", "n", "n", ap("Nat.dvd_refl", "n"))),
	)
}
