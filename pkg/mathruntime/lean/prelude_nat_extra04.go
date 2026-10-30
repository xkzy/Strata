package lean

func iffN(a, b string) string  { return "(Iff " + a + " " + b + ")" }
func andN(a, b string) string  { return "(And " + a + " " + b + ")" }
func fa(v, body string) string { return "((" + v + " : Nat) → " + body + ")" }
func imp(a, b string) string   { return "(" + a + " → " + b + ")" }

// iffI builds `Iff.intro` with its two directions as lambdas over `x`.
func iffI(a, b, x1, mp, x2, mpr string) string {
	return "(Iff.intro " + a + " " + b + " (fun (" + x1 + " : " + a + ") => " + mp + ") (fun (" + x2 + " : " + b + ") => " + mpr + "))"
}
func iffMp(a, b, i, h string) string  { return "(Iff.mp " + a + " " + b + " " + i + " " + h + ")" }
func iffMpr(a, b, i, h string) string { return "(Iff.mpr " + a + " " + b + " " + i + " " + h + ")" }

// gcd characterised by what divides it (Init/Data/Nat/Gcd.lean), and `gcd` against products.
func init() {
	no := [][2]string{}
	comm := func(a, b string) string { return ap("Nat.gcd_comm", a, b) }
	// gcd_right_eq_iff
	rEq := func(m, n, n2 string) string { // statement body
		return iffN(eqN(gcdN(m, n), gcdN(m, n2)),
			fa("k", imp(dvdN("k", m), iffN(dvdN("k", n), dvdN("k", n2)))))
	}
	gmn, gmn2 := gcdN("m", "n"), gcdN("m", "n2")
	preludeExtra = append(preludeExtra,
		thm("Nat.gcd_right_eq_iff", "m n n2", no, rEq("m", "n", "n2"),
			iffI(eqN(gmn, gmn2), fa("k", imp(dvdN("k", "m"), iffN(dvdN("k", "n"), dvdN("k", "n2")))), "h",
				"fun (k : Nat) (hkm : "+dvdN("k", "m")+") => "+iffI(dvdN("k", "n"), dvdN("k", "n2"), "hkn",
					ap("Nat.dvd_trans", "k", gmn2, "n2",
						tp(gmn, gmn2, "h", func(z string) string { return dvdN("k", z) }, ap("Nat.dvd_gcd", "k", "m", "n", "hkm", "hkn")),
						ap("Nat.gcd_dvd_right", "m", "n2")),
					"hkn2",
					ap("Nat.dvd_trans", "k", gmn, "n",
						tp(gmn2, gmn, sym(gmn, gmn2, "h"), func(z string) string { return dvdN("k", z) }, ap("Nat.dvd_gcd", "k", "m", "n2", "hkm", "hkn2")),
						ap("Nat.gcd_dvd_right", "m", "n"))),
				"H",
				ap("Nat.gcd_right_ext", "m", "n", "n2",
					iffMp(dvdN(gmn, "n"), dvdN(gmn, "n2"), ap("H", gmn, ap("Nat.gcd_dvd_left", "m", "n")), ap("Nat.gcd_dvd_right", "m", "n")),
					iffMpr(dvdN(gmn2, "n"), dvdN(gmn2, "n2"), ap("H", gmn2, ap("Nat.gcd_dvd_left", "m", "n2")), ap("Nat.gcd_dvd_right", "m", "n2"))))),
	)
	// gcd_left_eq_iff (m m2 n)
	lhs := eqN(gcdN("m", "n"), gcdN("m2", "n"))
	rhs := fa("k", imp(dvdN("k", "n"), iffN(dvdN("k", "m"), dvdN("k", "m2"))))
	rEqNM := iffN(eqN(gcdN("n", "m"), gcdN("n", "m2")), fa("k", imp(dvdN("k", "n"), iffN(dvdN("k", "m"), dvdN("k", "m2")))))
	rEqI := ap("Nat.gcd_right_eq_iff", "n", "m", "m2")
	preludeExtra = append(preludeExtra,
		thm("Nat.gcd_left_eq_iff", "m m2 n", no, iffN(lhs, rhs),
			iffI(lhs, rhs, "h",
				iffMp(eqN(gcdN("n", "m"), gcdN("n", "m2")), rhs, rEqI,
					tr(gcdN("n", "m"), gcdN("m", "n"), gcdN("n", "m2"), comm("n", "m"),
						tr(gcdN("m", "n"), gcdN("m2", "n"), gcdN("n", "m2"), "h", comm("m2", "n")))),
				"H",
				tr(gcdN("m", "n"), gcdN("n", "m"), gcdN("m2", "n"), comm("m", "n"),
					tr(gcdN("n", "m"), gcdN("n", "m2"), gcdN("m2", "n"),
						iffMpr(eqN(gcdN("n", "m"), gcdN("n", "m2")), rhs, rEqI, "H"), comm("n", "m2"))))),
	)
	_ = rEqNM
	// gcd_eq_iff (a b g)
	rhsG := func(g string) string {
		return andN(dvdN(g, "a"), andN(dvdN(g, "b"), fa("c", imp(dvdN("c", "a"), imp(dvdN("c", "b"), dvdN("c", g))))))
	}
	gab := gcdN("a", "b")
	preludeExtra = append(preludeExtra,
		thm("Nat.gcd_eq_iff", "a b g", no, iffN(eqN(gab, "g"), rhsG("g")),
			iffI(eqN(gab, "g"), rhsG("g"), "h",
				tp(gab, "g", "h", rhsG,
					ap("And.intro", dvdN(gab, "a"), andN(dvdN(gab, "b"), fa("c", imp(dvdN("c", "a"), imp(dvdN("c", "b"), dvdN("c", gab))))),
						ap("Nat.gcd_dvd_left", "a", "b"),
						ap("And.intro", dvdN(gab, "b"), fa("c", imp(dvdN("c", "a"), imp(dvdN("c", "b"), dvdN("c", gab)))),
							ap("Nat.gcd_dvd_right", "a", "b"),
							"(fun (c : Nat) (ha : "+dvdN("c", "a")+") (hb : "+dvdN("c", "b")+") => "+ap("Nat.dvd_gcd", "c", "a", "b", "ha", "hb")+")"))),
				"H",
				ap("Nat.dvd_antisymm", gab, "g",
					"("+ap("And.right", dvdN("g", "b"), fa("c", imp(dvdN("c", "a"), imp(dvdN("c", "b"), dvdN("c", "g")))),
						ap("And.right", dvdN("g", "a"), andN(dvdN("g", "b"), fa("c", imp(dvdN("c", "a"), imp(dvdN("c", "b"), dvdN("c", "g"))))), "H"))+
						" "+gab+" "+ap("Nat.gcd_dvd_left", "a", "b")+" "+ap("Nat.gcd_dvd_right", "a", "b")+")",
					ap("Nat.dvd_gcd", "g", "a", "b",
						ap("And.left", dvdN("g", "a"), andN(dvdN("g", "b"), fa("c", imp(dvdN("c", "a"), imp(dvdN("c", "b"), dvdN("c", "g"))))), "H"),
						ap("And.left", dvdN("g", "b"), fa("c", imp(dvdN("c", "a"), imp(dvdN("c", "b"), dvdN("c", "g")))),
							ap("And.right", dvdN("g", "a"), andN(dvdN("g", "b"), fa("c", imp(dvdN("c", "a"), imp(dvdN("c", "b"), dvdN("c", "g"))))), "H")))))),
	)
	// gcd_eq_one_iff (m n)
	oneR := fa("c", imp(dvdN("c", "m"), imp(dvdN("c", "n"), eqN("c", "1"))))
	preludeExtra = append(preludeExtra,
		thm("Nat.gcd_eq_one_iff", "m n", no, iffN(eqN(gmn, "1"), oneR),
			iffI(eqN(gmn, "1"), oneR, "h",
				"(fun (c : Nat) (hm : "+dvdN("c", "m")+") (hn : "+dvdN("c", "n")+") => "+ap("Nat.eq_one_of_dvd_one", "c",
					tp(gmn, "1", "h", func(z string) string { return dvdN("c", z) }, ap("Nat.dvd_gcd", "c", "m", "n", "hm", "hn")))+")",
				"H",
				ap("H", gmn, ap("Nat.gcd_dvd_left", "m", "n"), ap("Nat.gcd_dvd_right", "m", "n")))),
	)
}
