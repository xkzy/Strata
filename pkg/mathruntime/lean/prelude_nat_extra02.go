package lean

// gcd with multiples and sums (Init/Data/Nat/Gcd.lean). `Nat.gcd_right_ext` is a port helper: two gcds with the
// same left argument are equal when each divides the other's second argument.
func init() {
	gm := func(m string) func(string) string { return func(z string) string { return gcdN(m, z) } }
	gR := func(z string) func(string) string { return func(a string) string { return gcdN(a, z) } }
	_ = gR
	no := [][2]string{}
	comm := func(a, b string) string { return ap("Nat.gcd_comm", a, b) }
	preludeExtra = append(preludeExtra,
		thm("Nat.gcd_right_ext", "m a b", [][2]string{{"h1", dvdN(gcdN("m", "a"), "b")}, {"h2", dvdN(gcdN("m", "b"), "a")}},
			eqN(gcdN("m", "a"), gcdN("m", "b")),
			ap("Nat.dvd_antisymm", gcdN("m", "a"), gcdN("m", "b"),
				ap("Nat.dvd_gcd", gcdN("m", "a"), "m", "b", ap("Nat.gcd_dvd_left", "m", "a"), "h1"),
				ap("Nat.dvd_gcd", gcdN("m", "b"), "m", "a", ap("Nat.gcd_dvd_left", "m", "b"), "h2"))),
		thm("Nat.gcd_mul_left_left", "m n", no, eqN(gcdN(mulN("m", "n"), "n"), "n"),
			ap("Nat.dvd_antisymm", gcdN(mulN("m", "n"), "n"), "n", ap("Nat.gcd_dvd_right", mulN("m", "n"), "n"),
				ap("Nat.dvd_gcd", "n", mulN("m", "n"), "n", ap("Nat.dvd_mul_left", "n", "m"), ap("Nat.dvd_refl", "n")))),
		thm("Nat.gcd_mul_left_right", "m n", no, eqN(gcdN("n", mulN("m", "n")), "n"),
			tr(gcdN("n", mulN("m", "n")), gcdN(mulN("m", "n"), "n"), "n", comm("n", mulN("m", "n")), ap("Nat.gcd_mul_left_left", "m", "n"))),
		thm("Nat.gcd_mul_right_left", "m n", no, eqN(gcdN(mulN("n", "m"), "n"), "n"),
			tr(gcdN(mulN("n", "m"), "n"), gcdN(mulN("m", "n"), "n"), "n",
				cg(func(z string) string { return gcdN(z, "n") }, mulN("n", "m"), mulN("m", "n"), ap("Nat.mul_comm", "n", "m")),
				ap("Nat.gcd_mul_left_left", "m", "n"))),
		thm("Nat.gcd_mul_right_right", "m n", no, eqN(gcdN("n", mulN("n", "m")), "n"),
			tr(gcdN("n", mulN("n", "m")), gcdN(mulN("n", "m"), "n"), "n", comm("n", mulN("n", "m")), ap("Nat.gcd_mul_right_left", "m", "n"))),
		thm("Nat.gcd_gcd_self_right_left", "m n", no, eqN(gcdN("m", gcdN("m", "n")), gcdN("m", "n")),
			ap("Nat.dvd_antisymm", gcdN("m", gcdN("m", "n")), gcdN("m", "n"), ap("Nat.gcd_dvd_right", "m", gcdN("m", "n")),
				ap("Nat.dvd_gcd", gcdN("m", "n"), "m", gcdN("m", "n"), ap("Nat.gcd_dvd_left", "m", "n"), ap("Nat.dvd_refl", gcdN("m", "n"))))),
		thm("Nat.gcd_gcd_self_right_right", "m n", no, eqN(gcdN("m", gcdN("n", "m")), gcdN("n", "m")),
			tr(gcdN("m", gcdN("n", "m")), gcdN("m", gcdN("m", "n")), gcdN("n", "m"),
				cg(gm("m"), gcdN("n", "m"), gcdN("m", "n"), comm("n", "m")),
				tr(gcdN("m", gcdN("m", "n")), gcdN("m", "n"), gcdN("n", "m"), ap("Nat.gcd_gcd_self_right_left", "m", "n"), comm("m", "n")))),
		thm("Nat.gcd_gcd_self_left_right", "m n", no, eqN(gcdN(gcdN("n", "m"), "m"), gcdN("n", "m")),
			tr(gcdN(gcdN("n", "m"), "m"), gcdN("m", gcdN("n", "m")), gcdN("n", "m"),
				comm(gcdN("n", "m"), "m"), ap("Nat.gcd_gcd_self_right_right", "m", "n"))),
		thm("Nat.gcd_gcd_self_left_left", "m n", no, eqN(gcdN(gcdN("m", "n"), "m"), gcdN("m", "n")),
			tr(gcdN(gcdN("m", "n"), "m"), gcdN("m", gcdN("m", "n")), gcdN("m", "n"),
				comm(gcdN("m", "n"), "m"), ap("Nat.gcd_gcd_self_right_left", "m", "n"))),
	)
	// n + k*m
	nkm := addN("n", mulN("k", "m"))
	gk := gcdN("m", nkm)
	preludeExtra = append(preludeExtra,
		thm("Nat.gcd_add_mul_right_right", "m n k", no, eqN(gcdN("m", nkm), gcdN("m", "n")),
			ap("Nat.gcd_right_ext", "m", nkm, "n",
				tp(subN(nkm, mulN("k", "m")), "n", ap("Nat.add_sub_cancel", "n", mulN("k", "m")),
					func(z string) string { return dvdN(gk, z) },
					ap("Nat.dvd_sub", gk, nkm, mulN("k", "m"), ap("Nat.gcd_dvd_right", "m", nkm),
						ap("Nat.dvd_trans", gk, "m", mulN("k", "m"), ap("Nat.gcd_dvd_left", "m", nkm), ap("Nat.dvd_mul_left", "m", "k")))),
				ap("Nat.dvd_add", gcdN("m", "n"), "n", mulN("k", "m"), ap("Nat.gcd_dvd_right", "m", "n"),
					ap("Nat.dvd_trans", gcdN("m", "n"), "m", mulN("k", "m"), ap("Nat.gcd_dvd_left", "m", "n"), ap("Nat.dvd_mul_left", "m", "k"))))),
		thm("Nat.gcd_add_mul_left_right", "m n k", no, eqN(gcdN("m", addN("n", mulN("m", "k"))), gcdN("m", "n")),
			tr(gcdN("m", addN("n", mulN("m", "k"))), gk, gcdN("m", "n"),
				cg(func(z string) string { return gcdN("m", addN("n", z)) }, mulN("m", "k"), mulN("k", "m"), ap("Nat.mul_comm", "m", "k")),
				ap("Nat.gcd_add_mul_right_right", "m", "n", "k"))),
		thm("Nat.gcd_mul_right_add_right", "m n k", no, eqN(gcdN("m", addN(mulN("k", "m"), "n")), gcdN("m", "n")),
			tr(gcdN("m", addN(mulN("k", "m"), "n")), gk, gcdN("m", "n"),
				cg(gm("m"), addN(mulN("k", "m"), "n"), nkm, ap("Nat.add_comm", mulN("k", "m"), "n")),
				ap("Nat.gcd_add_mul_right_right", "m", "n", "k"))),
		thm("Nat.gcd_mul_left_add_right", "m n k", no, eqN(gcdN("m", addN(mulN("m", "k"), "n")), gcdN("m", "n")),
			tr(gcdN("m", addN(mulN("m", "k"), "n")), gcdN("m", addN("n", mulN("m", "k"))), gcdN("m", "n"),
				cg(gm("m"), addN(mulN("m", "k"), "n"), addN("n", mulN("m", "k")), ap("Nat.add_comm", mulN("m", "k"), "n")),
				ap("Nat.gcd_add_mul_left_right", "m", "n", "k"))),
	)
	flip := func(name, src, lhs, mid string, args string) valDecl {
		// gcd (X) m = gcd n m from the right-hand form gcd m (X) = gcd m n
		return thm(name, "m n k", no, eqN(gcdN(lhs, "m"), gcdN("n", "m")),
			tr(gcdN(lhs, "m"), gcdN("m", lhs), gcdN("n", "m"), comm(lhs, "m"),
				tr(gcdN("m", lhs), gcdN("m", "n"), gcdN("n", "m"), ap(src, "m", "n", "k"), comm("m", "n"))))
	}
	preludeExtra = append(preludeExtra,
		flip("Nat.gcd_add_mul_right_left", "Nat.gcd_add_mul_right_right", nkm, "", ""),
		flip("Nat.gcd_add_mul_left_left", "Nat.gcd_add_mul_left_right", addN("n", mulN("m", "k")), "", ""),
		flip("Nat.gcd_mul_right_add_left", "Nat.gcd_mul_right_add_right", addN(mulN("k", "m"), "n"), "", ""),
		flip("Nat.gcd_mul_left_add_left", "Nat.gcd_mul_left_add_right", addN(mulN("m", "k"), "n"), "", ""),
	)
	// k = 1
	one := func(z string) string { return z }
	_ = one
	nm := addN("n", "m")
	preludeExtra = append(preludeExtra,
		thm("Nat.gcd_add_self_right", "m n", no, eqN(gcdN("m", nm), gcdN("m", "n")),
			tr(gcdN("m", nm), gcdN("m", addN("n", mulN("1", "m"))), gcdN("m", "n"),
				cg(func(z string) string { return gcdN("m", addN("n", z)) }, "m", mulN("1", "m"), sym(mulN("1", "m"), "m", ap("Nat.one_mul", "m"))),
				ap("Nat.gcd_add_mul_right_right", "m", "n", "1"))),
		thm("Nat.gcd_self_add_right", "m n", no, eqN(gcdN("m", addN("m", "n")), gcdN("m", "n")),
			tr(gcdN("m", addN("m", "n")), gcdN("m", nm), gcdN("m", "n"),
				cg(gm("m"), addN("m", "n"), nm, ap("Nat.add_comm", "m", "n")), ap("Nat.gcd_add_self_right", "m", "n"))),
		thm("Nat.gcd_add_self_left", "m n", no, eqN(gcdN(nm, "m"), gcdN("n", "m")),
			tr(gcdN(nm, "m"), gcdN("m", nm), gcdN("n", "m"), comm(nm, "m"),
				tr(gcdN("m", nm), gcdN("m", "n"), gcdN("n", "m"), ap("Nat.gcd_add_self_right", "m", "n"), comm("m", "n")))),
		thm("Nat.gcd_self_add_left", "m n", no, eqN(gcdN(addN("m", "n"), "m"), gcdN("n", "m")),
			tr(gcdN(addN("m", "n"), "m"), gcdN("m", addN("m", "n")), gcdN("n", "m"), comm(addN("m", "n"), "m"),
				tr(gcdN("m", addN("m", "n")), gcdN("m", "n"), gcdN("n", "m"), ap("Nat.gcd_self_add_right", "m", "n"), comm("m", "n")))),
	)
	// the `_of_dvd` forms: m ∣ k, k = m * l
	hd := func(m, k string) [][2]string { return [][2]string{{"h", dvdN(m, k)}} }
	preludeExtra = append(preludeExtra,
		thm("Nat.gcd_add_left_left_of_dvd", "n m k", hd("m", "k"), eqN(gcdN(addN("k", "n"), "m"), gcdN("n", "m")),
			dvIntro("m", "k", "h", eqN(gcdN(addN("k", "n"), "m"), gcdN("n", "m")), func(l, hl string) string {
				return tp(mulN("m", l), "k", sym("k", mulN("m", l), hl),
					func(z string) string { return eqN(gcdN(addN(z, "n"), "m"), gcdN("n", "m")) },
					ap("Nat.gcd_mul_left_add_left", "m", "n", l))
			})),
		thm("Nat.gcd_add_right_left_of_dvd", "n m k", hd("m", "k"), eqN(gcdN(addN("n", "k"), "m"), gcdN("n", "m")),
			dvIntro("m", "k", "h", eqN(gcdN(addN("n", "k"), "m"), gcdN("n", "m")), func(l, hl string) string {
				return tp(mulN("m", l), "k", sym("k", mulN("m", l), hl),
					func(z string) string { return eqN(gcdN(addN("n", z), "m"), gcdN("n", "m")) },
					ap("Nat.gcd_add_mul_left_left", "m", "n", l))
			})),
		thm("Nat.gcd_add_left_right_of_dvd", "m n k", hd("n", "k"), eqN(gcdN("n", addN("k", "m")), gcdN("n", "m")),
			dvIntro("n", "k", "h", eqN(gcdN("n", addN("k", "m")), gcdN("n", "m")), func(l, hl string) string {
				return tp(mulN("n", l), "k", sym("k", mulN("n", l), hl),
					func(z string) string { return eqN(gcdN("n", addN(z, "m")), gcdN("n", "m")) },
					ap("Nat.gcd_mul_left_add_right", "n", "m", l))
			})),
		thm("Nat.gcd_add_right_right_of_dvd", "m n k", hd("n", "k"), eqN(gcdN("n", addN("m", "k")), gcdN("n", "m")),
			dvIntro("n", "k", "h", eqN(gcdN("n", addN("m", "k")), gcdN("n", "m")), func(l, hl string) string {
				return tp(mulN("n", l), "k", sym("k", mulN("n", l), hl),
					func(z string) string { return eqN(gcdN("n", addN("m", z)), gcdN("n", "m")) },
					ap("Nat.gcd_add_mul_left_right", "n", "m", l))
			})),
	)
}
