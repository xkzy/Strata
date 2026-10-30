package lean

// gcd with truncated subtraction (Init/Data/Nat/Gcd.lean): `gcd m (k*m - n) = gcd m n` is the base case, the other
// forms rewrite `m*k`, `1*m` and the argument order into it.
func init() {
	comm := func(a, b string) string { return ap("Nat.gcd_comm", a, b) }
	gm := func(m string) func(string) string { return func(z string) string { return gcdN(m, z) } }
	km := mulN("k", "m")
	mk := mulN("m", "k")
	x := subN(km, "n")
	g := gcdN("m", x)
	gn := gcdN("m", "n")
	hLe := func(a, b string) [][2]string { return [][2]string{{"h", leN(a, b)}} }
	// h with `a ≤ m*k` turned into `a ≤ k*m`
	swapLe := func(a string) string {
		return tp(mk, km, ap("Nat.mul_comm", "m", "k"), func(z string) string { return leN(a, z) }, "h")
	}
	swapLe2 := func(b string) string { // h : m*k ≤ b → k*m ≤ b
		return tp(mk, km, ap("Nat.mul_comm", "m", "k"), func(z string) string { return leN(z, b) }, "h")
	}
	flipL := func(name, src string, hyps [][2]string, lhs, a string) valDecl {
		// gcd lhs m = gcd a m from `gcd m lhs = gcd m a` (src)
		return thm(name, "m n k", hyps, eqN(gcdN(lhs, "m"), gcdN(a, "m")),
			tr(gcdN(lhs, "m"), gcdN("m", lhs), gcdN(a, "m"), comm(lhs, "m"),
				tr(gcdN("m", lhs), gcdN("m", a), gcdN(a, "m"), ap(src, "m", "n", "k", "h"), comm("m", a))))
	}
	preludeExtra = append(preludeExtra,
		thm("Nat.gcd_mul_right_sub_right", "m n k", hLe("n", km), eqN(g, gn),
			ap("Nat.gcd_right_ext", "m", x, "n",
				tp(subN(addN(x, "n"), x), "n", ap("Nat.add_sub_cancel_left", x, "n"),
					func(z string) string { return dvdN(g, z) },
					ap("Nat.dvd_sub", g, addN(x, "n"), x,
						tp(km, addN(x, "n"), sym(addN(x, "n"), km, ap("Nat.sub_add_cancel", km, "n", "h")),
							func(z string) string { return dvdN(g, z) },
							ap("Nat.dvd_trans", g, "m", km, ap("Nat.gcd_dvd_left", "m", x), ap("Nat.dvd_mul_left", "m", "k"))),
						ap("Nat.gcd_dvd_right", "m", x))),
				ap("Nat.dvd_sub", gn, km, "n",
					ap("Nat.dvd_trans", gn, "m", km, ap("Nat.gcd_dvd_left", "m", "n"), ap("Nat.dvd_mul_left", "m", "k")),
					ap("Nat.gcd_dvd_right", "m", "n")))),
		thm("Nat.gcd_mul_left_sub_right", "m n k", hLe("n", mk), eqN(gcdN("m", subN(mk, "n")), gn),
			tr(gcdN("m", subN(mk, "n")), g, gn,
				cg(func(z string) string { return gcdN("m", subN(z, "n")) }, mk, km, ap("Nat.mul_comm", "m", "k")),
				ap("Nat.gcd_mul_right_sub_right", "m", "n", "k", swapLe("n")))),
		thm("Nat.gcd_sub_mul_right_right", "m n k", hLe(km, "n"), eqN(gcdN("m", subN("n", km)), gn),
			tr(gcdN("m", subN("n", km)), gcdN("m", addN(subN("n", km), km)), gn,
				sym(gcdN("m", addN(subN("n", km), km)), gcdN("m", subN("n", km)), ap("Nat.gcd_add_mul_right_right", "m", subN("n", km), "k")),
				cg(gm("m"), addN(subN("n", km), km), "n", ap("Nat.sub_add_cancel", "n", km, "h")))),
		thm("Nat.gcd_sub_mul_left_right", "m n k", hLe(mk, "n"), eqN(gcdN("m", subN("n", mk)), gn),
			tr(gcdN("m", subN("n", mk)), gcdN("m", subN("n", km)), gn,
				cg(func(z string) string { return gcdN("m", subN("n", z)) }, mk, km, ap("Nat.mul_comm", "m", "k")),
				ap("Nat.gcd_sub_mul_right_right", "m", "n", "k", swapLe2("n")))),
		flipL("Nat.gcd_sub_mul_right_left", "Nat.gcd_sub_mul_right_right", hLe(km, "n"), subN("n", km), "n"),
		flipL("Nat.gcd_sub_mul_left_left", "Nat.gcd_sub_mul_left_right", hLe(mk, "n"), subN("n", mk), "n"),
		flipL("Nat.gcd_mul_right_sub_left", "Nat.gcd_mul_right_sub_right", hLe("n", km), x, "n"),
		flipL("Nat.gcd_mul_left_sub_left", "Nat.gcd_mul_left_sub_right", hLe("n", mk), subN(mk, "n"), "n"),
	)
	// k = 1
	oneM := mulN("1", "m")
	h1 := func(le func(z string) string) string { // transport h along m = 1*m
		return tp("m", oneM, sym(oneM, "m", ap("Nat.one_mul", "m")), le, "h")
	}
	preludeExtra = append(preludeExtra,
		thm("Nat.gcd_sub_self_right", "m n", [][2]string{{"h", leN("m", "n")}}, eqN(gcdN("m", subN("n", "m")), gcdN("m", "n")),
			tr(gcdN("m", subN("n", "m")), gcdN("m", subN("n", oneM)), gcdN("m", "n"),
				cg(func(z string) string { return gcdN("m", subN("n", z)) }, "m", oneM, sym(oneM, "m", ap("Nat.one_mul", "m"))),
				ap("Nat.gcd_sub_mul_right_right", "m", "n", "1", h1(func(z string) string { return leN(z, "n") })))),
		thm("Nat.gcd_self_sub_right", "m n", [][2]string{{"h", leN("n", "m")}}, eqN(gcdN("m", subN("m", "n")), gcdN("m", "n")),
			tr(gcdN("m", subN("m", "n")), gcdN("m", subN(oneM, "n")), gcdN("m", "n"),
				cg(func(z string) string { return gcdN("m", subN(z, "n")) }, "m", oneM, sym(oneM, "m", ap("Nat.one_mul", "m"))),
				ap("Nat.gcd_mul_right_sub_right", "m", "n", "1", h1(func(z string) string { return leN("n", z) })))),
		thm("Nat.gcd_sub_self_left", "m n", [][2]string{{"h", leN("m", "n")}}, eqN(gcdN(subN("n", "m"), "m"), gcdN("n", "m")),
			tr(gcdN(subN("n", "m"), "m"), gcdN("m", subN("n", "m")), gcdN("n", "m"), comm(subN("n", "m"), "m"),
				tr(gcdN("m", subN("n", "m")), gcdN("m", "n"), gcdN("n", "m"), ap("Nat.gcd_sub_self_right", "m", "n", "h"), comm("m", "n")))),
		thm("Nat.gcd_self_sub_left", "m n", [][2]string{{"h", leN("n", "m")}}, eqN(gcdN(subN("m", "n"), "m"), gcdN("n", "m")),
			tr(gcdN(subN("m", "n"), "m"), gcdN("m", subN("m", "n")), gcdN("n", "m"), comm(subN("m", "n"), "m"),
				tr(gcdN("m", subN("m", "n")), gcdN("m", "n"), gcdN("n", "m"), ap("Nat.gcd_self_sub_right", "m", "n", "h"), comm("m", "n")))),
	)
	// the `_of_dvd` forms: k = m*l, so `h` moves along `hl` and the goal along its symmetry
	ofDvd := func(name string, le [2]string, m, k string, goal func(z string) string, src string, srcArgs []string) valDecl {
		hyps := [][2]string{le, {"hd", dvdN(m, k)}}
		return thm(name, "m n k", hyps, goal(k),
			dvIntro(m, k, "hd", goal(k), func(l, hl string) string {
				hl2 := tp(k, mulN(m, l), hl, func(z string) string {
					if le[0] == "h" && le[1] == leN("n", k) {
						return leN("n", z)
					}
					return leN(z, "n")
				}, "h")
				return tp(mulN(m, l), k, sym(k, mulN(m, l), hl), goal, ap(src, append(srcArgs, l, hl2)...))
			}))
	}
	preludeExtra = append(preludeExtra,
		ofDvd("Nat.gcd_sub_left_left_of_dvd", [2]string{"h", leN("n", "k")}, "m", "k",
			func(z string) string { return eqN(gcdN(subN(z, "n"), "m"), gcdN("n", "m")) },
			"Nat.gcd_mul_left_sub_left", []string{"m", "n"}),
		ofDvd("Nat.gcd_sub_right_left_of_dvd", [2]string{"h", leN("k", "n")}, "m", "k",
			func(z string) string { return eqN(gcdN(subN("n", z), "m"), gcdN("n", "m")) },
			"Nat.gcd_sub_mul_left_left", []string{"m", "n"}),
		ofDvd("Nat.gcd_sub_left_right_of_dvd", [2]string{"h", leN("n", "k")}, "m", "k",
			func(z string) string { return eqN(gcdN("m", subN(z, "n")), gcdN("m", "n")) },
			"Nat.gcd_mul_left_sub_right", []string{"m", "n"}),
		ofDvd("Nat.gcd_sub_right_right_of_dvd", [2]string{"h", leN("k", "n")}, "m", "k",
			func(z string) string { return eqN(gcdN("m", subN("n", z)), gcdN("m", "n")) },
			"Nat.gcd_sub_mul_left_right", []string{"m", "n"}),
	)
}
