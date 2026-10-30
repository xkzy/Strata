package lean

// Int gcd with sums and differences: every variant is the core `gcd m (n + k*m) = gcd m n` plus a rewrite of the argument.
func init() {
	one := io()
	mc := func(a, b string) string { return ap("Int.mul_comm", a, b) }
	ac := func(a, b string) string { return ap("Int.add_comm", a, b) }
	g := igcd
	comm := func(a, b string) string { return ap("Int.gcd_comm", a, b) }
	// gcd m X = R from eq : X = Y and tail : gcd m Y = R
	rw := func(mm, X, Y, eq, R, tail string) string {
		return tr(g(mm, X), g(mm, Y), R, icgN(func(z string) string { return g(mm, z) }, X, Y, eq), tail)
	}
	// gcd X m = R from the right-slot proof rp : gcd m X = gcd m n2 (R = gcd n2 m)
	lf := func(X, mm, n2, rp string) string {
		return tr(g(X, mm), g(mm, X), g(n2, mm), comm(X, mm), tr(g(mm, X), g(mm, n2), g(n2, mm), rp, comm(mm, n2)))
	}
	km := imul("k", "m")
	mk := imul("m", "k")
	// ---- sums
	preludeExtra = append(preludeExtra,
		ithm("Int.gcd_add_mul_left_right", "m n k", nil, eqN(g("m", iaddN("n", mk)), g("m", "n")),
			rw("m", iaddN("n", mk), iaddN("n", km), icg(func(z string) string { return iaddN("n", z) }, mk, km, mc("m", "k")), g("m", "n"),
				ap("Int.gcd_add_mul_right_right", "m", "n", "k"))),
		ithm("Int.gcd_mul_right_add_right", "m n k", nil, eqN(g("m", iaddN(km, "n")), g("m", "n")),
			rw("m", iaddN(km, "n"), iaddN("n", km), ac(km, "n"), g("m", "n"), ap("Int.gcd_add_mul_right_right", "m", "n", "k"))),
		ithm("Int.gcd_mul_left_add_right", "m n k", nil, eqN(g("m", iaddN(mk, "n")), g("m", "n")),
			rw("m", iaddN(mk, "n"), iaddN("n", mk), ac(mk, "n"), g("m", "n"), ap("Int.gcd_add_mul_left_right", "m", "n", "k"))),
		ithm("Int.gcd_add_mul_right_left", "m n k", nil, eqN(g(iaddN("n", km), "m"), g("n", "m")),
			lf(iaddN("n", km), "m", "n", ap("Int.gcd_add_mul_right_right", "m", "n", "k"))),
		ithm("Int.gcd_add_mul_left_left", "m n k", nil, eqN(g(iaddN("n", mk), "m"), g("n", "m")),
			lf(iaddN("n", mk), "m", "n", ap("Int.gcd_add_mul_left_right", "m", "n", "k"))),
		ithm("Int.gcd_mul_right_add_left", "m n k", nil, eqN(g(iaddN(km, "n"), "m"), g("n", "m")),
			lf(iaddN(km, "n"), "m", "n", ap("Int.gcd_mul_right_add_right", "m", "n", "k"))),
		ithm("Int.gcd_mul_left_add_left", "m n k", nil, eqN(g(iaddN(mk, "n"), "m"), g("n", "m")),
			lf(iaddN(mk, "n"), "m", "n", ap("Int.gcd_mul_left_add_right", "m", "n", "k"))),
	)
	oneM := imul(one, "m")
	o1 := ap("Int.one_mul", "m")
	preludeExtra = append(preludeExtra,
		ithm("Int.gcd_add_self_right", "m n", nil, eqN(g("m", iaddN("n", "m")), g("m", "n")),
			rw("m", iaddN("n", "m"), iaddN("n", oneM), icg(func(z string) string { return iaddN("n", z) }, "m", oneM, isym(oneM, "m", o1)), g("m", "n"),
				ap("Int.gcd_add_mul_right_right", "m", "n", one))),
		ithm("Int.gcd_self_add_right", "m n", nil, eqN(g("m", iaddN("m", "n")), g("m", "n")),
			rw("m", iaddN("m", "n"), iaddN(oneM, "n"), icg(func(z string) string { return iaddN(z, "n") }, "m", oneM, isym(oneM, "m", o1)), g("m", "n"),
				ap("Int.gcd_mul_right_add_right", "m", "n", one))),
		ithm("Int.gcd_add_self_left", "m n", nil, eqN(g(iaddN("n", "m"), "m"), g("n", "m")),
			lf(iaddN("n", "m"), "m", "n", ap("Int.gcd_add_self_right", "m", "n"))),
		ithm("Int.gcd_self_add_left", "m n", nil, eqN(g(iaddN("m", "n"), "m"), g("n", "m")),
			lf(iaddN("m", "n"), "m", "n", ap("Int.gcd_self_add_right", "m", "n"))),
	)
	// ---- differences
	negk := ineg("k")
	nm := ap("Int.neg_mul", "k", "m")
	subEq := func(x, y string) string { return ap("Int.sub_eq_add_neg", x, y) }
	preludeExtra = append(preludeExtra,
		ithm("Int.gcd_sub_mul_right_right", "m n k", nil, eqN(g("m", isubN("n", km)), g("m", "n")),
			rw("m", isubN("n", km), iaddN("n", imul(negk, "m")),
				itr(isubN("n", km), iaddN("n", ineg(km)), iaddN("n", imul(negk, "m")), subEq("n", km),
					icg(func(z string) string { return iaddN("n", z) }, ineg(km), imul(negk, "m"), isym(imul(negk, "m"), ineg(km), nm))),
				g("m", "n"), ap("Int.gcd_add_mul_right_right", "m", "n", negk))),
		ithm("Int.gcd_sub_mul_left_right", "m n k", nil, eqN(g("m", isubN("n", mk)), g("m", "n")),
			rw("m", isubN("n", mk), isubN("n", km), icg(func(z string) string { return isubN("n", z) }, mk, km, mc("m", "k")), g("m", "n"),
				ap("Int.gcd_sub_mul_right_right", "m", "n", "k"))),
		ithm("Int.gcd_mul_right_sub_right", "m n k", nil, eqN(g("m", isubN(km, "n")), g("m", "n")),
			rw("m", isubN(km, "n"), iaddN(km, ineg("n")), subEq(km, "n"), g("m", "n"),
				tr(g("m", iaddN(km, ineg("n"))), g("m", ineg("n")), g("m", "n"), ap("Int.gcd_mul_right_add_right", "m", ineg("n"), "k"), ap("Int.gcd_neg_right", "m", "n")))),
		ithm("Int.gcd_mul_left_sub_right", "m n k", nil, eqN(g("m", isubN(mk, "n")), g("m", "n")),
			rw("m", isubN(mk, "n"), iaddN(mk, ineg("n")), subEq(mk, "n"), g("m", "n"),
				tr(g("m", iaddN(mk, ineg("n"))), g("m", ineg("n")), g("m", "n"), ap("Int.gcd_mul_left_add_right", "m", ineg("n"), "k"), ap("Int.gcd_neg_right", "m", "n")))),
		ithm("Int.gcd_sub_mul_right_left", "m n k", nil, eqN(g(isubN("n", km), "m"), g("n", "m")),
			lf(isubN("n", km), "m", "n", ap("Int.gcd_sub_mul_right_right", "m", "n", "k"))),
		ithm("Int.gcd_sub_mul_left_left", "m n k", nil, eqN(g(isubN("n", mk), "m"), g("n", "m")),
			lf(isubN("n", mk), "m", "n", ap("Int.gcd_sub_mul_left_right", "m", "n", "k"))),
		ithm("Int.gcd_mul_right_sub_left", "m n k", nil, eqN(g(isubN(km, "n"), "m"), g("n", "m")),
			lf(isubN(km, "n"), "m", "n", ap("Int.gcd_mul_right_sub_right", "m", "n", "k"))),
		ithm("Int.gcd_mul_left_sub_left", "m n k", nil, eqN(g(isubN(mk, "n"), "m"), g("n", "m")),
			lf(isubN(mk, "n"), "m", "n", ap("Int.gcd_mul_left_sub_right", "m", "n", "k"))),
		ithm("Int.gcd_sub_self_right", "m n", nil, eqN(g("m", isubN("n", "m")), g("m", "n")),
			rw("m", isubN("n", "m"), isubN("n", oneM), icg(func(z string) string { return isubN("n", z) }, "m", oneM, isym(oneM, "m", o1)), g("m", "n"),
				ap("Int.gcd_sub_mul_right_right", "m", "n", one))),
		ithm("Int.gcd_self_sub_right", "m n", nil, eqN(g("m", isubN("m", "n")), g("m", "n")),
			rw("m", isubN("m", "n"), isubN(oneM, "n"), icg(func(z string) string { return isubN(z, "n") }, "m", oneM, isym(oneM, "m", o1)), g("m", "n"),
				ap("Int.gcd_mul_right_sub_right", "m", "n", one))),
		ithm("Int.gcd_sub_self_left", "m n", nil, eqN(g(isubN("n", "m"), "m"), g("n", "m")),
			lf(isubN("n", "m"), "m", "n", ap("Int.gcd_sub_self_right", "m", "n"))),
		ithm("Int.gcd_self_sub_left", "m n", nil, eqN(g(isubN("m", "n"), "m"), g("n", "m")),
			lf(isubN("m", "n"), "m", "n", ap("Int.gcd_self_sub_right", "m", "n"))),
	)
	// ---- _of_dvd: k = m * l turns the sum / difference into one of the lemmas above
	ofDvd := func(name, vars, hypA, hypB string, goalOf func(z string) string, use string, args func(l string) []string) valDecl {
		h := [][2]string{{"h", idvd(hypA, hypB)}}
		goal := goalOf(hypB)
		return ithm(name, vars, h, goal,
			iDvIntro(hypA, hypB, "h", goal, func(l, hl string) string {
				return tpI(imul(hypA, l), hypB, isym(hypB, imul(hypA, l), hl), func(z string) string { return goalOf(z) }, ap(use, args(l)...))
			}))
	}
	gl := func(op func(a, b string) string, pos int) func(z string) string { // gcd (k op n) m / gcd (n op k) m
		return func(z string) string {
			if pos == 0 {
				return eqN(g(op(z, "n"), "m"), g("n", "m"))
			}
			return eqN(g(op("n", z), "m"), g("n", "m"))
		}
	}
	gr := func(op func(a, b string) string, pos int) func(z string) string { // gcd m (k op n) etc, here with divisor named m and free arg n
		return func(z string) string {
			if pos == 0 {
				return eqN(g("m", op(z, "n")), g("m", "n"))
			}
			return eqN(g("m", op("n", z)), g("m", "n"))
		}
	}
	preludeExtra = append(preludeExtra,
		ofDvd("Int.gcd_add_left_left_of_dvd", "n m k", "m", "k", gl(iaddN, 0), "Int.gcd_mul_left_add_left", func(l string) []string { return []string{"m", "n", l} }),
		ofDvd("Int.gcd_add_right_left_of_dvd", "n m k", "m", "k", gl(iaddN, 1), "Int.gcd_add_mul_left_left", func(l string) []string { return []string{"m", "n", l} }),
		ofDvd("Int.gcd_add_left_right_of_dvd", "n m k", "m", "k", gr(iaddN, 0), "Int.gcd_mul_left_add_right", func(l string) []string { return []string{"m", "n", l} }),
		ofDvd("Int.gcd_add_right_right_of_dvd", "n m k", "m", "k", gr(iaddN, 1), "Int.gcd_add_mul_left_right", func(l string) []string { return []string{"m", "n", l} }),
		ofDvd("Int.gcd_sub_left_left_of_dvd", "n m k", "m", "k", gl(isubN, 0), "Int.gcd_mul_left_sub_left", func(l string) []string { return []string{"m", "n", l} }),
		ofDvd("Int.gcd_sub_right_left_of_dvd", "n m k", "m", "k", gl(isubN, 1), "Int.gcd_sub_mul_left_left", func(l string) []string { return []string{"m", "n", l} }),
		ofDvd("Int.gcd_sub_left_right_of_dvd", "n m k", "m", "k", gr(isubN, 0), "Int.gcd_mul_left_sub_right", func(l string) []string { return []string{"m", "n", l} }),
		ofDvd("Int.gcd_sub_right_right_of_dvd", "n m k", "m", "k", gr(isubN, 1), "Int.gcd_sub_mul_left_right", func(l string) []string { return []string{"m", "n", l} }),
	)
}
