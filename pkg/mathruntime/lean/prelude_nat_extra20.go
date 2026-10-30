package lean

// More of Init/Data/Nat/{Div/Basic,Dvd,Mod}.lean built on add_mul_div_left / add_mul_mod_self_left. Variables named `t` where Lean says
// `z`, because cg/tp bind a variable named z.
func init() {
	md := func(x, y string) string { return "(Nat.mod " + x + " " + y + ")" }
	dv := func(x, y string) string { return "(Nat.div " + x + " " + y + ")" }
	hp := [][2]string{{"H", ltN("0", "t")}}
	// t*1 = t inside a division / modulus
	tOne := sym(mulN("t", "1"), "t", ap("Nat.mul_one", "t"))
	ac := func(a, b string) string { return ap("Nat.add_comm", a, b) }
	xt := addN("x", "t")
	preludeExtra = append(preludeExtra,
		thm("Nat.add_div_right", "x t", hp, eqN(dv(xt, "t"), addN(dv("x", "t"), "1")),
			tr(dv(xt, "t"), dv(addN("x", mulN("t", "1")), "t"), addN(dv("x", "t"), "1"),
				cg(func(w string) string { return dv(addN("x", w), "t") }, "t", mulN("t", "1"), tOne), ap("Nat.add_mul_div_left", "x", "1", "t", "H"))),
		thm("Nat.add_div_left", "x t", hp, eqN(dv(addN("t", "x"), "t"), addN(dv("x", "t"), "1")),
			tr(dv(addN("t", "x"), "t"), dv(xt, "t"), addN(dv("x", "t"), "1"),
				cg(func(w string) string { return dv(w, "t") }, addN("t", "x"), xt, ac("t", "x")), ap("Nat.add_div_right", "x", "t", "H"))),
		thm("Nat.add_mod_right", "x t", nil, eqN(md(xt, "t"), md("x", "t")),
			tr(md(xt, "t"), md(addN("x", mulN("t", "1")), "t"), md("x", "t"),
				cg(func(w string) string { return md(addN("x", w), "t") }, "t", mulN("t", "1"), tOne), ap("Nat.add_mul_mod_self_left", "x", "t", "1"))),
		thm("Nat.add_mod_left", "x t", nil, eqN(md(xt, "x"), md("t", "x")),
			tr(md(xt, "x"), md(addN("t", "x"), "x"), md("t", "x"),
				cg(func(w string) string { return md(w, "x") }, xt, addN("t", "x"), ac("x", "t")), ap("Nat.add_mod_right", "t", "x"))),
		thm("Nat.mul_add_mod_self_left", "a b c", nil, eqN(md(addN(mulN("a", "b"), "c"), "a"), md("c", "a")),
			tr(md(addN(mulN("a", "b"), "c"), "a"), md(addN("c", mulN("a", "b")), "a"), md("c", "a"),
				cg(func(w string) string { return md(w, "a") }, addN(mulN("a", "b"), "c"), addN("c", mulN("a", "b")), ac(mulN("a", "b"), "c")),
				ap("Nat.add_mul_mod_self_left", "c", "a", "b"))),
		thm("Nat.add_mul_mod_self_right", "x y t", nil, eqN(md(addN("x", mulN("y", "t")), "t"), md("x", "t")),
			tr(md(addN("x", mulN("y", "t")), "t"), md(addN("x", mulN("t", "y")), "t"), md("x", "t"),
				cg(func(w string) string { return md(addN("x", w), "t") }, mulN("y", "t"), mulN("t", "y"), ap("Nat.mul_comm", "y", "t")),
				ap("Nat.add_mul_mod_self_left", "x", "t", "y"))),
		thm("Nat.mul_add_mod_self_right", "a b c", nil, eqN(md(addN(mulN("a", "b"), "c"), "b"), md("c", "b")),
			tr(md(addN(mulN("a", "b"), "c"), "b"), md(addN("c", mulN("a", "b")), "b"), md("c", "b"),
				cg(func(w string) string { return md(w, "b") }, addN(mulN("a", "b"), "c"), addN("c", mulN("a", "b")), ac(mulN("a", "b"), "c")),
				ap("Nat.add_mul_mod_self_right", "c", "a", "b"))),
		thm("Nat.mul_div_right", "n m", [][2]string{{"H", ltN("0", "m")}}, eqN(dv(mulN("m", "n"), "m"), "n"), ap("Nat.mul_div_cancel_left", "n", "m", "H")),
		thm("Nat.mul_div_left", "m n", [][2]string{{"H", ltN("0", "n")}}, eqN(dv(mulN("m", "n"), "n"), "m"), ap("Nat.mul_div_cancel", "m", "n", "H")),
	)
	// a = (a - b) + b, for b ≤ a
	back := func(a, b string) string { return sym(addN(subN(a, b), b), a, ap("Nat.sub_add_cancel", a, b, "h")) }
	preludeExtra = append(preludeExtra,
		thm("Nat.mod_eq_sub_mod", "a b", [][2]string{{"h", leN("b", "a")}}, eqN(md("a", "b"), md(subN("a", "b"), "b")),
			tr(md("a", "b"), md(addN(subN("a", "b"), "b"), "b"), md(subN("a", "b"), "b"),
				cg(func(w string) string { return md(w, "b") }, "a", addN(subN("a", "b"), "b"), back("a", "b")),
				ap("Nat.add_mod_right", subN("a", "b"), "b"))),
		thm("Nat.div_eq_sub_div", "a b", [][2]string{{"h1", ltN("0", "b")}, {"h", leN("b", "a")}}, eqN(dv("a", "b"), addN(dv(subN("a", "b"), "b"), "1")),
			tr(dv("a", "b"), dv(addN(subN("a", "b"), "b"), "b"), addN(dv(subN("a", "b"), "b"), "1"),
				cg(func(w string) string { return dv(w, "b") }, "a", addN(subN("a", "b"), "b"), back("a", "b")),
				ap("Nat.add_div_right", subN("a", "b"), "b", "h1"))),
	)
	// zero_dvd, dvd_one, mul_lt_mul_left / right
	preludeExtra = append(preludeExtra,
		thm("Nat.zero_dvd", "n", nil, iffN(dvdN("0", "n"), eqN("n", "0")),
			iffI(dvdN("0", "n"), eqN("n", "0"), "h", ap("Nat.eq_zero_of_zero_dvd", "n", "h"),
				"h", tp("0", "n", sym("n", "0", "h"), func(z string) string { return dvdN("0", z) }, ap("Nat.dvd_zero", "0")))),
		thm("Nat.dvd_one", "n", nil, iffN(dvdN("n", "1"), eqN("n", "1")),
			iffI(dvdN("n", "1"), eqN("n", "1"), "h", ap("Nat.eq_one_of_dvd_one", "n", "h"),
				"h", tp("1", "n", sym("n", "1", "h"), func(z string) string { return dvdN(z, "1") }, ap("Nat.dvd_refl", "1")))),
		thm("Nat.mul_lt_mul_left", "a b c", [][2]string{{"a0", ltN("0", "a")}}, iffN(ltN(mulN("a", "b"), mulN("a", "c")), ltN("b", "c")),
			iffI(ltN(mulN("a", "b"), mulN("a", "c")), ltN("b", "c"), "h", ap("Nat.lt_of_mul_lt_mul_left", "a", "b", "c", "h"),
				"h", ap("Nat.mul_lt_mul_of_pos_left", "a", "b", "c", "h", "a0"))),
		thm("Nat.mul_lt_mul_right", "a b c", [][2]string{{"a0", ltN("0", "a")}}, iffN(ltN(mulN("b", "a"), mulN("c", "a")), ltN("b", "c")),
			iffI(ltN(mulN("b", "a"), mulN("c", "a")), ltN("b", "c"), "h",
				ap("Nat.lt_of_mul_lt_mul_left", "a", "b", "c", tp(mulN("b", "a"), mulN("a", "b"), ap("Nat.mul_comm", "b", "a"), func(z string) string { return ltN(z, mulN("a", "c")) },
					tp(mulN("c", "a"), mulN("a", "c"), ap("Nat.mul_comm", "c", "a"), func(z string) string { return ltN(mulN("b", "a"), z) }, "h"))),
				"h", tp(mulN("a", "b"), mulN("b", "a"), ap("Nat.mul_comm", "a", "b"), func(z string) string { return ltN(z, mulN("c", "a")) },
					tp(mulN("a", "c"), mulN("c", "a"), ap("Nat.mul_comm", "a", "c"), func(z string) string { return ltN(mulN("a", "b"), z) },
						ap("Nat.mul_lt_mul_of_pos_left", "a", "b", "c", "h", "a0"))))),
	)
}
