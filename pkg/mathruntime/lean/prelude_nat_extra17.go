package lean

// Nat.add_mul_div_left and the Nat.mod_mul family (Init/Data/Nat/Basic.lean, Init/Data/Nat/Mod.lean). add_mul_div_left comes from
// mul_add_div_of_lt: x + y*z = y*(x/y + z) + x%y with x%y < y.
func init() {
	md := func(x, y string) string { return "(Nat.mod " + x + " " + y + ")" }
	dv := func(x, y string) string { return "(Nat.div " + x + " " + y + ")" }
	pw := func(a, e string) string { return ap("Nat.pow", a, e) }
	// add_mul_div_left
	{
		a, r := dv("x", "y"), md("x", "y")
		ya := mulN("y", a)
		x1 := addN("x", mulN("y", "t"))
		x2 := addN(addN(r, ya), mulN("y", "t"))
		x3 := addN(r, addN(ya, mulN("y", "t")))
		x4 := addN(addN(ya, mulN("y", "t")), r)
		x5 := addN(mulN("y", addN(a, "t")), r)
		eqS := tr(x1, x3, x5,
			tr(x1, x2, x3, cg(func(w string) string { return addN(w, mulN("y", "t")) }, "x", addN(r, ya), sym(addN(r, ya), "x", ap("Nat.mod_add_div", "x", "y"))),
				ap("Nat.add_assoc", r, ya, mulN("y", "t"))),
			tr(x3, x4, x5, ap("Nat.add_comm", r, addN(ya, mulN("y", "t"))),
				cg(func(w string) string { return addN(w, r) }, addN(ya, mulN("y", "t")), mulN("y", addN(a, "t")),
					sym(mulN("y", addN(a, "t")), addN(ya, mulN("y", "t")), ap("Nat.mul_add", "y", a, "t")))))
		preludeExtra = append(preludeExtra,
			thm("Nat.add_mul_div_left", "x t y", [][2]string{{"H", ltN("0", "y")}}, eqN(dv(x1, "y"), addN(a, "t")),
				tr(dv(x1, "y"), dv(x5, "y"), addN(a, "t"),
					cg(func(w string) string { return dv(w, "y") }, x1, x5, eqS),
					ap("Nat.mul_add_div_of_lt", "y", addN(a, "t"), r, "H", ap("Nat.mod_lt", "x", "y", "H")))),
		)
	}
	// div_lt_of_lt_mul (h : m < n * k) : m / n < k
	{
		q := mulN("n", dv("m", "n"))
		le := tp(addN(md("m", "n"), q), "m", ap("Nat.mod_add_div", "m", "n"), func(z string) string { return leN(q, z) }, ap("Nat.le_add_left", q, md("m", "n")))
		preludeExtra = append(preludeExtra,
			thm("Nat.div_lt_of_lt_mul", "m n k", [][2]string{{"h", ltN("m", mulN("n", "k"))}}, ltN(dv("m", "n"), "k"),
				ap("Nat.lt_of_mul_lt_mul_left", "n", dv("m", "n"), "k", ap("Nat.lt_of_le_of_lt", q, "m", mulN("n", "k"), le, "h"))),
		)
	}
	// mod_mod_of_dvd (h : m ∣ k) : n % k % m = n % m
	{
		nk := md("n", "k")
		goal := eqN(md(nk, "m"), md("n", "m"))
		body := func(l, hl string) string {
			d := dv("n", "k")
			w := mulN(l, d)
			e1 := tr(mulN("k", d), mulN(mulN("m", l), d), mulN("m", w),
				cg(func(z string) string { return mulN(z, d) }, "k", mulN("m", l), hl), ap("Nat.mul_assoc", "m", l, d))
			sA := sym(md(addN(nk, mulN("m", w)), "m"), md(nk, "m"), ap("Nat.add_mul_mod_self_left", nk, "m", w))
			sB := cg(func(z string) string { return md(addN(nk, z), "m") }, mulN("m", w), mulN("k", d), sym(mulN("k", d), mulN("m", w), e1))
			sC := cg(func(z string) string { return md(z, "m") }, addN(nk, mulN("k", d)), "n", ap("Nat.mod_add_div", "n", "k"))
			return tr(md(nk, "m"), md(addN(nk, mulN("m", w)), "m"), md("n", "m"), sA,
				tr(md(addN(nk, mulN("m", w)), "m"), md(addN(nk, mulN("k", d)), "m"), md("n", "m"), sB, sC))
		}
		preludeExtra = append(preludeExtra,
			thm("Nat.mod_mod_of_dvd", "n m k", [][2]string{{"h", dvdN("m", "k")}}, goal, dvIntro("m", "k", "h", goal, body)),
			thm("Nat.mod_mul_right_mod", "a b c", nil, eqN(md(md("a", mulN("b", "c")), "b"), md("a", "b")),
				ap("Nat.mod_mod_of_dvd", "a", "b", mulN("b", "c"), ap("Nat.dvd_mul_right", "b", "c"))),
			thm("Nat.mod_mul_left_mod", "a b c", nil, eqN(md(md("a", mulN("b", "c")), "c"), md("a", "c")),
				ap("Nat.mod_mod_of_dvd", "a", "c", mulN("b", "c"), ap("Nat.dvd_mul_left", "c", "b"))),
		)
	}
	// mod_mul_right_div_self (m n k) : m % (n * k) / n = m / n % k
	{
		nk := mulN("n", "k")
		r, q := md("m", nk), dv("m", nk)
		gl := func(n, k string) string { return eqN(dv(md("m", mulN(n, k)), n), md(dv("m", n), k)) }
		goal := gl("n", "k")
		n0 := tp("0", "n", sym("n", "0", "h0"), func(z string) string { return gl(z, "k") },
			tr(dv(md("m", mulN("0", "k")), "0"), "0", md(dv("m", "0"), "k"), ap("Nat.div_zero", md("m", mulN("0", "k"))),
				sym(md(dv("m", "0"), "k"), "0", tr(md(dv("m", "0"), "k"), md("0", "k"), "0",
					cg(func(z string) string { return md(z, "k") }, dv("m", "0"), "0", ap("Nat.div_zero", "m")), ap("Nat.zero_mod", "k")))))
		k0 := tp("0", "k", sym("k", "0", "h1"), func(z string) string { return gl("n", z) },
			tr(dv(md("m", mulN("n", "0")), "n"), dv("m", "n"), md(dv("m", "n"), "0"),
				cg(func(z string) string { return dv(z, "n") }, md("m", mulN("n", "0")), "m", ap("Nat.mod_zero", "m")),
				sym(md(dv("m", "n"), "0"), dv("m", "n"), ap("Nat.mod_zero", dv("m", "n")))))
		rn := dv(r, "n")
		hlt := ap("Nat.div_lt_of_lt_mul", r, "n", "k", ap("Nat.mod_lt", "m", nk, ap("Nat.mul_pos", "n", "k", "hn", "hk")))
		kq := mulN("k", q)
		e := tr(addN(r, mulN("n", kq)), addN(r, mulN(nk, q)), "m",
			cg(func(z string) string { return addN(r, z) }, mulN("n", kq), mulN(nk, q), sym(mulN(nk, q), mulN("n", kq), ap("Nat.mul_assoc", "n", "k", q))),
			ap("Nat.mod_add_div", "m", nk))
		pos := tr(rn, md(rn, "k"), md(dv("m", "n"), "k"), sym(md(rn, "k"), rn, ap("Nat.mod_eq_of_lt", rn, "k", hlt)),
			tr(md(rn, "k"), md(addN(rn, kq), "k"), md(dv("m", "n"), "k"), sym(md(addN(rn, kq), "k"), md(rn, "k"), ap("Nat.add_mul_mod_self_left", rn, "k", q)),
				tr(md(addN(rn, kq), "k"), md(dv(addN(r, mulN("n", kq)), "n"), "k"), md(dv("m", "n"), "k"),
					cg(func(z string) string { return md(z, "k") }, addN(rn, kq), dv(addN(r, mulN("n", kq)), "n"),
						sym(dv(addN(r, mulN("n", kq)), "n"), addN(rn, kq), ap("Nat.add_mul_div_left", r, kq, "n", "hn"))),
					cg(func(z string) string { return md(dv(z, "n"), "k") }, addN(r, mulN("n", kq)), "m", e))))
		kCase := "(Or.rec " + eqN("k", "0") + " " + ltN("0", "k") + " (fun (_ : (Or " + eqN("k", "0") + " " + ltN("0", "k") + ")) => " + goal + ") " +
			"(fun (h1 : " + eqN("k", "0") + ") => " + k0 + ") (fun (hk : " + ltN("0", "k") + ") => " + pos + ") " + ap("Nat.eq_zero_or_pos", "k") + ")"
		preludeExtra = append(preludeExtra,
			thm("Nat.mod_mul_right_div_self", "m n k", nil, goal,
				"(Or.rec "+eqN("n", "0")+" "+ltN("0", "n")+" (fun (_ : (Or "+eqN("n", "0")+" "+ltN("0", "n")+")) => "+goal+") "+
					"(fun (h0 : "+eqN("n", "0")+") => "+n0+") (fun (hn : "+ltN("0", "n")+") => "+kCase+") "+ap("Nat.eq_zero_or_pos", "n")+")"),
			thm("Nat.mod_mul_left_div_self", "m n k", nil, eqN(dv(md("m", mulN("k", "n")), "n"), md(dv("m", "n"), "k")),
				tr(dv(md("m", mulN("k", "n")), "n"), dv(md("m", mulN("n", "k")), "n"), md(dv("m", "n"), "k"),
					cg(func(z string) string { return dv(md("m", z), "n") }, mulN("k", "n"), mulN("n", "k"), ap("Nat.mul_comm", "k", "n")),
					ap("Nat.mod_mul_right_div_self", "m", "n", "k"))),
		)
	}
	// mod_mul : x % (a * b) = x % a + a * (x / a % b), mod_pow_succ
	{
		ab := mulN("a", "b")
		r := md("x", ab)
		xa := md(dv("x", "a"), "b")
		mid := addN(mulN("a", dv(r, "a")), md(r, "a"))
		fin := addN(mulN("a", xa), md("x", "a"))
		goalRhs := addN(md("x", "a"), mulN("a", xa))
		preludeExtra = append(preludeExtra,
			thm("Nat.mod_mul", "a b x", nil, eqN(r, goalRhs),
				tr(r, mid, goalRhs, sym(mid, r, ap("Nat.div_add_mod", "a", r)),
					tr(mid, fin, goalRhs,
						tr(mid, addN(mulN("a", xa), md(r, "a")), fin,
							cg(func(z string) string { return addN(mulN("a", z), md(r, "a")) }, dv(r, "a"), xa, ap("Nat.mod_mul_right_div_self", "x", "a", "b")),
							cg(func(z string) string { return addN(mulN("a", xa), z) }, md(r, "a"), md("x", "a"), ap("Nat.mod_mul_right_mod", "x", "a", "b"))),
						ap("Nat.add_comm", mulN("a", xa), md("x", "a"))))),
		)
		pb := pw("b", "k")
		rhs := addN(md("x", pb), mulN(pb, md(dv("x", pb), "b")))
		preludeExtra = append(preludeExtra,
			thm("Nat.mod_pow_succ", "x b k", nil, eqN(md("x", pw("b", addN("k", "1"))), rhs),
				tr(md("x", pw("b", addN("k", "1"))), md("x", mulN(pb, "b")), rhs,
					cg(func(z string) string { return md("x", z) }, pw("b", "(Nat.succ k)"), mulN(pb, "b"), ap("Nat.pow_succ", "b", "k")),
					ap("Nat.mod_mul", pb, "b", "x"))),
		)
	}
}
