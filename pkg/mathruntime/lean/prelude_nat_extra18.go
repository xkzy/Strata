package lean

// Init/Data/Nat/Div/Basic.lean: the lemmas that do not unfold `Nat.div` / `Nat.mod` through `ite` (the kernel has no `ite`).
func init() {
	md := func(x, y string) string { return "(Nat.mod " + x + " " + y + ")" }
	dv := func(x, y string) string { return "(Nat.div " + x + " " + y + ")" }
	orZ := func(x string) string { return "(Or " + eqN(x, "0") + " " + ltN("0", x) + ")" }
	refl := func(x string) string { return ap("Nat.le.refl", x) }
	// div_eq_of_lt_le {k n m} (lo : k * n ≤ m) (hi : m < (k + 1) * n) : m / n = k
	{
		sk := "(Nat.succ k)"
		bad := "(fun (h0 : " + eqN("n", "0") + ") => False.elim.{0} " + ltN("0", "n") + " (" + ap("Nat.lt_irrefl", "m") + " " +
			ap("Nat.lt_of_lt_of_le", "m", "0", "m", tp("n", "0", "h0", func(z string) string { return ltN("m", mulN(sk, z)) }, "hi"), ap("Nat.zero_le", "m")) + "))"
		npos := "(Or.rec " + eqN("n", "0") + " " + ltN("0", "n") + " (fun (_ : " + orZ("n") + ") => " + ltN("0", "n") + ") " + bad +
			" (fun (hp : " + ltN("0", "n") + ") => hp) " + ap("Nat.eq_zero_or_pos", "n") + ")"
		preludeExtra = append(preludeExtra,
			thm("Nat.div_eq_of_lt_le", "k n m", [][2]string{{"lo", leN(mulN("k", "n"), "m")}, {"hi", ltN("m", mulN("(Nat.add k 1)", "n"))}}, eqN(dv("m", "n"), "k"),
				"(fun (hp : "+ltN("0", "n")+") => "+ap("Nat.le_antisymm", dv("m", "n"), "k",
					ap("Nat.le_of_lt_succ", dv("m", "n"), "k", iffMpr(ltN(dv("m", "n"), sk), ltN("m", mulN(sk, "n")), ap("Nat.div_lt_iff_lt_mul", "m", sk, "n", "hp"), "hi")),
					iffMpr(leN("k", dv("m", "n")), leN(mulN("k", "n"), "m"), ap("Nat.le_div_iff_mul_le", "k", "m", "n", "hp"), "lo"))+") "+npos),
		)
	}
	// div_eq_of_eq_mul_left / right, add_mul_div_right
	preludeExtra = append(preludeExtra,
		thm("Nat.div_eq_of_eq_mul_left", "n m k", [][2]string{{"H1", ltN("0", "n")}, {"H2", eqN("m", mulN("k", "n"))}}, eqN(dv("m", "n"), "k"),
			tr(dv("m", "n"), dv(mulN("k", "n"), "n"), "k", cg(func(z string) string { return dv(z, "n") }, "m", mulN("k", "n"), "H2"), ap("Nat.mul_div_cancel", "k", "n", "H1"))),
		thm("Nat.div_eq_of_eq_mul_right", "n m k", [][2]string{{"H1", ltN("0", "n")}, {"H2", eqN("m", mulN("n", "k"))}}, eqN(dv("m", "n"), "k"),
			tr(dv("m", "n"), dv(mulN("n", "k"), "n"), "k", cg(func(z string) string { return dv(z, "n") }, "m", mulN("n", "k"), "H2"), ap("Nat.mul_div_cancel_left", "k", "n", "H1"))),
		thm("Nat.add_mul_div_right", "x y t", [][2]string{{"H", ltN("0", "t")}}, eqN(dv(addN("x", mulN("y", "t")), "t"), addN(dv("x", "t"), "y")),
			tr(dv(addN("x", mulN("y", "t")), "t"), dv(addN("x", mulN("t", "y")), "t"), addN(dv("x", "t"), "y"),
				cg(func(w string) string { return dv(addN("x", w), "t") }, mulN("y", "t"), mulN("t", "y"), ap("Nat.mul_comm", "y", "t")),
				ap("Nat.add_mul_div_left", "x", "y", "t", "H"))),
		thm("Nat.mod_lt_of_lt", "a b c", [][2]string{{"h", ltN("a", "c")}}, ltN(md("a", "b"), "c"),
			ap("Nat.lt_of_le_of_lt", md("a", "b"), "a", "c", ap("Nat.mod_le", "a", "b"), "h")),
	)
	// mod_def (m k) : m % k = m - k * (m / k), and its two spellings
	{
		q := mulN("k", dv("m", "k"))
		def := tr(md("m", "k"), subN(addN(md("m", "k"), q), q), subN("m", q),
			sym(subN(addN(md("m", "k"), q), q), md("m", "k"), ap("Nat.add_sub_cancel", md("m", "k"), q)),
			cg(func(z string) string { return subN(z, q) }, addN(md("m", "k"), q), "m", ap("Nat.mod_add_div", "m", "k")))
		preludeExtra = append(preludeExtra,
			thm("Nat.mod_def", "m k", nil, eqN(md("m", "k"), subN("m", q)), def),
			thm("Nat.mod_eq_sub_mul_div", "x k", nil, eqN(md("x", "k"), subN("x", mulN("k", dv("x", "k")))), ap("Nat.mod_def", "x", "k")),
			thm("Nat.mod_eq_sub_div_mul", "x k", nil, eqN(md("x", "k"), subN("x", mulN(dv("x", "k"), "k"))),
				tr(md("x", "k"), subN("x", mulN("k", dv("x", "k"))), subN("x", mulN(dv("x", "k"), "k")), ap("Nat.mod_def", "x", "k"),
					cg(func(z string) string { return subN("x", z) }, mulN("k", dv("x", "k")), mulN(dv("x", "k"), "k"), ap("Nat.mul_comm", "k", dv("x", "k"))))),
		)
	}
	// pos_of_div_pos {a b} (h : 0 < a / b) : 0 < a
	preludeExtra = append(preludeExtra,
		thm("Nat.pos_of_div_pos", "a b", [][2]string{{"h", ltN("0", dv("a", "b"))}}, ltN("0", "a"),
			"(Or.rec "+eqN("a", "0")+" "+ltN("0", "a")+" (fun (_ : "+orZ("a")+") => "+ltN("0", "a")+") "+
				"(fun (h0 : "+eqN("a", "0")+") => False.elim.{0} "+ltN("0", "a")+" ("+ap("Nat.lt_irrefl", "0")+" "+
				tp(dv("0", "b"), "0", ap("Nat.zero_div", "b"), func(z string) string { return ltN("0", z) },
					tp("a", "0", "h0", func(z string) string { return ltN("0", dv(z, "b")) }, "h"))+")) (fun (hp : "+ltN("0", "a")+") => hp) "+ap("Nat.eq_zero_or_pos", "a")+")"),
	)
	// div_div_eq_div_mul (m n k) : m / n / k = m / (n * k)
	{
		nk := mulN("n", "k")
		a, b := dv("m", nk), dv(dv("m", "n"), "k")
		hnk := ap("Nat.mul_pos", "n", "k", "hn", "hk")
		sw := func(x string) string { // x * (n * k) = (x * k) * n
			return tr(mulN(x, nk), mulN(x, mulN("k", "n")), mulN(mulN(x, "k"), "n"),
				cg(func(z string) string { return mulN(x, z) }, nk, mulN("k", "n"), ap("Nat.mul_comm", "n", "k")),
				sym(mulN(mulN(x, "k"), "n"), mulN(x, mulN("k", "n")), ap("Nat.mul_assoc", x, "k", "n")))
		}
		// a ≤ b
		aLe := iffMpr(leN(a, b), leN(mulN(a, "k"), dv("m", "n")), ap("Nat.le_div_iff_mul_le", a, dv("m", "n"), "k", "hk"),
			iffMpr(leN(mulN(a, "k"), dv("m", "n")), leN(mulN(mulN(a, "k"), "n"), "m"), ap("Nat.le_div_iff_mul_le", mulN(a, "k"), "m", "n", "hn"),
				tp(mulN(a, nk), mulN(mulN(a, "k"), "n"), sw(a), func(z string) string { return leN(z, "m") },
					iffMp(leN(a, a), leN(mulN(a, nk), "m"), ap("Nat.le_div_iff_mul_le", a, "m", nk, hnk), refl(a)))))
		// b ≤ a
		bk := iffMp(leN(b, b), leN(mulN(b, "k"), dv("m", "n")), ap("Nat.le_div_iff_mul_le", b, dv("m", "n"), "k", "hk"), refl(b))
		mn := iffMp(leN(dv("m", "n"), dv("m", "n")), leN(mulN(dv("m", "n"), "n"), "m"), ap("Nat.le_div_iff_mul_le", dv("m", "n"), "m", "n", "hn"), refl(dv("m", "n")))
		bnk := ap("Nat.le_trans", mulN(mulN(b, "k"), "n"), mulN(dv("m", "n"), "n"), "m", ap("Nat.mul_le_mul_right", "n", mulN(b, "k"), dv("m", "n"), bk), mn)
		bLe := iffMpr(leN(b, a), leN(mulN(b, nk), "m"), ap("Nat.le_div_iff_mul_le", b, "m", nk, hnk),
			tp(mulN(mulN(b, "k"), "n"), mulN(b, nk), sym(mulN(b, nk), mulN(mulN(b, "k"), "n"), sw(b)), func(z string) string { return leN(z, "m") }, bnk))
		pos := ap("Nat.le_antisymm", b, a, bLe, aLe)
		g := eqN(b, a)
		zk := tp("0", "k", sym("k", "0", "h1"), func(z string) string { return eqN(dv(dv("m", "n"), z), dv("m", mulN("n", z))) },
			tr(dv(dv("m", "n"), "0"), "0", dv("m", mulN("n", "0")), ap("Nat.div_zero", dv("m", "n")), sym(dv("m", mulN("n", "0")), "0", ap("Nat.div_zero", "m"))))
		zn := tp("0", "n", sym("n", "0", "h0"), func(z string) string { return eqN(dv(dv("m", z), "k"), dv("m", mulN(z, "k"))) },
			tr(dv(dv("m", "0"), "k"), "0", dv("m", mulN("0", "k")),
				tr(dv(dv("m", "0"), "k"), dv("0", "k"), "0", cg(func(z string) string { return dv(z, "k") }, dv("m", "0"), "0", ap("Nat.div_zero", "m")), ap("Nat.zero_div", "k")),
				sym(dv("m", mulN("0", "k")), "0", tr(dv("m", mulN("0", "k")), dv("m", "0"), "0", cg(func(z string) string { return dv("m", z) }, mulN("0", "k"), "0", ap("Nat.zero_mul", "k")), ap("Nat.div_zero", "m")))))
		inner := "(Or.rec " + eqN("n", "0") + " " + ltN("0", "n") + " (fun (_ : " + orZ("n") + ") => " + g + ") (fun (h0 : " + eqN("n", "0") + ") => " + zn + ") (fun (hn : " + ltN("0", "n") + ") => " + pos + ") " + ap("Nat.eq_zero_or_pos", "n") + ")"
		preludeExtra = append(preludeExtra,
			thm("Nat.div_div_eq_div_mul", "m n k", nil, g,
				"(Or.rec "+eqN("k", "0")+" "+ltN("0", "k")+" (fun (_ : "+orZ("k")+") => "+g+") (fun (h1 : "+eqN("k", "0")+") => "+zk+") (fun (hk : "+ltN("0", "k")+") => "+inner+") "+ap("Nat.eq_zero_or_pos", "k")+")"),
		)
	}
	// mul_div_mul_left / right
	preludeExtra = append(preludeExtra,
		thm("Nat.mul_div_mul_left", "m n k", [][2]string{{"H", ltN("0", "m")}}, eqN(dv(mulN("m", "n"), mulN("m", "k")), dv("n", "k")),
			tr(dv(mulN("m", "n"), mulN("m", "k")), dv(dv(mulN("m", "n"), "m"), "k"), dv("n", "k"),
				sym(dv(dv(mulN("m", "n"), "m"), "k"), dv(mulN("m", "n"), mulN("m", "k")), ap("Nat.div_div_eq_div_mul", mulN("m", "n"), "m", "k")),
				cg(func(z string) string { return dv(z, "k") }, dv(mulN("m", "n"), "m"), "n", ap("Nat.mul_div_cancel_left", "n", "m", "H")))),
		thm("Nat.mul_div_mul_right", "m n k", [][2]string{{"H", ltN("0", "m")}}, eqN(dv(mulN("n", "m"), mulN("k", "m")), dv("n", "k")),
			tr(dv(mulN("n", "m"), mulN("k", "m")), dv(mulN("m", "n"), mulN("k", "m")), dv("n", "k"),
				cg(func(z string) string { return dv(z, mulN("k", "m")) }, mulN("n", "m"), mulN("m", "n"), ap("Nat.mul_comm", "n", "m")),
				tr(dv(mulN("m", "n"), mulN("k", "m")), dv(mulN("m", "n"), mulN("m", "k")), dv("n", "k"),
					cg(func(z string) string { return dv(mulN("m", "n"), z) }, mulN("k", "m"), mulN("m", "k"), ap("Nat.mul_comm", "k", "m")),
					ap("Nat.mul_div_mul_left", "m", "n", "k", "H")))),
	)
}
