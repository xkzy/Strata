package lean

// Init/Data/Nat/MinMax.lean. Nat.min / Nat.max here are Bool.rec on `Nat.ble a b`, so min_eq_left and its three twins
// are cases on that Bool; the rest follow from them and le_total, as in Lean.
func init() {
	mn := func(a, b string) string { return ap("Nat.min", a, b) }
	mx := func(a, b string) string { return ap("Nat.max", a, b) }
	ble := func(a, b string) string { return ap("Nat.ble", a, b) }
	br := func(f, t, x string) string {
		return "(Bool.rec.{1} (fun (_ : Bool) => Nat) " + f + " " + t + " " + x + ")"
	}
	// transport over Bool: body : P a to P b along p : a = b
	tpB := func(a, b, p string, P func(x string) string, body string) string {
		return "(Eq.rec.{0, 1} Bool " + a + " (fun (x : Bool) (_ : Eq.{1} Bool " + a + " x) => " + P("x") + ") " + body + " " + b + " " + p + ")"
	}
	symB := func(a, b, p string) string { return "(Eq.symm.{1} Bool " + a + " " + b + " " + p + ")" }
	// cases on `ble a b`: pf for false, pt for true (each may mention hf / ht)
	bc := func(a, b string, G func(x string) string, pf, pt string) string {
		return "(Bool.rec.{0} (fun (x : Bool) => Eq.{1} Bool " + ble(a, b) + " x → " + G("x") + ") (fun (hf : Eq.{1} Bool " + ble(a, b) + " Bool.false) => " + pf +
			") (fun (ht : Eq.{1} Bool " + ble(a, b) + " Bool.true) => " + pt + ") " + ble(a, b) + " (Eq.refl.{1} Bool " + ble(a, b) + "))"
	}
	hLe := [][2]string{{"h", leN("a", "b")}}
	hGe := [][2]string{{"h", leN("b", "a")}}
	fromLe := func(a, b string) string { return ap("Nat.le_of_ble_eq_true", a, b, "ht") }
	preludeExtra = append(preludeExtra,
		thm("Nat.min_eq_left", "a b", hLe, eqN(mn("a", "b"), "a"),
			tpB("Bool.true", ble("a", "b"), symB(ble("a", "b"), "Bool.true", ap("Nat.ble_true_of_le", "a", "b", "h")),
				func(x string) string { return eqN(br("b", "a", x), "a") }, "(Eq.refl.{1} Nat a)")),
		thm("Nat.min_eq_right", "a b", hGe, eqN(mn("a", "b"), "b"),
			bc("a", "b", func(x string) string { return eqN(br("b", "a", x), "b") }, "(Eq.refl.{1} Nat b)",
				"("+ap("Nat.le_antisymm", "a", "b", fromLe("a", "b"), "h")+")")),
		thm("Nat.max_eq_right", "a b", hLe, eqN(mx("a", "b"), "b"),
			tpB("Bool.true", ble("a", "b"), symB(ble("a", "b"), "Bool.true", ap("Nat.ble_true_of_le", "a", "b", "h")),
				func(x string) string { return eqN(br("a", "b", x), "b") }, "(Eq.refl.{1} Nat b)")),
		thm("Nat.max_eq_left", "a b", hGe, eqN(mx("a", "b"), "a"),
			bc("a", "b", func(x string) string { return eqN(br("a", "b", x), "a") }, "(Eq.refl.{1} Nat a)",
				"("+ap("Nat.le_antisymm", "b", "a", "h", fromLe("a", "b"))+")")),
	)
	orT := func(a, b string) string { return "(Or " + leN(a, b) + " " + leN(b, a) + ")" }
	// by cases on le_total a b: l for a ≤ b (hypothesis h), r for b ≤ a
	split := func(a, b, goal, l, r string) string {
		return "(Or.rec " + leN(a, b) + " " + leN(b, a) + " (fun (_ : " + orT(a, b) + ") => " + goal + ") (fun (h : " + leN(a, b) + ") => " + l + ") (fun (h : " + leN(b, a) + ") => " + r + ") " + ap("Nat.le_total", a, b) + ")"
	}
	eqL, eqR := func(a, b string) string { return ap("Nat.min_eq_left", a, b, "h") }, func(a, b string) string { return ap("Nat.min_eq_right", a, b, "h") }
	mxR, mxL := func(a, b string) string { return ap("Nat.max_eq_right", a, b, "h") }, func(a, b string) string { return ap("Nat.max_eq_left", a, b, "h") }
	rf := func(x string) string { return ap("Nat.le.refl", x) }
	preludeExtra = append(preludeExtra,
		thm("Nat.min_comm", "a b", nil, eqN(mn("a", "b"), mn("b", "a")),
			split("a", "b", eqN(mn("a", "b"), mn("b", "a")),
				tr(mn("a", "b"), "a", mn("b", "a"), eqL("a", "b"), sym(mn("b", "a"), "a", eqR("b", "a"))),
				tr(mn("a", "b"), "b", mn("b", "a"), eqR("a", "b"), sym(mn("b", "a"), "b", eqL("b", "a"))))),
		thm("Nat.max_comm", "a b", nil, eqN(mx("a", "b"), mx("b", "a")),
			split("a", "b", eqN(mx("a", "b"), mx("b", "a")),
				tr(mx("a", "b"), "b", mx("b", "a"), mxR("a", "b"), sym(mx("b", "a"), "b", mxL("b", "a"))),
				tr(mx("a", "b"), "a", mx("b", "a"), mxL("a", "b"), sym(mx("b", "a"), "a", mxR("b", "a"))))),
		thm("Nat.zero_min", "a", nil, eqN(mn("0", "a"), "0"), ap("Nat.min_eq_left", "0", "a", ap("Nat.zero_le", "a"))),
		thm("Nat.min_zero", "a", nil, eqN(mn("a", "0"), "0"), ap("Nat.min_eq_right", "a", "0", ap("Nat.zero_le", "a"))),
		thm("Nat.zero_max", "a", nil, eqN(mx("0", "a"), "a"), ap("Nat.max_eq_right", "0", "a", ap("Nat.zero_le", "a"))),
		thm("Nat.max_zero", "a", nil, eqN(mx("a", "0"), "a"), ap("Nat.max_eq_left", "a", "0", ap("Nat.zero_le", "a"))),
	)
	lf := func(v, w, e string, P func(z string) string, body string) string {
		return tp(w, v, sym(v, w, e), P, body)
	} // body : P w, e : v = w
	preludeExtra = append(preludeExtra,
		thm("Nat.min_le_right", "a b", nil, leN(mn("a", "b"), "b"),
			split("a", "b", leN(mn("a", "b"), "b"),
				lf(mn("a", "b"), "a", eqL("a", "b"), func(z string) string { return leN(z, "b") }, "h"),
				lf(mn("a", "b"), "b", eqR("a", "b"), func(z string) string { return leN(z, "b") }, rf("b")))),
		thm("Nat.min_le_left", "a b", nil, leN(mn("a", "b"), "a"),
			split("a", "b", leN(mn("a", "b"), "a"),
				lf(mn("a", "b"), "a", eqL("a", "b"), func(z string) string { return leN(z, "a") }, rf("a")),
				lf(mn("a", "b"), "b", eqR("a", "b"), func(z string) string { return leN(z, "a") }, "h"))),
		thm("Nat.le_max_left", "a b", nil, leN("a", mx("a", "b")),
			split("a", "b", leN("a", mx("a", "b")),
				lf(mx("a", "b"), "b", mxR("a", "b"), func(z string) string { return leN("a", z) }, "h"),
				lf(mx("a", "b"), "a", mxL("a", "b"), func(z string) string { return leN("a", z) }, rf("a")))),
		thm("Nat.le_max_right", "a b", nil, leN("b", mx("a", "b")),
			split("a", "b", leN("b", mx("a", "b")),
				lf(mx("a", "b"), "b", mxR("a", "b"), func(z string) string { return leN("b", z) }, rf("b")),
				lf(mx("a", "b"), "a", mxL("a", "b"), func(z string) string { return leN("b", z) }, "h"))),
	)
	// c-dependent ones: split on b, c (resp. a, b) with the three-variable names kept as in Lean
	hs2 := func(x, y string) [][2]string { return [][2]string{{"h1", x}, {"h2", y}} }
	e3L := func(a, b string) string { return ap("Nat.min_eq_left", a, b, "h") }
	e3R := func(a, b string) string { return ap("Nat.min_eq_right", a, b, "h") }
	preludeExtra = append(preludeExtra,
		thm("Nat.le_min_of_le_of_le", "a b c", hs2(leN("a", "b"), leN("a", "c")), leN("a", mn("b", "c")),
			split("b", "c", leN("a", mn("b", "c")),
				lf(mn("b", "c"), "b", e3L("b", "c"), func(z string) string { return leN("a", z) }, "h1"),
				lf(mn("b", "c"), "c", e3R("b", "c"), func(z string) string { return leN("a", z) }, "h2"))),
		thm("Nat.max_le_of_le_of_le", "a b c", hs2(leN("a", "c"), leN("b", "c")), leN(mx("a", "b"), "c"),
			split("a", "b", leN(mx("a", "b"), "c"),
				lf(mx("a", "b"), "b", mxR("a", "b"), func(z string) string { return leN(z, "c") }, "h2"),
				lf(mx("a", "b"), "a", mxL("a", "b"), func(z string) string { return leN(z, "c") }, "h1"))),
	)
	andN2 := func(x, y string) string { return andN(x, y) }
	preludeExtra = append(preludeExtra,
		thm("Nat.le_min", "a b c", nil, iffN(leN("a", mn("b", "c")), andN2(leN("a", "b"), leN("a", "c"))),
			iffI(leN("a", mn("b", "c")), andN2(leN("a", "b"), leN("a", "c")), "h",
				ap("And.intro", leN("a", "b"), leN("a", "c"), ap("Nat.le_trans", "a", mn("b", "c"), "b", "h", ap("Nat.min_le_left", "b", "c")),
					ap("Nat.le_trans", "a", mn("b", "c"), "c", "h", ap("Nat.min_le_right", "b", "c"))),
				"p", ap("Nat.le_min_of_le_of_le", "a", "b", "c", ap("And.left", leN("a", "b"), leN("a", "c"), "p"), ap("And.right", leN("a", "b"), leN("a", "c"), "p")))),
		thm("Nat.lt_min", "a b c", nil, iffN(ltN("a", mn("b", "c")), andN2(ltN("a", "b"), ltN("a", "c"))), ap("Nat.le_min", "(Nat.succ a)", "b", "c")),
		thm("Nat.max_le", "a b c", nil, iffN(leN(mx("a", "b"), "c"), andN2(leN("a", "c"), leN("b", "c"))),
			iffI(leN(mx("a", "b"), "c"), andN2(leN("a", "c"), leN("b", "c")), "h",
				ap("And.intro", leN("a", "c"), leN("b", "c"), ap("Nat.le_trans", "a", mx("a", "b"), "c", ap("Nat.le_max_left", "a", "b"), "h"),
					ap("Nat.le_trans", "b", mx("a", "b"), "c", ap("Nat.le_max_right", "a", "b"), "h")),
				"p", ap("Nat.max_le_of_le_of_le", "a", "b", "c", ap("And.left", leN("a", "c"), leN("b", "c"), "p"), ap("And.right", leN("a", "c"), leN("b", "c"), "p")))),
		thm("Nat.max_lt", "a b c", nil, iffN(ltN(mx("a", "b"), "c"), andN2(ltN("a", "c"), ltN("b", "c"))),
			iffI(ltN(mx("a", "b"), "c"), andN2(ltN("a", "c"), ltN("b", "c")), "h",
				ap("And.intro", ltN("a", "c"), ltN("b", "c"), ap("Nat.lt_of_le_of_lt", "a", mx("a", "b"), "c", ap("Nat.le_max_left", "a", "b"), "h"),
					ap("Nat.lt_of_le_of_lt", "b", mx("a", "b"), "c", ap("Nat.le_max_right", "a", "b"), "h")),
				"p", split("a", "b", ltN(mx("a", "b"), "c"),
					lf(mx("a", "b"), "b", mxR("a", "b"), func(z string) string { return ltN(z, "c") }, ap("And.right", ltN("a", "c"), ltN("b", "c"), "p")),
					lf(mx("a", "b"), "a", mxL("a", "b"), func(z string) string { return ltN(z, "c") }, ap("And.left", ltN("a", "c"), ltN("b", "c"), "p"))))),
	)
}
