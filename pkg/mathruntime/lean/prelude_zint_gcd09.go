package lean

// Int gcd / lcm: the `= g`, `= 1` characterisations, lcm_eq_mul_iff and gcd_le_left/right.
func init() {
	ab := iabs
	oN := icast
	idv := func(a, b string) string { return "(Int.dvd " + a + " " + b + ")" }
	toN := func(a, b, h string) string {
		return iffMpr(dvdN(ab(a), ab(b)), idv(a, b), ap("Int.natAbs_dvd_natAbs", a, b), h)
	}
	iff := func(l, r string) string { return "(Iff " + l + " " + r + ")" }
	faI := func(v, body string) string { return "((" + v + " : Int) → " + body + ")" }
	imp2 := func(a, b string) string { return "(" + a + " → " + b + ")" }
	gab, lab := igcd("a", "b"), ilcm("a", "b")
	// gcd a b = g ↔ ↑g ∣ a ∧ ↑g ∣ b ∧ ∀ c, c ∣ a → c ∣ b → c ∣ ↑g
	rg := func(g string) string {
		return andN(idv(oN(g), "a"), andN(idv(oN(g), "b"), faI("c", imp2(idv("c", "a"), imp2(idv("c", "b"), idv("c", oN(g)))))))
	}
	preludeExtra = append(preludeExtra,
		ithm("Int.gcd_eq_iff", "a b", [][2]string{{"g", "Nat"}}, iff(eqN(gab, "g"), rg("g")),
			iffI(eqN(gab, "g"), rg("g"), "h",
				tp(gab, "g", "h", rg,
					ap("And.intro", idv(oN(gab), "a"), andN(idv(oN(gab), "b"), faI("c", imp2(idv("c", "a"), imp2(idv("c", "b"), idv("c", oN(gab)))))),
						ap("Int.gcd_dvd_left", "a", "b"),
						ap("And.intro", idv(oN(gab), "b"), faI("c", imp2(idv("c", "a"), imp2(idv("c", "b"), idv("c", oN(gab))))),
							ap("Int.gcd_dvd_right", "a", "b"),
							"(fun (c : Int) (ha : "+idv("c", "a")+") (hb : "+idv("c", "b")+") => "+ap("Int.dvd_coe_gcd", "c", "a", "b", "ha", "hb")+")"))),
				"h", ap("Nat.dvd_antisymm", gab, "g",
					toN(oN(gab), oN("g"), "("+ap("And.right", idv(oN("g"), "b"), faI("c", imp2(idv("c", "a"), imp2(idv("c", "b"), idv("c", oN("g"))))),
						ap("And.right", idv(oN("g"), "a"), andN(idv(oN("g"), "b"), faI("c", imp2(idv("c", "a"), imp2(idv("c", "b"), idv("c", oN("g")))))), "h"))+" "+oN(gab)+" "+
						ap("Int.gcd_dvd_left", "a", "b")+" "+ap("Int.gcd_dvd_right", "a", "b")+")"),
					ap("Int.dvd_gcd", "a", "b", "g", ap("And.left", idv(oN("g"), "a"), andN(idv(oN("g"), "b"), faI("c", imp2(idv("c", "a"), imp2(idv("c", "b"), idv("c", oN("g")))))), "h"),
						ap("And.left", idv(oN("g"), "b"), faI("c", imp2(idv("c", "a"), imp2(idv("c", "b"), idv("c", oN("g"))))),
							ap("And.right", idv(oN("g"), "a"), andN(idv(oN("g"), "b"), faI("c", imp2(idv("c", "a"), imp2(idv("c", "b"), idv("c", oN("g")))))), "h")))))),
	)
	one := io()
	r1 := faI("c", imp2(idv("c", "a"), imp2(idv("c", "b"), idv("c", one))))
	preludeExtra = append(preludeExtra,
		ithm("Int.gcd_eq_one_iff", "a b", nil, iff(eqN(gab, "1"), r1),
			iffI(eqN(gab, "1"), r1, "h",
				"(fun (c : Int) (ha : "+idv("c", "a")+") (hb : "+idv("c", "b")+") => "+
					tp(gab, "1", "h", func(z string) string { return idv("c", oN(z)) }, ap("Int.dvd_coe_gcd", "c", "a", "b", "ha", "hb"))+")",
				"h", ap("Nat.eq_one_of_dvd_one", gab,
					toN(oN(gab), one, "("+"h"+" "+oN(gab)+" "+ap("Int.gcd_dvd_left", "a", "b")+" "+ap("Int.gcd_dvd_right", "a", "b")+")")))),
	)
	// lcm
	rl := func(l string) string {
		return andN(idv("a", oN(l)), andN(idv("b", oN(l)), faI("c", imp2(idv("a", "c"), imp2(idv("b", "c"), idv(oN(l), "c"))))))
	}
	rlb := faI("c", imp2(idv("a", "c"), imp2(idv("b", "c"), idv(oN(lab), "c"))))
	preludeExtra = append(preludeExtra,
		ithm("Int.lcm_eq_iff", "a b", [][2]string{{"l", "Nat"}}, iff(eqN(lab, "l"), rl("l")),
			iffI(eqN(lab, "l"), rl("l"), "h",
				tp(lab, "l", "h", rl,
					ap("And.intro", idv("a", oN(lab)), andN(idv("b", oN(lab)), rlb), ap("Int.dvd_lcm_left", "a", "b"),
						ap("And.intro", idv("b", oN(lab)), rlb, ap("Int.dvd_lcm_right", "a", "b"),
							"(fun (c : Int) (ha : "+idv("a", "c")+") (hb : "+idv("b", "c")+") => "+ap("Int.coe_lcm_dvd", "a", "b", "c", "ha", "hb")+")"))),
				"h", ap("Nat.dvd_antisymm", lab, "l",
					ap("Int.lcm_dvd", "a", "b", "l", ap("And.left", idv("a", oN("l")), andN(idv("b", oN("l")), faI("c", imp2(idv("a", "c"), imp2(idv("b", "c"), idv(oN("l"), "c"))))), "h"),
						ap("And.left", idv("b", oN("l")), faI("c", imp2(idv("a", "c"), imp2(idv("b", "c"), idv(oN("l"), "c")))),
							ap("And.right", idv("a", oN("l")), andN(idv("b", oN("l")), faI("c", imp2(idv("a", "c"), imp2(idv("b", "c"), idv(oN("l"), "c"))))), "h"))),
					toN(oN("l"), oN(lab), "("+ap("And.right", idv("b", oN("l")), faI("c", imp2(idv("a", "c"), imp2(idv("b", "c"), idv(oN("l"), "c")))),
						ap("And.right", idv("a", oN("l")), andN(idv("b", oN("l")), faI("c", imp2(idv("a", "c"), imp2(idv("b", "c"), idv(oN("l"), "c"))))), "h"))+" "+oN(lab)+" "+
						ap("Int.dvd_lcm_left", "a", "b")+" "+ap("Int.dvd_lcm_right", "a", "b")+")")))),
		ithm("Int.lcm_eq_one_iff", "a b", nil, iff(eqN(lab, "1"), andN(idv("a", one), idv("b", one))),
			iffI(eqN(lab, "1"), andN(idv("a", one), idv("b", one)), "h",
				tp(lab, "1", "h", func(z string) string { return andN(idv("a", oN(z)), idv("b", oN(z))) },
					ap("And.intro", idv("a", oN(lab)), idv("b", oN(lab)), ap("Int.dvd_lcm_left", "a", "b"), ap("Int.dvd_lcm_right", "a", "b"))),
				"h", ap("Nat.eq_one_of_dvd_one", lab,
					ap("Int.lcm_dvd", "a", "b", "1", ap("And.left", idv("a", one), idv("b", one), "h"), ap("And.right", idv("a", one), idv("b", one), "h"))))),
	)
	// lcm a b = |a| * |b| ↔ a = 0 ∨ b = 0 ∨ gcd a b = 1
	z := func(x string) string { return ieq(x, iz()) }
	zn := func(x string) string { return eqN(ab(x), "0") }
	lm := eqN(lab, mulN(ab("a"), ab("b")))
	oI := "(Or " + z("a") + " (Or " + z("b") + " " + eqN(gab, "1") + "))"
	oNt := "(Or " + zn("a") + " (Or " + zn("b") + " " + eqN(gab, "1") + "))"
	inner := "(Or " + zn("b") + " " + eqN(gab, "1") + ")"
	innerI := "(Or " + z("b") + " " + eqN(gab, "1") + ")"
	fz := func(x, h string) string { return iffMp(zn(x), z(x), ap("Int.natAbs_eq_zero", x), h) }
	tz := func(x, h string) string { return iffMpr(zn(x), z(x), ap("Int.natAbs_eq_zero", x), h) }
	nat := ap("Nat.lcm_eq_mul_iff", ab("a"), ab("b"))
	preludeExtra = append(preludeExtra,
		ithm("Int.lcm_eq_mul_iff", "a b", nil, iff(lm, oI),
			iffI(lm, oI, "h",
				"(Or.rec "+zn("a")+" "+inner+" (fun (_ : "+oNt+") => "+oI+") "+
					"(fun (x : "+zn("a")+") => "+ap("Or.inl", z("a"), innerI, fz("a", "x"))+") "+
					"(fun (r : "+inner+") => "+ap("Or.inr", z("a"), innerI,
					"(Or.rec "+zn("b")+" "+eqN(gab, "1")+" (fun (_ : "+inner+") => "+innerI+") "+
						"(fun (x : "+zn("b")+") => "+ap("Or.inl", z("b"), eqN(gab, "1"), fz("b", "x"))+") "+
						"(fun (x : "+eqN(gab, "1")+") => "+ap("Or.inr", z("b"), eqN(gab, "1"), "x")+") r)")+") "+
					iffMp(lm, oNt, nat, "h")+")",
				"h", iffMpr(lm, oNt, nat,
					"(Or.rec "+z("a")+" "+innerI+" (fun (_ : "+oI+") => "+oNt+") "+
						"(fun (x : "+z("a")+") => "+ap("Or.inl", zn("a"), inner, tz("a", "x"))+") "+
						"(fun (r : "+innerI+") => "+ap("Or.inr", zn("a"), inner,
						"(Or.rec "+z("b")+" "+eqN(gab, "1")+" (fun (_ : "+innerI+") => "+inner+") "+
							"(fun (x : "+z("b")+") => "+ap("Or.inl", zn("b"), eqN(gab, "1"), tz("b", "x"))+") "+
							"(fun (x : "+eqN(gab, "1")+") => "+ap("Or.inr", zn("b"), eqN(gab, "1"), "x")+") r)")+") h)"))),
	)
	// gcd_le_left / right: ↑(gcd a b) ≤ a for 0 < a
	pos := func(x string) string { return "(Int.lt (Int.ofNat 0) " + x + ")" }
	le := func(x, y string) string { return "(Int.le " + x + " " + y + ")" }
	neZ := func(x string) string {
		return "(fun (e : " + z(x) + ") => " + ap("Int.lt_irrefl", iz(), tpI(x, iz(), "e", func(w string) string { return pos(w) }, "h")) + ")"
	}
	bound := func(x, natLe string) string {
		return ap("Int.le_trans", oN(gab), oN(ab(x)), x,
			iffMpr(le(oN(gab), oN(ab(x))), leN(gab, ab(x)), ap("Int.ofNat_le", gab, ab(x)), natLe),
			ap("Int.le_of_eq", oN(ab(x)), x, ap("Int.ofNat_natAbs_of_nonneg", x, ap("Int.le_of_lt", iz(), x, "h"))))
	}
	preludeExtra = append(preludeExtra,
		ithm("Int.gcd_le_left", "b a", [][2]string{{"h", pos("a")}}, le(oN(igcd("a", "b")), "a"),
			bound("a", ap("Int.gcd_le_natAbs_left", "b", "a", neZ("a")))),
		ithm("Int.gcd_le_right", "a b", [][2]string{{"h", pos("b")}}, le(oN(gab), "b"),
			bound("b", ap("Int.gcd_le_natAbs_right", "a", "b", neZ("b")))),
	)
}
