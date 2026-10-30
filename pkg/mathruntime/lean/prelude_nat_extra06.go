package lean

func init() {
	no := [][2]string{}
	comm := func(a, b string) string { return ap("Nat.gcd_comm", a, b) }
	gm := func(m string) func(string) string { return func(z string) string { return gcdN(m, z) } }
	dvk := func(z string) string { return dvdN("k", z) }
	_ = gm
	// k ∣ a and k ∣ b % a give k ∣ b, since b = a * (b / a) + b % a.
	ba := "(Nat.mod b a)"
	abq := mulN("a", "(Nat.div b a)")
	preludeExtra = append(preludeExtra,
		thm("Nat.dvd_of_dvd_mod", "k b a", [][2]string{{"h1", dvdN("k", "a")}, {"h2", dvdN("k", ba)}}, dvdN("k", "b"),
			tp(addN(abq, ba), "b", ap("Nat.div_add_mod", "a", "b"), dvk,
				ap("Nat.dvd_add", "k", abq, ba, ap("Nat.dvd_mul_right_of_dvd", "k", "a", "(Nat.div b a)", "h1"), "h2"))),
		thm("Nat.gcd_rec", "m n", no, eqN(gcdN("m", "n"), gcdN("(Nat.mod n m)", "m")),
			ap("Nat.dvd_antisymm", gcdN("m", "n"), gcdN("(Nat.mod n m)", "m"),
				ap("Nat.dvd_gcd", gcdN("m", "n"), "(Nat.mod n m)", "m",
					ap("Nat.dvd_mod_of_dvd", gcdN("m", "n"), "n", "m", ap("Nat.gcd_dvd_left", "m", "n"), ap("Nat.gcd_dvd_right", "m", "n")),
					ap("Nat.gcd_dvd_left", "m", "n")),
				ap("Nat.dvd_gcd", gcdN("(Nat.mod n m)", "m"), "m", "n", ap("Nat.gcd_dvd_right", "(Nat.mod n m)", "m"),
					ap("Nat.dvd_of_dvd_mod", gcdN("(Nat.mod n m)", "m"), "n", "m", ap("Nat.gcd_dvd_right", "(Nat.mod n m)", "m"),
						ap("Nat.gcd_dvd_left", "(Nat.mod n m)", "m"))))),
		thm("Nat.gcd_add_one", "x y", no, eqN(gcdN(addN("x", "1"), "y"), gcdN("(Nat.mod y (Nat.add x 1))", addN("x", "1"))),
			ap("Nat.gcd_rec", addN("x", "1"), "y")),
	)
	// gcd.induction
	P := func(a, b string) string { return "(P " + a + " " + b + ")" }
	h0ty := "((n : Nat) → " + P("0", "n") + ")"
	h1ty := "((m : Nat) → (n : Nat) → " + ltN("0", "m") + " → " + P("(Nat.mod n m)", "m") + " → " + P("m", "n") + ")"
	motiveF := func(f string) string { return "(m : Nat) → " + leN("m", f) + " → (n : Nat) → " + P("m", "n") }
	preludeExtra = append(preludeExtra, valDecl{"thm", "Nat.gcd.induction", nil,
		"(P : Nat → Nat → Prop) → (m : Nat) → (n : Nat) → " + h0ty + " → " + h1ty + " → " + P("m", "n"),
		"fun (P : Nat → Nat → Prop) (m : Nat) (n : Nat) (H0 : " + h0ty + ") (H1 : " + h1ty + ") => " +
			"Nat.rec.{0} (fun (f : Nat) => " + motiveF("f") + ") " +
			"(fun (m : Nat) (hm : " + leN("m", "0") + ") (n : Nat) => " +
			tp("0", "m", sym("m", "0", ap("Nat.le_zero", "m", "hm")), func(z string) string { return P(z, "n") }, "(H0 n)") + ") " +
			"(fun (f : Nat) (ih : " + motiveF("f") + ") (m : Nat) => " +
			"Nat.rec.{0} (fun (m : Nat) => " + leN("m", "(Nat.succ f)") + " → (n : Nat) → " + P("m", "n") + ") " +
			"(fun (_ : " + leN("0", "(Nat.succ f)") + ") (n : Nat) => H0 n) " +
			"(fun (m' : Nat) (_ : " + leN("m'", "(Nat.succ f)") + " → (n : Nat) → " + P("m'", "n") + ") (hm : " + leN("(Nat.succ m')", "(Nat.succ f)") + ") (n : Nat) => " +
			"H1 (Nat.succ m') n (Nat.zero_lt_succ m') (ih (Nat.mod n (Nat.succ m')) " +
			ap("Nat.le_of_succ_le_succ", "(Nat.mod n (Nat.succ m'))", "f",
				ap("Nat.le_trans", "(Nat.succ (Nat.mod n (Nat.succ m')))", "(Nat.succ m')", "(Nat.succ f)",
					ap("Nat.mod_lt", "n", "(Nat.succ m')", "(Nat.zero_lt_succ m')"), "hm")) + " (Nat.succ m'))) m) m m (Nat.le.refl m) n"})
	// gcd_div (m n k)
	mk, nk := "(Nat.div m k)", "(Nat.div n k)"
	hdv := [][2]string{{"H1", dvdN("k", "m")}, {"H2", dvdN("k", "n")}}
	gdiv := eqN(gcdN(mk, nk), "(Nat.div "+gcdN("m", "n")+" k)")
	posCase := func(k string) string { // k = succ k': the multiplication is cancelled on the right
		kp := "(Nat.succ k')"
		_ = kp
		return ""
	}
	_ = posCase
	preludeExtra = append(preludeExtra,
		thm("Nat.gcd_div", "m n k", hdv, gdiv,
			"Nat.rec.{0} (fun (k : Nat) => "+dvdN("k", "m")+" → "+dvdN("k", "n")+" → "+gdiv+") "+
				"(fun (_ : "+dvdN("0", "m")+") (_ : "+dvdN("0", "n")+") => "+
				tr(gcdN("(Nat.div m 0)", "(Nat.div n 0)"), gcdN("0", "(Nat.div n 0)"), "(Nat.div "+gcdN("m", "n")+" 0)",
					cg(func(z string) string { return gcdN(z, "(Nat.div n 0)") }, "(Nat.div m 0)", "0", ap("Nat.div_zero", "m")),
					tr(gcdN("0", "(Nat.div n 0)"), gcdN("0", "0"), "(Nat.div "+gcdN("m", "n")+" 0)",
						cg(func(z string) string { return gcdN("0", z) }, "(Nat.div n 0)", "0", ap("Nat.div_zero", "n")),
						tr(gcdN("0", "0"), "0", "(Nat.div "+gcdN("m", "n")+" 0)", "(Nat.gcd_zero_left 0)",
							sym("(Nat.div "+gcdN("m", "n")+" 0)", "0", ap("Nat.div_zero", gcdN("m", "n"))))))+") "+
				"(fun (k' : Nat) (_ : "+dvdN("k'", "m")+" → "+dvdN("k'", "n")+" → "+eqN(gcdN("(Nat.div m k')", "(Nat.div n k')"), "(Nat.div "+gcdN("m", "n")+" k')")+") "+
				"(H1 : "+dvdN("(Nat.succ k')", "m")+") (H2 : "+dvdN("(Nat.succ k')", "n")+") => "+
				func() string {
					k := "(Nat.succ k')"
					mq, nq := "(Nat.div m "+k+")", "(Nat.div n "+k+")"
					g := gcdN("m", "n")
					lhs := mulN(gcdN(mq, nq), k)
					mid := gcdN(mulN(mq, k), mulN(nq, k))
					return ap("Nat.mul_right_cancel", gcdN(mq, nq), k, "(Nat.div "+g+" "+k+")", "(Nat.zero_lt_succ k')",
						tr(lhs, mid, mulN("(Nat.div "+g+" "+k+")", k),
							sym(mid, lhs, ap("Nat.gcd_mul_right", mq, k, nq)),
							tr(mid, g, mulN("(Nat.div "+g+" "+k+")", k),
								tr(mid, gcdN("m", mulN(nq, k)), g,
									cg(func(z string) string { return gcdN(z, mulN(nq, k)) }, mulN(mq, k), "m", ap("Nat.div_mul_cancel", k, "m", "H1")),
									cg(func(z string) string { return gcdN("m", z) }, mulN(nq, k), "n", ap("Nat.div_mul_cancel", k, "n", "H2"))),
								sym(mulN("(Nat.div "+g+" "+k+")", k), g, ap("Nat.div_mul_cancel", k, g, ap("Nat.dvd_gcd", k, "m", "n", "H1", "H2"))))))
				}()+") k H1 H2"),
	)
	// div_gcd_pos, and the gcd of the two quotients
	gab := gcdN("a", "b")
	one := func(z string) string { return leN(z, "a") }
	_ = one
	preludeExtra = append(preludeExtra,
		thm("Nat.div_gcd_pos_of_pos_left", "a b", [][2]string{{"h", ltN("0", "a")}}, ltN("0", "(Nat.div a "+gab+")"),
			iffMpr(leN("1", "(Nat.div a "+gab+")"), leN(mulN("1", gab), "a"),
				ap("Nat.le_div_iff_mul_le", "1", "a", gab, ap("Nat.gcd_pos_of_pos_left", "a", "b", "h")),
				tp(gab, mulN("1", gab), sym(mulN("1", gab), gab, ap("Nat.one_mul", gab)), func(z string) string { return leN(z, "a") },
					ap("Nat.gcd_le_left", "a", "b", "h")))),
		thm("Nat.div_gcd_pos_of_pos_right", "a b", [][2]string{{"h", ltN("0", "b")}}, ltN("0", "(Nat.div b "+gab+")"),
			iffMpr(leN("1", "(Nat.div b "+gab+")"), leN(mulN("1", gab), "b"),
				ap("Nat.le_div_iff_mul_le", "1", "b", gab, ap("Nat.gcd_pos_of_pos_right", "a", "b", "h")),
				tp(gab, mulN("1", gab), sym(mulN("1", gab), gab, ap("Nat.one_mul", gab)), func(z string) string { return leN(z, "b") },
					ap("Nat.gcd_le_right", "a", "b", "h")))),
		thm("Nat.gcd_div_gcd_div_gcd_of_pos_left", "n m", [][2]string{{"h", ltN("0", "n")}},
			eqN(gcdN("(Nat.div n "+gcdN("n", "m")+")", "(Nat.div m "+gcdN("n", "m")+")"), "1"),
			tr(gcdN("(Nat.div n "+gcdN("n", "m")+")", "(Nat.div m "+gcdN("n", "m")+")"), "(Nat.div "+gcdN("n", "m")+" "+gcdN("n", "m")+")", "1",
				ap("Nat.gcd_div", "n", "m", gcdN("n", "m"), ap("Nat.gcd_dvd_left", "n", "m"), ap("Nat.gcd_dvd_right", "n", "m")),
				ap("Nat.div_self", gcdN("n", "m"), ap("Nat.gcd_pos_of_pos_left", "n", "m", "h")))),
		thm("Nat.gcd_div_gcd_div_gcd_of_pos_right", "n m", [][2]string{{"h", ltN("0", "m")}},
			eqN(gcdN("(Nat.div n "+gcdN("n", "m")+")", "(Nat.div m "+gcdN("n", "m")+")"), "1"),
			tr(gcdN("(Nat.div n "+gcdN("n", "m")+")", "(Nat.div m "+gcdN("n", "m")+")"), "(Nat.div "+gcdN("n", "m")+" "+gcdN("n", "m")+")", "1",
				ap("Nat.gcd_div", "n", "m", gcdN("n", "m"), ap("Nat.gcd_dvd_left", "n", "m"), ap("Nat.gcd_dvd_right", "n", "m")),
				ap("Nat.div_self", gcdN("n", "m"), ap("Nat.gcd_pos_of_pos_right", "n", "m", "h")))),
	)
	// pow with gcd n m = 1
	h1 := [][2]string{{"h", eqN(gcdN("n", "m"), "1")}}
	pw := func(a, e string) string { return ap("Nat.pow", a, e) }
	hs := tr(gcdN("m", "n"), gcdN("n", "m"), "1", comm("m", "n"), "h")
	preludeExtra = append(preludeExtra,
		thm("Nat.gcd_pow_right_of_gcd_eq_one", "n m k", h1, eqN(gcdN("n", pw("m", "k")), "1"),
			tr(gcdN("n", pw("m", "k")), gcdN(pw("m", "k"), "n"), "1", comm("n", pw("m", "k")),
				ap("Nat.gcd_pow_left_of_gcd_eq_one", "m", "n", "k", hs))),
		thm("Nat.pow_gcd_pow_of_gcd_eq_one", "n m k l", h1, eqN(gcdN(pw("n", "k"), pw("m", "l")), "1"),
			"(fun (hh : "+eqN(gcdN("n", pw("m", "l")), "1")+") => "+
				ap("Nat.gcd_pow_left_of_gcd_eq_one", "n", pw("m", "l"), "k", "hh")+") "+ap("Nat.gcd_pow_right_of_gcd_eq_one", "n", "m", "l", "h")),
	)
	_ = dvk
}
