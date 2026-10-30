package lean

// Int gcd / lcm, continued: positivity, bounds, lcm basics.
func absNe(x, hx string) string { // |x| ≠ 0 from x ≠ 0
	return "(fun (e : " + eqN(iabs(x), "0") + ") => " + hx + " " + iffMp(eqN(iabs(x), "0"), ieq(x, iz()), ap("Int.natAbs_eq_zero", x), "e") + ")"
}
func absPos(x, hx string) string { return ap("Nat.pos_of_ne_zero", iabs(x), absNe(x, hx)) }

func init() {
	a, b := "a", "b"
	ga, gb := iabs(a), iabs(b)
	gab := igcd(a, b)
	lab := ilcm(a, b)
	nz := func(x string) [][2]string { return [][2]string{{"h", inz(x)}} }
	nz2 := [][2]string{{"ha", inz(a)}, {"hb", inz(b)}}
	zeroEq := func(x, h string) string { // Nat.gcd/lcm = 0 gives |x| = 0, hence x = 0
		return iffMp(eqN(iabs(x), "0"), ieq(x, iz()), ap("Int.natAbs_eq_zero", x), h)
	}
	nat2 := func(name string, concl, proof string) valDecl {
		return valDecl{"thm", name, nil, "(a : Nat) → (b : Nat) → " + concl, "fun (a : Nat) (b : Nat) => " + proof}
	}
	preludeExtra = append(preludeExtra,
		nat2("Int.gcd_natCast_natCast", eqN(igcd("(Int.ofNat a)", "(Int.ofNat b)"), gcdN("a", "b")), "Eq.refl.{1} Nat "+gcdN("a", "b")),
		nat2("Int.lcm_natCast_natCast", eqN(ilcm("(Int.ofNat a)", "(Int.ofNat b)"), lcmN("a", "b")), "Eq.refl.{1} Nat "+lcmN("a", "b")),
		ithm("Int.gcd_le_natAbs_left", "b a", nz(a), leN(gab, ga), ap("Nat.gcd_le_left", ga, gb, absPos(a, "h"))),
		ithm("Int.gcd_le_natAbs_right", "a b", nz(b), leN(gab, gb), ap("Nat.gcd_le_right", ga, gb, absPos(b, "h"))),
		ithm("Int.gcd_pos_of_ne_zero_left", "b a", nz(a), ltN("0", gab), ap("Nat.gcd_pos_of_pos_left", ga, gb, absPos(a, "h"))),
		ithm("Int.gcd_pos_of_ne_zero_right", "a b", nz(b), ltN("0", gab), ap("Nat.gcd_pos_of_pos_right", ga, gb, absPos(b, "h"))),
		ithm("Int.gcd_ne_zero_left", "a b", nz(a), nzN(gab), ap("Nat.gcd_ne_zero_left", ga, gb, absNe(a, "h"))),
		ithm("Int.gcd_ne_zero_right", "a b", nz(b), nzN(gab), ap("Nat.gcd_ne_zero_right", ga, gb, absNe(b, "h"))),
		ithm("Int.natAbs_div_gcd_pos_of_ne_zero_left", "b a", nz(a), ltN("0", "(Nat.div "+ga+" "+gab+")"), ap("Nat.div_gcd_pos_of_pos_left", ga, gb, absPos(a, "h"))),
		ithm("Int.natAbs_div_gcd_pos_of_ne_zero_right", "a b", nz(b), ltN("0", "(Nat.div "+gb+" "+gab+")"), ap("Nat.div_gcd_pos_of_pos_right", ga, gb, absPos(b, "h"))),
		ithm("Int.eq_zero_of_gcd_eq_zero_left", "a b", [][2]string{{"h", eqN(gab, "0")}}, ieq(a, iz()), zeroEq(a, ap("Nat.eq_zero_of_gcd_eq_zero_left", ga, gb, "h"))),
		ithm("Int.eq_zero_of_gcd_eq_zero_right", "a b", [][2]string{{"h", eqN(gab, "0")}}, ieq(b, iz()), zeroEq(b, ap("Nat.eq_zero_of_gcd_eq_zero_right", ga, gb, "h"))),
		ithm("Int.gcd_left_comm", "a b c", nil, eqN(igcd(a, icast(igcd("b", "c"))), igcd(b, icast(igcd("a", "c")))), ap("Nat.gcd_left_comm", ga, gb, iabs("c"))),
		ithm("Int.gcd_gcd_self_right_left", "m n", nil, eqN(igcd("m", icast(igcd("m", "n"))), igcd("m", "n")), ap("Nat.gcd_gcd_self_right_left", iabs("m"), iabs("n"))),
		ithm("Int.gcd_gcd_self_right_right", "m n", nil, eqN(igcd("m", icast(igcd("n", "m"))), igcd("n", "m")), ap("Nat.gcd_gcd_self_right_right", iabs("m"), iabs("n"))),
		ithm("Int.gcd_gcd_self_left_right", "m n", nil, eqN(igcd(icast(igcd("n", "m")), "m"), igcd("n", "m")), ap("Nat.gcd_gcd_self_left_right", iabs("m"), iabs("n"))),
		ithm("Int.gcd_gcd_self_left_left", "m n", nil, eqN(igcd(icast(igcd("m", "n")), "m"), igcd("m", "n")), ap("Nat.gcd_gcd_self_left_left", iabs("m"), iabs("n"))),
		ithm("Int.gcd_le_natAbs_mul", "a b", nz2, leN(gab, mulN(ga, gb)), ap("Nat.gcd_le_mul", ga, gb, absPos(a, "ha"), absPos(b, "hb"))),
		ithm("Int.lcm_le_natAbs_mul", "a b", nz2, leN(lab, mulN(ga, gb)), ap("Nat.lcm_le_mul", ga, gb, absPos(a, "ha"), absPos(b, "hb"))),
		ithm("Int.lcm_ne_zero", "m n", [][2]string{{"hm", inz("m")}, {"hn", inz("n")}}, nzN(ilcm("m", "n")),
			ap("Nat.lcm_ne_zero", iabs("m"), iabs("n"), absNe("m", "hm"), absNe("n", "hn"))),
		ithm("Int.gcd_mul_lcm", "m n", nil, eqN(mulN(igcd("m", "n"), ilcm("m", "n")), mulN(iabs("m"), iabs("n"))), ap("Nat.gcd_mul_lcm", iabs("m"), iabs("n"))),
		ithm("Int.lcm_mul_gcd", "m n", nil, eqN(mulN(ilcm("m", "n"), igcd("m", "n")), mulN(iabs("m"), iabs("n"))),
			tr(mulN(ilcm("m", "n"), igcd("m", "n")), mulN(igcd("m", "n"), ilcm("m", "n")), mulN(iabs("m"), iabs("n")),
				ap("Nat.mul_comm", ilcm("m", "n"), igcd("m", "n")), ap("Int.gcd_mul_lcm", "m", "n"))),
		ithm("Int.lcm_dvd_natAbs_mul", "m n", nil, dvdN(ilcm("m", "n"), mulN(iabs("m"), iabs("n"))), ap("Nat.lcm_dvd_mul", iabs("m"), iabs("n"))),
		ithm("Int.gcd_dvd_natAbs_mul", "m n", nil, dvdN(igcd("m", "n"), mulN(iabs("m"), iabs("n"))), ap("Nat.gcd_dvd_mul", iabs("m"), iabs("n"))),
		ithm("Int.lcm_eq_natAbs_lcm_natAbs", "m n", nil, eqN(ilcm("m", "n"), lcmN(iabs("m"), iabs("n"))), "Eq.refl.{1} Nat "+lcmN(iabs("m"), iabs("n"))),
		ithm("Int.natAbs_dvd_lcm_left", "a b", nil, dvdN(ga, lab), ap("Nat.dvd_lcm_left", ga, gb)),
		ithm("Int.natAbs_dvd_lcm_right", "a b", nil, dvdN(gb, lab), ap("Nat.dvd_lcm_right", ga, gb)),
		ithm("Int.natAbs_le_lcm_left", "a b", [][2]string{{"hb", inz(b)}}, leN(ga, lab), ap("Nat.le_lcm_left", ga, gb, absPos(b, "hb"))),
		ithm("Int.natAbs_le_lcm_right", "b a", [][2]string{{"ha", inz(a)}}, leN(gb, lab), ap("Nat.le_lcm_right", ga, gb, absPos(a, "ha"))),
		ithm("Int.lcm_assoc", "a b c", nil, eqN(ilcm(icast(ilcm(a, b)), "c"), ilcm(a, icast(ilcm("b", "c")))), ap("Nat.lcm_assoc", ga, gb, iabs("c"))),
		ithm("Int.one_lcm", "a", nil, eqN(ilcm(io(), a), ga), ap("Nat.lcm_one_left", ga)),
		ithm("Int.lcm_one", "a", nil, eqN(ilcm(a, io()), ga), ap("Nat.lcm_one_right", ga)),
		ithm("Int.zero_lcm", "a", nil, eqN(ilcm(iz(), a), "0"), ap("Nat.lcm_zero_left", ga)),
		ithm("Int.lcm_zero", "a", nil, eqN(ilcm(a, iz()), "0"), ap("Nat.lcm_zero_right", ga)),
		ithm("Int.lcm_one_left", "a", nil, eqN(ilcm(io(), a), ga), ap("Nat.lcm_one_left", ga)),
		ithm("Int.lcm_one_right", "a", nil, eqN(ilcm(a, io()), ga), ap("Nat.lcm_one_right", ga)),
		ithm("Int.lcm_zero_left", "a", nil, eqN(ilcm(iz(), a), "0"), ap("Nat.lcm_zero_left", ga)),
		ithm("Int.lcm_zero_right", "a", nil, eqN(ilcm(a, iz()), "0"), ap("Nat.lcm_zero_right", ga)),
		ithm("Int.lcm_self", "a", nil, eqN(ilcm(a, a), ga), ap("Nat.lcm_self", ga)),
		ithm("Int.neg_lcm", "a b", nil, eqN(ilcm("(Int.neg a)", b), lab),
			cg(func(z string) string { return lcmN(z, gb) }, iabs("(Int.neg a)"), ga, ap("Int.natAbs_neg", a))),
		ithm("Int.lcm_neg", "a b", nil, eqN(ilcm(a, "(Int.neg b)"), lab),
			cg(func(z string) string { return lcmN(ga, z) }, iabs("(Int.neg b)"), gb, ap("Int.natAbs_neg", b))),
	)
}
