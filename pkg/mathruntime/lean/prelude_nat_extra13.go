package lean

// lcm of powers: lcm (n^k) (m^k) = (lcm n m)^k (Init/Data/Nat/Lcm.lean `pow_lcm_pow`), by cancelling gcd^k.
func init() {
	pw := func(a, e string) string { return ap("Nat.pow", a, e) }
	z0 := func(k string) string { return eqN(lcmN(pw("0", k), pw("m", k)), pw(lcmN("0", "m"), k)) }
	S := "(Nat.succ j)"
	zeroStep := tr(lcmN(pw("0", S), pw("m", S)), "0", pw(lcmN("0", "m"), S),
		tr(lcmN(pw("0", S), pw("m", S)), lcmN("0", pw("m", S)), "0",
			cg(func(z string) string { return lcmN(z, pw("m", S)) }, pw("0", S), "0",
				tr(pw("0", S), mulN(pw("0", "j"), "0"), "0", ap("Nat.pow_succ", "0", "j"), ap("Nat.mul_zero", pw("0", "j")))),
			ap("Nat.lcm_zero_left", pw("m", S))),
		sym(pw(lcmN("0", "m"), S), "0",
			tr(pw(lcmN("0", "m"), S), pw("0", S), "0",
				cg(func(z string) string { return pw(z, S) }, lcmN("0", "m"), "0", ap("Nat.lcm_zero_left", "m")),
				tr(pw("0", S), mulN(pw("0", "j"), "0"), "0", ap("Nat.pow_succ", "0", "j"), ap("Nat.mul_zero", pw("0", "j"))))))
	preludeExtra = append(preludeExtra,
		thm("Nat.pow_lcm_pow_zero_left", "m k", nil, z0("k"),
			"Nat.rec.{0} (fun (k : Nat) => "+z0("k")+") "+ap("Nat.lcm_one_left", "1")+
				" (fun (j : Nat) (_ : "+z0("j")+") => "+zeroStep+") k"),
	)
	L, G := lcmN("n", "m"), gcdN("n", "m")
	Gk, Lk := pw(G, "k"), pw(L, "k")
	nk, mk := pw("n", "k"), pw("m", "k")
	Lp := lcmN(nk, mk)
	// Gk * Lp = n^k * m^k = Gk * Lk
	lhs := tr(mulN(Gk, Lp), mulN(gcdN(nk, mk), Lp), mulN(nk, mk),
		cg(func(z string) string { return mulN(z, Lp) }, Gk, gcdN(nk, mk), sym(gcdN(nk, mk), Gk, ap("Nat.pow_gcd_pow", "n", "m", "k"))),
		ap("Nat.gcd_mul_lcm", nk, mk))
	rhs := tr(mulN(Gk, Lk), pw(mulN(G, L), "k"), mulN(nk, mk),
		sym(pw(mulN(G, L), "k"), mulN(Gk, Lk), ap("Nat.mul_pow", G, L, "k")),
		tr(pw(mulN(G, L), "k"), pw(mulN("n", "m"), "k"), mulN(nk, mk),
			cg(func(z string) string { return pw(z, "k") }, mulN(G, L), mulN("n", "m"), ap("Nat.gcd_mul_lcm", "n", "m")),
			ap("Nat.mul_pow", "n", "m", "k")))
	gpos := ap("Nat.pos_pow_of_pos", G, "k", ap("Nat.gcd_pos_of_pos_left", "n", "m", "hn"))
	pos := eqN(Lp, Lk)
	preludeExtra = append(preludeExtra,
		thm("Nat.pow_lcm_pow_pos", "n m k", [][2]string{{"hn", ltN("0", "n")}}, pos,
			ap("Nat.mul_left_cancel", Gk, Lp, Lk, gpos, tr(mulN(Gk, Lp), mulN(nk, mk), mulN(Gk, Lk), lhs, sym(mulN(Gk, Lk), mulN(nk, mk), rhs)))),
		thm("Nat.pow_lcm_pow", "n m k", nil, eqN(lcmN(nk, mk), Lk),
			"Nat.rec.{0} (fun (n : Nat) => "+eqN(lcmN(pw("n", "k"), mk), pw(lcmN("n", "m"), "k"))+") "+
				ap("Nat.pow_lcm_pow_zero_left", "m", "k")+" "+
				"(fun (i : Nat) (_ : "+eqN(lcmN(pw("i", "k"), mk), pw(lcmN("i", "m"), "k"))+") => "+
				ap("Nat.pow_lcm_pow_pos", "(Nat.succ i)", "m", "k", "(Nat.zero_lt_succ i)")+") n"),
	)
}
