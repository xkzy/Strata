package lean

// (m * n + p) / m = n for p < m: the quotient step Int.ediv of a product needs (Init/Data/Nat/Basic.lean `Nat.mul_add_div`, restricted to p < m).
func init() {
	x := addN(mulN("m", "n"), "p")
	dx := "(Nat.div " + x + " m)"
	sn := "(Nat.succ n)"
	mc := func(a, b string) string { return ap("Nat.mul_comm", a, b) }
	nm := mulN("n", "m")
	lt1 := ap("Nat.add_lt_add_left", "p", "m", nm, "hp")
	lt2 := tp(addN(nm, "m"), mulN(sn, "m"), sym(mulN(sn, "m"), addN(nm, "m"), ap("Nat.succ_mul", "n", "m")),
		func(z string) string { return ltN(addN(nm, "p"), z) }, lt1)
	lt3 := tp(nm, mulN("m", "n"), mc("n", "m"), func(z string) string { return ltN(addN(z, "p"), mulN(sn, "m")) }, lt2)
	up := iffMpr(ltN(dx, sn), ltN(x, mulN(sn, "m")), ap("Nat.div_lt_iff_lt_mul", x, sn, "m", "hm"), lt3)
	le3 := tp(nm, mulN("m", "n"), mc("n", "m"), func(z string) string { return leN(nm, addN(z, "p")) }, ap("Nat.le_add_right", nm, "p"))
	lo := iffMpr(leN("n", dx), leN(nm, x), ap("Nat.le_div_iff_mul_le", "n", x, "m", "hm"), le3)
	preludeExtra = append(preludeExtra,
		thm("Nat.mul_add_div_of_lt", "m n p", [][2]string{{"hm", ltN("0", "m")}, {"hp", ltN("p", "m")}}, eqN(dx, "n"),
			ap("Nat.le_antisymm", dx, "n", ap("Nat.le_of_lt_succ", dx, "n", up), lo)),
	)
}
