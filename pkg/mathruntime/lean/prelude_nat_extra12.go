package lean

// the other three lcm_mul_*_eq_mul_of_lcm_eq_mul shapes, derived from the right_right one.
func init() {
	hl := [][2]string{{"h", eqN(lcmN("n", "m"), mulN("n", "m"))}}
	mc := func(a, b string) string { return ap("Nat.mul_comm", a, b) }
	base := "Nat.lcm_mul_right_right_eq_mul_of_lcm_eq_mul"
	// lcm n (k*m) = lcm n k * m
	lr := tr(lcmN("n", mulN("k", "m")), lcmN("n", mulN("m", "k")), mulN(lcmN("n", "k"), "m"),
		cg(func(z string) string { return lcmN("n", z) }, mulN("k", "m"), mulN("m", "k"), mc("k", "m")),
		ap(base, "n", "m", "k", "h"))
	// h' : lcm m n = m * n
	hr := tr(lcmN("m", "n"), lcmN("n", "m"), mulN("m", "n"), ap("Nat.lcm_comm", "m", "n"),
		tr(lcmN("n", "m"), mulN("n", "m"), mulN("m", "n"), "h", mc("n", "m")))
	// lcm (n*k) m = n * lcm k m
	rl := tr(lcmN(mulN("n", "k"), "m"), lcmN("m", mulN("n", "k")), mulN("n", lcmN("k", "m")),
		ap("Nat.lcm_comm", mulN("n", "k"), "m"),
		tr(lcmN("m", mulN("n", "k")), mulN(lcmN("m", "k"), "n"), mulN("n", lcmN("k", "m")),
			ap(base, "m", "n", "k", hr),
			tr(mulN(lcmN("m", "k"), "n"), mulN("n", lcmN("m", "k")), mulN("n", lcmN("k", "m")),
				mc(lcmN("m", "k"), "n"),
				cg(func(z string) string { return mulN("n", z) }, lcmN("m", "k"), lcmN("k", "m"), ap("Nat.lcm_comm", "m", "k")))))
	ll := tr(lcmN(mulN("k", "n"), "m"), lcmN(mulN("n", "k"), "m"), mulN("n", lcmN("k", "m")),
		cg(func(z string) string { return lcmN(z, "m") }, mulN("k", "n"), mulN("n", "k"), mc("k", "n")), rl)
	preludeExtra = append(preludeExtra,
		thm("Nat.lcm_mul_left_right_eq_mul_of_lcm_eq_mul", "n m k", hl, eqN(lcmN("n", mulN("k", "m")), mulN(lcmN("n", "k"), "m")), lr),
		thm("Nat.lcm_mul_right_left_eq_mul_of_lcm_eq_mul", "n m k", hl, eqN(lcmN(mulN("n", "k"), "m"), mulN("n", lcmN("k", "m"))), rl),
		thm("Nat.lcm_mul_left_left_eq_mul_of_lcm_eq_mul", "n m k", hl, eqN(lcmN(mulN("k", "n"), "m"), mulN("n", lcmN("k", "m"))), ll),
	)
}
