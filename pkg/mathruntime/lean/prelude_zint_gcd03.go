package lean

// Int gcd / lcm against products and divisibility, through `natAbs_mul` and `natAbs_dvd_natAbs`.
func init() {
	nm := func(x, y string) string { return ap("Int.natAbs_mul", x, y) } // |x*y| = |x|*|y|
	ab := iabs
	// body : P(|x|*|y|) carried to P(|x*y|)
	via := func(x, y string, P func(z string) string, body string) string {
		return tp(mulN(ab(x), ab(y)), ab(imul(x, y)), sym(ab(imul(x, y)), mulN(ab(x), ab(y)), nm(x, y)), P, body)
	}
	dv := func(h, a, b string) string { // Int.dvd a b -> |a| ∣ |b|
		return iffMpr(dvdN(ab(a), ab(b)), "(Int.dvd "+a+" "+b+")", ap("Int.natAbs_dvd_natAbs", a, b), h)
	}
	hd := func(a, b string) [][2]string { return [][2]string{{"h", "(Int.dvd " + a + " " + b + ")"}} }
	g := func(x, y string) string { return gcdN(x, y) }
	preludeExtra = append(preludeExtra,
		ithm("Int.gcd_dvd_gcd_of_dvd_left", "c a b", hd("a", "b"), dvdN(igcd("a", "c"), igcd("b", "c")),
			ap("Nat.gcd_dvd_gcd_of_dvd_left", ab("c"), ab("a"), ab("b"), dv("h", "a", "b"))),
		ithm("Int.gcd_dvd_gcd_of_dvd_right", "c a b", hd("a", "b"), dvdN(igcd("c", "a"), igcd("c", "b")),
			ap("Nat.gcd_dvd_gcd_of_dvd_right", ab("c"), ab("a"), ab("b"), dv("h", "a", "b"))),
		ithm("Int.gcd_dvd_gcd_mul_left_left", "a b c", nil, dvdN(igcd("a", "b"), igcd(imul("c", "a"), "b")),
			via("c", "a", func(z string) string { return dvdN(igcd("a", "b"), g(z, ab("b"))) }, ap("Nat.gcd_dvd_gcd_mul_left_left", ab("a"), ab("b"), ab("c")))),
		ithm("Int.gcd_dvd_gcd_mul_right_left", "a b c", nil, dvdN(igcd("a", "b"), igcd(imul("a", "c"), "b")),
			via("a", "c", func(z string) string { return dvdN(igcd("a", "b"), g(z, ab("b"))) }, ap("Nat.gcd_dvd_gcd_mul_right_left", ab("a"), ab("b"), ab("c")))),
		ithm("Int.gcd_dvd_gcd_mul_left_right", "a b c", nil, dvdN(igcd("a", "b"), igcd("a", imul("c", "b"))),
			via("c", "b", func(z string) string { return dvdN(igcd("a", "b"), g(ab("a"), z)) }, ap("Nat.gcd_dvd_gcd_mul_left_right", ab("a"), ab("b"), ab("c")))),
		ithm("Int.gcd_dvd_gcd_mul_right_right", "a b c", nil, dvdN(igcd("a", "b"), igcd("a", imul("b", "c"))),
			via("b", "c", func(z string) string { return dvdN(igcd("a", "b"), g(ab("a"), z)) }, ap("Nat.gcd_dvd_gcd_mul_right_right", ab("a"), ab("b"), ab("c")))),
		// gcd (a*b) b = |b| and friends
		ithm("Int.gcd_mul_left_left", "a b", nil, eqN(igcd(imul("a", "b"), "b"), ab("b")),
			via("a", "b", func(z string) string { return eqN(g(z, ab("b")), ab("b")) }, ap("Nat.gcd_mul_left_left", ab("a"), ab("b")))),
		ithm("Int.gcd_mul_left_right", "a b", nil, eqN(igcd("a", imul("b", "a")), ab("a")),
			via("b", "a", func(z string) string { return eqN(g(ab("a"), z), ab("a")) }, ap("Nat.gcd_mul_left_right", ab("b"), ab("a")))),
		ithm("Int.gcd_mul_right_left", "a b", nil, eqN(igcd(imul("b", "a"), "b"), ab("b")),
			via("b", "a", func(z string) string { return eqN(g(z, ab("b")), ab("b")) }, ap("Nat.gcd_mul_right_left", ab("a"), ab("b")))),
		ithm("Int.gcd_mul_right_right", "a b", nil, eqN(igcd("a", imul("a", "b")), ab("a")),
			via("a", "b", func(z string) string { return eqN(g(ab("a"), z), ab("a")) }, ap("Nat.gcd_mul_right_right", ab("b"), ab("a")))),
	)
	// both slots: gcd (m*n) (m*k) and friends
	two := func(f func(x, y string) string, x1, x2, y1, y2 string, rhs, body string) string { // f(|x1*x2|, |y1*y2|) = rhs
		e1 := cg(func(z string) string { return f(z, ab(imul(y1, y2))) }, ab(imul(x1, x2)), mulN(ab(x1), ab(x2)), nm(x1, x2))
		e2 := cg(func(z string) string { return f(mulN(ab(x1), ab(x2)), z) }, ab(imul(y1, y2)), mulN(ab(y1), ab(y2)), nm(y1, y2))
		return tr(f(ab(imul(x1, x2)), ab(imul(y1, y2))), f(mulN(ab(x1), ab(x2)), ab(imul(y1, y2))), rhs, e1,
			tr(f(mulN(ab(x1), ab(x2)), ab(imul(y1, y2))), f(mulN(ab(x1), ab(x2)), mulN(ab(y1), ab(y2))), rhs, e2, body))
	}
	lc := func(x, y string) string { return lcmN(x, y) }
	preludeExtra = append(preludeExtra,
		ithm("Int.gcd_mul_left", "m n k", nil, eqN(igcd(imul("m", "n"), imul("m", "k")), mulN(ab("m"), igcd("n", "k"))),
			two(g, "m", "n", "m", "k", mulN(ab("m"), igcd("n", "k")), ap("Nat.gcd_mul_left", ab("m"), ab("n"), ab("k")))),
		ithm("Int.gcd_mul_right", "m n k", nil, eqN(igcd(imul("m", "n"), imul("k", "n")), mulN(igcd("m", "k"), ab("n"))),
			two(g, "m", "n", "k", "n", mulN(igcd("m", "k"), ab("n")), ap("Nat.gcd_mul_right", ab("m"), ab("n"), ab("k")))),
		ithm("Int.lcm_mul_left", "m n k", nil, eqN(ilcm(imul("m", "n"), imul("m", "k")), mulN(ab("m"), ilcm("n", "k"))),
			two(lc, "m", "n", "m", "k", mulN(ab("m"), ilcm("n", "k")), ap("Nat.lcm_mul_left", ab("m"), ab("n"), ab("k")))),
		ithm("Int.lcm_mul_right", "m n k", nil, eqN(ilcm(imul("m", "n"), imul("k", "n")), mulN(ilcm("m", "k"), ab("n"))),
			two(lc, "m", "n", "k", "n", mulN(ilcm("m", "k"), ab("n")), ap("Nat.lcm_mul_right", ab("m"), ab("n"), ab("k")))),
		ithm("Int.gcd_mul_right_dvd_mul_gcd", "k m n", nil, dvdN(igcd("k", imul("m", "n")), mulN(igcd("k", "m"), igcd("k", "n"))),
			via("m", "n", func(z string) string { return dvdN(g(ab("k"), z), mulN(igcd("k", "m"), igcd("k", "n"))) }, ap("Nat.gcd_mul_right_dvd_mul_gcd", ab("k"), ab("m"), ab("n")))),
		ithm("Int.gcd_mul_left_dvd_mul_gcd", "k m n", nil, dvdN(igcd(imul("m", "n"), "k"), mulN(igcd("m", "k"), igcd("n", "k"))),
			via("m", "n", func(z string) string { return dvdN(g(z, ab("k")), mulN(igcd("m", "k"), igcd("n", "k"))) }, ap("Nat.gcd_mul_left_dvd_mul_gcd", ab("k"), ab("m"), ab("n")))),
	)
}
