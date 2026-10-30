package lean

import "fmt"

// Int arithmetic the gcd-with-sums lemmas need, then gcd (m) (n + k*m) = gcd m n and its sum / difference variants.

// Int-valued equality helpers: itr = Eq.trans, isym = Eq.symm, icg = congrArg Int Int, icgN = congrArg Int Nat.
func itr(a, b, c, p, q string) string {
	return "(Eq.trans.{1} Int " + a + " " + b + " " + c + " " + p + " " + q + ")"
}
func isym(a, b, p string) string { return "(Eq.symm.{1} Int " + a + " " + b + " " + p + ")" }
func icg(f func(z string) string, a, b, p string) string {
	return "(congrArg.{1, 1} Int Int (fun (z : Int) => " + f("z") + ") " + a + " " + b + " " + p + ")"
}
func icgN(f func(z string) string, a, b, p string) string {
	return "(congrArg.{1, 1} Int Nat (fun (z : Int) => " + f("z") + ") " + a + " " + b + " " + p + ")"
}
func iaddN(a, b string) string { return "(Int.add " + a + " " + b + ")" }
func isubN(a, b string) string { return "(Int.sub " + a + " " + b + ")" }
func ineg(a string) string     { return "(Int.neg " + a + ")" }
func idvd(a, b string) string  { return "(Int.dvd " + a + " " + b + ")" }

// iDvIntro destructs `h : m ∣ k` (Int) into `k = m * l` for the goal.
var iDvN int

func iDvIntro(m, k, h, goal string, body func(l, hl string) string) string {
	iDvN++
	l, hl := fmt.Sprintf("l%d_", iDvN), fmt.Sprintf("hl%d_", iDvN)
	return "(Exists.rec.{1} Int (fun (w_ : Int) => Eq.{1} Int " + k + " " + imul(m, "w_") + ") (fun (_ : " + idvd(m, k) +
		") => " + goal + ") (fun (" + l + " : Int) (" + hl + " : Eq.{1} Int " + k + " " + imul(m, l) + ") => " + body(l, hl) + ") " + h + ")"
}

func init() {
	// dvd_add / dvd_sub / add_sub_cancel
	two := func(op string, name string, mulOp string) valDecl {
		res := func(x, y string) string { return "(Int." + op + " " + x + " " + y + ")" }
		goal := idvd("a", res("b", "c"))
		return ithm(name, "a b c", [][2]string{{"h1", idvd("a", "b")}, {"h2", idvd("a", "c")}}, goal,
			iDvIntro("a", "b", "h1", goal, func(x, hx string) string {
				return iDvIntro("a", "c", "h2", goal, func(y, hy string) string {
					sum := res(imul("a", x), imul("a", y))
					return "(Exists.intro.{1} Int (fun (d : Int) => Eq.{1} Int " + res("b", "c") + " " + imul("a", "d") + ") " + res(x, y) + " " +
						itr(res("b", "c"), sum, imul("a", res(x, y)),
							itr(res("b", "c"), res(imul("a", x), "c"), sum,
								icg(func(z string) string { return res(z, "c") }, "b", imul("a", x), hx),
								icg(func(z string) string { return res(imul("a", x), z) }, "c", imul("a", y), hy)),
							isym(imul("a", res(x, y)), sum, ap(mulOp, "a", x, y))) + ")"
				})
			}))
	}
	preludeExtra = append(preludeExtra,
		two("add", "Int.dvd_add", "Int.mul_add"),
		two("sub", "Int.dvd_sub", "Int.mul_sub"),
		ithm("Int.add_sub_cancel", "a b", nil, ieq(isubN(iaddN("a", "b"), "b"), "a"),
			itr(isubN(iaddN("a", "b"), "b"), iaddN(iaddN("a", "b"), ineg("b")), "a", ap("Int.sub_eq_add_neg", iaddN("a", "b"), "b"), ap("Int.add_neg_cancel_right", "a", "b"))),
	)
	// core: gcd m (n + k*m) = gcd m n, from gcd_right_eq_iff
	m, n, k := "m", "n", "k"
	km := imul(k, m)
	X := iaddN(n, km)
	hkm := func(c string) string { return ap("Int.dvd_trans", c, m, km, "hc", ap("Int.dvd_mul_left", m, k)) }
	core := iffMpr(eqN(igcd(m, X), igcd(m, n)), "((c : Int) → "+idvd("c", m)+" → (Iff "+idvd("c", X)+" "+idvd("c", n)+"))",
		ap("Int.gcd_right_eq_iff", m, X, n),
		"(fun (c : Int) (hc : "+idvd("c", m)+") => "+iffI(idvd("c", X), idvd("c", n), "x",
			tpI(isubN(X, km), n, ap("Int.add_sub_cancel", n, km), func(z string) string { return idvd("c", z) },
				ap("Int.dvd_sub", "c", X, km, "x", hkm("c"))),
			"x", ap("Int.dvd_add", "c", n, km, "x", hkm("c")))+")")
	preludeExtra = append(preludeExtra,
		ithm("Int.gcd_add_mul_right_right", "m n k", nil, eqN(igcd(m, X), igcd(m, n)), core))
}
