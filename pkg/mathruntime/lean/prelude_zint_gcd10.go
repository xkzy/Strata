package lean

// Int: k ∣ (gcd ..) * m ↔ k ∣ n * m and its two twins, through natAbs_mul.
func init() {
	ab := iabs
	oN := icast
	nm := func(x, y string) string { return ap("Int.natAbs_mul", x, y) }
	idv := func(a, b string) string { return "(Int.dvd " + a + " " + b + ")" }
	toN := func(a, b, h string) string {
		return iffMpr(dvdN(ab(a), ab(b)), idv(a, b), ap("Int.natAbs_dvd_natAbs", a, b), h)
	}
	toI := func(a, b, h string) string {
		return iffMp(dvdN(ab(a), ab(b)), idv(a, b), ap("Int.natAbs_dvd_natAbs", a, b), h)
	}
	// k ∣ u*v ↔ k ∣ u2*v2 from natIff : |k| ∣ |u|*|v| ↔ |k| ∣ |u2|*|v2|
	mulIff := func(k, u, v, u2, v2, natIff string) string {
		L, R := idv(k, imul(u, v)), idv(k, imul(u2, v2))
		nl, nr := dvdN(ab(k), mulN(ab(u), ab(v))), dvdN(ab(k), mulN(ab(u2), ab(v2)))
		down := func(u, v, h string) string { // k ∣ u*v -> |k| ∣ |u|*|v|
			return tp(ab(imul(u, v)), mulN(ab(u), ab(v)), nm(u, v), func(z string) string { return dvdN(ab(k), z) }, toN(k, imul(u, v), h))
		}
		up := func(u, v, h string) string { // |k| ∣ |u|*|v| -> k ∣ u*v
			return toI(k, imul(u, v), tp(mulN(ab(u), ab(v)), ab(imul(u, v)), sym(ab(imul(u, v)), mulN(ab(u), ab(v)), nm(u, v)), func(z string) string { return dvdN(ab(k), z) }, h))
		}
		return iffI(L, R, "h", up(u2, v2, iffMp(nl, nr, natIff, down(u, v, "h"))), "h", up(u, v, iffMpr(nl, nr, natIff, down(u2, v2, "h"))))
	}
	iff := func(l, r string) string { return "(Iff " + l + " " + r + ")" }
	G := func(a, b string) string { return oN(igcd(a, b)) }
	preludeExtra = append(preludeExtra,
		ithm("Int.dvd_gcd_mul_iff_dvd_mul", "k n m", nil, iff(idv("k", imul(G("k", "n"), "m")), idv("k", imul("n", "m"))),
			mulIff("k", G("k", "n"), "m", "n", "m", ap("Nat.dvd_gcd_mul_iff_dvd_mul", ab("k"), ab("n"), ab("m")))),
		ithm("Int.dvd_mul_gcd_iff_dvd_mul", "k n m", nil, iff(idv("k", imul("n", G("k", "m"))), idv("k", imul("n", "m"))),
			mulIff("k", "n", G("k", "m"), "n", "m", ap("Nat.dvd_mul_gcd_iff_dvd_mul", ab("k"), ab("n"), ab("m")))),
		ithm("Int.dvd_gcd_mul_gcd_iff_dvd_mul", "k n m", nil, iff(idv("k", imul(G("k", "n"), G("k", "m"))), idv("k", imul("n", "m"))),
			mulIff("k", G("k", "n"), G("k", "m"), "n", "m", ap("Nat.dvd_gcd_mul_gcd_iff_dvd_mul", ab("k"), ab("n"), ab("m")))),
	)
}
