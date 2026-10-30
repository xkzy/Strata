package lean

// ex2 destructs `h : l ∣ m * n` as `a * b = l` with `a ∣ m`, `b ∣ n` (Nat.dvdProd_exists) for the goal.
func ex2(l, m, n, h, goal string, body func(a, b, ha, hb, e string) string) string {
	inner := func(a string) string {
		return "(fun (b : Nat) => And (And " + dvdN(a, m) + " " + dvdN("b", n) + ") " + eqN(mulN(a, "b"), l) + ")"
	}
	outer := "(fun (a : Nat) => Exists.{1} Nat " + inner("a") + ")"
	ex := func(a string) string { return "(Exists.{1} Nat " + inner(a) + ")" }
	pair := func(a, b string) string {
		return andN(andN(dvdN(a, m), dvdN(b, n)), eqN(mulN(a, b), l))
	}
	left := func(a, b, hb string) string {
		return ap("And.left", andN(dvdN(a, m), dvdN(b, n)), eqN(mulN(a, b), l), hb)
	}
	return "(Exists.rec.{1} Nat " + outer + " (fun (_ : Exists.{1} Nat " + outer + ") => " + goal + ") " +
		"(fun (a : Nat) (ha : " + ex("a") + ") => Exists.rec.{1} Nat " + inner("a") + " (fun (_ : " + ex("a") + ") => " + goal + ") " +
		"(fun (b : Nat) (hb : " + pair("a", "b") + ") => " + body("a", "b",
		ap("And.left", dvdN("a", m), dvdN("b", n), left("a", "b", "hb")),
		ap("And.right", dvdN("a", m), dvdN("b", n), left("a", "b", "hb")),
		ap("And.right", andN(dvdN("a", m), dvdN("b", n)), eqN(mulN("a", "b"), l), "hb")) + ") ha) " +
		ap("Nat.dvdProd_exists", l, m, n, h) + ")"
}

func init() {
	no := [][2]string{}
	comm := func(a, b string) string { return ap("Nat.gcd_comm", a, b) }
	dvk := func(z string) string { return dvdN("k", z) }
	mc := func(a, b string) string { return ap("Nat.mul_comm", a, b) }
	// dvd_gcd_mul_iff_dvd_mul (k n m): k ∣ gcd k n * m ↔ k ∣ n * m
	gkn := gcdN("k", "n")
	L1, R1 := dvdN("k", mulN(gkn, "m")), dvdN("k", mulN("n", "m"))
	nm := mulN("n", "m")
	preludeExtra = append(preludeExtra,
		thm("Nat.dvd_gcd_mul_iff_dvd_mul", "k n m", no, iffN(L1, R1),
			iffI(L1, R1, "h",
				ap("Nat.dvd_trans", "k", mulN(gkn, "m"), nm, "h",
					ap("Nat.mul_dvd_mul_right", gkn, "n", "m", ap("Nat.gcd_dvd_right", "k", "n"))),
				"h",
				dvIntro("k", nm, "h", L1, func(y, hy string) string {
					gmy := mulN("k", gcdN("m", y))
					e := tr(mulN(gkn, "m"), gcdN(mulN("k", "m"), nm), gmy,
						sym(gcdN(mulN("k", "m"), nm), mulN(gkn, "m"), ap("Nat.gcd_mul_right", "k", "m", "n")),
						tr(gcdN(mulN("k", "m"), nm), gcdN(mulN("k", "m"), mulN("k", y)), gmy,
							cg(func(z string) string { return gcdN(mulN("k", "m"), z) }, nm, mulN("k", y), hy),
							ap("Nat.gcd_mul_left", "k", "m", y)))
					return tp(gmy, mulN(gkn, "m"), sym(mulN(gkn, "m"), gmy, e), dvk, ap("Nat.dvd_mul_right", "k", gcdN("m", y)))
				}))),
	)
	// dvd_mul_gcd_iff_dvd_mul (k n m): k ∣ n * gcd k m ↔ k ∣ n * m
	gkm := gcdN("k", "m")
	L2, R2 := dvdN("k", mulN("n", gkm)), dvdN("k", nm)
	A := ap("Nat.dvd_gcd_mul_iff_dvd_mul", "k", "m", "n") // k ∣ gcd k m * n ↔ k ∣ m * n
	Aa, Ab := dvdN("k", mulN(gkm, "n")), dvdN("k", mulN("m", "n"))
	preludeExtra = append(preludeExtra,
		thm("Nat.dvd_mul_gcd_iff_dvd_mul", "k n m", no, iffN(L2, R2),
			iffI(L2, R2, "h",
				tp(mulN("m", "n"), nm, mc("m", "n"), dvk,
					iffMp(Aa, Ab, A, tp(mulN("n", gkm), mulN(gkm, "n"), mc("n", gkm), dvk, "h"))),
				"h",
				tp(mulN(gkm, "n"), mulN("n", gkm), mc(gkm, "n"), dvk,
					iffMpr(Aa, Ab, A, tp(nm, mulN("m", "n"), mc("n", "m"), dvk, "h"))))),
	)
	// dvd_gcd_mul_gcd_iff_dvd_mul (k n m)
	L3 := dvdN("k", mulN(gkn, gkm))
	B1 := ap("Nat.dvd_gcd_mul_iff_dvd_mul", "k", "n", gkm) // k ∣ gcd k n * gcd k m ↔ k ∣ n * gcd k m
	B1b := dvdN("k", mulN("n", gkm))
	B2 := ap("Nat.dvd_mul_gcd_iff_dvd_mul", "k", "n", "m")
	preludeExtra = append(preludeExtra,
		thm("Nat.dvd_gcd_mul_gcd_iff_dvd_mul", "k n m", no, iffN(L3, R2),
			iffI(L3, R2, "h", iffMp(L2, R2, B2, iffMp(L3, B1b, B1, "h")), "h",
				iffMpr(L3, B1b, B1, iffMpr(L2, R2, B2, "h")))),
		thm("Nat.gcd_mul_left_dvd_mul_gcd", "k m n", no,
			dvdN(gcdN(mulN("m", "n"), "k"), mulN(gcdN("m", "k"), gcdN("n", "k"))),
			tp(gcdN("k", mulN("m", "n")), gcdN(mulN("m", "n"), "k"), comm("k", mulN("m", "n")),
				func(z string) string { return dvdN(z, mulN(gcdN("m", "k"), gcdN("n", "k"))) },
				tp(mulN(gcdN("k", "m"), gcdN("k", "n")), mulN(gcdN("m", "k"), gcdN("n", "k")),
					tr(mulN(gcdN("k", "m"), gcdN("k", "n")), mulN(gcdN("m", "k"), gcdN("k", "n")), mulN(gcdN("m", "k"), gcdN("n", "k")),
						cg(func(z string) string { return mulN(z, gcdN("k", "n")) }, gcdN("k", "m"), gcdN("m", "k"), comm("k", "m")),
						cg(func(z string) string { return mulN(gcdN("m", "k"), z) }, gcdN("k", "n"), gcdN("n", "k"), comm("k", "n"))),
					func(z string) string { return dvdN(gcdN("k", mulN("m", "n")), z) },
					ap("Nat.gcd_mul_right_dvd_mul_gcd", "k", "m", "n")))),
	)
	// the `_of_gcd_eq_one` family: gcd n m = 1
	h1 := [][2]string{{"h", eqN(gcdN("n", "m"), "1")}}
	oneIff := func(l string) string { return "" }
	_ = oneIff
	preludeExtra = append(preludeExtra,
		thm("Nat.gcd_mul_right_right_of_gcd_eq_one", "n m k", h1, eqN(gcdN("n", mulN("m", "k")), gcdN("n", "k")),
			iffMpr(eqN(gcdN("n", mulN("m", "k")), gcdN("n", "k")),
				fa("l", imp(dvdN("l", "n"), iffN(dvdN("l", mulN("m", "k")), dvdN("l", "k")))),
				ap("Nat.gcd_right_eq_iff", "n", mulN("m", "k"), "k"),
				"(fun (l : Nat) (hl : "+dvdN("l", "n")+") => "+iffI(dvdN("l", mulN("m", "k")), dvdN("l", "k"), "hd",
					ex2("l", "m", "k", "hd", dvdN("l", "k"), func(a, b, ha, hb, e string) string {
						// a ∣ l ∣ n and a ∣ m, so a ∣ gcd n m = 1
						dl := tp(mulN(a, b), "l", e, func(z string) string { return dvdN(a, z) }, ap("Nat.dvd_mul_right", a, b))
						a1 := ap("Nat.eq_one_of_dvd_one", a, tp(gcdN("n", "m"), "1", "h", func(z string) string { return dvdN(a, z) },
							ap("Nat.dvd_gcd", a, "n", "m", ap("Nat.dvd_trans", a, "l", "n", dl, "hl"), ha)))
						leq := tr("l", mulN(a, b), b, sym(mulN(a, b), "l", e),
							tr(mulN(a, b), mulN("1", b), b, cg(func(z string) string { return mulN(z, b) }, a, "1", a1), ap("Nat.one_mul", b)))
						return tp(b, "l", sym("l", b, leq), func(z string) string { return dvdN(z, "k") }, hb)
					}),
					"hk",
					ap("Nat.dvd_trans", "l", "k", mulN("m", "k"), "hk", ap("Nat.dvd_mul_left", "k", "m")))+")")),
		thm("Nat.gcd_mul_left_right_of_gcd_eq_one", "n m k", h1, eqN(gcdN("n", mulN("k", "m")), gcdN("n", "k")),
			tr(gcdN("n", mulN("k", "m")), gcdN("n", mulN("m", "k")), gcdN("n", "k"),
				cg(func(z string) string { return gcdN("n", z) }, mulN("k", "m"), mulN("m", "k"), mc("k", "m")),
				ap("Nat.gcd_mul_right_right_of_gcd_eq_one", "n", "m", "k", "h"))),
		thm("Nat.gcd_mul_right_left_of_gcd_eq_one", "n m k", h1, eqN(gcdN(mulN("n", "k"), "m"), gcdN("k", "m")),
			tr(gcdN(mulN("n", "k"), "m"), gcdN("m", mulN("n", "k")), gcdN("k", "m"), comm(mulN("n", "k"), "m"),
				tr(gcdN("m", mulN("n", "k")), gcdN("m", "k"), gcdN("k", "m"),
					ap("Nat.gcd_mul_right_right_of_gcd_eq_one", "m", "n", "k", tr(gcdN("m", "n"), gcdN("n", "m"), "1", comm("m", "n"), "h")),
					comm("m", "k")))),
		thm("Nat.gcd_mul_left_left_of_gcd_eq_one", "n m k", h1, eqN(gcdN(mulN("k", "n"), "m"), gcdN("k", "m")),
			tr(gcdN(mulN("k", "n"), "m"), gcdN(mulN("n", "k"), "m"), gcdN("k", "m"),
				cg(func(z string) string { return gcdN(z, "m") }, mulN("k", "n"), mulN("n", "k"), mc("k", "n")),
				ap("Nat.gcd_mul_right_left_of_gcd_eq_one", "n", "m", "k", "h"))),
		// n ^ k with gcd n m = 1
		thm("Nat.gcd_pow_left_of_gcd_eq_one", "n m k", h1, eqN(gcdN(ap("Nat.pow", "n", "k"), "m"), "1"),
			"Nat.rec.{0} (fun (k : Nat) => "+eqN(gcdN(ap("Nat.pow", "n", "k"), "m"), "1")+") "+
				"(Nat.gcd_one_left m) "+
				"(fun (k : Nat) (ih : "+eqN(gcdN(ap("Nat.pow", "n", "k"), "m"), "1")+") => "+
				tr(gcdN(ap("Nat.pow", "n", "(Nat.succ k)"), "m"), gcdN(ap("Nat.pow", "n", "k"), "m"), "1",
					tr(gcdN(ap("Nat.pow", "n", "(Nat.succ k)"), "m"), gcdN(mulN(ap("Nat.pow", "n", "k"), "n"), "m"), gcdN(ap("Nat.pow", "n", "k"), "m"),
						cg(func(z string) string { return gcdN(z, "m") }, ap("Nat.pow", "n", "(Nat.succ k)"), mulN(ap("Nat.pow", "n", "k"), "n"), ap("Nat.pow_succ", "n", "k")),
						ap("Nat.gcd_mul_left_left_of_gcd_eq_one", "n", "m", ap("Nat.pow", "n", "k"), "h")), "ih")+") k"),
	)
}
