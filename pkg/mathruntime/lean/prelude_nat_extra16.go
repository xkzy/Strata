package lean

// The rest of Init/Data/Nat/Dvd.lean that has no `propext` / `eq_true` in it.
func init() {
	d := dvdN
	mm := func(x, y string) string { return "(Nat.mod " + x + " " + y + ")" }
	dd := func(x, y string) string { return "(Nat.div " + x + " " + y + ")" }
	tdz := func(k string) func(z string) string { return func(z string) string { return d(k, z) } }
	// dvd_add_iff_right (h : k ∣ m) : k ∣ n ↔ k ∣ m + n ; dvd_add_iff_left (h : k ∣ n) : k ∣ m ↔ k ∣ m + n
	mn := addN("m", "n")
	preludeExtra = append(preludeExtra,
		thm("Nat.dvd_mul_left_of_dvd", "a b", [][2]string{{"h", d("a", "b")}, {"c", "Nat"}}, d("a", mulN("c", "b")),
			ap("Nat.dvd_trans", "a", "b", mulN("c", "b"), "h", ap("Nat.dvd_mul_left", "b", "c"))),
		thm("Nat.dvd_add_iff_right", "k m n", [][2]string{{"h", d("k", "m")}}, iffN(d("k", "n"), d("k", mn)),
			iffI(d("k", "n"), d("k", mn), "H", ap("Nat.dvd_add", "k", "m", "n", "h", "H"),
				"H", tp(subN(mn, "m"), "n", ap("Nat.add_sub_cancel_left", "m", "n"), tdz("k"), ap("Nat.dvd_sub", "k", mn, "m", "H", "h")))),
		thm("Nat.dvd_add_iff_left", "k m n", [][2]string{{"h", d("k", "n")}}, iffN(d("k", "m"), d("k", mn)),
			iffI(d("k", "m"), d("k", mn), "H", ap("Nat.dvd_add", "k", "m", "n", "H", "h"),
				"H", tp(subN(mn, "n"), "m", ap("Nat.add_sub_cancel", "m", "n"), tdz("k"), ap("Nat.dvd_sub", "k", mn, "n", "H", "h")))),
	)
	// dvd_mod_iff (h : k ∣ n) : k ∣ m % n ↔ k ∣ m
	r, q := mm("m", "n"), mulN("n", dd("m", "n"))
	hq := ap("Nat.dvd_trans", "k", "n", q, "h", ap("Nat.dvd_mul_right", "n", dd("m", "n")))
	rq := addN(r, q)
	preludeExtra = append(preludeExtra,
		thm("Nat.dvd_mod_iff", "k m n", [][2]string{{"h", d("k", "n")}}, iffN(d("k", r), d("k", "m")),
			iffI(d("k", r), d("k", "m"), "H", tp(rq, "m", ap("Nat.mod_add_div", "m", "n"), tdz("k"), ap("Nat.dvd_add", "k", r, q, "H", hq)),
				"H", tp(subN(rq, q), r, ap("Nat.add_sub_cancel", r, q), tdz("k"),
					ap("Nat.dvd_sub", "k", rq, q, tp("m", rq, sym(rq, "m", ap("Nat.mod_add_div", "m", "n")), tdz("k"), "H"), hq)))),
		thm("Nat.emod_pos_of_not_dvd", "a b", [][2]string{{"h", notN(d("a", "b"))}}, ltN("0", mm("b", "a")),
			ap("Nat.pos_of_ne_zero", mm("b", "a"), "(fun (e : "+eqN(mm("b", "a"), "0")+") => h "+
				iffMpr(d("a", "b"), eqN(mm("b", "a"), "0"), ap("Nat.dvd_iff_mod_eq_zero", "a", "b"), "e")+")")),
		thm("Nat.dvd_of_mul_dvd_mul_right", "k m n", [][2]string{{"kpos", ltN("0", "k")}, {"H", d(mulN("m", "k"), mulN("n", "k"))}}, d("m", "n"),
			ap("Nat.dvd_of_mul_dvd_mul_left", "k", "m", "n", "kpos",
				tp(mulN("m", "k"), mulN("k", "m"), ap("Nat.mul_comm", "m", "k"), func(z string) string { return d(z, mulN("k", "n")) },
					tp(mulN("n", "k"), mulN("k", "n"), ap("Nat.mul_comm", "n", "k"), func(z string) string { return d(mulN("m", "k"), z) }, "H")))),
	)
	// dvd_sub_iff_right (hkn : k ≤ n) (h : m ∣ n) : m ∣ n - k ↔ m ∣ k
	nk := subN("n", "k")
	preludeExtra = append(preludeExtra,
		thm("Nat.dvd_sub_iff_right", "m n k", [][2]string{{"hkn", leN("k", "n")}, {"h", d("m", "n")}}, iffN(d("m", nk), d("m", "k")),
			iffI(d("m", nk), d("m", "k"), "H",
				iffMpr(d("m", "k"), d("m", addN(nk, "k")), ap("Nat.dvd_add_iff_right", "m", nk, "k", "H"),
					tp("n", addN(nk, "k"), sym(addN(nk, "k"), "n", ap("Nat.sub_add_cancel", "n", "k", "hkn")), tdz("m"), "h")),
				"H", ap("Nat.dvd_sub", "m", "n", "k", "h", "H"))),
		thm("Nat.dvd_sub_iff_left", "m n k", [][2]string{{"hkn", leN("k", "n")}, {"h", d("m", "k")}}, iffN(d("m", nk), d("m", "n")),
			iffI(d("m", nk), d("m", "n"), "H",
				tp(addN(nk, "k"), "n", ap("Nat.sub_add_cancel", "n", "k", "hkn"), tdz("m"), iffMp(d("m", nk), d("m", addN(nk, "k")), ap("Nat.dvd_add_iff_left", "m", nk, "k", "h"), "H")),
				"H", iffMpr(d("m", nk), d("m", addN(nk, "k")), ap("Nat.dvd_add_iff_left", "m", nk, "k", "h"),
					tp("n", addN(nk, "k"), sym(addN(nk, "k"), "n", ap("Nat.sub_add_cancel", "n", "k", "hkn")), tdz("m"), "H")))),
	)
}
