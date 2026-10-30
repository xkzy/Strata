package lean

// Int lcm characterised by divisibility, and the zero / positivity iffs.
func init() {
	ab := iabs
	idv := func(a, b string) string { return "(Int.dvd " + a + " " + b + ")" }
	toN := func(a, b, h string) string {
		return iffMpr(dvdN(ab(a), ab(b)), idv(a, b), ap("Int.natAbs_dvd_natAbs", a, b), h)
	}
	toI := func(a, b, h string) string {
		return iffMp(dvdN(ab(a), ab(b)), idv(a, b), ap("Int.natAbs_dvd_natAbs", a, b), h)
	}
	oN := icast
	iff := func(l, r string) string { return "(Iff " + l + " " + r + ")" }
	lab := ilcm("a", "b")
	L := oN(lab)
	hs := [][2]string{{"ha", idv("a", "c")}, {"hb", idv("b", "c")}}
	preludeExtra = append(preludeExtra,
		ithm("Int.dvd_lcm_left", "a b", nil, idv("a", L), toI("a", L, ap("Nat.dvd_lcm_left", ab("a"), ab("b")))),
		ithm("Int.dvd_lcm_right", "a b", nil, idv("b", L), toI("b", L, ap("Nat.dvd_lcm_right", ab("a"), ab("b")))),
		ithm("Int.coe_lcm_dvd", "a b c", hs, idv(L, "c"),
			toI(L, "c", ap("Nat.lcm_dvd", ab("a"), ab("b"), ab("c"), toN("a", "c", "ha"), toN("b", "c", "hb")))),
		ithm("Int.lcm_dvd", "a b", [][2]string{{"c", "Nat"}, {"ha", idv("a", oN("c"))}, {"hb", idv("b", oN("c"))}}, dvdN(lab, "c"),
			ap("Nat.lcm_dvd", ab("a"), ab("b"), "c", toN("a", oN("c"), "ha"), toN("b", oN("c"), "hb"))),
		ithm("Int.coe_lcm_dvd_iff", "a b c", nil, iff(idv(L, "c"), andN(idv("a", "c"), idv("b", "c"))),
			iffI(idv(L, "c"), andN(idv("a", "c"), idv("b", "c")), "h",
				ap("And.intro", idv("a", "c"), idv("b", "c"),
					ap("Int.dvd_trans", "a", L, "c", ap("Int.dvd_lcm_left", "a", "b"), "h"),
					ap("Int.dvd_trans", "b", L, "c", ap("Int.dvd_lcm_right", "a", "b"), "h")),
				"h", ap("Int.coe_lcm_dvd", "a", "b", "c", ap("And.left", idv("a", "c"), idv("b", "c"), "h"), ap("And.right", idv("a", "c"), idv("b", "c"), "h")))),
		ithm("Int.lcm_dvd_iff", "a b", [][2]string{{"c", "Nat"}}, iff(dvdN(lab, "c"), andN(idv("a", oN("c")), idv("b", oN("c")))),
			iffI(dvdN(lab, "c"), andN(idv("a", oN("c")), idv("b", oN("c"))), "h",
				ap("And.intro", idv("a", oN("c")), idv("b", oN("c")),
					ap("Int.dvd_trans", "a", L, oN("c"), ap("Int.dvd_lcm_left", "a", "b"), toI(L, oN("c"), "h")),
					ap("Int.dvd_trans", "b", L, oN("c"), ap("Int.dvd_lcm_right", "a", "b"), toI(L, oN("c"), "h"))),
				"h", ap("Int.lcm_dvd", "a", "b", "c", ap("And.left", idv("a", oN("c")), idv("b", oN("c")), "h"), ap("And.right", idv("a", oN("c")), idv("b", oN("c")), "h")))),
	)
	nl, nr := ap("Nat.lcm_eq_left_iff_dvd", ab("a"), ab("b")), ap("Nat.lcm_eq_right_iff_dvd", ab("a"), ab("b"))
	preludeExtra = append(preludeExtra,
		ithm("Int.lcm_eq_natAbs_left_iff_dvd", "a b", nil, iff(eqN(lab, ab("a")), idv("b", "a")),
			iffI(eqN(lab, ab("a")), idv("b", "a"), "h", toI("b", "a", iffMp(eqN(lab, ab("a")), dvdN(ab("b"), ab("a")), nl, "h")),
				"h", iffMpr(eqN(lab, ab("a")), dvdN(ab("b"), ab("a")), nl, toN("b", "a", "h")))),
		ithm("Int.lcm_eq_natAbs_right_iff_dvd", "a b", nil, iff(eqN(lab, ab("b")), idv("a", "b")),
			iffI(eqN(lab, ab("b")), idv("a", "b"), "h", toI("a", "b", iffMp(eqN(lab, ab("b")), dvdN(ab("a"), ab("b")), nr, "h")),
				"h", iffMpr(eqN(lab, ab("b")), dvdN(ab("a"), ab("b")), nr, toN("a", "b", "h")))),
		ithm("Int.lcm_eq_natAbs_left", "a b", [][2]string{{"h", idv("b", "a")}}, eqN(lab, ab("a")),
			iffMpr(eqN(lab, ab("a")), idv("b", "a"), ap("Int.lcm_eq_natAbs_left_iff_dvd", "a", "b"), "h")),
		ithm("Int.lcm_eq_natAbs_right", "a b", [][2]string{{"h", idv("a", "b")}}, eqN(lab, ab("b")),
			iffMpr(eqN(lab, ab("b")), idv("a", "b"), ap("Int.lcm_eq_natAbs_right_iff_dvd", "a", "b"), "h")),
	)
	le0 := func(x string) string { return "(Int.le (Int.ofNat 0) " + x + ")" }
	cI := func(x, y, p string) string {
		return "(congrArg.{1, 1} Int Nat (fun (z : Int) => (Int.natAbs z)) " + x + " " + y + " " + p + ")"
	}
	castIff := func(x, y, natIff string) string { // Iff (ofNat lcm = x) (y ∣ x) from natIff : Iff (lcm = |x|) (y ∣ x)
		return iffI(ieq(L, x), idv(y, x), "h",
			iffMp(eqN(lab, ab(x)), idv(y, x), natIff, "(Eq.trans.{1} Nat "+lab+" "+ab(L)+" "+ab(x)+" (Eq.refl.{1} Nat "+lab+") "+cI(L, x, "h")+")"),
			"h", "(Eq.trans.{1} Int "+L+" "+oN(ab(x))+" "+x+" (congrArg.{1, 1} Nat Int (fun (z : Nat) => Int.ofNat z) "+lab+" "+ab(x)+" "+
				iffMpr(eqN(lab, ab(x)), idv(y, x), natIff, "h")+") "+ap("Int.ofNat_natAbs_of_nonneg", x, "hx")+")")
	}
	preludeExtra = append(preludeExtra,
		ithm("Int.lcm_eq_left_iff_dvd", "a b", [][2]string{{"hx", le0("a")}}, iff(ieq(L, "a"), idv("b", "a")),
			castIff("a", "b", ap("Int.lcm_eq_natAbs_left_iff_dvd", "a", "b"))),
		ithm("Int.lcm_eq_right_iff_dvd", "a b", [][2]string{{"hx", le0("b")}}, iff(ieq(L, "b"), idv("a", "b")),
			castIff("b", "a", ap("Int.lcm_eq_natAbs_right_iff_dvd", "a", "b"))),
	)
	// gcd = 0 ↔ both 0; 0 < gcd ↔ one is nonzero
	z := func(x string) string { return ieq(x, iz()) }
	fromNat0 := func(x, h string) string { return iffMp(eqN(ab(x), "0"), z(x), ap("Int.natAbs_eq_zero", x), h) }
	toNat0 := func(x, h string) string { return iffMpr(eqN(ab(x), "0"), z(x), ap("Int.natAbs_eq_zero", x), h) }
	gz := eqN(igcd("a", "b"), "0")
	pz := andN(z("a"), z("b"))
	nat0 := ap("Nat.gcd_eq_zero_iff", ab("a"), ab("b"))
	preludeExtra = append(preludeExtra,
		ithm("Int.gcd_eq_zero_iff", "a b", nil, iff(gz, pz),
			iffI(gz, pz, "h",
				ap("And.intro", z("a"), z("b"),
					fromNat0("a", ap("And.left", eqN(ab("a"), "0"), eqN(ab("b"), "0"), iffMp(gz, andN(eqN(ab("a"), "0"), eqN(ab("b"), "0")), nat0, "h"))),
					fromNat0("b", ap("And.right", eqN(ab("a"), "0"), eqN(ab("b"), "0"), iffMp(gz, andN(eqN(ab("a"), "0"), eqN(ab("b"), "0")), nat0, "h")))),
				"h", iffMpr(gz, andN(eqN(ab("a"), "0"), eqN(ab("b"), "0")), nat0,
					ap("And.intro", eqN(ab("a"), "0"), eqN(ab("b"), "0"),
						toNat0("a", ap("And.left", z("a"), z("b"), "h")), toNat0("b", ap("And.right", z("a"), z("b"), "h")))))),
	)
}
