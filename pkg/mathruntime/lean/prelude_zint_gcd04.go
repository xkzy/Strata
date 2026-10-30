package lean

// Int gcd characterised by divisibility: dvd_gcd, gcd_eq_*_iff_dvd and friends.
func init() {
	ab := iabs
	idv := func(a, b string) string { return "(Int.dvd " + a + " " + b + ")" }
	toN := func(a, b, h string) string {
		return iffMpr(dvdN(ab(a), ab(b)), idv(a, b), ap("Int.natAbs_dvd_natAbs", a, b), h)
	} // Int.dvd -> |a| ∣ |b|
	toI := func(a, b, h string) string {
		return iffMp(dvdN(ab(a), ab(b)), idv(a, b), ap("Int.natAbs_dvd_natAbs", a, b), h)
	}
	oN := icast
	gab := igcd("a", "b")
	chk := [][2]string{{"ha", idv("c", "a")}, {"hb", idv("c", "b")}}
	dgcdIff := func(l, r string) string { return "(Iff " + l + " " + r + ")" }
	preludeExtra = append(preludeExtra,
		ithm("Int.dvd_coe_gcd", "c a b", chk, idv("c", oN(gab)),
			toI("c", oN(gab), ap("Nat.dvd_gcd", ab("c"), ab("a"), ab("b"), toN("c", "a", "ha"), toN("c", "b", "hb")))),
		ithm("Int.dvd_gcd", "a b", [][2]string{{"c", "Nat"}, {"ha", idv(oN("c"), "a")}, {"hb", idv(oN("c"), "b")}}, dvdN("c", gab),
			ap("Nat.dvd_gcd", "c", ab("a"), ab("b"), toN(oN("c"), "a", "ha"), toN(oN("c"), "b", "hb"))),
		ithm("Int.dvd_coe_gcd_iff", "a b c", nil, dgcdIff(idv("c", oN(gab)), andN(idv("c", "a"), idv("c", "b"))),
			iffI(idv("c", oN(gab)), andN(idv("c", "a"), idv("c", "b")), "h",
				ap("And.intro", idv("c", "a"), idv("c", "b"),
					ap("Int.dvd_trans", "c", oN(gab), "a", "h", ap("Int.gcd_dvd_left", "a", "b")),
					ap("Int.dvd_trans", "c", oN(gab), "b", "h", ap("Int.gcd_dvd_right", "a", "b"))),
				"h", ap("Int.dvd_coe_gcd", "c", "a", "b", ap("And.left", idv("c", "a"), idv("c", "b"), "h"), ap("And.right", idv("c", "a"), idv("c", "b"), "h")))),
		ithm("Int.dvd_gcd_iff", "a b", [][2]string{{"c", "Nat"}}, dgcdIff(dvdN("c", gab), andN(idv(oN("c"), "a"), idv(oN("c"), "b"))),
			iffI(dvdN("c", gab), andN(idv(oN("c"), "a"), idv(oN("c"), "b")), "h",
				ap("And.intro", idv(oN("c"), "a"), idv(oN("c"), "b"),
					ap("Int.dvd_trans", oN("c"), oN(gab), "a", toI(oN("c"), oN(gab), "h"), ap("Int.gcd_dvd_left", "a", "b")),
					ap("Int.dvd_trans", oN("c"), oN(gab), "b", toI(oN("c"), oN(gab), "h"), ap("Int.gcd_dvd_right", "a", "b"))),
				"h", ap("Int.dvd_gcd", "a", "b", "c", ap("And.left", idv(oN("c"), "a"), idv(oN("c"), "b"), "h"), ap("And.right", idv(oN("c"), "a"), idv(oN("c"), "b"), "h")))),
	)
	// gcd = |a| ↔ a ∣ b, and the right-hand twin
	nl, nr := ap("Nat.gcd_eq_left_iff_dvd", ab("a"), ab("b")), ap("Nat.gcd_eq_right_iff_dvd", ab("a"), ab("b"))
	preludeExtra = append(preludeExtra,
		ithm("Int.gcd_eq_natAbs_left_iff_dvd", "a b", nil, dgcdIff(eqN(gab, ab("a")), idv("a", "b")),
			iffI(eqN(gab, ab("a")), idv("a", "b"), "h", toI("a", "b", iffMp(eqN(gab, ab("a")), dvdN(ab("a"), ab("b")), nl, "h")),
				"h", iffMpr(eqN(gab, ab("a")), dvdN(ab("a"), ab("b")), nl, toN("a", "b", "h")))),
		ithm("Int.gcd_eq_natAbs_right_iff_dvd", "a b", nil, dgcdIff(eqN(gab, ab("b")), idv("b", "a")),
			iffI(eqN(gab, ab("b")), idv("b", "a"), "h", toI("b", "a", iffMp(eqN(gab, ab("b")), dvdN(ab("b"), ab("a")), nr, "h")),
				"h", iffMpr(eqN(gab, ab("b")), dvdN(ab("b"), ab("a")), nr, toN("b", "a", "h")))),
		ithm("Int.gcd_eq_natAbs_left", "a b", [][2]string{{"h", idv("a", "b")}}, eqN(gab, ab("a")),
			iffMpr(eqN(gab, ab("a")), idv("a", "b"), ap("Int.gcd_eq_natAbs_left_iff_dvd", "a", "b"), "h")),
		ithm("Int.gcd_eq_natAbs_right", "a b", [][2]string{{"h", idv("b", "a")}}, eqN(gab, ab("b")),
			iffMpr(eqN(gab, ab("b")), idv("b", "a"), ap("Int.gcd_eq_natAbs_right_iff_dvd", "a", "b"), "h")),
	)
	// gcd a b = a (as Int) for 0 ≤ a
	le0 := func(x string) string { return "(Int.le (Int.ofNat 0) " + x + ")" }
	cI := func(f, x, y, p string) string {
		return "(congrArg.{1, 1} Int Nat (fun (z : Int) => " + f + ") " + x + " " + y + " " + p + ")"
	}
	mkEq := func(x, y, natIff string) string { // Iff (ofNat (gcd a b) = x) (x ∣ y) from natIff : Iff (gcd a b = |x|) (x ∣ y)
		gg := oN(gab)
		return iffI(ieq(gg, x), idv(x, y), "h",
			iffMp(eqN(gab, ab(x)), idv(x, y), natIff, "(Eq.trans.{1} Nat "+gab+" "+ab(gg)+" "+ab(x)+" (Eq.refl.{1} Nat "+gab+") "+cI(ab("z"), gg, x, "h")+")"),
			"h", "(Eq.trans.{1} Int "+gg+" "+oN(ab(x))+" "+x+" (congrArg.{1, 1} Nat Int (fun (z : Nat) => Int.ofNat z) "+gab+" "+ab(x)+" "+
				iffMpr(eqN(gab, ab(x)), idv(x, y), natIff, "h")+") "+ap("Int.ofNat_natAbs_of_nonneg", x, "hx")+")")
	}
	preludeExtra = append(preludeExtra,
		ithm("Int.gcd_eq_left_iff_dvd", "a b", [][2]string{{"hx", le0("a")}}, dgcdIff(ieq(oN(gab), "a"), idv("a", "b")),
			mkEq("a", "b", ap("Int.gcd_eq_natAbs_left_iff_dvd", "a", "b"))),
		ithm("Int.gcd_eq_right_iff_dvd", "a b", [][2]string{{"hx", le0("b")}}, dgcdIff(ieq(oN(gab), "b"), idv("b", "a")),
			mkEq("b", "a", ap("Int.gcd_eq_natAbs_right_iff_dvd", "a", "b"))),
		ithm("Int.gcd_eq_left", "a b", [][2]string{{"ha", le0("a")}, {"h", idv("a", "b")}}, ieq(oN(gab), "a"),
			iffMpr(ieq(oN(gab), "a"), idv("a", "b"), ap("Int.gcd_eq_left_iff_dvd", "a", "b", "ha"), "h")),
		ithm("Int.gcd_eq_right", "a b", [][2]string{{"hb", le0("b")}, {"h", idv("b", "a")}}, ieq(oN(gab), "b"),
			iffMpr(ieq(oN(gab), "b"), idv("b", "a"), ap("Int.gcd_eq_right_iff_dvd", "a", "b", "hb"), "h")),
	)
}
