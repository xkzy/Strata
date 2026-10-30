package lean

// Int gcd: the `∀ c` characterisations, lifted from the Nat ones through |c| and Int.ofNat k.
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
	faI := func(v, body string) string { return "((" + v + " : Int) → " + body + ")" }
	faN := func(v, body string) string { return "((" + v + " : Nat) → " + body + ")" }
	imp2 := func(a, b string) string { return "(" + a + " → " + b + ")" }
	pos := func(x string) string { return ltN("0", ab(x)) }
	nzI := func(x string) string { return inz(x) }
	// 0 < gcd a b ↔ a ≠ 0 ∨ b ≠ 0
	natPos := ap("Nat.gcd_pos_iff", ab("a"), ab("b"))
	orN := "(Or " + pos("a") + " " + pos("b") + ")"
	orI := "(Or " + nzI("a") + " " + nzI("b") + ")"
	gp := ltN("0", igcd("a", "b"))
	np := func(x string) string { return ap("Int.natAbs_pos", x) }
	pa := func(h string, x string) string { return iffMp(pos(x), nzI(x), np(x), h) }
	pb := func(h string, x string) string { return iffMpr(pos(x), nzI(x), np(x), h) }
	preludeExtra = append(preludeExtra,
		ithm("Int.gcd_pos_iff", "a b", nil, iff(gp, orI),
			iffI(gp, orI, "h",
				"Or.rec "+pos("a")+" "+pos("b")+" (fun (_ : "+orN+") => "+orI+") "+
					"(fun (x : "+pos("a")+") => "+ap("Or.inl", nzI("a"), nzI("b"), pa("x", "a"))+") "+
					"(fun (x : "+pos("b")+") => "+ap("Or.inr", nzI("a"), nzI("b"), pa("x", "b"))+") "+iffMp(gp, orN, natPos, "h"),
				"h", iffMpr(gp, orN, natPos,
					"(Or.rec "+nzI("a")+" "+nzI("b")+" (fun (_ : "+orI+") => "+orN+") "+
						"(fun (x : "+nzI("a")+") => "+ap("Or.inl", pos("a"), pos("b"), pb("x", "a"))+") "+
						"(fun (x : "+nzI("b")+") => "+ap("Or.inr", pos("a"), pos("b"), pb("x", "b"))+") h)"))),
	)
	// gcd a b = gcd a b' ↔ ∀ c, c ∣ a → (c ∣ b ↔ c ∣ b')
	rhsR := faI("c", imp2(idv("c", "a"), iff(idv("c", "b"), idv("c", "b2"))))
	natR := ap("Nat.gcd_right_eq_iff", ab("a"), ab("b"), ab("b2"))
	eqR := eqN(igcd("a", "b"), igcd("a", "b2"))
	NR := faN("k", imp2(dvdN("k", ab("a")), iff(dvdN("k", ab("b")), dvdN("k", ab("b2")))))
	preludeExtra = append(preludeExtra,
		ithm("Int.gcd_right_eq_iff", "a b b2", nil, iff(eqR, rhsR),
			iffI(eqR, rhsR, "h",
				"fun (c : Int) (hc : "+idv("c", "a")+") => "+iffI(idv("c", "b"), idv("c", "b2"), "x",
					toI("c", "b2", iffMp(dvdN(ab("c"), ab("b")), dvdN(ab("c"), ab("b2")), "("+iffMp(eqR, NR, natR, "h")+" "+ab("c")+" "+toN("c", "a", "hc")+")", toN("c", "b", "x"))),
					"x", toI("c", "b", iffMpr(dvdN(ab("c"), ab("b")), dvdN(ab("c"), ab("b2")), "("+iffMp(eqR, NR, natR, "h")+" "+ab("c")+" "+toN("c", "a", "hc")+")", toN("c", "b2", "x")))),
				"h", iffMpr(eqR, NR, natR,
					"(fun (k : Nat) (hk : "+dvdN("k", ab("a"))+") => "+iffI(dvdN("k", ab("b")), dvdN("k", ab("b2")), "x",
						toN(oN("k"), "b2", iffMp(idv(oN("k"), "b"), idv(oN("k"), "b2"), "("+"h"+" "+oN("k")+" "+toI(oN("k"), "a", "hk")+")", toI(oN("k"), "b", "x"))),
						"x", toN(oN("k"), "b", iffMpr(idv(oN("k"), "b"), idv(oN("k"), "b2"), "("+"h"+" "+oN("k")+" "+toI(oN("k"), "a", "hk")+")", toI(oN("k"), "b2", "x"))))+")"))),
	)
	// gcd a b = gcd a2 b ↔ ∀ c, c ∣ b → (c ∣ a ↔ c ∣ a2)
	rhsL := faI("c", imp2(idv("c", "b"), iff(idv("c", "a"), idv("c", "a2"))))
	natL := ap("Nat.gcd_left_eq_iff", ab("a"), ab("a2"), ab("b"))
	eqL := eqN(igcd("a", "b"), igcd("a2", "b"))
	NL := faN("k", imp2(dvdN("k", ab("b")), iff(dvdN("k", ab("a")), dvdN("k", ab("a2")))))
	preludeExtra = append(preludeExtra,
		ithm("Int.gcd_left_eq_iff", "a a2 b", nil, iff(eqL, rhsL),
			iffI(eqL, rhsL, "h",
				"fun (c : Int) (hc : "+idv("c", "b")+") => "+iffI(idv("c", "a"), idv("c", "a2"), "x",
					toI("c", "a2", iffMp(dvdN(ab("c"), ab("a")), dvdN(ab("c"), ab("a2")), "("+iffMp(eqL, NL, natL, "h")+" "+ab("c")+" "+toN("c", "b", "hc")+")", toN("c", "a", "x"))),
					"x", toI("c", "a", iffMpr(dvdN(ab("c"), ab("a")), dvdN(ab("c"), ab("a2")), "("+iffMp(eqL, NL, natL, "h")+" "+ab("c")+" "+toN("c", "b", "hc")+")", toN("c", "a2", "x")))),
				"h", iffMpr(eqL, NL, natL,
					"(fun (k : Nat) (hk : "+dvdN("k", ab("b"))+") => "+iffI(dvdN("k", ab("a")), dvdN("k", ab("a2")), "x",
						toN(oN("k"), "a2", iffMp(idv(oN("k"), "a"), idv(oN("k"), "a2"), "(h "+oN("k")+" "+toI(oN("k"), "b", "hk")+")", toI(oN("k"), "a", "x"))),
						"x", toN(oN("k"), "a", iffMpr(idv(oN("k"), "a"), idv(oN("k"), "a2"), "(h "+oN("k")+" "+toI(oN("k"), "b", "hk")+")", toI(oN("k"), "a2", "x"))))+")"))),
	)
}
