package lean

// Init/Data/Int/DivMod/Basic.lean: Int.ediv / Int.emod on constructors are `rfl` there, and `Eq.refl` here. Int.zero_ne_one (Init/Data/Int/Basic.lean).
func init() {
	ofN := func(x string) string { return "(Int.ofNat " + x + ")" }
	ns := func(x string) string { return "(Int.negSucc " + x + ")" }
	dv := func(x, y string) string { return "(Nat.div " + x + " " + y + ")" }
	sc := func(x string) string { return "(Nat.succ " + x + ")" }
	refl := func(t string) string { return "(Eq.refl.{1} Int " + t + ")" }
	rf := func(name string, lhs, rhs string) valDecl {
		return thm(name, "a b", nil, ieq(lhs, rhs), refl(rhs))
	}
	e := func(x, y string) string { return "(Int.ediv " + x + " " + y + ")" }
	em := func(x, y string) string { return "(Int.emod " + x + " " + y + ")" }
	preludeExtra = append(preludeExtra,
		rf("Int.ofNat_ediv_ofNat", e(ofN("a"), ofN("b")), ofN(dv("a", "b"))),
		rf("Int.negSucc_ediv_ofNat_succ", e(ns("a"), ofN(sc("b"))), ns(dv("a", sc("b")))),
		rf("Int.negSucc_ediv_negSucc", e(ns("a"), ns("b")), ofN(sc(dv("a", sc("b"))))),
		rf("Int.ofNat_ediv_negSucc", e(ofN("a"), ns("b")), "(Int.neg "+ofN(dv("a", sc("b")))+")"),
		rf("Int.negSucc_emod_ofNat", em(ns("a"), ofN("b")), "(Int.subNatNat b "+sc("(Nat.mod a b)")+")"),
		rf("Int.negSucc_emod_negSucc", em(ns("a"), ns("b")), "(Int.subNatNat "+sc("b")+" "+sc("(Nat.mod a "+sc("b")+")")+")"),
		valDecl{"thm", "Int.zero_ne_one", nil, "Not " + ieq(ofN("0"), ofN("1")),
			"fun (h : " + ieq(ofN("0"), ofN("1")) + ") => Nat.succ_ne_zero 0 (Eq.symm.{1} Nat 0 1 (Int.ofNat.inj 0 1 h))"},
	)
}
