package lean

// (a * b) / a = b for a != 0 (Init/Data/Int/DivMod/Lemmas.lean `Int.mul_ediv_cancel_left`), by cases on both signs. The reduction of
// Int.mul / Int.ediv on constructors does the rewriting; the Nat facts are mul_div_cancel_left and mul_add_div_of_lt.
func init() {
	z := iz()
	goal := func(a, b string) string { return ieq("(Int.ediv (Int.mul "+a+" "+b+") "+a+")", b) }
	nz := func(a string) string { return notN(ieq(a, z)) }
	pre := func(a string) string { return "(b : Int) → " + nz(a) + " → " + goal(a, "b") }
	succ := func(a string) string { return "(Nat.succ " + a + ")" }
	cgI := func(f, a, b, p string) string {
		return "(congrArg.{1, 1} Nat Int " + f + " " + a + " " + b + " " + p + ")"
	}
	cgN := func(a, b, p string) string { return "(congrArg.{1, 1} Nat Nat Nat.succ " + a + " " + b + " " + p + ")" }
	mS := succ("p")
	// a = ofNat (succ p)
	pos := "(Int.ofNat " + mS + ")"
	rA := "(Nat.add (Nat.mul " + mS + " n) p)" // succ p * succ n = succ rA
	a1 := cgI("Int.ofNat", "(Nat.div (Nat.mul "+mS+" n) "+mS+")", "n", ap("Nat.mul_div_cancel_left", "n", mS, ap("Nat.zero_lt_succ", "p")))
	a2 := cgI("Int.negSucc", "(Nat.div "+rA+" "+mS+")", "n",
		ap("Nat.mul_add_div_of_lt", mS, "n", "p", ap("Nat.zero_lt_succ", "p"), ap("Nat.lt_succ_self", "p")))
	caseA := "(fun (p : Nat) (_ : " + pre("(Int.ofNat p)") + ") => Int.rec.{0} (fun (b : Int) => " + nz(pos) + " → " + goal(pos, "b") + ") " +
		"(fun (n : Nat) (_ : " + nz(pos) + ") => " + a1 + ") (fun (n : Nat) (_ : " + nz(pos) + ") => " + a2 + "))"
	// a = negSucc m
	neg := "(Int.negSucc m)"
	mM := succ("m")
	rB := "(Nat.add (Nat.mul " + mM + " q) m)"
	bq := cgI("Int.ofNat", "(Nat.succ (Nat.div "+rB+" "+mM+"))", "(Nat.succ q)",
		cgN("(Nat.div "+rB+" "+mM+")", "q", ap("Nat.mul_add_div_of_lt", mM, "q", "m", ap("Nat.zero_lt_succ", "m"), ap("Nat.lt_succ_self", "m"))))
	b0 := cgI("Int.negOfNat", "(Nat.div 0 "+mM+")", "0", ap("Nat.zero_div", mM))
	motB := "(fun (n : Nat) => " + goal(neg, "(Int.ofNat n)") + ")"
	b1 := "Nat.rec.{0} " + motB + " " + b0 + " (fun (q : Nat) (_ : " + goal(neg, "(Int.ofNat q)") + ") => " + bq + ") n"
	b2 := cgI("Int.negOfNat", "(Nat.div (Nat.mul "+mM+" "+succ("n")+") "+mM+")", succ("n"),
		ap("Nat.mul_div_cancel_left", succ("n"), mM, ap("Nat.zero_lt_succ", "m")))
	caseB := "(fun (m : Nat) => Int.rec.{0} (fun (b : Int) => " + nz(neg) + " → " + goal(neg, "b") + ") " +
		"(fun (n : Nat) (_ : " + nz(neg) + ") => " + b1 + ") (fun (n : Nat) (_ : " + nz(neg) + ") => " + b2 + "))"
	zeroA := "(fun (b : Int) (h : " + nz("(Int.ofNat 0)") + ") => False.elim.{0} " + goal("(Int.ofNat 0)", "b") + " (h (Eq.refl.{1} Int " + z + ")))"
	preludeExtra = append(preludeExtra,
		ithm("Int.mul_ediv_cancel_left", "a b", [][2]string{{"ha", nz("a")}}, goal("a", "b"),
			"Int.rec.{0} (fun (a : Int) => "+pre("a")+") (fun (m : Nat) => Nat.rec.{0} (fun (m : Nat) => "+pre("(Int.ofNat m)")+") "+zeroA+" "+caseA+" m) "+caseB+" a b ha"),
	)
}
