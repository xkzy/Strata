package lean

import "strings"

// Further Nat gcd / lcm / Coprime lemmas under Lean's names (Init/Data/Nat/{Gcd,Lcm,Coprime}.lean). They are built
// from the same raw proof terms as prelude.go, with a few string helpers so the rewrites stay readable.

// eqN is `a = b` on Nat; sym, tr, cg are Eq.symm / Eq.trans / congrArg with every argument explicit.
func eqN(a, b string) string    { return "(Eq.{1} Nat " + a + " " + b + ")" }
func sym(a, b, p string) string { return "(Eq.symm.{1} Nat " + a + " " + b + " " + p + ")" }
func tr(a, b, c, p, q string) string {
	return "(Eq.trans.{1} Nat " + a + " " + b + " " + c + " " + p + " " + q + ")"
}

// cg lifts `p : a = b` through `f`, written as the body of `fun (z : Nat) => ...` by f(z).
func cg(f func(z string) string, a, b, p string) string {
	return "(congrArg.{1, 1} Nat Nat (fun (z : Nat) => " + f("z") + ") " + a + " " + b + " " + p + ")"
}

// tp carries `body : P a` to `P b` along `p : a = b`.
func tp(a, b, p string, P func(z string) string, body string) string {
	return "(Eq.rec.{0, 1} Nat " + a + " (fun (z : Nat) (_ : Eq.{1} Nat " + a + " z) => " + P("z") + ") " + body + " " + b + " " + p + ")"
}

func gcdN(a, b string) string { return "(Nat.gcd " + a + " " + b + ")" }
func dvdN(a, b string) string { return "(Nat.dvd " + a + " " + b + ")" }
func mulN(a, b string) string { return "(Nat.mul " + a + " " + b + ")" }
func ltN(a, b string) string  { return "(Nat.lt " + a + " " + b + ")" }
func notN(p string) string    { return "(Not " + p + ")" }

const zN = "0"

func addN(a, b string) string { return "(Nat.add " + a + " " + b + ")" }
func subN(a, b string) string { return "(Nat.sub " + a + " " + b + ")" }
func leN(a, b string) string  { return "(Nat.le " + a + " " + b + ")" }

// ap applies a prelude constant to explicit arguments.
func ap(f string, args ...string) string {
	s := "(" + f
	for _, a := range args {
		s += " " + a
	}
	return s + ")"
}

// thm builds a theorem over Nat variables `vars` (space separated) and hypotheses hyps (name, type pairs).
func thm(name, vars string, hyps [][2]string, concl, proof string) valDecl {
	var pi, lam string
	for _, v := range strings.Fields(vars) {
		pi += "(" + v + " : Nat) → "
		lam += " (" + v + " : Nat)"
	}
	for _, h := range hyps {
		pi += "(" + h[0] + " : " + h[1] + ") → "
		lam += " (" + h[0] + " : " + h[1] + ")"
	}
	return valDecl{"thm", name, nil, pi + concl, "fun" + lam + " => " + proof}
}

// dvIntro destructs `h : m ∣ k` into `k = m * l` for the goal.
func dvIntro(m, k, h, goal string, body func(l, hl string) string) string {
	return "(Exists.rec.{1} Nat (fun (c : Nat) => Eq.{1} Nat " + k + " " + mulN(m, "c") + ") (fun (_ : " + dvdN(m, k) +
		") => " + goal + ") (fun (l : Nat) (hl : Eq.{1} Nat " + k + " " + mulN(m, "l") + ") => " + body("l", "hl") + ") " + h + ")"
}

var preludeExtra = []any{
	valDecl{"thm", "Nat.eq_zero_of_gcd_eq_zero_left", nil,
		"(m : Nat) → (n : Nat) → " + eqN(gcdN("m", "n"), zN) + " → " + eqN("m", zN),
		"fun (m : Nat) (n : Nat) (h : " + eqN(gcdN("m", "n"), zN) + ") => Nat.eq_zero_of_zero_dvd m " +
			tp(gcdN("m", "n"), zN, "h", func(z string) string { return dvdN(z, "m") }, "(Nat.gcd_dvd_left m n)")},
	valDecl{"thm", "Nat.eq_zero_of_gcd_eq_zero_right", nil,
		"(m : Nat) → (n : Nat) → " + eqN(gcdN("m", "n"), zN) + " → " + eqN("n", zN),
		"fun (m : Nat) (n : Nat) (h : " + eqN(gcdN("m", "n"), zN) + ") => Nat.eq_zero_of_zero_dvd n " +
			tp(gcdN("m", "n"), zN, "h", func(z string) string { return dvdN(z, "n") }, "(Nat.gcd_dvd_right m n)")},
	valDecl{"thm", "Nat.gcd_ne_zero_left", nil,
		"(m : Nat) → (n : Nat) → " + notN(eqN("m", zN)) + " → " + notN(eqN(gcdN("m", "n"), zN)),
		"fun (m : Nat) (n : Nat) (hm : " + notN(eqN("m", zN)) + ") (h : " + eqN(gcdN("m", "n"), zN) +
			") => hm (Nat.eq_zero_of_gcd_eq_zero_left m n h)"},
	valDecl{"thm", "Nat.gcd_ne_zero_right", nil,
		"(m : Nat) → (n : Nat) → " + notN(eqN("n", zN)) + " → " + notN(eqN(gcdN("m", "n"), zN)),
		"fun (m : Nat) (n : Nat) (hn : " + notN(eqN("n", zN)) + ") (h : " + eqN(gcdN("m", "n"), zN) +
			") => hn (Nat.eq_zero_of_gcd_eq_zero_right m n h)"},
	valDecl{"thm", "Nat.gcd_eq_zero_iff", nil,
		"(m : Nat) → (n : Nat) → Iff " + eqN(gcdN("m", "n"), zN) + " (And " + eqN("m", zN) + " " + eqN("n", zN) + ")",
		"fun (m : Nat) (n : Nat) => Iff.intro " + eqN(gcdN("m", "n"), zN) + " (And " + eqN("m", zN) + " " + eqN("n", zN) + ") " +
			"(fun (h : " + eqN(gcdN("m", "n"), zN) + ") => And.intro " + eqN("m", zN) + " " + eqN("n", zN) + " " +
			"(Nat.eq_zero_of_gcd_eq_zero_left m n h) (Nat.eq_zero_of_gcd_eq_zero_right m n h)) " +
			"(fun (h : And " + eqN("m", zN) + " " + eqN("n", zN) + ") => " +
			tp(zN, "m", sym("m", zN, "(And.left "+eqN("m", zN)+" "+eqN("n", zN)+" h)"),
				func(z string) string { return eqN(gcdN(z, "n"), zN) },
				tp(zN, "n", sym("n", zN, "(And.right "+eqN("m", zN)+" "+eqN("n", zN)+" h)"),
					func(w string) string { return eqN(gcdN(zN, w), zN) }, "(Nat.gcd_zero_left 0)")) + ")"},
	valDecl{"thm", "Nat.gcd_pos_iff", nil,
		"(m : Nat) → (n : Nat) → Iff " + ltN(zN, gcdN("m", "n")) + " (Or " + ltN(zN, "m") + " " + ltN(zN, "n") + ")",
		"fun (m : Nat) (n : Nat) => Iff.intro " + ltN(zN, gcdN("m", "n")) + " (Or " + ltN(zN, "m") + " " + ltN(zN, "n") + ") " +
			"(Nat.rec.{0} (fun (m : Nat) => " + ltN(zN, gcdN("m", "n")) + " → Or " + ltN(zN, "m") + " " + ltN(zN, "n") + ") " +
			"(fun (h : " + ltN(zN, gcdN(zN, "n")) + ") => Or.inr " + ltN(zN, zN) + " " + ltN(zN, "n") + " " +
			tp(gcdN(zN, "n"), "n", "(Nat.gcd_zero_left n)", func(z string) string { return ltN(zN, z) }, "h") + ") " +
			"(fun (i : Nat) (_ : " + ltN(zN, gcdN("i", "n")) + " → Or " + ltN(zN, "i") + " " + ltN(zN, "n") + ") (_ : " +
			ltN(zN, gcdN("(Nat.succ i)", "n")) + ") => Or.inl " + ltN(zN, "(Nat.succ i)") + " " + ltN(zN, "n") +
			" (Nat.zero_lt_succ i)) m) " +
			"(fun (h : Or " + ltN(zN, "m") + " " + ltN(zN, "n") + ") => Or.rec " + ltN(zN, "m") + " " + ltN(zN, "n") +
			" (fun (_ : Or " + ltN(zN, "m") + " " + ltN(zN, "n") + ") => " + ltN(zN, gcdN("m", "n")) + ") " +
			"(fun (hm : " + ltN(zN, "m") + ") => Nat.gcd_pos_of_pos_left m n hm) " +
			"(fun (hn : " + ltN(zN, "n") + ") => Nat.gcd_pos_of_pos_right m n hn) h)"},
	valDecl{"thm", "Nat.dvd_gcd_iff", nil,
		"(k : Nat) → (m : Nat) → (n : Nat) → Iff " + dvdN("k", gcdN("m", "n")) + " (And " + dvdN("k", "m") + " " + dvdN("k", "n") + ")",
		"fun (k : Nat) (m : Nat) (n : Nat) => Iff.intro " + dvdN("k", gcdN("m", "n")) + " (And " + dvdN("k", "m") + " " + dvdN("k", "n") + ") " +
			"(fun (h : " + dvdN("k", gcdN("m", "n")) + ") => And.intro " + dvdN("k", "m") + " " + dvdN("k", "n") + " " +
			"(Nat.dvd_trans k " + gcdN("m", "n") + " m h (Nat.gcd_dvd_left m n)) " +
			"(Nat.dvd_trans k " + gcdN("m", "n") + " n h (Nat.gcd_dvd_right m n))) " +
			"(fun (h : And " + dvdN("k", "m") + " " + dvdN("k", "n") + ") => Nat.dvd_gcd k m n " +
			"(And.left " + dvdN("k", "m") + " " + dvdN("k", "n") + " h) (And.right " + dvdN("k", "m") + " " + dvdN("k", "n") + " h))"},
	valDecl{"thm", "Nat.gcd_eq_left_iff_dvd", nil,
		"(m : Nat) → (n : Nat) → Iff " + eqN(gcdN("m", "n"), "m") + " " + dvdN("m", "n"),
		"fun (m : Nat) (n : Nat) => Iff.intro " + eqN(gcdN("m", "n"), "m") + " " + dvdN("m", "n") + " " +
			"(fun (h : " + eqN(gcdN("m", "n"), "m") + ") => " +
			tp(gcdN("m", "n"), "m", "h", func(z string) string { return dvdN(z, "n") }, "(Nat.gcd_dvd_right m n)") + ") " +
			"(fun (h : " + dvdN("m", "n") + ") => Nat.gcd_eq_left m n h)"},
	valDecl{"thm", "Nat.gcd_eq_right", nil,
		"(m : Nat) → (n : Nat) → " + dvdN("n", "m") + " → " + eqN(gcdN("m", "n"), "n"),
		"fun (m : Nat) (n : Nat) (h : " + dvdN("n", "m") + ") => " +
			tr(gcdN("m", "n"), gcdN("n", "m"), "n", "(Nat.gcd_comm m n)", "(Nat.gcd_eq_left n m h)")},
	valDecl{"thm", "Nat.gcd_eq_right_iff_dvd", nil,
		"(n : Nat) → (m : Nat) → Iff " + eqN(gcdN("n", "m"), "m") + " " + dvdN("m", "n"),
		"fun (n : Nat) (m : Nat) => Iff.intro " + eqN(gcdN("n", "m"), "m") + " " + dvdN("m", "n") + " " +
			"(fun (h : " + eqN(gcdN("n", "m"), "m") + ") => " +
			tp(gcdN("n", "m"), "m", "h", func(z string) string { return dvdN(z, "n") }, "(Nat.gcd_dvd_left n m)") + ") " +
			"(fun (h : " + dvdN("m", "n") + ") => Nat.gcd_eq_right n m h)"},
	valDecl{"thm", "Nat.gcd_left_comm", nil,
		"(m : Nat) → (n : Nat) → (k : Nat) → " + eqN(gcdN("m", gcdN("n", "k")), gcdN("n", gcdN("m", "k"))),
		"fun (m : Nat) (n : Nat) (k : Nat) => " +
			tr(gcdN("m", gcdN("n", "k")), gcdN(gcdN("m", "n"), "k"), gcdN("n", gcdN("m", "k")),
				sym(gcdN(gcdN("m", "n"), "k"), gcdN("m", gcdN("n", "k")), "(Nat.gcd_assoc m n k)"),
				tr(gcdN(gcdN("m", "n"), "k"), gcdN(gcdN("n", "m"), "k"), gcdN("n", gcdN("m", "k")),
					cg(func(z string) string { return gcdN(z, "k") }, gcdN("m", "n"), gcdN("n", "m"), "(Nat.gcd_comm m n)"),
					"(Nat.gcd_assoc n m k)"))},
}
