package lean

// |a / b| = |a| / |b| when b ∣ a (Init/Data/Int/DivMod/Lemmas.lean `Int.natAbs_ediv_of_dvd`): b = 0 divides both sides to 0;
// otherwise a = b * l gives a / b = l by mul_ediv_cancel_left, and |a| / |b| = |b| * |l| / |b| = |l|.
func init() {
	a, b := "a", "b"
	ab := iabs
	dv := func(z string) string { return "(Nat.div " + ab(a) + " " + ab(z) + ")" }
	goalOf := func(z string) string { return eqN(ab("(Int.ediv "+a+" "+z+")"), dv(z)) }
	goal := goalOf(b)
	orT := "(Or " + eqN(ab(b), "0") + " " + ltN("0", ab(b)) + ")"
	zeroCase := "(fun (h0 : " + eqN(ab(b), "0") + ") => " + tpI(iz(), b, isym(b, iz(), iffMp(eqN(ab(b), "0"), ieq(b, iz()), ap("Int.natAbs_eq_zero", b), "h0")), goalOf,
		tr(ab("(Int.ediv "+a+" "+iz()+")"), "0", dv(iz()),
			icgN(ab, "(Int.ediv "+a+" "+iz()+")", iz(), ap("Int.ediv_zero", a)),
			sym(dv(iz()), "0", ap("Nat.div_zero", ab(a))))) + ")"
	body := func(l, hl string) string {
		hbne := "(fun (h0 : " + ieq(b, iz()) + ") => Nat.lt_irrefl 0 " + tp(ab(b), "0", icgN(ab, b, iz(), "h0"), func(z string) string { return ltN("0", z) }, "hp") + ")"
		hdiv := tpI(imul(b, l), a, isym(a, imul(b, l), hl), func(z string) string { return ieq("(Int.ediv "+z+" "+b+")", l) },
			ap("Int.mul_ediv_cancel_left", b, l, hbne))
		habs := tr(ab(a), ab(imul(b, l)), mulN(ab(b), ab(l)), icgN(ab, a, imul(b, l), hl), ap("Int.natAbs_mul", b, l))
		return "(fun (hp : " + ltN("0", ab(b)) + ") => " + tr(ab("(Int.ediv "+a+" "+b+")"), ab(l), dv(b),
			icgN(ab, "(Int.ediv "+a+" "+b+")", l, hdiv),
			sym(dv(b), ab(l), tr(dv(b), "(Nat.div "+mulN(ab(b), ab(l))+" "+ab(b)+")", ab(l),
				cg(func(z string) string { return "(Nat.div " + z + " " + ab(b) + ")" }, ab(a), mulN(ab(b), ab(l)), habs),
				ap("Nat.mul_div_cancel_left", ab(l), ab(b), "hp")))) + ")"
	}
	preludeExtra = append(preludeExtra,
		ithm("Int.natAbs_ediv_of_dvd", "a b", [][2]string{{"h", idvd(b, a)}}, goal,
			iDvIntro(b, a, "h", goal, func(l, hl string) string {
				return "(Or.rec " + eqN(ab(b), "0") + " " + ltN("0", ab(b)) + " (fun (_ : " + orT + ") => " + goal + ") " + zeroCase + " " + body(l, hl) + " " + ap("Nat.eq_zero_or_pos", ab(b)) + ")"
			})),
	)
}
