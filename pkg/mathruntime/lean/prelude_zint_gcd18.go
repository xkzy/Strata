package lean

// Int.dvd_mul (Init/Data/Int/Gcd.lean): c ∣ a * b ↔ ∃ c₁ c₂, c₁ ∣ a ∧ c₂ ∣ b ∧ c₁ * c₂ = c. Lean's `dvdProdDvdOfDvdProd` is data;
// the Prop form Nat.dvdProd_exists at the absolute values gives the factors, and the sign of c goes to the first one.
// Variables are p q r (= a b c) because ex2 binds Nat variables named a and b.
func init() {
	p, q, r := "p", "q", "r"
	ab := iabs
	pair := func(c1, c2 string) string {
		return andN(idvd(c1, p), andN(idvd(c2, q), ieq(imul(c1, c2), r)))
	}
	innerI := func(c1 string) string { return "(fun (c2 : Int) => " + pair(c1, "c2") + ")" }
	outerI := "(fun (c1 : Int) => Exists.{1} Int " + innerI("c1") + ")"
	exI := func(c1 string) string { return "(Exists.{1} Int " + innerI(c1) + ")" }
	goalEx := "(Exists.{1} Int " + outerI + ")"
	toN := func(c, x, h string) string {
		return iffMpr(dvdN(ab(c), ab(x)), idvd(c, x), ap("Int.natAbs_dvd_natAbs", c, x), h)
	}
	frN := func(c, x, h string) string {
		return iffMp(dvdN(ab(c), ab(x)), idvd(c, x), ap("Int.natAbs_dvd_natAbs", c, x), h)
	}
	mk := func(c1, c2, pa, pb, pe string) string {
		return "(Exists.intro.{1} Int " + outerI + " " + c1 + " (Exists.intro.{1} Int " + innerI(c1) + " " + c2 + " " +
			ap("And.intro", idvd(c1, p), andN(idvd(c2, q), ieq(imul(c1, c2), r)), pa,
				ap("And.intro", idvd(c2, q), ieq(imul(c1, c2), r), pb, pe)) + "))"
	}
	ofN := func(x string) string { return "(Int.ofNat " + x + ")" }
	orT := "(Or " + ieq(r, ofN(ab(r))) + " " + ieq(r, ineg(ofN(ab(r)))) + ")"
	// forward
	hN := tp(ab(imul(p, q)), mulN(ab(p), ab(q)), ap("Int.natAbs_mul", p, q), func(z string) string { return dvdN(ab(r), z) }, toN(r, imul(p, q), "h"))
	fwd := ex2(ab(r), ab(p), ab(q), hN, goalEx, func(u, v, hu, hv, e string) string {
		xu, xv := ofN(u), ofN(v)
		prod := "(congrArg.{1, 1} Nat Int Int.ofNat " + mulN(u, v) + " " + ab(r) + " " + e + ")"
		pos := mk(xu, xv, frN(xu, p, hu), frN(xv, q, hv), itr(imul(xu, xv), ofN(ab(r)), r, prod, isym(r, ofN(ab(r)), "hr")))
		nx := ineg(xu)
		negPe := itr(imul(nx, xv), ineg(imul(xu, xv)), r, ap("Int.neg_mul", xu, xv),
			itr(ineg(imul(xu, xv)), ineg(ofN(ab(r))), r, icg(ineg, imul(xu, xv), ofN(ab(r)), prod), isym(r, ineg(ofN(ab(r))), "hr")))
		neg := mk(nx, xv, iffMpr(idvd(nx, p), idvd(xu, p), ap("Int.neg_dvd", xu, p), frN(xu, p, hu)), frN(xv, q, hv), negPe)
		return "(Or.rec " + ieq(r, ofN(ab(r))) + " " + ieq(r, ineg(ofN(ab(r)))) + " (fun (_ : " + orT + ") => " + goalEx + ") " +
			"(fun (hr : " + ieq(r, ofN(ab(r))) + ") => " + pos + ") (fun (hr : " + ieq(r, ineg(ofN(ab(r)))) + ") => " + neg + ") " + ap("Int.natAbs_eq", r) + ")"
	})
	// backward
	hc := ap("And.right", idvd("c2", q), ieq(imul("c1", "c2"), r), ap("And.right", idvd("c1", p), andN(idvd("c2", q), ieq(imul("c1", "c2"), r)), "h2"))
	hb := ap("And.left", idvd("c2", q), ieq(imul("c1", "c2"), r), ap("And.right", idvd("c1", p), andN(idvd("c2", q), ieq(imul("c1", "c2"), r)), "h2"))
	ha := ap("And.left", idvd("c1", p), andN(idvd("c2", q), ieq(imul("c1", "c2"), r)), "h2")
	m := ap("Nat.mul_dvd_mul", ab("c1"), ab(p), ab("c2"), ab(q), toN("c1", p, ha), toN("c2", q, hb))
	e1 := tr(mulN(ab("c1"), ab("c2")), ab(imul("c1", "c2")), ab(r), sym(ab(imul("c1", "c2")), mulN(ab("c1"), ab("c2")), ap("Int.natAbs_mul", "c1", "c2")),
		icgN(ab, imul("c1", "c2"), r, hc))
	d1 := tp(mulN(ab("c1"), ab("c2")), ab(r), e1, func(z string) string { return dvdN(z, mulN(ab(p), ab(q))) }, m)
	d2 := tp(mulN(ab(p), ab(q)), ab(imul(p, q)), sym(ab(imul(p, q)), mulN(ab(p), ab(q)), ap("Int.natAbs_mul", p, q)), func(z string) string { return dvdN(ab(r), z) }, d1)
	bwd := "Exists.rec.{1} Int " + outerI + " (fun (_ : " + goalEx + ") => " + idvd(r, imul(p, q)) + ") (fun (c1 : Int) (h1 : " + exI("c1") + ") => " +
		"Exists.rec.{1} Int " + innerI("c1") + " (fun (_ : " + exI("c1") + ") => " + idvd(r, imul(p, q)) + ") (fun (c2 : Int) (h2 : " + pair("c1", "c2") + ") => " +
		frN(r, imul(p, q), d2) + ") h1) h"
	preludeExtra = append(preludeExtra,
		ithm("Int.dvd_mul", "p q r", nil, "(Iff "+idvd(r, imul(p, q))+" "+goalEx+")",
			iffI(idvd(r, imul(p, q)), goalEx, "h", fwd, "h", bwd)),
	)
}
