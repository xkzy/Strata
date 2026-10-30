package lean

import (
	"fmt"
	"strings"
	"sync"
)

// PreludeAxioms are the declarations the prelude admits without a proof: none.  Every constant in the prelude - the
// Nat and Int operations, the division lemmas, `Nat.ble` and the order it decides - is either a definition or a
// theorem proved from `Nat.rec`, `Nat.le.rec` and the two inductive eliminators.  The list stays, and the check stays:
// an admitted statement added here later fails the build unless this list says so.
var PreludeAxioms = []string{}

type indDecl struct {
	name    string
	lparams []string
	nparams int
	ty      string
	ctors   [][2]string
}

type valDecl struct {
	kind    string // def, thm, opaque, axiom
	name    string
	lparams []string
	ty, val string
}

// The division lemmas below are raw proof terms, so the text that repeats inside them is named here once. `n` is the
// induction variable, `k` the divisor, and `hv` the branch of `Nat.beq (n % k + 1) k` the step took.
var (
	dmState  = "(Nat.DivMod.mk (Nat.div n k) (Nat.mod n k))"
	dmStateF = "(Nat.DivMod.mk (Nat.div n k) (Nat.succ (Nat.mod n k)))"
	dmStateT = "(Nat.DivMod.mk (Nat.succ (Nat.div n k)) 0)"
	dmStateS = "(Nat.DivMod.mk (Nat.div (Nat.succ n) k) (Nat.mod (Nat.succ n) k))"
	dmAlpha  = "Eq.{1} Nat.DivMod (Nat.divMod k n) " + dmState
	dmAlphaS = "Eq.{1} Nat.DivMod (Nat.divMod k (Nat.succ n)) " + dmStateS
	dmBeta   = "Eq.{1} Nat (Nat.add (Nat.mul k (Nat.div n k)) (Nat.mod n k)) n"
	dmBetaS  = "Eq.{1} Nat (Nat.add (Nat.mul k (Nat.div (Nat.succ n) k)) (Nat.mod (Nat.succ n) k)) (Nat.succ n)"
	dmIh     = "And (" + dmAlpha + ") (" + dmBeta + ")"
	dmIhA    = "(And.left (" + dmAlpha + ") (" + dmBeta + ") ih)"
	dmIhB    = "(And.right (" + dmAlpha + ") (" + dmBeta + ") ih)"
	dmBeq    = "(Nat.beq (Nat.succ (Nat.mod n k)) k)"
	dmFst    = "(fun (s : Nat.DivMod) => Nat.DivMod.rec.{1} (fun (_ : Nat.DivMod) => Nat) (fun (q : Nat) (_ : Nat) => q) s)"
	dmSnd    = "(fun (s : Nat.DivMod) => Nat.DivMod.rec.{1} (fun (_ : Nat.DivMod) => Nat) (fun (_ : Nat) (r : Nat) => r) s)"
	// one step of the machine, as an equation about the state after `n + 1` steps
	dmStepF = "(Eq.trans.{1} Nat.DivMod (Nat.divStep k (Nat.divMod k n)) (Nat.divStep k " + dmState + ") " + dmStateF +
		" (congrArg.{1, 1} Nat.DivMod Nat.DivMod (Nat.divStep k) (Nat.divMod k n) " + dmState + " " + dmIhA + ") " +
		"(Nat.divStep_false k (Nat.div n k) (Nat.mod n k) hv))"
	dmStepT = "(Eq.trans.{1} Nat.DivMod (Nat.divStep k (Nat.divMod k n)) (Nat.divStep k " + dmState + ") " + dmStateT +
		" (congrArg.{1, 1} Nat.DivMod Nat.DivMod (Nat.divStep k) (Nat.divMod k n) " + dmState + " " + dmIhA + ") " +
		"(Nat.divStep_true k (Nat.div n k) (Nat.mod n k) hv))"
)

// The statement `Nat.dvdProd_exists` proves, for `k m n` in scope: `k = a * b` with `a ∣ m`, `b ∣ n`.
const (
	dvdProdF  = "(fun (a : Nat) => Exists.{1} Nat (fun (b : Nat) => And (And (Nat.dvd a m) (Nat.dvd b n)) (Eq.{1} Nat (Nat.mul a b) k)))"
	dvdProdR  = "Exists.{1} Nat " + dvdProdF
	dvdProdRK = "Exists.{1} Nat (fun (a : Nat) => Exists.{1} Nat (fun (b : Nat) => And (And (Nat.dvd a m) (Nat.dvd b n)) (Eq.{1} Nat (Nat.mul a b) k)))"
	dvdProdRG = "Exists.{1} Nat (fun (a : Nat) => Exists.{1} Nat (fun (b : Nat) => And (And (Nat.dvd a m) (Nat.dvd b n)) (Eq.{1} Nat (Nat.mul a b) (Nat.gcd k (Nat.mul m n)))))"
)

// The step `Nat.gcdIter (succ f') (succ a') b = Nat.gcdIter f' (b % succ a') (succ a')`, as used by
// `Nat.gcdIter_dvd`: read from the front it is one unfold, read backwards it is the value the induction hypothesis
// covers. `gcdDvdB` is `b` rewritten into `succ a' * (b / succ a') + b % succ a'`, which `Nat.divMod_spec` proves, so
// that dividing the two summands divides `b`.
var (
	gcdA    = "(Nat.mod b (Nat.succ a'))"
	gcdB    = "(Nat.succ a')"
	gcdD    = "(Nat.gcdIter f' " + gcdA + " " + gcdB + ")"
	gcdLe   = "(Nat.le_trans " + gcdA + " a' f' (Nat.le_of_succ_le_succ " + gcdA + " a' (Nat.mod_lt b (Nat.succ a') (Nat.zero_lt_succ a'))) (Nat.le_of_succ_le_succ a' f' h))"
	gcdIh   = "(ih " + gcdA + " " + gcdB + " " + gcdLe + ")"
	gcdIhA  = "(And.left (Nat.dvd " + gcdD + " " + gcdA + ") (Nat.dvd " + gcdD + " " + gcdB + ") " + gcdIh + ")"
	gcdIhB  = "(And.right (Nat.dvd " + gcdD + " " + gcdA + ") (Nat.dvd " + gcdD + " " + gcdB + ") " + gcdIh + ")"
	gcdSum  = "(Nat.add (Nat.mul " + gcdB + " (Nat.div b " + gcdB + ")) " + gcdA + ")"
	gcdBeta = "(And.right (Eq.{1} Nat.DivMod (Nat.divMod " + gcdB + " b) (Nat.DivMod.mk (Nat.div b " + gcdB + ") " + gcdA + ")) (Eq.{1} Nat " + gcdSum + " b) (Nat.divMod_spec " + gcdB + " b))"
	gcdDvdS = "(Nat.dvd_add " + gcdD + " (Nat.mul " + gcdB + " (Nat.div b " + gcdB + ")) " + gcdA + " (Nat.dvd_mul_right_of_dvd " + gcdD + " " + gcdB + " (Nat.div b " + gcdB + ") " + gcdIhB + ") " + gcdIhA + ")"
	gcdDvdB = "(Eq.rec.{0, 1} Nat " + gcdSum + " (fun (z : Nat) (_ : Eq.{1} Nat " + gcdSum + " z) => Nat.dvd " + gcdD + " z) " + gcdDvdS + " b " + gcdBeta + ")"
	// `b = a * (b / a) + b % a`, and subtracting the multiple leaves `b % a`.
	modB   = "(Nat.add (Nat.mul a (Nat.div b a)) (Nat.mod b a))"
	modSub = "(Eq.trans.{1} Nat (Nat.sub b (Nat.mul a (Nat.div b a))) " +
		"(Nat.sub " + modB + " (Nat.mul a (Nat.div b a))) (Nat.mod b a) " +
		"(congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.sub z (Nat.mul a (Nat.div b a))) b " + modB +
		" (Eq.symm.{1} Nat " + modB + " b (And.right (Eq.{1} Nat.DivMod (Nat.divMod a b) " +
		"(Nat.DivMod.mk (Nat.div b a) (Nat.mod b a))) (Eq.{1} Nat " + modB + " b) (Nat.divMod_spec a b)))) " +
		"(Nat.add_sub_cancel_left (Nat.mul a (Nat.div b a)) (Nat.mod b a)))"
)

// Each entry is added in order through Env.AddInductive / AddDefn / AddTheorem / ..., so the kernel checks all of it.
var preludeDecls = []any{
	indDecl{"True", nil, 0, "Prop", [][2]string{{"True.intro", "True"}}},
	indDecl{"False", nil, 0, "Prop", nil},
	indDecl{"Bool", nil, 0, "Type", [][2]string{{"Bool.false", "Bool"}, {"Bool.true", "Bool"}}},
	indDecl{"Nat", nil, 0, "Type", [][2]string{{"Nat.zero", "Nat"}, {"Nat.succ", "(n : Nat) → Nat"}}},
	indDecl{"Eq", []string{"u"}, 2, "(α : Sort u) → (a : α) → α → Prop",
		[][2]string{{"Eq.refl", "(α : Sort u) → (a : α) → Eq.{u} α a a"}}},
	indDecl{"And", nil, 2, "Prop → Prop → Prop",
		[][2]string{{"And.intro", "(a : Prop) → (b : Prop) → (left : a) → (right : b) → And a b"}}},
	indDecl{"Or", nil, 2, "Prop → Prop → Prop", [][2]string{
		{"Or.inl", "(a : Prop) → (b : Prop) → (h : a) → Or a b"},
		{"Or.inr", "(a : Prop) → (b : Prop) → (h : b) → Or a b"}}},
	indDecl{"Iff", nil, 2, "Prop → Prop → Prop",
		[][2]string{{"Iff.intro", "(a : Prop) → (b : Prop) → (mp : a → b) → (mpr : b → a) → Iff a b"}}},
	indDecl{"Exists", []string{"u"}, 2, "(α : Sort u) → (α → Prop) → Prop",
		[][2]string{{"Exists.intro", "(α : Sort u) → (p : α → Prop) → (w : α) → (h : p w) → Exists.{u} α p"}}},
	valDecl{"def", "Not", nil, "Prop → Prop", "fun (a : Prop) => a → False"},
	indDecl{"Decidable", nil, 1, "Prop → Type", [][2]string{
		{"Decidable.isFalse", "(p : Prop) → (h : Not p) → Decidable p"},
		{"Decidable.isTrue", "(p : Prop) → (h : p) → Decidable p"}}},
	indDecl{"Nat.le", nil, 1, "Nat → Nat → Prop", [][2]string{
		{"Nat.le.refl", "(n : Nat) → Nat.le n n"},
		{"Nat.le.step", "(n : Nat) → (m : Nat) → Nat.le n m → Nat.le n (Nat.succ m)"}}},

	valDecl{"def", "Ne", []string{"u"}, "(α : Sort u) → α → α → Prop",
		"fun (α : Sort u) (a : α) (b : α) => Not (Eq.{u} α a b)"},
	valDecl{"def", "False.elim", []string{"u"}, "(C : Sort u) → False → C",
		"fun (C : Sort u) (h : False) => False.rec.{u} (fun (_ : False) => C) h"},
	valDecl{"thm", "Eq.symm", []string{"u"}, "(α : Sort u) → (a : α) → (b : α) → Eq.{u} α a b → Eq.{u} α b a",
		"fun (α : Sort u) (a : α) (b : α) (h : Eq.{u} α a b) => Eq.rec.{0, u} α a (fun (x : α) (_ : Eq.{u} α a x) => Eq.{u} α x a) (Eq.refl.{u} α a) b h"},
	valDecl{"thm", "Eq.trans", []string{"u"}, "(α : Sort u) → (a : α) → (b : α) → (c : α) → Eq.{u} α a b → Eq.{u} α b c → Eq.{u} α a c",
		"fun (α : Sort u) (a : α) (b : α) (c : α) (h1 : Eq.{u} α a b) (h2 : Eq.{u} α b c) => Eq.rec.{0, u} α b (fun (x : α) (_ : Eq.{u} α b x) => Eq.{u} α a x) h1 c h2"},
	valDecl{"thm", "congrArg", []string{"u", "v"},
		"(α : Sort u) → (β : Sort v) → (f : α → β) → (a : α) → (b : α) → Eq.{u} α a b → Eq.{v} β (f a) (f b)",
		"fun (α : Sort u) (β : Sort v) (f : α → β) (a : α) (b : α) (h : Eq.{u} α a b) => Eq.rec.{0, u} α a (fun (x : α) (_ : Eq.{u} α a x) => Eq.{v} β (f a) (f x)) (Eq.refl.{v} β (f a)) b h"},
	valDecl{"thm", "And.left", nil, "(a : Prop) → (b : Prop) → And a b → a",
		"fun (a : Prop) (b : Prop) (h : And a b) => And.rec.{0} a b (fun (_ : And a b) => a) (fun (l : a) (r : b) => l) h"},
	valDecl{"thm", "And.right", nil, "(a : Prop) → (b : Prop) → And a b → b",
		"fun (a : Prop) (b : Prop) (h : And a b) => And.rec.{0} a b (fun (_ : And a b) => b) (fun (l : a) (r : b) => r) h"},
	valDecl{"thm", "Iff.mp", nil, "(a : Prop) → (b : Prop) → Iff a b → a → b",
		"fun (a : Prop) (b : Prop) (h : Iff a b) => Iff.rec.{0} a b (fun (_ : Iff a b) => a → b) (fun (f : a → b) (g : b → a) => f) h"},
	valDecl{"thm", "Iff.mpr", nil, "(a : Prop) → (b : Prop) → Iff a b → b → a",
		"fun (a : Prop) (b : Prop) (h : Iff a b) => Iff.rec.{0} a b (fun (_ : Iff a b) => b → a) (fun (f : a → b) (g : b → a) => g) h"},
	valDecl{"thm", "Bool.false_ne_true", nil, "Eq.{1} Bool Bool.false Bool.true → False",
		"fun (h : Eq.{1} Bool Bool.false Bool.true) => Eq.rec.{0, 1} Bool Bool.false (fun (b : Bool) (_ : Eq.{1} Bool Bool.false b) => Bool.rec.{1} (fun (_ : Bool) => Prop) True False b) True.intro Bool.true h"},
	valDecl{"thm", "Bool.true_ne_false", nil, "Eq.{1} Bool Bool.true Bool.false → False",
		"fun (h : Eq.{1} Bool Bool.true Bool.false) => Eq.rec.{0, 1} Bool Bool.true (fun (b : Bool) (_ : Eq.{1} Bool Bool.true b) => Bool.rec.{1} (fun (_ : Bool) => Prop) False True b) True.intro Bool.false h"},

	valDecl{"def", "Nat.pred", nil, "Nat → Nat",
		"fun (n : Nat) => Nat.rec.{1} (fun (_ : Nat) => Nat) Nat.zero (fun (k : Nat) (_ : Nat) => k) n"},
	valDecl{"def", "Nat.add", nil, "Nat → Nat → Nat",
		"fun (n : Nat) (m : Nat) => Nat.rec.{1} (fun (_ : Nat) => Nat) n (fun (_ : Nat) (ih : Nat) => Nat.succ ih) m"},
	valDecl{"def", "Nat.sub", nil, "Nat → Nat → Nat",
		"fun (n : Nat) (m : Nat) => Nat.rec.{1} (fun (_ : Nat) => Nat) n (fun (_ : Nat) (ih : Nat) => Nat.pred ih) m"},
	valDecl{"def", "Nat.mul", nil, "Nat → Nat → Nat",
		"fun (n : Nat) (m : Nat) => Nat.rec.{1} (fun (_ : Nat) => Nat) Nat.zero (fun (_ : Nat) (ih : Nat) => Nat.add ih n) m"},
	valDecl{"def", "Nat.pow", nil, "Nat → Nat → Nat",
		"fun (n : Nat) (m : Nat) => Nat.rec.{1} (fun (_ : Nat) => Nat) 1 (fun (_ : Nat) (ih : Nat) => Nat.mul ih n) m"},
	valDecl{"def", "Nat.beq", nil, "Nat → Nat → Bool",
		"fun (n : Nat) => Nat.rec.{1} (fun (_ : Nat) => Nat → Bool) (fun (m : Nat) => Nat.rec.{1} (fun (_ : Nat) => Bool) Bool.true (fun (_ : Nat) (_ : Bool) => Bool.false) m) (fun (k : Nat) (ih : Nat → Bool) (m : Nat) => Nat.rec.{1} (fun (_ : Nat) => Bool) Bool.false (fun (m' : Nat) (_ : Bool) => ih m') m) n"},
	// Nat.div and Nat.mod are well-founded recursions in Lean, which this kernel cannot state. They are modelled
	// instead as a state machine run once per unit of the dividend: Nat.DivMod.mk q r keeps `q * k + r` equal to the
	// number of steps taken, and the step either bumps the remainder or, when it would reach `k`, wraps to
	// `mk (q+1) 0`. For `k = 0` the remainder never wraps, so `n / 0 = 0` and `n % 0 = n` as in Lean. Literals still
	// go through the kernel's Nat extension (`reduceNat`), so a big dividend costs one step, not one per unit.
	indDecl{"Nat.DivMod", nil, 0, "Type", [][2]string{{"Nat.DivMod.mk", "Nat → Nat → Nat.DivMod"}}},
	valDecl{"def", "Nat.divStep", nil, "Nat → Nat.DivMod → Nat.DivMod",
		"fun (k : Nat) (s : Nat.DivMod) => Nat.DivMod.rec.{1} (fun (_ : Nat.DivMod) => Nat.DivMod) " +
			"(fun (q : Nat) (r : Nat) => Bool.rec.{1} (fun (_ : Bool) => Nat.DivMod) " +
			"(Nat.DivMod.mk q (Nat.succ r)) (Nat.DivMod.mk (Nat.succ q) 0) (Nat.beq (Nat.succ r) k)) s"},
	valDecl{"def", "Nat.divIter", nil, "Nat → Nat → Nat.DivMod → Nat.DivMod",
		"fun (k : Nat) (j : Nat) (s : Nat.DivMod) => Nat.rec.{1} (fun (_ : Nat) => Nat.DivMod) s " +
			"(fun (_ : Nat) (ih : Nat.DivMod) => Nat.divStep k ih) j"},
	valDecl{"def", "Nat.divMod", nil, "Nat → Nat → Nat.DivMod",
		"fun (k : Nat) (n : Nat) => Nat.divIter k n (Nat.DivMod.mk 0 0)"},
	// Lean's `Nat.div`/`Nat.mod` take the dividend first, and so do these: `Nat.div a b` is `a / b`.
	valDecl{"def", "Nat.div", nil, "Nat → Nat → Nat",
		"fun (a : Nat) (b : Nat) => Nat.DivMod.rec.{1} (fun (_ : Nat.DivMod) => Nat) " +
			"(fun (q : Nat) (_ : Nat) => q) (Nat.divMod b a)"},
	valDecl{"def", "Nat.mod", nil, "Nat → Nat → Nat",
		"fun (a : Nat) (b : Nat) => Nat.DivMod.rec.{1} (fun (_ : Nat.DivMod) => Nat) " +
			"(fun (_ : Nat) (r : Nat) => r) (Nat.divMod b a)"},
	// Lean's `Nat.gcd` is a real definition (Init/Data/Nat/Gcd.lean): `gcd m n = if m = 0 then n else gcd (n % m) m`,
	// by well-founded recursion, which this kernel cannot state. Here it is the same recursion with the countdown made
	// explicit: a step maps `(a, b)` to `(b % a, a)`, whose first argument is smaller, so `a` steps always suffice, and
	// `Nat.gcd a b` runs the countdown from `a`. Nothing is opaque; the equations `gcd 0 b = b` and
	// `gcd (succ a) b = gcdIter a (b % succ a) (succ a)` hold definitionally, as in Lean. Literals still go through
	// the kernel's Nat extension, so a gcd of two numerals costs one step, not one per Euclid step.
	valDecl{"def", "Nat.gcdIter", nil, "Nat → Nat → Nat → Nat",
		"fun (f : Nat) (a : Nat) (b : Nat) => Nat.rec.{1} (fun (_ : Nat) => Nat → Nat → Nat) " +
			"(fun (a1 : Nat) (b1 : Nat) => b1) " +
			"(fun (k : Nat) (ih : Nat → Nat → Nat) (a1 : Nat) (b1 : Nat) => " +
			"Bool.rec.{1} (fun (_ : Bool) => Nat) (ih (Nat.mod b1 a1) a1) b1 (Nat.beq a1 0)) f a b"},
	valDecl{"def", "Nat.gcd", nil, "Nat → Nat → Nat", "fun (a : Nat) (b : Nat) => Nat.gcdIter a a b"},
	valDecl{"def", "Nat.ble", nil, "Nat → Nat → Bool",
		"fun (n : Nat) => Nat.rec.{1} (fun (_ : Nat) => Nat → Bool) (fun (m : Nat) => Bool.true) (fun (k : Nat) (ih : Nat → Bool) (m : Nat) => Nat.rec.{1} (fun (_ : Nat) => Bool) Bool.false (fun (m' : Nat) (_ : Bool) => ih m') m) n"},
	valDecl{"def", "Nat.lt", nil, "Nat → Nat → Prop", "fun (n : Nat) (m : Nat) => Nat.le (Nat.succ n) m"},
	valDecl{"def", "Nat.min", nil, "Nat → Nat → Nat",
		"fun (a : Nat) (b : Nat) => Bool.rec.{1} (fun (_ : Bool) => Nat) b a (Nat.ble a b)"},
	valDecl{"def", "Nat.max", nil, "Nat → Nat → Nat",
		"fun (a : Nat) (b : Nat) => Bool.rec.{1} (fun (_ : Bool) => Nat) a b (Nat.ble a b)"},
	valDecl{"def", "Nat.lcm", nil, "Nat → Nat → Nat", "fun (a : Nat) (b : Nat) => Nat.div (Nat.mul a b) (Nat.gcd a b)"},
	valDecl{"def", "Nat.factorial", nil, "Nat → Nat",
		"fun (n : Nat) => Nat.rec.{1} (fun (_ : Nat) => Nat) 1 (fun (k : Nat) (ih : Nat) => Nat.mul (Nat.succ k) ih) n"},
	valDecl{"def", "Nat.dvd", nil, "Nat → Nat → Prop",
		"fun (a : Nat) (b : Nat) => Exists.{1} Nat (fun (c : Nat) => Eq.{1} Nat b (Nat.mul a c))"},

	valDecl{"thm", "Nat.beq_refl", nil, "(n : Nat) → Eq.{1} Bool (Nat.beq n n) Bool.true",
		"fun (n : Nat) => Nat.rec.{0} (fun (k : Nat) => Eq.{1} Bool (Nat.beq k k) Bool.true) (Eq.refl.{1} Bool Bool.true) (fun (k : Nat) (ih : Eq.{1} Bool (Nat.beq k k) Bool.true) => ih) n"},
	valDecl{"thm", "Nat.eq_of_beq_eq_true", nil, "(n : Nat) → (m : Nat) → Eq.{1} Bool (Nat.beq n m) Bool.true → Eq.{1} Nat n m",
		"fun (n : Nat) => Nat.rec.{0} (fun (n : Nat) => (m : Nat) → Eq.{1} Bool (Nat.beq n m) Bool.true → Eq.{1} Nat n m) " +
			"(fun (m : Nat) => Nat.rec.{0} (fun (m : Nat) => Eq.{1} Bool (Nat.beq Nat.zero m) Bool.true → Eq.{1} Nat Nat.zero m) " +
			"(fun (h : Eq.{1} Bool (Nat.beq Nat.zero Nat.zero) Bool.true) => Eq.refl.{1} Nat Nat.zero) " +
			"(fun (m' : Nat) (_ : Eq.{1} Bool (Nat.beq Nat.zero m') Bool.true → Eq.{1} Nat Nat.zero m') (h : Eq.{1} Bool (Nat.beq Nat.zero (Nat.succ m')) Bool.true) => False.elim.{0} (Eq.{1} Nat Nat.zero (Nat.succ m')) (Bool.false_ne_true h)) m) " +
			"(fun (k : Nat) (ih : (m : Nat) → Eq.{1} Bool (Nat.beq k m) Bool.true → Eq.{1} Nat k m) (m : Nat) => " +
			"Nat.rec.{0} (fun (m : Nat) => Eq.{1} Bool (Nat.beq (Nat.succ k) m) Bool.true → Eq.{1} Nat (Nat.succ k) m) " +
			"(fun (h : Eq.{1} Bool (Nat.beq (Nat.succ k) Nat.zero) Bool.true) => False.elim.{0} (Eq.{1} Nat (Nat.succ k) Nat.zero) (Bool.false_ne_true h)) " +
			"(fun (m' : Nat) (_ : Eq.{1} Bool (Nat.beq (Nat.succ k) m') Bool.true → Eq.{1} Nat (Nat.succ k) m') (h : Eq.{1} Bool (Nat.beq (Nat.succ k) (Nat.succ m')) Bool.true) => congrArg.{1, 1} Nat Nat Nat.succ k m' (ih m' h)) m) n"},
	valDecl{"thm", "Nat.ne_of_beq_eq_false", nil, "(n : Nat) → (m : Nat) → Eq.{1} Bool (Nat.beq n m) Bool.false → Not (Eq.{1} Nat n m)",
		"fun (n : Nat) (m : Nat) (h : Eq.{1} Bool (Nat.beq n m) Bool.false) (e : Eq.{1} Nat n m) => " +
			"Eq.rec.{0, 1} Nat n (fun (x : Nat) (_ : Eq.{1} Nat n x) => Eq.{1} Bool (Nat.beq n x) Bool.false → False) " +
			"(fun (h' : Eq.{1} Bool (Nat.beq n n) Bool.false) => Bool.true_ne_false (Eq.trans.{1} Bool Bool.true (Nat.beq n n) Bool.false (Eq.symm.{1} Bool (Nat.beq n n) Bool.true (Nat.beq_refl n)) h')) m e h"},

	// The Nat facts the division lemmas rest on: the strict order over `Nat.le`, and zero for `+` and `*`
	// (Lean gets them from its equation compiler and `omega`; here they are inductions on `Nat.rec`/`Nat.le.rec`).
	valDecl{"thm", "Nat.succ_ne_zero", nil, "(n : Nat) → Not (Eq.{1} Nat (Nat.succ n) 0)",
		"fun (n : Nat) (h : Eq.{1} Nat (Nat.succ n) 0) => " +
			"Eq.rec.{0, 1} Nat (Nat.succ n) " +
			"(fun (z : Nat) (_ : Eq.{1} Nat (Nat.succ n) z) => " +
			"Nat.rec.{1} (fun (_ : Nat) => Prop) False (fun (_ : Nat) (_ : Prop) => True) z) " +
			"True.intro 0 h"},
	valDecl{"thm", "Nat.succ.inj", nil, "(a : Nat) → (b : Nat) → Eq.{1} Nat (Nat.succ a) (Nat.succ b) → Eq.{1} Nat a b",
		"fun (a : Nat) (b : Nat) (h : Eq.{1} Nat (Nat.succ a) (Nat.succ b)) => " +
			"congrArg.{1, 1} Nat Nat Nat.pred (Nat.succ a) (Nat.succ b) h"},
	valDecl{"thm", "Nat.le_zero", nil, "(n : Nat) → Nat.le n 0 → Eq.{1} Nat n 0",
		"fun (n : Nat) (h : Nat.le n 0) => Nat.le.rec n " +
			"(fun (a : Nat) (_ : Nat.le n a) => Eq.{1} Nat a 0 → Eq.{1} Nat n 0) " +
			"(fun (e : Eq.{1} Nat n 0) => e) " +
			"(fun (m : Nat) (_ : Nat.le n m) (_ : Eq.{1} Nat m 0 → Eq.{1} Nat n 0) (e : Eq.{1} Nat (Nat.succ m) 0) => " +
			"False.elim.{0} (Eq.{1} Nat n 0) (Nat.succ_ne_zero m e)) " +
			"0 h (Eq.refl.{1} Nat 0)"},
	valDecl{"thm", "Nat.le_of_le_pred", nil, "(a : Nat) → (m : Nat) → Nat.le a (Nat.pred m) → Nat.le a m",
		"fun (a : Nat) => Nat.rec.{0} (fun (m : Nat) => Nat.le a (Nat.pred m) → Nat.le a m) " +
			"(fun (h : Nat.le a (Nat.pred 0)) => h) " +
			"(fun (k : Nat) (_ : Nat.le a (Nat.pred k) → Nat.le a k) (h : Nat.le a (Nat.pred (Nat.succ k))) => " +
			"Nat.le.step a k h)"},
	valDecl{"thm", "Nat.succ_le_succ", nil, "(a : Nat) → (b : Nat) → Nat.le a b → Nat.le (Nat.succ a) (Nat.succ b)",
		"fun (a : Nat) (b : Nat) (h : Nat.le a b) => Nat.le.rec a " +
			"(fun (y : Nat) (_ : Nat.le a y) => Nat.le (Nat.succ a) (Nat.succ y)) " +
			"(Nat.le.refl (Nat.succ a)) " +
			"(fun (m : Nat) (_ : Nat.le a m) (ih : Nat.le (Nat.succ a) (Nat.succ m)) => Nat.le.step (Nat.succ a) (Nat.succ m) ih) " +
			"b h"},
	valDecl{"thm", "Nat.le_of_succ_le_succ", nil, "(a : Nat) → (b : Nat) → Nat.le (Nat.succ a) (Nat.succ b) → Nat.le a b",
		"fun (a : Nat) (b : Nat) (h : Nat.le (Nat.succ a) (Nat.succ b)) => Nat.le.rec (Nat.succ a) " +
			"(fun (y : Nat) (_ : Nat.le (Nat.succ a) y) => Nat.le a (Nat.pred y)) " +
			"(Nat.le.refl a) " +
			"(fun (m : Nat) (_ : Nat.le (Nat.succ a) m) (ih : Nat.le a (Nat.pred m)) => Nat.le_of_le_pred a m ih) " +
			"(Nat.succ b) h"},
	valDecl{"thm", "Nat.le_of_succ_le", nil, "(n : Nat) → (m : Nat) → Nat.le (Nat.succ n) m → Nat.le n m",
		"fun (n : Nat) (m : Nat) (h : Nat.le (Nat.succ n) m) => Nat.le_of_le_pred n m (Nat.le.rec (Nat.succ n) " +
			"(fun (y : Nat) (_ : Nat.le (Nat.succ n) y) => Nat.le n (Nat.pred y)) " +
			"(Nat.le.refl n) " +
			"(fun (k : Nat) (_ : Nat.le (Nat.succ n) k) (ih : Nat.le n (Nat.pred k)) => Nat.le_of_le_pred n k ih) " +
			"m h)"},
	valDecl{"thm", "Nat.not_le_succ_self", nil, "(n : Nat) → Not (Nat.le (Nat.succ n) n)",
		"fun (n : Nat) => Nat.rec.{0} (fun (k : Nat) => Not (Nat.le (Nat.succ k) k)) " +
			"(fun (h : Nat.le (Nat.succ 0) 0) => Nat.succ_ne_zero 0 (Nat.le_zero (Nat.succ 0) h)) " +
			"(fun (k : Nat) (ih : Not (Nat.le (Nat.succ k) k)) (h : Nat.le (Nat.succ (Nat.succ k)) (Nat.succ k)) => " +
			"ih (Nat.le_of_succ_le_succ (Nat.succ k) k h)) n"},
	valDecl{"thm", "Nat.lt_of_le_ne", nil,
		"(a : Nat) → (b : Nat) → Nat.le a b → Not (Eq.{1} Nat a b) → Nat.le (Nat.succ a) b",
		"fun (a : Nat) (b : Nat) (h : Nat.le a b) => Nat.le.rec a " +
			"(fun (y : Nat) (_ : Nat.le a y) => Not (Eq.{1} Nat a y) → Nat.le (Nat.succ a) y) " +
			"(fun (ne : Not (Eq.{1} Nat a a)) => False.elim.{0} (Nat.le (Nat.succ a) a) (ne (Eq.refl.{1} Nat a))) " +
			"(fun (m : Nat) (hm : Nat.le a m) (_ : Not (Eq.{1} Nat a m) → Nat.le (Nat.succ a) m) " +
			"(_ : Not (Eq.{1} Nat a (Nat.succ m))) => Nat.succ_le_succ a m hm) " +
			"b h"},
	valDecl{"thm", "Nat.ne_of_lt", nil, "(a : Nat) → (b : Nat) → Nat.lt a b → Not (Eq.{1} Nat a b)",
		"fun (a : Nat) (b : Nat) (h : Nat.lt a b) (e : Eq.{1} Nat a b) => " +
			"Eq.rec.{0, 1} Nat a (fun (z : Nat) (_ : Eq.{1} Nat a z) => Nat.le (Nat.succ a) z → False) " +
			"(Nat.not_le_succ_self a) b e h"},
	valDecl{"thm", "Nat.zero_add", nil, "(n : Nat) → Eq.{1} Nat (Nat.add 0 n) n",
		"fun (n : Nat) => Nat.rec.{0} (fun (k : Nat) => Eq.{1} Nat (Nat.add 0 k) k) " +
			"(Eq.refl.{1} Nat 0) " +
			"(fun (k : Nat) (ih : Eq.{1} Nat (Nat.add 0 k) k) => " +
			"congrArg.{1, 1} Nat Nat Nat.succ (Nat.add 0 k) k ih) n"},
	valDecl{"thm", "Nat.zero_mul", nil, "(n : Nat) → Eq.{1} Nat (Nat.mul 0 n) 0",
		"fun (n : Nat) => Nat.rec.{0} (fun (k : Nat) => Eq.{1} Nat (Nat.mul 0 k) 0) " +
			"(Eq.refl.{1} Nat 0) " +
			"(fun (k : Nat) (ih : Eq.{1} Nat (Nat.mul 0 k) 0) => ih) n"},
	valDecl{"thm", "Nat.beq_eq_false_of_ne", nil,
		"(a : Nat) → (b : Nat) → Not (Eq.{1} Nat a b) → Eq.{1} Bool (Nat.beq a b) Bool.false",
		"fun (a : Nat) (b : Nat) (h : Not (Eq.{1} Nat a b)) => " +
			"Bool.rec.{0} (fun (v : Bool) => Eq.{1} Bool (Nat.beq a b) v → Eq.{1} Bool (Nat.beq a b) Bool.false) " +
			"(fun (hv : Eq.{1} Bool (Nat.beq a b) Bool.false) => hv) " +
			"(fun (hv : Eq.{1} Bool (Nat.beq a b) Bool.true) => " +
			"False.elim.{0} (Eq.{1} Bool (Nat.beq a b) Bool.false) (h (Nat.eq_of_beq_eq_true a b hv))) " +
			"(Nat.beq a b) (Eq.refl.{1} Bool (Nat.beq a b))"},

	// The two ways one step of the division machine can go; both read the Bool the step tested.
	valDecl{"thm", "Nat.divStep_true", nil,
		"(k : Nat) → (q : Nat) → (r : Nat) → Eq.{1} Bool (Nat.beq (Nat.succ r) k) Bool.true → " +
			"Eq.{1} Nat.DivMod (Nat.divStep k (Nat.DivMod.mk q r)) (Nat.DivMod.mk (Nat.succ q) 0)",
		"fun (k : Nat) (q : Nat) (r : Nat) (h : Eq.{1} Bool (Nat.beq (Nat.succ r) k) Bool.true) => " +
			"Eq.rec.{0, 1} Bool Bool.true " +
			"(fun (v : Bool) (_ : Eq.{1} Bool Bool.true v) => Eq.{1} Nat.DivMod " +
			"(Bool.rec.{1} (fun (_ : Bool) => Nat.DivMod) (Nat.DivMod.mk q (Nat.succ r)) (Nat.DivMod.mk (Nat.succ q) 0) v) " +
			"(Nat.DivMod.mk (Nat.succ q) 0)) " +
			"(Eq.refl.{1} Nat.DivMod (Nat.DivMod.mk (Nat.succ q) 0)) (Nat.beq (Nat.succ r) k) " +
			"(Eq.symm.{1} Bool (Nat.beq (Nat.succ r) k) Bool.true h)"},
	valDecl{"thm", "Nat.divStep_false", nil,
		"(k : Nat) → (q : Nat) → (r : Nat) → Eq.{1} Bool (Nat.beq (Nat.succ r) k) Bool.false → " +
			"Eq.{1} Nat.DivMod (Nat.divStep k (Nat.DivMod.mk q r)) (Nat.DivMod.mk q (Nat.succ r))",
		"fun (k : Nat) (q : Nat) (r : Nat) (h : Eq.{1} Bool (Nat.beq (Nat.succ r) k) Bool.false) => " +
			"Eq.rec.{0, 1} Bool Bool.false " +
			"(fun (v : Bool) (_ : Eq.{1} Bool Bool.false v) => Eq.{1} Nat.DivMod " +
			"(Bool.rec.{1} (fun (_ : Bool) => Nat.DivMod) (Nat.DivMod.mk q (Nat.succ r)) (Nat.DivMod.mk (Nat.succ q) 0) v) " +
			"(Nat.DivMod.mk q (Nat.succ r))) " +
			"(Eq.refl.{1} Nat.DivMod (Nat.DivMod.mk q (Nat.succ r))) (Nat.beq (Nat.succ r) k) " +
			"(Eq.symm.{1} Bool (Nat.beq (Nat.succ r) k) Bool.false h)"},
	valDecl{"thm", "Nat.divIter_add", nil,
		"(k : Nat) → (a : Nat) → (b : Nat) → (s : Nat.DivMod) → " +
			"Eq.{1} Nat.DivMod (Nat.divIter k (Nat.add a b) s) (Nat.divIter k b (Nat.divIter k a s))",
		"fun (k : Nat) (a : Nat) => Nat.rec.{0} " +
			"(fun (b : Nat) => (s : Nat.DivMod) → Eq.{1} Nat.DivMod (Nat.divIter k (Nat.add a b) s) " +
			"(Nat.divIter k b (Nat.divIter k a s))) " +
			"(fun (s : Nat.DivMod) => Eq.refl.{1} Nat.DivMod (Nat.divIter k a s)) " +
			"(fun (b : Nat) (ih : (s : Nat.DivMod) → Eq.{1} Nat.DivMod (Nat.divIter k (Nat.add a b) s) " +
			"(Nat.divIter k b (Nat.divIter k a s))) (s : Nat.DivMod) => " +
			"congrArg.{1, 1} Nat.DivMod Nat.DivMod (Nat.divStep k) (Nat.divIter k (Nat.add a b) s) " +
			"(Nat.divIter k b (Nat.divIter k a s)) (ih s))"},

	// Where the machine stands after `j` steps that never wrap, and after the `k - r` steps that do.
	valDecl{"thm", "Nat.divIter_lt", nil,
		"(k : Nat) → (q : Nat) → (r : Nat) → (j : Nat) → Nat.lt (Nat.add r j) k → " +
			"Eq.{1} Nat.DivMod (Nat.divIter k j (Nat.DivMod.mk q r)) (Nat.DivMod.mk q (Nat.add r j))",
		"fun (k : Nat) (q : Nat) (r : Nat) => Nat.rec.{0} " +
			"(fun (j : Nat) => Nat.lt (Nat.add r j) k → Eq.{1} Nat.DivMod (Nat.divIter k j (Nat.DivMod.mk q r)) " +
			"(Nat.DivMod.mk q (Nat.add r j))) " +
			"(fun (h : Nat.lt (Nat.add r 0) k) => Eq.refl.{1} Nat.DivMod (Nat.DivMod.mk q (Nat.add r 0))) " +
			"(fun (j : Nat) (ih : Nat.lt (Nat.add r j) k → Eq.{1} Nat.DivMod (Nat.divIter k j (Nat.DivMod.mk q r)) " +
			"(Nat.DivMod.mk q (Nat.add r j))) (h : Nat.lt (Nat.add r (Nat.succ j)) k) => " +
			"Eq.trans.{1} Nat.DivMod (Nat.divStep k (Nat.divIter k j (Nat.DivMod.mk q r))) " +
			"(Nat.divStep k (Nat.DivMod.mk q (Nat.add r j))) (Nat.DivMod.mk q (Nat.add r (Nat.succ j))) " +
			"(congrArg.{1, 1} Nat.DivMod Nat.DivMod (Nat.divStep k) (Nat.divIter k j (Nat.DivMod.mk q r)) " +
			"(Nat.DivMod.mk q (Nat.add r j)) (ih (Nat.le_of_succ_le (Nat.succ (Nat.add r j)) k h))) " +
			"(Nat.divStep_false k q (Nat.add r j) (Nat.beq_eq_false_of_ne (Nat.succ (Nat.add r j)) k " +
			"(Nat.ne_of_lt (Nat.succ (Nat.add r j)) k h))))"},
	valDecl{"thm", "Nat.divIter_wrap", nil,
		"(k : Nat) → (q : Nat) → (r : Nat) → (j : Nat) → Eq.{1} Nat (Nat.add r j) k → Nat.lt r k → " +
			"Eq.{1} Nat.DivMod (Nat.divIter k j (Nat.DivMod.mk q r)) (Nat.DivMod.mk (Nat.succ q) 0)",
		"fun (k : Nat) (q : Nat) (r : Nat) => Nat.rec.{0} " +
			"(fun (j : Nat) => Eq.{1} Nat (Nat.add r j) k → Nat.lt r k → " +
			"Eq.{1} Nat.DivMod (Nat.divIter k j (Nat.DivMod.mk q r)) (Nat.DivMod.mk (Nat.succ q) 0)) " +
			"(fun (h : Eq.{1} Nat (Nat.add r 0) k) (hr : Nat.lt r k) => " +
			"False.elim.{0} (Eq.{1} Nat.DivMod (Nat.divIter k 0 (Nat.DivMod.mk q r)) (Nat.DivMod.mk (Nat.succ q) 0)) " +
			"(Nat.not_le_succ_self r (Eq.rec.{0, 1} Nat k (fun (z : Nat) (_ : Eq.{1} Nat k z) => Nat.le (Nat.succ r) z) " +
			"hr r (Eq.symm.{1} Nat r k h)))) " +
			"(fun (j : Nat) (_ : Eq.{1} Nat (Nat.add r j) k → Nat.lt r k → " +
			"Eq.{1} Nat.DivMod (Nat.divIter k j (Nat.DivMod.mk q r)) (Nat.DivMod.mk (Nat.succ q) 0)) " +
			"(h : Eq.{1} Nat (Nat.add r (Nat.succ j)) k) (hr : Nat.lt r k) => " +
			"Eq.trans.{1} Nat.DivMod (Nat.divStep k (Nat.divIter k j (Nat.DivMod.mk q r))) " +
			"(Nat.divStep k (Nat.DivMod.mk q (Nat.add r j))) (Nat.DivMod.mk (Nat.succ q) 0) " +
			"(congrArg.{1, 1} Nat.DivMod Nat.DivMod (Nat.divStep k) (Nat.divIter k j (Nat.DivMod.mk q r)) " +
			"(Nat.DivMod.mk q (Nat.add r j)) (Nat.divIter_lt k q r j " +
			"(Eq.rec.{0, 1} Nat (Nat.succ (Nat.add r j)) " +
			"(fun (z : Nat) (_ : Eq.{1} Nat (Nat.succ (Nat.add r j)) z) => Nat.le (Nat.succ (Nat.add r j)) z) " +
			"(Nat.le.refl (Nat.succ (Nat.add r j))) k h))) " +
			"(Nat.divStep_true k q (Nat.add r j) (Eq.rec.{0, 1} Nat (Nat.succ (Nat.add r j)) " +
			"(fun (z : Nat) (_ : Eq.{1} Nat (Nat.succ (Nat.add r j)) z) => " +
			"Eq.{1} Bool (Nat.beq (Nat.succ (Nat.add r j)) z) Bool.true) (Nat.beq_refl (Nat.succ (Nat.add r j))) k h)))"},

	valDecl{"thm", "Nat.DivMod.mk_congr", nil,
		"(a : Nat) → (b : Nat) → (c : Nat) → (d : Nat) → Eq.{1} Nat a c → Eq.{1} Nat b d → " +
			"Eq.{1} Nat.DivMod (Nat.DivMod.mk a b) (Nat.DivMod.mk c d)",
		"fun (a : Nat) (b : Nat) (c : Nat) (d : Nat) (h1 : Eq.{1} Nat a c) (h2 : Eq.{1} Nat b d) => " +
			"Eq.rec.{0, 1} Nat a (fun (z : Nat) (_ : Eq.{1} Nat a z) => " +
			"Eq.{1} Nat.DivMod (Nat.DivMod.mk a b) (Nat.DivMod.mk z d)) " +
			"(Eq.rec.{0, 1} Nat b (fun (w : Nat) (_ : Eq.{1} Nat b w) => " +
			"Eq.{1} Nat.DivMod (Nat.DivMod.mk a b) (Nat.DivMod.mk a w)) " +
			"(Eq.refl.{1} Nat.DivMod (Nat.DivMod.mk a b)) d h2) c h1"},
	valDecl{"thm", "Nat.add_congr", nil,
		"(a : Nat) → (b : Nat) → (c : Nat) → (d : Nat) → Eq.{1} Nat a c → Eq.{1} Nat b d → " +
			"Eq.{1} Nat (Nat.add a b) (Nat.add c d)",
		"fun (a : Nat) (b : Nat) (c : Nat) (d : Nat) (h1 : Eq.{1} Nat a c) (h2 : Eq.{1} Nat b d) => " +
			"Eq.rec.{0, 1} Nat a (fun (z : Nat) (_ : Eq.{1} Nat a z) => " +
			"Eq.{1} Nat (Nat.add a b) (Nat.add z d)) " +
			"(Eq.rec.{0, 1} Nat b (fun (w : Nat) (_ : Eq.{1} Nat b w) => " +
			"Eq.{1} Nat (Nat.add a b) (Nat.add a w)) " +
			"(Eq.refl.{1} Nat (Nat.add a b)) d h2) c h1"},

	// `n / k` and `n % k` are the two components of the machine's state after `n` steps.
	valDecl{"thm", "Nat.div_proj", nil,
		"(k : Nat) → (n : Nat) → (q : Nat) → (r : Nat) → Eq.{1} Nat.DivMod (Nat.divMod k n) (Nat.DivMod.mk q r) → " +
			"Eq.{1} Nat (Nat.div n k) q",
		"fun (k : Nat) (n : Nat) (q : Nat) (r : Nat) (h : Eq.{1} Nat.DivMod (Nat.divMod k n) (Nat.DivMod.mk q r)) => " +
			"congrArg.{1, 1} Nat.DivMod Nat " + dmFst + " (Nat.divMod k n) (Nat.DivMod.mk q r) h"},
	valDecl{"thm", "Nat.mod_proj", nil,
		"(k : Nat) → (n : Nat) → (q : Nat) → (r : Nat) → Eq.{1} Nat.DivMod (Nat.divMod k n) (Nat.DivMod.mk q r) → " +
			"Eq.{1} Nat (Nat.mod n k) r",
		"fun (k : Nat) (n : Nat) (q : Nat) (r : Nat) (h : Eq.{1} Nat.DivMod (Nat.divMod k n) (Nat.DivMod.mk q r)) => " +
			"congrArg.{1, 1} Nat.DivMod Nat " + dmSnd + " (Nat.divMod k n) (Nat.DivMod.mk q r) h"},

	// The state after `n + 1` steps and the identity at `n + 1`, one branch of the step at a time.
	valDecl{"thm", "Nat.divMod_mk_false", nil,
		"(k : Nat) → (n : Nat) → " + dmIh + " → Eq.{1} Bool " + dmBeq + " Bool.false → " + dmAlphaS,
		"fun (k : Nat) (n : Nat) (ih : " + dmIh + ") (hv : Eq.{1} Bool " + dmBeq + " Bool.false) => " +
			// the state is `mk (n/k) (n%k + 1)`, so `div` and `mod` at n+1 are the components of that
			"Eq.trans.{1} Nat.DivMod (Nat.divMod k (Nat.succ n)) " + dmStateF + " " + dmStateS + " " + dmStepF + " " +
			"(Nat.DivMod.mk_congr (Nat.div n k) (Nat.succ (Nat.mod n k)) (Nat.div (Nat.succ n) k) " +
			"(Nat.mod (Nat.succ n) k) " +
			"(Eq.symm.{1} Nat (Nat.div (Nat.succ n) k) (Nat.div n k) (Nat.div_proj k (Nat.succ n) (Nat.div n k) " +
			"(Nat.succ (Nat.mod n k)) " + dmStepF + ")) " +
			"(Eq.symm.{1} Nat (Nat.mod (Nat.succ n) k) (Nat.succ (Nat.mod n k)) (Nat.mod_proj k (Nat.succ n) " +
			"(Nat.div n k) (Nat.succ (Nat.mod n k)) " + dmStepF + ")))"},
	valDecl{"thm", "Nat.divMod_add_false", nil,
		"(k : Nat) → (n : Nat) → " + dmIh + " → Eq.{1} Bool " + dmBeq + " Bool.false → " + dmBetaS,
		"fun (k : Nat) (n : Nat) (ih : " + dmIh + ") (hv : Eq.{1} Bool " + dmBeq + " Bool.false) => " +
			// `k * (n/k) + (n%k + 1) = (k * (n/k) + n%k) + 1 = n + 1`
			"Eq.trans.{1} Nat (Nat.add (Nat.mul k (Nat.div (Nat.succ n) k)) (Nat.mod (Nat.succ n) k)) " +
			"(Nat.add (Nat.mul k (Nat.div n k)) (Nat.succ (Nat.mod n k))) (Nat.succ n) " +
			"(Nat.add_congr (Nat.mul k (Nat.div (Nat.succ n) k)) (Nat.mod (Nat.succ n) k) (Nat.mul k (Nat.div n k)) " +
			"(Nat.succ (Nat.mod n k)) " +
			"(congrArg.{1, 1} Nat Nat (Nat.mul k) (Nat.div (Nat.succ n) k) (Nat.div n k) " +
			"(Nat.div_proj k (Nat.succ n) (Nat.div n k) (Nat.succ (Nat.mod n k)) " + dmStepF + ")) " +
			"(Nat.mod_proj k (Nat.succ n) (Nat.div n k) (Nat.succ (Nat.mod n k)) " + dmStepF + ")) " +
			"(congrArg.{1, 1} Nat Nat Nat.succ (Nat.add (Nat.mul k (Nat.div n k)) (Nat.mod n k)) n " + dmIhB + ")"},
	valDecl{"thm", "Nat.divMod_mk_true", nil,
		"(k : Nat) → (n : Nat) → " + dmIh + " → Eq.{1} Bool " + dmBeq + " Bool.true → " + dmAlphaS,
		"fun (k : Nat) (n : Nat) (ih : " + dmIh + ") (hv : Eq.{1} Bool " + dmBeq + " Bool.true) => " +
			// the state is `mk (n/k + 1) 0`
			"Eq.trans.{1} Nat.DivMod (Nat.divMod k (Nat.succ n)) " + dmStateT + " " + dmStateS + " " + dmStepT + " " +
			"(Nat.DivMod.mk_congr (Nat.succ (Nat.div n k)) 0 (Nat.div (Nat.succ n) k) (Nat.mod (Nat.succ n) k) " +
			"(Eq.symm.{1} Nat (Nat.div (Nat.succ n) k) (Nat.succ (Nat.div n k)) (Nat.div_proj k (Nat.succ n) " +
			"(Nat.succ (Nat.div n k)) 0 " + dmStepT + ")) " +
			"(Eq.symm.{1} Nat (Nat.mod (Nat.succ n) k) 0 (Nat.mod_proj k (Nat.succ n) (Nat.succ (Nat.div n k)) 0 " +
			dmStepT + ")))"},
	valDecl{"thm", "Nat.divMod_add_true", nil,
		"(k : Nat) → (n : Nat) → " + dmIh + " → Eq.{1} Bool " + dmBeq + " Bool.true → " + dmBetaS,
		"fun (k : Nat) (n : Nat) (ih : " + dmIh + ") (hv : Eq.{1} Bool " + dmBeq + " Bool.true) => " +
			// `k * (n/k + 1) + 0 = k * (n/k) + k = k * (n/k) + (n%k + 1) = n + 1`, the middle step using `n%k + 1 = k`
			"Eq.trans.{1} Nat (Nat.add (Nat.mul k (Nat.div (Nat.succ n) k)) (Nat.mod (Nat.succ n) k)) " +
			"(Nat.add (Nat.mul k (Nat.succ (Nat.div n k))) 0) (Nat.succ n) " +
			"(Nat.add_congr (Nat.mul k (Nat.div (Nat.succ n) k)) (Nat.mod (Nat.succ n) k) " +
			"(Nat.mul k (Nat.succ (Nat.div n k))) 0 " +
			"(congrArg.{1, 1} Nat Nat (Nat.mul k) (Nat.div (Nat.succ n) k) (Nat.succ (Nat.div n k)) " +
			"(Nat.div_proj k (Nat.succ n) (Nat.succ (Nat.div n k)) 0 " + dmStepT + ")) " +
			"(Nat.mod_proj k (Nat.succ n) (Nat.succ (Nat.div n k)) 0 " + dmStepT + ")) " +
			"(Eq.trans.{1} Nat (Nat.add (Nat.mul k (Nat.succ (Nat.div n k))) 0) " +
			"(Nat.succ (Nat.add (Nat.mul k (Nat.div n k)) (Nat.mod n k))) (Nat.succ n) " +
			"(Eq.rec.{0, 1} Nat (Nat.succ (Nat.mod n k)) " +
			"(fun (z : Nat) (_ : Eq.{1} Nat (Nat.succ (Nat.mod n k)) z) => " +
			"Eq.{1} Nat (Nat.add (Nat.mul k (Nat.div n k)) z) " +
			"(Nat.succ (Nat.add (Nat.mul k (Nat.div n k)) (Nat.mod n k)))) " +
			"(Eq.refl.{1} Nat (Nat.succ (Nat.add (Nat.mul k (Nat.div n k)) (Nat.mod n k)))) k " +
			"(Nat.eq_of_beq_eq_true (Nat.succ (Nat.mod n k)) k hv)) " +
			"(congrArg.{1, 1} Nat Nat Nat.succ (Nat.add (Nat.mul k (Nat.div n k)) (Nat.mod n k)) n " + dmIhB + "))"},

	// `n = k * (n / k) + n % k` for every `n`, proved together with `divMod k n = mk (n / k) (n % k)`.
	valDecl{"thm", "Nat.divMod_spec", nil, "(k : Nat) → (n : Nat) → " + dmIh,
		"fun (k : Nat) => Nat.rec.{0} (fun (n : Nat) => " + dmIh + ") " +
			"(And.intro (Eq.{1} Nat.DivMod (Nat.divMod k 0) (Nat.DivMod.mk (Nat.div 0 k) (Nat.mod 0 k))) " +
			"(Eq.{1} Nat (Nat.add (Nat.mul k (Nat.div 0 k)) (Nat.mod 0 k)) 0) " +
			"(Eq.refl.{1} Nat.DivMod (Nat.DivMod.mk 0 0)) (Eq.refl.{1} Nat 0)) " +
			"(fun (n : Nat) (ih : " + dmIh + ") => Bool.rec.{0} " +
			"(fun (v : Bool) => Eq.{1} Bool " + dmBeq + " v → And (" + dmAlphaS + ") (" + dmBetaS + ")) " +
			"(fun (hv : Eq.{1} Bool " + dmBeq + " Bool.false) => " +
			"And.intro (" + dmAlphaS + ") (" + dmBetaS + ") (Nat.divMod_mk_false k n ih hv) " +
			"(Nat.divMod_add_false k n ih hv)) " +
			"(fun (hv : Eq.{1} Bool " + dmBeq + " Bool.true) => " +
			"And.intro (" + dmAlphaS + ") (" + dmBetaS + ") (Nat.divMod_mk_true k n ih hv) " +
			"(Nat.divMod_add_true k n ih hv)) " +
			dmBeq + " (Eq.refl.{1} Bool " + dmBeq + "))"},
	valDecl{"thm", "Nat.zero_lt_succ", nil, "(n : Nat) → Nat.lt 0 (Nat.succ n)",
		"fun (n : Nat) => Nat.rec.{0} (fun (n : Nat) => Nat.lt 0 (Nat.succ n)) (Nat.le.refl 1) " +
			"(fun (k : Nat) (ih : Nat.lt 0 (Nat.succ k)) => Nat.le.step (Nat.succ 0) (Nat.succ k) ih) n"},

	// From `mk c 0` the machine walks `k` steps back to `mk (c+1) 0`, which is the induction that `k` divides `k * c`.
	valDecl{"thm", "Nat.divMod_mul", nil,
		"(k : Nat) → (c : Nat) → Nat.lt 0 k → Eq.{1} Nat.DivMod (Nat.divMod k (Nat.mul k c)) (Nat.DivMod.mk c 0)",
		"fun (k : Nat) (c : Nat) (hk : Nat.lt 0 k) => Nat.rec.{0} " +
			"(fun (c : Nat) => Eq.{1} Nat.DivMod (Nat.divMod k (Nat.mul k c)) (Nat.DivMod.mk c 0)) " +
			"(Eq.refl.{1} Nat.DivMod (Nat.DivMod.mk 0 0)) " +
			"(fun (c : Nat) (ih : Eq.{1} Nat.DivMod (Nat.divMod k (Nat.mul k c)) (Nat.DivMod.mk c 0)) => " +
			"Eq.trans.{1} Nat.DivMod (Nat.divIter k (Nat.add (Nat.mul k c) k) (Nat.DivMod.mk 0 0)) " +
			"(Nat.divIter k k (Nat.divMod k (Nat.mul k c))) (Nat.DivMod.mk (Nat.succ c) 0) " +
			"(Nat.divIter_add k (Nat.mul k c) k (Nat.DivMod.mk 0 0)) " +
			"(Eq.trans.{1} Nat.DivMod (Nat.divIter k k (Nat.divMod k (Nat.mul k c))) " +
			"(Nat.divIter k k (Nat.DivMod.mk c 0)) (Nat.DivMod.mk (Nat.succ c) 0) " +
			"(congrArg.{1, 1} Nat.DivMod Nat.DivMod (Nat.divIter k k) (Nat.divMod k (Nat.mul k c)) " +
			"(Nat.DivMod.mk c 0) ih) " +
			"(Nat.divIter_wrap k c 0 k (Nat.zero_add k) hk))) " +
			"c"},

	// `k ∣ n → n % k = 0`: with `n = k * c` the machine ends at `mk c 0`, so the remainder is `0`. For `k = 0` the
	// same equation says `n = 0`, whose remainder is `0`.
	valDecl{"thm", "Nat.mod_eq_zero_of_dvd", nil, "(k : Nat) → (n : Nat) → Nat.dvd k n → Eq.{1} Nat (Nat.mod n k) 0",
		"fun (k : Nat) => Nat.rec.{0} (fun (k : Nat) => (n : Nat) → Nat.dvd k n → Eq.{1} Nat (Nat.mod n k) 0) " +
			"(fun (n : Nat) (h : Nat.dvd 0 n) => Exists.rec.{1} Nat (fun (c : Nat) => Eq.{1} Nat n (Nat.mul 0 c)) " +
			"(fun (_ : Nat.dvd 0 n) => Eq.{1} Nat (Nat.mod n 0) 0) " +
			"(fun (c : Nat) (e : Eq.{1} Nat n (Nat.mul 0 c)) => " +
			"Eq.rec.{0, 1} Nat 0 (fun (z : Nat) (_ : Eq.{1} Nat 0 z) => Eq.{1} Nat (Nat.mod z 0) 0) " +
			"(Eq.refl.{1} Nat 0) n " +
			"(Eq.symm.{1} Nat n 0 (Eq.trans.{1} Nat n (Nat.mul 0 c) 0 e (Nat.zero_mul c)))) h) " +
			"(fun (k' : Nat) (_ : (n : Nat) → Nat.dvd k' n → Eq.{1} Nat (Nat.mod n k') 0) (n : Nat) " +
			"(h : Nat.dvd (Nat.succ k') n) => Exists.rec.{1} Nat " +
			"(fun (c : Nat) => Eq.{1} Nat n (Nat.mul (Nat.succ k') c)) " +
			"(fun (_ : Nat.dvd (Nat.succ k') n) => Eq.{1} Nat (Nat.mod n (Nat.succ k')) 0) " +
			"(fun (c : Nat) (e : Eq.{1} Nat n (Nat.mul (Nat.succ k') c)) => " +
			"Eq.rec.{0, 1} Nat (Nat.mul (Nat.succ k') c) " +
			"(fun (z : Nat) (_ : Eq.{1} Nat (Nat.mul (Nat.succ k') c) z) => " +
			"Eq.{1} Nat (Nat.mod z (Nat.succ k')) 0) " +
			"(Nat.mod_proj (Nat.succ k') (Nat.mul (Nat.succ k') c) c 0 " +
			"(Nat.divMod_mul (Nat.succ k') c (Nat.zero_lt_succ k'))) n " +
			"(Eq.symm.{1} Nat n (Nat.mul (Nat.succ k') c) e)) h) k"},

	// `n % k = 0 → k ∣ n`, with witness `n / k`: the identity `n = k * (n / k) + n % k` loses its last term.
	valDecl{"thm", "Nat.dvd_of_mod_eq_zero", nil,
		"(k : Nat) → (n : Nat) → Eq.{1} Nat (Nat.mod n k) 0 → Nat.dvd k n",
		"fun (k : Nat) (n : Nat) (e : Eq.{1} Nat (Nat.mod n k) 0) => " +
			"Exists.intro.{1} Nat (fun (c : Nat) => Eq.{1} Nat n (Nat.mul k c)) (Nat.div n k) " +
			"(Eq.symm.{1} Nat (Nat.mul k (Nat.div n k)) n " +
			"(Eq.rec.{0, 1} Nat (Nat.mod n k) (fun (z : Nat) (_ : Eq.{1} Nat (Nat.mod n k) z) => " +
			"Eq.{1} Nat (Nat.add (Nat.mul k (Nat.div n k)) z) n) " +
			"(And.right (" + dmAlpha + ") (" + dmBeta + ") (Nat.divMod_spec k n)) 0 e))"},

	// `Nat.ble` decides `Nat.le` because it is a definition here: Lean proves these two with its equation compiler,
	// here they are inductions on `Nat.rec` (both arguments) and `Nat.le.rec`.
	valDecl{"thm", "Nat.ble_refl", nil, "(n : Nat) → Eq.{1} Bool (Nat.ble n n) Bool.true",
		"fun (n : Nat) => Nat.rec.{0} (fun (k : Nat) => Eq.{1} Bool (Nat.ble k k) Bool.true) " +
			"(Eq.refl.{1} Bool Bool.true) " +
			"(fun (k : Nat) (ih : Eq.{1} Bool (Nat.ble k k) Bool.true) => ih) n"},
	valDecl{"thm", "Nat.ble_step", nil,
		"(n : Nat) → (m : Nat) → Eq.{1} Bool (Nat.ble n m) Bool.true → Eq.{1} Bool (Nat.ble n (Nat.succ m)) Bool.true",
		"fun (n : Nat) => Nat.rec.{0} " +
			"(fun (k : Nat) => (m : Nat) → Eq.{1} Bool (Nat.ble k m) Bool.true → " +
			"Eq.{1} Bool (Nat.ble k (Nat.succ m)) Bool.true) " +
			"(fun (m : Nat) (_ : Eq.{1} Bool (Nat.ble 0 m) Bool.true) => Eq.refl.{1} Bool Bool.true) " +
			"(fun (k : Nat) (ih : (m : Nat) → Eq.{1} Bool (Nat.ble k m) Bool.true → " +
			"Eq.{1} Bool (Nat.ble k (Nat.succ m)) Bool.true) (m : Nat) => " +
			"Nat.rec.{0} (fun (m : Nat) => Eq.{1} Bool (Nat.ble (Nat.succ k) m) Bool.true → " +
			"Eq.{1} Bool (Nat.ble (Nat.succ k) (Nat.succ m)) Bool.true) " +
			"(fun (h : Eq.{1} Bool (Nat.ble (Nat.succ k) 0) Bool.true) => " +
			"False.elim.{0} (Eq.{1} Bool (Nat.ble (Nat.succ k) (Nat.succ 0)) Bool.true) (Bool.false_ne_true h)) " +
			"(fun (m' : Nat) (_ : Eq.{1} Bool (Nat.ble (Nat.succ k) m') Bool.true → " +
			"Eq.{1} Bool (Nat.ble (Nat.succ k) (Nat.succ m')) Bool.true) " +
			"(h : Eq.{1} Bool (Nat.ble (Nat.succ k) (Nat.succ m')) Bool.true) => ih m' h) " +
			"m) n"},
	// `n ≤ m → Nat.ble n m = true`: the reflection of `Nat.le` into `Bool`.
	valDecl{"thm", "Nat.ble_true_of_le", nil,
		"(n : Nat) → (m : Nat) → Nat.le n m → Eq.{1} Bool (Nat.ble n m) Bool.true",
		"fun (n : Nat) (m : Nat) (h : Nat.le n m) => Nat.le.rec n " +
			"(fun (y : Nat) (_ : Nat.le n y) => Eq.{1} Bool (Nat.ble n y) Bool.true) " +
			"(Nat.ble_refl n) " +
			"(fun (y : Nat) (_ : Nat.le n y) (ih : Eq.{1} Bool (Nat.ble n y) Bool.true) => Nat.ble_step n y ih) " +
			"m h"},
	valDecl{"thm", "Nat.le_of_ble_eq_true", nil,
		"(n : Nat) → (m : Nat) → Eq.{1} Bool (Nat.ble n m) Bool.true → Nat.le n m",
		"fun (n : Nat) => Nat.rec.{0} " +
			"(fun (k : Nat) => (m : Nat) → Eq.{1} Bool (Nat.ble k m) Bool.true → Nat.le k m) " +
			"(fun (m : Nat) (_ : Eq.{1} Bool (Nat.ble 0 m) Bool.true) => Nat.rec.{0} " +
			"(fun (m : Nat) => Nat.le 0 m) (Nat.le.refl 0) " +
			"(fun (m' : Nat) (ih : Nat.le 0 m') => Nat.le.step 0 m' ih) m) " +
			"(fun (k : Nat) (ih : (m : Nat) → Eq.{1} Bool (Nat.ble k m) Bool.true → Nat.le k m) (m : Nat) => " +
			"Nat.rec.{0} (fun (m : Nat) => Eq.{1} Bool (Nat.ble (Nat.succ k) m) Bool.true → Nat.le (Nat.succ k) m) " +
			"(fun (h : Eq.{1} Bool (Nat.ble (Nat.succ k) 0) Bool.true) => " +
			"False.elim.{0} (Nat.le (Nat.succ k) 0) (Bool.false_ne_true h)) " +
			"(fun (m' : Nat) (_ : Eq.{1} Bool (Nat.ble (Nat.succ k) m') Bool.true → Nat.le (Nat.succ k) m') " +
			"(h : Eq.{1} Bool (Nat.ble (Nat.succ k) (Nat.succ m')) Bool.true) => " +
			"Nat.succ_le_succ k m' (ih m' h)) " +
			"m) n"},
	// The other direction of the reflection: a false `Nat.ble` refutes `Nat.le` (the contrapositive of `ble_true_of_le`).
	valDecl{"thm", "Nat.not_le_of_not_ble_eq_true", nil,
		"(n : Nat) → (m : Nat) → Not (Eq.{1} Bool (Nat.ble n m) Bool.true) → Not (Nat.le n m)",
		"fun (n : Nat) (m : Nat) (h : Not (Eq.{1} Bool (Nat.ble n m) Bool.true)) (hle : Nat.le n m) => " +
			"h (Nat.ble_true_of_le n m hle)"},
	valDecl{"thm", "Nat.dvd_iff_mod_eq_zero", nil, "(k : Nat) → (n : Nat) → Iff (Nat.dvd k n) (Eq.{1} Nat (Nat.mod n k) 0)",
		"fun (k : Nat) (n : Nat) => Iff.intro (Nat.dvd k n) (Eq.{1} Nat (Nat.mod n k) 0) " +
			"(Nat.mod_eq_zero_of_dvd k n) (Nat.dvd_of_mod_eq_zero k n)"},

	// The gcd equations Lean gets from its equation compiler. `gcd 0 b = b` and the step unfold the definition; `gcd n 0
	// = n` needs `gcdIter_zero`, since `gcd n 0` takes one step before its first argument is 0.
	valDecl{"thm", "Nat.gcdIter_zero", nil, "(f : Nat) → (b : Nat) → Eq.{1} Nat (Nat.gcdIter f 0 b) b",
		"fun (f : Nat) => Nat.rec.{0} (fun (f : Nat) => (b : Nat) → Eq.{1} Nat (Nat.gcdIter f 0 b) b) " +
			"(fun (b : Nat) => Eq.refl.{1} Nat b) " +
			"(fun (k : Nat) (_ : (b : Nat) → Eq.{1} Nat (Nat.gcdIter k 0 b) b) (b : Nat) => Eq.refl.{1} Nat b) f"},
	valDecl{"thm", "Nat.gcd_zero_left", nil, "(b : Nat) → Eq.{1} Nat (Nat.gcd 0 b) b",
		"fun (b : Nat) => Eq.refl.{1} Nat b"},
	valDecl{"thm", "Nat.gcd_succ", nil,
		"(a : Nat) → (b : Nat) → Eq.{1} Nat (Nat.gcd (Nat.succ a) b) (Nat.gcdIter a (Nat.mod b (Nat.succ a)) (Nat.succ a))",
		"fun (a : Nat) (b : Nat) => Eq.refl.{1} Nat (Nat.gcdIter a (Nat.mod b (Nat.succ a)) (Nat.succ a))"},
	valDecl{"thm", "Nat.gcd_one_left", nil, "(b : Nat) → Eq.{1} Nat (Nat.gcd 1 b) 1",
		"fun (b : Nat) => Eq.refl.{1} Nat 1"},
	valDecl{"thm", "Nat.gcd_zero_right", nil, "(n : Nat) → Eq.{1} Nat (Nat.gcd n 0) n",
		"fun (n : Nat) => Nat.rec.{0} (fun (n : Nat) => Eq.{1} Nat (Nat.gcd n 0) n) (Nat.gcd_zero_left 0) " +
			"(fun (a : Nat) (_ : Eq.{1} Nat (Nat.gcd a 0) a) => Nat.gcdIter_zero a (Nat.succ a)) n"},

	// What `Nat.gcd_dvd_left`/`right` rest on, under Lean's names and statements (`Init/Data/Nat/Basic.lean`,
	// `Dvd.lean`, `Mod.lean`, `Gcd.lean`): transitivity of the order, the `*` lemmas their induction needs, the
	// remainder bound, and the two divisibility laws that turn `b = a * (b / a) + b % a` into a divisor of `b`. Lean
	// writes these with its equation compiler; here each is an induction on `Nat.rec`/`Nat.le.rec`.
	valDecl{"thm", "Nat.le_trans", nil, "(a : Nat) → (b : Nat) → (c : Nat) → Nat.le a b → Nat.le b c → Nat.le a c",
		"fun (a : Nat) (b : Nat) (c : Nat) (h1 : Nat.le a b) (h2 : Nat.le b c) => " +
			"Nat.le.rec b (fun (z : Nat) (_ : Nat.le b z) => Nat.le a z) h1 " +
			"(fun (m : Nat) (_ : Nat.le b m) (ih : Nat.le a m) => Nat.le.step a m ih) c h2"},
	valDecl{"thm", "Nat.mul_zero", nil, "(n : Nat) → Eq.{1} Nat (Nat.mul n 0) 0",
		"fun (n : Nat) => Eq.refl.{1} Nat 0"},
	valDecl{"thm", "Nat.mul_one", nil, "(n : Nat) → Eq.{1} Nat (Nat.mul n 1) n",
		"fun (n : Nat) => Nat.zero_add n"},
	valDecl{"thm", "Nat.add_assoc", nil,
		"(n : Nat) → (m : Nat) → (k : Nat) → Eq.{1} Nat (Nat.add (Nat.add n m) k) (Nat.add n (Nat.add m k))",
		"fun (n : Nat) (m : Nat) => Nat.rec.{0} " +
			"(fun (k : Nat) => Eq.{1} Nat (Nat.add (Nat.add n m) k) (Nat.add n (Nat.add m k))) " +
			"(Eq.refl.{1} Nat (Nat.add n m)) " +
			"(fun (k : Nat) (ih : Eq.{1} Nat (Nat.add (Nat.add n m) k) (Nat.add n (Nat.add m k))) => " +
			"congrArg.{1, 1} Nat Nat Nat.succ (Nat.add (Nat.add n m) k) (Nat.add n (Nat.add m k)) ih)"},
	valDecl{"thm", "Nat.mul_add", nil,
		"(n : Nat) → (m : Nat) → (k : Nat) → Eq.{1} Nat (Nat.mul n (Nat.add m k)) (Nat.add (Nat.mul n m) (Nat.mul n k))",
		"fun (n : Nat) (m : Nat) => Nat.rec.{0} " +
			"(fun (k : Nat) => Eq.{1} Nat (Nat.mul n (Nat.add m k)) (Nat.add (Nat.mul n m) (Nat.mul n k))) " +
			"(Eq.refl.{1} Nat (Nat.mul n m)) " +
			"(fun (k : Nat) (ih : Eq.{1} Nat (Nat.mul n (Nat.add m k)) (Nat.add (Nat.mul n m) (Nat.mul n k))) => " +
			"Eq.trans.{1} Nat (Nat.mul n (Nat.add m (Nat.succ k))) " +
			"(Nat.add (Nat.add (Nat.mul n m) (Nat.mul n k)) n) (Nat.add (Nat.mul n m) (Nat.mul n (Nat.succ k))) " +
			"(congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.add z n) (Nat.mul n (Nat.add m k)) " +
			"(Nat.add (Nat.mul n m) (Nat.mul n k)) ih) " +
			"(Nat.add_assoc (Nat.mul n m) (Nat.mul n k) n))"},
	valDecl{"thm", "Nat.mul_assoc", nil,
		"(n : Nat) → (m : Nat) → (k : Nat) → Eq.{1} Nat (Nat.mul (Nat.mul n m) k) (Nat.mul n (Nat.mul m k))",
		"fun (n : Nat) (m : Nat) => Nat.rec.{0} " +
			"(fun (k : Nat) => Eq.{1} Nat (Nat.mul (Nat.mul n m) k) (Nat.mul n (Nat.mul m k))) " +
			"(Eq.refl.{1} Nat 0) " +
			"(fun (k : Nat) (ih : Eq.{1} Nat (Nat.mul (Nat.mul n m) k) (Nat.mul n (Nat.mul m k))) => " +
			"Eq.trans.{1} Nat (Nat.mul (Nat.mul n m) (Nat.succ k)) " +
			"(Nat.add (Nat.mul n (Nat.mul m k)) (Nat.mul n m)) (Nat.mul n (Nat.mul m (Nat.succ k))) " +
			"(congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.add z (Nat.mul n m)) (Nat.mul (Nat.mul n m) k) " +
			"(Nat.mul n (Nat.mul m k)) ih) " +
			"(Eq.symm.{1} Nat (Nat.mul n (Nat.mul m (Nat.succ k))) " +
			"(Nat.add (Nat.mul n (Nat.mul m k)) (Nat.mul n m)) (Nat.mul_add n (Nat.mul m k) m)))"},

	// The machine's remainder is never `k` or more once `k` is not 0: a step either adds one to a remainder below `k`,
	// which `Nat.lt_of_le_ne` turns into `r + 1 < k` because that `Nat.beq` came out false, or wraps it to 0, which is
	// below `k` because `k` is a successor. `Nat.divIter_rem_lt` carries that over every number of steps, and
	// `Nat.mod_lt` reads it off `Nat.mod`, which is the machine's second component.
	valDecl{"thm", "Nat.divStep_rem_lt", nil,
		"(k : Nat) → Nat.lt 0 k → (s : Nat.DivMod) → Nat.lt (" + dmSnd + " s) k → " +
			"Nat.lt (" + dmSnd + " (Nat.divStep k s)) k",
		"fun (k : Nat) (hk : Nat.lt 0 k) (s : Nat.DivMod) => Nat.DivMod.rec.{0} " +
			"(fun (s : Nat.DivMod) => Nat.lt (" + dmSnd + " s) k → Nat.lt (" + dmSnd + " (Nat.divStep k s)) k) " +
			"(fun (q : Nat) (r : Nat) (hr : Nat.lt r k) => Bool.rec.{0} " +
			"(fun (v : Bool) => Eq.{1} Bool (Nat.beq (Nat.succ r) k) v → Nat.lt (" + dmSnd +
			" (Bool.rec.{1} (fun (_ : Bool) => Nat.DivMod) (Nat.DivMod.mk q (Nat.succ r)) " +
			"(Nat.DivMod.mk (Nat.succ q) 0) v)) k) " +
			"(fun (hv : Eq.{1} Bool (Nat.beq (Nat.succ r) k) Bool.false) => " +
			"Nat.lt_of_le_ne (Nat.succ r) k hr (Nat.ne_of_beq_eq_false (Nat.succ r) k hv)) " +
			"(fun (_ : Eq.{1} Bool (Nat.beq (Nat.succ r) k) Bool.true) => hk) " +
			"(Nat.beq (Nat.succ r) k) (Eq.refl.{1} Bool (Nat.beq (Nat.succ r) k))) s"},
	valDecl{"thm", "Nat.divIter_rem_lt", nil,
		"(k : Nat) → Nat.lt 0 k → (j : Nat) → (s : Nat.DivMod) → Nat.lt (" + dmSnd + " s) k → " +
			"Nat.lt (" + dmSnd + " (Nat.divIter k j s)) k",
		"fun (k : Nat) (hk : Nat.lt 0 k) => Nat.rec.{0} " +
			"(fun (j : Nat) => (s : Nat.DivMod) → Nat.lt (" + dmSnd + " s) k → " +
			"Nat.lt (" + dmSnd + " (Nat.divIter k j s)) k) " +
			"(fun (s : Nat.DivMod) (hs : Nat.lt (" + dmSnd + " s) k) => hs) " +
			"(fun (j : Nat) (ih : (s : Nat.DivMod) → Nat.lt (" + dmSnd + " s) k → " +
			"Nat.lt (" + dmSnd + " (Nat.divIter k j s)) k) (s : Nat.DivMod) (hs : Nat.lt (" + dmSnd + " s) k) => " +
			"Nat.divStep_rem_lt k hk (Nat.divIter k j s) (ih s hs))"},
	valDecl{"thm", "Nat.mod_lt", nil, "(n : Nat) → (k : Nat) → Nat.lt 0 k → Nat.lt (Nat.mod n k) k",
		"fun (n : Nat) (k : Nat) (hk : Nat.lt 0 k) => Nat.divIter_rem_lt k hk n (Nat.DivMod.mk 0 0) hk"},

	valDecl{"thm", "Nat.dvd_mul_right_of_dvd", nil,
		"(a : Nat) → (b : Nat) → (c : Nat) → Nat.dvd a b → Nat.dvd a (Nat.mul b c)",
		"fun (a : Nat) (b : Nat) (c : Nat) (h : Nat.dvd a b) => Exists.rec.{1} Nat " +
			"(fun (d : Nat) => Eq.{1} Nat b (Nat.mul a d)) (fun (_ : Nat.dvd a b) => Nat.dvd a (Nat.mul b c)) " +
			"(fun (d : Nat) (e : Eq.{1} Nat b (Nat.mul a d)) => Exists.intro.{1} Nat " +
			"(fun (w : Nat) => Eq.{1} Nat (Nat.mul b c) (Nat.mul a w)) (Nat.mul d c) " +
			"(Eq.trans.{1} Nat (Nat.mul b c) (Nat.mul (Nat.mul a d) c) (Nat.mul a (Nat.mul d c)) " +
			"(congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.mul z c) b (Nat.mul a d) e) " +
			"(Nat.mul_assoc a d c))) h"},
	valDecl{"thm", "Nat.dvd_add", nil,
		"(a : Nat) → (b : Nat) → (c : Nat) → Nat.dvd a b → Nat.dvd a c → Nat.dvd a (Nat.add b c)",
		"fun (a : Nat) (b : Nat) (c : Nat) (h1 : Nat.dvd a b) (h2 : Nat.dvd a c) => Exists.rec.{1} Nat " +
			"(fun (d : Nat) => Eq.{1} Nat b (Nat.mul a d)) (fun (_ : Nat.dvd a b) => Nat.dvd a (Nat.add b c)) " +
			"(fun (d : Nat) (e1 : Eq.{1} Nat b (Nat.mul a d)) => Exists.rec.{1} Nat " +
			"(fun (e : Nat) => Eq.{1} Nat c (Nat.mul a e)) (fun (_ : Nat.dvd a c) => Nat.dvd a (Nat.add b c)) " +
			"(fun (e : Nat) (e2 : Eq.{1} Nat c (Nat.mul a e)) => Exists.intro.{1} Nat " +
			"(fun (w : Nat) => Eq.{1} Nat (Nat.add b c) (Nat.mul a w)) (Nat.add d e) " +
			"(Eq.trans.{1} Nat (Nat.add b c) (Nat.add (Nat.mul a d) (Nat.mul a e)) (Nat.mul a (Nat.add d e)) " +
			"(Nat.add_congr b c (Nat.mul a d) (Nat.mul a e) e1 e2) " +
			"(Eq.symm.{1} Nat (Nat.mul a (Nat.add d e)) (Nat.add (Nat.mul a d) (Nat.mul a e)) " +
			"(Nat.mul_add a d e)))) h2) h1"},

	// `gcd m n` divides both arguments. The countdown keeps `a ≤ f`, so it only stops on `a = 0`, where the value is `b`
	// and divides `0` and `b`; backwards, if the value divides `b % a` and `a` it divides `b`, which is
	// `a * (b / a) + b % a`. That is the induction Lean's `Nat.gcd_dvd` runs on its well-founded recursion.
	valDecl{"thm", "Nat.gcdIter_dvd", nil,
		"(f : Nat) → (a : Nat) → (b : Nat) → Nat.le a f → " +
			"And (Nat.dvd (Nat.gcdIter f a b) a) (Nat.dvd (Nat.gcdIter f a b) b)",
		"fun (f : Nat) => Nat.rec.{0} " +
			"(fun (f : Nat) => (a : Nat) → (b : Nat) → Nat.le a f → " +
			"And (Nat.dvd (Nat.gcdIter f a b) a) (Nat.dvd (Nat.gcdIter f a b) b)) " +
			// fuel 0: `Nat.le a 0` leaves `a = 0`, so the value `b` divides `0` and itself
			"(fun (a : Nat) (b : Nat) (h : Nat.le a 0) => And.intro (Nat.dvd b a) (Nat.dvd b b) " +
			"(Eq.rec.{0, 1} Nat 0 (fun (z : Nat) (_ : Eq.{1} Nat 0 z) => Nat.dvd b z) " +
			"(Exists.intro.{1} Nat (fun (w : Nat) => Eq.{1} Nat 0 (Nat.mul b w)) 0 " +
			"(Eq.symm.{1} Nat (Nat.mul b 0) 0 (Nat.mul_zero b))) a " +
			"(Eq.symm.{1} Nat a 0 (Nat.le_zero a h))) " +
			"(Exists.intro.{1} Nat (fun (w : Nat) => Eq.{1} Nat b (Nat.mul b w)) 1 " +
			"(Eq.symm.{1} Nat (Nat.mul b 1) b (Nat.mul_one b)))) " +
			"(fun (f' : Nat) (ih : (a : Nat) → (b : Nat) → Nat.le a f' → " +
			"And (Nat.dvd (Nat.gcdIter f' a b) a) (Nat.dvd (Nat.gcdIter f' a b) b)) (a : Nat) => Nat.rec.{0} " +
			"(fun (a : Nat) => (b : Nat) → Nat.le a (Nat.succ f') → " +
			"And (Nat.dvd (Nat.gcdIter (Nat.succ f') a b) a) (Nat.dvd (Nat.gcdIter (Nat.succ f') a b) b)) " +
			// `a = 0`: the step returns `b`
			"(fun (b : Nat) (_ : Nat.le 0 (Nat.succ f')) => And.intro (Nat.dvd b 0) (Nat.dvd b b) " +
			"(Exists.intro.{1} Nat (fun (w : Nat) => Eq.{1} Nat 0 (Nat.mul b w)) 0 " +
			"(Eq.symm.{1} Nat (Nat.mul b 0) 0 (Nat.mul_zero b))) " +
			"(Exists.intro.{1} Nat (fun (w : Nat) => Eq.{1} Nat b (Nat.mul b w)) 1 " +
			"(Eq.symm.{1} Nat (Nat.mul b 1) b (Nat.mul_one b)))) " +
			// `a = succ a'`: one step on, the value is `gcd d`, which the induction hypothesis covers because
			// `b % succ a' < succ a' ≤ succ f'`
			"(fun (a' : Nat) (_ : (b : Nat) → Nat.le a' (Nat.succ f') → " +
			"And (Nat.dvd (Nat.gcdIter (Nat.succ f') a' b) a') " +
			"(Nat.dvd (Nat.gcdIter (Nat.succ f') a' b) b)) (b : Nat) (h : Nat.le (Nat.succ a') (Nat.succ f')) => " +
			"And.intro (Nat.dvd " + gcdD + " " + gcdB + ") (Nat.dvd " + gcdD + " b) " + gcdIhB + " " + gcdDvdB + ")" +
			" a) f"},
	valDecl{"thm", "Nat.gcd_dvd_left", nil, "(m : Nat) → (n : Nat) → Nat.dvd (Nat.gcd m n) m",
		"fun (m : Nat) (n : Nat) => And.left (Nat.dvd (Nat.gcdIter m m n) m) (Nat.dvd (Nat.gcdIter m m n) n) " +
			"(Nat.gcdIter_dvd m m n (Nat.le.refl m))"},
	valDecl{"thm", "Nat.gcd_dvd_right", nil, "(m : Nat) → (n : Nat) → Nat.dvd (Nat.gcd m n) n",
		"fun (m : Nat) (n : Nat) => And.right (Nat.dvd (Nat.gcdIter m m n) m) (Nat.dvd (Nat.gcdIter m m n) n) " +
			"(Nat.gcdIter_dvd m m n (Nat.le.refl m))"},
	valDecl{"thm", "Nat.gcd_dvd", nil,
		"(m : Nat) → (n : Nat) → And (Nat.dvd (Nat.gcd m n) m) (Nat.dvd (Nat.gcd m n) n)",
		"fun (m : Nat) (n : Nat) => Nat.gcdIter_dvd m m n (Nat.le.refl m)"},

	// The arithmetic `Nat.dvd_gcd` and `Nat.gcd_le_left` need, under Lean's names and statements
	// (`Init/Data/Nat/Basic.lean`, `Dvd.lean`): order and `*` first, then subtraction (`-` recurses on the subtrahend,
	// so each of these is an induction on the argument the definition consumes), then `dvd_sub`, which is where
	// `(m - n)` stops being a black box.
	valDecl{"thm", "Nat.succ_add", nil, "(n : Nat) → (m : Nat) → Eq.{1} Nat (Nat.add (Nat.succ n) m) (Nat.succ (Nat.add n m))",
		"fun (n : Nat) => Nat.rec.{0} (fun (m : Nat) => Eq.{1} Nat (Nat.add (Nat.succ n) m) (Nat.succ (Nat.add n m))) " +
			"(Eq.refl.{1} Nat (Nat.succ n)) " +
			"(fun (m : Nat) (ih : Eq.{1} Nat (Nat.add (Nat.succ n) m) (Nat.succ (Nat.add n m))) => " +
			"congrArg.{1, 1} Nat Nat Nat.succ (Nat.add (Nat.succ n) m) (Nat.succ (Nat.add n m)) ih)"},
	valDecl{"thm", "Nat.add_comm", nil, "(n : Nat) → (m : Nat) → Eq.{1} Nat (Nat.add n m) (Nat.add m n)",
		"fun (n : Nat) => Nat.rec.{0} (fun (n : Nat) => (m : Nat) → Eq.{1} Nat (Nat.add n m) (Nat.add m n)) " +
			"(fun (m : Nat) => Eq.trans.{1} Nat (Nat.add 0 m) m (Nat.add m 0) (Nat.zero_add m) " +
			"(Eq.refl.{1} Nat (Nat.add m 0))) " +
			"(fun (n' : Nat) (ih : (m : Nat) → Eq.{1} Nat (Nat.add n' m) (Nat.add m n')) (m : Nat) => " +
			"Eq.trans.{1} Nat (Nat.add (Nat.succ n') m) (Nat.succ (Nat.add n' m)) (Nat.add m (Nat.succ n')) " +
			"(Nat.succ_add n' m) " +
			"(congrArg.{1, 1} Nat Nat Nat.succ (Nat.add n' m) (Nat.add m n') (ih m))) n"},
	valDecl{"thm", "Nat.add_right_comm", nil,
		"(a : Nat) → (b : Nat) → (c : Nat) → Eq.{1} Nat (Nat.add (Nat.add a b) c) (Nat.add (Nat.add a c) b)",
		"fun (a : Nat) (b : Nat) (c : Nat) => Eq.trans.{1} Nat (Nat.add (Nat.add a b) c) (Nat.add a (Nat.add b c)) " +
			"(Nat.add (Nat.add a c) b) (Nat.add_assoc a b c) " +
			"(Eq.trans.{1} Nat (Nat.add a (Nat.add b c)) (Nat.add a (Nat.add c b)) (Nat.add (Nat.add a c) b) " +
			"(congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.add a z) (Nat.add b c) (Nat.add c b) (Nat.add_comm b c)) " +
			"(Eq.symm.{1} Nat (Nat.add (Nat.add a c) b) (Nat.add a (Nat.add c b)) (Nat.add_assoc a c b)))"},
	valDecl{"thm", "Nat.zero_le", nil, "(n : Nat) → Nat.le 0 n",
		"fun (n : Nat) => Nat.rec.{0} (fun (k : Nat) => Nat.le 0 k) (Nat.le.refl 0) " +
			"(fun (k : Nat) (ih : Nat.le 0 k) => Nat.le.step 0 k ih) n"},
	valDecl{"thm", "Nat.le_add_left", nil, "(n : Nat) → (m : Nat) → Nat.le n (Nat.add m n)",
		"fun (n : Nat) => Nat.rec.{0} (fun (m : Nat) => Nat.le n (Nat.add m n)) " +
			"(Eq.rec.{0, 1} Nat n (fun (z : Nat) (_ : Eq.{1} Nat n z) => Nat.le n z) " +
			"(Nat.le.refl n) (Nat.add 0 n) (Eq.symm.{1} Nat (Nat.add 0 n) n (Nat.zero_add n))) " +
			"(fun (m : Nat) (ih : Nat.le n (Nat.add m n)) => Eq.rec.{0, 1} Nat (Nat.succ (Nat.add m n)) " +
			"(fun (z : Nat) (_ : Eq.{1} Nat (Nat.succ (Nat.add m n)) z) => Nat.le n z) " +
			"(Nat.le.step n (Nat.add m n) ih) (Nat.add (Nat.succ m) n) " +
			"(Eq.symm.{1} Nat (Nat.add (Nat.succ m) n) (Nat.succ (Nat.add m n)) (Nat.succ_add m n)))"},
	valDecl{"thm", "Nat.sub_congr", nil,
		"(a : Nat) → (b : Nat) → (c : Nat) → (d : Nat) → Eq.{1} Nat a c → Eq.{1} Nat b d → " +
			"Eq.{1} Nat (Nat.sub a b) (Nat.sub c d)",
		"fun (a : Nat) (b : Nat) (c : Nat) (d : Nat) (h1 : Eq.{1} Nat a c) (h2 : Eq.{1} Nat b d) => " +
			"Eq.rec.{0, 1} Nat a (fun (z : Nat) (_ : Eq.{1} Nat a z) => Eq.{1} Nat (Nat.sub a b) (Nat.sub z d)) " +
			"(Eq.rec.{0, 1} Nat b (fun (w : Nat) (_ : Eq.{1} Nat b w) => Eq.{1} Nat (Nat.sub a b) (Nat.sub a w)) " +
			"(Eq.refl.{1} Nat (Nat.sub a b)) d h2) c h1"},
	valDecl{"thm", "Nat.succ_sub_succ", nil,
		"(a : Nat) → (b : Nat) → Eq.{1} Nat (Nat.sub (Nat.succ a) (Nat.succ b)) (Nat.sub a b)",
		"fun (a : Nat) => Nat.rec.{0} (fun (b : Nat) => Eq.{1} Nat (Nat.sub (Nat.succ a) (Nat.succ b)) (Nat.sub a b)) " +
			"(Eq.refl.{1} Nat a) " +
			"(fun (k : Nat) (ih : Eq.{1} Nat (Nat.sub (Nat.succ a) (Nat.succ k)) (Nat.sub a k)) => " +
			"congrArg.{1, 1} Nat Nat Nat.pred (Nat.sub (Nat.succ a) (Nat.succ k)) (Nat.sub a k) ih)"},
	valDecl{"thm", "Nat.sub_self", nil, "(n : Nat) → Eq.{1} Nat (Nat.sub n n) 0",
		"fun (n : Nat) => Nat.rec.{0} (fun (k : Nat) => Eq.{1} Nat (Nat.sub k k) 0) (Eq.refl.{1} Nat 0) " +
			"(fun (k : Nat) (ih : Eq.{1} Nat (Nat.sub k k) 0) => Eq.trans.{1} Nat (Nat.sub (Nat.succ k) (Nat.succ k)) " +
			"(Nat.sub k k) 0 (Nat.succ_sub_succ k k) ih) n"},
	valDecl{"thm", "Nat.zero_sub", nil, "(n : Nat) → Eq.{1} Nat (Nat.sub 0 n) 0",
		"fun (n : Nat) => Nat.rec.{0} (fun (k : Nat) => Eq.{1} Nat (Nat.sub 0 k) 0) (Eq.refl.{1} Nat 0) " +
			"(fun (k : Nat) (ih : Eq.{1} Nat (Nat.sub 0 k) 0) => " +
			"congrArg.{1, 1} Nat Nat Nat.pred (Nat.sub 0 k) 0 ih) n"},
	valDecl{"thm", "Nat.add_sub_add_right", nil,
		"(n : Nat) → (m : Nat) → (k : Nat) → Eq.{1} Nat (Nat.sub (Nat.add n k) (Nat.add m k)) (Nat.sub n m)",
		"fun (n : Nat) (m : Nat) => Nat.rec.{0} " +
			"(fun (k : Nat) => Eq.{1} Nat (Nat.sub (Nat.add n k) (Nat.add m k)) (Nat.sub n m)) " +
			"(Eq.refl.{1} Nat (Nat.sub n m)) " +
			"(fun (k : Nat) (ih : Eq.{1} Nat (Nat.sub (Nat.add n k) (Nat.add m k)) (Nat.sub n m)) => " +
			"Eq.trans.{1} Nat (Nat.sub (Nat.add n (Nat.succ k)) (Nat.add m (Nat.succ k))) " +
			"(Nat.sub (Nat.add n k) (Nat.add m k)) (Nat.sub n m) " +
			"(Nat.succ_sub_succ (Nat.add n k) (Nat.add m k)) ih)"},
	valDecl{"thm", "Nat.add_sub_add_left", nil,
		"(k : Nat) → (n : Nat) → (m : Nat) → Eq.{1} Nat (Nat.sub (Nat.add k n) (Nat.add k m)) (Nat.sub n m)",
		"fun (k : Nat) (n : Nat) (m : Nat) => Eq.trans.{1} Nat (Nat.sub (Nat.add k n) (Nat.add k m)) " +
			"(Nat.sub (Nat.add n k) (Nat.add m k)) (Nat.sub n m) " +
			"(Nat.sub_congr (Nat.add k n) (Nat.add k m) (Nat.add n k) (Nat.add m k) (Nat.add_comm k n) " +
			"(Nat.add_comm k m)) (Nat.add_sub_add_right n m k)"},
	valDecl{"thm", "Nat.add_sub_cancel", nil, "(n : Nat) → (m : Nat) → Eq.{1} Nat (Nat.sub (Nat.add n m) m) n",
		"fun (n : Nat) (m : Nat) => Eq.trans.{1} Nat (Nat.sub (Nat.add n m) m) " +
			"(Nat.sub (Nat.add n m) (Nat.add 0 m)) n " +
			"(Eq.symm.{1} Nat (Nat.sub (Nat.add n m) (Nat.add 0 m)) (Nat.sub (Nat.add n m) m) " +
			"(Nat.sub_congr (Nat.add n m) (Nat.add 0 m) (Nat.add n m) m (Eq.refl.{1} Nat (Nat.add n m)) " +
			"(Nat.zero_add m))) (Nat.add_sub_add_right n 0 m)"},
	valDecl{"thm", "Nat.add_sub_cancel_left", nil, "(n : Nat) → (m : Nat) → Eq.{1} Nat (Nat.sub (Nat.add n m) n) m",
		"fun (n : Nat) (m : Nat) => Nat.add_sub_add_left n m 0"},
	valDecl{"thm", "Nat.sub_sub", nil,
		"(n : Nat) → (m : Nat) → (k : Nat) → Eq.{1} Nat (Nat.sub (Nat.sub n m) k) (Nat.sub n (Nat.add m k))",
		"fun (n : Nat) (m : Nat) => Nat.rec.{0} " +
			"(fun (k : Nat) => Eq.{1} Nat (Nat.sub (Nat.sub n m) k) (Nat.sub n (Nat.add m k))) " +
			"(Eq.refl.{1} Nat (Nat.sub n m)) " +
			"(fun (k : Nat) (ih : Eq.{1} Nat (Nat.sub (Nat.sub n m) k) (Nat.sub n (Nat.add m k))) => " +
			"congrArg.{1, 1} Nat Nat Nat.pred (Nat.sub (Nat.sub n m) k) (Nat.sub n (Nat.add m k)) ih)"},
	// `Nat.mul` recurses on its second argument here, so `(succ n) * m = n * m + m` and `n * pred m = n * m - n` are
	// inductions too, the first needing `add_right_comm`.
	valDecl{"thm", "Nat.succ_mul", nil, "(n : Nat) → (m : Nat) → Eq.{1} Nat (Nat.mul (Nat.succ n) m) (Nat.add (Nat.mul n m) m)",
		"fun (n : Nat) => Nat.rec.{0} (fun (m : Nat) => Eq.{1} Nat (Nat.mul (Nat.succ n) m) (Nat.add (Nat.mul n m) m)) " +
			"(Eq.refl.{1} Nat 0) " +
			"(fun (k : Nat) (ih : Eq.{1} Nat (Nat.mul (Nat.succ n) k) (Nat.add (Nat.mul n k) k)) => " +
			"Eq.trans.{1} Nat (Nat.mul (Nat.succ n) (Nat.succ k)) (Nat.add (Nat.add (Nat.mul n k) k) (Nat.succ n)) " +
			"(Nat.add (Nat.add (Nat.mul n k) n) (Nat.succ k)) " +
			"(congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.add z (Nat.succ n)) (Nat.mul (Nat.succ n) k) " +
			"(Nat.add (Nat.mul n k) k) ih) " +
			"(congrArg.{1, 1} Nat Nat Nat.succ (Nat.add (Nat.add (Nat.mul n k) k) n) " +
			"(Nat.add (Nat.add (Nat.mul n k) n) k) (Nat.add_right_comm (Nat.mul n k) k n)))"},
	valDecl{"thm", "Nat.mul_pred", nil, "(n : Nat) → (m : Nat) → Eq.{1} Nat (Nat.mul n (Nat.pred m)) (Nat.sub (Nat.mul n m) n)",
		"fun (n : Nat) => Nat.rec.{0} " +
			"(fun (m : Nat) => Eq.{1} Nat (Nat.mul n (Nat.pred m)) (Nat.sub (Nat.mul n m) n)) " +
			"(Eq.symm.{1} Nat (Nat.sub 0 n) 0 (Nat.zero_sub n)) " +
			"(fun (k : Nat) (_ : Eq.{1} Nat (Nat.mul n (Nat.pred k)) (Nat.sub (Nat.mul n k) n)) => " +
			"Eq.symm.{1} Nat (Nat.sub (Nat.add (Nat.mul n k) n) n) (Nat.mul n k) (Nat.add_sub_cancel (Nat.mul n k) n))"},
	// `n * (m - k) = n * m - n * k`, unconditionally (Nat subtraction truncates, so the `k > m` case is `0 = 0`).
	valDecl{"thm", "Nat.mul_sub_left_distrib", nil,
		"(n : Nat) → (m : Nat) → (k : Nat) → Eq.{1} Nat (Nat.mul n (Nat.sub m k)) (Nat.sub (Nat.mul n m) (Nat.mul n k))",
		"fun (n : Nat) (m : Nat) => Nat.rec.{0} " +
			"(fun (k : Nat) => Eq.{1} Nat (Nat.mul n (Nat.sub m k)) (Nat.sub (Nat.mul n m) (Nat.mul n k))) " +
			"(Eq.refl.{1} Nat (Nat.mul n m)) " +
			"(fun (k : Nat) (ih : Eq.{1} Nat (Nat.mul n (Nat.sub m k)) (Nat.sub (Nat.mul n m) (Nat.mul n k))) => " +
			"Eq.trans.{1} Nat (Nat.mul n (Nat.sub m (Nat.succ k))) (Nat.sub (Nat.mul n (Nat.sub m k)) n) " +
			"(Nat.sub (Nat.mul n m) (Nat.mul n (Nat.succ k))) " +
			"(Nat.mul_pred n (Nat.sub m k)) " +
			"(Eq.trans.{1} Nat (Nat.sub (Nat.mul n (Nat.sub m k)) n) " +
			"(Nat.sub (Nat.sub (Nat.mul n m) (Nat.mul n k)) n) " +
			"(Nat.sub (Nat.mul n m) (Nat.mul n (Nat.succ k))) " +
			"(congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.sub z n) (Nat.mul n (Nat.sub m k)) " +
			"(Nat.sub (Nat.mul n m) (Nat.mul n k)) ih) " +
			"(Nat.sub_sub (Nat.mul n m) (Nat.mul n k) n)))"},
	valDecl{"thm", "Nat.dvd_sub", nil,
		"(k : Nat) → (m : Nat) → (n : Nat) → Nat.dvd k m → Nat.dvd k n → Nat.dvd k (Nat.sub m n)",
		"fun (k : Nat) (m : Nat) (n : Nat) (h1 : Nat.dvd k m) (h2 : Nat.dvd k n) => Exists.rec.{1} Nat " +
			"(fun (c : Nat) => Eq.{1} Nat m (Nat.mul k c)) (fun (_ : Nat.dvd k m) => Nat.dvd k (Nat.sub m n)) " +
			"(fun (c : Nat) (e1 : Eq.{1} Nat m (Nat.mul k c)) => Exists.rec.{1} Nat " +
			"(fun (d : Nat) => Eq.{1} Nat n (Nat.mul k d)) (fun (_ : Nat.dvd k n) => Nat.dvd k (Nat.sub m n)) " +
			"(fun (d : Nat) (e2 : Eq.{1} Nat n (Nat.mul k d)) => Exists.intro.{1} Nat " +
			"(fun (w : Nat) => Eq.{1} Nat (Nat.sub m n) (Nat.mul k w)) (Nat.sub c d) " +
			"(Eq.rec.{0, 1} Nat (Nat.mul k c) " +
			"(fun (z : Nat) (_ : Eq.{1} Nat (Nat.mul k c) z) => Eq.{1} Nat (Nat.sub z n) (Nat.mul k (Nat.sub c d))) " +
			"(Eq.rec.{0, 1} Nat (Nat.mul k d) " +
			"(fun (w : Nat) (_ : Eq.{1} Nat (Nat.mul k d) w) => " +
			"Eq.{1} Nat (Nat.sub (Nat.mul k c) w) (Nat.mul k (Nat.sub c d))) " +
			"(Eq.symm.{1} Nat (Nat.mul k (Nat.sub c d)) (Nat.sub (Nat.mul k c) (Nat.mul k d)) " +
			"(Nat.mul_sub_left_distrib k c d)) n (Eq.symm.{1} Nat n (Nat.mul k d) e2)) m " +
			"(Eq.symm.{1} Nat m (Nat.mul k c) e1))) h2) h1"},

	// `Nat.gcd` divides a common divisor: the countdown keeps both arguments multiples of `k`, and one step replaces
	// `(a, b)` by `(b % a, a)`, where `b % a` is a multiple because `b = a * (b / a) + b % a` and `b - a * (b / a) = b % a`.
	valDecl{"thm", "Nat.dvd_mod_of_dvd", nil,
		"(k : Nat) → (b : Nat) → (a : Nat) → Nat.dvd k a → Nat.dvd k b → Nat.dvd k (Nat.mod b a)",
		"fun (k : Nat) (b : Nat) (a : Nat) (ha : Nat.dvd k a) (hb : Nat.dvd k b) => " +
			"Eq.rec.{0, 1} Nat (Nat.sub b (Nat.mul a (Nat.div b a))) " +
			"(fun (z : Nat) (_ : Eq.{1} Nat (Nat.sub b (Nat.mul a (Nat.div b a))) z) => Nat.dvd k z) " +
			"(Nat.dvd_sub k b (Nat.mul a (Nat.div b a)) hb (Nat.dvd_mul_right_of_dvd k a (Nat.div b a) ha)) " +
			"(Nat.mod b a) " + modSub},
	valDecl{"thm", "Nat.gcdIter_dvd_of_dvd", nil,
		"(f : Nat) → (a : Nat) → (b : Nat) → (k : Nat) → Nat.dvd k a → Nat.dvd k b → " +
			"Nat.dvd k (Nat.gcdIter f a b)",
		"fun (f : Nat) => Nat.rec.{0} " +
			"(fun (f : Nat) => (a : Nat) → (b : Nat) → (k : Nat) → Nat.dvd k a → Nat.dvd k b → " +
			"Nat.dvd k (Nat.gcdIter f a b)) " +
			// fuel 0 leaves `b`
			"(fun (a : Nat) (b : Nat) (k : Nat) (_ : Nat.dvd k a) (hb : Nat.dvd k b) => hb) " +
			"(fun (f' : Nat) (ih : (a : Nat) → (b : Nat) → (k : Nat) → Nat.dvd k a → Nat.dvd k b → " +
			"Nat.dvd k (Nat.gcdIter f' a b)) (a : Nat) => Nat.rec.{0} " +
			"(fun (a : Nat) => (b : Nat) → (k : Nat) → Nat.dvd k a → Nat.dvd k b → " +
			"Nat.dvd k (Nat.gcdIter (Nat.succ f') a b)) " +
			// `a = 0`: the step returns `b`
			"(fun (b : Nat) (k : Nat) (_ : Nat.dvd k 0) (hb : Nat.dvd k b) => hb) " +
			// `a = succ a'`: the step asks for `b % succ a'` and `succ a'` to be multiples of `k`
			"(fun (a' : Nat) (_ : (b : Nat) → (k : Nat) → Nat.dvd k a' → Nat.dvd k b → " +
			"Nat.dvd k (Nat.gcdIter (Nat.succ f') a' b)) (b : Nat) (k : Nat) (ha : Nat.dvd k (Nat.succ a')) " +
			"(hb : Nat.dvd k b) => ih (Nat.mod b (Nat.succ a')) (Nat.succ a') k " +
			"(Nat.dvd_mod_of_dvd k b (Nat.succ a') ha hb) ha) a) f"},
	valDecl{"thm", "Nat.dvd_gcd", nil,
		"(k : Nat) → (m : Nat) → (n : Nat) → Nat.dvd k m → Nat.dvd k n → Nat.dvd k (Nat.gcd m n)",
		"fun (k : Nat) (m : Nat) (n : Nat) (hm : Nat.dvd k m) (hn : Nat.dvd k n) => " +
			"Nat.gcdIter_dvd_of_dvd m m n k hm hn"},
	// `a ∣ b` with `b > 0` bounds `a` by `b`: write `b = a * c`, and `c` cannot be `0`.
	valDecl{"thm", "Nat.le_of_dvd", nil, "(a : Nat) → (b : Nat) → Nat.lt 0 b → Nat.dvd a b → Nat.le a b",
		"fun (a : Nat) (b : Nat) (h : Nat.lt 0 b) (hdiv : Nat.dvd a b) => Exists.rec.{1} Nat " +
			"(fun (c : Nat) => Eq.{1} Nat b (Nat.mul a c)) (fun (_ : Nat.dvd a b) => Nat.le a b) " +
			"(fun (c : Nat) (e : Eq.{1} Nat b (Nat.mul a c)) => Nat.rec.{0} " +
			"(fun (c : Nat) => Eq.{1} Nat b (Nat.mul a c) → Nat.le a b) " +
			// `c = 0` makes `b = 0`, which `0 < b` rules out
			"(fun (e0 : Eq.{1} Nat b (Nat.mul a 0)) => False.elim.{0} (Nat.le a b) " +
			"(Nat.succ_ne_zero 0 (Nat.le_zero 1 (Eq.rec.{0, 1} Nat b " +
			"(fun (z : Nat) (_ : Eq.{1} Nat b z) => Nat.le 1 z) h 0 " +
			"(Eq.trans.{1} Nat b (Nat.mul a 0) 0 e0 (Nat.mul_zero a)))))) " +
			// `c = succ c'` puts `a` under `b = a * c' + a`
			"(fun (c' : Nat) (_ : Eq.{1} Nat b (Nat.mul a c') → Nat.le a b) " +
			"(ec : Eq.{1} Nat b (Nat.mul a (Nat.succ c'))) => Eq.rec.{0, 1} Nat " +
			"(Nat.add (Nat.mul a c') a) " +
			"(fun (z : Nat) (_ : Eq.{1} Nat (Nat.add (Nat.mul a c') a) z) => Nat.le a z) " +
			"(Nat.le_add_left a (Nat.mul a c')) b " +
			"(Eq.symm.{1} Nat b (Nat.add (Nat.mul a c') a) ec)) c e) hdiv"},
	valDecl{"thm", "Nat.gcd_le_left", nil, "(m : Nat) → (n : Nat) → Nat.lt 0 m → Nat.le (Nat.gcd m n) m",
		"fun (m : Nat) (n : Nat) (h : Nat.lt 0 m) => Nat.le_of_dvd (Nat.gcd m n) m h (Nat.gcd_dvd_left m n)"},
	valDecl{"thm", "Nat.gcd_le_right", nil, "(m : Nat) → (n : Nat) → Nat.lt 0 n → Nat.le (Nat.gcd m n) n",
		"fun (m : Nat) (n : Nat) (h : Nat.lt 0 n) => Nat.le_of_dvd (Nat.gcd m n) n h (Nat.gcd_dvd_right m n)"},

	// `gcd_comm` and `gcd_assoc` do not touch the countdown: both sides of each are common divisors of the same
	// arguments, so `dvd_gcd`, `gcd_dvd` and antisymmetry decide it. Antisymmetry is where the order and `*` lemmas here
	// are used: `le_antisymm` from the two `Nat.le`s, `dvd_antisymm` from `le_of_dvd`, `dvd_trans` from `mul_assoc`.
	valDecl{"thm", "Nat.mul_comm", nil, "(n : Nat) → (m : Nat) → Eq.{1} Nat (Nat.mul n m) (Nat.mul m n)",
		"fun (n : Nat) => Nat.rec.{0} (fun (n : Nat) => (m : Nat) → Eq.{1} Nat (Nat.mul n m) (Nat.mul m n)) " +
			"(fun (m : Nat) => Nat.zero_mul m) " +
			"(fun (n' : Nat) (ih : (m : Nat) → Eq.{1} Nat (Nat.mul n' m) (Nat.mul m n')) (m : Nat) => " +
			"Eq.trans.{1} Nat (Nat.mul (Nat.succ n') m) (Nat.add (Nat.mul n' m) m) (Nat.mul m (Nat.succ n')) " +
			"(Nat.succ_mul n' m) " +
			"(congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.add z m) (Nat.mul n' m) (Nat.mul m n') (ih m))) n"},
	valDecl{"thm", "Nat.add_le_add_right", nil,
		"(n : Nat) → (m : Nat) → (k : Nat) → Nat.le n m → Nat.le (Nat.add n k) (Nat.add m k)",
		"fun (n : Nat) (m : Nat) (k : Nat) (h : Nat.le n m) => Nat.le.rec n " +
			"(fun (y : Nat) (_ : Nat.le n y) => Nat.le (Nat.add n k) (Nat.add y k)) " +
			"(Nat.le.refl (Nat.add n k)) " +
			"(fun (m' : Nat) (_ : Nat.le n m') (ih : Nat.le (Nat.add n k) (Nat.add m' k)) => " +
			"Eq.rec.{0, 1} Nat (Nat.succ (Nat.add m' k)) " +
			"(fun (z : Nat) (_ : Eq.{1} Nat (Nat.succ (Nat.add m' k)) z) => Nat.le (Nat.add n k) z) " +
			"(Nat.le.step (Nat.add n k) (Nat.add m' k) ih) (Nat.add (Nat.succ m') k) " +
			"(Eq.symm.{1} Nat (Nat.add (Nat.succ m') k) (Nat.succ (Nat.add m' k)) (Nat.succ_add m' k))) m h"},
	valDecl{"thm", "Nat.add_le_add_left", nil,
		"(n : Nat) → (m : Nat) → (k : Nat) → Nat.le n m → Nat.le (Nat.add k n) (Nat.add k m)",
		"fun (n : Nat) (m : Nat) (k : Nat) (h : Nat.le n m) => Nat.le.rec n " +
			"(fun (y : Nat) (_ : Nat.le n y) => Nat.le (Nat.add k n) (Nat.add k y)) " +
			"(Nat.le.refl (Nat.add k n)) " +
			"(fun (m' : Nat) (_ : Nat.le n m') (ih : Nat.le (Nat.add k n) (Nat.add k m')) => " +
			"Nat.le.step (Nat.add k n) (Nat.add k m') ih) m h"},
	valDecl{"thm", "Nat.add_le_add", nil,
		"(a : Nat) → (b : Nat) → (c : Nat) → (d : Nat) → Nat.le a b → Nat.le c d → " +
			"Nat.le (Nat.add a c) (Nat.add b d)",
		"fun (a : Nat) (b : Nat) (c : Nat) (d : Nat) (h1 : Nat.le a b) (h2 : Nat.le c d) => " +
			"Nat.le_trans (Nat.add a c) (Nat.add b c) (Nat.add b d) (Nat.add_le_add_right a b c h1) " +
			"(Nat.add_le_add_left c d b h2)"},
	valDecl{"thm", "Nat.le_antisymm", nil, "(a : Nat) → (b : Nat) → Nat.le a b → Nat.le b a → Eq.{1} Nat a b",
		"fun (a : Nat) (b : Nat) (h1 : Nat.le a b) (h2 : Nat.le b a) => Nat.le.rec a " +
			"(fun (y : Nat) (_ : Nat.le a y) => Nat.le y a → Eq.{1} Nat a y) " +
			"(fun (_ : Nat.le a a) => Eq.refl.{1} Nat a) " +
			"(fun (m : Nat) (hm : Nat.le a m) (_ : Nat.le m a → Eq.{1} Nat a m) (h : Nat.le (Nat.succ m) a) => " +
			"False.elim.{0} (Eq.{1} Nat a (Nat.succ m)) (Nat.not_le_succ_self m (Nat.le_trans (Nat.succ m) a m h hm))) " +
			"b h1 h2"},
	valDecl{"thm", "Nat.eq_zero_of_zero_dvd", nil, "(n : Nat) → Nat.dvd 0 n → Eq.{1} Nat n 0",
		"fun (n : Nat) (h : Nat.dvd 0 n) => Exists.rec.{1} Nat (fun (c : Nat) => Eq.{1} Nat n (Nat.mul 0 c)) " +
			"(fun (_ : Nat.dvd 0 n) => Eq.{1} Nat n 0) " +
			"(fun (c : Nat) (e : Eq.{1} Nat n (Nat.mul 0 c)) => Eq.trans.{1} Nat n (Nat.mul 0 c) 0 e (Nat.zero_mul c)) h"},
	valDecl{"thm", "Nat.pos_of_dvd_of_pos", nil, "(a : Nat) → (b : Nat) → Nat.dvd a b → Nat.lt 0 b → Nat.lt 0 a",
		"fun (a : Nat) => Nat.rec.{0} (fun (a : Nat) => (b : Nat) → Nat.dvd a b → Nat.lt 0 b → Nat.lt 0 a) " +
			// `0 ∣ b` forces `b = 0`, which `0 < b` rules out
			"(fun (b : Nat) (h0 : Nat.dvd 0 b) (hb : Nat.lt 0 b) => False.elim.{0} (Nat.lt 0 0) " +
			"(Nat.succ_ne_zero 0 (Nat.le_zero 1 (Eq.rec.{0, 1} Nat b (fun (z : Nat) (_ : Eq.{1} Nat b z) => Nat.le 1 z) " +
			"hb 0 (Nat.eq_zero_of_zero_dvd b h0))))) " +
			"(fun (a' : Nat) (_ : (b : Nat) → Nat.dvd a' b → Nat.lt 0 b → Nat.lt 0 a') (b : Nat) " +
			"(_ : Nat.dvd (Nat.succ a') b) (_ : Nat.lt 0 b) => Nat.zero_lt_succ a') a"},
	valDecl{"thm", "Nat.dvd_antisymm", nil, "(a : Nat) → (b : Nat) → Nat.dvd a b → Nat.dvd b a → Eq.{1} Nat a b",
		"fun (a : Nat) => Nat.rec.{0} " +
			"(fun (a : Nat) => (b : Nat) → Nat.dvd a b → Nat.dvd b a → Eq.{1} Nat a b) " +
			"(fun (b : Nat) (hab : Nat.dvd 0 b) (_ : Nat.dvd b 0) => " +
			"Eq.symm.{1} Nat b 0 (Nat.eq_zero_of_zero_dvd b hab)) " +
			"(fun (a' : Nat) (_ : (b : Nat) → Nat.dvd a' b → Nat.dvd b a' → Eq.{1} Nat a' b) (b : Nat) " +
			"(hab : Nat.dvd (Nat.succ a') b) (hba : Nat.dvd b (Nat.succ a')) => Nat.le_antisymm (Nat.succ a') b " +
			"(Nat.le_of_dvd (Nat.succ a') b (Nat.pos_of_dvd_of_pos b (Nat.succ a') hba (Nat.zero_lt_succ a')) hab) " +
			"(Nat.le_of_dvd b (Nat.succ a') (Nat.zero_lt_succ a') hba)) a"},
	valDecl{"thm", "Nat.dvd_trans", nil,
		"(a : Nat) → (b : Nat) → (c : Nat) → Nat.dvd a b → Nat.dvd b c → Nat.dvd a c",
		"fun (a : Nat) (b : Nat) (c : Nat) (hab : Nat.dvd a b) (hbc : Nat.dvd b c) => Exists.rec.{1} Nat " +
			"(fun (d : Nat) => Eq.{1} Nat b (Nat.mul a d)) (fun (_ : Nat.dvd a b) => Nat.dvd a c) " +
			"(fun (d : Nat) (ed : Eq.{1} Nat b (Nat.mul a d)) => Exists.rec.{1} Nat " +
			"(fun (e : Nat) => Eq.{1} Nat c (Nat.mul b e)) (fun (_ : Nat.dvd b c) => Nat.dvd a c) " +
			"(fun (e : Nat) (ee : Eq.{1} Nat c (Nat.mul b e)) => Exists.intro.{1} Nat " +
			"(fun (w : Nat) => Eq.{1} Nat c (Nat.mul a w)) (Nat.mul d e) " +
			"(Eq.trans.{1} Nat c (Nat.mul b e) (Nat.mul a (Nat.mul d e)) ee " +
			"(Eq.trans.{1} Nat (Nat.mul b e) (Nat.mul (Nat.mul a d) e) (Nat.mul a (Nat.mul d e)) " +
			"(congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.mul z e) b (Nat.mul a d) ed) " +
			"(Nat.mul_assoc a d e)))) hbc) hab"},
	valDecl{"thm", "Nat.gcd_comm", nil, "(m : Nat) → (n : Nat) → Eq.{1} Nat (Nat.gcd m n) (Nat.gcd n m)",
		"fun (m : Nat) (n : Nat) => Nat.dvd_antisymm (Nat.gcd m n) (Nat.gcd n m) " +
			"(Nat.dvd_gcd (Nat.gcd m n) n m (Nat.gcd_dvd_right m n) (Nat.gcd_dvd_left m n)) " +
			"(Nat.dvd_gcd (Nat.gcd n m) m n (Nat.gcd_dvd_right n m) (Nat.gcd_dvd_left n m))"},
	valDecl{"thm", "Nat.gcd_assoc", nil,
		"(m : Nat) → (n : Nat) → (k : Nat) → Eq.{1} Nat (Nat.gcd (Nat.gcd m n) k) (Nat.gcd m (Nat.gcd n k))",
		"fun (m : Nat) (n : Nat) (k : Nat) => Nat.dvd_antisymm (Nat.gcd (Nat.gcd m n) k) " +
			"(Nat.gcd m (Nat.gcd n k)) " +
			// `gcd (gcd m n) k` divides `m`, `n` and `k`
			"(Nat.dvd_gcd (Nat.gcd (Nat.gcd m n) k) m (Nat.gcd n k) " +
			"(Nat.dvd_trans (Nat.gcd (Nat.gcd m n) k) (Nat.gcd m n) m (Nat.gcd_dvd_left (Nat.gcd m n) k) " +
			"(Nat.gcd_dvd_left m n)) " +
			"(Nat.dvd_gcd (Nat.gcd (Nat.gcd m n) k) n k " +
			"(Nat.dvd_trans (Nat.gcd (Nat.gcd m n) k) (Nat.gcd m n) n (Nat.gcd_dvd_left (Nat.gcd m n) k) " +
			"(Nat.gcd_dvd_right m n)) (Nat.gcd_dvd_right (Nat.gcd m n) k))) " +
			// and `gcd m (gcd n k)` divides `gcd m n` and `k`
			"(Nat.dvd_gcd (Nat.gcd m (Nat.gcd n k)) (Nat.gcd m n) k " +
			"(Nat.dvd_gcd (Nat.gcd m (Nat.gcd n k)) m n (Nat.gcd_dvd_left m (Nat.gcd n k)) " +
			"(Nat.dvd_trans (Nat.gcd m (Nat.gcd n k)) (Nat.gcd n k) n (Nat.gcd_dvd_right m (Nat.gcd n k)) " +
			"(Nat.gcd_dvd_left n k))) " +
			"(Nat.dvd_trans (Nat.gcd m (Nat.gcd n k)) (Nat.gcd n k) k (Nat.gcd_dvd_right m (Nat.gcd n k)) " +
			"(Nat.gcd_dvd_right n k)))"},

	// The `*`, order and divisibility facts the div/mod uniqueness lemmas and the lcm proofs rest on, under Lean's
	// names and statements (`Init/Data/Nat/Basic.lean`, `Lemmas.lean`, `Dvd.lean`). `mul` recurses on its second
	// argument here, so `k * (succ n)` is `k * n + k` definitionally and these inductions need no distributivity.
	valDecl{"thm", "Nat.le_of_eq", nil, "(a : Nat) → (b : Nat) → Eq.{1} Nat a b → Nat.le a b",
		"fun (a : Nat) (b : Nat) (h : Eq.{1} Nat a b) => Eq.rec.{0, 1} Nat a " +
			"(fun (z : Nat) (_ : Eq.{1} Nat a z) => Nat.le a z) (Nat.le.refl a) b h"},
	valDecl{"thm", "Nat.le_add_right", nil, "(n : Nat) → (m : Nat) → Nat.le n (Nat.add n m)",
		"fun (n : Nat) (m : Nat) => Eq.rec.{0, 1} Nat (Nat.add m n) " +
			"(fun (z : Nat) (_ : Eq.{1} Nat (Nat.add m n) z) => Nat.le n z) (Nat.le_add_left n m) " +
			"(Nat.add n m) (Nat.add_comm m n)"},
	valDecl{"thm", "Nat.mul_le_mul_left", nil,
		"(k : Nat) → (n : Nat) → (m : Nat) → Nat.le n m → Nat.le (Nat.mul k n) (Nat.mul k m)",
		"fun (k : Nat) (n : Nat) (m : Nat) (h : Nat.le n m) => Nat.le.rec n " +
			"(fun (y : Nat) (_ : Nat.le n y) => Nat.le (Nat.mul k n) (Nat.mul k y)) " +
			"(Nat.le.refl (Nat.mul k n)) " +
			"(fun (y : Nat) (_ : Nat.le n y) (ih : Nat.le (Nat.mul k n) (Nat.mul k y)) => " +
			"Nat.le_trans (Nat.mul k n) (Nat.mul k y) (Nat.mul k (Nat.succ y)) ih " +
			"(Nat.le_add_right (Nat.mul k y) k)) m h"},
	valDecl{"thm", "Nat.lt_of_lt_of_le", nil,
		"(a : Nat) → (b : Nat) → (c : Nat) → Nat.lt a b → Nat.le b c → Nat.lt a c",
		"fun (a : Nat) (b : Nat) (c : Nat) (h1 : Nat.lt a b) (h2 : Nat.le b c) => " +
			"Nat.le_trans (Nat.succ a) b c h1 h2"},
	valDecl{"thm", "Nat.add_lt_add_left", nil,
		"(a : Nat) → (b : Nat) → (c : Nat) → Nat.lt a b → Nat.lt (Nat.add c a) (Nat.add c b)",
		"fun (a : Nat) (b : Nat) (c : Nat) (h : Nat.lt a b) => Nat.add_le_add_left (Nat.succ a) b c h"},
	valDecl{"thm", "Nat.mul_lt_mul_of_pos_left", nil,
		"(k : Nat) → (n : Nat) → (m : Nat) → Nat.lt n m → Nat.lt 0 k → Nat.lt (Nat.mul k n) (Nat.mul k m)",
		"fun (k : Nat) (n : Nat) (m : Nat) (h : Nat.lt n m) (hk : Nat.lt 0 k) => " +
			"Nat.lt_of_lt_of_le (Nat.mul k n) (Nat.add (Nat.mul k n) k) (Nat.mul k m) " +
			"(Nat.add_lt_add_left 0 k (Nat.mul k n) hk) (Nat.mul_le_mul_left k (Nat.succ n) m h)"},
	valDecl{"thm", "Nat.not_le_of_gt", nil, "(a : Nat) → (b : Nat) → Nat.lt a b → Not (Nat.le b a)",
		"fun (a : Nat) (b : Nat) (h : Nat.lt a b) (hba : Nat.le b a) => " +
			"Nat.not_le_succ_self a (Nat.le_trans (Nat.succ a) b a h hba)"},
	valDecl{"thm", "Nat.dvd_refl", nil, "(n : Nat) → Nat.dvd n n",
		"fun (n : Nat) => Exists.intro.{1} Nat (fun (c : Nat) => Eq.{1} Nat n (Nat.mul n c)) 1 " +
			"(Eq.symm.{1} Nat (Nat.mul n 1) n (Nat.mul_one n))"},
	valDecl{"thm", "Nat.one_mul", nil, "(n : Nat) → Eq.{1} Nat (Nat.mul 1 n) n",
		"fun (n : Nat) => Nat.rec.{0} (fun (n : Nat) => Eq.{1} Nat (Nat.mul 1 n) n) (Eq.refl.{1} Nat 0) " +
			"(fun (k : Nat) (ih : Eq.{1} Nat (Nat.mul 1 k) k) => " +
			"congrArg.{1, 1} Nat Nat Nat.succ (Nat.mul 1 k) k ih) n"},
	valDecl{"thm", "Nat.dvd_mul_right", nil, "(a : Nat) → (b : Nat) → Nat.dvd a (Nat.mul a b)",
		"fun (a : Nat) (b : Nat) => Exists.intro.{1} Nat (fun (c : Nat) => Eq.{1} Nat (Nat.mul a b) (Nat.mul a c)) b " +
			"(Eq.refl.{1} Nat (Nat.mul a b))"},
	valDecl{"thm", "Nat.dvd_mul_left", nil, "(a : Nat) → (b : Nat) → Nat.dvd a (Nat.mul b a)",
		"fun (a : Nat) (b : Nat) => Exists.intro.{1} Nat (fun (c : Nat) => Eq.{1} Nat (Nat.mul b a) (Nat.mul a c)) b " +
			"(Nat.mul_comm b a)"},
	valDecl{"thm", "Nat.mul_dvd_mul_left", nil,
		"(k : Nat) → (a : Nat) → (b : Nat) → Nat.dvd a b → Nat.dvd (Nat.mul k a) (Nat.mul k b)",
		"fun (k : Nat) (a : Nat) (b : Nat) (h : Nat.dvd a b) => Exists.rec.{1} Nat " +
			"(fun (c : Nat) => Eq.{1} Nat b (Nat.mul a c)) " +
			"(fun (_ : Nat.dvd a b) => Nat.dvd (Nat.mul k a) (Nat.mul k b)) " +
			"(fun (c : Nat) (e : Eq.{1} Nat b (Nat.mul a c)) => Exists.intro.{1} Nat " +
			"(fun (w : Nat) => Eq.{1} Nat (Nat.mul k b) (Nat.mul (Nat.mul k a) w)) c " +
			"(Eq.trans.{1} Nat (Nat.mul k b) (Nat.mul k (Nat.mul a c)) (Nat.mul (Nat.mul k a) c) " +
			"(congrArg.{1, 1} Nat Nat (Nat.mul k) b (Nat.mul a c) e) " +
			"(Eq.symm.{1} Nat (Nat.mul (Nat.mul k a) c) (Nat.mul k (Nat.mul a c)) (Nat.mul_assoc k a c)))) h"},
	valDecl{"thm", "Nat.mul_dvd_mul_right", nil,
		"(a : Nat) → (b : Nat) → (c : Nat) → Nat.dvd a b → Nat.dvd (Nat.mul a c) (Nat.mul b c)",
		"fun (a : Nat) (b : Nat) (c : Nat) (h : Nat.dvd a b) => Exists.rec.{1} Nat " +
			"(fun (d : Nat) => Eq.{1} Nat b (Nat.mul a d)) " +
			"(fun (_ : Nat.dvd a b) => Nat.dvd (Nat.mul a c) (Nat.mul b c)) " +
			"(fun (d : Nat) (e : Eq.{1} Nat b (Nat.mul a d)) => Exists.intro.{1} Nat " +
			"(fun (w : Nat) => Eq.{1} Nat (Nat.mul b c) (Nat.mul (Nat.mul a c) w)) d " +
			"(Eq.trans.{1} Nat (Nat.mul b c) (Nat.mul (Nat.mul a d) c) (Nat.mul (Nat.mul a c) d) " +
			"(congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.mul z c) b (Nat.mul a d) e) " +
			"(Eq.trans.{1} Nat (Nat.mul (Nat.mul a d) c) (Nat.mul a (Nat.mul d c)) (Nat.mul (Nat.mul a c) d) " +
			"(Nat.mul_assoc a d c) " +
			"(Eq.trans.{1} Nat (Nat.mul a (Nat.mul d c)) (Nat.mul a (Nat.mul c d)) (Nat.mul (Nat.mul a c) d) " +
			"(congrArg.{1, 1} Nat Nat (Nat.mul a) (Nat.mul d c) (Nat.mul c d) (Nat.mul_comm d c)) " +
			"(Eq.symm.{1} Nat (Nat.mul (Nat.mul a c) d) (Nat.mul a (Nat.mul c d)) (Nat.mul_assoc a c d)))))) h"},
	valDecl{"thm", "Nat.mul_dvd_mul", nil,
		"(a : Nat) → (b : Nat) → (c : Nat) → (d : Nat) → Nat.dvd a b → Nat.dvd c d → Nat.dvd (Nat.mul a c) (Nat.mul b d)",
		"fun (a : Nat) (b : Nat) (c : Nat) (d : Nat) (hab : Nat.dvd a b) (hcd : Nat.dvd c d) => " +
			"Nat.dvd_trans (Nat.mul a c) (Nat.mul b c) (Nat.mul b d) " +
			"(Nat.mul_dvd_mul_right a b c hab) (Nat.mul_dvd_mul_left b c d hcd)"},

	// `Nat.div` and `Nat.mod` are a state machine here (see `Nat.divMod`), so their uniqueness lemma is the machine's
	// invariant: from `mk q 0` it takes `k` steps to `mk (q+1) 0`, and fewer than `k` steps never wrap. Together the
	// two say `divMod k (k * q + r) = mk q r` whenever `r < k`, which is where `a / b` and `a % b` stop being opaque.
	valDecl{"thm", "Nat.divIter_zero", nil,
		"(j : Nat) → (q : Nat) → (r : Nat) → Eq.{1} Nat.DivMod (Nat.divIter 0 j (Nat.DivMod.mk q r)) " +
			"(Nat.DivMod.mk q (Nat.add r j))",
		"fun (j : Nat) => Nat.rec.{0} (fun (j : Nat) => (q : Nat) → (r : Nat) → " +
			"Eq.{1} Nat.DivMod (Nat.divIter 0 j (Nat.DivMod.mk q r)) (Nat.DivMod.mk q (Nat.add r j))) " +
			"(fun (q : Nat) (r : Nat) => Eq.refl.{1} Nat.DivMod (Nat.DivMod.mk q (Nat.add r 0))) " +
			"(fun (j : Nat) (ih : (q : Nat) → (r : Nat) → Eq.{1} Nat.DivMod (Nat.divIter 0 j (Nat.DivMod.mk q r)) " +
			"(Nat.DivMod.mk q (Nat.add r j))) (q : Nat) (r : Nat) => " +
			"Eq.trans.{1} Nat.DivMod (Nat.divStep 0 (Nat.divIter 0 j (Nat.DivMod.mk q r))) " +
			"(Nat.divStep 0 (Nat.DivMod.mk q (Nat.add r j))) (Nat.DivMod.mk q (Nat.add r (Nat.succ j))) " +
			"(congrArg.{1, 1} Nat.DivMod Nat.DivMod (Nat.divStep 0) (Nat.divIter 0 j (Nat.DivMod.mk q r)) " +
			"(Nat.DivMod.mk q (Nat.add r j)) (ih q r)) " +
			"(Eq.refl.{1} Nat.DivMod (Nat.DivMod.mk q (Nat.succ (Nat.add r j))))) j"},
	valDecl{"thm", "Nat.div_zero", nil, "(a : Nat) → Eq.{1} Nat (Nat.div a 0) 0",
		"fun (a : Nat) => Nat.div_proj 0 a 0 (Nat.add 0 a) (Nat.divIter_zero a 0 0)"},
	valDecl{"thm", "Nat.mod_zero", nil, "(a : Nat) → Eq.{1} Nat (Nat.mod a 0) a",
		"fun (a : Nat) => Eq.trans.{1} Nat (Nat.mod a 0) (Nat.add 0 a) a " +
			"(Nat.mod_proj 0 a 0 (Nat.add 0 a) (Nat.divIter_zero a 0 0)) (Nat.zero_add a)"},
	valDecl{"thm", "Nat.divMod_add_mul", nil,
		"(k : Nat) → (q : Nat) → (r : Nat) → Nat.lt 0 k → Nat.lt r k → " +
			"Eq.{1} Nat.DivMod (Nat.divMod k (Nat.add (Nat.mul k q) r)) (Nat.DivMod.mk q r)",
		"fun (k : Nat) (q : Nat) (r : Nat) (hk : Nat.lt 0 k) (hr : Nat.lt r k) => " +
			"Eq.trans.{1} Nat.DivMod (Nat.divIter k (Nat.add (Nat.mul k q) r) (Nat.DivMod.mk 0 0)) " +
			"(Nat.divIter k r (Nat.divIter k (Nat.mul k q) (Nat.DivMod.mk 0 0))) (Nat.DivMod.mk q r) " +
			"(Nat.divIter_add k (Nat.mul k q) r (Nat.DivMod.mk 0 0)) " +
			"(Eq.trans.{1} Nat.DivMod (Nat.divIter k r (Nat.divIter k (Nat.mul k q) (Nat.DivMod.mk 0 0))) " +
			"(Nat.divIter k r (Nat.DivMod.mk q 0)) (Nat.DivMod.mk q r) " +
			"(congrArg.{1, 1} Nat.DivMod Nat.DivMod (Nat.divIter k r) " +
			"(Nat.divIter k (Nat.mul k q) (Nat.DivMod.mk 0 0)) (Nat.DivMod.mk q 0) (Nat.divMod_mul k q hk)) " +
			"(Eq.trans.{1} Nat.DivMod (Nat.divIter k r (Nat.DivMod.mk q 0)) " +
			"(Nat.DivMod.mk q (Nat.add 0 r)) (Nat.DivMod.mk q r) " +
			"(Nat.divIter_lt k q 0 r (Eq.rec.{0, 1} Nat r (fun (z : Nat) (_ : Eq.{1} Nat r z) => " +
			"Nat.lt z k) hr (Nat.add 0 r) (Eq.symm.{1} Nat (Nat.add 0 r) r (Nat.zero_add r)))) " +
			"(Nat.DivMod.mk_congr q (Nat.add 0 r) q r (Eq.refl.{1} Nat q) (Nat.zero_add r))))"},
	valDecl{"thm", "Nat.divMod_of_mul_add", nil,
		"(k : Nat) → (q : Nat) → (r : Nat) → (n : Nat) → Eq.{1} Nat n (Nat.add (Nat.mul k q) r) → " +
			"Nat.lt 0 k → Nat.lt r k → Eq.{1} Nat.DivMod (Nat.divMod k n) (Nat.DivMod.mk q r)",
		"fun (k : Nat) (q : Nat) (r : Nat) (n : Nat) (h : Eq.{1} Nat n (Nat.add (Nat.mul k q) r)) " +
			"(hk : Nat.lt 0 k) (hr : Nat.lt r k) => Eq.trans.{1} Nat.DivMod (Nat.divMod k n) " +
			"(Nat.divMod k (Nat.add (Nat.mul k q) r)) (Nat.DivMod.mk q r) " +
			"(congrArg.{1, 1} Nat Nat.DivMod (Nat.divMod k) n (Nat.add (Nat.mul k q) r) h) " +
			"(Nat.divMod_add_mul k q r hk hr)"},
	valDecl{"thm", "Nat.div_eq_of_mul_eq", nil,
		"(k : Nat) → (q : Nat) → (n : Nat) → Eq.{1} Nat n (Nat.mul k q) → Nat.lt 0 k → " +
			"Eq.{1} Nat (Nat.div n k) q",
		"fun (k : Nat) (q : Nat) (n : Nat) (h : Eq.{1} Nat n (Nat.mul k q)) (hk : Nat.lt 0 k) => " +
			"Nat.div_proj k n q 0 (Eq.trans.{1} Nat.DivMod (Nat.divMod k n) (Nat.divMod k (Nat.mul k q)) " +
			"(Nat.DivMod.mk q 0) (congrArg.{1, 1} Nat Nat.DivMod (Nat.divMod k) n (Nat.mul k q) h) " +
			"(Nat.divMod_add_mul k q 0 hk hk))"},
	valDecl{"thm", "Nat.div_eq_of_lt", nil, "(a : Nat) → (b : Nat) → Nat.lt a b → Eq.{1} Nat (Nat.div a b) 0",
		"fun (a : Nat) (b : Nat) (h : Nat.lt a b) => Nat.div_proj b a 0 a " +
			"(Nat.divMod_of_mul_add b 0 a a (Eq.rec.{0, 1} Nat (Nat.mul b 0) (fun (z : Nat) " +
			"(_ : Eq.{1} Nat (Nat.mul b 0) z) => Eq.{1} Nat a (Nat.add z a)) " +
			"(Eq.symm.{1} Nat (Nat.add 0 a) a (Nat.zero_add a)) 0 (Nat.mul_zero b)) " +
			"(Nat.le_trans (Nat.succ 0) (Nat.succ a) b (Nat.succ_le_succ 0 a (Nat.zero_le a)) h) h)"},
	valDecl{"thm", "Nat.mod_eq_of_lt", nil, "(a : Nat) → (b : Nat) → Nat.lt a b → Eq.{1} Nat (Nat.mod a b) a",
		"fun (a : Nat) (b : Nat) (h : Nat.lt a b) => Nat.mod_proj b a 0 a " +
			"(Nat.divMod_of_mul_add b 0 a a (Eq.rec.{0, 1} Nat (Nat.mul b 0) (fun (z : Nat) " +
			"(_ : Eq.{1} Nat (Nat.mul b 0) z) => Eq.{1} Nat a (Nat.add z a)) " +
			"(Eq.symm.{1} Nat (Nat.add 0 a) a (Nat.zero_add a)) 0 (Nat.mul_zero b)) " +
			"(Nat.le_trans (Nat.succ 0) (Nat.succ a) b (Nat.succ_le_succ 0 a (Nat.zero_le a)) h) h)"},
	valDecl{"thm", "Nat.div_self", nil, "(n : Nat) → Nat.lt 0 n → Eq.{1} Nat (Nat.div n n) 1",
		"fun (n : Nat) (h : Nat.lt 0 n) => Nat.div_proj n n 1 0 " +
			"(Nat.divMod_of_mul_add n 1 0 n (Eq.symm.{1} Nat (Nat.mul n 1) n (Nat.mul_one n)) h h)"},
	valDecl{"thm", "Nat.mul_div_cancel", nil, "(m : Nat) → (n : Nat) → Nat.lt 0 n → Eq.{1} Nat (Nat.div (Nat.mul m n) n) m",
		"fun (m : Nat) (n : Nat) (h : Nat.lt 0 n) => Nat.div_proj n (Nat.mul m n) m 0 " +
			"(Eq.trans.{1} Nat.DivMod (Nat.divMod n (Nat.mul m n)) (Nat.divMod n (Nat.mul n m)) " +
			"(Nat.DivMod.mk m 0) " +
			"(congrArg.{1, 1} Nat Nat.DivMod (Nat.divMod n) (Nat.mul m n) (Nat.mul n m) (Nat.mul_comm m n)) " +
			"(Nat.divMod_add_mul n m 0 h h))"},
	valDecl{"thm", "Nat.mul_div_cancel_left", nil,
		"(m : Nat) → (n : Nat) → Nat.lt 0 n → Eq.{1} Nat (Nat.div (Nat.mul n m) n) m",
		"fun (m : Nat) (n : Nat) (h : Nat.lt 0 n) => Eq.trans.{1} Nat (Nat.div (Nat.mul n m) n) " +
			"(Nat.div (Nat.mul m n) n) m " +
			"(congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.div z n) (Nat.mul n m) (Nat.mul m n) (Nat.mul_comm n m)) " +
			"(Nat.mul_div_cancel m n h)"},
	valDecl{"thm", "Nat.mul_div_cancel'", nil,
		"(n : Nat) → (m : Nat) → Nat.dvd n m → Eq.{1} Nat (Nat.mul n (Nat.div m n)) m",
		"fun (n : Nat) => Nat.rec.{0} (fun (n : Nat) => (m : Nat) → Nat.dvd n m → " +
			"Eq.{1} Nat (Nat.mul n (Nat.div m n)) m) " +
			// `n = 0` forces `m = 0`, and both sides are then `0`
			"(fun (m : Nat) (h0 : Nat.dvd 0 m) => Eq.rec.{0, 1} Nat 0 (fun (z : Nat) (_ : Eq.{1} Nat 0 z) => " +
			"Eq.{1} Nat (Nat.mul 0 (Nat.div z 0)) z) (Nat.zero_mul (Nat.div 0 0)) m " +
			"(Eq.symm.{1} Nat m 0 (Nat.eq_zero_of_zero_dvd m h0))) " +
			// `n = succ n'`: `m = n * c` makes `m / n` reduce to `c`
			"(fun (n' : Nat) (_ : (m : Nat) → Nat.dvd n' m → Eq.{1} Nat (Nat.mul n' (Nat.div m n')) m) " +
			"(m : Nat) (h : Nat.dvd (Nat.succ n') m) => Exists.rec.{1} Nat " +
			"(fun (c : Nat) => Eq.{1} Nat m (Nat.mul (Nat.succ n') c)) " +
			"(fun (_ : Nat.dvd (Nat.succ n') m) => Eq.{1} Nat (Nat.mul (Nat.succ n') (Nat.div m (Nat.succ n'))) m) " +
			"(fun (c : Nat) (e : Eq.{1} Nat m (Nat.mul (Nat.succ n') c)) => " +
			"Eq.trans.{1} Nat (Nat.mul (Nat.succ n') (Nat.div m (Nat.succ n'))) (Nat.mul (Nat.succ n') c) m " +
			"(congrArg.{1, 1} Nat Nat (Nat.mul (Nat.succ n')) (Nat.div m (Nat.succ n')) c " +
			"(Nat.div_proj (Nat.succ n') m c 0 (Eq.trans.{1} Nat.DivMod (Nat.divMod (Nat.succ n') m) " +
			"(Nat.divMod (Nat.succ n') (Nat.mul (Nat.succ n') c)) (Nat.DivMod.mk c 0) " +
			"(congrArg.{1, 1} Nat Nat.DivMod (Nat.divMod (Nat.succ n')) m (Nat.mul (Nat.succ n') c) e) " +
			"(Nat.divMod_mul (Nat.succ n') c (Nat.zero_lt_succ n'))))) " +
			"(Eq.symm.{1} Nat m (Nat.mul (Nat.succ n') c) e)) h) n"},
	valDecl{"thm", "Nat.div_mul_cancel", nil,
		"(n : Nat) → (m : Nat) → Nat.dvd n m → Eq.{1} Nat (Nat.mul (Nat.div m n) n) m",
		"fun (n : Nat) (m : Nat) (h : Nat.dvd n m) => Eq.trans.{1} Nat (Nat.mul (Nat.div m n) n) " +
			"(Nat.mul n (Nat.div m n)) m (Nat.mul_comm (Nat.div m n) n) (Nat.mul_div_cancel' n m h)"},
	valDecl{"thm", "Nat.mul_div_assoc", nil,
		"(a : Nat) → (b : Nat) → (c : Nat) → Nat.dvd c b → " +
			"Eq.{1} Nat (Nat.div (Nat.mul a b) c) (Nat.mul a (Nat.div b c))",
		"fun (a : Nat) (b : Nat) (c : Nat) (h : Nat.dvd c b) => Nat.rec.{0} " +
			"(fun (c : Nat) => Nat.dvd c b → Eq.{1} Nat (Nat.div (Nat.mul a b) c) (Nat.mul a (Nat.div b c))) " +
			// `c = 0`: both sides are `0`
			"(fun (_ : Nat.dvd 0 b) => Eq.trans.{1} Nat (Nat.div (Nat.mul a b) 0) 0 " +
			"(Nat.mul a (Nat.div b 0)) (Nat.div_zero (Nat.mul a b)) " +
			"(Eq.symm.{1} Nat (Nat.mul a (Nat.div b 0)) 0 (Eq.trans.{1} Nat (Nat.mul a (Nat.div b 0)) " +
			"(Nat.mul a 0) 0 (congrArg.{1, 1} Nat Nat (Nat.mul a) (Nat.div b 0) 0 (Nat.div_zero b)) " +
			"(Nat.mul_zero a)))) " +
			// `c = succ c'`: `b = c * d`, and `a * b = c * (a * d)` makes `(a * b) / c` reduce to `a * d`
			"(fun (c' : Nat) (_ : Nat.dvd c' b → Eq.{1} Nat (Nat.div (Nat.mul a b) c') (Nat.mul a (Nat.div b c'))) " +
			"(h' : Nat.dvd (Nat.succ c') b) => Exists.rec.{1} Nat " +
			"(fun (d : Nat) => Eq.{1} Nat b (Nat.mul (Nat.succ c') d)) " +
			"(fun (_ : Nat.dvd (Nat.succ c') b) => " +
			"Eq.{1} Nat (Nat.div (Nat.mul a b) (Nat.succ c')) (Nat.mul a (Nat.div b (Nat.succ c')))) " +
			"(fun (d : Nat) (e : Eq.{1} Nat b (Nat.mul (Nat.succ c') d)) => " +
			"Eq.trans.{1} Nat (Nat.div (Nat.mul a b) (Nat.succ c')) (Nat.mul a d) " +
			"(Nat.mul a (Nat.div b (Nat.succ c'))) " +
			"(Nat.div_eq_of_mul_eq (Nat.succ c') (Nat.mul a d) (Nat.mul a b) " +
			"(Eq.trans.{1} Nat (Nat.mul a b) (Nat.mul a (Nat.mul (Nat.succ c') d)) " +
			"(Nat.mul (Nat.succ c') (Nat.mul a d)) " +
			"(congrArg.{1, 1} Nat Nat (Nat.mul a) b (Nat.mul (Nat.succ c') d) e) " +
			"(Eq.trans.{1} Nat (Nat.mul a (Nat.mul (Nat.succ c') d)) " +
			"(Nat.mul (Nat.mul a (Nat.succ c')) d) (Nat.mul (Nat.succ c') (Nat.mul a d)) " +
			"(Eq.symm.{1} Nat (Nat.mul (Nat.mul a (Nat.succ c')) d) " +
			"(Nat.mul a (Nat.mul (Nat.succ c') d)) (Nat.mul_assoc a (Nat.succ c') d)) " +
			"(Eq.trans.{1} Nat (Nat.mul (Nat.mul a (Nat.succ c')) d) (Nat.mul (Nat.mul (Nat.succ c') a) d) " +
			"(Nat.mul (Nat.succ c') (Nat.mul a d)) " +
			"(congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.mul z d) (Nat.mul a (Nat.succ c')) " +
			"(Nat.mul (Nat.succ c') a) (Nat.mul_comm a (Nat.succ c'))) " +
			"(Nat.mul_assoc (Nat.succ c') a d)))) (Nat.zero_lt_succ c')) " +
			"(congrArg.{1, 1} Nat Nat (Nat.mul a) d (Nat.div b (Nat.succ c')) " +
			"(Eq.symm.{1} Nat (Nat.div b (Nat.succ c')) d (Nat.div_eq_of_mul_eq (Nat.succ c') d b e " +
			"(Nat.zero_lt_succ c'))))) h') c h"},
	valDecl{"thm", "Nat.mod_mul_add", nil,
		"(k : Nat) → (q : Nat) → (r : Nat) → (n : Nat) → Eq.{1} Nat n (Nat.add (Nat.mul k q) r) → " +
			"Nat.lt 0 k → Nat.lt r k → Eq.{1} Nat (Nat.mod n k) r",
		"fun (k : Nat) (q : Nat) (r : Nat) (n : Nat) (h : Eq.{1} Nat n (Nat.add (Nat.mul k q) r)) " +
			"(hk : Nat.lt 0 k) (hr : Nat.lt r k) => " +
			"Nat.mod_proj k n q r (Nat.divMod_of_mul_add k q r n h hk hr)"},
	valDecl{"thm", "Nat.mod_mul_zero_left", nil,
		"(a : Nat) → (b : Nat) → Eq.{1} Nat (Nat.mod (Nat.mul 0 b) (Nat.mul 0 a)) (Nat.mul 0 (Nat.mod b a))",
		"fun (a : Nat) (b : Nat) => Eq.trans.{1} Nat (Nat.mod (Nat.mul 0 b) (Nat.mul 0 a)) " +
			"(Nat.mod 0 (Nat.mul 0 a)) (Nat.mul 0 (Nat.mod b a)) " +
			"(Eq.trans.{1} Nat (Nat.mod (Nat.mul 0 b) (Nat.mul 0 a)) (Nat.mod 0 (Nat.mul 0 a)) " +
			"(Nat.mod 0 0) " +
			"(congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.mod z (Nat.mul 0 a)) (Nat.mul 0 b) 0 " +
			"(Nat.zero_mul b)) " +
			"(congrArg.{1, 1} Nat Nat (fun (w : Nat) => Nat.mod 0 w) (Nat.mul 0 a) 0 (Nat.zero_mul a))) " +
			"(Eq.trans.{1} Nat (Nat.mod 0 0) 0 (Nat.mul 0 (Nat.mod b a)) (Eq.refl.{1} Nat 0) " +
			"(Eq.symm.{1} Nat (Nat.mul 0 (Nat.mod b a)) 0 (Nat.zero_mul (Nat.mod b a))))"},
	valDecl{"thm", "Nat.mod_mul_zero_right", nil,
		"(m : Nat) → (b : Nat) → Eq.{1} Nat (Nat.mod (Nat.mul (Nat.succ m) b) (Nat.mul (Nat.succ m) 0)) " +
			"(Nat.mul (Nat.succ m) (Nat.mod b 0))",
		"fun (m : Nat) (b : Nat) => Eq.trans.{1} Nat " +
			"(Nat.mod (Nat.mul (Nat.succ m) b) (Nat.mul (Nat.succ m) 0)) (Nat.mul (Nat.succ m) b) " +
			"(Nat.mul (Nat.succ m) (Nat.mod b 0)) (Nat.mod_zero (Nat.mul (Nat.succ m) b)) " +
			"(Eq.symm.{1} Nat (Nat.mul (Nat.succ m) (Nat.mod b 0)) (Nat.mul (Nat.succ m) b) " +
			"(congrArg.{1, 1} Nat Nat (Nat.mul (Nat.succ m)) (Nat.mod b 0) b (Nat.mod_zero b)))"},
	valDecl{"thm", "Nat.mod_mul_succ", nil,
		"(m : Nat) → (a : Nat) → (b : Nat) → Eq.{1} Nat (Nat.mod (Nat.mul (Nat.succ m) b) " +
			"(Nat.mul (Nat.succ m) (Nat.succ a))) (Nat.mul (Nat.succ m) (Nat.mod b (Nat.succ a)))",
		"fun (m : Nat) (a : Nat) (b : Nat) => Nat.mod_mul_add (Nat.mul (Nat.succ m) (Nat.succ a)) " +
			"(Nat.div b (Nat.succ a)) (Nat.mul (Nat.succ m) (Nat.mod b (Nat.succ a))) " +
			"(Nat.mul (Nat.succ m) b) " +
			"(Eq.trans.{1} Nat (Nat.mul (Nat.succ m) b) " +
			"(Nat.mul (Nat.succ m) (Nat.add (Nat.mul (Nat.succ a) (Nat.div b (Nat.succ a))) " +
			"(Nat.mod b (Nat.succ a)))) " +
			"(Nat.add (Nat.mul (Nat.mul (Nat.succ m) (Nat.succ a)) (Nat.div b (Nat.succ a))) " +
			"(Nat.mul (Nat.succ m) (Nat.mod b (Nat.succ a)))) " +
			"(congrArg.{1, 1} Nat Nat (Nat.mul (Nat.succ m)) b " +
			"(Nat.add (Nat.mul (Nat.succ a) (Nat.div b (Nat.succ a))) (Nat.mod b (Nat.succ a))) " +
			"(Eq.symm.{1} Nat (Nat.add (Nat.mul (Nat.succ a) (Nat.div b (Nat.succ a))) " +
			"(Nat.mod b (Nat.succ a))) b (And.right " +
			"(Eq.{1} Nat.DivMod (Nat.divMod (Nat.succ a) b) (Nat.DivMod.mk (Nat.div b (Nat.succ a)) " +
			"(Nat.mod b (Nat.succ a)))) (Eq.{1} Nat (Nat.add (Nat.mul (Nat.succ a) (Nat.div b (Nat.succ a))) " +
			"(Nat.mod b (Nat.succ a))) b) (Nat.divMod_spec (Nat.succ a) b)))) " +
			"(Eq.trans.{1} Nat (Nat.mul (Nat.succ m) (Nat.add (Nat.mul (Nat.succ a) " +
			"(Nat.div b (Nat.succ a))) (Nat.mod b (Nat.succ a)))) " +
			"(Nat.add (Nat.mul (Nat.succ m) (Nat.mul (Nat.succ a) (Nat.div b (Nat.succ a)))) " +
			"(Nat.mul (Nat.succ m) (Nat.mod b (Nat.succ a)))) " +
			"(Nat.add (Nat.mul (Nat.mul (Nat.succ m) (Nat.succ a)) (Nat.div b (Nat.succ a))) " +
			"(Nat.mul (Nat.succ m) (Nat.mod b (Nat.succ a)))) " +
			"(Nat.mul_add (Nat.succ m) (Nat.mul (Nat.succ a) (Nat.div b (Nat.succ a))) " +
			"(Nat.mod b (Nat.succ a))) " +
			"(congrArg.{1, 1} Nat Nat " +
			"(fun (z : Nat) => Nat.add z (Nat.mul (Nat.succ m) (Nat.mod b (Nat.succ a)))) " +
			"(Nat.mul (Nat.succ m) (Nat.mul (Nat.succ a) (Nat.div b (Nat.succ a)))) " +
			"(Nat.mul (Nat.mul (Nat.succ m) (Nat.succ a)) (Nat.div b (Nat.succ a))) " +
			"(Eq.symm.{1} Nat (Nat.mul (Nat.mul (Nat.succ m) (Nat.succ a)) (Nat.div b (Nat.succ a))) " +
			"(Nat.mul (Nat.succ m) (Nat.mul (Nat.succ a) (Nat.div b (Nat.succ a)))) " +
			"(Nat.mul_assoc (Nat.succ m) (Nat.succ a) (Nat.div b (Nat.succ a))))))) " +
			"(Nat.zero_lt_succ (Nat.add (Nat.mul (Nat.succ m) a) m)) " +
			"(Nat.mul_lt_mul_of_pos_left (Nat.succ m) (Nat.mod b (Nat.succ a)) (Nat.succ a) " +
			"(Nat.mod_lt b (Nat.succ a) (Nat.zero_lt_succ a)) (Nat.zero_lt_succ m))"},
	valDecl{"thm", "Nat.mul_mod_mul_left", nil,
		"(m : Nat) → (a : Nat) → (b : Nat) → Eq.{1} Nat (Nat.mod (Nat.mul m b) (Nat.mul m a)) " +
			"(Nat.mul m (Nat.mod b a))",
		"fun (m : Nat) => Nat.rec.{0} (fun (m : Nat) => (a : Nat) → (b : Nat) → " +
			"Eq.{1} Nat (Nat.mod (Nat.mul m b) (Nat.mul m a)) (Nat.mul m (Nat.mod b a))) " +
			"(fun (a : Nat) (b : Nat) => Nat.mod_mul_zero_left a b) " +
			"(fun (m' : Nat) (_ : (a : Nat) → (b : Nat) → Eq.{1} Nat (Nat.mod (Nat.mul m' b) " +
			"(Nat.mul m' a)) (Nat.mul m' (Nat.mod b a))) (a : Nat) => Nat.rec.{0} " +
			"(fun (a : Nat) => (b : Nat) → Eq.{1} Nat (Nat.mod (Nat.mul (Nat.succ m') b) " +
			"(Nat.mul (Nat.succ m') a)) (Nat.mul (Nat.succ m') (Nat.mod b a))) " +
			"(fun (b : Nat) => Nat.mod_mul_zero_right m' b) " +
			"(fun (a' : Nat) (_ : (b : Nat) → Eq.{1} Nat (Nat.mod (Nat.mul (Nat.succ m') b) " +
			"(Nat.mul (Nat.succ m') a')) (Nat.mul (Nat.succ m') (Nat.mod b a'))) (b : Nat) => " +
			"Nat.mod_mul_succ m' a' b) a) m"},

	// The countdown of `gcdIter` scaled by `m`: with `a <= f <= F`, `gcdIter F (m*a) (m*b) = m * gcdIter f a b`. The two fuels
	// differ because `gcd (m*n) (m*k)` starts with fuel `m*n` and `gcd n k` with `n`; the induction is on `f`, and each
	// step needs only `a`, the remainder, to stay under it.
	valDecl{"thm", "Nat.gcdIter_mul_left", nil,
		"(m' : Nat) → (f : Nat) → (a : Nat) → (b : Nat) → (F : Nat) → Nat.le a f → Nat.le f F → Eq.{1} Nat (Nat.gcdIter F (Nat.mul (Nat.succ m') a) " +
			"(Nat.mul (Nat.succ m') b)) (Nat.mul (Nat.succ m') (Nat.gcdIter f a b))",
		"fun (m' : Nat) (f : Nat) => Nat.rec.{0} (fun (f : Nat) => (a : Nat) → (b : Nat) → (F : Nat) → Nat.le a f → Nat.le f F → Eq.{1} Nat " +
			"(Nat.gcdIter F (Nat.mul (Nat.succ m') a) (Nat.mul (Nat.succ m') b)) (Nat.mul (Nat.succ m') (Nat.gcdIter f a b))) (fun (a : Nat) (b : Nat) (F " +
			": Nat) (ha : Nat.le a 0) (hF : Nat.le 0 F) => Nat.rec.{0} (fun (a : Nat) => Nat.le a 0 → Eq.{1} Nat (Nat.gcdIter F (Nat.mul (Nat.succ m') a) " +
			"(Nat.mul (Nat.succ m') b)) (Nat.mul (Nat.succ m') (Nat.gcdIter 0 a b))) (fun (_ : Nat.le 0 0) => Nat.rec.{0} (fun (F : Nat) => Eq.{1} Nat " +
			"(Nat.gcdIter F (Nat.mul (Nat.succ m') 0) (Nat.mul (Nat.succ m') b)) (Nat.mul (Nat.succ m') (Nat.gcdIter 0 0 b))) (Eq.refl.{1} Nat (Nat.mul " +
			"(Nat.succ m') b)) (fun (F' : Nat) (_ : Eq.{1} Nat (Nat.gcdIter F' (Nat.mul (Nat.succ m') 0) (Nat.mul (Nat.succ m') b)) (Nat.mul (Nat.succ " +
			"m') (Nat.gcdIter 0 0 b))) => Eq.refl.{1} Nat (Nat.mul (Nat.succ m') b)) F) (fun (a' : Nat) (_ : Nat.le a' 0 → Eq.{1} Nat (Nat.gcdIter F " +
			"(Nat.mul (Nat.succ m') a') (Nat.mul (Nat.succ m') b)) (Nat.mul (Nat.succ m') (Nat.gcdIter 0 a' b))) (h : Nat.le (Nat.succ a') 0) => " +
			"False.elim.{0} (Eq.{1} Nat (Nat.gcdIter F (Nat.mul (Nat.succ m') (Nat.succ a')) (Nat.mul (Nat.succ m') b)) (Nat.mul (Nat.succ m') " +
			"(Nat.gcdIter 0 (Nat.succ a') b))) (Nat.succ_ne_zero a' (Nat.le_zero (Nat.succ a') h))) a ha) (fun (f' : Nat) (ih : (a : Nat) → (b : Nat) → " +
			"(F : Nat) → Nat.le a f' → Nat.le f' F → Eq.{1} Nat (Nat.gcdIter F (Nat.mul (Nat.succ m') a) (Nat.mul (Nat.succ m') b)) (Nat.mul (Nat.succ " +
			"m') (Nat.gcdIter f' a b))) (a : Nat) (b : Nat) (F : Nat) (ha : Nat.le a (Nat.succ f')) (hF : Nat.le (Nat.succ f') F) => Nat.rec.{0} (fun (F " +
			": Nat) => Nat.le (Nat.succ f') F → Eq.{1} Nat (Nat.gcdIter F (Nat.mul (Nat.succ m') a) (Nat.mul (Nat.succ m') b)) (Nat.mul (Nat.succ m') " +
			"(Nat.gcdIter (Nat.succ f') a b))) (fun (h : Nat.le (Nat.succ f') 0) => False.elim.{0} (Eq.{1} Nat (Nat.gcdIter 0 (Nat.mul (Nat.succ m') a) " +
			"(Nat.mul (Nat.succ m') b)) (Nat.mul (Nat.succ m') (Nat.gcdIter (Nat.succ f') a b))) (Nat.succ_ne_zero f' (Nat.le_zero (Nat.succ f') h))) " +
			"(fun (F' : Nat) (_ : Nat.le (Nat.succ f') F' → Eq.{1} Nat (Nat.gcdIter F' (Nat.mul (Nat.succ m') a) (Nat.mul (Nat.succ m') b)) (Nat.mul " +
			"(Nat.succ m') (Nat.gcdIter (Nat.succ f') a b))) (hF' : Nat.le (Nat.succ f') (Nat.succ F')) => Nat.rec.{0} (fun (a : Nat) => Nat.le a " +
			"(Nat.succ f') → Eq.{1} Nat (Nat.gcdIter (Nat.succ F') (Nat.mul (Nat.succ m') a) (Nat.mul (Nat.succ m') b)) (Nat.mul (Nat.succ m') " +
			"(Nat.gcdIter (Nat.succ f') a b))) (fun (_ : Nat.le 0 (Nat.succ f')) => Eq.refl.{1} Nat (Nat.mul (Nat.succ m') b)) (fun (a' : Nat) (_ : " +
			"Nat.le a' (Nat.succ f') → Eq.{1} Nat (Nat.gcdIter (Nat.succ F') (Nat.mul (Nat.succ m') a') (Nat.mul (Nat.succ m') b)) (Nat.mul (Nat.succ m') " +
			"(Nat.gcdIter (Nat.succ f') a' b))) (ha : Nat.le (Nat.succ a') (Nat.succ f')) => Eq.trans.{1} Nat (Nat.gcdIter (Nat.succ F') (Nat.mul " +
			"(Nat.succ m') (Nat.succ a')) (Nat.mul (Nat.succ m') b)) (Nat.gcdIter F' (Nat.mul (Nat.succ m') (Nat.mod b (Nat.succ a'))) (Nat.mul (Nat.succ " +
			"m') (Nat.succ a'))) (Nat.mul (Nat.succ m') (Nat.gcdIter (Nat.succ f') (Nat.succ a') b)) (congrArg.{1, 1} Nat Nat (fun (z : Nat) => " +
			"Nat.gcdIter F' z (Nat.mul (Nat.succ m') (Nat.succ a'))) (Nat.mod (Nat.mul (Nat.succ m') b) (Nat.mul (Nat.succ m') (Nat.succ a'))) (Nat.mul " +
			"(Nat.succ m') (Nat.mod b (Nat.succ a'))) (Nat.mul_mod_mul_left (Nat.succ m') (Nat.succ a') b)) (ih (Nat.mod b (Nat.succ a')) (Nat.succ a') " +
			"F' (Nat.le_trans (Nat.mod b (Nat.succ a')) a' f' (Nat.le_of_succ_le_succ (Nat.mod b (Nat.succ a')) a' (Nat.mod_lt b (Nat.succ a') " +
			"(Nat.zero_lt_succ a'))) (Nat.le_of_succ_le_succ a' f' ha)) (Nat.le_of_succ_le_succ f' F' hF'))) a ha) F hF) f"},
	// `gcd` multiplies: `gcd (m * n) (m * k) = m * gcd n k`. Lean proves it by `gcd.induction` on the step of the
	// algorithm; here the countdown's fuel is the induction, and the step is `Nat.mul_mod_mul_left` followed by the
	// hypothesis. The `m = 0` case is separate because `0 * n` does not reduce on its own.
	valDecl{"thm", "Nat.gcd_mul_left", nil,
		"(m : Nat) → (n : Nat) → (k : Nat) → Eq.{1} Nat (Nat.gcd (Nat.mul m n) (Nat.mul m k)) " +
			"(Nat.mul m (Nat.gcd n k))",
		"fun (m : Nat) => Nat.rec.{0} (fun (m : Nat) => (n : Nat) → (k : Nat) → " +
			"Eq.{1} Nat (Nat.gcd (Nat.mul m n) (Nat.mul m k)) (Nat.mul m (Nat.gcd n k))) " +
			// `m = 0`: `gcd 0 0 = 0 = 0 * gcd n k`
			"(fun (n : Nat) (k : Nat) => Eq.trans.{1} Nat (Nat.gcd (Nat.mul 0 n) (Nat.mul 0 k)) " +
			"(Nat.gcd 0 0) (Nat.mul 0 (Nat.gcd n k)) " +
			"(Eq.trans.{1} Nat (Nat.gcd (Nat.mul 0 n) (Nat.mul 0 k)) (Nat.gcd 0 (Nat.mul 0 k)) " +
			"(Nat.gcd 0 0) " +
			"(congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.gcd z (Nat.mul 0 k)) (Nat.mul 0 n) 0 " +
			"(Nat.zero_mul n)) " +
			"(congrArg.{1, 1} Nat Nat (fun (w : Nat) => Nat.gcd 0 w) (Nat.mul 0 k) 0 (Nat.zero_mul k))) " +
			"(Eq.trans.{1} Nat (Nat.gcd 0 0) 0 (Nat.mul 0 (Nat.gcd n k)) (Nat.gcd_zero_left 0) " +
			"(Eq.symm.{1} Nat (Nat.mul 0 (Nat.gcd n k)) 0 (Nat.zero_mul (Nat.gcd n k))))) " +
			// `m = succ m'`: `Nat.gcdIter_mul_left` with fuels `n` and `m * n`
			"(fun (m' : Nat) (_ : (n : Nat) → (k : Nat) → Eq.{1} Nat (Nat.gcd (Nat.mul m' n) (Nat.mul m' k)) (Nat.mul m' (Nat.gcd n k))) (n : Nat) (k : " +
			"Nat) => Nat.gcdIter_mul_left m' n n k (Nat.mul (Nat.succ m') n) (Nat.le.refl n) (Eq.rec.{0, 1} Nat (Nat.add (Nat.mul m' n) n) (fun (z : Nat) " +
			"(_ : Eq.{1} Nat (Nat.add (Nat.mul m' n) n) z) => Nat.le n z) (Nat.le_add_left n (Nat.mul m' n)) (Nat.mul (Nat.succ m') n) (Eq.symm.{1} Nat " +
			"(Nat.mul (Nat.succ m') n) (Nat.add (Nat.mul m' n) n) (Nat.succ_mul m' n)))) m"},
	valDecl{"thm", "Nat.gcd_dvd_gcd_of_dvd_left", nil,
		"(n : Nat) → (m : Nat) → (k : Nat) → Nat.dvd m k → Nat.dvd (Nat.gcd m n) (Nat.gcd k n)",
		"fun (n : Nat) (m : Nat) (k : Nat) (h : Nat.dvd m k) => Nat.dvd_gcd (Nat.gcd m n) k n " +
			"(Nat.dvd_trans (Nat.gcd m n) m k (Nat.gcd_dvd_left m n) h) (Nat.gcd_dvd_right m n)"},
	valDecl{"thm", "Nat.gcd_dvd_gcd_of_dvd_right", nil,
		"(n : Nat) → (m : Nat) → (k : Nat) → Nat.dvd m k → Nat.dvd (Nat.gcd n m) (Nat.gcd n k)",
		"fun (n : Nat) (m : Nat) (k : Nat) (h : Nat.dvd m k) => Nat.dvd_gcd (Nat.gcd n m) n k " +
			"(Nat.gcd_dvd_left n m) (Nat.dvd_trans (Nat.gcd n m) m k (Nat.gcd_dvd_right n m) h)"},
	valDecl{"thm", "Nat.gcd_dvd_gcd_mul_left_left", nil,
		"(m : Nat) → (n : Nat) → (k : Nat) → Nat.dvd (Nat.gcd m n) (Nat.gcd (Nat.mul k m) n)",
		"fun (m : Nat) (n : Nat) (k : Nat) => Nat.gcd_dvd_gcd_of_dvd_left n m (Nat.mul k m) (Nat.dvd_mul_left m k)"},
	valDecl{"thm", "Nat.gcd_dvd_gcd_mul_right_left", nil,
		"(m : Nat) → (n : Nat) → (k : Nat) → Nat.dvd (Nat.gcd m n) (Nat.gcd (Nat.mul m k) n)",
		"fun (m : Nat) (n : Nat) (k : Nat) => Nat.gcd_dvd_gcd_of_dvd_left n m (Nat.mul m k) (Nat.dvd_mul_right m k)"},
	valDecl{"thm", "Nat.gcd_dvd_gcd_mul_left_right", nil,
		"(m : Nat) → (n : Nat) → (k : Nat) → Nat.dvd (Nat.gcd m n) (Nat.gcd m (Nat.mul k n))",
		"fun (m : Nat) (n : Nat) (k : Nat) => Nat.gcd_dvd_gcd_of_dvd_right m n (Nat.mul k n) (Nat.dvd_mul_left n k)"},
	valDecl{"thm", "Nat.gcd_dvd_gcd_mul_right_right", nil,
		"(m : Nat) → (n : Nat) → (k : Nat) → Nat.dvd (Nat.gcd m n) (Nat.gcd m (Nat.mul n k))",
		"fun (m : Nat) (n : Nat) (k : Nat) => Nat.gcd_dvd_gcd_of_dvd_right m n (Nat.mul n k) (Nat.dvd_mul_right n k)"},
	valDecl{"thm", "Nat.gcd_mul_right", nil,
		"(m : Nat) → (n : Nat) → (k : Nat) → Eq.{1} Nat (Nat.gcd (Nat.mul m n) (Nat.mul k n)) " +
			"(Nat.mul (Nat.gcd m k) n)",
		"fun (m : Nat) (n : Nat) (k : Nat) => Eq.trans.{1} Nat (Nat.gcd (Nat.mul m n) (Nat.mul k n)) " +
			"(Nat.gcd (Nat.mul n m) (Nat.mul n k)) (Nat.mul (Nat.gcd m k) n) " +
			"(Eq.trans.{1} Nat (Nat.gcd (Nat.mul m n) (Nat.mul k n)) (Nat.gcd (Nat.mul n m) (Nat.mul k n)) " +
			"(Nat.gcd (Nat.mul n m) (Nat.mul n k)) " +
			"(congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.gcd z (Nat.mul k n)) (Nat.mul m n) " +
			"(Nat.mul n m) (Nat.mul_comm m n)) " +
			"(congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.gcd (Nat.mul n m) z) (Nat.mul k n) " +
			"(Nat.mul n k) (Nat.mul_comm k n))) " +
			"(Eq.trans.{1} Nat (Nat.gcd (Nat.mul n m) (Nat.mul n k)) (Nat.mul n (Nat.gcd m k)) " +
			"(Nat.mul (Nat.gcd m k) n) (Nat.gcd_mul_left n m k) (Nat.mul_comm n (Nat.gcd m k)))"},
	valDecl{"thm", "Nat.gcd_pos_of_pos_left", nil,
		"(m : Nat) → (n : Nat) → Nat.lt 0 m → Nat.lt 0 (Nat.gcd m n)",
		"fun (m : Nat) (n : Nat) (h : Nat.lt 0 m) => " +
			"Nat.pos_of_dvd_of_pos (Nat.gcd m n) m (Nat.gcd_dvd_left m n) h"},
	valDecl{"thm", "Nat.gcd_pos_of_pos_right", nil,
		"(m : Nat) → (n : Nat) → Nat.lt 0 n → Nat.lt 0 (Nat.gcd m n)",
		"fun (m : Nat) (n : Nat) (h : Nat.lt 0 n) => " +
			"Nat.pos_of_dvd_of_pos (Nat.gcd m n) n (Nat.gcd_dvd_right m n) h"},
	// `a * b ∣ a * c` gives `b ∣ c`: divide both sides by `a`, using `a * x / a = x` (which needs `0 < a`).
	valDecl{"thm", "Nat.dvd_of_mul_dvd_mul_left", nil,
		"(a : Nat) → (b : Nat) → (c : Nat) → Nat.lt 0 a → Nat.dvd (Nat.mul a b) (Nat.mul a c) → " +
			"Nat.dvd b c",
		"fun (a : Nat) (b : Nat) (c : Nat) (ha : Nat.lt 0 a) " +
			"(h : Nat.dvd (Nat.mul a b) (Nat.mul a c)) => Exists.rec.{1} Nat " +
			"(fun (d : Nat) => Eq.{1} Nat (Nat.mul a c) (Nat.mul (Nat.mul a b) d)) " +
			"(fun (_ : Nat.dvd (Nat.mul a b) (Nat.mul a c)) => Nat.dvd b c) " +
			"(fun (d : Nat) (e : Eq.{1} Nat (Nat.mul a c) (Nat.mul (Nat.mul a b) d)) => " +
			"Exists.intro.{1} Nat (fun (w : Nat) => Eq.{1} Nat c (Nat.mul b w)) d " +
			"(Eq.trans.{1} Nat c (Nat.div (Nat.mul a c) a) (Nat.mul b d) " +
			"(Eq.symm.{1} Nat (Nat.div (Nat.mul a c) a) c (Nat.mul_div_cancel_left c a ha)) " +
			"(Eq.trans.{1} Nat (Nat.div (Nat.mul a c) a) (Nat.div (Nat.mul a (Nat.mul b d)) a) " +
			"(Nat.mul b d) " +
			"(congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.div z a) (Nat.mul a c) " +
			"(Nat.mul a (Nat.mul b d)) (Eq.trans.{1} Nat (Nat.mul a c) (Nat.mul (Nat.mul a b) d) " +
			"(Nat.mul a (Nat.mul b d)) e (Nat.mul_assoc a b d))) " +
			"(Nat.mul_div_cancel_left (Nat.mul b d) a ha)))) h"},
	// `k ∣ m * n` splits as `k₁ * k₂` with `k₁ ∣ m`, `k₂ ∣ n` (Lean's `dvdProdDvdOfDvdProd`, here as a statement
	// instead of a pair). The witness is `gcd k m` and `k / gcd k m`; `gcd k m = 0` forces `k = m = 0`.
	valDecl{"thm", "Nat.div_gcd_dvd_of_dvd_mul", nil,
		"(k : Nat) → (m : Nat) → (n : Nat) → Nat.lt 0 (Nat.gcd k m) → Nat.dvd k (Nat.mul m n) → " +
			"Nat.dvd (Nat.div k (Nat.gcd k m)) n",
		"fun (k : Nat) (m : Nat) (n : Nat) (hp : Nat.lt 0 (Nat.gcd k m)) (h : Nat.dvd k (Nat.mul m n)) => " +
			"Nat.dvd_of_mul_dvd_mul_left (Nat.gcd k m) (Nat.div k (Nat.gcd k m)) n hp " +
			"(Eq.rec.{0, 1} Nat k (fun (z : Nat) (_ : Eq.{1} Nat k z) => Nat.dvd z (Nat.mul (Nat.gcd k m) n)) " +
			"(Eq.rec.{0, 1} Nat (Nat.gcd (Nat.mul k n) (Nat.mul m n)) " +
			"(fun (z : Nat) (_ : Eq.{1} Nat (Nat.gcd (Nat.mul k n) (Nat.mul m n)) z) => Nat.dvd k z) " +
			"(Nat.dvd_gcd k (Nat.mul k n) (Nat.mul m n) (Nat.dvd_mul_right k n) h) " +
			"(Nat.mul (Nat.gcd k m) n) (Nat.gcd_mul_right k n m)) " +
			"(Nat.mul (Nat.gcd k m) (Nat.div k (Nat.gcd k m))) " +
			"(Eq.symm.{1} Nat (Nat.mul (Nat.gcd k m) (Nat.div k (Nat.gcd k m))) k " +
			"(Nat.mul_div_cancel' (Nat.gcd k m) k (Nat.gcd_dvd_left k m))))"},
	valDecl{"thm", "Nat.dvdProd_exists", nil,
		"(k : Nat) → (m : Nat) → (n : Nat) → Nat.dvd k (Nat.mul m n) → " + dvdProdR,
		"fun (k : Nat) (m : Nat) (n : Nat) (h : Nat.dvd k (Nat.mul m n)) => " +
			"Nat.rec.{0} (fun (g : Nat) => Eq.{1} Nat (Nat.gcd k m) g → " + dvdProdR + ") " +
			// `gcd k m = 0`: `k = 0`, `m = 0`; take `0` and `n`
			"(fun (hg : Eq.{1} Nat (Nat.gcd k m) 0) => Exists.intro.{1} Nat " + dvdProdF + " 0 " +
			"(Exists.intro.{1} Nat (fun (b : Nat) => And (And (Nat.dvd 0 m) (Nat.dvd b n)) (Eq.{1} Nat (Nat.mul 0 b) k)) n " +
			"(And.intro (And (Nat.dvd 0 m) (Nat.dvd n n)) (Eq.{1} Nat (Nat.mul 0 n) k) " +
			"(And.intro (Nat.dvd 0 m) (Nat.dvd n n) " +
			"(Eq.rec.{0, 1} Nat 0 (fun (z : Nat) (_ : Eq.{1} Nat 0 z) => Nat.dvd 0 z) (Nat.dvd_refl 0) m " +
			"(Eq.symm.{1} Nat m 0 (Nat.eq_zero_of_zero_dvd m (Eq.rec.{0, 1} Nat (Nat.gcd k m) " +
			"(fun (z : Nat) (_ : Eq.{1} Nat (Nat.gcd k m) z) => Nat.dvd z m) (Nat.gcd_dvd_right k m) 0 hg)))) " +
			"(Nat.dvd_refl n)) " +
			"(Eq.trans.{1} Nat (Nat.mul 0 n) 0 k (Nat.zero_mul n) (Eq.symm.{1} Nat k 0 " +
			"(Nat.eq_zero_of_zero_dvd k (Eq.rec.{0, 1} Nat (Nat.gcd k m) " +
			"(fun (z : Nat) (_ : Eq.{1} Nat (Nat.gcd k m) z) => Nat.dvd z k) (Nat.gcd_dvd_left k m) 0 hg))))))) " +
			// `gcd k m = succ g'`: take `gcd k m` and `k / gcd k m`
			"(fun (g' : Nat) (_ : Eq.{1} Nat (Nat.gcd k m) g' → " + dvdProdR + ") " +
			"(hg : Eq.{1} Nat (Nat.gcd k m) (Nat.succ g')) => Exists.intro.{1} Nat " + dvdProdF + " (Nat.gcd k m) " +
			"(Exists.intro.{1} Nat (fun (b : Nat) => And (And (Nat.dvd (Nat.gcd k m) m) (Nat.dvd b n)) " +
			"(Eq.{1} Nat (Nat.mul (Nat.gcd k m) b) k)) (Nat.div k (Nat.gcd k m)) " +
			"(And.intro (And (Nat.dvd (Nat.gcd k m) m) (Nat.dvd (Nat.div k (Nat.gcd k m)) n)) " +
			"(Eq.{1} Nat (Nat.mul (Nat.gcd k m) (Nat.div k (Nat.gcd k m))) k) " +
			"(And.intro (Nat.dvd (Nat.gcd k m) m) (Nat.dvd (Nat.div k (Nat.gcd k m)) n) (Nat.gcd_dvd_right k m) " +
			"(Nat.div_gcd_dvd_of_dvd_mul k m n (Eq.rec.{0, 1} Nat (Nat.succ g') " +
			"(fun (z : Nat) (_ : Eq.{1} Nat (Nat.succ g') z) => Nat.lt 0 z) (Nat.zero_lt_succ g') (Nat.gcd k m) " +
			"(Eq.symm.{1} Nat (Nat.gcd k m) (Nat.succ g') hg)) h)) " +
			"(Nat.mul_div_cancel' (Nat.gcd k m) k (Nat.gcd_dvd_left k m))))) " +
			"(Nat.gcd k m) (Eq.refl.{1} Nat (Nat.gcd k m))"},
	valDecl{"thm", "Nat.dvd_mul", nil,
		"(k : Nat) → (m : Nat) → (n : Nat) → Iff (Nat.dvd k (Nat.mul m n)) (" + dvdProdRK + ")",
		"fun (k : Nat) (m : Nat) (n : Nat) => Iff.intro (Nat.dvd k (Nat.mul m n)) (" + dvdProdRK + ") " +
			"(Nat.dvdProd_exists k m n) " +
			"(fun (w : " + dvdProdRK + ") => Exists.rec.{1} Nat " + dvdProdF + " " +
			"(fun (_ : " + dvdProdRK + ") => Nat.dvd k (Nat.mul m n)) " +
			"(fun (a : Nat) (ha : Exists.{1} Nat (fun (b : Nat) => And (And (Nat.dvd a m) (Nat.dvd b n)) " +
			"(Eq.{1} Nat (Nat.mul a b) k))) => Exists.rec.{1} Nat " +
			"(fun (b : Nat) => And (And (Nat.dvd a m) (Nat.dvd b n)) (Eq.{1} Nat (Nat.mul a b) k)) " +
			"(fun (_ : Exists.{1} Nat (fun (b : Nat) => And (And (Nat.dvd a m) (Nat.dvd b n)) " +
			"(Eq.{1} Nat (Nat.mul a b) k))) => Nat.dvd k (Nat.mul m n)) " +
			"(fun (b : Nat) (hb : And (And (Nat.dvd a m) (Nat.dvd b n)) (Eq.{1} Nat (Nat.mul a b) k)) => " +
			"Eq.rec.{0, 1} Nat (Nat.mul a b) (fun (z : Nat) (_ : Eq.{1} Nat (Nat.mul a b) z) => " +
			"Nat.dvd z (Nat.mul m n)) (Nat.mul_dvd_mul a m b n " +
			"(And.left (Nat.dvd a m) (Nat.dvd b n) (And.left (And (Nat.dvd a m) (Nat.dvd b n)) " +
			"(Eq.{1} Nat (Nat.mul a b) k) hb)) " +
			"(And.right (Nat.dvd a m) (Nat.dvd b n) (And.left (And (Nat.dvd a m) (Nat.dvd b n)) " +
			"(Eq.{1} Nat (Nat.mul a b) k) hb))) k " +
			"(And.right (And (Nat.dvd a m) (Nat.dvd b n)) (Eq.{1} Nat (Nat.mul a b) k) hb)) ha) w)"},
	valDecl{"thm", "Nat.gcd_mul_right_dvd_mul_gcd", nil,
		"(k : Nat) → (m : Nat) → (n : Nat) → Nat.dvd (Nat.gcd k (Nat.mul m n)) (Nat.mul (Nat.gcd k m) (Nat.gcd k n))",
		"fun (k : Nat) (m : Nat) (n : Nat) => Exists.rec.{1} Nat " +
			"(fun (a : Nat) => Exists.{1} Nat (fun (b : Nat) => And (And (Nat.dvd a m) (Nat.dvd b n)) " +
			"(Eq.{1} Nat (Nat.mul a b) (Nat.gcd k (Nat.mul m n))))) " +
			"(fun (_ : " + dvdProdRG + ") => Nat.dvd (Nat.gcd k (Nat.mul m n)) (Nat.mul (Nat.gcd k m) (Nat.gcd k n))) " +
			"(fun (a : Nat) (ha : Exists.{1} Nat (fun (b : Nat) => And (And (Nat.dvd a m) (Nat.dvd b n)) " +
			"(Eq.{1} Nat (Nat.mul a b) (Nat.gcd k (Nat.mul m n))))) => Exists.rec.{1} Nat " +
			"(fun (b : Nat) => And (And (Nat.dvd a m) (Nat.dvd b n)) (Eq.{1} Nat (Nat.mul a b) (Nat.gcd k (Nat.mul m n)))) " +
			"(fun (_ : Exists.{1} Nat (fun (b : Nat) => And (And (Nat.dvd a m) (Nat.dvd b n)) " +
			"(Eq.{1} Nat (Nat.mul a b) (Nat.gcd k (Nat.mul m n)))))  => " +
			"Nat.dvd (Nat.gcd k (Nat.mul m n)) (Nat.mul (Nat.gcd k m) (Nat.gcd k n))) " +
			"(fun (b : Nat) (hb : And (And (Nat.dvd a m) (Nat.dvd b n)) (Eq.{1} Nat (Nat.mul a b) (Nat.gcd k (Nat.mul m n)))) => " +
			"Eq.rec.{0, 1} Nat (Nat.mul a b) (fun (z : Nat) (_ : Eq.{1} Nat (Nat.mul a b) z) => " +
			"Nat.dvd z (Nat.mul (Nat.gcd k m) (Nat.gcd k n))) " +
			"(Nat.mul_dvd_mul a (Nat.gcd k m) b (Nat.gcd k n) " +
			"(Nat.dvd_gcd a k m (Nat.dvd_trans a (Nat.mul a b) k (Nat.dvd_mul_right a b) " +
			"(Eq.rec.{0, 1} Nat (Nat.gcd k (Nat.mul m n)) (fun (z : Nat) (_ : Eq.{1} Nat (Nat.gcd k (Nat.mul m n)) z) => " +
			"Nat.dvd z k) (Nat.gcd_dvd_left k (Nat.mul m n)) (Nat.mul a b) " +
			"(Eq.symm.{1} Nat (Nat.mul a b) (Nat.gcd k (Nat.mul m n)) " +
			"(And.right (And (Nat.dvd a m) (Nat.dvd b n)) (Eq.{1} Nat (Nat.mul a b) (Nat.gcd k (Nat.mul m n))) hb)))) " +
			"(And.left (Nat.dvd a m) (Nat.dvd b n) (And.left (And (Nat.dvd a m) (Nat.dvd b n)) " +
			"(Eq.{1} Nat (Nat.mul a b) (Nat.gcd k (Nat.mul m n))) hb))) " +
			"(Nat.dvd_gcd b k n (Nat.dvd_trans b (Nat.mul a b) k (Nat.dvd_mul_left b a) " +
			"(Eq.rec.{0, 1} Nat (Nat.gcd k (Nat.mul m n)) (fun (z : Nat) (_ : Eq.{1} Nat (Nat.gcd k (Nat.mul m n)) z) => " +
			"Nat.dvd z k) (Nat.gcd_dvd_left k (Nat.mul m n)) (Nat.mul a b) " +
			"(Eq.symm.{1} Nat (Nat.mul a b) (Nat.gcd k (Nat.mul m n)) " +
			"(And.right (And (Nat.dvd a m) (Nat.dvd b n)) (Eq.{1} Nat (Nat.mul a b) (Nat.gcd k (Nat.mul m n))) hb)))) " +
			"(And.right (Nat.dvd a m) (Nat.dvd b n) (And.left (And (Nat.dvd a m) (Nat.dvd b n)) " +
			"(Eq.{1} Nat (Nat.mul a b) (Nat.gcd k (Nat.mul m n))) hb)))) (Nat.gcd k (Nat.mul m n)) " +
			"(And.right (And (Nat.dvd a m) (Nat.dvd b n)) (Eq.{1} Nat (Nat.mul a b) (Nat.gcd k (Nat.mul m n))) hb)) ha) " +
			"(Nat.dvdProd_exists (Nat.gcd k (Nat.mul m n)) m n (Nat.gcd_dvd_right k (Nat.mul m n)))"},
	valDecl{"thm", "Nat.gcd_one_right", nil, "(n : Nat) → Eq.{1} Nat (Nat.gcd n 1) 1",
		"fun (n : Nat) => Nat.dvd_antisymm (Nat.gcd n 1) 1 (Nat.gcd_dvd_right n 1) " +
			"(Exists.intro.{1} Nat (fun (c : Nat) => Eq.{1} Nat (Nat.gcd n 1) (Nat.mul 1 c)) " +
			"(Nat.gcd n 1) (Eq.symm.{1} Nat (Nat.mul 1 (Nat.gcd n 1)) (Nat.gcd n 1) " +
			"(Nat.one_mul (Nat.gcd n 1))))"},
	valDecl{"thm", "Nat.gcd_self", nil, "(n : Nat) → Eq.{1} Nat (Nat.gcd n n) n",
		"fun (n : Nat) => Nat.dvd_antisymm (Nat.gcd n n) n (Nat.gcd_dvd_left n n) " +
			"(Nat.dvd_gcd n n n (Nat.dvd_refl n) (Nat.dvd_refl n))"},
	valDecl{"thm", "Nat.gcd_eq_left", nil,
		"(m : Nat) → (n : Nat) → Nat.dvd m n → Eq.{1} Nat (Nat.gcd m n) m",
		"fun (m : Nat) (n : Nat) (h : Nat.dvd m n) => Nat.dvd_antisymm (Nat.gcd m n) m " +
			"(Nat.gcd_dvd_left m n) (Nat.dvd_gcd m m n (Nat.dvd_refl m) h)"},
	// `n ∣ 1` forces `n = 1`: `1` is positive, so `n ≤ 1`, and `n = 0` is ruled out by `0 ∣ 1`.
	valDecl{"thm", "Nat.eq_one_of_dvd_one", nil, "(n : Nat) → Nat.dvd n 1 → Eq.{1} Nat n 1",
		"fun (n : Nat) => Nat.rec.{0} (fun (n : Nat) => Nat.dvd n 1 → Eq.{1} Nat n 1) " +
			"(fun (h0 : Nat.dvd 0 1) => False.elim.{0} (Eq.{1} Nat 0 1) " +
			"(Nat.succ_ne_zero 0 (Nat.eq_zero_of_zero_dvd 1 h0))) " +
			"(fun (n' : Nat) (_ : Nat.dvd n' 1 → Eq.{1} Nat n' 1) (h : Nat.dvd (Nat.succ n') 1) => " +
			"congrArg.{1, 1} Nat Nat Nat.succ n' 0 (Nat.le_zero n' (Nat.le_of_succ_le_succ n' 0 " +
			"(Nat.le_of_dvd (Nat.succ n') 1 (Nat.zero_lt_succ 0) h)))) n"},

	// `Nat.lcm` is a definition here (`m * n / gcd m n`, as in Lean), so its lemmas are about `*`, `gcd` and `/`.
	valDecl{"thm", "Nat.lcm_eq_mul_div", nil,
		"(m : Nat) → (n : Nat) → Eq.{1} Nat (Nat.lcm m n) (Nat.div (Nat.mul m n) (Nat.gcd m n))",
		"fun (m : Nat) (n : Nat) => Eq.refl.{1} Nat (Nat.div (Nat.mul m n) (Nat.gcd m n))"},
	valDecl{"thm", "Nat.gcd_mul_lcm", nil,
		"(m : Nat) → (n : Nat) → Eq.{1} Nat (Nat.mul (Nat.gcd m n) (Nat.lcm m n)) (Nat.mul m n)",
		"fun (m : Nat) (n : Nat) => Nat.mul_div_cancel' (Nat.gcd m n) (Nat.mul m n) " +
			"(Nat.dvd_mul_right_of_dvd (Nat.gcd m n) m n (Nat.gcd_dvd_left m n))"},
	valDecl{"thm", "Nat.lcm_mul_gcd", nil,
		"(m : Nat) → (n : Nat) → Eq.{1} Nat (Nat.mul (Nat.lcm m n) (Nat.gcd m n)) (Nat.mul m n)",
		"fun (m : Nat) (n : Nat) => Eq.trans.{1} Nat (Nat.mul (Nat.lcm m n) (Nat.gcd m n)) " +
			"(Nat.mul (Nat.gcd m n) (Nat.lcm m n)) (Nat.mul m n) " +
			"(Nat.mul_comm (Nat.lcm m n) (Nat.gcd m n)) (Nat.gcd_mul_lcm m n)"},
	valDecl{"thm", "Nat.lcm_dvd_mul", nil,
		"(m : Nat) → (n : Nat) → Nat.dvd (Nat.lcm m n) (Nat.mul m n)",
		"fun (m : Nat) (n : Nat) => Exists.intro.{1} Nat " +
			"(fun (c : Nat) => Eq.{1} Nat (Nat.mul m n) (Nat.mul (Nat.lcm m n) c)) (Nat.gcd m n) " +
			"(Eq.symm.{1} Nat (Nat.mul (Nat.lcm m n) (Nat.gcd m n)) (Nat.mul m n) (Nat.lcm_mul_gcd m n))"},
	valDecl{"thm", "Nat.gcd_dvd_mul", nil,
		"(m : Nat) → (n : Nat) → Nat.dvd (Nat.gcd m n) (Nat.mul m n)",
		"fun (m : Nat) (n : Nat) => Exists.intro.{1} Nat " +
			"(fun (c : Nat) => Eq.{1} Nat (Nat.mul m n) (Nat.mul (Nat.gcd m n) c)) (Nat.lcm m n) " +
			"(Eq.symm.{1} Nat (Nat.mul (Nat.gcd m n) (Nat.lcm m n)) (Nat.mul m n) (Nat.gcd_mul_lcm m n))"},
	valDecl{"thm", "Nat.lcm_zero_left", nil, "(m : Nat) → Eq.{1} Nat (Nat.lcm 0 m) 0",
		"fun (m : Nat) => congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.div z (Nat.gcd 0 m)) " +
			"(Nat.mul 0 m) 0 (Nat.zero_mul m)"},
	valDecl{"thm", "Nat.lcm_zero_right", nil, "(m : Nat) → Eq.{1} Nat (Nat.lcm m 0) 0",
		"fun (m : Nat) => Eq.refl.{1} Nat 0"},
	valDecl{"thm", "Nat.lcm_comm", nil,
		"(m : Nat) → (n : Nat) → Eq.{1} Nat (Nat.lcm m n) (Nat.lcm n m)",
		"fun (m : Nat) (n : Nat) => Eq.trans.{1} Nat (Nat.lcm m n) (Nat.div (Nat.mul n m) (Nat.gcd m n)) " +
			"(Nat.lcm n m) " +
			"(congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.div z (Nat.gcd m n)) (Nat.mul m n) " +
			"(Nat.mul n m) (Nat.mul_comm m n)) " +
			"(congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.div (Nat.mul n m) z) (Nat.gcd m n) " +
			"(Nat.gcd n m) (Nat.gcd_comm m n))"},
	valDecl{"thm", "Nat.dvd_lcm_left", nil, "(m : Nat) → (n : Nat) → Nat.dvd m (Nat.lcm m n)",
		"fun (m : Nat) (n : Nat) => Exists.intro.{1} Nat " +
			"(fun (c : Nat) => Eq.{1} Nat (Nat.lcm m n) (Nat.mul m c)) (Nat.div n (Nat.gcd m n)) " +
			"(Nat.mul_div_assoc m n (Nat.gcd m n) (Nat.gcd_dvd_right m n))"},
	valDecl{"thm", "Nat.dvd_lcm_right", nil, "(m : Nat) → (n : Nat) → Nat.dvd n (Nat.lcm m n)",
		"fun (m : Nat) (n : Nat) => Eq.rec.{0, 1} Nat (Nat.lcm n m) (fun (z : Nat) " +
			"(_ : Eq.{1} Nat (Nat.lcm n m) z) => Nat.dvd n z) (Nat.dvd_lcm_left n m) (Nat.lcm m n) " +
			"(Nat.lcm_comm n m)"},
	// `lcm m n` divides every common multiple: multiply the goal by `gcd m n`, where `gcd_mul_lcm` turns the left
	// side into `m * n` and `gcd_mul_right` turns the right side into a `gcd` of two multiples of it.
	valDecl{"thm", "Nat.lcm_dvd", nil,
		"(m : Nat) → (n : Nat) → (k : Nat) → Nat.dvd m k → Nat.dvd n k → Nat.dvd (Nat.lcm m n) k",
		"fun (m : Nat) (n : Nat) (k : Nat) (h1 : Nat.dvd m k) (h2 : Nat.dvd n k) => Nat.rec.{0} " +
			"(fun (k : Nat) => Nat.dvd m k → Nat.dvd n k → Nat.dvd (Nat.lcm m n) k) " +
			// `k = 0`: `lcm m n * 0 = 0`
			"(fun (_ : Nat.dvd m 0) (_ : Nat.dvd n 0) => Exists.intro.{1} Nat " +
			"(fun (c : Nat) => Eq.{1} Nat 0 (Nat.mul (Nat.lcm m n) c)) 0 (Eq.refl.{1} Nat 0)) " +
			// `k = succ k'`: `gcd m n * lcm m n = m * n` and `gcd m n * k = gcd (m * k) (n * k)`
			"(fun (k' : Nat) (_ : Nat.dvd m k' → Nat.dvd n k' → Nat.dvd (Nat.lcm m n) k') " +
			"(h1' : Nat.dvd m (Nat.succ k')) (h2' : Nat.dvd n (Nat.succ k')) => " +
			"Nat.dvd_of_mul_dvd_mul_left (Nat.gcd m n) (Nat.lcm m n) (Nat.succ k') " +
			"(Nat.gcd_pos_of_pos_left m n (Nat.pos_of_dvd_of_pos m (Nat.succ k') h1' " +
			"(Nat.zero_lt_succ k'))) " +
			"(Eq.rec.{0, 1} Nat (Nat.mul m n) (fun (z : Nat) " +
			"(_ : Eq.{1} Nat (Nat.mul m n) z) => Nat.dvd z (Nat.mul (Nat.gcd m n) (Nat.succ k'))) " +
			"(Eq.rec.{0, 1} Nat (Nat.gcd (Nat.mul m (Nat.succ k')) (Nat.mul n (Nat.succ k'))) " +
			"(fun (w : Nat) (_ : Eq.{1} Nat (Nat.gcd (Nat.mul m (Nat.succ k')) " +
			"(Nat.mul n (Nat.succ k'))) w) => Nat.dvd (Nat.mul m n) w) " +
			"(Nat.dvd_gcd (Nat.mul m n) (Nat.mul m (Nat.succ k')) (Nat.mul n (Nat.succ k')) " +
			"(Nat.mul_dvd_mul_left m n (Nat.succ k') h2') " +
			"(Eq.rec.{0, 1} Nat (Nat.mul n m) (fun (z : Nat) " +
			"(_ : Eq.{1} Nat (Nat.mul n m) z) => Nat.dvd z (Nat.mul n (Nat.succ k'))) " +
			"(Nat.mul_dvd_mul_left n m (Nat.succ k') h1') (Nat.mul m n) " +
			"(Eq.symm.{1} Nat (Nat.mul m n) (Nat.mul n m) (Nat.mul_comm m n)))) " +
			"(Nat.mul (Nat.gcd m n) (Nat.succ k')) (Nat.gcd_mul_right m (Nat.succ k') n)) " +
			"(Nat.mul (Nat.gcd m n) (Nat.lcm m n)) (Eq.symm.{1} Nat (Nat.mul (Nat.gcd m n) (Nat.lcm m n)) " +
			"(Nat.mul m n) (Nat.gcd_mul_lcm m n)))) k h1 h2"},
	valDecl{"thm", "Nat.lcm_dvd_iff", nil,
		"(m : Nat) → (n : Nat) → (k : Nat) → Iff (Nat.dvd (Nat.lcm m n) k) (And (Nat.dvd m k) (Nat.dvd n k))",
		"fun (m : Nat) (n : Nat) (k : Nat) => Iff.intro (Nat.dvd (Nat.lcm m n) k) (And (Nat.dvd m k) (Nat.dvd n k)) " +
			"(fun (h : Nat.dvd (Nat.lcm m n) k) => And.intro (Nat.dvd m k) (Nat.dvd n k) " +
			"(Nat.dvd_trans m (Nat.lcm m n) k (Nat.dvd_lcm_left m n) h) " +
			"(Nat.dvd_trans n (Nat.lcm m n) k (Nat.dvd_lcm_right m n) h)) " +
			"(fun (h : And (Nat.dvd m k) (Nat.dvd n k)) => Nat.lcm_dvd m n k " +
			"(And.left (Nat.dvd m k) (Nat.dvd n k) h) (And.right (Nat.dvd m k) (Nat.dvd n k) h))"},
	valDecl{"thm", "Nat.lcm_dvd_lcm_of_dvd_left", nil,
		"(n : Nat) → (m : Nat) → (k : Nat) → Nat.dvd m k → Nat.dvd (Nat.lcm m n) (Nat.lcm k n)",
		"fun (n : Nat) (m : Nat) (k : Nat) (h : Nat.dvd m k) => Nat.lcm_dvd m n (Nat.lcm k n) " +
			"(Nat.dvd_trans m k (Nat.lcm k n) h (Nat.dvd_lcm_left k n)) (Nat.dvd_lcm_right k n)"},
	valDecl{"thm", "Nat.lcm_dvd_lcm_of_dvd_right", nil,
		"(n : Nat) → (m : Nat) → (k : Nat) → Nat.dvd m k → Nat.dvd (Nat.lcm n m) (Nat.lcm n k)",
		"fun (n : Nat) (m : Nat) (k : Nat) (h : Nat.dvd m k) => Nat.lcm_dvd n m (Nat.lcm n k) " +
			"(Nat.dvd_lcm_left n k) (Nat.dvd_trans m k (Nat.lcm n k) h (Nat.dvd_lcm_right n k))"},
	valDecl{"thm", "Nat.lcm_assoc", nil,
		"(m : Nat) → (n : Nat) → (k : Nat) → Eq.{1} Nat (Nat.lcm (Nat.lcm m n) k) " +
			"(Nat.lcm m (Nat.lcm n k))",
		"fun (m : Nat) (n : Nat) (k : Nat) => Nat.dvd_antisymm (Nat.lcm (Nat.lcm m n) k) " +
			"(Nat.lcm m (Nat.lcm n k)) " +
			// `lcm (lcm m n) k ∣ lcm m (lcm n k)`: both `lcm m n` and `k` divide the right side
			"(Nat.lcm_dvd (Nat.lcm m n) k (Nat.lcm m (Nat.lcm n k)) " +
			"(Nat.lcm_dvd m n (Nat.lcm m (Nat.lcm n k)) (Nat.dvd_lcm_left m (Nat.lcm n k)) " +
			"(Nat.dvd_trans n (Nat.lcm n k) (Nat.lcm m (Nat.lcm n k)) (Nat.dvd_lcm_left n k) " +
			"(Nat.dvd_lcm_right m (Nat.lcm n k)))) " +
			"(Nat.dvd_trans k (Nat.lcm n k) (Nat.lcm m (Nat.lcm n k)) (Nat.dvd_lcm_right n k) " +
			"(Nat.dvd_lcm_right m (Nat.lcm n k)))) " +
			// and conversely, both `m` and `lcm n k` divide the left side
			"(Nat.lcm_dvd m (Nat.lcm n k) (Nat.lcm (Nat.lcm m n) k) " +
			"(Nat.dvd_trans m (Nat.lcm m n) (Nat.lcm (Nat.lcm m n) k) (Nat.dvd_lcm_left m n) " +
			"(Nat.dvd_lcm_left (Nat.lcm m n) k)) " +
			"(Nat.lcm_dvd n k (Nat.lcm (Nat.lcm m n) k) " +
			"(Nat.dvd_trans n (Nat.lcm m n) (Nat.lcm (Nat.lcm m n) k) (Nat.dvd_lcm_right m n) " +
			"(Nat.dvd_lcm_left (Nat.lcm m n) k)) " +
			"(Nat.dvd_lcm_right (Nat.lcm m n) k)))"},

	valDecl{"def", "Decidable.decide", nil, "(p : Prop) → Decidable p → Bool",
		"fun (p : Prop) (h : Decidable p) => Decidable.rec.{1} p (fun (_ : Decidable p) => Bool) (fun (_ : Not p) => Bool.false) (fun (_ : p) => Bool.true) h"},
	valDecl{"thm", "of_decide_eq_true", nil, "(p : Prop) → (inst : Decidable p) → Eq.{1} Bool (Decidable.decide p inst) Bool.true → p",
		"fun (p : Prop) (inst : Decidable p) => Decidable.rec.{0} p (fun (d : Decidable p) => Eq.{1} Bool (Decidable.decide p d) Bool.true → p) " +
			"(fun (h1 : Not p) (h2 : Eq.{1} Bool (Decidable.decide p (Decidable.isFalse p h1)) Bool.true) => False.elim.{0} p (Bool.false_ne_true h2)) " +
			"(fun (h1 : p) (h2 : Eq.{1} Bool (Decidable.decide p (Decidable.isTrue p h1)) Bool.true) => h1) inst"},

	valDecl{"def", "Nat.decEq", nil, "(n : Nat) → (m : Nat) → Decidable (Eq.{1} Nat n m)",
		"fun (n : Nat) (m : Nat) => Bool.rec.{1} (fun (b : Bool) => Eq.{1} Bool (Nat.beq n m) b → Decidable (Eq.{1} Nat n m)) " +
			"(fun (h : Eq.{1} Bool (Nat.beq n m) Bool.false) => Decidable.isFalse (Eq.{1} Nat n m) (Nat.ne_of_beq_eq_false n m h)) " +
			"(fun (h : Eq.{1} Bool (Nat.beq n m) Bool.true) => Decidable.isTrue (Eq.{1} Nat n m) (Nat.eq_of_beq_eq_true n m h)) " +
			"(Nat.beq n m) (Eq.refl.{1} Bool (Nat.beq n m))"},
	valDecl{"def", "Nat.decLe", nil, "(n : Nat) → (m : Nat) → Decidable (Nat.le n m)",
		"fun (n : Nat) (m : Nat) => Bool.rec.{1} (fun (b : Bool) => Eq.{1} Bool (Nat.ble n m) b → Decidable (Nat.le n m)) " +
			"(fun (h : Eq.{1} Bool (Nat.ble n m) Bool.false) => Decidable.isFalse (Nat.le n m) (Nat.not_le_of_not_ble_eq_true n m (fun (e : Eq.{1} Bool (Nat.ble n m) Bool.true) => Bool.false_ne_true (Eq.trans.{1} Bool Bool.false (Nat.ble n m) Bool.true (Eq.symm.{1} Bool (Nat.ble n m) Bool.false h) e)))) " +
			"(fun (h : Eq.{1} Bool (Nat.ble n m) Bool.true) => Decidable.isTrue (Nat.le n m) (Nat.le_of_ble_eq_true n m h)) " +
			"(Nat.ble n m) (Eq.refl.{1} Bool (Nat.ble n m))"},
	valDecl{"def", "Nat.decLt", nil, "(n : Nat) → (m : Nat) → Decidable (Nat.lt n m)",
		"fun (n : Nat) (m : Nat) => Nat.decLe (Nat.succ n) m"},
	// Order lemmas under Lean's names (`Init/Data/Nat/Basic.lean`).
	valDecl{"thm", "Nat.lt_irrefl", nil, "(n : Nat) → Not (Nat.lt n n)", "fun (n : Nat) => Nat.not_le_succ_self n"},
	valDecl{"thm", "Nat.le_succ", nil, "(n : Nat) → Nat.le n (Nat.succ n)", "fun (n : Nat) => Nat.le.step n n (Nat.le.refl n)"},
	valDecl{"thm", "Nat.lt_succ_self", nil, "(n : Nat) → Nat.lt n (Nat.succ n)", "fun (n : Nat) => Nat.le.refl (Nat.succ n)"},
	valDecl{"thm", "Nat.le_of_lt", nil, "(n : Nat) → (m : Nat) → Nat.lt n m → Nat.le n m",
		"fun (n : Nat) (m : Nat) (h : Nat.lt n m) => Nat.le_of_succ_le n m h"},
	valDecl{"thm", "Nat.succ_le_of_lt", nil, "(n : Nat) → (m : Nat) → Nat.lt n m → Nat.le (Nat.succ n) m",
		"fun (n : Nat) (m : Nat) (h : Nat.lt n m) => h"},
	valDecl{"thm", "Nat.lt_of_succ_le", nil, "(n : Nat) → (m : Nat) → Nat.le (Nat.succ n) m → Nat.lt n m",
		"fun (n : Nat) (m : Nat) (h : Nat.le (Nat.succ n) m) => h"},
	valDecl{"thm", "Nat.lt_of_le_of_lt", nil, "(a : Nat) → (b : Nat) → (c : Nat) → Nat.le a b → Nat.lt b c → Nat.lt a c",
		"fun (a : Nat) (b : Nat) (c : Nat) (h1 : Nat.le a b) (h2 : Nat.lt b c) => " +
			"Nat.le_trans (Nat.succ a) (Nat.succ b) c (Nat.succ_le_succ a b h1) h2"},
	valDecl{"thm", "Nat.lt_trans", nil, "(a : Nat) → (b : Nat) → (c : Nat) → Nat.lt a b → Nat.lt b c → Nat.lt a c",
		"fun (a : Nat) (b : Nat) (c : Nat) (h1 : Nat.lt a b) (h2 : Nat.lt b c) => Nat.lt_of_lt_of_le a b c h1 (Nat.le_of_lt b c h2)"},
	valDecl{"thm", "Nat.pos_of_ne_zero", nil, "(n : Nat) → Not (Eq.{1} Nat n 0) → Nat.lt 0 n",
		"fun (n : Nat) => Nat.rec.{0} (fun (n : Nat) => Not (Eq.{1} Nat n 0) → Nat.lt 0 n) " +
			"(fun (h : Not (Eq.{1} Nat 0 0)) => False.elim.{0} (Nat.lt 0 0) (h (Eq.refl.{1} Nat 0))) " +
			"(fun (k : Nat) (_ : Not (Eq.{1} Nat k 0) → Nat.lt 0 k) (_ : Not (Eq.{1} Nat (Nat.succ k) 0)) => Nat.zero_lt_succ k) n"},
	valDecl{"thm", "Nat.le_total", nil, "(n : Nat) → (m : Nat) → Or (Nat.le n m) (Nat.le m n)",
		"fun (n : Nat) => Nat.rec.{0} (fun (n : Nat) => (m : Nat) → Or (Nat.le n m) (Nat.le m n)) " +
			"(fun (m : Nat) => Or.inl (Nat.le 0 m) (Nat.le m 0) (Nat.zero_le m)) " +
			"(fun (k : Nat) (ih : (m : Nat) → Or (Nat.le k m) (Nat.le m k)) (m : Nat) => " +
			"Nat.rec.{0} (fun (m : Nat) => Or (Nat.le (Nat.succ k) m) (Nat.le m (Nat.succ k))) " +
			"(Or.inr (Nat.le (Nat.succ k) 0) (Nat.le 0 (Nat.succ k)) (Nat.zero_le (Nat.succ k))) " +
			"(fun (j : Nat) (_ : Or (Nat.le (Nat.succ k) j) (Nat.le j (Nat.succ k))) => " +
			"Or.rec (Nat.le k j) (Nat.le j k) (fun (_ : Or (Nat.le k j) (Nat.le j k)) => Or (Nat.le (Nat.succ k) (Nat.succ j)) (Nat.le (Nat.succ j) (Nat.succ k))) " +
			"(fun (h : Nat.le k j) => Or.inl (Nat.le (Nat.succ k) (Nat.succ j)) (Nat.le (Nat.succ j) (Nat.succ k)) (Nat.succ_le_succ k j h)) " +
			"(fun (h : Nat.le j k) => Or.inr (Nat.le (Nat.succ k) (Nat.succ j)) (Nat.le (Nat.succ j) (Nat.succ k)) (Nat.succ_le_succ j k h)) (ih j)) m) n"},
	valDecl{"thm", "Nat.lt_or_ge", nil, "(a : Nat) → (b : Nat) → Or (Nat.lt a b) (Nat.le b a)",
		"fun (a : Nat) (b : Nat) => Or.rec (Nat.le (Nat.succ a) b) (Nat.le b (Nat.succ a)) " +
			"(fun (_ : Or (Nat.le (Nat.succ a) b) (Nat.le b (Nat.succ a))) => Or (Nat.lt a b) (Nat.le b a)) " +
			"(fun (h : Nat.le (Nat.succ a) b) => Or.inl (Nat.lt a b) (Nat.le b a) h) " +
			"(fun (h : Nat.le b (Nat.succ a)) => Decidable.rec.{0} (Eq.{1} Nat b (Nat.succ a)) " +
			"(fun (_ : Decidable (Eq.{1} Nat b (Nat.succ a))) => Or (Nat.lt a b) (Nat.le b a)) " +
			"(fun (ne : Not (Eq.{1} Nat b (Nat.succ a))) => Or.inr (Nat.lt a b) (Nat.le b a) (Nat.le_of_succ_le_succ b a (Nat.lt_of_le_ne b (Nat.succ a) h ne))) " +
			"(fun (e : Eq.{1} Nat b (Nat.succ a)) => Or.inl (Nat.lt a b) (Nat.le b a) " +
			"(Eq.rec.{0, 1} Nat (Nat.succ a) (fun (z : Nat) (_ : Eq.{1} Nat (Nat.succ a) z) => Nat.lt a z) (Nat.le.refl (Nat.succ a)) b " +
			"(Eq.symm.{1} Nat b (Nat.succ a) e))) (Nat.decEq b (Nat.succ a))) " +
			"(Nat.le_total (Nat.succ a) b)"},
	// The division identity and the bounds that follow from it (`Init/Data/Nat/Basic.lean`, `Div.lean`).
	valDecl{"thm", "Nat.div_add_mod", nil, "(k : Nat) → (n : Nat) → " + dmBeta,
		"fun (k : Nat) (n : Nat) => And.right (" + dmAlpha + ") (" + dmBeta + ") (Nat.divMod_spec k n)"},
	valDecl{"thm", "Nat.mod_add_div", nil, "(m : Nat) → (k : Nat) → Eq.{1} Nat (Nat.add (Nat.mod m k) (Nat.mul k (Nat.div m k))) m",
		"fun (m : Nat) (k : Nat) => Eq.trans.{1} Nat (Nat.add (Nat.mod m k) (Nat.mul k (Nat.div m k))) " +
			"(Nat.add (Nat.mul k (Nat.div m k)) (Nat.mod m k)) m (Nat.add_comm (Nat.mod m k) (Nat.mul k (Nat.div m k))) (Nat.div_add_mod k m)"},
	valDecl{"thm", "Nat.mod_le", nil, "(n : Nat) → (k : Nat) → Nat.le (Nat.mod n k) n",
		"fun (n : Nat) (k : Nat) => Eq.rec.{0, 1} Nat (Nat.add (Nat.mod n k) (Nat.mul k (Nat.div n k))) " +
			"(fun (z : Nat) (_ : Eq.{1} Nat (Nat.add (Nat.mod n k) (Nat.mul k (Nat.div n k))) z) => Nat.le (Nat.mod n k) z) " +
			"(Nat.le_add_right (Nat.mod n k) (Nat.mul k (Nat.div n k))) n (Nat.mod_add_div n k)"},
	// `n / k ≤ n`: for `k = succ j`, `q ≤ j * q + q = succ j * q ≤ succ j * q + r = n`; `n / 0 = 0`.
	valDecl{"thm", "Nat.div_le_self", nil, "(n : Nat) → (k : Nat) → Nat.le (Nat.div n k) n",
		"fun (n : Nat) (k : Nat) => Nat.rec.{0} (fun (k : Nat) => Nat.le (Nat.div n k) n) " +
			"(Eq.rec.{0, 1} Nat 0 (fun (z : Nat) (_ : Eq.{1} Nat 0 z) => Nat.le z n) (Nat.zero_le n) (Nat.div n 0) " +
			"(Eq.symm.{1} Nat (Nat.div n 0) 0 (Nat.div_zero n))) " +
			"(fun (j : Nat) (_ : Nat.le (Nat.div n j) n) => Nat.le_trans (Nat.div n (Nat.succ j)) (Nat.mul (Nat.succ j) (Nat.div n (Nat.succ j))) n " +
			"(Eq.rec.{0, 1} Nat (Nat.add (Nat.mul j (Nat.div n (Nat.succ j))) (Nat.div n (Nat.succ j))) " +
			"(fun (z : Nat) (_ : Eq.{1} Nat (Nat.add (Nat.mul j (Nat.div n (Nat.succ j))) (Nat.div n (Nat.succ j))) z) => Nat.le (Nat.div n (Nat.succ j)) z) " +
			"(Nat.le_add_left (Nat.div n (Nat.succ j)) (Nat.mul j (Nat.div n (Nat.succ j)))) " +
			"(Nat.mul (Nat.succ j) (Nat.div n (Nat.succ j))) " +
			"(Eq.symm.{1} Nat (Nat.mul (Nat.succ j) (Nat.div n (Nat.succ j))) (Nat.add (Nat.mul j (Nat.div n (Nat.succ j))) (Nat.div n (Nat.succ j))) " +
			"(Nat.succ_mul j (Nat.div n (Nat.succ j))))) " +
			"(Eq.rec.{0, 1} Nat (Nat.add (Nat.mul (Nat.succ j) (Nat.div n (Nat.succ j))) (Nat.mod n (Nat.succ j))) " +
			"(fun (z : Nat) (_ : Eq.{1} Nat (Nat.add (Nat.mul (Nat.succ j) (Nat.div n (Nat.succ j))) (Nat.mod n (Nat.succ j))) z) => " +
			"Nat.le (Nat.mul (Nat.succ j) (Nat.div n (Nat.succ j))) z) " +
			"(Nat.le_add_right (Nat.mul (Nat.succ j) (Nat.div n (Nat.succ j))) (Nat.mod n (Nat.succ j))) n (Nat.div_add_mod (Nat.succ j) n))) k"},
	// `(a + b * n) % b = a % b`: `b = 0` is `mod_zero`; otherwise `mod_mul_add` with quotient `a / b + n`.
	valDecl{"thm", "Nat.add_mul_mod_self_left", nil,
		"(a : Nat) → (b : Nat) → (n : Nat) → Eq.{1} Nat (Nat.mod (Nat.add a (Nat.mul b n)) b) (Nat.mod a b)",
		"fun (a : Nat) (b : Nat) (n : Nat) => Nat.rec.{0} (fun (b : Nat) => Eq.{1} Nat (Nat.mod (Nat.add a (Nat.mul b n)) b) (Nat.mod a b)) " +
			"(Eq.trans.{1} Nat (Nat.mod (Nat.add a (Nat.mul 0 n)) 0) (Nat.add a (Nat.mul 0 n)) (Nat.mod a 0) (Nat.mod_zero (Nat.add a (Nat.mul 0 n))) " +
			"(Eq.trans.{1} Nat (Nat.add a (Nat.mul 0 n)) a (Nat.mod a 0) (congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.add a z) (Nat.mul 0 n) 0 " +
			"(Nat.zero_mul n)) (Eq.symm.{1} Nat (Nat.mod a 0) a (Nat.mod_zero a)))) (fun (j : Nat) (_ : Eq.{1} Nat (Nat.mod (Nat.add a (Nat.mul j n)) j) " +
			"(Nat.mod a j)) => (Nat.mod_mul_add (Nat.succ j) (Nat.add (Nat.div a (Nat.succ j)) n) (Nat.mod a (Nat.succ j)) (Nat.add a (Nat.mul (Nat.succ " +
			"j) n)) (Eq.trans.{1} Nat (Nat.add a (Nat.mul (Nat.succ j) n)) (Nat.add (Nat.add (Nat.mul (Nat.succ j) (Nat.div a (Nat.succ j))) (Nat.mod a " +
			"(Nat.succ j))) (Nat.mul (Nat.succ j) n)) (Nat.add (Nat.mul (Nat.succ j) (Nat.add (Nat.div a (Nat.succ j)) n)) (Nat.mod a (Nat.succ j))) " +
			"(congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.add z (Nat.mul (Nat.succ j) n)) a (Nat.add (Nat.mul (Nat.succ j) (Nat.div a (Nat.succ j))) " +
			"(Nat.mod a (Nat.succ j))) (Eq.symm.{1} Nat (Nat.add (Nat.mul (Nat.succ j) (Nat.div a (Nat.succ j))) (Nat.mod a (Nat.succ j))) a " +
			"(Nat.div_add_mod (Nat.succ j) a))) (Eq.trans.{1} Nat (Nat.add (Nat.add (Nat.mul (Nat.succ j) (Nat.div a (Nat.succ j))) (Nat.mod a (Nat.succ " +
			"j))) (Nat.mul (Nat.succ j) n)) (Nat.add (Nat.add (Nat.mul (Nat.succ j) (Nat.div a (Nat.succ j))) (Nat.mul (Nat.succ j) n)) (Nat.mod a " +
			"(Nat.succ j))) (Nat.add (Nat.mul (Nat.succ j) (Nat.add (Nat.div a (Nat.succ j)) n)) (Nat.mod a (Nat.succ j))) (Nat.add_right_comm (Nat.mul " +
			"(Nat.succ j) (Nat.div a (Nat.succ j))) (Nat.mod a (Nat.succ j)) (Nat.mul (Nat.succ j) n)) (congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.add " +
			"z (Nat.mod a (Nat.succ j))) (Nat.add (Nat.mul (Nat.succ j) (Nat.div a (Nat.succ j))) (Nat.mul (Nat.succ j) n)) (Nat.mul (Nat.succ j) " +
			"(Nat.add (Nat.div a (Nat.succ j)) n)) (Eq.symm.{1} Nat (Nat.mul (Nat.succ j) (Nat.add (Nat.div a (Nat.succ j)) n)) (Nat.add (Nat.mul " +
			"(Nat.succ j) (Nat.div a (Nat.succ j))) (Nat.mul (Nat.succ j) n)) (Nat.mul_add (Nat.succ j) (Nat.div a (Nat.succ j)) n))))) (Nat.zero_lt_succ " +
			"j) (Nat.mod_lt a (Nat.succ j) (Nat.zero_lt_succ j)))) b"},
	valDecl{"thm", "Nat.mod_add_mod", nil,
		"(m : Nat) → (n : Nat) → (k : Nat) → Eq.{1} Nat (Nat.mod (Nat.add (Nat.mod m n) k) n) (Nat.mod (Nat.add m k) n)",
		"fun (m : Nat) (n : Nat) (k : Nat) => Eq.trans.{1} Nat (Nat.mod (Nat.add (Nat.mod m n) k) n) (Nat.mod (Nat.add (Nat.add (Nat.mod m n) k) " +
			"(Nat.mul n (Nat.div m n))) n) (Nat.mod (Nat.add m k) n) (Eq.symm.{1} Nat (Nat.mod (Nat.add (Nat.add (Nat.mod m n) k) (Nat.mul n (Nat.div m " +
			"n))) n) (Nat.mod (Nat.add (Nat.mod m n) k) n) (Nat.add_mul_mod_self_left (Nat.add (Nat.mod m n) k) n (Nat.div m n))) (congrArg.{1, 1} Nat " +
			"Nat (fun (z : Nat) => Nat.mod z n) (Nat.add (Nat.add (Nat.mod m n) k) (Nat.mul n (Nat.div m n))) (Nat.add m k) (Eq.trans.{1} Nat (Nat.add " +
			"(Nat.add (Nat.mod m n) k) (Nat.mul n (Nat.div m n))) (Nat.add (Nat.add (Nat.mod m n) (Nat.mul n (Nat.div m n))) k) (Nat.add m k) " +
			"(Eq.symm.{1} Nat (Nat.add (Nat.add (Nat.mod m n) (Nat.mul n (Nat.div m n))) k) (Nat.add (Nat.add (Nat.mod m n) k) (Nat.mul n (Nat.div m n))) " +
			"(Nat.add_right_comm (Nat.mod m n) (Nat.mul n (Nat.div m n)) k)) (congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.add z k) (Nat.add (Nat.mod m " +
			"n) (Nat.mul n (Nat.div m n))) m (Nat.mod_add_div m n))))"},
	valDecl{"thm", "Nat.add_mod_mod", nil,
		"(m : Nat) → (n : Nat) → (k : Nat) → Eq.{1} Nat (Nat.mod (Nat.add m (Nat.mod n k)) k) (Nat.mod (Nat.add m n) k)",
		"fun (m : Nat) (n : Nat) (k : Nat) => Eq.trans.{1} Nat (Nat.mod (Nat.add m (Nat.mod n k)) k) (Nat.mod (Nat.add (Nat.mod n k) m) k) (Nat.mod " +
			"(Nat.add m n) k) (congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.mod z k) (Nat.add m (Nat.mod n k)) (Nat.add (Nat.mod n k) m) (Nat.add_comm m " +
			"(Nat.mod n k))) (Eq.trans.{1} Nat (Nat.mod (Nat.add (Nat.mod n k) m) k) (Nat.mod (Nat.add n m) k) (Nat.mod (Nat.add m n) k) (Nat.mod_add_mod " +
			"n k m) (congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.mod z k) (Nat.add n m) (Nat.add m n) (Nat.add_comm n m)))"},
	valDecl{"thm", "Nat.add_mod", nil,
		"(a : Nat) → (b : Nat) → (n : Nat) → Eq.{1} Nat (Nat.mod (Nat.add a b) n) (Nat.mod (Nat.add (Nat.mod a n) (Nat.mod b n)) n)",
		"fun (a : Nat) (b : Nat) (n : Nat) => Eq.trans.{1} Nat (Nat.mod (Nat.add a b) n) (Nat.mod (Nat.add (Nat.mod a n) b) n) (Nat.mod (Nat.add " +
			"(Nat.mod a n) (Nat.mod b n)) n) (Eq.symm.{1} Nat (Nat.mod (Nat.add (Nat.mod a n) b) n) (Nat.mod (Nat.add a b) n) (Nat.mod_add_mod a n b)) " +
			"(Eq.symm.{1} Nat (Nat.mod (Nat.add (Nat.mod a n) (Nat.mod b n)) n) (Nat.mod (Nat.add (Nat.mod a n) b) n) (Nat.add_mod_mod (Nat.mod a n) b " +
			"n))"},
	// Right distributivity, from `mul_comm` and `mul_add`.
	valDecl{"thm", "Nat.add_mul", nil,
		"(a : Nat) → (b : Nat) → (c : Nat) → Eq.{1} Nat (Nat.mul (Nat.add a b) c) (Nat.add (Nat.mul a c) (Nat.mul b c))",
		"fun (a : Nat) (b : Nat) (c : Nat) => Eq.trans.{1} Nat (Nat.mul (Nat.add a b) c) (Nat.mul c (Nat.add a b)) (Nat.add (Nat.mul a c) (Nat.mul b " +
			"c)) (Nat.mul_comm (Nat.add a b) c) (Eq.trans.{1} Nat (Nat.mul c (Nat.add a b)) (Nat.add (Nat.mul c a) (Nat.mul c b)) (Nat.add (Nat.mul a c) " +
			"(Nat.mul b c)) (Nat.mul_add c a b) (Eq.trans.{1} Nat (Nat.add (Nat.mul c a) (Nat.mul c b)) (Nat.add (Nat.mul a c) (Nat.mul c b)) (Nat.add " +
			"(Nat.mul a c) (Nat.mul b c)) (congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.add z (Nat.mul c b)) (Nat.mul c a) (Nat.mul a c) (Nat.mul_comm c " +
			"a)) (congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.add (Nat.mul a c) z) (Nat.mul c b) (Nat.mul b c) (Nat.mul_comm c b))))"},
	// `a % n % n = a % n`: `n = 0` is `mod_zero`; otherwise the remainder is below `n`.
	valDecl{"thm", "Nat.mod_mod", nil,
		"(a : Nat) → (n : Nat) → Eq.{1} Nat (Nat.mod (Nat.mod a n) n) (Nat.mod a n)",
		"fun (a : Nat) (n : Nat) => Nat.rec.{0} (fun (n : Nat) => Eq.{1} Nat (Nat.mod (Nat.mod a n) n) (Nat.mod a n)) (Nat.mod_zero (Nat.mod a 0)) " +
			"(fun (j : Nat) (_ : Eq.{1} Nat (Nat.mod (Nat.mod a j) j) (Nat.mod a j)) => Nat.mod_eq_of_lt (Nat.mod a (Nat.succ j)) (Nat.succ j) " +
			"(Nat.mod_lt a (Nat.succ j) (Nat.zero_lt_succ j))) n"},
	// `(a % n) * b % n = a * b % n`: write `a = a % n + n * (a / n)` and drop the multiple of `n` with `add_mul_mod_self_left`.
	valDecl{"thm", "Nat.mod_mul_mod", nil,
		"(a : Nat) → (b : Nat) → (n : Nat) → Eq.{1} Nat (Nat.mod (Nat.mul (Nat.mod a n) b) n) (Nat.mod (Nat.mul a b) n)",
		"fun (a : Nat) (b : Nat) (n : Nat) => Eq.trans.{1} Nat (Nat.mod (Nat.mul (Nat.mod a n) b) n) (Nat.mod (Nat.add (Nat.mul (Nat.mod a n) b) " +
			"(Nat.mul n (Nat.mul (Nat.div a n) b))) n) (Nat.mod (Nat.mul a b) n) (Eq.symm.{1} Nat (Nat.mod (Nat.add (Nat.mul (Nat.mod a n) b) (Nat.mul n " +
			"(Nat.mul (Nat.div a n) b))) n) (Nat.mod (Nat.mul (Nat.mod a n) b) n) (Nat.add_mul_mod_self_left (Nat.mul (Nat.mod a n) b) n (Nat.mul " +
			"(Nat.div a n) b))) (congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.mod z n) (Nat.add (Nat.mul (Nat.mod a n) b) (Nat.mul n (Nat.mul (Nat.div a " +
			"n) b))) (Nat.mul a b) (Eq.symm.{1} Nat (Nat.mul a b) (Nat.add (Nat.mul (Nat.mod a n) b) (Nat.mul n (Nat.mul (Nat.div a n) b))) (Eq.trans.{1} " +
			"Nat (Nat.mul a b) (Nat.mul (Nat.add (Nat.mod a n) (Nat.mul n (Nat.div a n))) b) (Nat.add (Nat.mul (Nat.mod a n) b) (Nat.mul n (Nat.mul " +
			"(Nat.div a n) b))) (congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.mul z b) a (Nat.add (Nat.mod a n) (Nat.mul n (Nat.div a n))) (Eq.symm.{1} " +
			"Nat (Nat.add (Nat.mod a n) (Nat.mul n (Nat.div a n))) a (Nat.mod_add_div a n))) (Eq.trans.{1} Nat (Nat.mul (Nat.add (Nat.mod a n) (Nat.mul n " +
			"(Nat.div a n))) b) (Nat.add (Nat.mul (Nat.mod a n) b) (Nat.mul (Nat.mul n (Nat.div a n)) b)) (Nat.add (Nat.mul (Nat.mod a n) b) (Nat.mul n " +
			"(Nat.mul (Nat.div a n) b))) (Nat.add_mul (Nat.mod a n) (Nat.mul n (Nat.div a n)) b) (congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.add " +
			"(Nat.mul (Nat.mod a n) b) z) (Nat.mul (Nat.mul n (Nat.div a n)) b) (Nat.mul n (Nat.mul (Nat.div a n) b)) (Nat.mul_assoc n (Nat.div a n) " +
			"b))))))"},
	valDecl{"thm", "Nat.mul_mod", nil,
		"(a : Nat) → (b : Nat) → (n : Nat) → Eq.{1} Nat (Nat.mod (Nat.mul a b) n) (Nat.mod (Nat.mul (Nat.mod a n) (Nat.mod b n)) n)",
		"fun (a : Nat) (b : Nat) (n : Nat) => Eq.trans.{1} Nat (Nat.mod (Nat.mul a b) n) (Nat.mod (Nat.mul (Nat.mod a n) b) n) (Nat.mod (Nat.mul " +
			"(Nat.mod a n) (Nat.mod b n)) n) (Eq.symm.{1} Nat (Nat.mod (Nat.mul (Nat.mod a n) b) n) (Nat.mod (Nat.mul a b) n) (Nat.mod_mul_mod a b n)) " +
			"(Eq.trans.{1} Nat (Nat.mod (Nat.mul (Nat.mod a n) b) n) (Nat.mod (Nat.mul b (Nat.mod a n)) n) (Nat.mod (Nat.mul (Nat.mod a n) (Nat.mod b n)) " +
			"n) (congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.mod z n) (Nat.mul (Nat.mod a n) b) (Nat.mul b (Nat.mod a n)) (Nat.mul_comm (Nat.mod a n) " +
			"b)) (Eq.trans.{1} Nat (Nat.mod (Nat.mul b (Nat.mod a n)) n) (Nat.mod (Nat.mul (Nat.mod b n) (Nat.mod a n)) n) (Nat.mod (Nat.mul (Nat.mod a " +
			"n) (Nat.mod b n)) n) (Eq.symm.{1} Nat (Nat.mod (Nat.mul (Nat.mod b n) (Nat.mod a n)) n) (Nat.mod (Nat.mul b (Nat.mod a n)) n) " +
			"(Nat.mod_mul_mod b (Nat.mod a n) n)) (congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.mod z n) (Nat.mul (Nat.mod b n) (Nat.mod a n)) (Nat.mul " +
			"(Nat.mod a n) (Nat.mod b n)) (Nat.mul_comm (Nat.mod b n) (Nat.mod a n)))))"},
	// `Nat.sub` lemmas (`Init/Data/Nat/Basic.lean`).
	valDecl{"thm", "Nat.sub_zero", nil, "(n : Nat) → Eq.{1} Nat (Nat.sub n 0) n", "fun (n : Nat) => Eq.refl.{1} Nat n"},
	valDecl{"thm", "Nat.pred_le", nil, "(n : Nat) → Nat.le (Nat.pred n) n",
		"fun (n : Nat) => Nat.rec.{0} (fun (n : Nat) => Nat.le (Nat.pred n) n) (Nat.le.refl 0) " +
			"(fun (k : Nat) (_ : Nat.le (Nat.pred k) k) => Nat.le_succ k) n"},
	valDecl{"thm", "Nat.sub_le", nil, "(n : Nat) → (m : Nat) → Nat.le (Nat.sub n m) n",
		"fun (n : Nat) (m : Nat) => Nat.rec.{0} (fun (m : Nat) => Nat.le (Nat.sub n m) n) (Nat.le.refl n) " +
			"(fun (k : Nat) (ih : Nat.le (Nat.sub n k) n) => Nat.le_trans (Nat.pred (Nat.sub n k)) (Nat.sub n k) n (Nat.pred_le (Nat.sub n k)) ih) m"},
	valDecl{"thm", "Nat.succ_pred", nil, "(n : Nat) → Not (Eq.{1} Nat n 0) → Eq.{1} Nat (Nat.succ (Nat.pred n)) n",
		"fun (n : Nat) => Nat.rec.{0} (fun (n : Nat) => Not (Eq.{1} Nat n 0) → Eq.{1} Nat (Nat.succ (Nat.pred n)) n) " +
			"(fun (h : Not (Eq.{1} Nat 0 0)) => False.elim.{0} (Eq.{1} Nat (Nat.succ (Nat.pred 0)) 0) (h (Eq.refl.{1} Nat 0))) " +
			"(fun (k : Nat) (_ : Not (Eq.{1} Nat k 0) → Eq.{1} Nat (Nat.succ (Nat.pred k)) k) (_ : Not (Eq.{1} Nat (Nat.succ k) 0)) => " +
			"Eq.refl.{1} Nat (Nat.succ k)) n"},
	// `m ≤ n` gives `n - m + m = n`: with `x = n - k` and `x + k = n`, `x` is not 0 (else `n = k < succ k ≤ n`), so
	// `x = succ (pred x)` and the step moves the `succ` across the sum.
	valDecl{"thm", "Nat.sub_add_cancel", nil, "(n : Nat) → (m : Nat) → Nat.le m n → Eq.{1} Nat (Nat.add (Nat.sub n m) m) n",
		"fun (n : Nat) (m : Nat) => Nat.rec.{0} (fun (m : Nat) => Nat.le m n → Eq.{1} Nat (Nat.add (Nat.sub n m) m) n) " +
			"(fun (_ : Nat.le 0 n) => Eq.refl.{1} Nat n) " +
			"(fun (k : Nat) (ih : Nat.le k n → Eq.{1} Nat (Nat.add (Nat.sub n k) k) n) (h : Nat.le (Nat.succ k) n) => " +
			"Eq.trans.{1} Nat (Nat.add (Nat.pred (Nat.sub n k)) (Nat.succ k)) (Nat.add (Nat.succ (Nat.pred (Nat.sub n k))) k) n " +
			"(Eq.symm.{1} Nat (Nat.add (Nat.succ (Nat.pred (Nat.sub n k))) k) (Nat.succ (Nat.add (Nat.pred (Nat.sub n k)) k)) " +
			"(Nat.succ_add (Nat.pred (Nat.sub n k)) k)) " +
			"(Eq.trans.{1} Nat (Nat.add (Nat.succ (Nat.pred (Nat.sub n k))) k) (Nat.add (Nat.sub n k) k) n " +
			"(congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.add z k) (Nat.succ (Nat.pred (Nat.sub n k))) (Nat.sub n k) " +
			"(Nat.succ_pred (Nat.sub n k) (fun (e : Eq.{1} Nat (Nat.sub n k) 0) => Nat.not_le_succ_self k " +
			"(Eq.rec.{0, 1} Nat n (fun (z : Nat) (_ : Eq.{1} Nat n z) => Nat.le (Nat.succ k) z) h k " +
			"(Eq.trans.{1} Nat n (Nat.add (Nat.sub n k) k) k " +
			"(Eq.symm.{1} Nat (Nat.add (Nat.sub n k) k) n (ih (Nat.le_of_succ_le k n h))) " +
			"(Eq.trans.{1} Nat (Nat.add (Nat.sub n k) k) (Nat.add 0 k) k " +
			"(congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.add z k) (Nat.sub n k) 0 e) (Nat.zero_add k))))))) " +
			"(ih (Nat.le_of_succ_le k n h)))) m"},
	valDecl{"thm", "Nat.add_sub_of_le", nil, "(n : Nat) → (m : Nat) → Nat.le n m → Eq.{1} Nat (Nat.add n (Nat.sub m n)) m",
		"fun (n : Nat) (m : Nat) (h : Nat.le n m) => Eq.trans.{1} Nat (Nat.add n (Nat.sub m n)) (Nat.add (Nat.sub m n) n) m " +
			"(Nat.add_comm n (Nat.sub m n)) (Nat.sub_add_cancel m n h)"},
	// `n ≤ m` gives `n - m = 0`: induction on `m`; `succ i - succ k = i - k` hands the step to the hypothesis.
	valDecl{"thm", "Nat.sub_eq_zero_of_le", nil,
		"(n : Nat) → (m : Nat) → Nat.le n m → Eq.{1} Nat (Nat.sub n m) 0",
		"fun (n : Nat) (m : Nat) => (Nat.rec.{0} (fun (m : Nat) => (n : Nat) → Nat.le n m → Eq.{1} Nat (Nat.sub n m) 0) (fun (n : Nat) (h : Nat.le n " +
			"0) => Nat.le_zero n h) (fun (k : Nat) (ih : (n : Nat) → Nat.le n k → Eq.{1} Nat (Nat.sub n k) 0) (n : Nat) => Nat.rec.{0} (fun (n : Nat) => " +
			"Nat.le n (Nat.succ k) → Eq.{1} Nat (Nat.sub n (Nat.succ k)) 0) (fun (_ : Nat.le 0 (Nat.succ k)) => Nat.zero_sub (Nat.succ k)) (fun (i : Nat) " +
			"(_ : Nat.le i (Nat.succ k) → Eq.{1} Nat (Nat.sub i (Nat.succ k)) 0) (h : Nat.le (Nat.succ i) (Nat.succ k)) => Eq.trans.{1} Nat (Nat.sub " +
			"(Nat.succ i) (Nat.succ k)) (Nat.sub i k) 0 (Nat.succ_sub_succ i k) (ih i (Nat.le_of_succ_le_succ i k h))) n) m) n"},
	// `0 < n` and `0 < m` give `n - m < n`: both are successors, and `succ i - succ j = i - j <= i`.
	valDecl{"thm", "Nat.sub_lt", nil,
		"(n : Nat) → (m : Nat) → Nat.lt 0 n → Nat.lt 0 m → Nat.lt (Nat.sub n m) n",
		"fun (n : Nat) (m : Nat) => Nat.rec.{0} (fun (n : Nat) => Nat.lt 0 n → Nat.lt 0 m → Nat.lt (Nat.sub n m) n) (fun (h : Nat.lt 0 0) (_ : Nat.lt " +
			"0 m) => False.elim.{0} (Nat.lt (Nat.sub 0 m) 0) (Nat.lt_irrefl 0 h)) (fun (i : Nat) (_ : Nat.lt 0 i → Nat.lt 0 m → Nat.lt (Nat.sub i m) i) " +
			"(_ : Nat.lt 0 (Nat.succ i)) (hm : Nat.lt 0 m) => Nat.rec.{0} (fun (m : Nat) => Nat.lt 0 m → Nat.lt (Nat.sub (Nat.succ i) m) (Nat.succ i)) " +
			"(fun (h : Nat.lt 0 0) => False.elim.{0} (Nat.lt (Nat.sub (Nat.succ i) 0) (Nat.succ i)) (Nat.lt_irrefl 0 h)) (fun (j : Nat) (_ : Nat.lt 0 j → " +
			"Nat.lt (Nat.sub (Nat.succ i) j) (Nat.succ i)) (_ : Nat.lt 0 (Nat.succ j)) => Eq.rec.{0, 1} Nat (Nat.sub i j) (fun (z : Nat) (_ : Eq.{1} Nat " +
			"(Nat.sub i j) z) => Nat.lt z (Nat.succ i)) (Nat.succ_le_succ (Nat.sub i j) i (Nat.sub_le i j)) (Nat.sub (Nat.succ i) (Nat.succ j)) " +
			"(Eq.symm.{1} Nat (Nat.sub (Nat.succ i) (Nat.succ j)) (Nat.sub i j) (Nat.succ_sub_succ i j))) m hm) n"},
	valDecl{"thm", "Nat.lt_succ_of_le", nil,
		"(n : Nat) → (m : Nat) → Nat.le n m → Nat.lt n (Nat.succ m)",
		"fun (n : Nat) (m : Nat) (h : Nat.le n m) => Nat.succ_le_succ n m h"},
	valDecl{"thm", "Nat.le_of_lt_succ", nil,
		"(n : Nat) → (m : Nat) → Nat.lt n (Nat.succ m) → Nat.le n m",
		"fun (n : Nat) (m : Nat) (h : Nat.lt n (Nat.succ m)) => Nat.le_of_succ_le_succ n m h"},
	// Cancellation: subtract `m` from both sides with `add_sub_cancel`.
	valDecl{"thm", "Nat.add_right_cancel", nil,
		"(n : Nat) → (m : Nat) → (k : Nat) → Eq.{1} Nat (Nat.add n m) (Nat.add k m) → Eq.{1} Nat n k",
		"fun (n : Nat) (m : Nat) (k : Nat) (h : Eq.{1} Nat (Nat.add n m) (Nat.add k m)) => Eq.trans.{1} Nat n (Nat.sub (Nat.add n m) m) k " +
			"(Eq.symm.{1} Nat (Nat.sub (Nat.add n m) m) n (Nat.add_sub_cancel n m)) (Eq.trans.{1} Nat (Nat.sub (Nat.add n m) m) (Nat.sub (Nat.add k m) m) " +
			"k (congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.sub z m) (Nat.add n m) (Nat.add k m) h) (Nat.add_sub_cancel k m))"},
	valDecl{"thm", "Nat.add_left_cancel", nil,
		"(n : Nat) → (m : Nat) → (k : Nat) → Eq.{1} Nat (Nat.add n m) (Nat.add n k) → Eq.{1} Nat m k",
		"fun (n : Nat) (m : Nat) (k : Nat) (h : Eq.{1} Nat (Nat.add n m) (Nat.add n k)) => Eq.trans.{1} Nat m (Nat.sub (Nat.add n m) n) k " +
			"(Eq.symm.{1} Nat (Nat.sub (Nat.add n m) n) m (Nat.add_sub_cancel_left n m)) (Eq.trans.{1} Nat (Nat.sub (Nat.add n m) n) (Nat.sub (Nat.add n " +
			"k) n) k (congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.sub z n) (Nat.add n m) (Nat.add n k) h) (Nat.add_sub_cancel_left n k))"},
	// `m ≤ n` gives `n - (n - m) = m`: rewrite `n` as `n - m + m`, then cancel `n - m`.
	valDecl{"thm", "Nat.sub_sub_self", nil,
		"(n : Nat) → (m : Nat) → Nat.le m n → Eq.{1} Nat (Nat.sub n (Nat.sub n m)) m",
		"fun (n : Nat) (m : Nat) (h : Nat.le m n) => Eq.trans.{1} Nat (Nat.sub n (Nat.sub n m)) (Nat.sub (Nat.add (Nat.sub n m) m) (Nat.sub n m)) m " +
			"(congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.sub z (Nat.sub n m)) n (Nat.add (Nat.sub n m) m) (Eq.symm.{1} Nat (Nat.add (Nat.sub n m) m) n " +
			"(Nat.sub_add_cancel n m h))) (Nat.add_sub_cancel_left (Nat.sub n m) m)"},
	// `k ≤ m` gives `n + m - k = n + (m - k)`: `(n + (m - k)) + k = n + m`, so subtract `k`.
	valDecl{"thm", "Nat.add_sub_assoc", nil,
		"(n : Nat) → (m : Nat) → (k : Nat) → Nat.le k m → Eq.{1} Nat (Nat.sub (Nat.add n m) k) (Nat.add n (Nat.sub m k))",
		"fun (n : Nat) (m : Nat) (k : Nat) (h : Nat.le k m) => Eq.trans.{1} Nat (Nat.sub (Nat.add n m) k) (Nat.sub (Nat.add (Nat.add n (Nat.sub m k)) " +
			"k) k) (Nat.add n (Nat.sub m k)) (congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.sub z k) (Nat.add n m) (Nat.add (Nat.add n (Nat.sub m k)) k) " +
			"(Eq.symm.{1} Nat (Nat.add (Nat.add n (Nat.sub m k)) k) (Nat.add n m) (Eq.trans.{1} Nat (Nat.add (Nat.add n (Nat.sub m k)) k) (Nat.add n " +
			"(Nat.add (Nat.sub m k) k)) (Nat.add n m) (Nat.add_assoc n (Nat.sub m k) k) (congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.add n z) (Nat.add " +
			"(Nat.sub m k) k) m (Nat.sub_add_cancel m k h))))) (Nat.add_sub_cancel (Nat.add n (Nat.sub m k)) k)"},
	valDecl{"thm", "Nat.pred_le_pred", nil,
		"(a : Nat) → (b : Nat) → Nat.le a b → Nat.le (Nat.pred a) (Nat.pred b)",
		"fun (a : Nat) (b : Nat) (h : Nat.le a b) => Nat.le.rec a (fun (y : Nat) (_ : Nat.le a y) => Nat.le (Nat.pred a) (Nat.pred y)) (Nat.le.refl " +
			"(Nat.pred a)) (fun (m : Nat) (_ : Nat.le a m) (ih : Nat.le (Nat.pred a) (Nat.pred m)) => Nat.le_trans (Nat.pred a) (Nat.pred m) m ih " +
			"(Nat.pred_le m)) b h"},
	// Subtracting `k` is monotone: `pred` is, and `n - succ j = pred (n - j)`.
	valDecl{"thm", "Nat.sub_le_sub_right", nil,
		"(n : Nat) → (m : Nat) → (k : Nat) → Nat.le n m → Nat.le (Nat.sub n k) (Nat.sub m k)",
		"fun (n : Nat) (m : Nat) (k : Nat) (h : Nat.le n m) => Nat.rec.{0} (fun (k : Nat) => Nat.le (Nat.sub n k) (Nat.sub m k)) h (fun (j : Nat) (ih " +
			": Nat.le (Nat.sub n j) (Nat.sub m j)) => Nat.pred_le_pred (Nat.sub n j) (Nat.sub m j) ih) k"},
	// Subtract `n` from both sides (`sub_le_sub_right`) and cancel with `add_sub_cancel_left`.
	valDecl{"thm", "Nat.le_of_add_le_add_left", nil,
		"(n : Nat) → (m : Nat) → (k : Nat) → Nat.le (Nat.add n m) (Nat.add n k) → Nat.le m k",
		"fun (n : Nat) (m : Nat) (k : Nat) (h : Nat.le (Nat.add n m) (Nat.add n k)) => Eq.rec.{0, 1} Nat (Nat.sub (Nat.add n k) n) (fun (z : Nat) (_ " +
			": Eq.{1} Nat (Nat.sub (Nat.add n k) n) z) => Nat.le m z) (Eq.rec.{0, 1} Nat (Nat.sub (Nat.add n m) n) (fun (z : Nat) (_ : Eq.{1} Nat " +
			"(Nat.sub (Nat.add n m) n) z) => Nat.le z (Nat.sub (Nat.add n k) n)) (Nat.sub_le_sub_right (Nat.add n m) (Nat.add n k) n h) m " +
			"(Nat.add_sub_cancel_left n m)) k (Nat.add_sub_cancel_left n k)"},
	// Commute both sums and use `le_of_add_le_add_left`.
	valDecl{"thm", "Nat.le_of_add_le_add_right", nil,
		"(n : Nat) → (m : Nat) → (k : Nat) → Nat.le (Nat.add n m) (Nat.add k m) → Nat.le n k",
		"fun (n : Nat) (m : Nat) (k : Nat) (h : Nat.le (Nat.add n m) (Nat.add k m)) => Nat.le_of_add_le_add_left m n k (Eq.rec.{0, 1} Nat (Nat.add k " +
			"m) (fun (z : Nat) (_ : Eq.{1} Nat (Nat.add k m) z) => Nat.le (Nat.add m n) z) (Eq.rec.{0, 1} Nat (Nat.add n m) (fun (z : Nat) (_ : Eq.{1} " +
			"Nat (Nat.add n m) z) => Nat.le z (Nat.add k m)) (h) (Nat.add m n) (Nat.add_comm n m)) (Nat.add m k) (Nat.add_comm k m))"},
	valDecl{"thm", "Nat.add_lt_add_right", nil,
		"(a : Nat) → (b : Nat) → (c : Nat) → Nat.lt a b → Nat.lt (Nat.add a c) (Nat.add b c)",
		"fun (a : Nat) (b : Nat) (c : Nat) (h : Nat.lt a b) => Eq.rec.{0, 1} Nat (Nat.add c b) (fun (z : Nat) (_ : Eq.{1} Nat (Nat.add c b) z) => " +
			"Nat.lt (Nat.add a c) z) (Eq.rec.{0, 1} Nat (Nat.add c a) (fun (z : Nat) (_ : Eq.{1} Nat (Nat.add c a) z) => Nat.lt z (Nat.add c b)) " +
			"(Nat.add_lt_add_left a b c h) (Nat.add a c) (Nat.add_comm c a)) (Nat.add b c) (Nat.add_comm c b)"},
	// `m * n = 0` gives `m = 0` or `n = 0`: `succ i * succ j` is a successor, so it is not 0.
	valDecl{"thm", "Nat.mul_eq_zero", nil,
		"(m : Nat) → (n : Nat) → Eq.{1} Nat (Nat.mul m n) 0 → Or (Eq.{1} Nat m 0) (Eq.{1} Nat n 0)",
		"fun (m : Nat) (n : Nat) => Nat.rec.{0} (fun (m : Nat) => Eq.{1} Nat (Nat.mul m n) 0 → Or (Eq.{1} Nat m 0) (Eq.{1} Nat n 0)) (fun (_ : Eq.{1} " +
			"Nat (Nat.mul 0 n) 0) => Or.inl (Eq.{1} Nat 0 0) (Eq.{1} Nat n 0) (Eq.refl.{1} Nat 0)) (fun (i : Nat) (_ : Eq.{1} Nat (Nat.mul i n) 0 → Or " +
			"(Eq.{1} Nat i 0) (Eq.{1} Nat n 0)) => Nat.rec.{0} (fun (n : Nat) => Eq.{1} Nat (Nat.mul (Nat.succ i) n) 0 → Or (Eq.{1} Nat (Nat.succ i) 0) " +
			"(Eq.{1} Nat n 0)) (fun (_ : Eq.{1} Nat (Nat.mul (Nat.succ i) 0) 0) => Or.inr (Eq.{1} Nat (Nat.succ i) 0) (Eq.{1} Nat 0 0) (Eq.refl.{1} Nat " +
			"0)) (fun (j : Nat) (_ : Eq.{1} Nat (Nat.mul (Nat.succ i) j) 0 → Or (Eq.{1} Nat (Nat.succ i) 0) (Eq.{1} Nat j 0)) (h : Eq.{1} Nat (Nat.mul " +
			"(Nat.succ i) (Nat.succ j)) 0) => False.elim.{0} (Or (Eq.{1} Nat (Nat.succ i) 0) (Eq.{1} Nat (Nat.succ j) 0)) (Nat.succ_ne_zero (Nat.add " +
			"(Nat.mul (Nat.succ i) j) i) h)) n) m"},
	// `0 < n` and `n * k = 0` give `k = 0`: `mul_eq_zero` leaves `n = 0`, which `0 < n` rules out.
	valDecl{"thm", "Nat.eq_zero_of_mul_eq_zero_pos", nil,
		"(n : Nat) → (k : Nat) → Nat.lt 0 n → Eq.{1} Nat (Nat.mul n k) 0 → Eq.{1} Nat k 0",
		"fun (n : Nat) (k : Nat) (hn : Nat.lt 0 n) (h : Eq.{1} Nat (Nat.mul n k) 0) => Or.rec (Eq.{1} Nat n 0) (Eq.{1} Nat k 0) (fun (_ : Or (Eq.{1} " +
			"Nat n 0) (Eq.{1} Nat k 0)) => Eq.{1} Nat k 0) (fun (e : Eq.{1} Nat n 0) => False.elim.{0} (Eq.{1} Nat k 0) (Nat.lt_irrefl 0 (Eq.rec.{0, 1} " +
			"Nat n (fun (z : Nat) (_ : Eq.{1} Nat n z) => Nat.lt 0 z) hn 0 e))) (fun (e : Eq.{1} Nat k 0) => e) (Nat.mul_eq_zero n k h)"},
	// Induction on `m` with `k` free: `n * succ a = n * a + n`, so `add_right_cancel` and the hypothesis finish; `k = 0` is
	// ruled out by `eq_zero_of_mul_eq_zero_pos`.
	valDecl{"thm", "Nat.mul_left_cancel", nil,
		"(n : Nat) → (m : Nat) → (k : Nat) → Nat.lt 0 n → Eq.{1} Nat (Nat.mul n m) (Nat.mul n k) → Eq.{1} Nat m k",
		"fun (n : Nat) (m : Nat) (k : Nat) (hn : Nat.lt 0 n) => Nat.rec.{0} (fun (m : Nat) => (k : Nat) → Eq.{1} Nat (Nat.mul n m) (Nat.mul n k) → " +
			"Eq.{1} Nat m k) (fun (k : Nat) (h : Eq.{1} Nat (Nat.mul n 0) (Nat.mul n k)) => Eq.symm.{1} Nat k 0 (Nat.eq_zero_of_mul_eq_zero_pos n k hn " +
			"(Eq.symm.{1} Nat (Nat.mul n 0) (Nat.mul n k) h))) (fun (a : Nat) (ih : (k : Nat) → Eq.{1} Nat (Nat.mul n a) (Nat.mul n k) → Eq.{1} Nat a k) " +
			"(k : Nat) => Nat.rec.{0} (fun (k : Nat) => Eq.{1} Nat (Nat.mul n (Nat.succ a)) (Nat.mul n k) → Eq.{1} Nat (Nat.succ a) k) (fun (h : Eq.{1} " +
			"Nat (Nat.mul n (Nat.succ a)) (Nat.mul n 0)) => False.elim.{0} (Eq.{1} Nat (Nat.succ a) 0) (Nat.succ_ne_zero a " +
			"(Nat.eq_zero_of_mul_eq_zero_pos n (Nat.succ a) hn h))) (fun (b : Nat) (_ : Eq.{1} Nat (Nat.mul n (Nat.succ a)) (Nat.mul n b) → Eq.{1} Nat " +
			"(Nat.succ a) b) (h : Eq.{1} Nat (Nat.mul n (Nat.succ a)) (Nat.mul n (Nat.succ b))) => congrArg.{1, 1} Nat Nat Nat.succ a b (ih b " +
			"(Nat.add_right_cancel (Nat.mul n a) n (Nat.mul n b) h))) k) m k"},
	// `mul_left_cancel` after commuting both products.
	valDecl{"thm", "Nat.mul_right_cancel", nil,
		"(n : Nat) → (m : Nat) → (k : Nat) → Nat.lt 0 m → Eq.{1} Nat (Nat.mul n m) (Nat.mul k m) → Eq.{1} Nat n k",
		"fun (n : Nat) (m : Nat) (k : Nat) (hm : Nat.lt 0 m) (h : Eq.{1} Nat (Nat.mul n m) (Nat.mul k m)) => Nat.mul_left_cancel m n k hm (Eq.trans.{1} Nat (Nat.mul m n) (Nat.mul n m) (Nat.mul m k) (Nat.mul_comm m n) (Eq.trans.{1} Nat (Nat.mul n m) (Nat.mul k m) (Nat.mul m k) h (Nat.mul_comm k m)))"},
	// `mul_left_cancel` for `<=`: `n * succ a = n * a + n`, so `le_of_add_le_add_right` feeds the hypothesis.
	valDecl{"thm", "Nat.le_of_mul_le_mul_left", nil,
		"(n : Nat) → (m : Nat) → (k : Nat) → Nat.lt 0 n → Nat.le (Nat.mul n m) (Nat.mul n k) → Nat.le m k",
		"fun (n : Nat) (m : Nat) (k : Nat) (hn : Nat.lt 0 n) => Nat.rec.{0} (fun (m : Nat) => (k : Nat) → Nat.le (Nat.mul n m) (Nat.mul n k) → Nat.le " +
			"m k) (fun (k : Nat) (_ : Nat.le (Nat.mul n 0) (Nat.mul n k)) => Nat.zero_le k) (fun (a : Nat) (ih : (k : Nat) → Nat.le (Nat.mul n a) " +
			"(Nat.mul n k) → Nat.le a k) (k : Nat) => Nat.rec.{0} (fun (k : Nat) => Nat.le (Nat.mul n (Nat.succ a)) (Nat.mul n k) → Nat.le (Nat.succ a) " +
			"k) (fun (h : Nat.le (Nat.mul n (Nat.succ a)) (Nat.mul n 0)) => False.elim.{0} (Nat.le (Nat.succ a) 0) (Nat.succ_ne_zero a " +
			"(Nat.eq_zero_of_mul_eq_zero_pos n (Nat.succ a) hn (Nat.le_zero (Nat.mul n (Nat.succ a)) h)))) (fun (b : Nat) (_ : Nat.le (Nat.mul n " +
			"(Nat.succ a)) (Nat.mul n b) → Nat.le (Nat.succ a) b) (h : Nat.le (Nat.mul n (Nat.succ a)) (Nat.mul n (Nat.succ b))) => Nat.succ_le_succ a b " +
			"(ih b (Nat.le_of_add_le_add_right (Nat.mul n a) n (Nat.mul n b) h))) k) m k"},
	valDecl{"thm", "Nat.one_dvd", nil,
		"(n : Nat) → Nat.dvd 1 n",
		"fun (n : Nat) => Exists.intro.{1} Nat (fun (c : Nat) => Eq.{1} Nat n (Nat.mul 1 c)) n (Eq.symm.{1} Nat (Nat.mul 1 n) n (Nat.one_mul n))"},
	valDecl{"thm", "Nat.dvd_zero", nil,
		"(n : Nat) → Nat.dvd n 0",
		"fun (n : Nat) => Exists.intro.{1} Nat (fun (c : Nat) => Eq.{1} Nat 0 (Nat.mul n c)) 0 (Eq.refl.{1} Nat 0)"},
	// Remainders from divisibility: `Nat.mod_eq_zero_of_dvd` with the matching `dvd` lemma.
	valDecl{"thm", "Nat.mul_mod_right", nil, "(a : Nat) → (b : Nat) → Eq.{1} Nat (Nat.mod (Nat.mul a b) a) 0",
		"fun (a : Nat) (b : Nat) => Nat.mod_eq_zero_of_dvd a (Nat.mul a b) (Nat.dvd_mul_right a b)"},
	valDecl{"thm", "Nat.mul_mod_left", nil, "(a : Nat) → (b : Nat) → Eq.{1} Nat (Nat.mod (Nat.mul a b) b) 0",
		"fun (a : Nat) (b : Nat) => Nat.mod_eq_zero_of_dvd b (Nat.mul a b) (Nat.dvd_mul_left b a)"},
	valDecl{"thm", "Nat.mod_self", nil, "(n : Nat) → Eq.{1} Nat (Nat.mod n n) 0",
		"fun (n : Nat) => Nat.mod_eq_zero_of_dvd n n (Nat.dvd_refl n)"},
	valDecl{"thm", "Nat.mod_one", nil, "(n : Nat) → Eq.{1} Nat (Nat.mod n 1) 0",
		"fun (n : Nat) => Nat.mod_eq_zero_of_dvd 1 n (Nat.one_dvd n)"},
	valDecl{"thm", "Nat.zero_mod", nil, "(n : Nat) → Eq.{1} Nat (Nat.mod 0 n) 0",
		"fun (n : Nat) => Nat.mod_eq_zero_of_dvd n 0 (Nat.dvd_zero n)"},
	// `0 / n = 0`: `n = 0` is `div_zero`; `n = succ j` is `div_eq_of_lt`.
	valDecl{"thm", "Nat.zero_div", nil,
		"(n : Nat) → Eq.{1} Nat (Nat.div 0 n) 0",
		"fun (n : Nat) => Nat.rec.{0} (fun (n : Nat) => Eq.{1} Nat (Nat.div 0 n) 0) (Nat.div_zero 0) (fun (j : Nat) (_ : Eq.{1} Nat (Nat.div 0 j) 0) " +
			"=> Nat.div_eq_of_lt 0 (Nat.succ j) (Nat.zero_lt_succ j)) n"},
	// `n * (m / n) <= n * (m / n) + m % n = m`.
	valDecl{"thm", "Nat.mul_div_le", nil,
		"(m : Nat) → (n : Nat) → Nat.le (Nat.mul n (Nat.div m n)) m",
		"fun (m : Nat) (n : Nat) => Eq.rec.{0, 1} Nat (Nat.add (Nat.mul n (Nat.div m n)) (Nat.mod m n)) (fun (z : Nat) (_ : Eq.{1} Nat (Nat.add " +
			"(Nat.mul n (Nat.div m n)) (Nat.mod m n)) z) => Nat.le (Nat.mul n (Nat.div m n)) z) (Nat.le_add_right (Nat.mul n (Nat.div m n)) (Nat.mod m " +
			"n)) m (Nat.div_add_mod n m)"},
	// `m < n * (m / n + 1)`: `m = n * (m / n) + m % n` and `m % n < n`.
	valDecl{"thm", "Nat.lt_mul_div_succ", nil,
		"(m : Nat) → (n : Nat) → Nat.lt 0 n → Nat.lt m (Nat.mul n (Nat.succ (Nat.div m n)))",
		"fun (m : Nat) (n : Nat) (hn : Nat.lt 0 n) => Eq.rec.{0, 1} Nat (Nat.add (Nat.mul n (Nat.div m n)) (Nat.mod m n)) (fun (z : Nat) (_ : Eq.{1} " +
			"Nat (Nat.add (Nat.mul n (Nat.div m n)) (Nat.mod m n)) z) => Nat.lt z (Nat.add (Nat.mul n (Nat.div m n)) n)) (Nat.add_lt_add_left (Nat.mod m " +
			"n) n (Nat.mul n (Nat.div m n)) (Nat.mod_lt m n hn)) m (Nat.div_add_mod n m)"},
	// `n * m < n * k` gives `m < k`: otherwise `k <= m`, so `n * k <= n * m` and `n * m < n * m`.
	valDecl{"thm", "Nat.lt_of_mul_lt_mul_left", nil,
		"(n : Nat) → (m : Nat) → (k : Nat) → Nat.lt (Nat.mul n m) (Nat.mul n k) → Nat.lt m k",
		"fun (n : Nat) (m : Nat) (k : Nat) (H : Nat.lt (Nat.mul n m) (Nat.mul n k)) => Or.rec (Nat.lt m k) (Nat.le k m) (fun (_ : Or (Nat.lt m k) " +
			"(Nat.le k m)) => Nat.lt m k) (fun (h : Nat.lt m k) => h) (fun (h : Nat.le k m) => False.elim.{0} (Nat.lt m k) (Nat.lt_irrefl (Nat.mul n m) " +
			"(Nat.lt_of_lt_of_le (Nat.mul n m) (Nat.mul n k) (Nat.mul n m) H (Nat.mul_le_mul_left n k m h)))) (Nat.lt_or_ge m k)"},
	// `Nat.pow` lemmas under Lean's names (`Init/Data/Nat/Basic.lean`, `Lemmas.lean`).
	valDecl{"thm", "Nat.pow_zero", nil, "(n : Nat) → Eq.{1} Nat (Nat.pow n 0) 1", "fun (n : Nat) => Eq.refl.{1} Nat 1"},
	valDecl{"thm", "Nat.pow_succ", nil, "(n : Nat) → (m : Nat) → Eq.{1} Nat (Nat.pow n (Nat.succ m)) (Nat.mul (Nat.pow n m) n)",
		"fun (n : Nat) (m : Nat) => Eq.refl.{1} Nat (Nat.mul (Nat.pow n m) n)"},
	valDecl{"thm", "Nat.pow_one", nil, "(n : Nat) → Eq.{1} Nat (Nat.pow n 1) n",
		"fun (n : Nat) => Eq.trans.{1} Nat (Nat.pow n 1) (Nat.mul 1 n) n (Eq.refl.{1} Nat (Nat.mul 1 n)) (Nat.one_mul n)"},
	valDecl{"thm", "Nat.one_pow", nil, "(n : Nat) → Eq.{1} Nat (Nat.pow 1 n) 1",
		"fun (n : Nat) => Nat.rec.{0} (fun (n : Nat) => Eq.{1} Nat (Nat.pow 1 n) 1) (Eq.refl.{1} Nat 1) " +
			"(fun (k : Nat) (ih : Eq.{1} Nat (Nat.pow 1 k) 1) => Eq.trans.{1} Nat (Nat.pow 1 (Nat.succ k)) (Nat.mul 1 1) 1 " +
			"(congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.mul z 1) (Nat.pow 1 k) 1 ih) (Eq.refl.{1} Nat 1)) n"},
	valDecl{"thm", "Nat.zero_pow", nil, "(n : Nat) → Nat.lt 0 n → Eq.{1} Nat (Nat.pow 0 n) 0",
		"fun (n : Nat) => Nat.rec.{0} (fun (n : Nat) => Nat.lt 0 n → Eq.{1} Nat (Nat.pow 0 n) 0) " +
			"(fun (h : Nat.lt 0 0) => False.elim.{0} (Eq.{1} Nat (Nat.pow 0 0) 0) (Nat.succ_ne_zero 0 (Nat.le_zero (Nat.succ 0) h))) " +
			"(fun (k : Nat) (_ : Nat.lt 0 k → Eq.{1} Nat (Nat.pow 0 k) 0) (_ : Nat.lt 0 (Nat.succ k)) => Eq.refl.{1} Nat 0) n"},
	valDecl{"thm", "Nat.pow_add", nil,
		"(n : Nat) → (m : Nat) → (k : Nat) → Eq.{1} Nat (Nat.pow n (Nat.add m k)) (Nat.mul (Nat.pow n m) (Nat.pow n k))",
		"fun (n : Nat) (m : Nat) (k : Nat) => Nat.rec.{0} (fun (k : Nat) => Eq.{1} Nat (Nat.pow n (Nat.add m k)) (Nat.mul (Nat.pow n m) (Nat.pow n k))) " +
			"(Eq.symm.{1} Nat (Nat.mul (Nat.pow n m) 1) (Nat.pow n m) (Nat.mul_one (Nat.pow n m))) " +
			"(fun (k : Nat) (ih : Eq.{1} Nat (Nat.pow n (Nat.add m k)) (Nat.mul (Nat.pow n m) (Nat.pow n k))) => " +
			"Eq.trans.{1} Nat (Nat.pow n (Nat.add m (Nat.succ k))) (Nat.mul (Nat.mul (Nat.pow n m) (Nat.pow n k)) n) " +
			"(Nat.mul (Nat.pow n m) (Nat.pow n (Nat.succ k))) " +
			"(congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.mul z n) (Nat.pow n (Nat.add m k)) (Nat.mul (Nat.pow n m) (Nat.pow n k)) ih) " +
			"(Nat.mul_assoc (Nat.pow n m) (Nat.pow n k) n)) k"},
	valDecl{"thm", "Nat.pow_two", nil, "(n : Nat) → Eq.{1} Nat (Nat.pow n 2) (Nat.mul n n)",
		"fun (n : Nat) => congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.mul z n) (Nat.pow n 1) n (Nat.pow_one n)"},
	valDecl{"thm", "Nat.mul_left_comm", nil,
		"(a : Nat) → (b : Nat) → (c : Nat) → Eq.{1} Nat (Nat.mul a (Nat.mul b c)) (Nat.mul b (Nat.mul a c))",
		"fun (a : Nat) (b : Nat) (c : Nat) => Eq.trans.{1} Nat (Nat.mul a (Nat.mul b c)) (Nat.mul (Nat.mul b a) c) (Nat.mul b (Nat.mul a c)) " +
			"(Eq.trans.{1} Nat (Nat.mul a (Nat.mul b c)) (Nat.mul (Nat.mul a b) c) (Nat.mul (Nat.mul b a) c) " +
			"(Eq.symm.{1} Nat (Nat.mul (Nat.mul a b) c) (Nat.mul a (Nat.mul b c)) (Nat.mul_assoc a b c)) " +
			"(congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.mul z c) (Nat.mul a b) (Nat.mul b a) (Nat.mul_comm a b))) " +
			"(Nat.mul_assoc b a c)"},
	valDecl{"thm", "Nat.mul_mul_mul_comm", nil,
		"(a : Nat) → (b : Nat) → (c : Nat) → (d : Nat) → Eq.{1} Nat (Nat.mul (Nat.mul a b) (Nat.mul c d)) (Nat.mul (Nat.mul a c) (Nat.mul b d))",
		"fun (a : Nat) (b : Nat) (c : Nat) (d : Nat) => Eq.trans.{1} Nat (Nat.mul (Nat.mul a b) (Nat.mul c d)) " +
			"(Nat.mul a (Nat.mul c (Nat.mul b d))) (Nat.mul (Nat.mul a c) (Nat.mul b d)) " +
			"(Eq.trans.{1} Nat (Nat.mul (Nat.mul a b) (Nat.mul c d)) (Nat.mul a (Nat.mul b (Nat.mul c d))) (Nat.mul a (Nat.mul c (Nat.mul b d))) " +
			"(Nat.mul_assoc a b (Nat.mul c d)) " +
			"(congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.mul a z) (Nat.mul b (Nat.mul c d)) (Nat.mul c (Nat.mul b d)) (Nat.mul_left_comm b c d))) " +
			"(Eq.symm.{1} Nat (Nat.mul (Nat.mul a c) (Nat.mul b d)) (Nat.mul a (Nat.mul c (Nat.mul b d))) (Nat.mul_assoc a c (Nat.mul b d)))"},
	valDecl{"thm", "Nat.mul_pow", nil,
		"(a : Nat) → (b : Nat) → (n : Nat) → Eq.{1} Nat (Nat.pow (Nat.mul a b) n) (Nat.mul (Nat.pow a n) (Nat.pow b n))",
		"fun (a : Nat) (b : Nat) (n : Nat) => Nat.rec.{0} (fun (n : Nat) => Eq.{1} Nat (Nat.pow (Nat.mul a b) n) (Nat.mul (Nat.pow a n) (Nat.pow b n))) " +
			"(Eq.refl.{1} Nat 1) " +
			"(fun (k : Nat) (ih : Eq.{1} Nat (Nat.pow (Nat.mul a b) k) (Nat.mul (Nat.pow a k) (Nat.pow b k))) => " +
			"Eq.trans.{1} Nat (Nat.pow (Nat.mul a b) (Nat.succ k)) (Nat.mul (Nat.mul (Nat.pow a k) (Nat.pow b k)) (Nat.mul a b)) " +
			"(Nat.mul (Nat.pow a (Nat.succ k)) (Nat.pow b (Nat.succ k))) " +
			"(congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.mul z (Nat.mul a b)) (Nat.pow (Nat.mul a b) k) (Nat.mul (Nat.pow a k) (Nat.pow b k)) ih) " +
			"(Nat.mul_mul_mul_comm (Nat.pow a k) (Nat.pow b k) a b)) n"},
	valDecl{"thm", "Nat.mul_pos", nil, "(m : Nat) → (n : Nat) → Nat.lt 0 m → Nat.lt 0 n → Nat.lt 0 (Nat.mul m n)",
		"fun (m : Nat) (n : Nat) (hm : Nat.lt 0 m) (hn : Nat.lt 0 n) => Nat.mul_lt_mul_of_pos_left m 0 n hn hm"},
	valDecl{"thm", "Nat.pos_pow_of_pos", nil, "(b : Nat) → (n : Nat) → Nat.lt 0 b → Nat.lt 0 (Nat.pow b n)",
		"fun (b : Nat) (n : Nat) (hb : Nat.lt 0 b) => Nat.rec.{0} (fun (n : Nat) => Nat.lt 0 (Nat.pow b n)) (Nat.le.refl 1) " +
			"(fun (k : Nat) (ih : Nat.lt 0 (Nat.pow b k)) => Nat.mul_pos (Nat.pow b k) b ih hb) n"},
	// `lcm m n = 0` forces a zero argument: `gcd m n * lcm m n = m * n` makes `m * n = 0`.
	valDecl{"thm", "Nat.lcm_ne_zero", nil,
		"(m : Nat) → (n : Nat) → Not (Eq.{1} Nat m 0) → Not (Eq.{1} Nat n 0) → Not (Eq.{1} Nat (Nat.lcm m n) 0)",
		"fun (m : Nat) (n : Nat) (hm : Not (Eq.{1} Nat m 0)) (hn : Not (Eq.{1} Nat n 0)) " +
			"(h : Eq.{1} Nat (Nat.lcm m n) 0) => Or.rec (Eq.{1} Nat m 0) (Eq.{1} Nat n 0) " +
			"(fun (_ : Or (Eq.{1} Nat m 0) (Eq.{1} Nat n 0)) => False) " +
			"(fun (e : Eq.{1} Nat m 0) => hm e) (fun (e : Eq.{1} Nat n 0) => hn e) " +
			"(Nat.mul_eq_zero m n (Eq.trans.{1} Nat (Nat.mul m n) (Nat.mul (Nat.gcd m n) (Nat.lcm m n)) 0 " +
			"(Eq.symm.{1} Nat (Nat.mul (Nat.gcd m n) (Nat.lcm m n)) (Nat.mul m n) (Nat.gcd_mul_lcm m n)) " +
			"(Eq.trans.{1} Nat (Nat.mul (Nat.gcd m n) (Nat.lcm m n)) (Nat.mul (Nat.gcd m n) 0) 0 " +
			"(congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.mul (Nat.gcd m n) z) (Nat.lcm m n) 0 h) " +
			"(Nat.mul_zero (Nat.gcd m n)))))"},
	valDecl{"thm", "Nat.lcm_pos", nil,
		"(m : Nat) → (n : Nat) → Nat.lt 0 m → Nat.lt 0 n → Nat.lt 0 (Nat.lcm m n)",
		"fun (m : Nat) (n : Nat) (hm : Nat.lt 0 m) (hn : Nat.lt 0 n) => Nat.pos_of_ne_zero (Nat.lcm m n) " +
			"(Nat.lcm_ne_zero m n " +
			"(fun (e : Eq.{1} Nat m 0) => Nat.lt_irrefl 0 (Eq.rec.{0, 1} Nat m " +
			"(fun (z : Nat) (_ : Eq.{1} Nat m z) => Nat.lt 0 z) hm 0 e)) " +
			"(fun (e : Eq.{1} Nat n 0) => Nat.lt_irrefl 0 (Eq.rec.{0, 1} Nat n " +
			"(fun (z : Nat) (_ : Eq.{1} Nat n z) => Nat.lt 0 z) hn 0 e)))"},
	valDecl{"thm", "Nat.eq_zero_of_lcm_eq_zero", nil,
		"(m : Nat) → (n : Nat) → Eq.{1} Nat (Nat.lcm m n) 0 → Or (Eq.{1} Nat m 0) (Eq.{1} Nat n 0)",
		"fun (m : Nat) (n : Nat) => Nat.rec.{0} (fun (m : Nat) => Eq.{1} Nat (Nat.lcm m n) 0 → " +
			"Or (Eq.{1} Nat m 0) (Eq.{1} Nat n 0)) " +
			"(fun (_ : Eq.{1} Nat (Nat.lcm 0 n) 0) => Or.inl (Eq.{1} Nat 0 0) (Eq.{1} Nat n 0) (Eq.refl.{1} Nat 0)) " +
			"(fun (i : Nat) (_ : Eq.{1} Nat (Nat.lcm i n) 0 → Or (Eq.{1} Nat i 0) (Eq.{1} Nat n 0)) => " +
			"Nat.rec.{0} (fun (n : Nat) => Eq.{1} Nat (Nat.lcm (Nat.succ i) n) 0 → " +
			"Or (Eq.{1} Nat (Nat.succ i) 0) (Eq.{1} Nat n 0)) " +
			"(fun (_ : Eq.{1} Nat (Nat.lcm (Nat.succ i) 0) 0) => Or.inr (Eq.{1} Nat (Nat.succ i) 0) (Eq.{1} Nat 0 0) " +
			"(Eq.refl.{1} Nat 0)) " +
			"(fun (j : Nat) (_ : Eq.{1} Nat (Nat.lcm (Nat.succ i) j) 0 → Or (Eq.{1} Nat (Nat.succ i) 0) (Eq.{1} Nat j 0)) " +
			"(hh : Eq.{1} Nat (Nat.lcm (Nat.succ i) (Nat.succ j)) 0) => " +
			"False.elim.{0} (Or (Eq.{1} Nat (Nat.succ i) 0) (Eq.{1} Nat (Nat.succ j) 0)) " +
			"(Nat.lcm_ne_zero (Nat.succ i) (Nat.succ j) (Nat.succ_ne_zero i) (Nat.succ_ne_zero j) hh)) n) m"},
	valDecl{"thm", "Nat.mul_le_mul_right", nil,
		"(k : Nat) → (n : Nat) → (m : Nat) → Nat.le n m → Nat.le (Nat.mul n k) (Nat.mul m k)",
		"fun (k : Nat) (n : Nat) (m : Nat) (h : Nat.le n m) => Eq.rec.{0, 1} Nat (Nat.mul k n) " +
			"(fun (z : Nat) (_ : Eq.{1} Nat (Nat.mul k n) z) => Nat.le z (Nat.mul m k)) " +
			"(Eq.rec.{0, 1} Nat (Nat.mul k m) (fun (w : Nat) (_ : Eq.{1} Nat (Nat.mul k m) w) => Nat.le (Nat.mul k n) w) " +
			"(Nat.mul_le_mul_left k n m h) (Nat.mul m k) (Nat.mul_comm k m)) (Nat.mul n k) (Nat.mul_comm k n)"},
	valDecl{"thm", "Nat.mul_le_mul", nil,
		"(a : Nat) → (b : Nat) → (c : Nat) → (d : Nat) → Nat.le a b → Nat.le c d → Nat.le (Nat.mul a c) (Nat.mul b d)",
		"fun (a : Nat) (b : Nat) (c : Nat) (d : Nat) (h1 : Nat.le a b) (h2 : Nat.le c d) => " +
			"Nat.le_trans (Nat.mul a c) (Nat.mul b c) (Nat.mul b d) (Nat.mul_le_mul_right c a b h1) (Nat.mul_le_mul_left b c d h2)"},
	valDecl{"thm", "Nat.pow_le_pow_left", nil, "(a : Nat) → (b : Nat) → (n : Nat) → Nat.le a b → Nat.le (Nat.pow a n) (Nat.pow b n)",
		"fun (a : Nat) (b : Nat) (n : Nat) (h : Nat.le a b) => Nat.rec.{0} (fun (n : Nat) => Nat.le (Nat.pow a n) (Nat.pow b n)) " +
			"(Nat.le.refl 1) (fun (k : Nat) (ih : Nat.le (Nat.pow a k) (Nat.pow b k)) => Nat.mul_le_mul (Nat.pow a k) (Nat.pow b k) a b ih h) n"},
	valDecl{"thm", "Nat.pow_succ'", nil, "(n : Nat) → (m : Nat) → Eq.{1} Nat (Nat.pow n (Nat.succ m)) (Nat.mul n (Nat.pow n m))",
		"fun (n : Nat) (m : Nat) => Eq.trans.{1} Nat (Nat.pow n (Nat.succ m)) (Nat.mul (Nat.pow n m) n) (Nat.mul n (Nat.pow n m)) " +
			"(Nat.pow_succ n m) (Nat.mul_comm (Nat.pow n m) n)"},
	valDecl{"thm", "Nat.pow_mul", nil, "(a : Nat) → (m : Nat) → (n : Nat) → Eq.{1} Nat (Nat.pow a (Nat.mul m n)) (Nat.pow (Nat.pow a m) n)",
		"fun (a : Nat) (m : Nat) (n : Nat) => Nat.rec.{0} (fun (n : Nat) => Eq.{1} Nat (Nat.pow a (Nat.mul m n)) (Nat.pow (Nat.pow a m) n)) " +
			"(Eq.refl.{1} Nat 1) " +
			"(fun (k : Nat) (ih : Eq.{1} Nat (Nat.pow a (Nat.mul m k)) (Nat.pow (Nat.pow a m) k)) => " +
			"Eq.trans.{1} Nat (Nat.pow a (Nat.mul m (Nat.succ k))) (Nat.mul (Nat.pow a (Nat.mul m k)) (Nat.pow a m)) " +
			"(Nat.pow (Nat.pow a m) (Nat.succ k)) (Nat.pow_add a (Nat.mul m k) m) " +
			"(congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.mul z (Nat.pow a m)) (Nat.pow a (Nat.mul m k)) (Nat.pow (Nat.pow a m) k) ih)) n"},
	valDecl{"thm", "Nat.lt_two_pow", nil, "(n : Nat) → Nat.lt n (Nat.pow 2 n)",
		"fun (n : Nat) => Nat.rec.{0} (fun (n : Nat) => Nat.lt n (Nat.pow 2 n)) (Nat.le.refl 1) " +
			"(fun (k : Nat) (ih : Nat.lt k (Nat.pow 2 k)) => Eq.rec.{0, 1} Nat (Nat.add (Nat.pow 2 k) (Nat.pow 2 k)) " +
			"(fun (z : Nat) (_ : Eq.{1} Nat (Nat.add (Nat.pow 2 k) (Nat.pow 2 k)) z) => Nat.lt (Nat.succ k) z) " +
			"(Nat.add_le_add (Nat.succ k) (Nat.pow 2 k) 1 (Nat.pow 2 k) ih (Nat.pos_pow_of_pos 2 k (Nat.zero_lt_succ 1))) " +
			"(Nat.pow 2 (Nat.succ k)) (Eq.symm.{1} Nat (Nat.pow 2 (Nat.succ k)) (Nat.add (Nat.pow 2 k) (Nat.pow 2 k)) " +
			"(congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.add z (Nat.pow 2 k)) (Nat.mul (Nat.pow 2 k) 1) (Nat.pow 2 k) (Nat.mul_one (Nat.pow 2 k))))) n"},
	// `x <= y / k` iff `x * k <= y`: forwards `x * k <= (y / k) * k <= y`; backwards, `y / k < x` would give
	// `y < k * (y / k + 1) <= k * x <= y`.
	valDecl{"thm", "Nat.le_div_iff_mul_le", nil,
		"(x : Nat) → (y : Nat) → (k : Nat) → Nat.lt 0 k → Iff (Nat.le x (Nat.div y k)) (Nat.le (Nat.mul x k) y)",
		"fun (x : Nat) (y : Nat) (k : Nat) (hk : Nat.lt 0 k) => Iff.intro (Nat.le x (Nat.div y k)) (Nat.le (Nat.mul x k) y) (fun (h : Nat.le x " +
			"(Nat.div y k)) => Nat.le_trans (Nat.mul x k) (Nat.mul (Nat.div y k) k) y (Nat.mul_le_mul_right k x (Nat.div y k) h) (Eq.rec.{0, 1} Nat " +
			"(Nat.mul k (Nat.div y k)) (fun (z : Nat) (_ : Eq.{1} Nat (Nat.mul k (Nat.div y k)) z) => Nat.le z y) (Nat.mul_div_le y k) (Nat.mul (Nat.div " +
			"y k) k) (Nat.mul_comm k (Nat.div y k)))) (fun (h : Nat.le (Nat.mul x k) y) => Or.rec (Nat.lt (Nat.div y k) x) (Nat.le x (Nat.div y k)) (fun " +
			"(_ : Or (Nat.lt (Nat.div y k) x) (Nat.le x (Nat.div y k))) => Nat.le x (Nat.div y k)) (fun (hq : Nat.lt (Nat.div y k) x) => False.elim.{0} " +
			"(Nat.le x (Nat.div y k)) (Nat.lt_irrefl y (Nat.lt_of_lt_of_le y (Nat.mul x k) y (Eq.rec.{0, 1} Nat (Nat.mul k x) (fun (z : Nat) (_ : Eq.{1} " +
			"Nat (Nat.mul k x) z) => Nat.lt y z) (Nat.lt_of_lt_of_le y (Nat.mul k (Nat.succ (Nat.div y k))) (Nat.mul k x) (Nat.lt_mul_div_succ y k hk) " +
			"(Nat.mul_le_mul_left k (Nat.succ (Nat.div y k)) x hq)) (Nat.mul x k) (Nat.mul_comm k x)) h))) (fun (hq : Nat.le x (Nat.div y k)) => hq) " +
			"(Nat.lt_or_ge (Nat.div y k) x))"},
	// `mul_div_le` with the factors swapped.
	valDecl{"thm", "Nat.div_mul_le_self", nil, "(m : Nat) → (n : Nat) → Nat.le (Nat.mul (Nat.div m n) n) m",
		"fun (m : Nat) (n : Nat) => Eq.rec.{0, 1} Nat (Nat.mul n (Nat.div m n)) (fun (z : Nat) (_ : Eq.{1} Nat (Nat.mul n (Nat.div m n)) z) => Nat.le z m) (Nat.mul_div_le m n) (Nat.mul (Nat.div m n) n) (Nat.mul_comm n (Nat.div m n))"},
	// `le_of_mul_le_mul_left` after commuting both products.
	valDecl{"thm", "Nat.le_of_mul_le_mul_right", nil,
		"(n : Nat) → (m : Nat) → (k : Nat) → Nat.lt 0 k → Nat.le (Nat.mul n k) (Nat.mul m k) → Nat.le n m",
		"fun (n : Nat) (m : Nat) (k : Nat) (hk : Nat.lt 0 k) (h : Nat.le (Nat.mul n k) (Nat.mul m k)) => Nat.le_of_mul_le_mul_left k n m hk (Eq.rec.{0, 1} Nat (Nat.mul m k) (fun (z : Nat) (_ : Eq.{1} Nat (Nat.mul m k) z) => Nat.le (Nat.mul k n) z) (Eq.rec.{0, 1} Nat (Nat.mul n k) (fun (z : Nat) (_ : Eq.{1} Nat (Nat.mul n k) z) => Nat.le z (Nat.mul m k)) h (Nat.mul k n) (Nat.mul_comm n k)) (Nat.mul k m) (Nat.mul_comm m k))"},
	valDecl{"thm", "Nat.not_lt_of_le", nil,
		"(a : Nat) → (b : Nat) → Nat.le a b → Not (Nat.lt b a)",
		"fun (a : Nat) (b : Nat) (h : Nat.le a b) (hl : Nat.lt b a) => Nat.not_le_succ_self b (Nat.le_trans (Nat.succ b) a b hl h)"},
	valDecl{"thm", "Nat.not_le_of_lt", nil,
		"(a : Nat) → (b : Nat) → Nat.lt a b → Not (Nat.le b a)",
		"fun (a : Nat) (b : Nat) (h : Nat.lt a b) (hl : Nat.le b a) => Nat.not_le_succ_self a (Nat.le_trans (Nat.succ a) b a h hl)"},
	valDecl{"thm", "Nat.lt_asymm", nil,
		"(a : Nat) → (b : Nat) → Nat.lt a b → Not (Nat.lt b a)",
		"fun (a : Nat) (b : Nat) (h : Nat.lt a b) => Nat.not_lt_of_le a b (Nat.le_of_lt a b h)"},
	// Trichotomy pieces: `lt_or_ge` and `le_antisymm` give the converses.
	valDecl{"thm", "Nat.le_of_not_lt", nil,
		"(a : Nat) → (b : Nat) → Not (Nat.lt a b) → Nat.le b a",
		"fun (a : Nat) (b : Nat) (h : Not (Nat.lt a b)) => Or.rec (Nat.lt a b) (Nat.le b a) (fun (_ : Or (Nat.lt a b) (Nat.le b a)) => Nat.le b a) " +
			"(fun (hl : Nat.lt a b) => False.elim.{0} (Nat.le b a) (h hl)) (fun (x : Nat.le b a) => x) (Nat.lt_or_ge a b)"},
	valDecl{"thm", "Nat.lt_of_not_le", nil,
		"(a : Nat) → (b : Nat) → Not (Nat.le a b) → Nat.lt b a",
		"fun (a : Nat) (b : Nat) (h : Not (Nat.le a b)) => Or.rec (Nat.lt b a) (Nat.le a b) (fun (_ : Or (Nat.lt b a) (Nat.le a b)) => Nat.lt b a) " +
			"(fun (x : Nat.lt b a) => x) (fun (hl : Nat.le a b) => False.elim.{0} (Nat.lt b a) (h hl)) (Nat.lt_or_ge b a)"},
	valDecl{"thm", "Nat.lt_or_eq_of_le", nil,
		"(a : Nat) → (b : Nat) → Nat.le a b → Or (Nat.lt a b) (Eq.{1} Nat a b)",
		"fun (a : Nat) (b : Nat) (h : Nat.le a b) => Or.rec (Nat.lt a b) (Nat.le b a) (fun (_ : Or (Nat.lt a b) (Nat.le b a)) => Or (Nat.lt a b) " +
			"(Eq.{1} Nat a b)) (fun (hl : Nat.lt a b) => Or.inl (Nat.lt a b) (Eq.{1} Nat a b) hl) (fun (hb : Nat.le b a) => Or.inr (Nat.lt a b) (Eq.{1} " +
			"Nat a b) (Nat.le_antisymm a b h hb)) (Nat.lt_or_ge a b)"},
	// `x / k < y` iff `x < y * k`: the negation of `le_div_iff_mul_le`.
	valDecl{"thm", "Nat.div_lt_iff_lt_mul", nil,
		"(x : Nat) → (y : Nat) → (k : Nat) → Nat.lt 0 k → Iff (Nat.lt (Nat.div x k) y) (Nat.lt x (Nat.mul y k))",
		"fun (x : Nat) (y : Nat) (k : Nat) (hk : Nat.lt 0 k) => Iff.intro (Nat.lt (Nat.div x k) y) (Nat.lt x (Nat.mul y k)) (fun (h : Nat.lt (Nat.div " +
			"x k) y) => Nat.lt_of_not_le (Nat.mul y k) x (fun (hle : Nat.le (Nat.mul y k) x) => Nat.not_le_of_lt (Nat.div x k) y h (Iff.mpr (Nat.le y " +
			"(Nat.div x k)) (Nat.le (Nat.mul y k) x) (Nat.le_div_iff_mul_le y x k hk) hle))) (fun (h : Nat.lt x (Nat.mul y k)) => Nat.lt_of_not_le y " +
			"(Nat.div x k) (fun (hle : Nat.le y (Nat.div x k)) => Nat.not_le_of_lt x (Nat.mul y k) h (Iff.mp (Nat.le y (Nat.div x k)) (Nat.le (Nat.mul y " +
			"k) x) (Nat.le_div_iff_mul_le y x k hk) hle)))"},
	// `m <= k * n` gives `m / k <= n`: `m < k * n + k = succ n * k`, so `m / k < succ n` by `div_lt_iff_lt_mul`.
	valDecl{"thm", "Nat.div_le_of_le_mul", nil,
		"(m : Nat) → (n : Nat) → (k : Nat) → Nat.lt 0 k → Nat.le m (Nat.mul k n) → Nat.le (Nat.div m k) n",
		"fun (m : Nat) (n : Nat) (k : Nat) (hk : Nat.lt 0 k) (h : Nat.le m (Nat.mul k n)) => Nat.le_of_lt_succ (Nat.div m k) n (Iff.mpr (Nat.lt " +
			"(Nat.div m k) (Nat.succ n)) (Nat.lt m (Nat.mul (Nat.succ n) k)) (Nat.div_lt_iff_lt_mul m (Nat.succ n) k hk) (Eq.rec.{0, 1} Nat (Nat.mul k " +
			"(Nat.succ n)) (fun (z : Nat) (_ : Eq.{1} Nat (Nat.mul k (Nat.succ n)) z) => Nat.lt m z) (Nat.lt_of_le_of_lt m (Nat.mul k n) (Nat.mul k " +
			"(Nat.succ n)) h (Nat.add_lt_add_left 0 k (Nat.mul k n) hk)) (Nat.mul (Nat.succ n) k) (Eq.symm.{1} Nat (Nat.mul (Nat.succ n) k) (Nat.mul k " +
			"(Nat.succ n)) (Nat.mul_comm (Nat.succ n) k))))"},
	// `b^y <= b^y * b = b^(succ y)` once `1 <= b`; the rest is induction on `n <= m`.
	valDecl{"thm", "Nat.pow_le_pow_right", nil,
		"(b : Nat) → (n : Nat) → (m : Nat) → Nat.lt 0 b → Nat.le n m → Nat.le (Nat.pow b n) (Nat.pow b m)",
		"fun (b : Nat) (n : Nat) (m : Nat) (hb : Nat.lt 0 b) (h : Nat.le n m) => Nat.le.rec n (fun (y : Nat) (_ : Nat.le n y) => Nat.le (Nat.pow b n) " +
			"(Nat.pow b y)) (Nat.le.refl (Nat.pow b n)) (fun (y : Nat) (_ : Nat.le n y) (ih : Nat.le (Nat.pow b n) (Nat.pow b y)) => Nat.le_trans " +
			"(Nat.pow b n) (Nat.pow b y) (Nat.pow b (Nat.succ y)) ih (Eq.rec.{0, 1} Nat (Nat.mul (Nat.pow b y) 1) (fun (z : Nat) (_ : Eq.{1} Nat (Nat.mul " +
			"(Nat.pow b y) 1) z) => Nat.le z (Nat.mul (Nat.pow b y) b)) (Nat.mul_le_mul_left (Nat.pow b y) 1 b hb) (Nat.pow b y) (Nat.mul_one (Nat.pow b " +
			"y)))) m h"},
	// `b^n ∣ b^y ∣ b^y * b = b^(succ y)`.
	valDecl{"thm", "Nat.pow_dvd_pow", nil,
		"(b : Nat) → (n : Nat) → (m : Nat) → Nat.le n m → Nat.dvd (Nat.pow b n) (Nat.pow b m)",
		"fun (b : Nat) (n : Nat) (m : Nat) (h : Nat.le n m) => Nat.le.rec n (fun (y : Nat) (_ : Nat.le n y) => Nat.dvd (Nat.pow b n) (Nat.pow b y)) " +
			"(Nat.dvd_refl (Nat.pow b n)) (fun (y : Nat) (_ : Nat.le n y) (ih : Nat.dvd (Nat.pow b n) (Nat.pow b y)) => Nat.dvd_trans (Nat.pow b n) " +
			"(Nat.pow b y) (Nat.pow b (Nat.succ y)) ih (Nat.dvd_mul_right (Nat.pow b y) b)) m h"},
	// `n / k < n` for `0 < n`, `1 < k`: by `div_lt_iff_lt_mul` it is `n < n * k`, which is `n * 1 < n * k`.
	valDecl{"thm", "Nat.div_lt_self", nil,
		"(n : Nat) → (k : Nat) → Nat.lt 0 n → Nat.lt 1 k → Nat.lt (Nat.div n k) n",
		"fun (n : Nat) (k : Nat) (hn : Nat.lt 0 n) (hk : Nat.lt 1 k) => Iff.mpr (Nat.lt (Nat.div n k) n) (Nat.lt n (Nat.mul n k)) " +
			"(Nat.div_lt_iff_lt_mul n n k (Nat.lt_trans 0 1 k (Nat.zero_lt_succ 0) hk)) " +
			"(Eq.rec.{0, 1} Nat (Nat.mul n 1) (fun (z : Nat) (_ : Eq.{1} Nat (Nat.mul n 1) z) => Nat.lt z (Nat.mul n k)) " +
			"(Nat.mul_lt_mul_of_pos_left n 1 k hk hn) n (Nat.mul_one n))"},
	// Positivity of sums: `n < n + k` for `0 < k` is `add_lt_add_left` (`n + 0` reduces to `n`); `0 < m + n` for `0 < m` is `lt_of_lt_of_le` with `le_add_right`.
	valDecl{"thm", "Nat.lt_add_of_pos_right", nil,
		"(n : Nat) → (k : Nat) → Nat.lt 0 k → Nat.lt n (Nat.add n k)",
		"fun (n : Nat) (k : Nat) (hk : Nat.lt 0 k) => Nat.add_lt_add_left 0 k n hk"},
	valDecl{"thm", "Nat.add_pos_left", nil,
		"(m : Nat) → (n : Nat) → Nat.lt 0 m → Nat.lt 0 (Nat.add m n)",
		"fun (m : Nat) (n : Nat) (h : Nat.lt 0 m) => Nat.lt_of_lt_of_le 0 m (Nat.add m n) h (Nat.le_add_right m n)"},
	// `a < b` gives `0 < b - a`: if `b - a <= 0` then `b - a = 0`, so `a + (b - a) = b` reads `a = b`, against `ne_of_lt`.
	valDecl{"thm", "Nat.sub_pos_of_lt", nil,
		"(a : Nat) → (b : Nat) → Nat.lt a b → Nat.lt 0 (Nat.sub b a)",
		"fun (a : Nat) (b : Nat) (hab : Nat.lt a b) => Nat.lt_of_not_le (Nat.sub b a) 0 (fun (hle : Nat.le (Nat.sub b a) 0) => " +
			"Nat.ne_of_lt a b hab (Eq.rec.{0, 1} Nat (Nat.sub b a) " +
			"(fun (z : Nat) (_ : Eq.{1} Nat (Nat.sub b a) z) => Eq.{1} Nat (Nat.add a z) b) " +
			"(Nat.add_sub_of_le a b (Nat.le_of_lt a b hab)) 0 (Nat.le_zero (Nat.sub b a) hle)))"},
	// `b ^ n < b ^ (n + 1)` for `1 < b`: `b ^ (n + 1)` is `b ^ n * b` by `rfl`, and `b ^ n * 1 < b ^ n * b` as `b ^ n > 0`.
	valDecl{"thm", "Nat.pow_lt_pow_succ", nil,
		"(b : Nat) → (n : Nat) → Nat.lt 1 b → Nat.lt (Nat.pow b n) (Nat.pow b (Nat.succ n))",
		"fun (b : Nat) (n : Nat) (hb : Nat.lt 1 b) => Eq.rec.{0, 1} Nat (Nat.mul (Nat.pow b n) 1) " +
			"(fun (z : Nat) (_ : Eq.{1} Nat (Nat.mul (Nat.pow b n) 1) z) => Nat.lt z (Nat.pow b (Nat.succ n))) " +
			"(Nat.mul_lt_mul_of_pos_left (Nat.pow b n) 1 b hb (Nat.pos_pow_of_pos b n (Nat.lt_trans 0 1 b (Nat.zero_lt_succ 0) hb))) " +
			"(Nat.pow b n) (Nat.mul_one (Nat.pow b n))"},
	// `b ^ m < b ^ n` for `1 < b`, `m < n`: induction on `succ m <= n`; the base is `pow_lt_pow_succ`, the step adds one more of it by `lt_trans`.
	valDecl{"thm", "Nat.pow_lt_pow_right", nil,
		"(b : Nat) → (m : Nat) → (n : Nat) → Nat.lt 1 b → Nat.lt m n → Nat.lt (Nat.pow b m) (Nat.pow b n)",
		"fun (b : Nat) (m : Nat) (n : Nat) (hb : Nat.lt 1 b) (h : Nat.lt m n) => Nat.le.rec (Nat.succ m) " +
			"(fun (y : Nat) (_ : Nat.le (Nat.succ m) y) => Nat.lt (Nat.pow b m) (Nat.pow b y)) " +
			"(Nat.pow_lt_pow_succ b m hb) " +
			"(fun (y : Nat) (_ : Nat.le (Nat.succ m) y) (ih : Nat.lt (Nat.pow b m) (Nat.pow b y)) => " +
			"Nat.lt_trans (Nat.pow b m) (Nat.pow b y) (Nat.pow b (Nat.succ y)) ih (Nat.pow_lt_pow_succ b y hb)) n h"},
	// `n + m < n + k` gives `m < k`: `succ (n + m)` is `n + succ m` by `rfl`, so it is `le_of_add_le_add_left` at `succ m`.
	valDecl{"thm", "Nat.lt_of_add_lt_add_left", nil,
		"(n : Nat) → (m : Nat) → (k : Nat) → Nat.lt (Nat.add n m) (Nat.add n k) → Nat.lt m k",
		"fun (n : Nat) (m : Nat) (k : Nat) (h : Nat.lt (Nat.add n m) (Nat.add n k)) => Nat.le_of_add_le_add_left n (Nat.succ m) k h"},
	// `a < b` and `c < d` give `a + c < b + d`: `a + c < b + c < b + d`.
	valDecl{"thm", "Nat.add_lt_add", nil,
		"(a : Nat) → (b : Nat) → (c : Nat) → (d : Nat) → Nat.lt a b → Nat.lt c d → Nat.lt (Nat.add a c) (Nat.add b d)",
		"fun (a : Nat) (b : Nat) (c : Nat) (d : Nat) (h1 : Nat.lt a b) (h2 : Nat.lt c d) => " +
			"Nat.lt_trans (Nat.add a c) (Nat.add b c) (Nat.add b d) (Nat.add_lt_add_right a b c h1) (Nat.add_lt_add_left c d b h2)"},
	// `n < m`, `0 < k` give `n * k < m * k`: `mul_lt_mul_of_pos_left` with both products turned round by `mul_comm`.
	valDecl{"thm", "Nat.mul_lt_mul_of_pos_right", nil,
		"(k : Nat) → (n : Nat) → (m : Nat) → Nat.lt n m → Nat.lt 0 k → Nat.lt (Nat.mul n k) (Nat.mul m k)",
		"fun (k : Nat) (n : Nat) (m : Nat) (h : Nat.lt n m) (hk : Nat.lt 0 k) => " +
			"Eq.rec.{0, 1} Nat (Nat.mul k n) (fun (z : Nat) (_ : Eq.{1} Nat (Nat.mul k n) z) => Nat.lt z (Nat.mul m k)) " +
			"(Eq.rec.{0, 1} Nat (Nat.mul k m) (fun (w : Nat) (_ : Eq.{1} Nat (Nat.mul k m) w) => Nat.lt (Nat.mul k n) w) " +
			"(Nat.mul_lt_mul_of_pos_left k n m h hk) (Nat.mul m k) (Nat.mul_comm k m)) (Nat.mul n k) (Nat.mul_comm k n)"},
	// `n * k < m * k` gives `n < m`: `lt_of_mul_lt_mul_left` after turning both products round by `mul_comm`.
	valDecl{"thm", "Nat.lt_of_mul_lt_mul_right", nil,
		"(k : Nat) → (n : Nat) → (m : Nat) → Nat.lt (Nat.mul n k) (Nat.mul m k) → Nat.lt n m",
		"fun (k : Nat) (n : Nat) (m : Nat) (h : Nat.lt (Nat.mul n k) (Nat.mul m k)) => Nat.lt_of_mul_lt_mul_left k n m " +
			"(Eq.rec.{0, 1} Nat (Nat.mul n k) (fun (z : Nat) (_ : Eq.{1} Nat (Nat.mul n k) z) => Nat.lt z (Nat.mul k m)) " +
			"(Eq.rec.{0, 1} Nat (Nat.mul m k) (fun (w : Nat) (_ : Eq.{1} Nat (Nat.mul m k) w) => Nat.lt (Nat.mul n k) w) " +
			"h (Nat.mul k m) (Nat.mul_comm m k)) (Nat.mul k n) (Nat.mul_comm n k))"},
	// `n + k < m + k` gives `n < m`: `lt_of_add_lt_add_left` after turning both sums round by `add_comm`.
	valDecl{"thm", "Nat.lt_of_add_lt_add_right", nil,
		"(k : Nat) → (n : Nat) → (m : Nat) → Nat.lt (Nat.add n k) (Nat.add m k) → Nat.lt n m",
		"fun (k : Nat) (n : Nat) (m : Nat) (h : Nat.lt (Nat.add n k) (Nat.add m k)) => Nat.lt_of_add_lt_add_left k n m " +
			"(Eq.rec.{0, 1} Nat (Nat.add n k) (fun (z : Nat) (_ : Eq.{1} Nat (Nat.add n k) z) => Nat.lt z (Nat.add k m)) " +
			"(Eq.rec.{0, 1} Nat (Nat.add m k) (fun (w : Nat) (_ : Eq.{1} Nat (Nat.add m k) w) => Nat.lt (Nat.add n k) w) " +
			"h (Nat.add k m) (Nat.add_comm m k)) (Nat.add k n) (Nat.add_comm n k))"},
	// Iff forms: `n < succ m <-> n <= m` pairs `lt_succ_of_le` with `le_of_lt_succ`; `succ n <= m <-> n < m` is the identity both ways.
	valDecl{"thm", "Nat.lt_succ_iff", nil,
		"(n : Nat) → (m : Nat) → Iff (Nat.lt n (Nat.succ m)) (Nat.le n m)",
		"fun (n : Nat) (m : Nat) => Iff.intro (Nat.lt n (Nat.succ m)) (Nat.le n m) (Nat.le_of_lt_succ n m) (Nat.lt_succ_of_le n m)"},
	valDecl{"thm", "Nat.succ_le_iff", nil,
		"(n : Nat) → (m : Nat) → Iff (Nat.le (Nat.succ n) m) (Nat.lt n m)",
		"fun (n : Nat) (m : Nat) => Iff.intro (Nat.le (Nat.succ n) m) (Nat.lt n m) (Nat.lt_of_succ_le n m) (Nat.succ_le_of_lt n m)"},
	// `0 < b - a` gives `a < b`, the converse of `sub_pos_of_lt`: if `b <= a` then `b - a = 0`, so `0 < 0`.
	valDecl{"thm", "Nat.lt_of_sub_pos", nil,
		"(a : Nat) → (b : Nat) → Nat.lt 0 (Nat.sub b a) → Nat.lt a b",
		"fun (a : Nat) (b : Nat) (hp : Nat.lt 0 (Nat.sub b a)) => Nat.lt_of_not_le b a (fun (hle : Nat.le b a) => " +
			"Nat.lt_irrefl 0 (Eq.rec.{0, 1} Nat (Nat.sub b a) (fun (z : Nat) (_ : Eq.{1} Nat (Nat.sub b a) z) => Nat.lt 0 z) hp 0 " +
			"(Nat.sub_eq_zero_of_le b a hle)))"},
	// Iff forms of subtraction: `0 < b - a <-> a < b` pairs the two lemmas above; `n - m = 0 <-> n <= m` has `sub_eq_zero_of_le` one way, and the other
	// is `le_of_not_lt`: `m < n` would give `0 < n - m` by `sub_pos_of_lt`.
	valDecl{"thm", "Nat.sub_pos_iff_lt", nil,
		"(a : Nat) → (b : Nat) → Iff (Nat.lt 0 (Nat.sub b a)) (Nat.lt a b)",
		"fun (a : Nat) (b : Nat) => Iff.intro (Nat.lt 0 (Nat.sub b a)) (Nat.lt a b) (Nat.lt_of_sub_pos a b) (Nat.sub_pos_of_lt a b)"},
	valDecl{"thm", "Nat.sub_eq_zero_iff_le", nil,
		"(n : Nat) → (m : Nat) → Iff (Eq.{1} Nat (Nat.sub n m) 0) (Nat.le n m)",
		"fun (n : Nat) (m : Nat) => Iff.intro (Eq.{1} Nat (Nat.sub n m) 0) (Nat.le n m) " +
			"(fun (e : Eq.{1} Nat (Nat.sub n m) 0) => Nat.le_of_not_lt m n (fun (hlt : Nat.lt m n) => Nat.lt_irrefl 0 " +
			"(Eq.rec.{0, 1} Nat (Nat.sub n m) (fun (z : Nat) (_ : Eq.{1} Nat (Nat.sub n m) z) => Nat.lt 0 z) (Nat.sub_pos_of_lt m n hlt) 0 e))) " +
			"(Nat.sub_eq_zero_of_le n m)"},
	// `n + (m + k) = m + (n + k)`.
	valDecl{"thm", "Nat.add_left_comm", nil,
		"(n : Nat) → (m : Nat) → (k : Nat) → Eq.{1} Nat (Nat.add n (Nat.add m k)) (Nat.add m (Nat.add n k))",
		"fun (n : Nat) (m : Nat) (k : Nat) => Eq.trans.{1} Nat (Nat.add n (Nat.add m k)) (Nat.add (Nat.add n m) k) (Nat.add m (Nat.add n k)) " +
			"(Eq.symm.{1} Nat (Nat.add (Nat.add n m) k) (Nat.add n (Nat.add m k)) (Nat.add_assoc n m k)) (Eq.trans.{1} Nat (Nat.add (Nat.add n m) k) " +
			"(Nat.add (Nat.add m n) k) (Nat.add m (Nat.add n k)) (congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.add z k) (Nat.add n m) (Nat.add m n) " +
			"(Nat.add_comm n m)) (Nat.add_assoc m n k))"},
	// `factorial` lemmas (`Mathlib/Data/Nat/Factorial/Basic.lean`; Lean core has the definition only).
	valDecl{"thm", "Nat.factorial_zero", nil, "Eq.{1} Nat (Nat.factorial 0) 1", "Eq.refl.{1} Nat 1"},
	valDecl{"thm", "Nat.factorial_succ", nil, "(n : Nat) → Eq.{1} Nat (Nat.factorial (Nat.succ n)) (Nat.mul (Nat.succ n) (Nat.factorial n))",
		"fun (n : Nat) => Eq.refl.{1} Nat (Nat.mul (Nat.succ n) (Nat.factorial n))"},
	valDecl{"thm", "Nat.factorial_pos", nil, "(n : Nat) → Nat.lt 0 (Nat.factorial n)",
		"fun (n : Nat) => Nat.rec.{0} (fun (n : Nat) => Nat.lt 0 (Nat.factorial n)) (Nat.le.refl 1) " +
			"(fun (k : Nat) (ih : Nat.lt 0 (Nat.factorial k)) => Nat.mul_pos (Nat.succ k) (Nat.factorial k) (Nat.zero_lt_succ k) ih) n"},
	valDecl{"thm", "Nat.dvd_factorial", nil, "(m : Nat) → (n : Nat) → Nat.lt 0 m → Nat.le m n → Nat.dvd m (Nat.factorial n)",
		"fun (m : Nat) (n : Nat) (hm : Nat.lt 0 m) => Nat.rec.{0} (fun (n : Nat) => Nat.le m n → Nat.dvd m (Nat.factorial n)) " +
			"(fun (h : Nat.le m 0) => False.elim.{0} (Nat.dvd m (Nat.factorial 0)) " +
			"(Nat.succ_ne_zero 0 (Nat.le_zero (Nat.succ 0) (Nat.le_trans (Nat.succ 0) m 0 hm h)))) " +
			"(fun (k : Nat) (ih : Nat.le m k → Nat.dvd m (Nat.factorial k)) (h : Nat.le m (Nat.succ k)) => " +
			"Decidable.rec.{0} (Eq.{1} Nat m (Nat.succ k)) " +
			"(fun (_ : Decidable (Eq.{1} Nat m (Nat.succ k))) => Nat.dvd m (Nat.factorial (Nat.succ k))) " +
			"(fun (ne : Not (Eq.{1} Nat m (Nat.succ k))) => Nat.dvd_trans m (Nat.factorial k) (Nat.factorial (Nat.succ k)) " +
			"(ih (Nat.le_of_succ_le_succ m k (Nat.lt_of_le_ne m (Nat.succ k) h ne))) (Nat.dvd_mul_left (Nat.factorial k) (Nat.succ k))) " +
			"(fun (e : Eq.{1} Nat m (Nat.succ k)) => Eq.rec.{0, 1} Nat (Nat.succ k) " +
			"(fun (z : Nat) (_ : Eq.{1} Nat (Nat.succ k) z) => Nat.dvd z (Nat.factorial (Nat.succ k))) " +
			"(Nat.dvd_mul_right (Nat.succ k) (Nat.factorial k)) m (Eq.symm.{1} Nat m (Nat.succ k) e)) " +
			"(Nat.decEq m (Nat.succ k))) n"},
	valDecl{"thm", "Nat.self_le_factorial", nil, "(n : Nat) → Nat.le n (Nat.factorial n)",
		"fun (n : Nat) => Nat.rec.{0} (fun (n : Nat) => Nat.le n (Nat.factorial n)) (Nat.zero_le (Nat.factorial 0)) " +
			"(fun (k : Nat) (_ : Nat.le k (Nat.factorial k)) => Nat.le_of_dvd (Nat.succ k) (Nat.factorial (Nat.succ k)) " +
			"(Nat.factorial_pos (Nat.succ k)) (Nat.dvd_factorial (Nat.succ k) (Nat.succ k) (Nat.zero_lt_succ k) (Nat.le.refl (Nat.succ k)))) n"},
	// Lean's `Nat.Coprime m n` is `Nat.gcd m n = 1`, decided by `Nat.decEq` (`Init/Data/Nat/Coprime.lean`).
	valDecl{"def", "Nat.Coprime", nil, "Nat → Nat → Prop",
		"fun (m : Nat) (n : Nat) => Eq.{1} Nat (Nat.gcd m n) 1"},
	valDecl{"def", "Nat.decCoprime", nil, "(m : Nat) → (n : Nat) → Decidable (Nat.Coprime m n)",
		"fun (m : Nat) (n : Nat) => Nat.decEq (Nat.gcd m n) 1"},
	valDecl{"thm", "Nat.coprime_iff_gcd_eq_one", nil,
		"(m : Nat) → (n : Nat) → Iff (Nat.Coprime m n) (Eq.{1} Nat (Nat.gcd m n) 1)",
		"fun (m : Nat) (n : Nat) => Iff.intro (Nat.Coprime m n) (Eq.{1} Nat (Nat.gcd m n) 1) (fun (h : Nat.Coprime m n) => h) (fun (h : Eq.{1} Nat (Nat.gcd m n) 1) => h)"},
	valDecl{"thm", "Nat.Coprime.gcd_eq_one", nil,
		"(m : Nat) → (n : Nat) → Nat.Coprime m n → Eq.{1} Nat (Nat.gcd m n) 1",
		"fun (m : Nat) (n : Nat) (h : Nat.Coprime m n) => h"},
	valDecl{"thm", "Nat.Coprime.symm", nil, "(m : Nat) → (n : Nat) → Nat.Coprime m n → Nat.Coprime n m",
		"fun (m : Nat) (n : Nat) (h : Nat.Coprime m n) => Eq.trans.{1} Nat (Nat.gcd n m) (Nat.gcd m n) 1 (Nat.gcd_comm n m) h"},
	valDecl{"thm", "Nat.coprime_comm", nil,
		"(m : Nat) → (n : Nat) → Iff (Nat.Coprime n m) (Nat.Coprime m n)",
		"fun (m : Nat) (n : Nat) => Iff.intro (Nat.Coprime n m) (Nat.Coprime m n) (Nat.Coprime.symm n m) (Nat.Coprime.symm m n)"},
	valDecl{"thm", "Nat.coprime_zero_left", nil,
		"(n : Nat) → Iff (Nat.Coprime 0 n) (Eq.{1} Nat n 1)",
		"fun (n : Nat) => Iff.intro (Nat.Coprime 0 n) (Eq.{1} Nat n 1) " +
			"(fun (h : Nat.Coprime 0 n) => Eq.trans.{1} Nat n (Nat.gcd 0 n) 1 (Eq.symm.{1} Nat (Nat.gcd 0 n) n (Nat.gcd_zero_left n)) h) " +
			"(fun (h : Eq.{1} Nat n 1) => Eq.trans.{1} Nat (Nat.gcd 0 n) n 1 (Nat.gcd_zero_left n) h)"},
	valDecl{"thm", "Nat.coprime_zero_right", nil,
		"(n : Nat) → Iff (Nat.Coprime n 0) (Eq.{1} Nat n 1)",
		"fun (n : Nat) => Iff.intro (Nat.Coprime n 0) (Eq.{1} Nat n 1) " +
			"(fun (h : Nat.Coprime n 0) => Eq.trans.{1} Nat n (Nat.gcd n 0) 1 (Eq.symm.{1} Nat (Nat.gcd n 0) n (Nat.gcd_zero_right n)) h) " +
			"(fun (h : Eq.{1} Nat n 1) => Eq.trans.{1} Nat (Nat.gcd n 0) n 1 (Nat.gcd_zero_right n) h)"},
	valDecl{"thm", "Nat.coprime_one_left", nil, "(n : Nat) → Nat.Coprime 1 n",
		"fun (n : Nat) => Nat.gcd_one_left n"},
	valDecl{"thm", "Nat.coprime_one_right", nil, "(n : Nat) → Nat.Coprime n 1",
		"fun (n : Nat) => Nat.gcd_one_right n"},
	valDecl{"thm", "Nat.coprime_self", nil,
		"(n : Nat) → Iff (Nat.Coprime n n) (Eq.{1} Nat n 1)",
		"fun (n : Nat) => Iff.intro (Nat.Coprime n n) (Eq.{1} Nat n 1) " +
			"(fun (h : Nat.Coprime n n) => Eq.trans.{1} Nat n (Nat.gcd n n) 1 (Eq.symm.{1} Nat (Nat.gcd n n) n (Nat.gcd_self n)) h) " +
			"(fun (h : Eq.{1} Nat n 1) => Eq.trans.{1} Nat (Nat.gcd n n) n 1 (Nat.gcd_self n) h)"},
	// `k ∣ m * n` and `gcd k n = 1` give `k ∣ m`: `k` divides `gcd (m * k) (m * n) = m * gcd k n = m` (Lean's proof).
	valDecl{"thm", "Nat.Coprime.dvd_of_dvd_mul_right", nil,
		"(k : Nat) → (n : Nat) → (m : Nat) → Nat.Coprime k n → Nat.dvd k (Nat.mul m n) → Nat.dvd k m",
		"fun (k : Nat) (n : Nat) (m : Nat) (h1 : Nat.Coprime k n) (h2 : Nat.dvd k (Nat.mul m n)) => " +
			"Eq.rec.{0, 1} Nat (Nat.gcd (Nat.mul m k) (Nat.mul m n)) (fun (z : Nat) (_ : Eq.{1} Nat (Nat.gcd (Nat.mul m k) (Nat.mul m n)) z) => Nat.dvd k z) " +
			"(Nat.dvd_gcd k (Nat.mul m k) (Nat.mul m n) (Nat.dvd_mul_left k m) h2) m " +
			"(Eq.trans.{1} Nat (Nat.gcd (Nat.mul m k) (Nat.mul m n)) (Nat.mul m (Nat.gcd k n)) m (Nat.gcd_mul_left m k n) " +
			"(Eq.trans.{1} Nat (Nat.mul m (Nat.gcd k n)) (Nat.mul m 1) m " +
			"(congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.mul m z) (Nat.gcd k n) 1 h1) (Nat.mul_one m)))"},
	valDecl{"thm", "Nat.Coprime.dvd_of_dvd_mul_left", nil,
		"(k : Nat) → (m : Nat) → (n : Nat) → Nat.Coprime k m → Nat.dvd k (Nat.mul m n) → Nat.dvd k n",
		"fun (k : Nat) (m : Nat) (n : Nat) (h1 : Nat.Coprime k m) (h2 : Nat.dvd k (Nat.mul m n)) => " +
			"Nat.Coprime.dvd_of_dvd_mul_right k m n h1 (Eq.rec.{0, 1} Nat (Nat.mul m n) " +
			"(fun (z : Nat) (_ : Eq.{1} Nat (Nat.mul m n) z) => Nat.dvd k z) h2 (Nat.mul n m) (Nat.mul_comm m n))"},
	// `k ∣ m` and `n ∣ m` with `k`, `n` coprime give `k * n ∣ m`: write `m = k * p`; `n ∣ k * p` gives `n ∣ p = n * q`.
	valDecl{"thm", "Nat.Coprime.mul_dvd_of_dvd_of_dvd", nil,
		"(k : Nat) → (n : Nat) → (m : Nat) → Nat.Coprime k n → Nat.dvd k m → Nat.dvd n m → Nat.dvd (Nat.mul k n) m",
		"fun (k : Nat) (n : Nat) (m : Nat) (h1 : Nat.Coprime k n) (h2 : Nat.dvd k m) (h3 : Nat.dvd n m) => Exists.rec.{1} Nat " +
			"(fun (p : Nat) => Eq.{1} Nat m (Nat.mul k p)) (fun (_ : Nat.dvd k m) => Nat.dvd (Nat.mul k n) m) " +
			"(fun (p : Nat) (hp : Eq.{1} Nat m (Nat.mul k p)) => Exists.rec.{1} Nat " +
			"(fun (q : Nat) => Eq.{1} Nat p (Nat.mul n q)) (fun (_ : Nat.dvd n p) => Nat.dvd (Nat.mul k n) m) " +
			"(fun (q : Nat) (hq : Eq.{1} Nat p (Nat.mul n q)) => Exists.intro.{1} Nat " +
			"(fun (w : Nat) => Eq.{1} Nat m (Nat.mul (Nat.mul k n) w)) q " +
			"(Eq.trans.{1} Nat m (Nat.mul k p) (Nat.mul (Nat.mul k n) q) hp " +
			"(Eq.trans.{1} Nat (Nat.mul k p) (Nat.mul k (Nat.mul n q)) (Nat.mul (Nat.mul k n) q) " +
			"(congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.mul k z) p (Nat.mul n q) hq) " +
			"(Eq.symm.{1} Nat (Nat.mul (Nat.mul k n) q) (Nat.mul k (Nat.mul n q)) (Nat.mul_assoc k n q))))) " +
			"(Nat.Coprime.dvd_of_dvd_mul_left n k p (Nat.Coprime.symm k n h1) " +
			"(Eq.rec.{0, 1} Nat m (fun (z : Nat) (_ : Eq.{1} Nat m z) => Nat.dvd n z) h3 (Nat.mul k p) hp))) h2"},
	valDecl{"thm", "Nat.div_one", nil, "(n : Nat) → Eq.{1} Nat (Nat.div n 1) n",
		"fun (n : Nat) => Eq.trans.{1} Nat (Nat.div n 1) (Nat.div (Nat.mul n 1) 1) n " +
			"(congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.div z 1) n (Nat.mul n 1) (Eq.symm.{1} Nat (Nat.mul n 1) n (Nat.mul_one n))) " +
			"(Nat.mul_div_cancel n 1 (Nat.le.refl 1))"},
	valDecl{"thm", "Nat.Coprime.lcm_eq_mul", nil, "(m : Nat) → (n : Nat) → Nat.Coprime m n → Eq.{1} Nat (Nat.lcm m n) (Nat.mul m n)",
		"fun (m : Nat) (n : Nat) (h : Nat.Coprime m n) => Eq.trans.{1} Nat (Nat.lcm m n) (Nat.div (Nat.mul m n) 1) (Nat.mul m n) " +
			"(congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.div (Nat.mul m n) z) (Nat.gcd m n) 1 h) (Nat.div_one (Nat.mul m n))"},
	// A divisor of a number coprime to `k` is coprime to `k`: `gcd m k` divides `gcd m' k = 1`.
	valDecl{"thm", "Nat.Coprime.coprime_dvd_left", nil,
		"(m : Nat) → (m' : Nat) → (k : Nat) → Nat.dvd m m' → Nat.Coprime m' k → Nat.Coprime m k",
		"fun (m : Nat) (m' : Nat) (k : Nat) (hd : Nat.dvd m m') (h : Nat.Coprime m' k) => Nat.eq_one_of_dvd_one (Nat.gcd m k) " +
			"(Eq.rec.{0, 1} Nat (Nat.gcd m' k) (fun (z : Nat) (_ : Eq.{1} Nat (Nat.gcd m' k) z) => Nat.dvd (Nat.gcd m k) z) " +
			"(Nat.dvd_gcd (Nat.gcd m k) m' k (Nat.dvd_trans (Nat.gcd m k) m m' (Nat.gcd_dvd_left m k) hd) (Nat.gcd_dvd_right m k)) 1 h)"},
	valDecl{"thm", "Nat.Coprime.coprime_dvd_right", nil,
		"(k : Nat) → (m : Nat) → (n : Nat) → Nat.dvd n m → Nat.Coprime k m → Nat.Coprime k n",
		"fun (k : Nat) (m : Nat) (n : Nat) (hd : Nat.dvd n m) (h : Nat.Coprime k m) => Nat.Coprime.symm n k " +
			"(Nat.Coprime.coprime_dvd_left n m k hd (Nat.Coprime.symm k m h))"},
	valDecl{"thm", "Nat.Coprime.coprime_mul_left", nil,
		"(m : Nat) → (k : Nat) → (n : Nat) → Nat.Coprime (Nat.mul k m) n → Nat.Coprime m n",
		"fun (m : Nat) (k : Nat) (n : Nat) (h : Nat.Coprime (Nat.mul k m) n) => " +
			"Nat.Coprime.coprime_dvd_left m (Nat.mul k m) n (Nat.dvd_mul_left m k) h"},
	valDecl{"thm", "Nat.Coprime.coprime_mul_right", nil,
		"(m : Nat) → (k : Nat) → (n : Nat) → Nat.Coprime (Nat.mul m k) n → Nat.Coprime m n",
		"fun (m : Nat) (k : Nat) (n : Nat) (h : Nat.Coprime (Nat.mul m k) n) => " +
			"Nat.Coprime.coprime_dvd_left m (Nat.mul m k) n (Nat.dvd_mul_right m k) h"},
	valDecl{"thm", "Nat.Coprime.coprime_mul_left_right", nil,
		"(m : Nat) → (k : Nat) → (n : Nat) → Nat.Coprime m (Nat.mul k n) → Nat.Coprime m n",
		"fun (m : Nat) (k : Nat) (n : Nat) (h : Nat.Coprime m (Nat.mul k n)) => " +
			"Nat.Coprime.coprime_dvd_right m (Nat.mul k n) n (Nat.dvd_mul_left n k) h"},
	valDecl{"thm", "Nat.Coprime.coprime_mul_right_right", nil,
		"(m : Nat) → (n : Nat) → (k : Nat) → Nat.Coprime m (Nat.mul n k) → Nat.Coprime m n",
		"fun (m : Nat) (n : Nat) (k : Nat) (h : Nat.Coprime m (Nat.mul n k)) => " +
			"Nat.Coprime.coprime_dvd_right m (Nat.mul n k) n (Nat.dvd_mul_right n k) h"},
	valDecl{"thm", "Nat.Coprime.gcd_left", nil,
		"(m : Nat) → (n : Nat) → (k : Nat) → Nat.Coprime m n → Nat.Coprime (Nat.gcd k m) n",
		"fun (m : Nat) (n : Nat) (k : Nat) (h : Nat.Coprime m n) => " +
			"Nat.Coprime.coprime_dvd_left (Nat.gcd k m) m n (Nat.gcd_dvd_right k m) h"},
	valDecl{"thm", "Nat.Coprime.gcd_right", nil,
		"(m : Nat) → (n : Nat) → (k : Nat) → Nat.Coprime m n → Nat.Coprime m (Nat.gcd k n)",
		"fun (m : Nat) (n : Nat) (k : Nat) (h : Nat.Coprime m n) => " +
			"Nat.Coprime.coprime_dvd_right m n (Nat.gcd k n) (Nat.gcd_dvd_right k n) h"},
	valDecl{"thm", "Nat.Coprime.gcd_both", nil,
		"(m : Nat) → (n : Nat) → (k : Nat) → (l : Nat) → Nat.Coprime m n → Nat.Coprime (Nat.gcd k m) (Nat.gcd l n)",
		"fun (m : Nat) (n : Nat) (k : Nat) (l : Nat) (h : Nat.Coprime m n) => Nat.Coprime.gcd_right (Nat.gcd k m) n l (Nat.Coprime.gcd_left m n k h)"},
	// `gcd k (m * n) = gcd k m * gcd k n` for coprime `m n`: `∣` one way is `gcd_mul_right_dvd_mul_gcd`, the other
	// is `gcd k m` and `gcd k n` (coprime, as `gcd_both` gives) both dividing `gcd k (m * n)`.
	valDecl{"thm", "Nat.Coprime.gcd_mul", nil,
		"(k : Nat) → (m : Nat) → (n : Nat) → Nat.Coprime m n → " +
			"Eq.{1} Nat (Nat.gcd k (Nat.mul m n)) (Nat.mul (Nat.gcd k m) (Nat.gcd k n))",
		"fun (k : Nat) (m : Nat) (n : Nat) (h : Nat.Coprime m n) => " +
			"Nat.dvd_antisymm (Nat.gcd k (Nat.mul m n)) (Nat.mul (Nat.gcd k m) (Nat.gcd k n)) " +
			"(Nat.gcd_mul_right_dvd_mul_gcd k m n) " +
			"(Nat.Coprime.mul_dvd_of_dvd_of_dvd (Nat.gcd k m) (Nat.gcd k n) (Nat.gcd k (Nat.mul m n)) " +
			"(Nat.Coprime.gcd_both m n k k h) (Nat.gcd_dvd_gcd_mul_right_right k m n) " +
			"(Nat.gcd_dvd_gcd_mul_left_right k n m))"},
	valDecl{"thm", "Nat.Coprime.mul_gcd", nil,
		"(m : Nat) → (n : Nat) → (k : Nat) → Nat.Coprime m n → " +
			"Eq.{1} Nat (Nat.gcd (Nat.mul m n) k) (Nat.mul (Nat.gcd m k) (Nat.gcd n k))",
		"fun (m : Nat) (n : Nat) (k : Nat) (h : Nat.Coprime m n) => " +
			"Eq.trans.{1} Nat (Nat.gcd (Nat.mul m n) k) (Nat.gcd k (Nat.mul m n)) " +
			"(Nat.mul (Nat.gcd m k) (Nat.gcd n k)) (Nat.gcd_comm (Nat.mul m n) k) " +
			"(Eq.trans.{1} Nat (Nat.gcd k (Nat.mul m n)) (Nat.mul (Nat.gcd k m) (Nat.gcd k n)) " +
			"(Nat.mul (Nat.gcd m k) (Nat.gcd n k)) (Nat.Coprime.gcd_mul k m n h) " +
			"(Eq.trans.{1} Nat (Nat.mul (Nat.gcd k m) (Nat.gcd k n)) (Nat.mul (Nat.gcd m k) (Nat.gcd k n)) " +
			"(Nat.mul (Nat.gcd m k) (Nat.gcd n k)) " +
			"(congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.mul z (Nat.gcd k n)) (Nat.gcd k m) (Nat.gcd m k) " +
			"(Nat.gcd_comm k m)) " +
			"(congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.mul (Nat.gcd m k) z) (Nat.gcd k n) (Nat.gcd n k) " +
			"(Nat.gcd_comm k n))))"},
	valDecl{"thm", "Nat.Coprime.eq_one_of_dvd", nil,
		"(k : Nat) → (m : Nat) → Nat.Coprime k m → Nat.dvd k m → Eq.{1} Nat k 1",
		"fun (k : Nat) (m : Nat) (h : Nat.Coprime k m) (d : Nat.dvd k m) => " +
			"Eq.trans.{1} Nat k (Nat.gcd k m) 1 (Eq.symm.{1} Nat (Nat.gcd k m) k (Nat.gcd_eq_left k m d)) (Nat.Coprime.gcd_eq_one k m h)"},
	valDecl{"thm", "Nat.Coprime.of_mul_left", nil,
		"(m : Nat) → (n : Nat) → (k : Nat) → Nat.Coprime (Nat.mul m n) k → Nat.Coprime m k",
		"fun (m : Nat) (n : Nat) (k : Nat) (h : Nat.Coprime (Nat.mul m n) k) => " +
			"Nat.Coprime.coprime_dvd_left m (Nat.mul m n) k (Nat.dvd_mul_right m n) h"},
	valDecl{"thm", "Nat.Coprime.of_mul_right", nil,
		"(m : Nat) → (n : Nat) → (k : Nat) → Nat.Coprime (Nat.mul m n) k → Nat.Coprime n k",
		"fun (m : Nat) (n : Nat) (k : Nat) (h : Nat.Coprime (Nat.mul m n) k) => " +
			"Nat.Coprime.coprime_dvd_left n (Nat.mul m n) k (Nat.dvd_mul_left n m) h"},
	// `k` coprime to `m` and to `n` is coprime to `m * n`. With `g = gcd k (m * n)`: `g` is coprime to `m`, so `g ∣ n`,
	// so `g` divides `gcd k n = 1`. (No Bezout needed.)
	valDecl{"thm", "Nat.Coprime.mul_right", nil,
		"(k : Nat) → (m : Nat) → (n : Nat) → Nat.Coprime k m → Nat.Coprime k n → Nat.Coprime k (Nat.mul m n)",
		"fun (k : Nat) (m : Nat) (n : Nat) (h1 : Nat.Coprime k m) (h2 : Nat.Coprime k n) => " +
			"Nat.eq_one_of_dvd_one (Nat.gcd k (Nat.mul m n)) " +
			"(Eq.rec.{0, 1} Nat (Nat.gcd k n) (fun (z : Nat) (_ : Eq.{1} Nat (Nat.gcd k n) z) => Nat.dvd (Nat.gcd k (Nat.mul m n)) z) " +
			"(Nat.dvd_gcd (Nat.gcd k (Nat.mul m n)) k n (Nat.gcd_dvd_left k (Nat.mul m n)) " +
			"(Nat.Coprime.dvd_of_dvd_mul_left (Nat.gcd k (Nat.mul m n)) m n " +
			"(Nat.Coprime.coprime_dvd_left (Nat.gcd k (Nat.mul m n)) k m (Nat.gcd_dvd_left k (Nat.mul m n)) h1) " +
			"(Nat.gcd_dvd_right k (Nat.mul m n)))) 1 h2)"},
	valDecl{"thm", "Nat.Coprime.mul_left", nil,
		"(m : Nat) → (n : Nat) → (k : Nat) → Nat.Coprime m k → Nat.Coprime n k → Nat.Coprime (Nat.mul m n) k",
		"fun (m : Nat) (n : Nat) (k : Nat) (h1 : Nat.Coprime m k) (h2 : Nat.Coprime n k) => " +
			"Nat.Coprime.symm k (Nat.mul m n) (Nat.Coprime.mul_right k m n (Nat.Coprime.symm m k h1) (Nat.Coprime.symm n k h2))"},
	valDecl{"thm", "Nat.Coprime.pow_left", nil,
		"(m : Nat) → (k : Nat) → (n : Nat) → Nat.Coprime m k → Nat.Coprime (Nat.pow m n) k",
		"fun (m : Nat) (k : Nat) (n : Nat) (h : Nat.Coprime m k) => Nat.rec.{0} " +
			"(fun (n : Nat) => Nat.Coprime (Nat.pow m n) k) (Nat.coprime_one_left k) " +
			"(fun (n : Nat) (ih : Nat.Coprime (Nat.pow m n) k) => Nat.Coprime.mul_left (Nat.pow m n) m k ih h) n"},
	valDecl{"thm", "Nat.Coprime.pow_right", nil,
		"(k : Nat) → (m : Nat) → (n : Nat) → Nat.Coprime k m → Nat.Coprime k (Nat.pow m n)",
		"fun (k : Nat) (m : Nat) (n : Nat) (h : Nat.Coprime k m) => Nat.rec.{0} " +
			"(fun (n : Nat) => Nat.Coprime k (Nat.pow m n)) (Nat.coprime_one_right k) " +
			"(fun (n : Nat) (ih : Nat.Coprime k (Nat.pow m n)) => Nat.Coprime.mul_right k (Nat.pow m n) m ih h) n"},
	valDecl{"thm", "Nat.Coprime.pow", nil,
		"(k : Nat) → (l : Nat) → (m : Nat) → (n : Nat) → Nat.Coprime k l → Nat.Coprime (Nat.pow k m) (Nat.pow l n)",
		"fun (k : Nat) (l : Nat) (m : Nat) (n : Nat) (h : Nat.Coprime k l) => Nat.Coprime.pow_right (Nat.pow k m) l n " +
			"(Nat.Coprime.pow_left k l m h)"},
	valDecl{"thm", "Nat.coprime_mul_iff_left", nil,
		"(m : Nat) → (n : Nat) → (k : Nat) → Iff (Nat.Coprime (Nat.mul m n) k) (And (Nat.Coprime m k) (Nat.Coprime n k))",
		"fun (m : Nat) (n : Nat) (k : Nat) => Iff.intro (Nat.Coprime (Nat.mul m n) k) (And (Nat.Coprime m k) (Nat.Coprime n k)) " +
			"(fun (h : Nat.Coprime (Nat.mul m n) k) => And.intro (Nat.Coprime m k) (Nat.Coprime n k) " +
			"(Nat.Coprime.of_mul_left m n k h) (Nat.Coprime.of_mul_right m n k h)) " +
			"(fun (h : And (Nat.Coprime m k) (Nat.Coprime n k)) => Nat.Coprime.mul_left m n k " +
			"(And.left (Nat.Coprime m k) (Nat.Coprime n k) h) (And.right (Nat.Coprime m k) (Nat.Coprime n k) h))"},
	valDecl{"thm", "Nat.coprime_mul_iff_right", nil,
		"(k : Nat) → (m : Nat) → (n : Nat) → Iff (Nat.Coprime k (Nat.mul m n)) (And (Nat.Coprime k m) (Nat.Coprime k n))",
		"fun (k : Nat) (m : Nat) (n : Nat) => Iff.intro (Nat.Coprime k (Nat.mul m n)) (And (Nat.Coprime k m) (Nat.Coprime k n)) " +
			"(fun (h : Nat.Coprime k (Nat.mul m n)) => And.intro (Nat.Coprime k m) (Nat.Coprime k n) " +
			"(Nat.Coprime.coprime_mul_right_right k m n h) (Nat.Coprime.coprime_mul_left_right k m n h)) " +
			"(fun (h : And (Nat.Coprime k m) (Nat.Coprime k n)) => Nat.Coprime.mul_right k m n " +
			"(And.left (Nat.Coprime k m) (Nat.Coprime k n) h) (And.right (Nat.Coprime k m) (Nat.Coprime k n) h))"},
	valDecl{"def", "Nat.decDvd", nil, "(m : Nat) → (n : Nat) → Decidable (Nat.dvd m n)",
		"fun (m : Nat) (n : Nat) => Decidable.rec.{1} (Eq.{1} Nat (Nat.mod n m) 0) (fun (_ : Decidable (Eq.{1} Nat (Nat.mod n m) 0)) => Decidable (Nat.dvd m n)) " +
			"(fun (h : Not (Eq.{1} Nat (Nat.mod n m) 0)) => Decidable.isFalse (Nat.dvd m n) (fun (hd : Nat.dvd m n) => h (Iff.mp (Nat.dvd m n) (Eq.{1} Nat (Nat.mod n m) 0) (Nat.dvd_iff_mod_eq_zero m n) hd))) " +
			"(fun (h : Eq.{1} Nat (Nat.mod n m) 0) => Decidable.isTrue (Nat.dvd m n) (Iff.mpr (Nat.dvd m n) (Eq.{1} Nat (Nat.mod n m) 0) (Nat.dvd_iff_mod_eq_zero m n) h)) " +
			"(Nat.decEq (Nat.mod n m) 0)"},
	valDecl{"def", "instDecidableNot", nil, "(p : Prop) → Decidable p → Decidable (Not p)",
		"fun (p : Prop) (dp : Decidable p) => Decidable.rec.{1} p (fun (_ : Decidable p) => Decidable (Not p)) " +
			"(fun (hp : Not p) => Decidable.isTrue (Not p) hp) " +
			"(fun (hp : p) => Decidable.isFalse (Not p) (fun (hn : Not p) => hn hp)) dp"},
	valDecl{"def", "instDecidableAnd", nil, "(p : Prop) → (q : Prop) → Decidable p → Decidable q → Decidable (And p q)",
		"fun (p : Prop) (q : Prop) (dp : Decidable p) (dq : Decidable q) => Decidable.rec.{1} p (fun (_ : Decidable p) => Decidable (And p q)) " +
			"(fun (hp : Not p) => Decidable.isFalse (And p q) (fun (h : And p q) => hp (And.left p q h))) " +
			"(fun (hp : p) => Decidable.rec.{1} q (fun (_ : Decidable q) => Decidable (And p q)) " +
			"(fun (hq : Not q) => Decidable.isFalse (And p q) (fun (h : And p q) => hq (And.right p q h))) " +
			"(fun (hq : q) => Decidable.isTrue (And p q) (And.intro p q hp hq)) dq) dp"},
	valDecl{"def", "instDecidableOr", nil, "(p : Prop) → (q : Prop) → Decidable p → Decidable q → Decidable (Or p q)",
		"fun (p : Prop) (q : Prop) (dp : Decidable p) (dq : Decidable q) => Decidable.rec.{1} p (fun (_ : Decidable p) => Decidable (Or p q)) " +
			"(fun (hp : Not p) => Decidable.rec.{1} q (fun (_ : Decidable q) => Decidable (Or p q)) " +
			"(fun (hq : Not q) => Decidable.isFalse (Or p q) (fun (h : Or p q) => Or.rec p q (fun (_ : Or p q) => False) (fun (x : p) => hp x) (fun (y : q) => hq y) h)) " +
			"(fun (hq : q) => Decidable.isTrue (Or p q) (Or.inr p q hq)) dq) " +
			"(fun (hp : p) => Decidable.isTrue (Or p q) (Or.inl p q hp)) dp"},

	// Int: the two-constructor type of Init/Data/Int/Basic.lean, its arithmetic, and the decision procedures the
	// claims use. Lean writes these operations as match expressions; here they are recursor/Bool.rec applications with
	// the same values on every input. `Int.ctorNat`, `Int.ofNat.inj`, `Int.negSucc.inj` and the two cross-constructor
	// lemmas are port helpers: Lean derives them from `Int.noConfusion` (see `Int.decEq`).
	indDecl{"Int", nil, 0, "Type", [][2]string{{"Int.ofNat", "Nat → Int"}, {"Int.negSucc", "Nat → Int"}}},
	// NonNeg : Int → Prop is an indexed family; `Int.le` is `NonNeg (b - a)`, exactly as in Lean
	indDecl{"Int.NonNeg", nil, 0, "Int → Prop", [][2]string{{"Int.NonNeg.mk", "(n : Nat) → Int.NonNeg (Int.ofNat n)"}}},
	valDecl{"def", "Int.subNatNat", nil, "Nat → Nat → Int",
		"fun (m : Nat) (n : Nat) => Bool.rec.{1} (fun (_ : Bool) => Int) " +
			"(Int.negSucc (Nat.sub n (Nat.succ m))) " +
			"(Int.ofNat (Nat.sub m n)) " +
			"(Nat.ble n m)"},
	valDecl{"def", "Int.negOfNat", nil, "Nat → Int",
		"fun (n : Nat) => Bool.rec.{1} (fun (_ : Bool) => Int) " +
			"(Int.negSucc (Nat.pred n)) " +
			"(Int.ofNat 0) " +
			"(Nat.beq n 0)"},
	valDecl{"def", "Int.neg", nil, "Int → Int",
		"fun (n : Int) => Int.rec.{1} (fun (_ : Int) => Int) " +
			"(fun (k : Nat) => Int.negOfNat k) " +
			"(fun (k : Nat) => Int.ofNat (Nat.succ k)) n"},
	valDecl{"def", "Int.add", nil, "Int → Int → Int",
		"fun (m : Int) (n : Int) => Int.rec.{1} (fun (_ : Int) => Int → Int) " +
			"(fun (a : Nat) => Int.rec.{1} (fun (_ : Int) => Int) " +
			"(fun (b : Nat) => Int.ofNat (Nat.add a b)) " +
			"(fun (b : Nat) => Int.subNatNat a (Nat.succ b))) " +
			"(fun (a : Nat) => Int.rec.{1} (fun (_ : Int) => Int) " +
			"(fun (b : Nat) => Int.subNatNat b (Nat.succ a)) " +
			"(fun (b : Nat) => Int.negSucc (Nat.succ (Nat.add a b)))) " +
			"m n"},
	valDecl{"def", "Int.mul", nil, "Int → Int → Int",
		"fun (m : Int) (n : Int) => Int.rec.{1} (fun (_ : Int) => Int → Int) " +
			"(fun (a : Nat) => Int.rec.{1} (fun (_ : Int) => Int) " +
			"(fun (b : Nat) => Int.ofNat (Nat.mul a b)) " +
			"(fun (b : Nat) => Int.negOfNat (Nat.mul a (Nat.succ b)))) " +
			"(fun (a : Nat) => Int.rec.{1} (fun (_ : Int) => Int) " +
			"(fun (b : Nat) => Int.negOfNat (Nat.mul (Nat.succ a) b)) " +
			"(fun (b : Nat) => Int.ofNat (Nat.mul (Nat.succ a) (Nat.succ b)))) " +
			"m n"},
	valDecl{"def", "Int.sub", nil, "Int → Int → Int", "fun (m : Int) (n : Int) => Int.add m (Int.neg n)"},
	valDecl{"def", "Int.natAbs", nil, "Int → Nat",
		"fun (n : Int) => Int.rec.{1} (fun (_ : Int) => Nat) (fun (k : Nat) => k) (fun (k : Nat) => Nat.succ k) n"},
	// Int.pow branches on the constructor and on `n % 2` as Lean's model does; the parity test is a Bool.rec here
	// instead of an `ite`, and Nat.pow's literal extension makes a big power one step.
	valDecl{"def", "Int.pow", nil, "Int → Nat → Int",
		"fun (n : Int) (m : Nat) => Int.rec.{1} (fun (_ : Int) => Int) " +
			"(fun (a : Nat) => Int.ofNat (Nat.pow a m)) " +
			"(fun (a : Nat) => Bool.rec.{1} (fun (_ : Bool) => Int) " +
			"(Int.neg (Int.ofNat (Nat.pow (Int.natAbs (Int.negSucc a)) m))) " +
			"(Int.ofNat (Nat.pow (Int.natAbs (Int.negSucc a)) m)) " +
			"(Nat.beq (Nat.mod m 2) 0)) " +
			"n"},
	valDecl{"def", "Int.le", nil, "Int → Int → Prop", "fun (a : Int) (b : Int) => Int.NonNeg (Int.sub b a)"},
	valDecl{"def", "Int.lt", nil, "Int → Int → Prop", "fun (a : Int) (b : Int) => Int.le (Int.add a (Int.ofNat 1)) b"},
	// Int.ediv and Int.emod are Lean's `/` and `%` on Int (Init/Data/Int/DivMod/Basic.lean): E-rounding, so
	// `emod x y` is non-negative and `x = y * (x / y) + x % y`; `x / 0 = 0` and `x % 0 = x`. The `match` bodies are
	// recursor/Bool.rec terms here, and Nat.div/Nat.mod's literal extension keeps a big literal quotient one step.
	valDecl{"def", "Int.ediv", nil, "Int → Int → Int",
		"fun (m : Int) (n : Int) => Int.rec.{1} (fun (_ : Int) => Int → Int) " +
			"(fun (a : Nat) => Int.rec.{1} (fun (_ : Int) => Int) " +
			"(fun (b : Nat) => Int.ofNat (Nat.div a b)) " +
			"(fun (b : Nat) => Int.negOfNat (Nat.div a (Nat.succ b)))) " +
			"(fun (a : Nat) => Int.rec.{1} (fun (_ : Int) => Int) " +
			"(fun (b : Nat) => Bool.rec.{1} (fun (_ : Bool) => Int) " +
			"(Int.negSucc (Nat.div a b)) (Int.ofNat 0) (Nat.beq b 0)) " +
			"(fun (b : Nat) => Int.ofNat (Nat.succ (Nat.div a (Nat.succ b))))) " +
			"m n"},
	valDecl{"def", "Int.emod", nil, "Int → Int → Int",
		"fun (m : Int) (n : Int) => Int.rec.{1} (fun (_ : Int) => Int → Int) " +
			"(fun (a : Nat) => fun (n : Int) => Int.ofNat (Nat.mod a (Int.natAbs n))) " +
			"(fun (a : Nat) => fun (n : Int) => Int.subNatNat (Int.natAbs n) (Nat.succ (Nat.mod a (Int.natAbs n)))) " +
			"m n"},

	valDecl{"def", "Int.ctorNat", nil, "Int → Nat",
		"fun (n : Int) => Int.rec.{1} (fun (_ : Int) => Nat) (fun (k : Nat) => k) (fun (k : Nat) => k) n"},
	valDecl{"def", "Int.isNat", nil, "Int → Bool",
		"fun (n : Int) => Int.rec.{1} (fun (_ : Int) => Bool) (fun (k : Nat) => Bool.true) (fun (k : Nat) => Bool.false) n"},
	valDecl{"thm", "Int.ofNat.inj", nil, "(a : Nat) → (b : Nat) → Eq.{1} Int (Int.ofNat a) (Int.ofNat b) → Eq.{1} Nat a b",
		"fun (a : Nat) (b : Nat) (h : Eq.{1} Int (Int.ofNat a) (Int.ofNat b)) => " +
			"Eq.rec.{0, 1} Int (Int.ofNat a) (fun (x : Int) (_ : Eq.{1} Int (Int.ofNat a) x) => Eq.{1} Nat a (Int.ctorNat x)) " +
			"(Eq.refl.{1} Nat a) (Int.ofNat b) h"},
	valDecl{"thm", "Int.negSucc.inj", nil, "(a : Nat) → (b : Nat) → Eq.{1} Int (Int.negSucc a) (Int.negSucc b) → Eq.{1} Nat a b",
		"fun (a : Nat) (b : Nat) (h : Eq.{1} Int (Int.negSucc a) (Int.negSucc b)) => " +
			"Eq.rec.{0, 1} Int (Int.negSucc a) (fun (x : Int) (_ : Eq.{1} Int (Int.negSucc a) x) => Eq.{1} Nat a (Int.ctorNat x)) " +
			"(Eq.refl.{1} Nat a) (Int.negSucc b) h"},
	valDecl{"thm", "Int.ofNat_ne_negSucc", nil, "(a : Nat) → (b : Nat) → Not (Eq.{1} Int (Int.ofNat a) (Int.negSucc b))",
		"fun (a : Nat) (b : Nat) (h : Eq.{1} Int (Int.ofNat a) (Int.negSucc b)) => " +
			"Bool.true_ne_false (congrArg.{1, 1} Int Bool Int.isNat (Int.ofNat a) (Int.negSucc b) h)"},
	valDecl{"thm", "Int.negSucc_ne_ofNat", nil, "(a : Nat) → (b : Nat) → Not (Eq.{1} Int (Int.negSucc a) (Int.ofNat b))",
		"fun (a : Nat) (b : Nat) (h : Eq.{1} Int (Int.negSucc a) (Int.ofNat b)) => " +
			"Bool.false_ne_true (congrArg.{1, 1} Int Bool Int.isNat (Int.negSucc a) (Int.ofNat b) h)"},
	valDecl{"def", "Int.decNonneg", nil, "(m : Int) → Decidable (Int.NonNeg m)",
		"fun (m : Int) => Int.rec.{1} (fun (x : Int) => Decidable (Int.NonNeg x)) " +
			"(fun (k : Nat) => Decidable.isTrue (Int.NonNeg (Int.ofNat k)) (Int.NonNeg.mk k)) " +
			"(fun (k : Nat) => Decidable.isFalse (Int.NonNeg (Int.negSucc k)) (fun (h : Int.NonNeg (Int.negSucc k)) => " +
			"Int.NonNeg.rec (fun (j : Int) (_ : Int.NonNeg j) => Eq.{1} Int j (Int.negSucc k) → False) " +
			"(fun (j : Nat) (e : Eq.{1} Int (Int.ofNat j) (Int.negSucc k)) => " +
			"Bool.true_ne_false (congrArg.{1, 1} Int Bool Int.isNat (Int.ofNat j) (Int.negSucc k) e)) " +
			"(Int.negSucc k) h (Eq.refl.{1} Int (Int.negSucc k)))) " +
			"m"},
	valDecl{"def", "Int.decEq", nil, "(a : Int) → (b : Int) → Decidable (Eq.{1} Int a b)",
		"fun (a : Int) (b : Int) => Int.rec.{1} (fun (x : Int) => Decidable (Eq.{1} Int x b)) " +
			"(fun (na : Nat) => Int.rec.{1} (fun (y : Int) => Decidable (Eq.{1} Int (Int.ofNat na) y)) " +
			"(fun (nb : Nat) => Decidable.rec.{1} (Eq.{1} Nat na nb) " +
			"(fun (_ : Decidable (Eq.{1} Nat na nb)) => Decidable (Eq.{1} Int (Int.ofNat na) (Int.ofNat nb))) " +
			"(fun (h : Not (Eq.{1} Nat na nb)) => Decidable.isFalse (Eq.{1} Int (Int.ofNat na) (Int.ofNat nb)) " +
			"(fun (h' : Eq.{1} Int (Int.ofNat na) (Int.ofNat nb)) => h (Int.ofNat.inj na nb h'))) " +
			"(fun (h : Eq.{1} Nat na nb) => Decidable.isTrue (Eq.{1} Int (Int.ofNat na) (Int.ofNat nb)) " +
			"(congrArg.{1, 1} Nat Int Int.ofNat na nb h)) " +
			"(Nat.decEq na nb)) " +
			"(fun (nb : Nat) => Decidable.isFalse (Eq.{1} Int (Int.ofNat na) (Int.negSucc nb)) (Int.ofNat_ne_negSucc na nb)) b) " +
			"(fun (na : Nat) => Int.rec.{1} (fun (y : Int) => Decidable (Eq.{1} Int (Int.negSucc na) y)) " +
			"(fun (nb : Nat) => Decidable.isFalse (Eq.{1} Int (Int.negSucc na) (Int.ofNat nb)) (Int.negSucc_ne_ofNat na nb)) " +
			"(fun (nb : Nat) => Decidable.rec.{1} (Eq.{1} Nat na nb) " +
			"(fun (_ : Decidable (Eq.{1} Nat na nb)) => Decidable (Eq.{1} Int (Int.negSucc na) (Int.negSucc nb))) " +
			"(fun (h : Not (Eq.{1} Nat na nb)) => Decidable.isFalse (Eq.{1} Int (Int.negSucc na) (Int.negSucc nb)) " +
			"(fun (h' : Eq.{1} Int (Int.negSucc na) (Int.negSucc nb)) => h (Int.negSucc.inj na nb h'))) " +
			"(fun (h : Eq.{1} Nat na nb) => Decidable.isTrue (Eq.{1} Int (Int.negSucc na) (Int.negSucc nb)) " +
			"(congrArg.{1, 1} Nat Int Int.negSucc na nb h)) " +
			"(Nat.decEq na nb)) b) " +
			"a"},
	valDecl{"def", "Int.decLe", nil, "(a : Int) → (b : Int) → Decidable (Int.le a b)",
		"fun (a : Int) (b : Int) => Int.decNonneg (Int.sub b a)"},
	valDecl{"def", "Int.decLt", nil, "(a : Int) → (b : Int) → Decidable (Int.lt a b)",
		"fun (a : Int) (b : Int) => Int.decLe (Int.add a (Int.ofNat 1)) b"},

	// Int.dvd is Lean's `Dvd Int` instance (Basic.lean: `dvd a b := Exists (fun c => b = a * c)`).
	valDecl{"def", "Int.dvd", nil, "Int → Int → Prop",
		"fun (a : Int) (b : Int) => Exists.{1} Int (fun (c : Int) => Eq.{1} Int b (Int.mul a c))"},

	// Deciding Int.dvd. Lean decides it through `dvd_iff_emod_eq_zero` (DivMod/Bootstrap.lean), proved there from
	// `mul_ediv_cancel_of_emod_eq_zero`. `natAbs_dvd_natAbs` (same file, line 66) states the same decision without
	// mod, and it follows from `natAbs_mul` and constructor cases alone, so `Int.decDvd` goes that way and needs no
	// div/mod lemma, only `Nat.decDvd`.
	valDecl{"thm", "Int.natAbs_negOfNat", nil, "(n : Nat) → Eq.{1} Nat (Int.natAbs (Int.negOfNat n)) n",
		"fun (n : Nat) => Nat.rec.{0} (fun (k : Nat) => Eq.{1} Nat (Int.natAbs (Int.negOfNat k)) k) " +
			"(Eq.refl.{1} Nat 0) (fun (j : Nat) (_ : Eq.{1} Nat (Int.natAbs (Int.negOfNat j)) j) => " +
			"Eq.refl.{1} Nat (Nat.succ j)) n"},
	// `natAbs` of a constructor, and of a negation (`Int.natAbs_neg`: `negOfNat` handles `ofNat k`, `negSucc k` is a successor).
	valDecl{"thm", "Int.natAbs_ofNat", nil, "(n : Nat) → Eq.{1} Nat (Int.natAbs (Int.ofNat n)) n", "fun (n : Nat) => Eq.refl.{1} Nat n"},
	valDecl{"thm", "Int.natAbs_negSucc", nil, "(n : Nat) → Eq.{1} Nat (Int.natAbs (Int.negSucc n)) (Nat.succ n)",
		"fun (n : Nat) => Eq.refl.{1} Nat (Nat.succ n)"},
	valDecl{"thm", "Int.natAbs_neg", nil, "(a : Int) → Eq.{1} Nat (Int.natAbs (Int.neg a)) (Int.natAbs a)",
		"fun (a : Int) => Int.rec.{0} (fun (a : Int) => Eq.{1} Nat (Int.natAbs (Int.neg a)) (Int.natAbs a)) " +
			"(fun (k : Nat) => Int.natAbs_negOfNat k) " +
			"(fun (k : Nat) => Eq.refl.{1} Nat (Nat.succ k)) a"},
	// `- -a = a`: cases on `a`, and on `k` for `ofNat k` (`negOfNat 0` is `ofNat 0`, `negOfNat (succ j)` is `negSucc j`); each is `rfl`.
	valDecl{"thm", "Int.neg_neg", nil, "(a : Int) → Eq.{1} Int (Int.neg (Int.neg a)) a",
		"fun (a : Int) => Int.rec.{0} (fun (a : Int) => Eq.{1} Int (Int.neg (Int.neg a)) a) " +
			"(fun (k : Nat) => Nat.rec.{0} (fun (k : Nat) => Eq.{1} Int (Int.neg (Int.neg (Int.ofNat k))) (Int.ofNat k)) " +
			"(Eq.refl.{1} Int (Int.ofNat 0)) " +
			"(fun (j : Nat) (_ : Eq.{1} Int (Int.neg (Int.neg (Int.ofNat j))) (Int.ofNat j)) => Eq.refl.{1} Int (Int.ofNat (Nat.succ j))) k) " +
			"(fun (k : Nat) => Eq.refl.{1} Int (Int.negSucc k)) a"},
	// `natAbs a = 0 ↔ a = 0`: forwards by cases (`ofNat k` with `k = 0`; `negSucc k` has a successor as `natAbs`); backwards `congrArg natAbs`.
	valDecl{"thm", "Int.natAbs_eq_zero", nil,
		"(a : Int) → Iff (Eq.{1} Nat (Int.natAbs a) 0) (Eq.{1} Int a (Int.ofNat 0))",
		"fun (a : Int) => Iff.intro (Eq.{1} Nat (Int.natAbs a) 0) (Eq.{1} Int a (Int.ofNat 0)) " +
			"(Int.rec.{0} (fun (a : Int) => Eq.{1} Nat (Int.natAbs a) 0 → Eq.{1} Int a (Int.ofNat 0)) " +
			"(fun (k : Nat) (h : Eq.{1} Nat k 0) => congrArg.{1, 1} Nat Int Int.ofNat k 0 h) " +
			"(fun (k : Nat) (h : Eq.{1} Nat (Nat.succ k) 0) => False.elim.{0} (Eq.{1} Int (Int.negSucc k) (Int.ofNat 0)) (Nat.succ_ne_zero k h)) a) " +
			"(fun (h : Eq.{1} Int a (Int.ofNat 0)) => congrArg.{1, 1} Int Nat Int.natAbs a (Int.ofNat 0) h)"},
	valDecl{"thm", "Int.natAbs_mul", nil,
		"(a : Int) → (b : Int) → Eq.{1} Nat (Int.natAbs (Int.mul a b)) (Nat.mul (Int.natAbs a) (Int.natAbs b))",
		"fun (a : Int) (b : Int) => Int.rec.{0} " +
			"(fun (x : Int) => Eq.{1} Nat (Int.natAbs (Int.mul x b)) (Nat.mul (Int.natAbs x) (Int.natAbs b))) " +
			"(fun (m : Nat) => Int.rec.{0} " +
			"(fun (y : Int) => Eq.{1} Nat (Int.natAbs (Int.mul (Int.ofNat m) y)) (Nat.mul m (Int.natAbs y))) " +
			"(fun (n : Nat) => Eq.refl.{1} Nat (Nat.mul m n)) " +
			"(fun (n : Nat) => Int.natAbs_negOfNat (Nat.mul m (Nat.succ n))) b) " +
			"(fun (m : Nat) => Int.rec.{0} " +
			"(fun (y : Int) => Eq.{1} Nat (Int.natAbs (Int.mul (Int.negSucc m) y)) (Nat.mul (Nat.succ m) (Int.natAbs y))) " +
			"(fun (n : Nat) => Int.natAbs_negOfNat (Nat.mul (Nat.succ m) n)) " +
			"(fun (n : Nat) => Eq.refl.{1} Nat (Nat.mul (Nat.succ m) (Nat.succ n))) b) " +
			"a"},

	// The four sign cases of `Nat.dvd (natAbs a) (natAbs b) -> Int.dvd a b`; k = 0 is impossible where the dividend is
	// a successor, and k = succ j builds the witness from j and one transport.
	valDecl{"thm", "Int.dvd_ofNat_ofNat", nil,
		"(m : Nat) → (n : Nat) → Nat.dvd m n → Int.dvd (Int.ofNat m) (Int.ofNat n)",
		"fun (m : Nat) (n : Nat) (h : Nat.dvd m n) => " +
			"Exists.rec.{1} Nat (fun (k : Nat) => Eq.{1} Nat n (Nat.mul m k)) " +
			"(fun (_ : Exists.{1} Nat (fun (k : Nat) => Eq.{1} Nat n (Nat.mul m k))) => Int.dvd (Int.ofNat m) (Int.ofNat n)) " +
			"(fun (k : Nat) (hk : Eq.{1} Nat n (Nat.mul m k)) => " +
			"Exists.intro.{1} Int (fun (c : Int) => Eq.{1} Int (Int.ofNat n) (Int.mul (Int.ofNat m) c)) (Int.ofNat k) " +
			"(congrArg.{1, 1} Nat Int Int.ofNat n (Nat.mul m k) hk)) h"},
	valDecl{"thm", "Int.dvd_ofNat_negSucc", nil,
		"(m : Nat) → (n : Nat) → Nat.dvd m (Nat.succ n) → Int.dvd (Int.ofNat m) (Int.negSucc n)",
		"fun (m : Nat) (n : Nat) (h : Nat.dvd m (Nat.succ n)) => " +
			"Exists.rec.{1} Nat (fun (k : Nat) => Eq.{1} Nat (Nat.succ n) (Nat.mul m k)) " +
			"(fun (_ : Exists.{1} Nat (fun (k : Nat) => Eq.{1} Nat (Nat.succ n) (Nat.mul m k))) => Int.dvd (Int.ofNat m) (Int.negSucc n)) " +
			"(fun (k : Nat) (hk : Eq.{1} Nat (Nat.succ n) (Nat.mul m k)) => " +
			"Nat.rec.{0} (fun (j : Nat) => Eq.{1} Nat (Nat.succ n) (Nat.mul m j) → Int.dvd (Int.ofNat m) (Int.negSucc n)) " +
			"(fun (h0 : Eq.{1} Nat (Nat.succ n) (Nat.mul m 0)) => False.elim.{0} (Int.dvd (Int.ofNat m) (Int.negSucc n)) " +
			"(Nat.ne_of_beq_eq_false (Nat.succ n) 0 (Eq.refl.{1} Bool Bool.false) h0)) " +
			"(fun (j : Nat) (_ : Eq.{1} Nat (Nat.succ n) (Nat.mul m j) → Int.dvd (Int.ofNat m) (Int.negSucc n)) " +
			"(hk' : Eq.{1} Nat (Nat.succ n) (Nat.mul m (Nat.succ j))) => " +
			"Exists.intro.{1} Int (fun (c : Int) => Eq.{1} Int (Int.negSucc n) (Int.mul (Int.ofNat m) c)) (Int.negSucc j) " +
			"(Eq.rec.{0, 1} Nat (Nat.succ n) (fun (x : Nat) (_ : Eq.{1} Nat (Nat.succ n) x) => Eq.{1} Int (Int.negSucc n) (Int.negOfNat x)) " +
			"(Eq.refl.{1} Int (Int.negSucc n)) (Nat.mul m (Nat.succ j)) hk')) k hk) h"},
	valDecl{"thm", "Int.dvd_negSucc_ofNat", nil,
		"(m : Nat) → (n : Nat) → Nat.dvd (Nat.succ m) n → Int.dvd (Int.negSucc m) (Int.ofNat n)",
		"fun (m : Nat) (n : Nat) (h : Nat.dvd (Nat.succ m) n) => " +
			"Exists.rec.{1} Nat (fun (k : Nat) => Eq.{1} Nat n (Nat.mul (Nat.succ m) k)) " +
			"(fun (_ : Exists.{1} Nat (fun (k : Nat) => Eq.{1} Nat n (Nat.mul (Nat.succ m) k))) => Int.dvd (Int.negSucc m) (Int.ofNat n)) " +
			"(fun (k : Nat) (hk : Eq.{1} Nat n (Nat.mul (Nat.succ m) k)) => " +
			"Nat.rec.{0} (fun (j : Nat) => Eq.{1} Nat n (Nat.mul (Nat.succ m) j) → Int.dvd (Int.negSucc m) (Int.ofNat n)) " +
			"(fun (h0 : Eq.{1} Nat n (Nat.mul (Nat.succ m) 0)) => " +
			"Exists.intro.{1} Int (fun (c : Int) => Eq.{1} Int (Int.ofNat n) (Int.mul (Int.negSucc m) c)) (Int.ofNat 0) " +
			"(congrArg.{1, 1} Nat Int Int.ofNat n 0 h0)) " +
			"(fun (j : Nat) (_ : Eq.{1} Nat n (Nat.mul (Nat.succ m) j) → Int.dvd (Int.negSucc m) (Int.ofNat n)) " +
			"(hk' : Eq.{1} Nat n (Nat.mul (Nat.succ m) (Nat.succ j))) => " +
			"Exists.intro.{1} Int (fun (c : Int) => Eq.{1} Int (Int.ofNat n) (Int.mul (Int.negSucc m) c)) (Int.negSucc j) " +
			"(congrArg.{1, 1} Nat Int Int.ofNat n (Nat.mul (Nat.succ m) (Nat.succ j)) hk')) k hk) h"},
	valDecl{"thm", "Int.dvd_negSucc_negSucc", nil,
		"(m : Nat) → (n : Nat) → Nat.dvd (Nat.succ m) (Nat.succ n) → Int.dvd (Int.negSucc m) (Int.negSucc n)",
		"fun (m : Nat) (n : Nat) (h : Nat.dvd (Nat.succ m) (Nat.succ n)) => " +
			"Exists.rec.{1} Nat (fun (k : Nat) => Eq.{1} Nat (Nat.succ n) (Nat.mul (Nat.succ m) k)) " +
			"(fun (_ : Exists.{1} Nat (fun (k : Nat) => Eq.{1} Nat (Nat.succ n) (Nat.mul (Nat.succ m) k))) => Int.dvd (Int.negSucc m) (Int.negSucc n)) " +
			"(fun (k : Nat) (hk : Eq.{1} Nat (Nat.succ n) (Nat.mul (Nat.succ m) k)) => " +
			"Nat.rec.{0} (fun (j : Nat) => Eq.{1} Nat (Nat.succ n) (Nat.mul (Nat.succ m) j) → Int.dvd (Int.negSucc m) (Int.negSucc n)) " +
			"(fun (h0 : Eq.{1} Nat (Nat.succ n) (Nat.mul (Nat.succ m) 0)) => False.elim.{0} (Int.dvd (Int.negSucc m) (Int.negSucc n)) " +
			"(Nat.ne_of_beq_eq_false (Nat.succ n) 0 (Eq.refl.{1} Bool Bool.false) h0)) " +
			"(fun (j : Nat) (_ : Eq.{1} Nat (Nat.succ n) (Nat.mul (Nat.succ m) j) → Int.dvd (Int.negSucc m) (Int.negSucc n)) " +
			"(hk' : Eq.{1} Nat (Nat.succ n) (Nat.mul (Nat.succ m) (Nat.succ j))) => " +
			"Exists.intro.{1} Int (fun (c : Int) => Eq.{1} Int (Int.negSucc n) (Int.mul (Int.negSucc m) c)) (Int.ofNat (Nat.succ j)) " +
			"(Eq.rec.{0, 1} Nat (Nat.succ n) (fun (x : Nat) (_ : Eq.{1} Nat (Nat.succ n) x) => Eq.{1} Int (Int.negSucc n) (Int.negOfNat x)) " +
			"(Eq.refl.{1} Int (Int.negSucc n)) (Nat.mul (Nat.succ m) (Nat.succ j)) hk')) k hk) h"},
	valDecl{"thm", "Int.natAbs_dvd_natAbs", nil,
		"(a : Int) → (b : Int) → Iff (Nat.dvd (Int.natAbs a) (Int.natAbs b)) (Int.dvd a b)",
		"fun (a : Int) (b : Int) => Iff.intro (Nat.dvd (Int.natAbs a) (Int.natAbs b)) (Int.dvd a b) " +
			"(fun (h : Nat.dvd (Int.natAbs a) (Int.natAbs b)) => Int.rec.{0} " +
			"(fun (x : Int) => Nat.dvd (Int.natAbs x) (Int.natAbs b) → Int.dvd x b) " +
			"(fun (m : Nat) => Int.rec.{0} (fun (y : Int) => Nat.dvd m (Int.natAbs y) → Int.dvd (Int.ofNat m) y) " +
			"(fun (n : Nat) (h' : Nat.dvd m n) => Int.dvd_ofNat_ofNat m n h') " +
			"(fun (n : Nat) (h' : Nat.dvd m (Nat.succ n)) => Int.dvd_ofNat_negSucc m n h') b) " +
			"(fun (m : Nat) => Int.rec.{0} (fun (y : Int) => Nat.dvd (Nat.succ m) (Int.natAbs y) → Int.dvd (Int.negSucc m) y) " +
			"(fun (n : Nat) (h' : Nat.dvd (Nat.succ m) n) => Int.dvd_negSucc_ofNat m n h') " +
			"(fun (n : Nat) (h' : Nat.dvd (Nat.succ m) (Nat.succ n)) => Int.dvd_negSucc_negSucc m n h') b) a h) " +
			"(fun (h : Int.dvd a b) => Exists.rec.{1} Int (fun (c : Int) => Eq.{1} Int b (Int.mul a c)) " +
			"(fun (_ : Exists.{1} Int (fun (c : Int) => Eq.{1} Int b (Int.mul a c))) => Nat.dvd (Int.natAbs a) (Int.natAbs b)) " +
			"(fun (c : Int) (e : Eq.{1} Int b (Int.mul a c)) => " +
			"Exists.intro.{1} Nat (fun (d : Nat) => Eq.{1} Nat (Int.natAbs b) (Nat.mul (Int.natAbs a) d)) (Int.natAbs c) " +
			"(Eq.trans.{1} Nat (Int.natAbs b) (Int.natAbs (Int.mul a c)) (Nat.mul (Int.natAbs a) (Int.natAbs c)) " +
			"(congrArg.{1, 1} Int Nat Int.natAbs b (Int.mul a c) e) (Int.natAbs_mul a c))) h)"},
	// Int divisibility through `natAbs_dvd_natAbs`: `dvd_refl`, `dvd_trans`, and `dvd_neg` (`natAbs (-b) = natAbs b`).
	valDecl{"thm", "Int.dvd_refl", nil,
		"(a : Int) → Int.dvd a a",
		"fun (a : Int) => Iff.mp (Nat.dvd (Int.natAbs a) (Int.natAbs a)) (Int.dvd a a) (Int.natAbs_dvd_natAbs a a) (Nat.dvd_refl (Int.natAbs a))"},
	valDecl{"thm", "Int.dvd_trans", nil,
		"(a : Int) → (b : Int) → (c : Int) → Int.dvd a b → Int.dvd b c → Int.dvd a c",
		"fun (a : Int) (b : Int) (c : Int) (h1 : Int.dvd a b) (h2 : Int.dvd b c) => Iff.mp (Nat.dvd (Int.natAbs a) (Int.natAbs c)) (Int.dvd a c) " +
			"(Int.natAbs_dvd_natAbs a c) (Nat.dvd_trans (Int.natAbs a) (Int.natAbs b) (Int.natAbs c) (Iff.mpr (Nat.dvd (Int.natAbs a) (Int.natAbs b)) " +
			"(Int.dvd a b) (Int.natAbs_dvd_natAbs a b) h1) (Iff.mpr (Nat.dvd (Int.natAbs b) (Int.natAbs c)) (Int.dvd b c) (Int.natAbs_dvd_natAbs b c) " +
			"h2))"},
	valDecl{"thm", "Int.dvd_neg", nil,
		"(a : Int) → (b : Int) → Iff (Int.dvd a (Int.neg b)) (Int.dvd a b)",
		"fun (a : Int) (b : Int) => Iff.intro (Int.dvd a (Int.neg b)) (Int.dvd a b) (fun (h : Int.dvd a (Int.neg b)) => Iff.mp (Nat.dvd (Int.natAbs " +
			"a) (Int.natAbs b)) (Int.dvd a b) (Int.natAbs_dvd_natAbs a b) (Eq.rec.{0, 1} Nat (Int.natAbs (Int.neg b)) (fun (z : Nat) (_ : Eq.{1} Nat " +
			"(Int.natAbs (Int.neg b)) z) => Nat.dvd (Int.natAbs a) z) (Iff.mpr (Nat.dvd (Int.natAbs a) (Int.natAbs (Int.neg b))) (Int.dvd a (Int.neg b)) " +
			"(Int.natAbs_dvd_natAbs a (Int.neg b)) h) (Int.natAbs b) (Int.natAbs_neg b))) (fun (h : Int.dvd a b) => Iff.mp (Nat.dvd (Int.natAbs a) " +
			"(Int.natAbs (Int.neg b))) (Int.dvd a (Int.neg b)) (Int.natAbs_dvd_natAbs a (Int.neg b)) (Eq.rec.{0, 1} Nat (Int.natAbs b) (fun (z : Nat) (_ " +
			": Eq.{1} Nat (Int.natAbs b) z) => Nat.dvd (Int.natAbs a) z) (Iff.mpr (Nat.dvd (Int.natAbs a) (Int.natAbs b)) (Int.dvd a b) " +
			"(Int.natAbs_dvd_natAbs a b) h) (Int.natAbs (Int.neg b)) (Eq.symm.{1} Nat (Int.natAbs (Int.neg b)) (Int.natAbs b) (Int.natAbs_neg b))))"},
	// More Int divisibility through `natAbs_dvd_natAbs`, with `natAbs_mul` and `natAbs_neg` transporting the Nat lemma.
	valDecl{"thm", "Int.dvd_mul_right", nil,
		"(a : Int) → (b : Int) → Int.dvd a (Int.mul a b)",
		"fun (a : Int) (b : Int) => Iff.mp (Nat.dvd (Int.natAbs a) (Int.natAbs (Int.mul a b))) (Int.dvd a (Int.mul a b)) (Int.natAbs_dvd_natAbs a " +
			"(Int.mul a b)) (Eq.rec.{0, 1} Nat (Nat.mul (Int.natAbs a) (Int.natAbs b)) (fun (z : Nat) (_ : Eq.{1} Nat (Nat.mul (Int.natAbs a) (Int.natAbs " +
			"b)) z) => Nat.dvd (Int.natAbs a) z) (Nat.dvd_mul_right (Int.natAbs a) (Int.natAbs b)) (Int.natAbs (Int.mul a b)) (Eq.symm.{1} Nat " +
			"(Int.natAbs (Int.mul a b)) (Nat.mul (Int.natAbs a) (Int.natAbs b)) (Int.natAbs_mul a b)))"},
	valDecl{"thm", "Int.dvd_mul_left", nil,
		"(a : Int) → (b : Int) → Int.dvd a (Int.mul b a)",
		"fun (a : Int) (b : Int) => Iff.mp (Nat.dvd (Int.natAbs a) (Int.natAbs (Int.mul b a))) (Int.dvd a (Int.mul b a)) (Int.natAbs_dvd_natAbs a " +
			"(Int.mul b a)) (Eq.rec.{0, 1} Nat (Nat.mul (Int.natAbs b) (Int.natAbs a)) (fun (z : Nat) (_ : Eq.{1} Nat (Nat.mul (Int.natAbs b) (Int.natAbs " +
			"a)) z) => Nat.dvd (Int.natAbs a) z) (Nat.dvd_mul_left (Int.natAbs a) (Int.natAbs b)) (Int.natAbs (Int.mul b a)) (Eq.symm.{1} Nat (Int.natAbs " +
			"(Int.mul b a)) (Nat.mul (Int.natAbs b) (Int.natAbs a)) (Int.natAbs_mul b a)))"},
	valDecl{"thm", "Int.dvd_zero", nil,
		"(a : Int) → Int.dvd a (Int.ofNat 0)",
		"fun (a : Int) => Iff.mp (Nat.dvd (Int.natAbs a) (Int.natAbs (Int.ofNat 0))) (Int.dvd a (Int.ofNat 0)) (Int.natAbs_dvd_natAbs a (Int.ofNat " +
			"0)) (Nat.dvd_zero (Int.natAbs a))"},
	valDecl{"thm", "Int.neg_dvd", nil,
		"(a : Int) → (b : Int) → Iff (Int.dvd (Int.neg a) b) (Int.dvd a b)",
		"fun (a : Int) (b : Int) => Iff.intro (Int.dvd (Int.neg a) b) (Int.dvd a b) (fun (h : Int.dvd (Int.neg a) b) => Iff.mp (Nat.dvd (Int.natAbs " +
			"a) (Int.natAbs b)) (Int.dvd a b) (Int.natAbs_dvd_natAbs a b) (Eq.rec.{0, 1} Nat (Int.natAbs (Int.neg a)) (fun (z : Nat) (_ : Eq.{1} Nat " +
			"(Int.natAbs (Int.neg a)) z) => Nat.dvd z (Int.natAbs b)) (Iff.mpr (Nat.dvd (Int.natAbs (Int.neg a)) (Int.natAbs b)) (Int.dvd (Int.neg a) b) " +
			"(Int.natAbs_dvd_natAbs (Int.neg a) b) h) (Int.natAbs a) (Int.natAbs_neg a))) (fun (h : Int.dvd a b) => Iff.mp (Nat.dvd (Int.natAbs (Int.neg " +
			"a)) (Int.natAbs b)) (Int.dvd (Int.neg a) b) (Int.natAbs_dvd_natAbs (Int.neg a) b) (Eq.rec.{0, 1} Nat (Int.natAbs a) (fun (z : Nat) (_ : " +
			"Eq.{1} Nat (Int.natAbs a) z) => Nat.dvd z (Int.natAbs b)) (Iff.mpr (Nat.dvd (Int.natAbs a) (Int.natAbs b)) (Int.dvd a b) " +
			"(Int.natAbs_dvd_natAbs a b) h) (Int.natAbs (Int.neg a)) (Eq.symm.{1} Nat (Int.natAbs (Int.neg a)) (Int.natAbs a) (Int.natAbs_neg a))))"},
	// Int.gcd/Int.lcm are Nat-valued in Lean (Init/Data/Int/Gcd.lean: `m.natAbs.gcd n.natAbs`); Int.min/Int.max
	// are the `Min Int`/`Max Int` instances built from `minOfLe`/`maxOfLe` (`ite (a ≤ b) a b`), with that `ite` as a
	// Bool.rec on `Decidable.decide` here.
	valDecl{"def", "Int.gcd", nil, "Int → Int → Nat",
		"fun (m : Int) (n : Int) => Nat.gcd (Int.natAbs m) (Int.natAbs n)"},
	valDecl{"def", "Int.lcm", nil, "Int → Int → Nat",
		"fun (m : Int) (n : Int) => Nat.lcm (Int.natAbs m) (Int.natAbs n)"},
	// `Int.gcd` is `Nat.gcd` of the `natAbs` values, so the Nat lemmas apply as they are; signs go through `natAbs_neg`.
	valDecl{"thm", "Int.gcd_comm", nil,
		"(m : Int) → (n : Int) → Eq.{1} Nat (Int.gcd m n) (Int.gcd n m)",
		"fun (m : Int) (n : Int) => Nat.gcd_comm (Int.natAbs m) (Int.natAbs n)"},
	valDecl{"thm", "Int.lcm_comm", nil,
		"(m : Int) → (n : Int) → Eq.{1} Nat (Int.lcm m n) (Int.lcm n m)",
		"fun (m : Int) (n : Int) => Nat.lcm_comm (Int.natAbs m) (Int.natAbs n)"},
	valDecl{"thm", "Int.gcd_self", nil,
		"(a : Int) → Eq.{1} Nat (Int.gcd a a) (Int.natAbs a)",
		"fun (a : Int) => Nat.gcd_self (Int.natAbs a)"},
	valDecl{"thm", "Int.gcd_zero_right", nil,
		"(a : Int) → Eq.{1} Nat (Int.gcd a (Int.ofNat 0)) (Int.natAbs a)",
		"fun (a : Int) => Nat.gcd_zero_right (Int.natAbs a)"},
	valDecl{"thm", "Int.gcd_neg_left", nil,
		"(m : Int) → (n : Int) → Eq.{1} Nat (Int.gcd (Int.neg m) n) (Int.gcd m n)",
		"fun (m : Int) (n : Int) => congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.gcd z (Int.natAbs n)) (Int.natAbs (Int.neg m)) (Int.natAbs m) " +
			"(Int.natAbs_neg m)"},
	valDecl{"thm", "Int.gcd_neg_right", nil,
		"(m : Int) → (n : Int) → Eq.{1} Nat (Int.gcd m (Int.neg n)) (Int.gcd m n)",
		"fun (m : Int) (n : Int) => congrArg.{1, 1} Nat Nat (fun (z : Nat) => Nat.gcd (Int.natAbs m) z) (Int.natAbs (Int.neg n)) (Int.natAbs n) " +
			"(Int.natAbs_neg n)"},
	// `m * n = n * m` on Int: four constructor cases, each `congrArg` of `Nat.mul_comm` under `ofNat` or `negOfNat`.
	valDecl{"thm", "Int.mul_comm", nil,
		"(m : Int) → (n : Int) → Eq.{1} Int (Int.mul m n) (Int.mul n m)",
		"fun (m : Int) (n : Int) => Int.rec.{0} (fun (m : Int) => (n : Int) → Eq.{1} Int (Int.mul m n) (Int.mul n m)) (fun (a : Nat) (n : Int) => " +
			"Int.rec.{0} (fun (n : Int) => Eq.{1} Int (Int.mul (Int.ofNat a) n) (Int.mul n (Int.ofNat a))) (fun (b : Nat) => congrArg.{1, 1} Nat Int " +
			"Int.ofNat (Nat.mul a b) (Nat.mul b a) (Nat.mul_comm a b)) (fun (b : Nat) => congrArg.{1, 1} Nat Int Int.negOfNat (Nat.mul a (Nat.succ b)) " +
			"(Nat.mul (Nat.succ b) a) (Nat.mul_comm a (Nat.succ b))) n) (fun (a : Nat) (n : Int) => Int.rec.{0} (fun (n : Int) => Eq.{1} Int (Int.mul " +
			"(Int.negSucc a) n) (Int.mul n (Int.negSucc a))) (fun (b : Nat) => congrArg.{1, 1} Nat Int Int.negOfNat (Nat.mul (Nat.succ a) b) (Nat.mul b " +
			"(Nat.succ a)) (Nat.mul_comm (Nat.succ a) b)) (fun (b : Nat) => congrArg.{1, 1} Nat Int Int.ofNat (Nat.mul (Nat.succ a) (Nat.succ b)) " +
			"(Nat.mul (Nat.succ b) (Nat.succ a)) (Nat.mul_comm (Nat.succ a) (Nat.succ b))) n) m n"},
	// `ofNat a * negOfNat k = negOfNat (a * k)`: `k = 0` and `k = succ j` are `rfl`. Needed for `Int.mul_assoc`, whose mixed
	// sign cases are `negOfNat` of a product; the commuted form is `mul_comm` and `Nat.mul_comm`.
	valDecl{"thm", "Int.mul_ofNat_negOfNat", nil,
		"(a : Nat) → (k : Nat) → Eq.{1} Int (Int.mul (Int.ofNat a) (Int.negOfNat k)) (Int.negOfNat (Nat.mul a k))",
		"fun (a : Nat) (k : Nat) => Nat.rec.{0} (fun (k : Nat) => Eq.{1} Int (Int.mul (Int.ofNat a) (Int.negOfNat k)) (Int.negOfNat (Nat.mul a k))) " +
			"(Eq.refl.{1} Int (Int.ofNat 0)) (fun (j : Nat) (_ : Eq.{1} Int (Int.mul (Int.ofNat a) (Int.negOfNat j)) (Int.negOfNat (Nat.mul a j))) => " +
			"Eq.refl.{1} Int (Int.negOfNat (Nat.mul a (Nat.succ j)))) k"},
	valDecl{"thm", "Int.negOfNat_mul_ofNat", nil,
		"(k : Nat) → (a : Nat) → Eq.{1} Int (Int.mul (Int.negOfNat k) (Int.ofNat a)) (Int.negOfNat (Nat.mul k a))",
		"fun (k : Nat) (a : Nat) => Eq.trans.{1} Int (Int.mul (Int.negOfNat k) (Int.ofNat a)) (Int.mul (Int.ofNat a) (Int.negOfNat k)) (Int.negOfNat " +
			"(Nat.mul k a)) (Int.mul_comm (Int.negOfNat k) (Int.ofNat a)) (Eq.trans.{1} Int (Int.mul (Int.ofNat a) (Int.negOfNat k)) (Int.negOfNat " +
			"(Nat.mul a k)) (Int.negOfNat (Nat.mul k a)) (Int.mul_ofNat_negOfNat a k) (congrArg.{1, 1} Nat Int Int.negOfNat (Nat.mul a k) (Nat.mul k a) " +
			"(Nat.mul_comm a k)))"},
	// Mixed-sign helpers for `Int.mul_assoc`: `negOfNat k * negSucc c = ofNat (k * succ c)` (`k = 0` needs `zero_mul`; the
	// successor case is `rfl`) and `negSucc a * negOfNat k = ofNat (succ a * k)` (both cases `rfl`).
	valDecl{"thm", "Int.negOfNat_mul_negSucc", nil,
		"(k : Nat) → (c : Nat) → Eq.{1} Int (Int.mul (Int.negOfNat k) (Int.negSucc c)) (Int.ofNat (Nat.mul k (Nat.succ c)))",
		"fun (k : Nat) (c : Nat) => Nat.rec.{0} (fun (k : Nat) => Eq.{1} Int (Int.mul (Int.negOfNat k) (Int.negSucc c)) (Int.ofNat (Nat.mul k " +
			"(Nat.succ c)))) (Eq.trans.{1} Int (Int.negOfNat (Nat.mul 0 (Nat.succ c))) (Int.negOfNat 0) (Int.ofNat (Nat.mul 0 (Nat.succ c))) " +
			"(congrArg.{1, 1} Nat Int Int.negOfNat (Nat.mul 0 (Nat.succ c)) 0 (Nat.zero_mul (Nat.succ c))) (Eq.symm.{1} Int (Int.ofNat (Nat.mul 0 " +
			"(Nat.succ c))) (Int.ofNat 0) (congrArg.{1, 1} Nat Int Int.ofNat (Nat.mul 0 (Nat.succ c)) 0 (Nat.zero_mul (Nat.succ c))))) (fun (j : Nat) (_ " +
			": Eq.{1} Int (Int.mul (Int.negOfNat j) (Int.negSucc c)) (Int.ofNat (Nat.mul j (Nat.succ c)))) => Eq.refl.{1} Int (Int.ofNat (Nat.mul " +
			"(Nat.succ j) (Nat.succ c)))) k"},
	valDecl{"thm", "Int.negSucc_mul_negOfNat", nil,
		"(a : Nat) → (k : Nat) → Eq.{1} Int (Int.mul (Int.negSucc a) (Int.negOfNat k)) (Int.ofNat (Nat.mul (Nat.succ a) k))",
		"fun (a : Nat) (k : Nat) => Nat.rec.{0} (fun (k : Nat) => Eq.{1} Int (Int.mul (Int.negSucc a) (Int.negOfNat k)) (Int.ofNat (Nat.mul (Nat.succ " +
			"a) k))) (Eq.refl.{1} Int (Int.ofNat 0)) (fun (j : Nat) (_ : Eq.{1} Int (Int.mul (Int.negSucc a) (Int.negOfNat j)) (Int.ofNat (Nat.mul " +
			"(Nat.succ a) j))) => Eq.refl.{1} Int (Int.ofNat (Nat.mul (Nat.succ a) (Nat.succ j)))) k"},
	// `(m * n) * k = m * (n * k)` on Int: eight constructor cases, each `congrArg ofNat` or `congrArg negOfNat` of
	// `Nat.mul_assoc`, with the helpers above moving `negOfNat` across one factor.
	valDecl{"thm", "Int.mul_assoc", nil,
		"(m : Int) → (n : Int) → (k : Int) → Eq.{1} Int (Int.mul (Int.mul m n) k) (Int.mul m (Int.mul n k))",
		"fun (m : Int) (n : Int) (k : Int) => Int.rec.{0} (fun (m : Int) => (n : Int) → (k : Int) → Eq.{1} Int (Int.mul (Int.mul m n) k) (Int.mul m " +
			"(Int.mul n k))) (fun (a : Nat) (n : Int) (k : Int) => Int.rec.{0} (fun (n : Int) => (k : Int) → Eq.{1} Int (Int.mul (Int.mul (Int.ofNat a) " +
			"n) k) (Int.mul (Int.ofNat a) (Int.mul n k))) (fun (b : Nat) (k : Int) => Int.rec.{0} (fun (k : Int) => Eq.{1} Int (Int.mul (Int.mul " +
			"(Int.ofNat a) (Int.ofNat b)) k) (Int.mul (Int.ofNat a) (Int.mul (Int.ofNat b) k))) (fun (c : Nat) => congrArg.{1, 1} Nat Int Int.ofNat " +
			"(Nat.mul (Nat.mul a b) c) (Nat.mul a (Nat.mul b c)) (Nat.mul_assoc a b c)) (fun (c : Nat) => Eq.trans.{1} Int (Int.negOfNat (Nat.mul " +
			"(Nat.mul a b) (Nat.succ c))) (Int.negOfNat (Nat.mul a (Nat.mul b (Nat.succ c)))) (Int.mul (Int.ofNat a) (Int.negOfNat (Nat.mul b (Nat.succ " +
			"c)))) (congrArg.{1, 1} Nat Int Int.negOfNat (Nat.mul (Nat.mul a b) (Nat.succ c)) (Nat.mul a (Nat.mul b (Nat.succ c))) (Nat.mul_assoc a b " +
			"(Nat.succ c))) (Eq.symm.{1} Int (Int.mul (Int.ofNat a) (Int.negOfNat (Nat.mul b (Nat.succ c)))) (Int.negOfNat (Nat.mul a (Nat.mul b " +
			"(Nat.succ c)))) (Int.mul_ofNat_negOfNat a (Nat.mul b (Nat.succ c))))) k) (fun (b : Nat) (k : Int) => Int.rec.{0} (fun (k : Int) => Eq.{1} " +
			"Int (Int.mul (Int.mul (Int.ofNat a) (Int.negSucc b)) k) (Int.mul (Int.ofNat a) (Int.mul (Int.negSucc b) k))) (fun (c : Nat) => Eq.trans.{1} " +
			"Int (Int.mul (Int.negOfNat (Nat.mul a (Nat.succ b))) (Int.ofNat c)) (Int.negOfNat (Nat.mul (Nat.mul a (Nat.succ b)) c)) (Int.mul (Int.ofNat " +
			"a) (Int.negOfNat (Nat.mul (Nat.succ b) c))) (Int.negOfNat_mul_ofNat (Nat.mul a (Nat.succ b)) c) (Eq.trans.{1} Int (Int.negOfNat (Nat.mul " +
			"(Nat.mul a (Nat.succ b)) c)) (Int.negOfNat (Nat.mul a (Nat.mul (Nat.succ b) c))) (Int.mul (Int.ofNat a) (Int.negOfNat (Nat.mul (Nat.succ b) " +
			"c))) (congrArg.{1, 1} Nat Int Int.negOfNat (Nat.mul (Nat.mul a (Nat.succ b)) c) (Nat.mul a (Nat.mul (Nat.succ b) c)) (Nat.mul_assoc a " +
			"(Nat.succ b) c)) (Eq.symm.{1} Int (Int.mul (Int.ofNat a) (Int.negOfNat (Nat.mul (Nat.succ b) c))) (Int.negOfNat (Nat.mul a (Nat.mul " +
			"(Nat.succ b) c))) (Int.mul_ofNat_negOfNat a (Nat.mul (Nat.succ b) c))))) (fun (c : Nat) => Eq.trans.{1} Int (Int.mul (Int.negOfNat (Nat.mul " +
			"a (Nat.succ b))) (Int.negSucc c)) (Int.ofNat (Nat.mul (Nat.mul a (Nat.succ b)) (Nat.succ c))) (Int.ofNat (Nat.mul a (Nat.mul (Nat.succ b) " +
			"(Nat.succ c)))) (Int.negOfNat_mul_negSucc (Nat.mul a (Nat.succ b)) c) (congrArg.{1, 1} Nat Int Int.ofNat (Nat.mul (Nat.mul a (Nat.succ b)) " +
			"(Nat.succ c)) (Nat.mul a (Nat.mul (Nat.succ b) (Nat.succ c))) (Nat.mul_assoc a (Nat.succ b) (Nat.succ c)))) k) n k) (fun (a : Nat) (n : Int) " +
			"(k : Int) => Int.rec.{0} (fun (n : Int) => (k : Int) → Eq.{1} Int (Int.mul (Int.mul (Int.negSucc a) n) k) (Int.mul (Int.negSucc a) (Int.mul " +
			"n k))) (fun (b : Nat) (k : Int) => Int.rec.{0} (fun (k : Int) => Eq.{1} Int (Int.mul (Int.mul (Int.negSucc a) (Int.ofNat b)) k) (Int.mul " +
			"(Int.negSucc a) (Int.mul (Int.ofNat b) k))) (fun (c : Nat) => Eq.trans.{1} Int (Int.mul (Int.negOfNat (Nat.mul (Nat.succ a) b)) (Int.ofNat " +
			"c)) (Int.negOfNat (Nat.mul (Nat.mul (Nat.succ a) b) c)) (Int.negOfNat (Nat.mul (Nat.succ a) (Nat.mul b c))) (Int.negOfNat_mul_ofNat (Nat.mul " +
			"(Nat.succ a) b) c) (congrArg.{1, 1} Nat Int Int.negOfNat (Nat.mul (Nat.mul (Nat.succ a) b) c) (Nat.mul (Nat.succ a) (Nat.mul b c)) " +
			"(Nat.mul_assoc (Nat.succ a) b c))) (fun (c : Nat) => Eq.trans.{1} Int (Int.mul (Int.negOfNat (Nat.mul (Nat.succ a) b)) (Int.negSucc c)) " +
			"(Int.ofNat (Nat.mul (Nat.mul (Nat.succ a) b) (Nat.succ c))) (Int.mul (Int.negSucc a) (Int.negOfNat (Nat.mul b (Nat.succ c)))) " +
			"(Int.negOfNat_mul_negSucc (Nat.mul (Nat.succ a) b) c) (Eq.trans.{1} Int (Int.ofNat (Nat.mul (Nat.mul (Nat.succ a) b) (Nat.succ c))) " +
			"(Int.ofNat (Nat.mul (Nat.succ a) (Nat.mul b (Nat.succ c)))) (Int.mul (Int.negSucc a) (Int.negOfNat (Nat.mul b (Nat.succ c)))) (congrArg.{1, " +
			"1} Nat Int Int.ofNat (Nat.mul (Nat.mul (Nat.succ a) b) (Nat.succ c)) (Nat.mul (Nat.succ a) (Nat.mul b (Nat.succ c))) (Nat.mul_assoc " +
			"(Nat.succ a) b (Nat.succ c))) (Eq.symm.{1} Int (Int.mul (Int.negSucc a) (Int.negOfNat (Nat.mul b (Nat.succ c)))) (Int.ofNat (Nat.mul " +
			"(Nat.succ a) (Nat.mul b (Nat.succ c)))) (Int.negSucc_mul_negOfNat a (Nat.mul b (Nat.succ c)))))) k) (fun (b : Nat) (k : Int) => Int.rec.{0} " +
			"(fun (k : Int) => Eq.{1} Int (Int.mul (Int.mul (Int.negSucc a) (Int.negSucc b)) k) (Int.mul (Int.negSucc a) (Int.mul (Int.negSucc b) k))) " +
			"(fun (c : Nat) => Eq.trans.{1} Int (Int.ofNat (Nat.mul (Nat.mul (Nat.succ a) (Nat.succ b)) c)) (Int.ofNat (Nat.mul (Nat.succ a) (Nat.mul " +
			"(Nat.succ b) c))) (Int.mul (Int.negSucc a) (Int.negOfNat (Nat.mul (Nat.succ b) c))) (congrArg.{1, 1} Nat Int Int.ofNat (Nat.mul (Nat.mul " +
			"(Nat.succ a) (Nat.succ b)) c) (Nat.mul (Nat.succ a) (Nat.mul (Nat.succ b) c)) (Nat.mul_assoc (Nat.succ a) (Nat.succ b) c)) (Eq.symm.{1} Int " +
			"(Int.mul (Int.negSucc a) (Int.negOfNat (Nat.mul (Nat.succ b) c))) (Int.ofNat (Nat.mul (Nat.succ a) (Nat.mul (Nat.succ b) c))) " +
			"(Int.negSucc_mul_negOfNat a (Nat.mul (Nat.succ b) c)))) (fun (c : Nat) => congrArg.{1, 1} Nat Int Int.negOfNat (Nat.mul (Nat.mul (Nat.succ " +
			"a) (Nat.succ b)) (Nat.succ c)) (Nat.mul (Nat.succ a) (Nat.mul (Nat.succ b) (Nat.succ c))) (Nat.mul_assoc (Nat.succ a) (Nat.succ b) (Nat.succ " +
			"c))) k) n k) m n k"},
	// Units for Int multiplication: `mul_zero` is `rfl` on both constructors; `mul_one` is `congrArg` of `Nat.mul_one`; the left
	// versions are `mul_comm` followed by those.
	valDecl{"thm", "Int.mul_zero", nil,
		"(a : Int) → Eq.{1} Int (Int.mul a (Int.ofNat 0)) (Int.ofNat 0)",
		"fun (a : Int) => Int.rec.{0} (fun (a : Int) => Eq.{1} Int (Int.mul a (Int.ofNat 0)) (Int.ofNat 0)) (fun (k : Nat) => Eq.refl.{1} Int " +
			"(Int.ofNat 0)) (fun (k : Nat) => Eq.refl.{1} Int (Int.ofNat 0)) a"},
	valDecl{"thm", "Int.zero_mul", nil,
		"(a : Int) → Eq.{1} Int (Int.mul (Int.ofNat 0) a) (Int.ofNat 0)",
		"fun (a : Int) => Eq.trans.{1} Int (Int.mul (Int.ofNat 0) a) (Int.mul a (Int.ofNat 0)) (Int.ofNat 0) (Int.mul_comm (Int.ofNat 0) a) " +
			"(Int.mul_zero a)"},
	valDecl{"thm", "Int.mul_one", nil,
		"(a : Int) → Eq.{1} Int (Int.mul a (Int.ofNat 1)) a",
		"fun (a : Int) => Int.rec.{0} (fun (a : Int) => Eq.{1} Int (Int.mul a (Int.ofNat 1)) a) (fun (k : Nat) => congrArg.{1, 1} Nat Int Int.ofNat " +
			"(Nat.mul k 1) k (Nat.mul_one k)) (fun (k : Nat) => congrArg.{1, 1} Nat Int Int.negOfNat (Nat.mul (Nat.succ k) 1) (Nat.succ k) (Nat.mul_one " +
			"(Nat.succ k))) a"},
	valDecl{"thm", "Int.one_mul", nil,
		"(a : Int) → Eq.{1} Int (Int.mul (Int.ofNat 1) a) a",
		"fun (a : Int) => Eq.trans.{1} Int (Int.mul (Int.ofNat 1) a) (Int.mul a (Int.ofNat 1)) a (Int.mul_comm (Int.ofNat 1) a) (Int.mul_one a)"},
	// `m + n = n + m` on Int: the two same-sign cases are `congrArg` of `Nat.add_comm`; the two mixed cases are `rfl`, since
	// `Int.add` gives `subNatNat` of the same pair in either order.
	valDecl{"thm", "Int.add_comm", nil,
		"(m : Int) → (n : Int) → Eq.{1} Int (Int.add m n) (Int.add n m)",
		"fun (m : Int) (n : Int) => Int.rec.{0} (fun (m : Int) => (n : Int) → Eq.{1} Int (Int.add m n) (Int.add n m)) (fun (a : Nat) (n : Int) => " +
			"Int.rec.{0} (fun (n : Int) => Eq.{1} Int (Int.add (Int.ofNat a) n) (Int.add n (Int.ofNat a))) (fun (b : Nat) => congrArg.{1, 1} Nat Int " +
			"Int.ofNat (Nat.add a b) (Nat.add b a) (Nat.add_comm a b)) (fun (b : Nat) => Eq.refl.{1} Int (Int.subNatNat a (Nat.succ b))) n) (fun (a : " +
			"Nat) (n : Int) => Int.rec.{0} (fun (n : Int) => Eq.{1} Int (Int.add (Int.negSucc a) n) (Int.add n (Int.negSucc a))) (fun (b : Nat) => " +
			"Eq.refl.{1} Int (Int.subNatNat b (Nat.succ a))) (fun (b : Nat) => congrArg.{1, 1} Nat Int (fun (z : Nat) => Int.negSucc (Nat.succ z)) " +
			"(Nat.add a b) (Nat.add b a) (Nat.add_comm a b)) n) m n"},
	// `a + 0 = a` is `rfl` on both constructors (`subNatNat 0 (succ k)` reduces to `negSucc k`); `zero_add` is `add_comm` then that.
	valDecl{"thm", "Int.add_zero", nil,
		"(a : Int) → Eq.{1} Int (Int.add a (Int.ofNat 0)) a",
		"fun (a : Int) => Int.rec.{0} (fun (a : Int) => Eq.{1} Int (Int.add a (Int.ofNat 0)) a) (fun (k : Nat) => Eq.refl.{1} Int (Int.ofNat k)) (fun " +
			"(k : Nat) => Eq.refl.{1} Int (Int.negSucc k)) a"},
	valDecl{"thm", "Int.zero_add", nil,
		"(a : Int) → Eq.{1} Int (Int.add (Int.ofNat 0) a) a",
		"fun (a : Int) => Eq.trans.{1} Int (Int.add (Int.ofNat 0) a) (Int.add a (Int.ofNat 0)) a (Int.add_comm (Int.ofNat 0) a) (Int.add_zero a)"},
	valDecl{"thm", "Int.neg_zero", nil,
		"Eq.{1} Int (Int.neg (Int.ofNat 0)) (Int.ofNat 0)",
		"Eq.refl.{1} Int (Int.ofNat 0)"},
	// `subNatNat n n = 0`: its `Bool.rec` is on `Nat.ble n n`, which is `true` (`ble_refl`), leaving `ofNat (n - n)` and `sub_self`.
	valDecl{"thm", "Int.subNatNat_self", nil,
		"(n : Nat) → Eq.{1} Int (Int.subNatNat n n) (Int.ofNat 0)",
		"fun (n : Nat) => Eq.rec.{0, 1} Bool Bool.true (fun (z : Bool) (_ : Eq.{1} Bool Bool.true z) => Eq.{1} Int (Bool.rec.{1} (fun (_ : Bool) => " +
			"Int) (Int.negSucc (Nat.sub n (Nat.succ n))) (Int.ofNat (Nat.sub n n)) z) (Int.ofNat 0)) (congrArg.{1, 1} Nat Int Int.ofNat (Nat.sub n n) 0 " +
			"(Nat.sub_self n)) (Nat.ble n n) (Eq.symm.{1} Bool (Nat.ble n n) Bool.true (Nat.ble_refl n))"},
	// `a + -a = 0`: `k = 0` is `rfl`; `ofNat (succ j) + negSucc j` and `negSucc k + ofNat (succ k)` are `subNatNat (succ _) (succ _)`.
	valDecl{"thm", "Int.add_neg_cancel", nil,
		"(a : Int) → Eq.{1} Int (Int.add a (Int.neg a)) (Int.ofNat 0)",
		"fun (a : Int) => Int.rec.{0} (fun (a : Int) => Eq.{1} Int (Int.add a (Int.neg a)) (Int.ofNat 0)) (fun (k : Nat) => Nat.rec.{0} (fun (k : " +
			"Nat) => Eq.{1} Int (Int.add (Int.ofNat k) (Int.neg (Int.ofNat k))) (Int.ofNat 0)) (Eq.refl.{1} Int (Int.ofNat 0)) (fun (j : Nat) (_ : Eq.{1} " +
			"Int (Int.add (Int.ofNat j) (Int.neg (Int.ofNat j))) (Int.ofNat 0)) => Int.subNatNat_self (Nat.succ j)) k) (fun (k : Nat) => " +
			"Int.subNatNat_self (Nat.succ k)) a"},
	valDecl{"thm", "Int.neg_add_cancel", nil,
		"(a : Int) → Eq.{1} Int (Int.add (Int.neg a) a) (Int.ofNat 0)",
		"fun (a : Int) => Eq.trans.{1} Int (Int.add (Int.neg a) a) (Int.add a (Int.neg a)) (Int.ofNat 0) (Int.add_comm (Int.neg a) a) " +
			"(Int.add_neg_cancel a)"},
	// `-(negOfNat n) = ofNat n`: `n = 0` and `n = succ j` are `rfl`. Then `-a * b = -(a * b)`: four constructor cases, using the
	// `negOfNat` helpers of `mul_assoc`; `a * -b = -(a * b)` follows by `mul_comm`.
	valDecl{"thm", "Int.neg_negOfNat", nil,
		"(n : Nat) → Eq.{1} Int (Int.neg (Int.negOfNat n)) (Int.ofNat n)",
		"fun (n : Nat) => Nat.rec.{0} (fun (n : Nat) => Eq.{1} Int (Int.neg (Int.negOfNat n)) (Int.ofNat n)) (Eq.refl.{1} Int (Int.ofNat 0)) (fun (j " +
			": Nat) (_ : Eq.{1} Int (Int.neg (Int.negOfNat j)) (Int.ofNat j)) => Eq.refl.{1} Int (Int.ofNat (Nat.succ j))) n"},
	valDecl{"thm", "Int.neg_mul", nil,
		"(a : Int) → (b : Int) → Eq.{1} Int (Int.mul (Int.neg a) b) (Int.neg (Int.mul a b))",
		"fun (a : Int) (b : Int) => Int.rec.{0} (fun (a : Int) => (b : Int) → Eq.{1} Int (Int.mul (Int.neg a) b) (Int.neg (Int.mul a b))) (fun (k : " +
			"Nat) (b : Int) => Int.rec.{0} (fun (b : Int) => Eq.{1} Int (Int.mul (Int.neg (Int.ofNat k)) b) (Int.neg (Int.mul (Int.ofNat k) b))) (fun (c " +
			": Nat) => Int.negOfNat_mul_ofNat k c) (fun (c : Nat) => Eq.trans.{1} Int (Int.mul (Int.negOfNat k) (Int.negSucc c)) (Int.ofNat (Nat.mul k " +
			"(Nat.succ c))) (Int.neg (Int.negOfNat (Nat.mul k (Nat.succ c)))) (Int.negOfNat_mul_negSucc k c) (Eq.symm.{1} Int (Int.neg (Int.negOfNat " +
			"(Nat.mul k (Nat.succ c)))) (Int.ofNat (Nat.mul k (Nat.succ c))) (Int.neg_negOfNat (Nat.mul k (Nat.succ c))))) b) (fun (k : Nat) (b : Int) => " +
			"Int.rec.{0} (fun (b : Int) => Eq.{1} Int (Int.mul (Int.neg (Int.negSucc k)) b) (Int.neg (Int.mul (Int.negSucc k) b))) (fun (c : Nat) => " +
			"Eq.symm.{1} Int (Int.neg (Int.negOfNat (Nat.mul (Nat.succ k) c))) (Int.ofNat (Nat.mul (Nat.succ k) c)) (Int.neg_negOfNat (Nat.mul (Nat.succ " +
			"k) c))) (fun (c : Nat) => Eq.refl.{1} Int (Int.negOfNat (Nat.mul (Nat.succ k) (Nat.succ c)))) b) a b"},
	valDecl{"thm", "Int.mul_neg", nil,
		"(a : Int) → (b : Int) → Eq.{1} Int (Int.mul a (Int.neg b)) (Int.neg (Int.mul a b))",
		"fun (a : Int) (b : Int) => Eq.trans.{1} Int (Int.mul a (Int.neg b)) (Int.mul (Int.neg b) a) (Int.neg (Int.mul a b)) (Int.mul_comm a (Int.neg " +
			"b)) (Eq.trans.{1} Int (Int.mul (Int.neg b) a) (Int.neg (Int.mul b a)) (Int.neg (Int.mul a b)) (Int.neg_mul b a) (congrArg.{1, 1} Int Int " +
			"Int.neg (Int.mul b a) (Int.mul a b) (Int.mul_comm b a)))"},
	// Constructor facts, each `rfl`: `ofNat` commutes with `+` and `*`, and `neg` swaps `ofNat (succ n)` with `negSucc n`.
	valDecl{"thm", "Int.ofNat_add", nil,
		"(m : Nat) → (n : Nat) → Eq.{1} Int (Int.ofNat (Nat.add m n)) (Int.add (Int.ofNat m) (Int.ofNat n))",
		"fun (m : Nat) (n : Nat) => Eq.refl.{1} Int (Int.ofNat (Nat.add m n))"},
	valDecl{"thm", "Int.ofNat_mul", nil,
		"(m : Nat) → (n : Nat) → Eq.{1} Int (Int.ofNat (Nat.mul m n)) (Int.mul (Int.ofNat m) (Int.ofNat n))",
		"fun (m : Nat) (n : Nat) => Eq.refl.{1} Int (Int.ofNat (Nat.mul m n))"},
	valDecl{"thm", "Int.neg_ofNat_succ", nil,
		"(n : Nat) → Eq.{1} Int (Int.neg (Int.ofNat (Nat.succ n))) (Int.negSucc n)",
		"fun (n : Nat) => Eq.refl.{1} Int (Int.negSucc n)"},
	valDecl{"thm", "Int.neg_negSucc", nil,
		"(n : Nat) → Eq.{1} Int (Int.neg (Int.negSucc n)) (Int.ofNat (Nat.succ n))",
		"fun (n : Nat) => Eq.refl.{1} Int (Int.ofNat (Nat.succ n))"},
	// Subtraction is `a + -b`: `sub_eq_add_neg` is `rfl`, and `sub_self`, `sub_zero`, `zero_sub` are `add_neg_cancel`, `add_zero`, `zero_add`.
	valDecl{"thm", "Int.sub_eq_add_neg", nil,
		"(a : Int) → (b : Int) → Eq.{1} Int (Int.sub a b) (Int.add a (Int.neg b))",
		"fun (a : Int) (b : Int) => Eq.refl.{1} Int (Int.add a (Int.neg b))"},
	valDecl{"thm", "Int.sub_self", nil,
		"(a : Int) → Eq.{1} Int (Int.sub a a) (Int.ofNat 0)",
		"fun (a : Int) => Int.add_neg_cancel a"},
	valDecl{"thm", "Int.sub_zero", nil,
		"(a : Int) → Eq.{1} Int (Int.sub a (Int.ofNat 0)) a",
		"fun (a : Int) => Int.add_zero a"},
	valDecl{"thm", "Int.zero_sub", nil,
		"(a : Int) → Eq.{1} Int (Int.sub (Int.ofNat 0) a) (Int.neg a)",
		"fun (a : Int) => Int.zero_add (Int.neg a)"},
	// `subNatNat (m + n) m = ofNat n` (its `Bool.rec` is on `ble m (m + n)`, true) and `subNatNat m (succ (m + n)) = negSucc n`
	// (`ble (succ (m + n)) m` is false: true would give `succ m <= m`).
	valDecl{"thm", "Int.subNatNat_add_left", nil,
		"(m : Nat) → (n : Nat) → Eq.{1} Int (Int.subNatNat (Nat.add m n) m) (Int.ofNat n)",
		"fun (m : Nat) (n : Nat) => Eq.rec.{0, 1} Bool Bool.true (fun (z : Bool) (_ : Eq.{1} Bool Bool.true z) => Eq.{1} Int (Bool.rec.{1} (fun (_ : " +
			"Bool) => Int) (Int.negSucc (Nat.sub m (Nat.succ (Nat.add m n)))) (Int.ofNat (Nat.sub (Nat.add m n) m)) z) (Int.ofNat n)) (congrArg.{1, 1} " +
			"Nat Int Int.ofNat (Nat.sub (Nat.add m n) m) n (Nat.add_sub_cancel_left m n)) (Nat.ble m (Nat.add m n)) (Eq.symm.{1} Bool (Nat.ble m (Nat.add " +
			"m n)) Bool.true (Nat.ble_true_of_le m (Nat.add m n) (Nat.le_add_right m n)))"},
	valDecl{"thm", "Int.subNatNat_add_right", nil,
		"(m : Nat) → (n : Nat) → Eq.{1} Int (Int.subNatNat m (Nat.succ (Nat.add m n))) (Int.negSucc n)",
		"fun (m : Nat) (n : Nat) => Bool.rec.{0} (fun (z : Bool) => Eq.{1} Bool (Nat.ble (Nat.succ (Nat.add m n)) m) z → Eq.{1} Int (Bool.rec.{1} " +
			"(fun (_ : Bool) => Int) (Int.negSucc (Nat.sub (Nat.succ (Nat.add m n)) (Nat.succ m))) (Int.ofNat (Nat.sub m (Nat.succ (Nat.add m n)))) z) " +
			"(Int.negSucc n)) (fun (e : Eq.{1} Bool (Nat.ble (Nat.succ (Nat.add m n)) m) Bool.false) => congrArg.{1, 1} Nat Int Int.negSucc (Nat.sub " +
			"(Nat.succ (Nat.add m n)) (Nat.succ m)) n (Eq.trans.{1} Nat (Nat.sub (Nat.succ (Nat.add m n)) (Nat.succ m)) (Nat.sub (Nat.add m n) m) n " +
			"(Nat.succ_sub_succ (Nat.add m n) m) (Nat.add_sub_cancel_left m n))) (fun (e : Eq.{1} Bool (Nat.ble (Nat.succ (Nat.add m n)) m) Bool.true) => " +
			"False.elim.{0} (Eq.{1} Int (Bool.rec.{1} (fun (_ : Bool) => Int) (Int.negSucc (Nat.sub (Nat.succ (Nat.add m n)) (Nat.succ m))) (Int.ofNat " +
			"(Nat.sub m (Nat.succ (Nat.add m n)))) Bool.true) (Int.negSucc n)) (Nat.lt_irrefl m (Nat.le_trans (Nat.succ m) (Nat.succ (Nat.add m n)) m " +
			"(Nat.succ_le_succ m (Nat.add m n) (Nat.le_add_right m n)) (Nat.le_of_ble_eq_true (Nat.succ (Nat.add m n)) m e)))) (Nat.ble (Nat.succ " +
			"(Nat.add m n)) m) (Eq.refl.{1} Bool (Nat.ble (Nat.succ (Nat.add m n)) m))"},
	// `subNatNat (succ m) (succ n) = subNatNat m n`: `ble (succ n) (succ m)` reduces to `ble n m`, and each branch is a `succ_sub_succ`.
	valDecl{"thm", "Int.subNatNat_succ_succ", nil,
		"(m : Nat) → (n : Nat) → Eq.{1} Int (Int.subNatNat (Nat.succ m) (Nat.succ n)) (Int.subNatNat m n)",
		"fun (m : Nat) (n : Nat) => Bool.rec.{0} (fun (z : Bool) => Eq.{1} Bool (Nat.ble n m) z → Eq.{1} Int (Bool.rec.{1} (fun (_ : Bool) => Int) " +
			"(Int.negSucc (Nat.sub (Nat.succ n) (Nat.succ (Nat.succ m)))) (Int.ofNat (Nat.sub (Nat.succ m) (Nat.succ n))) z) (Bool.rec.{1} (fun (_ : " +
			"Bool) => Int) (Int.negSucc (Nat.sub n (Nat.succ m))) (Int.ofNat (Nat.sub m n)) z)) (fun (e : Eq.{1} Bool (Nat.ble n m) Bool.false) => " +
			"congrArg.{1, 1} Nat Int Int.negSucc (Nat.sub (Nat.succ n) (Nat.succ (Nat.succ m))) (Nat.sub n (Nat.succ m)) (Nat.succ_sub_succ n (Nat.succ " +
			"m))) (fun (e : Eq.{1} Bool (Nat.ble n m) Bool.true) => congrArg.{1, 1} Nat Int Int.ofNat (Nat.sub (Nat.succ m) (Nat.succ n)) (Nat.sub m n) " +
			"(Nat.succ_sub_succ m n)) (Nat.ble n m) (Eq.refl.{1} Bool (Nat.ble n m))"},
	// `subNatNat (m + k) (n + k) = subNatNat m n`: induction on `k`; `m + succ j` is `succ (m + j)` by `rfl`, so the step is `subNatNat_succ_succ`.
	valDecl{"thm", "Int.subNatNat_add_add", nil,
		"(m : Nat) → (n : Nat) → (k : Nat) → Eq.{1} Int (Int.subNatNat (Nat.add m k) (Nat.add n k)) (Int.subNatNat m n)",
		"fun (m : Nat) (n : Nat) (k : Nat) => Nat.rec.{0} (fun (k : Nat) => Eq.{1} Int (Int.subNatNat (Nat.add m k) (Nat.add n k)) (Int.subNatNat m n)) " +
			"(Eq.refl.{1} Int (Int.subNatNat m n)) " +
			"(fun (j : Nat) (ih : Eq.{1} Int (Int.subNatNat (Nat.add m j) (Nat.add n j)) (Int.subNatNat m n)) => " +
			"Eq.trans.{1} Int (Int.subNatNat (Nat.succ (Nat.add m j)) (Nat.succ (Nat.add n j))) (Int.subNatNat (Nat.add m j) (Nat.add n j)) " +
			"(Int.subNatNat m n) (Int.subNatNat_succ_succ (Nat.add m j) (Nat.add n j)) ih) k"},
	// `ofNat a + subNatNat b c = subNatNat (a + b) c` in its two shapes. For `b = c + d`, both sides are `ofNat (a + d)` by `subNatNat_add_left`.
	// For `c = succ (b + d)`, the left side is `subNatNat a (succ d)` (`subNatNat_add_right`) and so is the right, by `subNatNat_add_add`.
	valDecl{"thm", "Int.ofNat_add_subNatNat_le", nil,
		"(a : Nat) → (c : Nat) → (d : Nat) → Eq.{1} Int (Int.add (Int.ofNat a) (Int.subNatNat (Nat.add c d) c)) (Int.subNatNat (Nat.add a (Nat.add c " +
			"d)) c)",
		"fun (a : Nat) (c : Nat) (d : Nat) => Eq.trans.{1} Int (Int.add (Int.ofNat a) (Int.subNatNat (Nat.add c d) c)) (Int.ofNat (Nat.add a d)) " +
			"(Int.subNatNat (Nat.add a (Nat.add c d)) c) (congrArg.{1, 1} Int Int (Int.add (Int.ofNat a)) (Int.subNatNat (Nat.add c d) c) (Int.ofNat d) " +
			"(Int.subNatNat_add_left c d)) (Eq.trans.{1} Int (Int.ofNat (Nat.add a d)) (Int.subNatNat (Nat.add c (Nat.add a d)) c) (Int.subNatNat " +
			"(Nat.add a (Nat.add c d)) c) (Eq.symm.{1} Int (Int.subNatNat (Nat.add c (Nat.add a d)) c) (Int.ofNat (Nat.add a d)) (Int.subNatNat_add_left " +
			"c (Nat.add a d))) (congrArg.{1, 1} Nat Int (fun (z : Nat) => Int.subNatNat z c) (Nat.add c (Nat.add a d)) (Nat.add a (Nat.add c d)) " +
			"(Nat.add_left_comm c a d)))"},
	valDecl{"thm", "Int.ofNat_add_subNatNat_gt", nil,
		"(a : Nat) → (b : Nat) → (d : Nat) → Eq.{1} Int (Int.add (Int.ofNat a) (Int.subNatNat b (Nat.succ (Nat.add b d)))) (Int.subNatNat (Nat.add a " +
			"b) (Nat.succ (Nat.add b d)))",
		"fun (a : Nat) (b : Nat) (d : Nat) => Eq.trans.{1} Int (Int.add (Int.ofNat a) (Int.subNatNat b (Nat.succ (Nat.add b d)))) (Int.subNatNat a " +
			"(Nat.succ d)) (Int.subNatNat (Nat.add a b) (Nat.succ (Nat.add b d))) (congrArg.{1, 1} Int Int (Int.add (Int.ofNat a)) (Int.subNatNat b " +
			"(Nat.succ (Nat.add b d))) (Int.negSucc d) (Int.subNatNat_add_right b d)) (Eq.symm.{1} Int (Int.subNatNat (Nat.add a b) (Nat.succ (Nat.add b " +
			"d))) (Int.subNatNat a (Nat.succ d)) (Eq.trans.{1} Int (Int.subNatNat (Nat.add a b) (Nat.succ (Nat.add b d))) (Int.subNatNat (Nat.add a b) " +
			"(Nat.add (Nat.succ d) b)) (Int.subNatNat a (Nat.succ d)) (congrArg.{1, 1} Nat Int (Int.subNatNat (Nat.add a b)) (Nat.succ (Nat.add b d)) " +
			"(Nat.add (Nat.succ d) b) (Eq.trans.{1} Nat (Nat.succ (Nat.add b d)) (Nat.succ (Nat.add d b)) (Nat.add (Nat.succ d) b) (congrArg.{1, 1} Nat " +
			"Nat Nat.succ (Nat.add b d) (Nat.add d b) (Nat.add_comm b d)) (Eq.symm.{1} Nat (Nat.add (Nat.succ d) b) (Nat.succ (Nat.add d b)) " +
			"(Nat.succ_add d b)))) (Int.subNatNat_add_add a (Nat.succ d) b)))"},
	// `ofNat a + subNatNat b c = subNatNat (a + b) c`: split on `b < c` or `c <= b`, write `c` (resp. `b`) as the sum the two shapes need, and rewrite.
	valDecl{"thm", "Int.ofNat_add_subNatNat", nil,
		"(a : Nat) → (b : Nat) → (c : Nat) → Eq.{1} Int (Int.add (Int.ofNat a) (Int.subNatNat b c)) (Int.subNatNat (Nat.add a b) c)",
		"fun (a : Nat) (b : Nat) (c : Nat) => Or.rec (Nat.lt b c) (Nat.le c b) (fun (_ : Or (Nat.lt b c) (Nat.le c b)) => Eq.{1} Int (Int.add " +
			"(Int.ofNat a) (Int.subNatNat b c)) (Int.subNatNat (Nat.add a b) c)) (fun (h : Nat.lt b c) => Eq.rec.{0, 1} Nat (Nat.succ (Nat.add b (Nat.sub " +
			"c (Nat.succ b)))) (fun (z : Nat) (_ : Eq.{1} Nat (Nat.succ (Nat.add b (Nat.sub c (Nat.succ b)))) z) => Eq.{1} Int (Int.add (Int.ofNat a) " +
			"(Int.subNatNat b z)) (Int.subNatNat (Nat.add a b) z)) (Int.ofNat_add_subNatNat_gt a b (Nat.sub c (Nat.succ b))) c (Eq.trans.{1} Nat " +
			"(Nat.succ (Nat.add b (Nat.sub c (Nat.succ b)))) (Nat.add (Nat.succ b) (Nat.sub c (Nat.succ b))) c (Eq.symm.{1} Nat (Nat.add (Nat.succ b) " +
			"(Nat.sub c (Nat.succ b))) (Nat.succ (Nat.add b (Nat.sub c (Nat.succ b)))) (Nat.succ_add b (Nat.sub c (Nat.succ b)))) (Nat.add_sub_of_le " +
			"(Nat.succ b) c h))) (fun (h : Nat.le c b) => Eq.rec.{0, 1} Nat (Nat.add c (Nat.sub b c)) (fun (z : Nat) (_ : Eq.{1} Nat (Nat.add c (Nat.sub " +
			"b c)) z) => Eq.{1} Int (Int.add (Int.ofNat a) (Int.subNatNat z c)) (Int.subNatNat (Nat.add a z) c)) (Int.ofNat_add_subNatNat_le a c (Nat.sub " +
			"b c)) b (Nat.add_sub_of_le c b h)) (Nat.lt_or_ge b c)"},
	// `negSucc a + subNatNat b c = subNatNat b (succ (a + c))`: the same split as `ofNat_add_subNatNat`. For `b = c + d` both sides are `subNatNat d (succ a)`
	// (`subNatNat_add_add` after turning `c + d` and `succ (a + c)` round); for `c = succ (b + d)` both are `negSucc (succ (a + d))` (`subNatNat_add_right`).
	valDecl{"thm", "Int.negSucc_add_subNatNat", nil,
		"(a : Nat) → (b : Nat) → (c : Nat) → Eq.{1} Int (Int.add (Int.negSucc a) (Int.subNatNat b c)) (Int.subNatNat b (Nat.succ (Nat.add a c)))",
		"fun (a : Nat) (b : Nat) (c : Nat) => Or.rec (Nat.lt b c) (Nat.le c b) (fun (_ : Or (Nat.lt b c) (Nat.le c b)) => Eq.{1} Int (Int.add " +
			"(Int.negSucc a) (Int.subNatNat b c)) (Int.subNatNat b (Nat.succ (Nat.add a c)))) (fun (h : Nat.lt b c) => Eq.rec.{0, 1} Nat (Nat.succ " +
			"(Nat.add b (Nat.sub c (Nat.succ b)))) (fun (z : Nat) (_ : Eq.{1} Nat (Nat.succ (Nat.add b (Nat.sub c (Nat.succ b)))) z) => Eq.{1} Int " +
			"(Int.add (Int.negSucc a) (Int.subNatNat b z)) (Int.subNatNat b (Nat.succ (Nat.add a z)))) (Eq.trans.{1} Int (Int.add (Int.negSucc a) " +
			"(Int.subNatNat b (Nat.succ (Nat.add b (Nat.sub c (Nat.succ b)))))) (Int.negSucc (Nat.succ (Nat.add a (Nat.sub c (Nat.succ b))))) " +
			"(Int.subNatNat b (Nat.succ (Nat.add a (Nat.succ (Nat.add b (Nat.sub c (Nat.succ b))))))) (congrArg.{1, 1} Int Int (Int.add (Int.negSucc a)) " +
			"(Int.subNatNat b (Nat.succ (Nat.add b (Nat.sub c (Nat.succ b))))) (Int.negSucc (Nat.sub c (Nat.succ b))) (Int.subNatNat_add_right b (Nat.sub " +
			"c (Nat.succ b)))) (Eq.symm.{1} Int (Int.subNatNat b (Nat.succ (Nat.add a (Nat.succ (Nat.add b (Nat.sub c (Nat.succ b))))))) (Int.negSucc " +
			"(Nat.succ (Nat.add a (Nat.sub c (Nat.succ b))))) (Eq.trans.{1} Int (Int.subNatNat b (Nat.succ (Nat.add a (Nat.succ (Nat.add b (Nat.sub c " +
			"(Nat.succ b))))))) (Int.subNatNat b (Nat.succ (Nat.add b (Nat.succ (Nat.add a (Nat.sub c (Nat.succ b))))))) (Int.negSucc (Nat.succ (Nat.add " +
			"a (Nat.sub c (Nat.succ b))))) (congrArg.{1, 1} Nat Int (Int.subNatNat b) (Nat.succ (Nat.add a (Nat.succ (Nat.add b (Nat.sub c (Nat.succ " +
			"b)))))) (Nat.succ (Nat.add b (Nat.succ (Nat.add a (Nat.sub c (Nat.succ b)))))) (congrArg.{1, 1} Nat Nat Nat.succ (Nat.add a (Nat.succ " +
			"(Nat.add b (Nat.sub c (Nat.succ b))))) (Nat.add b (Nat.succ (Nat.add a (Nat.sub c (Nat.succ b))))) (congrArg.{1, 1} Nat Nat Nat.succ " +
			"(Nat.add a (Nat.add b (Nat.sub c (Nat.succ b)))) (Nat.add b (Nat.add a (Nat.sub c (Nat.succ b)))) (Nat.add_left_comm a b (Nat.sub c " +
			"(Nat.succ b)))))) (Int.subNatNat_add_right b (Nat.succ (Nat.add a (Nat.sub c (Nat.succ b)))))))) c (Eq.trans.{1} Nat (Nat.succ (Nat.add b " +
			"(Nat.sub c (Nat.succ b)))) (Nat.add (Nat.succ b) (Nat.sub c (Nat.succ b))) c (Eq.symm.{1} Nat (Nat.add (Nat.succ b) (Nat.sub c (Nat.succ " +
			"b))) (Nat.succ (Nat.add b (Nat.sub c (Nat.succ b)))) (Nat.succ_add b (Nat.sub c (Nat.succ b)))) (Nat.add_sub_of_le (Nat.succ b) c h))) (fun " +
			"(h : Nat.le c b) => Eq.rec.{0, 1} Nat (Nat.add c (Nat.sub b c)) (fun (z : Nat) (_ : Eq.{1} Nat (Nat.add c (Nat.sub b c)) z) => Eq.{1} Int " +
			"(Int.add (Int.negSucc a) (Int.subNatNat z c)) (Int.subNatNat z (Nat.succ (Nat.add a c)))) (Eq.trans.{1} Int (Int.add (Int.negSucc a) " +
			"(Int.subNatNat (Nat.add c (Nat.sub b c)) c)) (Int.subNatNat (Nat.sub b c) (Nat.succ a)) (Int.subNatNat (Nat.add c (Nat.sub b c)) (Nat.succ " +
			"(Nat.add a c))) (congrArg.{1, 1} Int Int (Int.add (Int.negSucc a)) (Int.subNatNat (Nat.add c (Nat.sub b c)) c) (Int.ofNat (Nat.sub b c)) " +
			"(Int.subNatNat_add_left c (Nat.sub b c))) (Eq.symm.{1} Int (Int.subNatNat (Nat.add c (Nat.sub b c)) (Nat.succ (Nat.add a c))) (Int.subNatNat " +
			"(Nat.sub b c) (Nat.succ a)) (Eq.trans.{1} Int (Int.subNatNat (Nat.add c (Nat.sub b c)) (Nat.succ (Nat.add a c))) (Int.subNatNat (Nat.add " +
			"(Nat.sub b c) c) (Nat.succ (Nat.add a c))) (Int.subNatNat (Nat.sub b c) (Nat.succ a)) (congrArg.{1, 1} Nat Int (fun (x : Nat) => " +
			"Int.subNatNat x (Nat.succ (Nat.add a c))) (Nat.add c (Nat.sub b c)) (Nat.add (Nat.sub b c) c) (Nat.add_comm c (Nat.sub b c))) (Eq.trans.{1} " +
			"Int (Int.subNatNat (Nat.add (Nat.sub b c) c) (Nat.succ (Nat.add a c))) (Int.subNatNat (Nat.add (Nat.sub b c) c) (Nat.add (Nat.succ a) c)) " +
			"(Int.subNatNat (Nat.sub b c) (Nat.succ a)) (congrArg.{1, 1} Nat Int (Int.subNatNat (Nat.add (Nat.sub b c) c)) (Nat.succ (Nat.add a c)) " +
			"(Nat.add (Nat.succ a) c) (Eq.symm.{1} Nat (Nat.add (Nat.succ a) c) (Nat.succ (Nat.add a c)) (Nat.succ_add a c))) (Int.subNatNat_add_add " +
			"(Nat.sub b c) (Nat.succ a) c))))) b (Nat.add_sub_of_le c b h)) (Nat.lt_or_ge b c)"},
	// `(a + b) + c = a + (b + c)`: eight constructor cases. Three of the same sign are `Nat.add_assoc` with `succ`s; the mixed ones are
	// `subNatNat` sums, moved by `ofNat_add_subNatNat` and `negSucc_add_subNatNat` (with `add_comm` when the `subNatNat` is on the left).
	valDecl{"thm", "Int.add_assoc", nil,
		"(A : Int) → (B : Int) → (C : Int) → Eq.{1} Int (Int.add (Int.add A B) C) (Int.add A (Int.add B C))",
		"fun (A : Int) (B : Int) (C : Int) => Int.rec.{0} (fun (A : Int) => (B : Int) → (C : Int) → Eq.{1} Int (Int.add (Int.add A B) C) (Int.add A " +
			"(Int.add B C))) (fun (a : Nat) (B : Int) (C : Int) => Int.rec.{0} (fun (B : Int) => (C : Int) → Eq.{1} Int (Int.add (Int.add (Int.ofNat a) " +
			"B) C) (Int.add (Int.ofNat a) (Int.add B C))) (fun (b : Nat) (C : Int) => Int.rec.{0} (fun (C : Int) => Eq.{1} Int (Int.add (Int.add " +
			"(Int.ofNat a) (Int.ofNat b)) C) (Int.add (Int.ofNat a) (Int.add (Int.ofNat b) C))) (fun (c : Nat) => (congrArg.{1, 1} Nat Int Int.ofNat " +
			"(Nat.add (Nat.add a b) c) (Nat.add a (Nat.add b c)) (Nat.add_assoc a b c))) (fun (c : Nat) => (Eq.symm.{1} Int (Int.add (Int.ofNat a) " +
			"(Int.subNatNat b (Nat.succ c))) (Int.subNatNat (Nat.add a b) (Nat.succ c)) (Int.ofNat_add_subNatNat a b (Nat.succ c)))) C) (fun (b : Nat) (C " +
			": Int) => Int.rec.{0} (fun (C : Int) => Eq.{1} Int (Int.add (Int.add (Int.ofNat a) (Int.negSucc b)) C) (Int.add (Int.ofNat a) (Int.add " +
			"(Int.negSucc b) C))) (fun (c : Nat) => (Eq.trans.{1} Int (Int.add (Int.subNatNat a (Nat.succ b)) (Int.ofNat c)) (Int.subNatNat (Nat.add a c) " +
			"(Nat.succ b)) (Int.add (Int.ofNat a) (Int.subNatNat c (Nat.succ b))) (Eq.trans.{1} Int (Int.add (Int.subNatNat a (Nat.succ b)) (Int.ofNat " +
			"c)) (Int.subNatNat (Nat.add c a) (Nat.succ b)) (Int.subNatNat (Nat.add a c) (Nat.succ b)) (Eq.trans.{1} Int (Int.add (Int.subNatNat a " +
			"(Nat.succ b)) (Int.ofNat c)) (Int.add (Int.ofNat c) (Int.subNatNat a (Nat.succ b))) (Int.subNatNat (Nat.add c a) (Nat.succ b)) (Int.add_comm " +
			"(Int.subNatNat a (Nat.succ b)) (Int.ofNat c)) (Int.ofNat_add_subNatNat c a (Nat.succ b))) (congrArg.{1, 1} Nat Int (fun (x : Nat) => " +
			"Int.subNatNat x (Nat.succ b)) (Nat.add c a) (Nat.add a c) (Nat.add_comm c a))) (Eq.symm.{1} Int (Int.add (Int.ofNat a) (Int.subNatNat c " +
			"(Nat.succ b))) (Int.subNatNat (Nat.add a c) (Nat.succ b)) (Int.ofNat_add_subNatNat a c (Nat.succ b))))) (fun (c : Nat) => (Eq.trans.{1} Int " +
			"(Int.add (Int.subNatNat a (Nat.succ b)) (Int.negSucc c)) (Int.subNatNat a (Nat.succ (Nat.add c (Nat.succ b)))) (Int.subNatNat a (Nat.succ " +
			"(Nat.succ (Nat.add b c)))) (Eq.trans.{1} Int (Int.add (Int.subNatNat a (Nat.succ b)) (Int.negSucc c)) (Int.add (Int.negSucc c) " +
			"(Int.subNatNat a (Nat.succ b))) (Int.subNatNat a (Nat.succ (Nat.add c (Nat.succ b)))) (Int.add_comm (Int.subNatNat a (Nat.succ b)) " +
			"(Int.negSucc c)) (Int.negSucc_add_subNatNat c a (Nat.succ b))) (congrArg.{1, 1} Nat Int (Int.subNatNat a) (Nat.succ (Nat.add c (Nat.succ " +
			"b))) (Nat.succ (Nat.succ (Nat.add b c))) (congrArg.{1, 1} Nat Nat Nat.succ (Nat.succ (Nat.add c b)) (Nat.succ (Nat.add b c)) (congrArg.{1, " +
			"1} Nat Nat Nat.succ (Nat.add c b) (Nat.add b c) (Nat.add_comm c b)))))) C) B C) (fun (a : Nat) (B : Int) (C : Int) => Int.rec.{0} (fun (B : " +
			"Int) => (C : Int) → Eq.{1} Int (Int.add (Int.add (Int.negSucc a) B) C) (Int.add (Int.negSucc a) (Int.add B C))) (fun (b : Nat) (C : Int) => " +
			"Int.rec.{0} (fun (C : Int) => Eq.{1} Int (Int.add (Int.add (Int.negSucc a) (Int.ofNat b)) C) (Int.add (Int.negSucc a) (Int.add (Int.ofNat b) " +
			"C))) (fun (c : Nat) => (Eq.trans.{1} Int (Int.add (Int.subNatNat b (Nat.succ a)) (Int.ofNat c)) (Int.subNatNat (Nat.add c b) (Nat.succ a)) " +
			"(Int.subNatNat (Nat.add b c) (Nat.succ a)) (Eq.trans.{1} Int (Int.add (Int.subNatNat b (Nat.succ a)) (Int.ofNat c)) (Int.add (Int.ofNat c) " +
			"(Int.subNatNat b (Nat.succ a))) (Int.subNatNat (Nat.add c b) (Nat.succ a)) (Int.add_comm (Int.subNatNat b (Nat.succ a)) (Int.ofNat c)) " +
			"(Int.ofNat_add_subNatNat c b (Nat.succ a))) (congrArg.{1, 1} Nat Int (fun (x : Nat) => Int.subNatNat x (Nat.succ a)) (Nat.add c b) (Nat.add " +
			"b c) (Nat.add_comm c b)))) (fun (c : Nat) => (Eq.trans.{1} Int (Int.add (Int.subNatNat b (Nat.succ a)) (Int.negSucc c)) (Int.subNatNat b " +
			"(Nat.succ (Nat.add a (Nat.succ c)))) (Int.add (Int.negSucc a) (Int.subNatNat b (Nat.succ c))) (Eq.trans.{1} Int (Int.add (Int.subNatNat b " +
			"(Nat.succ a)) (Int.negSucc c)) (Int.subNatNat b (Nat.succ (Nat.add c (Nat.succ a)))) (Int.subNatNat b (Nat.succ (Nat.add a (Nat.succ c)))) " +
			"(Eq.trans.{1} Int (Int.add (Int.subNatNat b (Nat.succ a)) (Int.negSucc c)) (Int.add (Int.negSucc c) (Int.subNatNat b (Nat.succ a))) " +
			"(Int.subNatNat b (Nat.succ (Nat.add c (Nat.succ a)))) (Int.add_comm (Int.subNatNat b (Nat.succ a)) (Int.negSucc c)) " +
			"(Int.negSucc_add_subNatNat c b (Nat.succ a))) (congrArg.{1, 1} Nat Int (Int.subNatNat b) (Nat.succ (Nat.add c (Nat.succ a))) (Nat.succ " +
			"(Nat.add a (Nat.succ c))) (congrArg.{1, 1} Nat Nat Nat.succ (Nat.add c (Nat.succ a)) (Nat.add a (Nat.succ c)) (congrArg.{1, 1} Nat Nat " +
			"Nat.succ (Nat.add c a) (Nat.add a c) (Nat.add_comm c a))))) (Eq.symm.{1} Int (Int.add (Int.negSucc a) (Int.subNatNat b (Nat.succ c))) " +
			"(Int.subNatNat b (Nat.succ (Nat.add a (Nat.succ c)))) (Int.negSucc_add_subNatNat a b (Nat.succ c))))) C) (fun (b : Nat) (C : Int) => " +
			"Int.rec.{0} (fun (C : Int) => Eq.{1} Int (Int.add (Int.add (Int.negSucc a) (Int.negSucc b)) C) (Int.add (Int.negSucc a) (Int.add " +
			"(Int.negSucc b) C))) (fun (c : Nat) => (Eq.symm.{1} Int (Int.add (Int.negSucc a) (Int.subNatNat c (Nat.succ b))) (Int.subNatNat c (Nat.succ " +
			"(Nat.succ (Nat.add a b)))) (Int.negSucc_add_subNatNat a c (Nat.succ b)))) (fun (c : Nat) => (congrArg.{1, 1} Nat Int Int.negSucc (Nat.succ " +
			"(Nat.add (Nat.succ (Nat.add a b)) c)) (Nat.succ (Nat.add a (Nat.succ (Nat.add b c)))) (Eq.trans.{1} Nat (Nat.succ (Nat.add (Nat.succ " +
			"(Nat.add a b)) c)) (Nat.succ (Nat.succ (Nat.add (Nat.add a b) c))) (Nat.succ (Nat.succ (Nat.add a (Nat.add b c)))) (congrArg.{1, 1} Nat Nat " +
			"Nat.succ (Nat.add (Nat.succ (Nat.add a b)) c) (Nat.succ (Nat.add (Nat.add a b) c)) (Nat.succ_add (Nat.add a b) c)) (congrArg.{1, 1} Nat Nat " +
			"Nat.succ (Nat.succ (Nat.add (Nat.add a b) c)) (Nat.succ (Nat.add a (Nat.add b c))) (congrArg.{1, 1} Nat Nat Nat.succ (Nat.add (Nat.add a b) " +
			"c) (Nat.add a (Nat.add b c)) (Nat.add_assoc a b c)))))) C) B C) A B C"},
	// From `add_assoc`: `a + b = a + c` gives `b = c` by adding `-a` on the left (`neg_add_cancel`, `zero_add`); the right-hand form goes through `add_comm`.
	// `a + (b + c) = b + (a + c)` is `add_assoc` both ways around an `add_comm`.
	valDecl{"thm", "Int.add_left_cancel", nil,
		"(a : Int) → (b : Int) → (c : Int) → Eq.{1} Int (Int.add a b) (Int.add a c) → Eq.{1} Int b c",
		"fun (a : Int) (b : Int) (c : Int) (h : Eq.{1} Int (Int.add a b) (Int.add a c)) => Eq.trans.{1} Int b (Int.add (Int.ofNat 0) c) c " +
			"(Eq.trans.{1} Int b (Int.add (Int.add (Int.neg a) a) c) (Int.add (Int.ofNat 0) c) (Eq.trans.{1} Int b (Int.add (Int.neg a) (Int.add a c)) " +
			"(Int.add (Int.add (Int.neg a) a) c) (Eq.trans.{1} Int b (Int.add (Int.neg a) (Int.add a b)) (Int.add (Int.neg a) (Int.add a c)) " +
			"(Eq.trans.{1} Int b (Int.add (Int.add (Int.neg a) a) b) (Int.add (Int.neg a) (Int.add a b)) (Eq.trans.{1} Int b (Int.add (Int.ofNat 0) b) " +
			"(Int.add (Int.add (Int.neg a) a) b) (Eq.symm.{1} Int (Int.add (Int.ofNat 0) b) b (Int.zero_add b)) (congrArg.{1, 1} Int Int (fun (x : Int) " +
			"=> Int.add x b) (Int.ofNat 0) (Int.add (Int.neg a) a) (Eq.symm.{1} Int (Int.add (Int.neg a) a) (Int.ofNat 0) (Int.neg_add_cancel a)))) " +
			"(Int.add_assoc (Int.neg a) a b)) (congrArg.{1, 1} Int Int (Int.add (Int.neg a)) (Int.add a b) (Int.add a c) (h))) (Eq.symm.{1} Int (Int.add " +
			"(Int.add (Int.neg a) a) c) (Int.add (Int.neg a) (Int.add a c)) (Int.add_assoc (Int.neg a) a c))) (congrArg.{1, 1} Int Int (fun (x : Int) => " +
			"Int.add x c) (Int.add (Int.neg a) a) (Int.ofNat 0) (Int.neg_add_cancel a))) (Int.zero_add c)"},
	valDecl{"thm", "Int.add_right_cancel", nil,
		"(a : Int) → (b : Int) → (c : Int) → Eq.{1} Int (Int.add b a) (Int.add c a) → Eq.{1} Int b c",
		"fun (a : Int) (b : Int) (c : Int) (h : Eq.{1} Int (Int.add b a) (Int.add c a)) => Int.add_left_cancel a b c (Eq.trans.{1} Int (Int.add a b) " +
			"(Int.add c a) (Int.add a c) (Eq.trans.{1} Int (Int.add a b) (Int.add b a) (Int.add c a) (Int.add_comm a b) (h)) (Int.add_comm c a))"},
	valDecl{"thm", "Int.add_left_comm", nil,
		"(a : Int) → (b : Int) → (c : Int) → Eq.{1} Int (Int.add a (Int.add b c)) (Int.add b (Int.add a c))",
		"fun (a : Int) (b : Int) (c : Int) => Eq.trans.{1} Int (Int.add a (Int.add b c)) (Int.add (Int.add b a) c) (Int.add b (Int.add a c)) " +
			"(Eq.trans.{1} Int (Int.add a (Int.add b c)) (Int.add (Int.add a b) c) (Int.add (Int.add b a) c) (Eq.symm.{1} Int (Int.add (Int.add a b) c) " +
			"(Int.add a (Int.add b c)) (Int.add_assoc a b c)) (congrArg.{1, 1} Int Int (fun (x : Int) => Int.add x c) (Int.add a b) (Int.add b a) " +
			"(Int.add_comm a b))) (Int.add_assoc b a c)"},
	// `-(a + b) = -a + -b`: by `add_left_cancel` at `a + b`, both `(a + b) + -(a + b)` and `(a + b) + (-a + -b)` are `0`; the second reassociates to `b + -b`.
	valDecl{"thm", "Int.neg_add", nil,
		"(a : Int) → (b : Int) → Eq.{1} Int (Int.neg (Int.add a b)) (Int.add (Int.neg a) (Int.neg b))",
		"fun (a : Int) (b : Int) => Int.add_left_cancel (Int.add a b) (Int.neg (Int.add a b)) (Int.add (Int.neg a) (Int.neg b)) (Eq.trans.{1} Int " +
			"(Int.add (Int.add a b) (Int.neg (Int.add a b))) (Int.ofNat 0) (Int.add (Int.add a b) (Int.add (Int.neg a) (Int.neg b))) (Int.add_neg_cancel " +
			"(Int.add a b)) (Eq.symm.{1} Int (Int.add (Int.add a b) (Int.add (Int.neg a) (Int.neg b))) (Int.ofNat 0) (Eq.trans.{1} Int (Int.add (Int.add " +
			"a b) (Int.add (Int.neg a) (Int.neg b))) (Int.add b (Int.neg b)) (Int.ofNat 0) (Eq.trans.{1} Int (Int.add (Int.add a b) (Int.add (Int.neg a) " +
			"(Int.neg b))) (Int.add (Int.add b (Int.ofNat 0)) (Int.neg b)) (Int.add b (Int.neg b)) (Eq.trans.{1} Int (Int.add (Int.add a b) (Int.add " +
			"(Int.neg a) (Int.neg b))) (Int.add (Int.add b (Int.add a (Int.neg a))) (Int.neg b)) (Int.add (Int.add b (Int.ofNat 0)) (Int.neg b)) " +
			"(Eq.trans.{1} Int (Int.add (Int.add a b) (Int.add (Int.neg a) (Int.neg b))) (Int.add (Int.add (Int.add b a) (Int.neg a)) (Int.neg b)) " +
			"(Int.add (Int.add b (Int.add a (Int.neg a))) (Int.neg b)) (Eq.trans.{1} Int (Int.add (Int.add a b) (Int.add (Int.neg a) (Int.neg b))) " +
			"(Int.add (Int.add (Int.add a b) (Int.neg a)) (Int.neg b)) (Int.add (Int.add (Int.add b a) (Int.neg a)) (Int.neg b)) (Eq.symm.{1} Int " +
			"(Int.add (Int.add (Int.add a b) (Int.neg a)) (Int.neg b)) (Int.add (Int.add a b) (Int.add (Int.neg a) (Int.neg b))) (Int.add_assoc (Int.add " +
			"a b) (Int.neg a) (Int.neg b))) (congrArg.{1, 1} Int Int (fun (x : Int) => Int.add (Int.add x (Int.neg a)) (Int.neg b)) (Int.add a b) " +
			"(Int.add b a) (Int.add_comm a b))) (congrArg.{1, 1} Int Int (fun (x : Int) => Int.add x (Int.neg b)) (Int.add (Int.add b a) (Int.neg a)) " +
			"(Int.add b (Int.add a (Int.neg a))) (Int.add_assoc b a (Int.neg a)))) (congrArg.{1, 1} Int Int (fun (x : Int) => Int.add (Int.add b x) " +
			"(Int.neg b)) (Int.add a (Int.neg a)) (Int.ofNat 0) (Int.add_neg_cancel a))) (congrArg.{1, 1} Int Int (fun (x : Int) => Int.add x (Int.neg " +
			"b)) (Int.add b (Int.ofNat 0)) b (Int.add_zero b))) (Int.add_neg_cancel b))))"},
	// `negOfNat (u + succ n) = negOfNat u + negSucc n`: `u = 0` is `negSucc (0 + n) = negSucc n`, `u = succ j` is `negSucc (succ j + n) = negSucc (succ (j + n))` by `succ_add`.
	// Then `ofNat (succ k) * x = ofNat k * x + x`: for `x = ofNat n` it is `Nat.succ_mul`; for `x = negSucc n` it is `succ_mul` under `negOfNat`, then the lemma above.
	valDecl{"thm", "Int.negOfNat_add_negSucc", nil,
		"(u : Nat) → (n : Nat) → Eq.{1} Int (Int.negOfNat (Nat.add u (Nat.succ n))) (Int.add (Int.negOfNat u) (Int.negSucc n))",
		"fun (u : Nat) (n : Nat) => Nat.rec.{0} (fun (u : Nat) => Eq.{1} Int (Int.negOfNat (Nat.add u (Nat.succ n))) (Int.add (Int.negOfNat u) " +
			"(Int.negSucc n))) (congrArg.{1, 1} Nat Int Int.negSucc (Nat.add 0 n) n (Nat.zero_add n)) (fun (j : Nat) (_ : Eq.{1} Int (Int.negOfNat " +
			"(Nat.add j (Nat.succ n))) (Int.add (Int.negOfNat j) (Int.negSucc n))) => (congrArg.{1, 1} Nat Int Int.negSucc (Nat.add (Nat.succ j) n) " +
			"(Nat.succ (Nat.add j n)) (Nat.succ_add j n))) u"},
	valDecl{"thm", "Int.ofNat_succ_mul", nil,
		"(k : Nat) → (x : Int) → Eq.{1} Int (Int.mul (Int.ofNat (Nat.succ k)) x) (Int.add (Int.mul (Int.ofNat k) x) x)",
		"fun (k : Nat) (x : Int) => Int.rec.{0} (fun (x : Int) => Eq.{1} Int (Int.mul (Int.ofNat (Nat.succ k)) x) (Int.add (Int.mul (Int.ofNat k) x) " +
			"x)) (fun (n : Nat) => (congrArg.{1, 1} Nat Int Int.ofNat (Nat.mul (Nat.succ k) n) (Nat.add (Nat.mul k n) n) (Nat.succ_mul k n))) (fun (n : " +
			"Nat) => Eq.trans.{1} Int (Int.negOfNat (Nat.mul (Nat.succ k) (Nat.succ n))) (Int.negOfNat (Nat.add (Nat.mul k (Nat.succ n)) (Nat.succ n))) " +
			"(Int.add (Int.negOfNat (Nat.mul k (Nat.succ n))) (Int.negSucc n)) (congrArg.{1, 1} Nat Int Int.negOfNat (Nat.mul (Nat.succ k) (Nat.succ n)) " +
			"(Nat.add (Nat.mul k (Nat.succ n)) (Nat.succ n)) (Nat.succ_mul k (Nat.succ n))) (Int.negOfNat_add_negSucc (Nat.mul k (Nat.succ n)) n)) x"},
	// Distributivity. `(p + q) + (b + c) = (p + b) + (q + c)` is `add_assoc` around `add_left_comm`. `ofNat k * (b + c)` is induction on `k`: `0` by `zero_mul`,
	// the step by `ofNat_succ_mul` and that rearrangement. `negSucc k` is `-(ofNat (succ k))`, so `neg_mul` and `neg_add` carry it over. `add_mul` is `mul_add` and `mul_comm`.
	valDecl{"thm", "Int.add_add_add_comm", nil,
		"(p : Int) → (q : Int) → (b : Int) → (c : Int) → Eq.{1} Int (Int.add (Int.add p q) (Int.add b c)) (Int.add (Int.add p b) (Int.add q c))",
		"fun (p : Int) (q : Int) (b : Int) (c : Int) => Eq.trans.{1} Int (Int.add (Int.add p q) (Int.add b c)) (Int.add p (Int.add b (Int.add q c))) " +
			"(Int.add (Int.add p b) (Int.add q c)) (Eq.trans.{1} Int (Int.add (Int.add p q) (Int.add b c)) (Int.add p (Int.add q (Int.add b c))) (Int.add " +
			"p (Int.add b (Int.add q c))) (Int.add_assoc p q (Int.add b c)) (congrArg.{1, 1} Int Int (Int.add p) (Int.add q (Int.add b c)) (Int.add b " +
			"(Int.add q c)) (Int.add_left_comm q b c))) (Eq.symm.{1} Int (Int.add (Int.add p b) (Int.add q c)) (Int.add p (Int.add b (Int.add q c))) " +
			"(Int.add_assoc p b (Int.add q c)))"},
	valDecl{"thm", "Int.ofNat_mul_add", nil,
		"(k : Nat) → (b : Int) → (c : Int) → Eq.{1} Int (Int.mul (Int.ofNat k) (Int.add b c)) (Int.add (Int.mul (Int.ofNat k) b) (Int.mul (Int.ofNat " +
			"k) c))",
		"fun (k : Nat) (b : Int) (c : Int) => Nat.rec.{0} (fun (k : Nat) => Eq.{1} Int (Int.mul (Int.ofNat k) (Int.add b c)) (Int.add (Int.mul " +
			"(Int.ofNat k) b) (Int.mul (Int.ofNat k) c))) (Eq.trans.{1} Int (Int.mul (Int.ofNat 0) (Int.add b c)) (Int.ofNat 0) (Int.add (Int.mul " +
			"(Int.ofNat 0) b) (Int.mul (Int.ofNat 0) c)) (Int.zero_mul (Int.add b c)) (Eq.symm.{1} Int (Int.add (Int.mul (Int.ofNat 0) b) (Int.mul " +
			"(Int.ofNat 0) c)) (Int.ofNat 0) (Eq.trans.{1} Int (Int.add (Int.mul (Int.ofNat 0) b) (Int.mul (Int.ofNat 0) c)) (Int.add (Int.ofNat 0) " +
			"(Int.mul (Int.ofNat 0) c)) (Int.add (Int.ofNat 0) (Int.ofNat 0)) (congrArg.{1, 1} Int Int (fun (x : Int) => Int.add x (Int.mul (Int.ofNat 0) " +
			"c)) (Int.mul (Int.ofNat 0) b) (Int.ofNat 0) (Int.zero_mul b)) (congrArg.{1, 1} Int Int (Int.add (Int.ofNat 0)) (Int.mul (Int.ofNat 0) c) " +
			"(Int.ofNat 0) (Int.zero_mul c))))) (fun (k : Nat) (ih : Eq.{1} Int (Int.mul (Int.ofNat k) (Int.add b c)) (Int.add (Int.mul (Int.ofNat k) b) " +
			"(Int.mul (Int.ofNat k) c))) => Eq.trans.{1} Int (Int.mul (Int.ofNat (Nat.succ k)) (Int.add b c)) (Int.add (Int.add (Int.mul (Int.ofNat k) b) " +
			"b) (Int.add (Int.mul (Int.ofNat k) c) c)) (Int.add (Int.mul (Int.ofNat (Nat.succ k)) b) (Int.mul (Int.ofNat (Nat.succ k)) c)) (Eq.trans.{1} " +
			"Int (Int.mul (Int.ofNat (Nat.succ k)) (Int.add b c)) (Int.add (Int.add (Int.mul (Int.ofNat k) b) (Int.mul (Int.ofNat k) c)) (Int.add b c)) " +
			"(Int.add (Int.add (Int.mul (Int.ofNat k) b) b) (Int.add (Int.mul (Int.ofNat k) c) c)) (Eq.trans.{1} Int (Int.mul (Int.ofNat (Nat.succ k)) " +
			"(Int.add b c)) (Int.add (Int.mul (Int.ofNat k) (Int.add b c)) (Int.add b c)) (Int.add (Int.add (Int.mul (Int.ofNat k) b) (Int.mul (Int.ofNat " +
			"k) c)) (Int.add b c)) (Int.ofNat_succ_mul k (Int.add b c)) (congrArg.{1, 1} Int Int (fun (x : Int) => Int.add x (Int.add b c)) (Int.mul " +
			"(Int.ofNat k) (Int.add b c)) (Int.add (Int.mul (Int.ofNat k) b) (Int.mul (Int.ofNat k) c)) (ih))) (Int.add_add_add_comm (Int.mul (Int.ofNat " +
			"k) b) (Int.mul (Int.ofNat k) c) b c)) (Eq.symm.{1} Int (Int.add (Int.mul (Int.ofNat (Nat.succ k)) b) (Int.mul (Int.ofNat (Nat.succ k)) c)) " +
			"(Int.add (Int.add (Int.mul (Int.ofNat k) b) b) (Int.add (Int.mul (Int.ofNat k) c) c)) (Eq.trans.{1} Int (Int.add (Int.mul (Int.ofNat " +
			"(Nat.succ k)) b) (Int.mul (Int.ofNat (Nat.succ k)) c)) (Int.add (Int.add (Int.mul (Int.ofNat k) b) b) (Int.mul (Int.ofNat (Nat.succ k)) c)) " +
			"(Int.add (Int.add (Int.mul (Int.ofNat k) b) b) (Int.add (Int.mul (Int.ofNat k) c) c)) (congrArg.{1, 1} Int Int (fun (x : Int) => Int.add x " +
			"(Int.mul (Int.ofNat (Nat.succ k)) c)) (Int.mul (Int.ofNat (Nat.succ k)) b) (Int.add (Int.mul (Int.ofNat k) b) b) (Int.ofNat_succ_mul k b)) " +
			"(congrArg.{1, 1} Int Int (Int.add (Int.add (Int.mul (Int.ofNat k) b) b)) (Int.mul (Int.ofNat (Nat.succ k)) c) (Int.add (Int.mul (Int.ofNat " +
			"k) c) c) (Int.ofNat_succ_mul k c))))) k"},
	valDecl{"thm", "Int.mul_add", nil,
		"(a : Int) → (b : Int) → (c : Int) → Eq.{1} Int (Int.mul a (Int.add b c)) (Int.add (Int.mul a b) (Int.mul a c))",
		"fun (a : Int) (b : Int) (c : Int) => Int.rec.{0} (fun (a : Int) => Eq.{1} Int (Int.mul a (Int.add b c)) (Int.add (Int.mul a b) (Int.mul a " +
			"c))) (fun (k : Nat) => Int.ofNat_mul_add k b c) (fun (k : Nat) => Eq.trans.{1} Int (Int.mul (Int.negSucc k) (Int.add b c)) (Int.add (Int.neg " +
			"(Int.mul (Int.ofNat (Nat.succ k)) b)) (Int.neg (Int.mul (Int.ofNat (Nat.succ k)) c))) (Int.add (Int.mul (Int.negSucc k) b) (Int.mul " +
			"(Int.negSucc k) c)) (Eq.trans.{1} Int (Int.mul (Int.negSucc k) (Int.add b c)) (Int.neg (Int.add (Int.mul (Int.ofNat (Nat.succ k)) b) " +
			"(Int.mul (Int.ofNat (Nat.succ k)) c))) (Int.add (Int.neg (Int.mul (Int.ofNat (Nat.succ k)) b)) (Int.neg (Int.mul (Int.ofNat (Nat.succ k)) " +
			"c))) (Eq.trans.{1} Int (Int.mul (Int.negSucc k) (Int.add b c)) (Int.neg (Int.mul (Int.ofNat (Nat.succ k)) (Int.add b c))) (Int.neg (Int.add " +
			"(Int.mul (Int.ofNat (Nat.succ k)) b) (Int.mul (Int.ofNat (Nat.succ k)) c))) (Int.neg_mul (Int.ofNat (Nat.succ k)) (Int.add b c)) " +
			"(congrArg.{1, 1} Int Int Int.neg (Int.mul (Int.ofNat (Nat.succ k)) (Int.add b c)) (Int.add (Int.mul (Int.ofNat (Nat.succ k)) b) (Int.mul " +
			"(Int.ofNat (Nat.succ k)) c)) (Int.ofNat_mul_add (Nat.succ k) b c))) (Int.neg_add (Int.mul (Int.ofNat (Nat.succ k)) b) (Int.mul (Int.ofNat " +
			"(Nat.succ k)) c))) (Eq.symm.{1} Int (Int.add (Int.mul (Int.negSucc k) b) (Int.mul (Int.negSucc k) c)) (Int.add (Int.neg (Int.mul (Int.ofNat " +
			"(Nat.succ k)) b)) (Int.neg (Int.mul (Int.ofNat (Nat.succ k)) c))) (Eq.trans.{1} Int (Int.add (Int.mul (Int.negSucc k) b) (Int.mul " +
			"(Int.negSucc k) c)) (Int.add (Int.neg (Int.mul (Int.ofNat (Nat.succ k)) b)) (Int.mul (Int.negSucc k) c)) (Int.add (Int.neg (Int.mul " +
			"(Int.ofNat (Nat.succ k)) b)) (Int.neg (Int.mul (Int.ofNat (Nat.succ k)) c))) (congrArg.{1, 1} Int Int (fun (x : Int) => Int.add x (Int.mul " +
			"(Int.negSucc k) c)) (Int.mul (Int.negSucc k) b) (Int.neg (Int.mul (Int.ofNat (Nat.succ k)) b)) (Int.neg_mul (Int.ofNat (Nat.succ k)) b)) " +
			"(congrArg.{1, 1} Int Int (Int.add (Int.neg (Int.mul (Int.ofNat (Nat.succ k)) b))) (Int.mul (Int.negSucc k) c) (Int.neg (Int.mul (Int.ofNat " +
			"(Nat.succ k)) c)) (Int.neg_mul (Int.ofNat (Nat.succ k)) c))))) a"},
	valDecl{"thm", "Int.add_mul", nil,
		"(a : Int) → (b : Int) → (c : Int) → Eq.{1} Int (Int.mul (Int.add a b) c) (Int.add (Int.mul a c) (Int.mul b c))",
		"fun (a : Int) (b : Int) (c : Int) => Eq.trans.{1} Int (Int.mul (Int.add a b) c) (Int.add (Int.mul a c) (Int.mul c b)) (Int.add (Int.mul a c) " +
			"(Int.mul b c)) (Eq.trans.{1} Int (Int.mul (Int.add a b) c) (Int.add (Int.mul c a) (Int.mul c b)) (Int.add (Int.mul a c) (Int.mul c b)) " +
			"(Eq.trans.{1} Int (Int.mul (Int.add a b) c) (Int.mul c (Int.add a b)) (Int.add (Int.mul c a) (Int.mul c b)) (Int.mul_comm (Int.add a b) c) " +
			"(Int.mul_add c a b)) (congrArg.{1, 1} Int Int (fun (x : Int) => Int.add x (Int.mul c b)) (Int.mul c a) (Int.mul a c) (Int.mul_comm c a))) " +
			"(congrArg.{1, 1} Int Int (Int.add (Int.mul a c)) (Int.mul c b) (Int.mul b c) (Int.mul_comm c b))"},
	// With `a - b = a + -b`: `mul_sub` and `sub_mul` are `mul_add`/`add_mul` then `mul_neg`/`neg_mul`; `-(a - b) = b - a` is `neg_add`, `neg_neg`, `add_comm`;
	// `a - b + b = a` is `add_assoc`, `neg_add_cancel`, `add_zero`.
	valDecl{"thm", "Int.mul_sub", nil,
		"(a : Int) → (b : Int) → (c : Int) → Eq.{1} Int (Int.mul a (Int.sub b c)) (Int.sub (Int.mul a b) (Int.mul a c))",
		"fun (a : Int) (b : Int) (c : Int) => Eq.trans.{1} Int (Int.mul a (Int.sub b c)) (Int.add (Int.mul a b) (Int.mul a (Int.neg c))) (Int.add " +
			"(Int.mul a b) (Int.neg (Int.mul a c))) (Int.mul_add a b (Int.neg c)) (congrArg.{1, 1} Int Int (Int.add (Int.mul a b)) (Int.mul a (Int.neg " +
			"c)) (Int.neg (Int.mul a c)) (Int.mul_neg a c))"},
	valDecl{"thm", "Int.sub_mul", nil,
		"(a : Int) → (b : Int) → (c : Int) → Eq.{1} Int (Int.mul (Int.sub a b) c) (Int.sub (Int.mul a c) (Int.mul b c))",
		"fun (a : Int) (b : Int) (c : Int) => Eq.trans.{1} Int (Int.mul (Int.sub a b) c) (Int.add (Int.mul a c) (Int.mul (Int.neg b) c)) (Int.add " +
			"(Int.mul a c) (Int.neg (Int.mul b c))) (Int.add_mul a (Int.neg b) c) (congrArg.{1, 1} Int Int (Int.add (Int.mul a c)) (Int.mul (Int.neg b) " +
			"c) (Int.neg (Int.mul b c)) (Int.neg_mul b c))"},
	valDecl{"thm", "Int.neg_sub", nil,
		"(a : Int) → (b : Int) → Eq.{1} Int (Int.neg (Int.sub a b)) (Int.sub b a)",
		"fun (a : Int) (b : Int) => Eq.trans.{1} Int (Int.neg (Int.sub a b)) (Int.add (Int.neg a) b) (Int.add b (Int.neg a)) (Eq.trans.{1} Int " +
			"(Int.neg (Int.sub a b)) (Int.add (Int.neg a) (Int.neg (Int.neg b))) (Int.add (Int.neg a) b) (Int.neg_add a (Int.neg b)) (congrArg.{1, 1} Int " +
			"Int (Int.add (Int.neg a)) (Int.neg (Int.neg b)) b (Int.neg_neg b))) (Int.add_comm (Int.neg a) b)"},
	valDecl{"thm", "Int.sub_add_cancel", nil,
		"(a : Int) → (b : Int) → Eq.{1} Int (Int.add (Int.sub a b) b) a",
		"fun (a : Int) (b : Int) => Eq.trans.{1} Int (Int.add (Int.sub a b) b) (Int.add a (Int.ofNat 0)) a (Eq.trans.{1} Int (Int.add (Int.sub a b) " +
			"b) (Int.add a (Int.add (Int.neg b) b)) (Int.add a (Int.ofNat 0)) (Int.add_assoc a (Int.neg b) b) (congrArg.{1, 1} Int Int (Int.add a) " +
			"(Int.add (Int.neg b) b) (Int.ofNat 0) (Int.neg_add_cancel b))) (Int.add_zero a)"},
	// `a - b = 0` iff `a = b`: `a = (a - b) + b = 0 + b = b` one way, `a - a = 0` (`sub_self`) the other. `-a = -b` gives `a = b` through `neg_neg`.
	valDecl{"thm", "Int.eq_of_sub_eq_zero", nil,
		"(a : Int) → (b : Int) → Eq.{1} Int (Int.sub a b) (Int.ofNat 0) → Eq.{1} Int a b",
		"fun (a : Int) (b : Int) (h : Eq.{1} Int (Int.sub a b) (Int.ofNat 0)) => Eq.trans.{1} Int a (Int.add (Int.ofNat 0) b) b (Eq.trans.{1} Int a " +
			"(Int.add (Int.sub a b) b) (Int.add (Int.ofNat 0) b) (Eq.symm.{1} Int (Int.add (Int.sub a b) b) a (Int.sub_add_cancel a b)) (congrArg.{1, 1} " +
			"Int Int (fun (x : Int) => Int.add x b) (Int.sub a b) (Int.ofNat 0) (h))) (Int.zero_add b)"},
	valDecl{"thm", "Int.sub_eq_zero_of_eq", nil,
		"(a : Int) → (b : Int) → Eq.{1} Int a b → Eq.{1} Int (Int.sub a b) (Int.ofNat 0)",
		"fun (a : Int) (b : Int) (h : Eq.{1} Int a b) => Eq.trans.{1} Int (Int.sub a b) (Int.sub b b) (Int.ofNat 0) (congrArg.{1, 1} Int Int (fun (x " +
			": Int) => Int.sub x b) a b (h)) (Int.sub_self b)"},
	valDecl{"thm", "Int.sub_eq_zero", nil,
		"(a : Int) → (b : Int) → Iff (Eq.{1} Int (Int.sub a b) (Int.ofNat 0)) (Eq.{1} Int a b)",
		"fun (a : Int) (b : Int) => Iff.intro (Eq.{1} Int (Int.sub a b) (Int.ofNat 0)) (Eq.{1} Int a b) (Int.eq_of_sub_eq_zero a b) " +
			"(Int.sub_eq_zero_of_eq a b)"},
	valDecl{"thm", "Int.neg_inj", nil,
		"(a : Int) → (b : Int) → Eq.{1} Int (Int.neg a) (Int.neg b) → Eq.{1} Int a b",
		"fun (a : Int) (b : Int) (h : Eq.{1} Int (Int.neg a) (Int.neg b)) => Eq.trans.{1} Int a (Int.neg (Int.neg b)) b (Eq.trans.{1} Int a (Int.neg " +
			"(Int.neg a)) (Int.neg (Int.neg b)) (Eq.symm.{1} Int (Int.neg (Int.neg a)) a (Int.neg_neg a)) (congrArg.{1, 1} Int Int Int.neg (Int.neg a) " +
			"(Int.neg b) (h))) (Int.neg_neg b)"},
	// `a * b = 0` gives `a = 0 or b = 0`: take `natAbs` (`natAbs_mul`), use `Nat.mul_eq_zero`, and come back through `natAbs_eq_zero`.
	// `mul_left_cancel`: for `a != 0`, `a * (b - c) = 0` by `mul_sub`, so `b - c = 0`.
	valDecl{"thm", "Int.mul_eq_zero", nil,
		"(a : Int) → (b : Int) → Eq.{1} Int (Int.mul a b) (Int.ofNat 0) → Or (Eq.{1} Int a (Int.ofNat 0)) (Eq.{1} Int b (Int.ofNat 0))",
		"fun (a : Int) (b : Int) (h : Eq.{1} Int (Int.mul a b) (Int.ofNat 0)) => Or.rec (Eq.{1} Nat (Int.natAbs a) 0) (Eq.{1} Nat (Int.natAbs b) 0) " +
			"(fun (_ : Or (Eq.{1} Nat (Int.natAbs a) 0) (Eq.{1} Nat (Int.natAbs b) 0)) => Or (Eq.{1} Int a (Int.ofNat 0)) (Eq.{1} Int b (Int.ofNat 0))) " +
			"(fun (h1 : Eq.{1} Nat (Int.natAbs a) 0) => Or.inl (Eq.{1} Int a (Int.ofNat 0)) (Eq.{1} Int b (Int.ofNat 0)) (Iff.mp (Eq.{1} Nat (Int.natAbs " +
			"a) 0) (Eq.{1} Int a (Int.ofNat 0)) (Int.natAbs_eq_zero a) h1)) (fun (h2 : Eq.{1} Nat (Int.natAbs b) 0) => Or.inr (Eq.{1} Int a (Int.ofNat " +
			"0)) (Eq.{1} Int b (Int.ofNat 0)) (Iff.mp (Eq.{1} Nat (Int.natAbs b) 0) (Eq.{1} Int b (Int.ofNat 0)) (Int.natAbs_eq_zero b) h2)) " +
			"(Nat.mul_eq_zero (Int.natAbs a) (Int.natAbs b) (Eq.trans.{1} Nat (Nat.mul (Int.natAbs a) (Int.natAbs b)) (Int.natAbs (Int.mul a b)) 0 " +
			"(Eq.symm.{1} Nat (Int.natAbs (Int.mul a b)) (Nat.mul (Int.natAbs a) (Int.natAbs b)) (Int.natAbs_mul a b)) (congrArg.{1, 1} Int Nat " +
			"Int.natAbs (Int.mul a b) (Int.ofNat 0) h)))"},
	valDecl{"thm", "Int.mul_left_cancel", nil,
		"(a : Int) → (b : Int) → (c : Int) → Not (Eq.{1} Int a (Int.ofNat 0)) → Eq.{1} Int (Int.mul a b) (Int.mul a c) → Eq.{1} Int b c",
		"fun (a : Int) (b : Int) (c : Int) (ha : Not (Eq.{1} Int a (Int.ofNat 0))) (h : Eq.{1} Int (Int.mul a b) (Int.mul a c)) => Or.rec (Eq.{1} Int " +
			"a (Int.ofNat 0)) (Eq.{1} Int (Int.sub b c) (Int.ofNat 0)) (fun (_ : Or (Eq.{1} Int a (Int.ofNat 0)) (Eq.{1} Int (Int.sub b c) (Int.ofNat " +
			"0))) => Eq.{1} Int b c) (fun (h1 : Eq.{1} Int a (Int.ofNat 0)) => False.elim.{0} (Eq.{1} Int b c) (ha h1)) (fun (h2 : Eq.{1} Int (Int.sub b " +
			"c) (Int.ofNat 0)) => Int.eq_of_sub_eq_zero b c h2) (Int.mul_eq_zero a (Int.sub b c) (Eq.trans.{1} Int (Int.mul a (Int.sub b c)) (Int.sub " +
			"(Int.mul a b) (Int.mul a c)) (Int.ofNat 0) (Int.mul_sub a b c) (Int.sub_eq_zero_of_eq (Int.mul a b) (Int.mul a c) h)))"},
	// `mul_right_cancel` is `mul_left_cancel` after `mul_comm` on both sides. `a + b = 0` gives `-a = b`: `-a = -a + 0 = -a + (a + b) = (-a + a) + b = b`.
	valDecl{"thm", "Int.mul_right_cancel", nil,
		"(a : Int) → (b : Int) → (c : Int) → Not (Eq.{1} Int c (Int.ofNat 0)) → Eq.{1} Int (Int.mul a c) (Int.mul b c) → Eq.{1} Int a b",
		"fun (a : Int) (b : Int) (c : Int) (hc : Not (Eq.{1} Int c (Int.ofNat 0))) (h : Eq.{1} Int (Int.mul a c) (Int.mul b c)) => " +
			"Int.mul_left_cancel c a b hc (Eq.trans.{1} Int (Int.mul c a) (Int.mul b c) (Int.mul c b) (Eq.trans.{1} Int (Int.mul c a) (Int.mul a c) " +
			"(Int.mul b c) (Int.mul_comm c a) (h)) (Int.mul_comm b c))"},
	valDecl{"thm", "Int.neg_eq_of_add_eq_zero", nil,
		"(a : Int) → (b : Int) → Eq.{1} Int (Int.add a b) (Int.ofNat 0) → Eq.{1} Int (Int.neg a) b",
		"fun (a : Int) (b : Int) (h : Eq.{1} Int (Int.add a b) (Int.ofNat 0)) => Eq.trans.{1} Int (Int.neg a) (Int.add (Int.ofNat 0) b) b " +
			"(Eq.trans.{1} Int (Int.neg a) (Int.add (Int.add (Int.neg a) a) b) (Int.add (Int.ofNat 0) b) (Eq.trans.{1} Int (Int.neg a) (Int.add (Int.neg " +
			"a) (Int.add a b)) (Int.add (Int.add (Int.neg a) a) b) (Eq.trans.{1} Int (Int.neg a) (Int.add (Int.neg a) (Int.ofNat 0)) (Int.add (Int.neg a) " +
			"(Int.add a b)) (Eq.symm.{1} Int (Int.add (Int.neg a) (Int.ofNat 0)) (Int.neg a) (Int.add_zero (Int.neg a))) (congrArg.{1, 1} Int Int " +
			"(Int.add (Int.neg a)) (Int.ofNat 0) (Int.add a b) (Eq.symm.{1} Int (Int.add a b) (Int.ofNat 0) (h)))) (Eq.symm.{1} Int (Int.add (Int.add " +
			"(Int.neg a) a) b) (Int.add (Int.neg a) (Int.add a b)) (Int.add_assoc (Int.neg a) a b))) (congrArg.{1, 1} Int Int (fun (x : Int) => Int.add x " +
			"b) (Int.add (Int.neg a) a) (Int.ofNat 0) (Int.neg_add_cancel a))) (Int.zero_add b)"},
	// Order on Int, where `a <= b` is `NonNeg (b - a)`. `NonNeg` is closed under `+` (its one constructor is on `ofNat`, so `ofNat n + ofNat m` is `ofNat (n + m)`);
	// `le_refl` is `NonNeg 0` by `sub_self`; `le_trans` is `(c - b) + (b - a) = c - a` and `NonNeg_add`.
	valDecl{"thm", "Int.NonNeg_add", nil,
		"(x : Int) → (y : Int) → (Int.NonNeg x) → (Int.NonNeg y) → (Int.NonNeg (Int.add x y))",
		"fun (x : Int) (y : Int) (hx : (Int.NonNeg x)) (hy : (Int.NonNeg y)) => Int.NonNeg.rec (fun (j : Int) (_ : (Int.NonNeg j)) => (y : Int) → " +
			"(Int.NonNeg y) → (Int.NonNeg (Int.add j y))) (fun (n : Nat) (y : Int) (hy : (Int.NonNeg y)) => Int.NonNeg.rec (fun (j : Int) (_ : " +
			"(Int.NonNeg j)) => (Int.NonNeg (Int.add (Int.ofNat n) j))) (fun (m : Nat) => Int.NonNeg.mk (Nat.add n m)) y hy) x hx y hy"},
	valDecl{"thm", "Int.le_refl", nil,
		"(a : Int) → (Int.le a a)",
		"fun (a : Int) => Eq.rec.{0, 1} Int (Int.ofNat 0) (fun (z : Int) (_ : Eq.{1} Int (Int.ofNat 0) z) => (Int.NonNeg z)) (Int.NonNeg.mk 0) " +
			"(Int.sub a a) (Eq.symm.{1} Int (Int.sub a a) (Int.ofNat 0) (Int.sub_self a))"},
	valDecl{"thm", "Int.le_trans", nil,
		"(a : Int) → (b : Int) → (c : Int) → (Int.le a b) → (Int.le b c) → (Int.le a c)",
		"fun (a : Int) (b : Int) (c : Int) (h1 : (Int.le a b)) (h2 : (Int.le b c)) => Eq.rec.{0, 1} Int (Int.add (Int.sub c b) (Int.sub b a)) (fun (z " +
			": Int) (_ : Eq.{1} Int (Int.add (Int.sub c b) (Int.sub b a)) z) => (Int.NonNeg z)) (Int.NonNeg_add (Int.sub c b) (Int.sub b a) h2 h1) " +
			"(Int.sub c a) (Eq.trans.{1} Int (Int.add (Int.sub c b) (Int.sub b a)) (Int.add c (Int.add (Int.ofNat 0) (Int.neg a))) (Int.sub c a) " +
			"(Eq.trans.{1} Int (Int.add (Int.sub c b) (Int.sub b a)) (Int.add c (Int.add (Int.add (Int.neg b) b) (Int.neg a))) (Int.add c (Int.add " +
			"(Int.ofNat 0) (Int.neg a))) (Eq.trans.{1} Int (Int.add (Int.sub c b) (Int.sub b a)) (Int.add c (Int.add (Int.neg b) (Int.sub b a))) (Int.add " +
			"c (Int.add (Int.add (Int.neg b) b) (Int.neg a))) (Int.add_assoc c (Int.neg b) (Int.sub b a)) (congrArg.{1, 1} Int Int (Int.add c) (Int.add " +
			"(Int.neg b) (Int.sub b a)) (Int.add (Int.add (Int.neg b) b) (Int.neg a)) (Eq.symm.{1} Int (Int.add (Int.add (Int.neg b) b) (Int.neg a)) " +
			"(Int.add (Int.neg b) (Int.sub b a)) (Int.add_assoc (Int.neg b) b (Int.neg a))))) (congrArg.{1, 1} Int Int (fun (x : Int) => Int.add c " +
			"(Int.add x (Int.neg a))) (Int.add (Int.neg b) b) (Int.ofNat 0) (Int.neg_add_cancel b))) (congrArg.{1, 1} Int Int (Int.add c) (Int.add " +
			"(Int.ofNat 0) (Int.neg a)) (Int.neg a) (Int.zero_add (Int.neg a))))"},
	// `a <= b` gives `a + c <= b + c`: `(b + c) - (a + c) = b - a` by `neg_add`, `add_add_add_comm`, `sub_self`, `add_zero`. The left-hand form turns the sums round by `add_comm`.
	valDecl{"thm", "Int.add_le_add_right", nil,
		"(a : Int) → (b : Int) → (c : Int) → (Int.le a b) → (Int.le (Int.add a c) (Int.add b c))",
		"fun (a : Int) (b : Int) (c : Int) (h : (Int.le a b)) => Eq.rec.{0, 1} Int (Int.sub b a) (fun (z : Int) (_ : Eq.{1} Int (Int.sub b a) z) => " +
			"(Int.NonNeg z)) h (Int.sub (Int.add b c) (Int.add a c)) (Eq.symm.{1} Int (Int.sub (Int.add b c) (Int.add a c)) (Int.sub b a) (Eq.trans.{1} " +
			"Int (Int.sub (Int.add b c) (Int.add a c)) (Int.add (Int.sub b a) (Int.ofNat 0)) (Int.sub b a) (Eq.trans.{1} Int (Int.sub (Int.add b c) " +
			"(Int.add a c)) (Int.add (Int.sub b a) (Int.sub c c)) (Int.add (Int.sub b a) (Int.ofNat 0)) (Eq.trans.{1} Int (Int.sub (Int.add b c) (Int.add " +
			"a c)) (Int.add (Int.add b c) (Int.add (Int.neg a) (Int.neg c))) (Int.add (Int.sub b a) (Int.sub c c)) (congrArg.{1, 1} Int Int (Int.add " +
			"(Int.add b c)) (Int.neg (Int.add a c)) (Int.add (Int.neg a) (Int.neg c)) (Int.neg_add a c)) (Int.add_add_add_comm b c (Int.neg a) (Int.neg " +
			"c))) (congrArg.{1, 1} Int Int (Int.add (Int.sub b a)) (Int.sub c c) (Int.ofNat 0) (Int.sub_self c))) (Int.add_zero (Int.sub b a))))"},
	valDecl{"thm", "Int.add_le_add_left", nil,
		"(a : Int) → (b : Int) → (c : Int) → (Int.le a b) → (Int.le (Int.add c a) (Int.add c b))",
		"fun (a : Int) (b : Int) (c : Int) (h : (Int.le a b)) => Eq.rec.{0, 1} Int (Int.add a c) (fun (z : Int) (_ : Eq.{1} Int (Int.add a c) z) => " +
			"(Int.le z (Int.add c b))) (Eq.rec.{0, 1} Int (Int.add b c) (fun (w : Int) (_ : Eq.{1} Int (Int.add b c) w) => (Int.le (Int.add a c) w)) " +
			"(Int.add_le_add_right a b c h) (Int.add c b) (Int.add_comm b c)) (Int.add c a) (Int.add_comm a c)"},
	// `NonNeg (negSucc k)` is false (the one constructor is on `ofNat`). `NonNeg (subNatNat m n) <-> n <= m`: split on `Nat.ble n m`; true leaves `ofNat (m - n)`,
	// false leaves `negSucc _`. `ofNat n - ofNat m` is `subNatNat n m` by `rfl` on `m = 0` and `m = succ k`; so `ofNat m <= ofNat n <-> m <= n`.
	valDecl{"thm", "Int.not_nonneg_negSucc", nil,
		"(k : Nat) → Not (Int.NonNeg (Int.negSucc k))",
		"fun (k : Nat) (h : (Int.NonNeg (Int.negSucc k))) => Int.NonNeg.rec (fun (j : Int) (_ : (Int.NonNeg j)) => Eq.{1} Int j (Int.negSucc k) → " +
			"False) (fun (j : Nat) (e : Eq.{1} Int (Int.ofNat j) (Int.negSucc k)) => Bool.true_ne_false (congrArg.{1, 1} Int Bool Int.isNat (Int.ofNat j) " +
			"(Int.negSucc k) e)) (Int.negSucc k) h (Eq.refl.{1} Int (Int.negSucc k))"},
	valDecl{"thm", "Int.subNatNat_nonneg", nil,
		"(m : Nat) → (n : Nat) → Iff (Int.NonNeg (Int.subNatNat m n)) (Nat.le n m)",
		"fun (m : Nat) (n : Nat) => Iff.intro (Int.NonNeg (Int.subNatNat m n)) (Nat.le n m) (fun (h : (Int.NonNeg (Int.subNatNat m n))) => " +
			"Bool.rec.{0} (fun (z : Bool) => Eq.{1} Bool (Nat.ble n m) z → (Int.NonNeg (Bool.rec.{1} (fun (_ : Bool) => Int) (Int.negSucc (Nat.sub n " +
			"(Nat.succ m))) (Int.ofNat (Nat.sub m n)) z)) → (Nat.le n m)) (fun (e : Eq.{1} Bool (Nat.ble n m) Bool.false) (hn : (Int.NonNeg (Int.negSucc " +
			"(Nat.sub n (Nat.succ m))))) => False.elim.{0} (Nat.le n m) (Int.not_nonneg_negSucc (Nat.sub n (Nat.succ m)) hn)) (fun (e : Eq.{1} Bool " +
			"(Nat.ble n m) Bool.true) (_ : (Int.NonNeg (Int.ofNat (Nat.sub m n)))) => Nat.le_of_ble_eq_true n m e) (Nat.ble n m) (Eq.refl.{1} Bool " +
			"(Nat.ble n m)) h) (fun (h : (Nat.le n m)) => Eq.rec.{0, 1} Bool Bool.true (fun (z : Bool) (_ : Eq.{1} Bool Bool.true z) => (Int.NonNeg " +
			"(Bool.rec.{1} (fun (_ : Bool) => Int) (Int.negSucc (Nat.sub n (Nat.succ m))) (Int.ofNat (Nat.sub m n)) z))) (Int.NonNeg.mk (Nat.sub m n)) " +
			"(Nat.ble n m) (Eq.symm.{1} Bool (Nat.ble n m) Bool.true (Nat.ble_true_of_le n m h)))"},
	valDecl{"thm", "Int.ofNat_sub_ofNat", nil,
		"(n : Nat) → (m : Nat) → Eq.{1} Int (Int.sub (Int.ofNat n) (Int.ofNat m)) (Int.subNatNat n m)",
		"fun (n : Nat) (m : Nat) => Nat.rec.{0} (fun (m : Nat) => Eq.{1} Int (Int.sub (Int.ofNat n) (Int.ofNat m)) (Int.subNatNat n m)) (Eq.refl.{1} " +
			"Int (Int.ofNat n)) (fun (k : Nat) (_ : Eq.{1} Int (Int.sub (Int.ofNat n) (Int.ofNat k)) (Int.subNatNat n k)) => Eq.refl.{1} Int " +
			"(Int.subNatNat n (Nat.succ k))) m"},
	valDecl{"thm", "Int.ofNat_le", nil,
		"(m : Nat) → (n : Nat) → Iff (Int.le (Int.ofNat m) (Int.ofNat n)) (Nat.le m n)",
		"fun (m : Nat) (n : Nat) => Eq.rec.{0, 1} Int (Int.subNatNat n m) (fun (z : Int) (_ : Eq.{1} Int (Int.subNatNat n m) z) => Iff (Int.NonNeg z) " +
			"(Nat.le m n)) (Int.subNatNat_nonneg n m) (Int.sub (Int.ofNat n) (Int.ofNat m)) (Eq.symm.{1} Int (Int.sub (Int.ofNat n) (Int.ofNat m)) " +
			"(Int.subNatNat n m) (Int.ofNat_sub_ofNat n m))"},
	// `a <= b` or `b <= a`: case on `b - a`. If it is `ofNat n`, that is `NonNeg.mk n`; if it is `negSucc k`, then `a - b = -(b - a) = ofNat (succ k)` by `neg_sub`.
	valDecl{"thm", "Int.le_total", nil,
		"(a : Int) → (b : Int) → Or (Int.le a b) (Int.le b a)",
		"fun (a : Int) (b : Int) => Int.rec.{0} (fun (z : Int) => Eq.{1} Int (Int.sub b a) z → Or (Int.le a b) (Int.le b a)) (fun (n : Nat) (e : " +
			"Eq.{1} Int (Int.sub b a) (Int.ofNat n)) => Or.inl (Int.le a b) (Int.le b a) (Eq.rec.{0, 1} Int (Int.ofNat n) (fun (w : Int) (_ : Eq.{1} Int " +
			"(Int.ofNat n) w) => (Int.NonNeg w)) (Int.NonNeg.mk n) (Int.sub b a) (Eq.symm.{1} Int (Int.sub b a) (Int.ofNat n) e))) (fun (k : Nat) (e : " +
			"Eq.{1} Int (Int.sub b a) (Int.negSucc k)) => Or.inr (Int.le a b) (Int.le b a) (Eq.rec.{0, 1} Int (Int.ofNat (Nat.succ k)) (fun (w : Int) (_ " +
			": Eq.{1} Int (Int.ofNat (Nat.succ k)) w) => (Int.NonNeg w)) (Int.NonNeg.mk (Nat.succ k)) (Int.sub a b) (Eq.symm.{1} Int (Int.sub a b) " +
			"(Int.ofNat (Nat.succ k)) (Eq.trans.{1} Int (Int.sub a b) (Int.neg (Int.sub b a)) (Int.ofNat (Nat.succ k)) (Eq.symm.{1} Int (Int.neg (Int.sub " +
			"b a)) (Int.sub a b) (Int.neg_sub b a)) (congrArg.{1, 1} Int Int Int.neg (Int.sub b a) (Int.negSucc k) (e)))))) (Int.sub b a) (Eq.refl.{1} " +
			"Int (Int.sub b a))"},
	valDecl{"thm", "Int.le_of_eq", nil,
		"(a : Int) → (b : Int) → Eq.{1} Int a b → (Int.le a b)",
		"fun (a : Int) (b : Int) (h : Eq.{1} Int a b) => Eq.rec.{0, 1} Int a (fun (z : Int) (_ : Eq.{1} Int a z) => (Int.le a z)) (Int.le_refl a) b h"},
	// `NonNeg x` and `NonNeg (-x)` give `x = 0`: `x = ofNat 0` is `rfl`, `ofNat (succ j)` has `-x = negSucc j` (not `NonNeg`), and `negSucc k` is not `NonNeg` itself.
	// `le_antisymm`: `b - a` and `-(b - a) = a - b` are both `NonNeg`, so `b - a = 0`.
	valDecl{"thm", "Int.eq_zero_of_nonneg_neg", nil,
		"(x : Int) → (Int.NonNeg x) → (Int.NonNeg (Int.neg x)) → Eq.{1} Int x (Int.ofNat 0)",
		"fun (x : Int) => Int.rec.{0} (fun (x : Int) => (Int.NonNeg x) → (Int.NonNeg (Int.neg x)) → Eq.{1} Int x (Int.ofNat 0)) (fun (n : Nat) (_ : " +
			"(Int.NonNeg (Int.ofNat n))) => Nat.rec.{0} (fun (m : Nat) => (Int.NonNeg (Int.neg (Int.ofNat m))) → Eq.{1} Int (Int.ofNat m) (Int.ofNat 0)) " +
			"(fun (_ : (Int.NonNeg (Int.neg (Int.ofNat 0)))) => Eq.refl.{1} Int (Int.ofNat 0)) (fun (j : Nat) (_ : (Int.NonNeg (Int.neg (Int.ofNat j))) → " +
			"Eq.{1} Int (Int.ofNat j) (Int.ofNat 0)) (h : (Int.NonNeg (Int.negSucc j))) => False.elim.{0} (Eq.{1} Int (Int.ofNat (Nat.succ j)) (Int.ofNat " +
			"0)) (Int.not_nonneg_negSucc j h)) n) (fun (k : Nat) (h1 : (Int.NonNeg (Int.negSucc k))) (_ : (Int.NonNeg (Int.neg (Int.negSucc k)))) => " +
			"False.elim.{0} (Eq.{1} Int (Int.negSucc k) (Int.ofNat 0)) (Int.not_nonneg_negSucc k h1)) x"},
	valDecl{"thm", "Int.le_antisymm", nil,
		"(a : Int) → (b : Int) → (Int.le a b) → (Int.le b a) → Eq.{1} Int a b",
		"fun (a : Int) (b : Int) (h1 : (Int.le a b)) (h2 : (Int.le b a)) => Eq.symm.{1} Int b a (Int.eq_of_sub_eq_zero b a (Int.eq_zero_of_nonneg_neg " +
			"(Int.sub b a) h1 (Eq.rec.{0, 1} Int (Int.sub a b) (fun (w : Int) (_ : Eq.{1} Int (Int.sub a b) w) => (Int.NonNeg w)) h2 (Int.neg (Int.sub b " +
			"a)) (Eq.symm.{1} Int (Int.neg (Int.sub b a)) (Int.sub a b) (Int.neg_sub b a)))))"},
	// Strict order, where `a < b` is `a + 1 <= b`. `(a + 1) - a = 1` and `a - (a + 1) = negSucc 0` give `a <= a + 1` and `not (a < a)` (a `NonNeg (negSucc 0)`).
	// `lt_trans` and the mixed forms are `le_trans` with `le_of_lt` or `add_le_add_right`.
	valDecl{"thm", "Int.add_one_sub", nil,
		"(a : Int) → Eq.{1} Int (Int.sub (Int.add a (Int.ofNat 1)) a) (Int.ofNat 1)",
		"fun (a : Int) => Eq.trans.{1} Int (Int.sub (Int.add a (Int.ofNat 1)) a) (Int.add (Int.ofNat 1) (Int.ofNat 0)) (Int.ofNat 1) (Eq.trans.{1} " +
			"Int (Int.sub (Int.add a (Int.ofNat 1)) a) (Int.add (Int.ofNat 1) (Int.add a (Int.neg a))) (Int.add (Int.ofNat 1) (Int.ofNat 0)) " +
			"(Eq.trans.{1} Int (Int.sub (Int.add a (Int.ofNat 1)) a) (Int.add (Int.add (Int.ofNat 1) a) (Int.neg a)) (Int.add (Int.ofNat 1) (Int.add a " +
			"(Int.neg a))) (congrArg.{1, 1} Int Int (fun (x : Int) => Int.add x (Int.neg a)) (Int.add a (Int.ofNat 1)) (Int.add (Int.ofNat 1) a) " +
			"(Int.add_comm a (Int.ofNat 1))) (Int.add_assoc (Int.ofNat 1) a (Int.neg a))) (congrArg.{1, 1} Int Int (Int.add (Int.ofNat 1)) (Int.add a " +
			"(Int.neg a)) (Int.ofNat 0) (Int.add_neg_cancel a))) (Int.add_zero (Int.ofNat 1))"},
	valDecl{"thm", "Int.sub_add_one_self", nil,
		"(a : Int) → Eq.{1} Int (Int.sub a (Int.add a (Int.ofNat 1))) (Int.negSucc 0)",
		"fun (a : Int) => Eq.trans.{1} Int (Int.sub a (Int.add a (Int.ofNat 1))) (Int.add (Int.ofNat 0) (Int.neg (Int.ofNat 1))) (Int.negSucc 0) " +
			"(Eq.trans.{1} Int (Int.sub a (Int.add a (Int.ofNat 1))) (Int.add (Int.add a (Int.neg a)) (Int.neg (Int.ofNat 1))) (Int.add (Int.ofNat 0) " +
			"(Int.neg (Int.ofNat 1))) (Eq.trans.{1} Int (Int.sub a (Int.add a (Int.ofNat 1))) (Int.add a (Int.add (Int.neg a) (Int.neg (Int.ofNat 1)))) " +
			"(Int.add (Int.add a (Int.neg a)) (Int.neg (Int.ofNat 1))) (congrArg.{1, 1} Int Int (Int.add a) (Int.neg (Int.add a (Int.ofNat 1))) (Int.add " +
			"(Int.neg a) (Int.neg (Int.ofNat 1))) (Int.neg_add a (Int.ofNat 1))) (Eq.symm.{1} Int (Int.add (Int.add a (Int.neg a)) (Int.neg (Int.ofNat " +
			"1))) (Int.add a (Int.add (Int.neg a) (Int.neg (Int.ofNat 1)))) (Int.add_assoc a (Int.neg a) (Int.neg (Int.ofNat 1))))) (congrArg.{1, 1} Int " +
			"Int (fun (x : Int) => Int.add x (Int.neg (Int.ofNat 1))) (Int.add a (Int.neg a)) (Int.ofNat 0) (Int.add_neg_cancel a))) (Int.zero_add " +
			"(Int.neg (Int.ofNat 1)))"},
	valDecl{"thm", "Int.le_add_one", nil,
		"(a : Int) → (Int.le a (Int.add a (Int.ofNat 1)))",
		"fun (a : Int) => Eq.rec.{0, 1} Int (Int.ofNat 1) (fun (w : Int) (_ : Eq.{1} Int (Int.ofNat 1) w) => (Int.NonNeg w)) (Int.NonNeg.mk 1) " +
			"(Int.sub (Int.add a (Int.ofNat 1)) a) (Eq.symm.{1} Int (Int.sub (Int.add a (Int.ofNat 1)) a) (Int.ofNat 1) (Int.add_one_sub a))"},
	valDecl{"thm", "Int.lt_irrefl", nil,
		"(a : Int) → Not (Int.lt a a)",
		"fun (a : Int) (h : (Int.lt a a)) => Int.not_nonneg_negSucc 0 (Eq.rec.{0, 1} Int (Int.sub a (Int.add a (Int.ofNat 1))) (fun (w : Int) (_ : " +
			"Eq.{1} Int (Int.sub a (Int.add a (Int.ofNat 1))) w) => (Int.NonNeg w)) h (Int.negSucc 0) (Int.sub_add_one_self a))"},
	valDecl{"thm", "Int.le_of_lt", nil,
		"(a : Int) → (b : Int) → (Int.lt a b) → (Int.le a b)",
		"fun (a : Int) (b : Int) (h : (Int.lt a b)) => Int.le_trans a (Int.add a (Int.ofNat 1)) b (Int.le_add_one a) h"},
	valDecl{"thm", "Int.lt_trans", nil,
		"(a : Int) → (b : Int) → (c : Int) → (Int.lt a b) → (Int.lt b c) → (Int.lt a c)",
		"fun (a : Int) (b : Int) (c : Int) (h1 : (Int.lt a b)) (h2 : (Int.lt b c)) => Int.le_trans (Int.add a (Int.ofNat 1)) b c h1 (Int.le_of_lt b c " +
			"h2)"},
	valDecl{"thm", "Int.lt_of_le_of_lt", nil,
		"(a : Int) → (b : Int) → (c : Int) → (Int.le a b) → (Int.lt b c) → (Int.lt a c)",
		"fun (a : Int) (b : Int) (c : Int) (h1 : (Int.le a b)) (h2 : (Int.lt b c)) => Int.le_trans (Int.add a (Int.ofNat 1)) (Int.add b (Int.ofNat " +
			"1)) c (Int.add_le_add_right a b (Int.ofNat 1) h1) h2"},
	valDecl{"thm", "Int.lt_of_lt_of_le", nil,
		"(a : Int) → (b : Int) → (c : Int) → (Int.lt a b) → (Int.le b c) → (Int.lt a c)",
		"fun (a : Int) (b : Int) (c : Int) (h1 : (Int.lt a b)) (h2 : (Int.le b c)) => Int.le_trans (Int.add a (Int.ofNat 1)) b c h1 h2"},
	// `NonNeg` is closed under `*` (the same two `NonNeg.rec` as for `+`, ending in `ofNat (n * m)`), so `a <= b` and `NonNeg c` give `c * a <= c * b`:
	// `c * (b - a)` is `NonNeg` and equals `c * b - c * a` by `mul_sub`.
	valDecl{"thm", "Int.NonNeg_mul", nil,
		"(x : Int) → (y : Int) → (Int.NonNeg x) → (Int.NonNeg y) → (Int.NonNeg (Int.mul x y))",
		"fun (x : Int) (y : Int) (hx : (Int.NonNeg x)) (hy : (Int.NonNeg y)) => Int.NonNeg.rec (fun (j : Int) (_ : (Int.NonNeg j)) => (y : Int) → " +
			"(Int.NonNeg y) → (Int.NonNeg (Int.mul j y))) (fun (n : Nat) (y : Int) (hy : (Int.NonNeg y)) => Int.NonNeg.rec (fun (j : Int) (_ : " +
			"(Int.NonNeg j)) => (Int.NonNeg (Int.mul (Int.ofNat n) j))) (fun (m : Nat) => Int.NonNeg.mk (Nat.mul n m)) y hy) x hx y hy"},
	valDecl{"thm", "Int.mul_le_mul_of_nonneg_left", nil,
		"(a : Int) → (b : Int) → (c : Int) → (Int.NonNeg c) → (Int.le a b) → (Int.le (Int.mul c a) (Int.mul c b))",
		"fun (a : Int) (b : Int) (c : Int) (hc : (Int.NonNeg c)) (h : (Int.le a b)) => Eq.rec.{0, 1} Int (Int.mul c (Int.sub b a)) (fun (w : Int) (_ " +
			": Eq.{1} Int (Int.mul c (Int.sub b a)) w) => (Int.NonNeg w)) (Int.NonNeg_mul c (Int.sub b a) hc h) (Int.sub (Int.mul c b) (Int.mul c a)) " +
			"(Int.mul_sub c b a)"},
	// `mul_le_mul_of_nonneg_right` is the left form after `mul_comm` on both products. `a < b` rules out `b <= a`: together they give `a < a`.
	valDecl{"thm", "Int.mul_le_mul_of_nonneg_right", nil,
		"(a : Int) → (b : Int) → (c : Int) → (Int.NonNeg c) → (Int.le a b) → (Int.le (Int.mul a c) (Int.mul b c))",
		"fun (a : Int) (b : Int) (c : Int) (hc : (Int.NonNeg c)) (h : (Int.le a b)) => Eq.rec.{0, 1} Int (Int.mul c a) (fun (z : Int) (_ : Eq.{1} Int " +
			"(Int.mul c a) z) => (Int.le z (Int.mul b c))) (Eq.rec.{0, 1} Int (Int.mul c b) (fun (w : Int) (_ : Eq.{1} Int (Int.mul c b) w) => (Int.le " +
			"(Int.mul c a) w)) (Int.mul_le_mul_of_nonneg_left a b c hc h) (Int.mul b c) (Int.mul_comm c b)) (Int.mul a c) (Int.mul_comm c a)"},
	valDecl{"thm", "Int.not_le_of_lt", nil,
		"(a : Int) → (b : Int) → (Int.lt a b) → Not (Int.le b a)",
		"fun (a : Int) (b : Int) (h : (Int.lt a b)) (h2 : (Int.le b a)) => Int.lt_irrefl a (Int.lt_of_lt_of_le a b a h h2)"},
	// Division by and of zero. `a / 0 = 0` and `a % 0 = a` (Lean's convention): `ofNat k` by `Nat.div_zero`/`Nat.mod_zero`, `negSucc k` by `rfl` (the `emod` case
	// needs `subNatNat 0 (succ k) = negSucc k`, also `rfl`, after rewriting `k % 0` to `k`). `0 / b = 0` and `0 % b = 0` by `Nat.zero_div` and `Nat.zero_mod`.
	valDecl{"thm", "Int.ediv_zero", nil,
		"(a : Int) → Eq.{1} Int (Int.ediv a (Int.ofNat 0)) (Int.ofNat 0)",
		"fun (a : Int) => Int.rec.{0} (fun (a : Int) => Eq.{1} Int (Int.ediv a (Int.ofNat 0)) (Int.ofNat 0)) (fun (k : Nat) => congrArg.{1, 1} Nat " +
			"Int Int.ofNat (Nat.div k 0) 0 (Nat.div_zero k)) (fun (k : Nat) => Eq.refl.{1} Int (Int.ofNat 0)) a"},
	valDecl{"thm", "Int.emod_zero", nil,
		"(a : Int) → Eq.{1} Int (Int.emod a (Int.ofNat 0)) a",
		"fun (a : Int) => Int.rec.{0} (fun (a : Int) => Eq.{1} Int (Int.emod a (Int.ofNat 0)) a) (fun (k : Nat) => congrArg.{1, 1} Nat Int Int.ofNat " +
			"(Nat.mod k 0) k (Nat.mod_zero k)) (fun (k : Nat) => Eq.rec.{0, 1} Nat k (fun (z : Nat) (_ : Eq.{1} Nat k z) => Eq.{1} Int (Int.subNatNat 0 " +
			"(Nat.succ z)) (Int.negSucc k)) (Eq.refl.{1} Int (Int.negSucc k)) (Nat.mod k 0) (Eq.symm.{1} Nat (Nat.mod k 0) k (Nat.mod_zero k))) a"},
	valDecl{"thm", "Int.zero_ediv", nil,
		"(b : Int) → Eq.{1} Int (Int.ediv (Int.ofNat 0) b) (Int.ofNat 0)",
		"fun (b : Int) => Int.rec.{0} (fun (b : Int) => Eq.{1} Int (Int.ediv (Int.ofNat 0) b) (Int.ofNat 0)) (fun (n : Nat) => congrArg.{1, 1} Nat " +
			"Int Int.ofNat (Nat.div 0 n) 0 (Nat.zero_div n)) (fun (n : Nat) => congrArg.{1, 1} Nat Int Int.negOfNat (Nat.div 0 (Nat.succ n)) 0 " +
			"(Nat.zero_div (Nat.succ n))) b"},
	valDecl{"thm", "Int.zero_emod", nil,
		"(b : Int) → Eq.{1} Int (Int.emod (Int.ofNat 0) b) (Int.ofNat 0)",
		"fun (b : Int) => congrArg.{1, 1} Nat Int Int.ofNat (Nat.mod 0 (Int.natAbs b)) 0 (Nat.zero_mod (Int.natAbs b))"},
	// Division by one: `a / 1 = a` and `a % 1 = 0`, by cases with `Nat.div_one` and `Nat.mod_one` (`emod` on `negSucc k` rewrites `k % 1` to `0`, then `rfl`).
	valDecl{"thm", "Int.ediv_one", nil,
		"(a : Int) → Eq.{1} Int (Int.ediv a (Int.ofNat 1)) a",
		"fun (a : Int) => Int.rec.{0} (fun (a : Int) => Eq.{1} Int (Int.ediv a (Int.ofNat 1)) a) (fun (k : Nat) => congrArg.{1, 1} Nat Int Int.ofNat " +
			"(Nat.div k 1) k (Nat.div_one k)) (fun (k : Nat) => congrArg.{1, 1} Nat Int Int.negSucc (Nat.div k 1) k (Nat.div_one k)) a"},
	valDecl{"thm", "Int.emod_one", nil,
		"(a : Int) → Eq.{1} Int (Int.emod a (Int.ofNat 1)) (Int.ofNat 0)",
		"fun (a : Int) => Int.rec.{0} (fun (a : Int) => Eq.{1} Int (Int.emod a (Int.ofNat 1)) (Int.ofNat 0)) (fun (k : Nat) => congrArg.{1, 1} Nat " +
			"Int Int.ofNat (Nat.mod k 1) 0 (Nat.mod_one k)) (fun (k : Nat) => Eq.rec.{0, 1} Nat 0 (fun (z : Nat) (_ : Eq.{1} Nat 0 z) => Eq.{1} Int " +
			"(Int.subNatNat 1 (Nat.succ z)) (Int.ofNat 0)) (Eq.refl.{1} Int (Int.ofNat 0)) (Nat.mod k 1) (Eq.symm.{1} Nat (Nat.mod k 1) 0 (Nat.mod_one " +
			"k))) a"},
	// `a % a = 0` for every `a`: `ofNat k` by `Nat.mod_self`; `negSucc k` rewrites `k % succ k` to `k` (`Nat.mod_eq_of_lt`), leaving `subNatNat (succ k) (succ k) = 0` by `rfl`.
	valDecl{"thm", "Int.emod_self", nil,
		"(a : Int) → Eq.{1} Int (Int.emod a a) (Int.ofNat 0)",
		"fun (a : Int) => Int.rec.{0} (fun (a : Int) => Eq.{1} Int (Int.emod a a) (Int.ofNat 0)) (fun (k : Nat) => congrArg.{1, 1} Nat Int Int.ofNat " +
			"(Nat.mod k k) 0 (Nat.mod_self k)) (fun (k : Nat) => Eq.rec.{0, 1} Nat k (fun (z : Nat) (_ : Eq.{1} Nat k z) => Eq.{1} Int (Int.subNatNat " +
			"(Nat.succ k) (Nat.succ z)) (Int.ofNat 0)) (Int.subNatNat_self (Nat.succ k)) (Nat.mod k (Nat.succ k)) (Eq.symm.{1} Nat (Nat.mod k (Nat.succ " +
			"k)) k (Nat.mod_eq_of_lt k (Nat.succ k) (Nat.lt_succ_self k)))) a"},
	// `a / a = 1` for `a != 0`: `ofNat 0` is absurd, `ofNat (succ m)` is `Nat.div_self`, and `negSucc k / negSucc k = ofNat (succ (k / succ k))` with `k / succ k = 0`.
	valDecl{"thm", "Int.ediv_self", nil,
		"(a : Int) → Not (Eq.{1} Int a (Int.ofNat 0)) → Eq.{1} Int (Int.ediv a a) (Int.ofNat 1)",
		"fun (a : Int) => Int.rec.{0} (fun (a : Int) => Not (Eq.{1} Int a (Int.ofNat 0)) → Eq.{1} Int (Int.ediv a a) (Int.ofNat 1)) (fun (k : Nat) => " +
			"Nat.rec.{0} (fun (k : Nat) => Not (Eq.{1} Int (Int.ofNat k) (Int.ofNat 0)) → Eq.{1} Int (Int.ediv (Int.ofNat k) (Int.ofNat k)) (Int.ofNat " +
			"1)) (fun (h : Not (Eq.{1} Int (Int.ofNat 0) (Int.ofNat 0))) => False.elim.{0} (Eq.{1} Int (Int.ediv (Int.ofNat 0) (Int.ofNat 0)) (Int.ofNat " +
			"1)) (h (Eq.refl.{1} Int (Int.ofNat 0)))) (fun (m : Nat) (ih : Not (Eq.{1} Int (Int.ofNat m) (Int.ofNat 0)) → Eq.{1} Int (Int.ediv (Int.ofNat " +
			"m) (Int.ofNat m)) (Int.ofNat 1)) (h : Not (Eq.{1} Int (Int.ofNat (Nat.succ m)) (Int.ofNat 0))) => congrArg.{1, 1} Nat Int Int.ofNat (Nat.div " +
			"(Nat.succ m) (Nat.succ m)) 1 (Nat.div_self (Nat.succ m) (Nat.zero_lt_succ m))) k) (fun (k : Nat) (h : Not (Eq.{1} Int (Int.negSucc k) " +
			"(Int.ofNat 0))) => congrArg.{1, 1} Nat Int Int.ofNat ((Nat.succ (Nat.div k (Nat.succ k)))) (Nat.succ 0) (congrArg.{1, 1} Nat Nat Nat.succ " +
			"(Nat.div k (Nat.succ k)) 0 (Nat.div_eq_of_lt k (Nat.succ k) (Nat.lt_succ_self k)))) a"},
	// `a < b` is `a + 1 <= b` by definition, so the `Iff` is the identity both ways; `not_lt_of_le` is `not_le_of_lt` turned around.
	valDecl{"thm", "Int.lt_iff_add_one_le", nil,
		"(a : Int) → (b : Int) → Iff (Int.lt a b) (Int.le (Int.add a (Int.ofNat 1)) b)",
		"fun (a : Int) (b : Int) => Iff.intro (Int.lt a b) (Int.le (Int.add a (Int.ofNat 1)) b) (fun (h : (Int.lt a b)) => h) (fun (h : (Int.le " +
			"(Int.add a (Int.ofNat 1)) b)) => h)"},
	valDecl{"thm", "Int.not_lt_of_le", nil,
		"(a : Int) → (b : Int) → (Int.le b a) → Not (Int.lt a b)",
		"fun (a : Int) (b : Int) (h : (Int.le b a)) (h2 : (Int.lt a b)) => Int.not_le_of_lt a b h2 h"},
	// A `NonNeg` integer is `0` or has `NonNeg (j - 1)`: `NonNeg.rec`, then `Nat.rec` on the payload; `ofNat (succ m) - 1` reduces to `ofNat m`.
	valDecl{"thm", "Int.NonNeg_cases", nil,
		"(j : Int) → (Int.NonNeg j) → Or (Eq.{1} Int j (Int.ofNat 0)) ((Int.NonNeg (Int.sub j (Int.ofNat 1))))",
		"fun (j : Int) (h : (Int.NonNeg j)) => Int.NonNeg.rec (fun (j : Int) (_ : (Int.NonNeg j)) => Or (Eq.{1} Int j (Int.ofNat 0)) ((Int.NonNeg " +
			"(Int.sub j (Int.ofNat 1))))) (fun (n : Nat) => Nat.rec.{0} (fun (n : Nat) => Or (Eq.{1} Int (Int.ofNat n) (Int.ofNat 0)) ((Int.NonNeg " +
			"(Int.sub (Int.ofNat n) (Int.ofNat 1))))) (Or.inl (Eq.{1} Int (Int.ofNat 0) (Int.ofNat 0)) ((Int.NonNeg (Int.sub (Int.ofNat 0) (Int.ofNat " +
			"1)))) (Eq.refl.{1} Int (Int.ofNat 0))) (fun (m : Nat) (_ : Or (Eq.{1} Int (Int.ofNat m) (Int.ofNat 0)) ((Int.NonNeg (Int.sub (Int.ofNat m) " +
			"(Int.ofNat 1))))) => Or.inr (Eq.{1} Int (Int.ofNat (Nat.succ m)) (Int.ofNat 0)) ((Int.NonNeg (Int.sub (Int.ofNat (Nat.succ m)) (Int.ofNat " +
			"1)))) (Int.NonNeg.mk m)) n) j h"},
	// `b - (a + c) = (b - a) - c` from `neg_add` and `add_assoc`. Then `a <= b` gives `a = b` or `a < b`: `NonNeg_cases` on `b - a`; the `0` case is
	// `eq_of_sub_eq_zero`, the other transports `NonNeg ((b - a) - 1)` to `NonNeg (b - (a + 1))`, which is `a < b`.
	valDecl{"thm", "Int.sub_add_eq_sub_sub", nil,
		"(a : Int) → (b : Int) → (c : Int) → Eq.{1} Int (Int.sub b (Int.add a c)) (Int.sub (Int.sub b a) c)",
		"fun (a : Int) (b : Int) (c : Int) => Eq.trans.{1} Int (Int.sub b (Int.add a c)) (Int.add b (Int.add (Int.neg a) (Int.neg c))) (Int.sub " +
			"(Int.sub b a) c) (congrArg.{1, 1} Int Int (Int.add b) (Int.neg (Int.add a c)) (Int.add (Int.neg a) (Int.neg c)) (Int.neg_add a c)) " +
			"(Eq.symm.{1} Int (Int.add (Int.add b (Int.neg a)) (Int.neg c)) (Int.add b (Int.add (Int.neg a) (Int.neg c))) (Int.add_assoc b (Int.neg a) " +
			"(Int.neg c)))"},
	valDecl{"thm", "Int.eq_or_lt_of_le", nil,
		"(a : Int) → (b : Int) → (Int.le a b) → Or (Eq.{1} Int a b) (Int.lt a b)",
		"fun (a : Int) (b : Int) (h : (Int.le a b)) => Or.rec (Eq.{1} Int (Int.sub b a) (Int.ofNat 0)) ((Int.NonNeg (Int.sub (Int.sub b a) (Int.ofNat " +
			"1)))) (fun (_ : Or (Eq.{1} Int (Int.sub b a) (Int.ofNat 0)) ((Int.NonNeg (Int.sub (Int.sub b a) (Int.ofNat 1))))) => Or (Eq.{1} Int a b) " +
			"(Int.lt a b)) (fun (h0 : Eq.{1} Int (Int.sub b a) (Int.ofNat 0)) => Or.inl (Eq.{1} Int a b) (Int.lt a b) (Eq.symm.{1} Int b a " +
			"(Int.eq_of_sub_eq_zero b a h0))) (fun (h1 : (Int.NonNeg (Int.sub (Int.sub b a) (Int.ofNat 1)))) => Or.inr (Eq.{1} Int a b) (Int.lt a b) " +
			"(Eq.rec.{0, 1} Int (Int.sub (Int.sub b a) (Int.ofNat 1)) (fun (w : Int) (_ : Eq.{1} Int (Int.sub (Int.sub b a) (Int.ofNat 1)) w) => " +
			"(Int.NonNeg w)) h1 (Int.sub b (Int.add a (Int.ofNat 1))) (Eq.symm.{1} Int (Int.sub b (Int.add a (Int.ofNat 1))) (Int.sub (Int.sub b a) " +
			"(Int.ofNat 1)) (Int.sub_add_eq_sub_sub a b (Int.ofNat 1))))) (Int.NonNeg_cases (Int.sub b a) h)"},
	// `a <= b` and `not (b <= a)` give `a < b`: `eq_or_lt_of_le` leaves `a = b` (then `b <= a`, absurd) or `a < b`. With `le_of_lt` and `not_le_of_lt` that is
	// `a < b` iff `a <= b` and `not (b <= a)`.
	valDecl{"thm", "Int.lt_of_le_of_not_le", nil,
		"(a : Int) → (b : Int) → (Int.le a b) → Not (Int.le b a) → (Int.lt a b)",
		"fun (a : Int) (b : Int) (h : (Int.le a b)) (h2 : Not (Int.le b a)) => Or.rec (Eq.{1} Int a b) (Int.lt a b) (fun (_ : Or (Eq.{1} Int a b) " +
			"(Int.lt a b)) => (Int.lt a b)) (fun (e : Eq.{1} Int a b) => False.elim.{0} (Int.lt a b) (h2 (Int.le_of_eq b a (Eq.symm.{1} Int a b e)))) " +
			"(fun (l : (Int.lt a b)) => l) (Int.eq_or_lt_of_le a b h)"},
	valDecl{"thm", "Int.lt_iff_le_not_le", nil,
		"(a : Int) → (b : Int) → Iff (Int.lt a b) (And (Int.le a b) (Not (Int.le b a)))",
		"fun (a : Int) (b : Int) => Iff.intro (Int.lt a b) (And (Int.le a b) (Not (Int.le b a))) (fun (h : (Int.lt a b)) => And.intro (Int.le a b) " +
			"(Not (Int.le b a)) (Int.le_of_lt a b h) (Int.not_le_of_lt a b h)) (fun (h : (And (Int.le a b) (Not (Int.le b a)))) => Int.lt_of_le_of_not_le " +
			"a b (And.left (Int.le a b) (Not (Int.le b a)) h) (And.right (Int.le a b) (Not (Int.le b a)) h))"},
	// `0 <= a % b` for `b != 0`: `ofNat k` is `NonNeg.mk`; `negSucc k` is `subNatNat |b| (succ (k % |b|))`, `NonNeg` by `subNatNat_nonneg` and `Nat.mod_lt`
	// (`|b| > 0` from `natAbs_eq_zero`), moved from `x` to `x - 0` by `sub_zero`.
	valDecl{"thm", "Int.emod_nonneg", nil,
		"(a : Int) → (b : Int) → Not (Eq.{1} Int b (Int.ofNat 0)) → (Int.le (Int.ofNat 0) (Int.emod a b))",
		"fun (a : Int) (b : Int) (hb : Not (Eq.{1} Int b (Int.ofNat 0))) => Int.rec.{0} (fun (a : Int) => (Int.le (Int.ofNat 0) (Int.emod a b))) (fun " +
			"(k : Nat) => Int.NonNeg.mk (Nat.mod k (Int.natAbs b))) (fun (k : Nat) => Eq.rec.{0, 1} Int (Int.subNatNat (Int.natAbs b) (Nat.succ (Nat.mod " +
			"k (Int.natAbs b)))) (fun (w : Int) (_ : Eq.{1} Int (Int.subNatNat (Int.natAbs b) (Nat.succ (Nat.mod k (Int.natAbs b)))) w) => (Int.NonNeg " +
			"w)) (Iff.mpr ((Int.NonNeg (Int.subNatNat (Int.natAbs b) (Nat.succ (Nat.mod k (Int.natAbs b)))))) (Nat.le (Nat.succ (Nat.mod k (Int.natAbs " +
			"b))) (Int.natAbs b)) (Int.subNatNat_nonneg (Int.natAbs b) (Nat.succ (Nat.mod k (Int.natAbs b)))) (Nat.mod_lt k (Int.natAbs b) " +
			"(Nat.pos_of_ne_zero (Int.natAbs b) (fun (h0 : Eq.{1} Nat (Int.natAbs b) 0) => hb (Iff.mp (Eq.{1} Nat (Int.natAbs b) 0) (Eq.{1} Int b " +
			"(Int.ofNat 0)) (Int.natAbs_eq_zero b) h0))))) (Int.sub (Int.subNatNat (Int.natAbs b) (Nat.succ (Nat.mod k (Int.natAbs b)))) (Int.ofNat 0)) " +
			"(Eq.symm.{1} Int (Int.sub (Int.subNatNat (Int.natAbs b) (Nat.succ (Nat.mod k (Int.natAbs b)))) (Int.ofNat 0)) (Int.subNatNat (Int.natAbs b) " +
			"(Nat.succ (Nat.mod k (Int.natAbs b)))) (Int.sub_zero (Int.subNatNat (Int.natAbs b) (Nat.succ (Nat.mod k (Int.natAbs b))))))) a"},
	// `0 <= a` is `NonNeg a` up to `sub_zero`. `0 <= a / b` for `0 <= a` and `0 <= b`: only `ofNat / ofNat` is left, the rest contradict `not_nonneg_negSucc`.
	valDecl{"thm", "Int.nonneg_of_le_zero", nil,
		"(a : Int) → (Int.le (Int.ofNat 0) a) → (Int.NonNeg a)",
		"fun (a : Int) (h : (Int.le (Int.ofNat 0) a)) => Eq.rec.{0, 1} Int (Int.sub a (Int.ofNat 0)) (fun (w : Int) (_ : Eq.{1} Int (Int.sub a " +
			"(Int.ofNat 0)) w) => (Int.NonNeg w)) h a (Int.sub_zero a)"},
	valDecl{"thm", "Int.le_zero_of_nonneg", nil,
		"(a : Int) → (Int.NonNeg a) → (Int.le (Int.ofNat 0) a)",
		"fun (a : Int) (h : (Int.NonNeg a)) => Eq.rec.{0, 1} Int a (fun (w : Int) (_ : Eq.{1} Int a w) => (Int.NonNeg w)) h (Int.sub a (Int.ofNat 0)) " +
			"(Eq.symm.{1} Int (Int.sub a (Int.ofNat 0)) a (Int.sub_zero a))"},
	valDecl{"thm", "Int.ediv_nonneg", nil,
		"(a : Int) → (b : Int) → (Int.le (Int.ofNat 0) a) → (Int.le (Int.ofNat 0) b) → (Int.le (Int.ofNat 0) (Int.ediv a b))",
		"fun (a : Int) => Int.rec.{0} (fun (a : Int) => (b : Int) → (Int.le (Int.ofNat 0) a) → (Int.le (Int.ofNat 0) b) → (Int.le (Int.ofNat 0) " +
			"(Int.ediv a b))) (fun (k : Nat) (b0 : Int) => Int.rec.{0} (fun (b : Int) => (Int.le (Int.ofNat 0) (Int.ofNat k)) → (Int.le (Int.ofNat 0) b) " +
			"→ (Int.le (Int.ofNat 0) (Int.ediv (Int.ofNat k) b))) (fun (n : Nat) (_ : (Int.le (Int.ofNat 0) (Int.ofNat k))) (_ : (Int.le (Int.ofNat 0) " +
			"(Int.ofNat n))) => Int.le_zero_of_nonneg (Int.ofNat (Nat.div k n)) (Int.NonNeg.mk (Nat.div k n))) (fun (n : Nat) (_ : (Int.le (Int.ofNat 0) " +
			"(Int.ofNat k))) (hb : (Int.le (Int.ofNat 0) (Int.negSucc n))) => False.elim.{0} (Int.le (Int.ofNat 0) (Int.ediv (Int.ofNat k) (Int.negSucc " +
			"n))) (Int.not_nonneg_negSucc n (Int.nonneg_of_le_zero (Int.negSucc n) hb))) b0) (fun (k : Nat) (b : Int) (ha : (Int.le (Int.ofNat 0) " +
			"(Int.negSucc k))) (_ : (Int.le (Int.ofNat 0) b)) => False.elim.{0} (Int.le (Int.ofNat 0) (Int.ediv (Int.negSucc k) b)) " +
			"(Int.not_nonneg_negSucc k (Int.nonneg_of_le_zero (Int.negSucc k) ha))) a"},
	// Division identity on naturals lifted to `Int`: `k * (m / k) + m % k = m` for `ofNat`. The `Int` operations on `ofNat` reduce to `ofNat` of the `Nat` ones,
	// so it is `congrArg ofNat` of `Nat.mod_add_div` after `add_comm`.
	valDecl{"thm", "Int.ediv_add_emod_ofNat", nil,
		"(m : Nat) → (k : Nat) → Eq.{1} Int (Int.add (Int.mul (Int.ofNat k) (Int.ediv (Int.ofNat m) (Int.ofNat k))) (Int.emod (Int.ofNat m) " +
			"(Int.ofNat k))) (Int.ofNat m)",
		"fun (m : Nat) (k : Nat) => congrArg.{1, 1} Nat Int Int.ofNat (Nat.add (Nat.mul k (Nat.div m k)) (Nat.mod m k)) m (Eq.trans.{1} Nat (Nat.add " +
			"(Nat.mul k (Nat.div m k)) (Nat.mod m k)) (Nat.add (Nat.mod m k) (Nat.mul k (Nat.div m k))) m (Nat.add_comm (Nat.mul k (Nat.div m k)) " +
			"(Nat.mod m k)) (Nat.mod_add_div m k))"},
	// Upper bound of the remainder on `ofNat`: `a % b < |b|` for `b != 0`. `(a % |b|) + 1 <= |b|` unfolds to `NonNeg (subNatNat |b| (succ (a % |b|)))`,
	// which is `subNatNat_nonneg` and `Nat.mod_lt`.
	valDecl{"thm", "Int.emod_lt_ofNat", nil,
		"(a : Nat) → (b : Int) → Not (Eq.{1} Int b (Int.ofNat 0)) → (Int.lt (Int.emod (Int.ofNat a) b) (Int.ofNat (Int.natAbs b)))",
		"fun (a : Nat) (b : Int) (hb : Not (Eq.{1} Int b (Int.ofNat 0))) => Iff.mpr ((Int.NonNeg (Int.subNatNat (Int.natAbs b) (Nat.succ (Nat.mod a " +
			"(Int.natAbs b)))))) (Nat.le (Nat.succ (Nat.mod a (Int.natAbs b))) (Int.natAbs b)) (Int.subNatNat_nonneg (Int.natAbs b) (Nat.succ (Nat.mod a " +
			"(Int.natAbs b)))) (Nat.mod_lt a (Int.natAbs b) (Nat.pos_of_ne_zero (Int.natAbs b) (fun (h0 : Eq.{1} Nat (Int.natAbs b) 0) => hb (Iff.mp " +
			"(Eq.{1} Nat (Int.natAbs b) 0) (Eq.{1} Int b (Int.ofNat 0)) (Int.natAbs_eq_zero b) h0))))"},
	// `a - (a - b) = b`. Then `subNatNat n (succ r) < n` for all `n`, `r`: it is `n - s` (`ofNat_sub_ofNat`), `n - ((n - s) + 1) = (n - (n - s)) - 1 = s - 1 = r`,
	// `NonNeg` by `mk`. With `emod_lt_ofNat` for the `ofNat` case that is `a % b < |b|` for `b != 0`.
	valDecl{"thm", "Int.sub_sub_cancel", nil,
		"(a : Int) → (b : Int) → Eq.{1} Int (Int.sub a (Int.sub a b)) b",
		"fun (a : Int) (b : Int) => Eq.trans.{1} Int (Int.sub a (Int.sub a b)) (Int.add (Int.ofNat 0) b) b (Eq.trans.{1} Int (Int.sub a (Int.sub a " +
			"b)) (Int.add (Int.add a (Int.neg a)) b) (Int.add (Int.ofNat 0) b) (Eq.trans.{1} Int (Int.sub a (Int.sub a b)) (Int.add a (Int.add (Int.neg " +
			"a) b)) (Int.add (Int.add a (Int.neg a)) b) (Eq.trans.{1} Int (Int.sub a (Int.sub a b)) (Int.add a (Int.add (Int.neg a) (Int.neg (Int.neg " +
			"b)))) (Int.add a (Int.add (Int.neg a) b)) (congrArg.{1, 1} Int Int (Int.add a) (Int.neg (Int.add a (Int.neg b))) (Int.add (Int.neg a) " +
			"(Int.neg (Int.neg b))) (Int.neg_add a (Int.neg b))) (congrArg.{1, 1} Int Int (fun (x : Int) => Int.add a (Int.add (Int.neg a) x)) (Int.neg " +
			"(Int.neg b)) b (Int.neg_neg b))) (Eq.symm.{1} Int (Int.add (Int.add a (Int.neg a)) b) (Int.add a (Int.add (Int.neg a) b)) (Int.add_assoc a " +
			"(Int.neg a) b))) (congrArg.{1, 1} Int Int (fun (x : Int) => Int.add x b) (Int.add a (Int.neg a)) (Int.ofNat 0) (Int.add_neg_cancel a))) " +
			"(Int.zero_add b)"},
	valDecl{"thm", "Int.subNatNat_succ_lt", nil,
		"(n : Nat) → (r : Nat) → (Int.lt (Int.subNatNat n (Nat.succ r)) (Int.ofNat n))",
		"fun (n : Nat) (r : Nat) => Eq.rec.{0, 1} Int (Int.sub (Int.ofNat (Nat.succ r)) (Int.ofNat 1)) (fun (w : Int) (_ : Eq.{1} Int (Int.sub " +
			"(Int.ofNat (Nat.succ r)) (Int.ofNat 1)) w) => (Int.NonNeg w)) (Int.NonNeg.mk r) (Int.sub (Int.ofNat n) (Int.add (Int.subNatNat n (Nat.succ " +
			"r)) (Int.ofNat 1))) (Eq.symm.{1} Int (Int.sub (Int.ofNat n) (Int.add (Int.subNatNat n (Nat.succ r)) (Int.ofNat 1))) (Int.sub (Int.ofNat " +
			"(Nat.succ r)) (Int.ofNat 1)) (Eq.trans.{1} Int (Int.sub (Int.ofNat n) (Int.add (Int.subNatNat n (Nat.succ r)) (Int.ofNat 1))) (Int.sub " +
			"(Int.sub (Int.ofNat n) (Int.subNatNat n (Nat.succ r))) (Int.ofNat 1)) (Int.sub (Int.ofNat (Nat.succ r)) (Int.ofNat 1)) " +
			"(Int.sub_add_eq_sub_sub (Int.subNatNat n (Nat.succ r)) (Int.ofNat n) (Int.ofNat 1)) (congrArg.{1, 1} Int Int (fun (z : Int) => Int.sub z " +
			"(Int.ofNat 1)) (Int.sub (Int.ofNat n) (Int.subNatNat n (Nat.succ r))) (Int.ofNat (Nat.succ r)) (Eq.trans.{1} Int (Int.sub (Int.ofNat n) " +
			"(Int.subNatNat n (Nat.succ r))) (Int.sub (Int.ofNat n) (Int.sub (Int.ofNat n) (Int.ofNat (Nat.succ r)))) (Int.ofNat (Nat.succ r)) " +
			"(congrArg.{1, 1} Int Int (Int.sub (Int.ofNat n)) (Int.subNatNat n (Nat.succ r)) (Int.sub (Int.ofNat n) (Int.ofNat (Nat.succ r))) " +
			"(Eq.symm.{1} Int (Int.sub (Int.ofNat n) (Int.ofNat (Nat.succ r))) (Int.subNatNat n (Nat.succ r)) (Int.ofNat_sub_ofNat n (Nat.succ r)))) " +
			"(Int.sub_sub_cancel (Int.ofNat n) (Int.ofNat (Nat.succ r)))))))"},
	valDecl{"thm", "Int.emod_lt", nil,
		"(a : Int) → (b : Int) → Not (Eq.{1} Int b (Int.ofNat 0)) → (Int.lt (Int.emod a b) (Int.ofNat (Int.natAbs b)))",
		"fun (a : Int) (b : Int) (hb : Not (Eq.{1} Int b (Int.ofNat 0))) => Int.rec.{0} (fun (a : Int) => (Int.lt (Int.emod a b) (Int.ofNat " +
			"(Int.natAbs b)))) (fun (k : Nat) => Int.emod_lt_ofNat k b hb) (fun (k : Nat) => Int.subNatNat_succ_lt (Int.natAbs b) (Nat.mod k (Int.natAbs " +
			"b))) a"},
	// With `negSucc_mul_negOfNat`, the division identity for `ofNat m` and divisor `negSucc n`:
	// `negSucc n * (m / negSucc n) + m % negSucc n = m`, from `Nat.mod_add_div` as for the `ofNat` divisor.
	valDecl{"thm", "Int.ediv_add_emod_ofNat_negSucc", nil,
		"(m : Nat) → (n : Nat) → Eq.{1} Int (Int.add (Int.mul (Int.negSucc n) (Int.ediv (Int.ofNat m) (Int.negSucc n))) (Int.emod (Int.ofNat m) " +
			"(Int.negSucc n))) (Int.ofNat m)",
		"fun (m : Nat) (n : Nat) => Eq.trans.{1} Int (Int.add (Int.mul (Int.negSucc n) (Int.ediv (Int.ofNat m) (Int.negSucc n))) (Int.emod (Int.ofNat " +
			"m) (Int.negSucc n))) (Int.add (Int.ofNat (Nat.mul (Nat.succ n) (Nat.div m (Nat.succ n)))) (Int.ofNat (Nat.mod m (Nat.succ n)))) (Int.ofNat " +
			"m) (congrArg.{1, 1} Int Int (fun (x : Int) => Int.add x (Int.ofNat (Nat.mod m (Nat.succ n)))) (Int.mul (Int.negSucc n) (Int.negOfNat " +
			"(Nat.div m (Nat.succ n)))) (Int.ofNat (Nat.mul (Nat.succ n) (Nat.div m (Nat.succ n)))) (Int.negSucc_mul_negOfNat n (Nat.div m (Nat.succ " +
			"n)))) (congrArg.{1, 1} Nat Int Int.ofNat (Nat.add (Nat.mul (Nat.succ n) (Nat.div m (Nat.succ n))) (Nat.mod m (Nat.succ n))) m (Eq.trans.{1} " +
			"Nat (Nat.add (Nat.mul (Nat.succ n) (Nat.div m (Nat.succ n))) (Nat.mod m (Nat.succ n))) (Nat.add (Nat.mod m (Nat.succ n)) (Nat.mul (Nat.succ " +
			"n) (Nat.div m (Nat.succ n)))) m (Nat.add_comm (Nat.mul (Nat.succ n) (Nat.div m (Nat.succ n))) (Nat.mod m (Nat.succ n))) (Nat.mod_add_div m " +
			"(Nat.succ n))))"},
	// The division identity for divisor `0`, every `a`: `0 * (a / 0) + a % 0 = a`, from `emod_zero`, `zero_mul` and `zero_add`.
	valDecl{"thm", "Int.ediv_add_emod_zero", nil,
		"(a : Int) → Eq.{1} Int (Int.add (Int.mul (Int.ofNat 0) (Int.ediv a (Int.ofNat 0))) (Int.emod a (Int.ofNat 0))) a",
		"fun (a : Int) => Eq.trans.{1} Int (Int.add (Int.mul (Int.ofNat 0) (Int.ediv a (Int.ofNat 0))) (Int.emod a (Int.ofNat 0))) (Int.add " +
			"(Int.ofNat 0) a) a (Eq.trans.{1} Int (Int.add (Int.mul (Int.ofNat 0) (Int.ediv a (Int.ofNat 0))) (Int.emod a (Int.ofNat 0))) (Int.add " +
			"(Int.mul (Int.ofNat 0) (Int.ediv a (Int.ofNat 0))) a) (Int.add (Int.ofNat 0) a) (congrArg.{1, 1} Int Int (Int.add (Int.mul (Int.ofNat 0) " +
			"(Int.ediv a (Int.ofNat 0)))) (Int.emod a (Int.ofNat 0)) a (Int.emod_zero a)) (congrArg.{1, 1} Int Int (fun (x : Int) => Int.add x a) " +
			"(Int.mul (Int.ofNat 0) (Int.ediv a (Int.ofNat 0))) (Int.ofNat 0) (Int.zero_mul (Int.ediv a (Int.ofNat 0))))) (Int.zero_add a)"},
	// `negOfNat (t + s) + subNatNat s (succ r) = negSucc (t + r)`: write both as `-(T + S) + (S - (R + 1))`, regroup, cancel `-S + S`, and `-T + -(R + 1) = -(T + (R + 1))`.
	// With `t = succ j * (a / succ j)` and `r = a % succ j` (`a = t + r` by `mod_add_div`) that is the division identity for `negSucc a` and divisor `succ j`.
	valDecl{"thm", "Int.negOfNat_add_subNatNat", nil,
		"(t : Nat) → (s : Nat) → (r : Nat) → Eq.{1} Int (Int.add (Int.negOfNat (Nat.add t s)) (Int.subNatNat s (Nat.succ r))) (Int.negSucc (Nat.add t " +
			"r))",
		"fun (t : Nat) (s : Nat) (r : Nat) => Eq.trans.{1} Int (Int.add (Int.negOfNat (Nat.add t s)) (Int.subNatNat s (Nat.succ r))) (Int.neg " +
			"(Int.add (Int.ofNat t) (Int.ofNat (Nat.succ r)))) (Int.neg (Int.ofNat (Nat.add t (Nat.succ r)))) (Eq.trans.{1} Int (Int.add (Int.negOfNat " +
			"(Nat.add t s)) (Int.subNatNat s (Nat.succ r))) (Int.add (Int.neg (Int.ofNat t)) (Int.neg (Int.ofNat (Nat.succ r)))) (Int.neg (Int.add " +
			"(Int.ofNat t) (Int.ofNat (Nat.succ r)))) (Eq.trans.{1} Int (Int.add (Int.negOfNat (Nat.add t s)) (Int.subNatNat s (Nat.succ r))) (Int.add " +
			"(Int.neg (Int.ofNat t)) (Int.add (Int.neg (Int.ofNat s)) (Int.sub (Int.ofNat s) (Int.ofNat (Nat.succ r))))) (Int.add (Int.neg (Int.ofNat t)) " +
			"(Int.neg (Int.ofNat (Nat.succ r)))) (Eq.trans.{1} Int (Int.add (Int.negOfNat (Nat.add t s)) (Int.subNatNat s (Nat.succ r))) (Int.add " +
			"(Int.add (Int.neg (Int.ofNat t)) (Int.neg (Int.ofNat s))) (Int.sub (Int.ofNat s) (Int.ofNat (Nat.succ r)))) (Int.add (Int.neg (Int.ofNat t)) " +
			"(Int.add (Int.neg (Int.ofNat s)) (Int.sub (Int.ofNat s) (Int.ofNat (Nat.succ r))))) (Eq.trans.{1} Int (Int.add (Int.negOfNat (Nat.add t s)) " +
			"(Int.subNatNat s (Nat.succ r))) (Int.add (Int.neg (Int.add (Int.ofNat t) (Int.ofNat s))) (Int.sub (Int.ofNat s) (Int.ofNat (Nat.succ r)))) " +
			"(Int.add (Int.add (Int.neg (Int.ofNat t)) (Int.neg (Int.ofNat s))) (Int.sub (Int.ofNat s) (Int.ofNat (Nat.succ r)))) (Eq.trans.{1} Int " +
			"(Int.add (Int.negOfNat (Nat.add t s)) (Int.subNatNat s (Nat.succ r))) (Int.add (Int.negOfNat (Nat.add t s)) (Int.sub (Int.ofNat s) " +
			"(Int.ofNat (Nat.succ r)))) (Int.add (Int.neg (Int.add (Int.ofNat t) (Int.ofNat s))) (Int.sub (Int.ofNat s) (Int.ofNat (Nat.succ r)))) " +
			"(congrArg.{1, 1} Int Int (Int.add (Int.negOfNat (Nat.add t s))) (Int.subNatNat s (Nat.succ r)) (Int.sub (Int.ofNat s) (Int.ofNat (Nat.succ " +
			"r))) (Eq.symm.{1} Int (Int.sub (Int.ofNat s) (Int.ofNat (Nat.succ r))) (Int.subNatNat s (Nat.succ r)) (Int.ofNat_sub_ofNat s (Nat.succ r)))) " +
			"(congrArg.{1, 1} Int Int (fun (x : Int) => Int.add x (Int.sub (Int.ofNat s) (Int.ofNat (Nat.succ r)))) (Int.negOfNat (Nat.add t s)) (Int.neg " +
			"(Int.add (Int.ofNat t) (Int.ofNat s))) (congrArg.{1, 1} Int Int Int.neg (Int.ofNat (Nat.add t s)) (Int.add (Int.ofNat t) (Int.ofNat s)) " +
			"(Int.ofNat_add t s)))) (congrArg.{1, 1} Int Int (fun (x : Int) => Int.add x (Int.sub (Int.ofNat s) (Int.ofNat (Nat.succ r)))) (Int.neg " +
			"(Int.add (Int.ofNat t) (Int.ofNat s))) (Int.add (Int.neg (Int.ofNat t)) (Int.neg (Int.ofNat s))) (Int.neg_add (Int.ofNat t) (Int.ofNat s)))) " +
			"(Int.add_assoc (Int.neg (Int.ofNat t)) (Int.neg (Int.ofNat s)) (Int.sub (Int.ofNat s) (Int.ofNat (Nat.succ r))))) (congrArg.{1, 1} Int Int " +
			"(Int.add (Int.neg (Int.ofNat t))) (Int.add (Int.neg (Int.ofNat s)) (Int.sub (Int.ofNat s) (Int.ofNat (Nat.succ r)))) (Int.neg (Int.ofNat " +
			"(Nat.succ r))) (Eq.trans.{1} Int (Int.add (Int.neg (Int.ofNat s)) (Int.sub (Int.ofNat s) (Int.ofNat (Nat.succ r)))) (Int.add (Int.ofNat 0) " +
			"(Int.neg (Int.ofNat (Nat.succ r)))) (Int.neg (Int.ofNat (Nat.succ r))) (Eq.trans.{1} Int (Int.add (Int.neg (Int.ofNat s)) (Int.sub " +
			"(Int.ofNat s) (Int.ofNat (Nat.succ r)))) (Int.add (Int.add (Int.neg (Int.ofNat s)) (Int.ofNat s)) (Int.neg (Int.ofNat (Nat.succ r)))) " +
			"(Int.add (Int.ofNat 0) (Int.neg (Int.ofNat (Nat.succ r)))) (Eq.symm.{1} Int (Int.add (Int.add (Int.neg (Int.ofNat s)) (Int.ofNat s)) " +
			"(Int.neg (Int.ofNat (Nat.succ r)))) (Int.add (Int.neg (Int.ofNat s)) (Int.add (Int.ofNat s) (Int.neg (Int.ofNat (Nat.succ r))))) " +
			"(Int.add_assoc (Int.neg (Int.ofNat s)) (Int.ofNat s) (Int.neg (Int.ofNat (Nat.succ r))))) (congrArg.{1, 1} Int Int (fun (x : Int) => Int.add " +
			"x (Int.neg (Int.ofNat (Nat.succ r)))) (Int.add (Int.neg (Int.ofNat s)) (Int.ofNat s)) (Int.ofNat 0) (Int.neg_add_cancel (Int.ofNat s)))) " +
			"(Int.zero_add (Int.neg (Int.ofNat (Nat.succ r))))))) (Eq.symm.{1} Int (Int.neg (Int.add (Int.ofNat t) (Int.ofNat (Nat.succ r)))) (Int.add " +
			"(Int.neg (Int.ofNat t)) (Int.neg (Int.ofNat (Nat.succ r)))) (Int.neg_add (Int.ofNat t) (Int.ofNat (Nat.succ r))))) (congrArg.{1, 1} Int Int " +
			"Int.neg (Int.add (Int.ofNat t) (Int.ofNat (Nat.succ r))) (Int.ofNat (Nat.add t (Nat.succ r))) (Eq.symm.{1} Int (Int.ofNat (Nat.add t " +
			"(Nat.succ r))) (Int.add (Int.ofNat t) (Int.ofNat (Nat.succ r))) (Int.ofNat_add t (Nat.succ r))))"},
	valDecl{"thm", "Int.ediv_add_emod_negSucc_ofNat_succ", nil,
		"(a : Nat) → (j : Nat) → Eq.{1} Int (Int.add (Int.mul (Int.ofNat (Nat.succ j)) (Int.ediv (Int.negSucc a) (Int.ofNat (Nat.succ j)))) (Int.emod " +
			"(Int.negSucc a) (Int.ofNat (Nat.succ j)))) (Int.negSucc a)",
		"fun (a : Nat) (j : Nat) => Eq.trans.{1} Int (Int.add (Int.mul (Int.ofNat (Nat.succ j)) (Int.ediv (Int.negSucc a) (Int.ofNat (Nat.succ j)))) " +
			"(Int.emod (Int.negSucc a) (Int.ofNat (Nat.succ j)))) (Int.negSucc (Nat.add (Nat.mul (Nat.succ j) (Nat.div a (Nat.succ j))) (Nat.mod a " +
			"(Nat.succ j)))) (Int.negSucc a) (Int.negOfNat_add_subNatNat (Nat.mul (Nat.succ j) (Nat.div a (Nat.succ j))) (Nat.succ j) (Nat.mod a " +
			"(Nat.succ j))) (congrArg.{1, 1} Nat Int Int.negSucc (Nat.add (Nat.mul (Nat.succ j) (Nat.div a (Nat.succ j))) (Nat.mod a (Nat.succ j))) a " +
			"(Eq.trans.{1} Nat (Nat.add (Nat.mul (Nat.succ j) (Nat.div a (Nat.succ j))) (Nat.mod a (Nat.succ j))) (Nat.add (Nat.mod a (Nat.succ j)) " +
			"(Nat.mul (Nat.succ j) (Nat.div a (Nat.succ j)))) a (Nat.add_comm (Nat.mul (Nat.succ j) (Nat.div a (Nat.succ j))) (Nat.mod a (Nat.succ j))) " +
			"(Nat.mod_add_div a (Nat.succ j))))"},
	// `negSucc a` with divisor `negSucc n` is the same computation as divisor `succ j` (`negSucc n * ofNat (succ q)` is `negOfNat (succ n * succ q)`).
	// The four sign cases and divisor `0` then give the division identity for every `Int`: `b * (a / b) + a % b = a`.
	valDecl{"thm", "Int.ediv_add_emod_negSucc_negSucc", nil,
		"(a : Nat) → (n : Nat) → Eq.{1} Int (Int.add (Int.mul (Int.negSucc n) (Int.ediv (Int.negSucc a) (Int.negSucc n))) (Int.emod (Int.negSucc a) " +
			"(Int.negSucc n))) (Int.negSucc a)",
		"fun (a : Nat) (n : Nat) => Eq.trans.{1} Int (Int.add (Int.mul (Int.negSucc n) (Int.ediv (Int.negSucc a) (Int.negSucc n))) (Int.emod " +
			"(Int.negSucc a) (Int.negSucc n))) (Int.negSucc (Nat.add (Nat.mul (Nat.succ n) (Nat.div a (Nat.succ n))) (Nat.mod a (Nat.succ n)))) " +
			"(Int.negSucc a) (Int.negOfNat_add_subNatNat (Nat.mul (Nat.succ n) (Nat.div a (Nat.succ n))) (Nat.succ n) (Nat.mod a (Nat.succ n))) " +
			"(congrArg.{1, 1} Nat Int Int.negSucc (Nat.add (Nat.mul (Nat.succ n) (Nat.div a (Nat.succ n))) (Nat.mod a (Nat.succ n))) a (Eq.trans.{1} Nat " +
			"(Nat.add (Nat.mul (Nat.succ n) (Nat.div a (Nat.succ n))) (Nat.mod a (Nat.succ n))) (Nat.add (Nat.mod a (Nat.succ n)) (Nat.mul (Nat.succ n) " +
			"(Nat.div a (Nat.succ n)))) a (Nat.add_comm (Nat.mul (Nat.succ n) (Nat.div a (Nat.succ n))) (Nat.mod a (Nat.succ n))) (Nat.mod_add_div a " +
			"(Nat.succ n))))"},
	valDecl{"thm", "Int.ediv_add_emod", nil,
		"(a : Int) → (b : Int) → Eq.{1} Int (Int.add (Int.mul b (Int.ediv a b)) (Int.emod a b)) a",
		"fun (a : Int) (b : Int) => Int.rec.{0} (fun (a : Int) => (b : Int) → Eq.{1} Int (Int.add (Int.mul b (Int.ediv a b)) (Int.emod a b)) a) (fun " +
			"(m : Nat) (b : Int) => Int.rec.{0} (fun (b : Int) => Eq.{1} Int (Int.add (Int.mul b (Int.ediv (Int.ofNat m) b)) (Int.emod (Int.ofNat m) b)) " +
			"(Int.ofNat m)) (fun (k : Nat) => Int.ediv_add_emod_ofNat m k) (fun (n : Nat) => Int.ediv_add_emod_ofNat_negSucc m n) b) (fun (a : Nat) (b : " +
			"Int) => Int.rec.{0} (fun (b : Int) => Eq.{1} Int (Int.add (Int.mul b (Int.ediv (Int.negSucc a) b)) (Int.emod (Int.negSucc a) b)) " +
			"(Int.negSucc a)) (fun (k : Nat) => Nat.rec.{0} (fun (k : Nat) => Eq.{1} Int (Int.add (Int.mul (Int.ofNat k) (Int.ediv (Int.negSucc a) " +
			"(Int.ofNat k))) (Int.emod (Int.negSucc a) (Int.ofNat k))) (Int.negSucc a)) (Int.ediv_add_emod_zero (Int.negSucc a)) (fun (j : Nat) (_ : " +
			"Eq.{1} Int (Int.add (Int.mul (Int.ofNat j) (Int.ediv (Int.negSucc a) (Int.ofNat j))) (Int.emod (Int.negSucc a) (Int.ofNat j))) (Int.negSucc " +
			"a)) => Int.ediv_add_emod_negSucc_ofNat_succ a j) k) (fun (n : Nat) => Int.ediv_add_emod_negSucc_negSucc a n) b) a b"},
	valDecl{"def", "Int.min", nil, "Int → Int → Int",
		"fun (a : Int) (b : Int) => Bool.rec.{1} (fun (_ : Bool) => Int) b a " +
			"(Decidable.decide (Int.le a b) (Int.decLe a b))"},
	valDecl{"def", "Int.max", nil, "Int → Int → Int",
		"fun (a : Int) (b : Int) => Bool.rec.{1} (fun (_ : Bool) => Int) a b " +
			"(Decidable.decide (Int.le a b) (Int.decLe a b))"},
	// `decide p = true` if `p`, `= false` if `not p`; so `min a b` and `max a b` (a `Bool.rec` on `decide (a <= b)`) are `a` or `b` as `a <= b` holds or not.
	// Named `decide_eq_true'`/`decide_eq_false'` to leave Lean's own names free.
	valDecl{"thm", "decide_eq_true'", nil,
		"(p : Prop) → (inst : Decidable p) → p → Eq.{1} Bool (Decidable.decide p inst) Bool.true",
		"fun (p : Prop) (inst : Decidable p) => Decidable.rec.{0} p (fun (d : Decidable p) => p → Eq.{1} Bool (Decidable.decide p d) Bool.true) (fun " +
			"(hn : Not p) (hp : p) => False.elim.{0} (Eq.{1} Bool Bool.false Bool.true) (hn hp)) (fun (hp : p) (_ : p) => Eq.refl.{1} Bool Bool.true) " +
			"inst"},
	valDecl{"thm", "decide_eq_false'", nil,
		"(p : Prop) → (inst : Decidable p) → Not p → Eq.{1} Bool (Decidable.decide p inst) Bool.false",
		"fun (p : Prop) (inst : Decidable p) => Decidable.rec.{0} p (fun (d : Decidable p) => Not p → Eq.{1} Bool (Decidable.decide p d) Bool.false) " +
			"(fun (hn : Not p) (_ : Not p) => Eq.refl.{1} Bool Bool.false) (fun (hp : p) (hn : Not p) => False.elim.{0} (Eq.{1} Bool Bool.true " +
			"Bool.false) (hn hp)) inst"},
	valDecl{"thm", "Int.min_eq_left", nil,
		"(a : Int) → (b : Int) → (Int.le a b) → Eq.{1} Int (Int.min a b) a",
		"fun (a : Int) (b : Int) (h : (Int.le a b)) => Eq.rec.{0, 1} Bool Bool.true (fun (y : Bool) (_ : Eq.{1} Bool Bool.true y) => Eq.{1} Int " +
			"(Bool.rec.{1} (fun (_ : Bool) => Int) b a y) a) (Eq.refl.{1} Int a) (Decidable.decide (Int.le a b) (Int.decLe a b)) (Eq.symm.{1} Bool " +
			"(Decidable.decide (Int.le a b) (Int.decLe a b)) Bool.true (decide_eq_true' (Int.le a b) (Int.decLe a b) h))"},
	valDecl{"thm", "Int.min_eq_right_of_not_le", nil,
		"(a : Int) → (b : Int) → Not (Int.le a b) → Eq.{1} Int (Int.min a b) b",
		"fun (a : Int) (b : Int) (h : Not (Int.le a b)) => Eq.rec.{0, 1} Bool Bool.false (fun (y : Bool) (_ : Eq.{1} Bool Bool.false y) => Eq.{1} Int " +
			"(Bool.rec.{1} (fun (_ : Bool) => Int) b a y) b) (Eq.refl.{1} Int b) (Decidable.decide (Int.le a b) (Int.decLe a b)) (Eq.symm.{1} Bool " +
			"(Decidable.decide (Int.le a b) (Int.decLe a b)) Bool.false (decide_eq_false' (Int.le a b) (Int.decLe a b) h))"},
	valDecl{"thm", "Int.max_eq_right", nil,
		"(a : Int) → (b : Int) → (Int.le a b) → Eq.{1} Int (Int.max a b) b",
		"fun (a : Int) (b : Int) (h : (Int.le a b)) => Eq.rec.{0, 1} Bool Bool.true (fun (y : Bool) (_ : Eq.{1} Bool Bool.true y) => Eq.{1} Int " +
			"(Bool.rec.{1} (fun (_ : Bool) => Int) a b y) b) (Eq.refl.{1} Int b) (Decidable.decide (Int.le a b) (Int.decLe a b)) (Eq.symm.{1} Bool " +
			"(Decidable.decide (Int.le a b) (Int.decLe a b)) Bool.true (decide_eq_true' (Int.le a b) (Int.decLe a b) h))"},
	valDecl{"thm", "Int.max_eq_left_of_not_le", nil,
		"(a : Int) → (b : Int) → Not (Int.le a b) → Eq.{1} Int (Int.max a b) a",
		"fun (a : Int) (b : Int) (h : Not (Int.le a b)) => Eq.rec.{0, 1} Bool Bool.false (fun (y : Bool) (_ : Eq.{1} Bool Bool.false y) => Eq.{1} Int " +
			"(Bool.rec.{1} (fun (_ : Bool) => Int) a b y) a) (Eq.refl.{1} Int a) (Decidable.decide (Int.le a b) (Int.decLe a b)) (Eq.symm.{1} Bool " +
			"(Decidable.decide (Int.le a b) (Int.decLe a b)) Bool.false (decide_eq_false' (Int.le a b) (Int.decLe a b) h))"},
	// `min a b <= a, b` and `a, b <= max a b`: split on `decide (a <= b)`; the equation moves the goal to `a` or `b`, closed by `le_refl`, the hypothesis,
	// or (when `not (a <= b)`) `b <= a` from `le_total`.
	valDecl{"thm", "Int.min_le_left", nil,
		"(a : Int) → (b : Int) → (Int.le (Int.min a b) a)",
		"fun (a : Int) (b : Int) => Decidable.rec.{0} (Int.le a b) (fun (_ : Decidable (Int.le a b)) => (Int.le (Int.min a b) a)) (fun (h : Not " +
			"(Int.le a b)) => Eq.rec.{0, 1} Int b (fun (w : Int) (_ : Eq.{1} Int b w) => (Int.le w a)) (Or.rec (Int.le a b) (Int.le b a) (fun (_ : Or " +
			"(Int.le a b) (Int.le b a)) => (Int.le b a)) (fun (h1 : (Int.le a b)) => False.elim.{0} (Int.le b a) (h h1)) (fun (h2 : (Int.le b a)) => h2) " +
			"(Int.le_total a b)) (Int.min a b) (Eq.symm.{1} Int (Int.min a b) b (Int.min_eq_right_of_not_le a b h))) (fun (h : (Int.le a b)) => " +
			"Eq.rec.{0, 1} Int a (fun (w : Int) (_ : Eq.{1} Int a w) => (Int.le w a)) (Int.le_refl a) (Int.min a b) (Eq.symm.{1} Int (Int.min a b) a " +
			"(Int.min_eq_left a b h))) (Int.decLe a b)"},
	valDecl{"thm", "Int.min_le_right", nil,
		"(a : Int) → (b : Int) → (Int.le (Int.min a b) b)",
		"fun (a : Int) (b : Int) => Decidable.rec.{0} (Int.le a b) (fun (_ : Decidable (Int.le a b)) => (Int.le (Int.min a b) b)) (fun (h : Not " +
			"(Int.le a b)) => Eq.rec.{0, 1} Int b (fun (w : Int) (_ : Eq.{1} Int b w) => (Int.le w b)) (Int.le_refl b) (Int.min a b) (Eq.symm.{1} Int " +
			"(Int.min a b) b (Int.min_eq_right_of_not_le a b h))) (fun (h : (Int.le a b)) => Eq.rec.{0, 1} Int a (fun (w : Int) (_ : Eq.{1} Int a w) => " +
			"(Int.le w b)) h (Int.min a b) (Eq.symm.{1} Int (Int.min a b) a (Int.min_eq_left a b h))) (Int.decLe a b)"},
	valDecl{"thm", "Int.le_max_left", nil,
		"(a : Int) → (b : Int) → (Int.le a (Int.max a b))",
		"fun (a : Int) (b : Int) => Decidable.rec.{0} (Int.le a b) (fun (_ : Decidable (Int.le a b)) => (Int.le a (Int.max a b))) (fun (h : Not " +
			"(Int.le a b)) => Eq.rec.{0, 1} Int a (fun (w : Int) (_ : Eq.{1} Int a w) => (Int.le a w)) (Int.le_refl a) (Int.max a b) (Eq.symm.{1} Int " +
			"(Int.max a b) a (Int.max_eq_left_of_not_le a b h))) (fun (h : (Int.le a b)) => Eq.rec.{0, 1} Int b (fun (w : Int) (_ : Eq.{1} Int b w) => " +
			"(Int.le a w)) h (Int.max a b) (Eq.symm.{1} Int (Int.max a b) b (Int.max_eq_right a b h))) (Int.decLe a b)"},
	valDecl{"thm", "Int.le_max_right", nil,
		"(a : Int) → (b : Int) → (Int.le b (Int.max a b))",
		"fun (a : Int) (b : Int) => Decidable.rec.{0} (Int.le a b) (fun (_ : Decidable (Int.le a b)) => (Int.le b (Int.max a b))) (fun (h : Not " +
			"(Int.le a b)) => Eq.rec.{0, 1} Int a (fun (w : Int) (_ : Eq.{1} Int a w) => (Int.le b w)) (Or.rec (Int.le a b) (Int.le b a) (fun (_ : Or " +
			"(Int.le a b) (Int.le b a)) => (Int.le b a)) (fun (h1 : (Int.le a b)) => False.elim.{0} (Int.le b a) (h h1)) (fun (h2 : (Int.le b a)) => h2) " +
			"(Int.le_total a b)) (Int.max a b) (Eq.symm.{1} Int (Int.max a b) a (Int.max_eq_left_of_not_le a b h))) (fun (h : (Int.le a b)) => Eq.rec.{0, " +
			"1} Int b (fun (w : Int) (_ : Eq.{1} Int b w) => (Int.le b w)) (Int.le_refl b) (Int.max a b) (Eq.symm.{1} Int (Int.max a b) b " +
			"(Int.max_eq_right a b h))) (Int.decLe a b)"},
	// `min` is the greatest lower bound and `max` the least upper bound (`le_min`, `max_le`: split on `decide (a <= b)` as for `min_le_left`).
	// So `min_comm` and `max_comm` are `le_antisymm` of two such bounds.
	valDecl{"thm", "Int.le_min", nil,
		"(a : Int) → (b : Int) → (c : Int) → (Int.le c a) → (Int.le c b) → (Int.le c (Int.min a b))",
		"fun (a : Int) (b : Int) (c : Int) (hca : (Int.le c a)) (hcb : (Int.le c b)) => Decidable.rec.{0} (Int.le a b) (fun (_ : Decidable (Int.le a " +
			"b)) => (Int.le c (Int.min a b))) (fun (h : Not (Int.le a b)) => Eq.rec.{0, 1} Int b (fun (w : Int) (_ : Eq.{1} Int b w) => (Int.le c w)) hcb " +
			"(Int.min a b) (Eq.symm.{1} Int (Int.min a b) b (Int.min_eq_right_of_not_le a b h))) (fun (h : (Int.le a b)) => Eq.rec.{0, 1} Int a (fun (w : " +
			"Int) (_ : Eq.{1} Int a w) => (Int.le c w)) hca (Int.min a b) (Eq.symm.{1} Int (Int.min a b) a (Int.min_eq_left a b h))) (Int.decLe a b)"},
	valDecl{"thm", "Int.max_le", nil,
		"(a : Int) → (b : Int) → (c : Int) → (Int.le a c) → (Int.le b c) → (Int.le (Int.max a b) c)",
		"fun (a : Int) (b : Int) (c : Int) (hac : (Int.le a c)) (hbc : (Int.le b c)) => Decidable.rec.{0} (Int.le a b) (fun (_ : Decidable (Int.le a " +
			"b)) => (Int.le (Int.max a b) c)) (fun (h : Not (Int.le a b)) => Eq.rec.{0, 1} Int a (fun (w : Int) (_ : Eq.{1} Int a w) => (Int.le w c)) hac " +
			"(Int.max a b) (Eq.symm.{1} Int (Int.max a b) a (Int.max_eq_left_of_not_le a b h))) (fun (h : (Int.le a b)) => Eq.rec.{0, 1} Int b (fun (w : " +
			"Int) (_ : Eq.{1} Int b w) => (Int.le w c)) hbc (Int.max a b) (Eq.symm.{1} Int (Int.max a b) b (Int.max_eq_right a b h))) (Int.decLe a b)"},
	valDecl{"thm", "Int.min_comm", nil,
		"(a : Int) → (b : Int) → Eq.{1} Int (Int.min a b) (Int.min b a)",
		"fun (a : Int) (b : Int) => Int.le_antisymm (Int.min a b) (Int.min b a) (Int.le_min b a (Int.min a b) (Int.min_le_right a b) (Int.min_le_left " +
			"a b)) (Int.le_min a b (Int.min b a) (Int.min_le_right b a) (Int.min_le_left b a))"},
	valDecl{"thm", "Int.max_comm", nil,
		"(a : Int) → (b : Int) → Eq.{1} Int (Int.max a b) (Int.max b a)",
		"fun (a : Int) (b : Int) => Int.le_antisymm (Int.max a b) (Int.max b a) (Int.max_le a b (Int.max b a) (Int.le_max_right b a) (Int.le_max_left " +
			"b a)) (Int.max_le b a (Int.max a b) (Int.le_max_right a b) (Int.le_max_left a b))"},
	// `min a a = a`, `max a a = a`, and associativity of both, each a `le_antisymm` of bounds from `min_le_*`, `le_min`, `le_max_*`, `max_le` and `le_trans`.
	valDecl{"thm", "Int.min_self", nil,
		"(a : Int) → Eq.{1} Int (Int.min a a) a",
		"fun (a : Int) => Int.le_antisymm (Int.min a a) a (Int.min_le_left a a) (Int.le_min a a a (Int.le_refl a) (Int.le_refl a))"},
	valDecl{"thm", "Int.max_self", nil,
		"(a : Int) → Eq.{1} Int (Int.max a a) a",
		"fun (a : Int) => Int.le_antisymm (Int.max a a) a (Int.max_le a a a (Int.le_refl a) (Int.le_refl a)) (Int.le_max_left a a)"},
	valDecl{"thm", "Int.min_assoc", nil,
		"(a : Int) → (b : Int) → (c : Int) → Eq.{1} Int (Int.min (Int.min a b) c) (Int.min a (Int.min b c))",
		"fun (a : Int) (b : Int) (c : Int) => Int.le_antisymm (Int.min (Int.min a b) c) (Int.min a (Int.min b c)) (Int.le_min a (Int.min b c) " +
			"(Int.min (Int.min a b) c) (Int.le_trans (Int.min (Int.min a b) c) (Int.min a b) a (Int.min_le_left (Int.min a b) c) (Int.min_le_left a b)) " +
			"(Int.le_min b c (Int.min (Int.min a b) c) (Int.le_trans (Int.min (Int.min a b) c) (Int.min a b) b (Int.min_le_left (Int.min a b) c) " +
			"(Int.min_le_right a b)) (Int.min_le_right (Int.min a b) c))) (Int.le_min (Int.min a b) c (Int.min a (Int.min b c)) (Int.le_min a b (Int.min " +
			"a (Int.min b c)) (Int.min_le_left a (Int.min b c)) (Int.le_trans (Int.min a (Int.min b c)) (Int.min b c) b (Int.min_le_right a (Int.min b " +
			"c)) (Int.min_le_left b c))) (Int.le_trans (Int.min a (Int.min b c)) (Int.min b c) c (Int.min_le_right a (Int.min b c)) (Int.min_le_right b " +
			"c)))"},
	valDecl{"thm", "Int.max_assoc", nil,
		"(a : Int) → (b : Int) → (c : Int) → Eq.{1} Int (Int.max (Int.max a b) c) (Int.max a (Int.max b c))",
		"fun (a : Int) (b : Int) (c : Int) => Int.le_antisymm (Int.max (Int.max a b) c) (Int.max a (Int.max b c)) (Int.max_le (Int.max a b) c " +
			"(Int.max a (Int.max b c)) (Int.max_le a b (Int.max a (Int.max b c)) (Int.le_max_left a (Int.max b c)) (Int.le_trans b (Int.max b c) (Int.max " +
			"a (Int.max b c)) (Int.le_max_left b c) (Int.le_max_right a (Int.max b c)))) (Int.le_trans c (Int.max b c) (Int.max a (Int.max b c)) " +
			"(Int.le_max_right b c) (Int.le_max_right a (Int.max b c)))) (Int.max_le a (Int.max b c) (Int.max (Int.max a b) c) (Int.le_trans a (Int.max a " +
			"b) (Int.max (Int.max a b) c) (Int.le_max_left a b) (Int.le_max_left (Int.max a b) c)) (Int.max_le b c (Int.max (Int.max a b) c) " +
			"(Int.le_trans b (Int.max a b) (Int.max (Int.max a b) c) (Int.le_max_right a b) (Int.le_max_left (Int.max a b) c)) (Int.le_max_right (Int.max " +
			"a b) c)))"},
	// `add_le_add` is `add_le_add_right` then `add_le_add_left` joined by `le_trans`. `neg_le_neg`: `-a - -b = -a + b = b + -a = b - a`, so `NonNeg (b - a)` carries over.
	valDecl{"thm", "Int.add_le_add", nil,
		"(a : Int) → (b : Int) → (c : Int) → (d : Int) → (Int.le a b) → (Int.le c d) → (Int.le (Int.add a c) (Int.add b d))",
		"fun (a : Int) (b : Int) (c : Int) (d : Int) (h1 : (Int.le a b)) (h2 : (Int.le c d)) => Int.le_trans (Int.add a c) (Int.add b c) (Int.add b " +
			"d) (Int.add_le_add_right a b c h1) (Int.add_le_add_left c d b h2)"},
	valDecl{"thm", "Int.neg_le_neg", nil,
		"(a : Int) → (b : Int) → (Int.le a b) → (Int.le (Int.neg b) (Int.neg a))",
		"fun (a : Int) (b : Int) (h : (Int.le a b)) => Eq.rec.{0, 1} Int (Int.sub b a) (fun (w : Int) (_ : Eq.{1} Int (Int.sub b a) w) => (Int.NonNeg " +
			"w)) h (Int.sub (Int.neg a) (Int.neg b)) (Eq.symm.{1} Int (Int.sub (Int.neg a) (Int.neg b)) (Int.sub b a) (Eq.trans.{1} Int (Int.sub (Int.neg " +
			"a) (Int.neg b)) (Int.add (Int.neg a) b) (Int.sub b a) (congrArg.{1, 1} Int Int (Int.add (Int.neg a)) (Int.neg (Int.neg b)) b (Int.neg_neg " +
			"b)) (Int.add_comm (Int.neg a) b)))"},
	// Strict order and `+`. `a < b` is `a + 1 <= b`; `add_le_add_right` on it needs `(a + c) + 1 = (a + 1) + c` (`add_assoc`, `add_comm`). `add_lt_add_left` is the right form
	// moved by `add_comm`, and `add_lt_add` joins the two with `lt_trans`.
	valDecl{"thm", "Int.add_lt_add_right", nil,
		"(a : Int) → (b : Int) → (c : Int) → (Int.lt a b) → (Int.lt (Int.add a c) (Int.add b c))",
		"fun (a : Int) (b : Int) (c : Int) (h : (Int.lt a b)) => Eq.rec.{0, 1} Int (Int.add (Int.add a (Int.ofNat 1)) c) (fun (w : Int) (_ : Eq.{1} " +
			"Int (Int.add (Int.add a (Int.ofNat 1)) c) w) => (Int.le w (Int.add b c))) (Int.add_le_add_right (Int.add a (Int.ofNat 1)) b c h) (Int.add " +
			"(Int.add a c) (Int.ofNat 1)) (Eq.symm.{1} Int (Int.add (Int.add a c) (Int.ofNat 1)) (Int.add (Int.add a (Int.ofNat 1)) c) (Eq.trans.{1} Int " +
			"(Int.add (Int.add a c) (Int.ofNat 1)) (Int.add a (Int.add (Int.ofNat 1) c)) (Int.add (Int.add a (Int.ofNat 1)) c) (Eq.trans.{1} Int (Int.add " +
			"(Int.add a c) (Int.ofNat 1)) (Int.add a (Int.add c (Int.ofNat 1))) (Int.add a (Int.add (Int.ofNat 1) c)) (Int.add_assoc a c (Int.ofNat 1)) " +
			"(congrArg.{1, 1} Int Int (Int.add a) (Int.add c (Int.ofNat 1)) (Int.add (Int.ofNat 1) c) (Int.add_comm c (Int.ofNat 1)))) (Eq.symm.{1} Int " +
			"(Int.add (Int.add a (Int.ofNat 1)) c) (Int.add a (Int.add (Int.ofNat 1) c)) (Int.add_assoc a (Int.ofNat 1) c))))"},
	valDecl{"thm", "Int.add_lt_add_left", nil,
		"(a : Int) → (b : Int) → (c : Int) → (Int.lt a b) → (Int.lt (Int.add c a) (Int.add c b))",
		"fun (a : Int) (b : Int) (c : Int) (h : (Int.lt a b)) => Eq.rec.{0, 1} Int (Int.add a c) (fun (w : Int) (_ : Eq.{1} Int (Int.add a c) w) => " +
			"(Int.lt w (Int.add c b))) (Eq.rec.{0, 1} Int (Int.add b c) (fun (v : Int) (_ : Eq.{1} Int (Int.add b c) v) => (Int.lt (Int.add a c) v)) " +
			"(Int.add_lt_add_right a b c h) (Int.add c b) (Int.add_comm b c)) (Int.add c a) (Int.add_comm a c)"},
	valDecl{"thm", "Int.add_lt_add", nil,
		"(a : Int) → (b : Int) → (c : Int) → (d : Int) → (Int.lt a b) → (Int.lt c d) → (Int.lt (Int.add a c) (Int.add b d))",
		"fun (a : Int) (b : Int) (c : Int) (d : Int) (h1 : (Int.lt a b)) (h2 : (Int.lt c d)) => Int.lt_trans (Int.add a c) (Int.add b c) (Int.add b " +
			"d) (Int.add_lt_add_right a b c h1) (Int.add_lt_add_left c d b h2)"},
	// Cancelling `+` in `<=`: add `-c` to both sides (`add_le_add_right`) and use `(x + c) + -c = x`. The left form is the right one after `add_comm`.
	valDecl{"thm", "Int.add_neg_cancel_right", nil,
		"(x : Int) → (c : Int) → Eq.{1} Int (Int.add (Int.add x c) (Int.neg c)) x",
		"fun (x : Int) (c : Int) => Eq.trans.{1} Int (Int.add (Int.add x c) (Int.neg c)) (Int.add x (Int.ofNat 0)) x (Eq.trans.{1} Int (Int.add " +
			"(Int.add x c) (Int.neg c)) (Int.add x (Int.add c (Int.neg c))) (Int.add x (Int.ofNat 0)) (Int.add_assoc x c (Int.neg c)) (congrArg.{1, 1} " +
			"Int Int (Int.add x) (Int.add c (Int.neg c)) (Int.ofNat 0) (Int.add_neg_cancel c))) (Int.add_zero x)"},
	valDecl{"thm", "Int.le_of_add_le_add_right", nil,
		"(a : Int) → (b : Int) → (c : Int) → (Int.le (Int.add a c) (Int.add b c)) → (Int.le a b)",
		"fun (a : Int) (b : Int) (c : Int) (h : (Int.le (Int.add a c) (Int.add b c))) => Eq.rec.{0, 1} Int (Int.add (Int.add a c) (Int.neg c)) (fun " +
			"(w : Int) (_ : Eq.{1} Int (Int.add (Int.add a c) (Int.neg c)) w) => (Int.le w b)) (Eq.rec.{0, 1} Int (Int.add (Int.add b c) (Int.neg c)) " +
			"(fun (v : Int) (_ : Eq.{1} Int (Int.add (Int.add b c) (Int.neg c)) v) => (Int.le (Int.add (Int.add a c) (Int.neg c)) v)) " +
			"(Int.add_le_add_right (Int.add a c) (Int.add b c) (Int.neg c) h) b (Int.add_neg_cancel_right b c)) a (Int.add_neg_cancel_right a c)"},
	valDecl{"thm", "Int.le_of_add_le_add_left", nil,
		"(a : Int) → (b : Int) → (c : Int) → (Int.le (Int.add c a) (Int.add c b)) → (Int.le a b)",
		"fun (a : Int) (b : Int) (c : Int) (h : (Int.le (Int.add c a) (Int.add c b))) => Int.le_of_add_le_add_right a b c (Eq.rec.{0, 1} Int (Int.add " +
			"c a) (fun (w : Int) (_ : Eq.{1} Int (Int.add c a) w) => (Int.le w (Int.add b c))) (Eq.rec.{0, 1} Int (Int.add c b) (fun (v : Int) (_ : " +
			"Eq.{1} Int (Int.add c b) v) => (Int.le (Int.add c a) v)) h (Int.add b c) (Int.add_comm c b)) (Int.add a c) (Int.add_comm c a))"},
	// Cancelling `+` in `<`: `(a + c) + 1 = (a + 1) + c` turns `a + c < b + c` into `(a + 1) + c <= b + c`, then `le_of_add_le_add_right`. The left form is the right one after `add_comm`.
	valDecl{"thm", "Int.lt_of_add_lt_add_right", nil,
		"(a : Int) → (b : Int) → (c : Int) → (Int.lt (Int.add a c) (Int.add b c)) → (Int.lt a b)",
		"fun (a : Int) (b : Int) (c : Int) (h : (Int.lt (Int.add a c) (Int.add b c))) => Int.le_of_add_le_add_right (Int.add a (Int.ofNat 1)) b c " +
			"(Eq.rec.{0, 1} Int (Int.add (Int.add a c) (Int.ofNat 1)) (fun (w : Int) (_ : Eq.{1} Int (Int.add (Int.add a c) (Int.ofNat 1)) w) => (Int.le " +
			"w (Int.add b c))) h (Int.add (Int.add a (Int.ofNat 1)) c) (Eq.trans.{1} Int (Int.add (Int.add a c) (Int.ofNat 1)) (Int.add a (Int.add " +
			"(Int.ofNat 1) c)) (Int.add (Int.add a (Int.ofNat 1)) c) (Eq.trans.{1} Int (Int.add (Int.add a c) (Int.ofNat 1)) (Int.add a (Int.add c " +
			"(Int.ofNat 1))) (Int.add a (Int.add (Int.ofNat 1) c)) (Int.add_assoc a c (Int.ofNat 1)) (congrArg.{1, 1} Int Int (Int.add a) (Int.add c " +
			"(Int.ofNat 1)) (Int.add (Int.ofNat 1) c) (Int.add_comm c (Int.ofNat 1)))) (Eq.symm.{1} Int (Int.add (Int.add a (Int.ofNat 1)) c) (Int.add a " +
			"(Int.add (Int.ofNat 1) c)) (Int.add_assoc a (Int.ofNat 1) c))))"},
	valDecl{"thm", "Int.lt_of_add_lt_add_left", nil,
		"(a : Int) → (b : Int) → (c : Int) → (Int.lt (Int.add c a) (Int.add c b)) → (Int.lt a b)",
		"fun (a : Int) (b : Int) (c : Int) (h : (Int.lt (Int.add c a) (Int.add c b))) => Int.lt_of_add_lt_add_right a b c (Eq.rec.{0, 1} Int (Int.add " +
			"c a) (fun (w : Int) (_ : Eq.{1} Int (Int.add c a) w) => (Int.lt w (Int.add b c))) (Eq.rec.{0, 1} Int (Int.add c b) (fun (v : Int) (_ : " +
			"Eq.{1} Int (Int.add c b) v) => (Int.lt (Int.add c a) v)) h (Int.add b c) (Int.add_comm c b)) (Int.add a c) (Int.add_comm c a))"},
	// `0 <= a - b` iff `b <= a` (`nonneg_of_le_zero`, `le_zero_of_nonneg`). `le_of_neg_le_neg` is `neg_le_neg` on `-b <= -a` with `--x = x`, so `neg_le_neg_iff` holds.
	valDecl{"thm", "Int.sub_nonneg", nil,
		"(a : Int) → (b : Int) → Iff (Int.le (Int.ofNat 0) (Int.sub a b)) (Int.le b a)",
		"fun (a : Int) (b : Int) => Iff.intro (Int.le (Int.ofNat 0) (Int.sub a b)) (Int.le b a) (fun (h : (Int.le (Int.ofNat 0) (Int.sub a b))) => " +
			"Int.nonneg_of_le_zero (Int.sub a b) h) (fun (h : (Int.le b a)) => Int.le_zero_of_nonneg (Int.sub a b) h)"},
	valDecl{"thm", "Int.le_of_neg_le_neg", nil,
		"(a : Int) → (b : Int) → (Int.le (Int.neg b) (Int.neg a)) → (Int.le a b)",
		"fun (a : Int) (b : Int) (h : (Int.le (Int.neg b) (Int.neg a))) => Eq.rec.{0, 1} Int (Int.neg (Int.neg a)) (fun (w : Int) (_ : Eq.{1} Int " +
			"(Int.neg (Int.neg a)) w) => (Int.le w b)) (Eq.rec.{0, 1} Int (Int.neg (Int.neg b)) (fun (v : Int) (_ : Eq.{1} Int (Int.neg (Int.neg b)) v) " +
			"=> (Int.le (Int.neg (Int.neg a)) v)) (Int.neg_le_neg (Int.neg b) (Int.neg a) h) b (Int.neg_neg b)) a (Int.neg_neg a)"},
	valDecl{"thm", "Int.neg_le_neg_iff", nil,
		"(a : Int) → (b : Int) → Iff (Int.le (Int.neg b) (Int.neg a)) (Int.le a b)",
		"fun (a : Int) (b : Int) => Iff.intro (Int.le (Int.neg b) (Int.neg a)) (Int.le a b) (fun (h : (Int.le (Int.neg b) (Int.neg a))) => " +
			"Int.le_of_neg_le_neg a b h) (fun (h : (Int.le a b)) => Int.neg_le_neg a b h)"},
	// `a - c` is `a + -c`, so `sub_le_sub_right` is `add_le_add_right`; the left and two-sided forms use `neg_le_neg`. `mul_nonneg` is `NonNeg_mul` seen through `0 <= x` iff `NonNeg x`.
	valDecl{"thm", "Int.sub_le_sub_right", nil,
		"(a : Int) → (b : Int) → (c : Int) → (Int.le a b) → (Int.le (Int.sub a c) (Int.sub b c))",
		"fun (a : Int) (b : Int) (c : Int) (h : (Int.le a b)) => Int.add_le_add_right a b (Int.neg c) h"},
	valDecl{"thm", "Int.sub_le_sub_left", nil,
		"(a : Int) → (b : Int) → (c : Int) → (Int.le a b) → (Int.le (Int.sub c b) (Int.sub c a))",
		"fun (a : Int) (b : Int) (c : Int) (h : (Int.le a b)) => Int.add_le_add_left (Int.neg b) (Int.neg a) c (Int.neg_le_neg a b h)"},
	valDecl{"thm", "Int.sub_le_sub", nil,
		"(a : Int) → (b : Int) → (c : Int) → (d : Int) → (Int.le a b) → (Int.le c d) → (Int.le (Int.sub a d) (Int.sub b c))",
		"fun (a : Int) (b : Int) (c : Int) (d : Int) (h1 : (Int.le a b)) (h2 : (Int.le c d)) => Int.add_le_add a b (Int.neg d) (Int.neg c) h1 " +
			"(Int.neg_le_neg c d h2)"},
	valDecl{"thm", "Int.mul_nonneg", nil,
		"(a : Int) → (b : Int) → (Int.le (Int.ofNat 0) a) → (Int.le (Int.ofNat 0) b) → (Int.le (Int.ofNat 0) (Int.mul a b))",
		"fun (a : Int) (b : Int) (ha : (Int.le (Int.ofNat 0) a)) (hb : (Int.le (Int.ofNat 0) b)) => Int.le_zero_of_nonneg (Int.mul a b) " +
			"(Int.NonNeg_mul a b (Int.nonneg_of_le_zero a ha) (Int.nonneg_of_le_zero b hb))"},
	// `0 < a` is `1 <= a`. `1 <= a` gives `1 * b <= a * b` (`mul_le_mul_of_nonneg_right`, `b` nonneg), `1 * b = b`, and `1 <= b`, so `1 <= a * b` by `le_trans`.
	valDecl{"thm", "Int.mul_pos", nil,
		"(a : Int) → (b : Int) → (Int.lt (Int.ofNat 0) a) → (Int.lt (Int.ofNat 0) b) → (Int.lt (Int.ofNat 0) (Int.mul a b))",
		"fun (a : Int) (b : Int) (ha : (Int.lt (Int.ofNat 0) a)) (hb : (Int.lt (Int.ofNat 0) b)) => Int.le_trans (Int.ofNat 1) b (Int.mul a b) hb " +
			"(Eq.rec.{0, 1} Int (Int.mul (Int.ofNat 1) b) (fun (w : Int) (_ : Eq.{1} Int (Int.mul (Int.ofNat 1) b) w) => (Int.le w (Int.mul a b))) " +
			"(Int.mul_le_mul_of_nonneg_right (Int.ofNat 1) a b (Int.nonneg_of_le_zero b (Int.le_of_lt (Int.ofNat 0) b hb)) ha) b (Int.one_mul b))"},
	// `neg_lt_neg`: `-b < -a` is `NonNeg (-a - (-b + 1))`, and `-a - (-b + 1) = (-a - -b) - 1 = (b - a) - 1 = b - (a + 1)` by `sub_add_eq_sub_sub` and `-a - -b = b - a`,
	// the `NonNeg` that `a < b` is.
	valDecl{"thm", "Int.neg_lt_neg", nil,
		"(a : Int) → (b : Int) → (Int.lt a b) → (Int.lt (Int.neg b) (Int.neg a))",
		"fun (a : Int) (b : Int) (h : (Int.lt a b)) => Eq.rec.{0, 1} Int (Int.sub b (Int.add a (Int.ofNat 1))) (fun (w : Int) (_ : Eq.{1} Int " +
			"(Int.sub b (Int.add a (Int.ofNat 1))) w) => (Int.NonNeg w)) h (Int.sub (Int.neg a) (Int.add (Int.neg b) (Int.ofNat 1))) (Eq.symm.{1} Int " +
			"(Int.sub (Int.neg a) (Int.add (Int.neg b) (Int.ofNat 1))) (Int.sub b (Int.add a (Int.ofNat 1))) (Eq.trans.{1} Int (Int.sub (Int.neg a) " +
			"(Int.add (Int.neg b) (Int.ofNat 1))) (Int.sub (Int.sub (Int.neg a) (Int.neg b)) (Int.ofNat 1)) (Int.sub b (Int.add a (Int.ofNat 1))) " +
			"(Eq.trans.{1} Int (Int.sub (Int.neg a) (Int.add (Int.neg b) (Int.ofNat 1))) (Int.sub (Int.sub (Int.neg a) (Int.neg b)) (Int.ofNat 1)) " +
			"(Int.sub (Int.sub (Int.neg a) (Int.neg b)) (Int.ofNat 1)) (Int.sub_add_eq_sub_sub (Int.neg b) (Int.neg a) (Int.ofNat 1)) (Eq.refl.{1} Int " +
			"(Int.sub (Int.sub (Int.neg a) (Int.neg b)) (Int.ofNat 1)))) (Eq.trans.{1} Int (Int.sub (Int.sub (Int.neg a) (Int.neg b)) (Int.ofNat 1)) " +
			"(Int.sub (Int.sub b a) (Int.ofNat 1)) (Int.sub b (Int.add a (Int.ofNat 1))) (congrArg.{1, 1} Int Int (fun (z : Int) => Int.sub z (Int.ofNat " +
			"1)) (Int.sub (Int.neg a) (Int.neg b)) (Int.sub b a) (Eq.trans.{1} Int (Int.sub (Int.neg a) (Int.neg b)) (Int.add (Int.neg a) b) (Int.sub b " +
			"a) (congrArg.{1, 1} Int Int (Int.add (Int.neg a)) (Int.neg (Int.neg b)) b (Int.neg_neg b)) (Int.add_comm (Int.neg a) b))) (Eq.symm.{1} Int " +
			"(Int.sub b (Int.add a (Int.ofNat 1))) (Int.sub (Int.sub b a) (Int.ofNat 1)) (Int.sub_add_eq_sub_sub a b (Int.ofNat 1))))))"},
	// `min (a + c) (b + c) = min a b + c` by `le_antisymm`. `min a b + c <= min (a + c) (b + c)` is `le_min` of two `add_le_add_right`. The other way, `m + -c <= a, b`
	// (`add_neg_cancel_right`) gives `m + -c <= min a b`, and `le_of_add_le_add_right` moves `-c` across.
	valDecl{"thm", "Int.min_add_add_right", nil,
		"(a : Int) → (b : Int) → (c : Int) → Eq.{1} Int (Int.min (Int.add a c) (Int.add b c)) (Int.add (Int.min a b) c)",
		"fun (a : Int) (b : Int) (c : Int) => Int.le_antisymm (Int.min (Int.add a c) (Int.add b c)) (Int.add (Int.min a b) c) " +
			"(Int.le_of_add_le_add_right (Int.min (Int.add a c) (Int.add b c)) (Int.add (Int.min a b) c) (Int.neg c) (Eq.rec.{0, 1} Int (Int.min a b) " +
			"(fun (w : Int) (_ : Eq.{1} Int (Int.min a b) w) => (Int.le (Int.add (Int.min (Int.add a c) (Int.add b c)) (Int.neg c)) w)) (Int.le_min a b " +
			"(Int.add (Int.min (Int.add a c) (Int.add b c)) (Int.neg c)) (Eq.rec.{0, 1} Int (Int.add (Int.add a c) (Int.neg c)) (fun (w : Int) (_ : " +
			"Eq.{1} Int (Int.add (Int.add a c) (Int.neg c)) w) => (Int.le (Int.add (Int.min (Int.add a c) (Int.add b c)) (Int.neg c)) w)) " +
			"(Int.add_le_add_right (Int.min (Int.add a c) (Int.add b c)) (Int.add a c) (Int.neg c) (Int.min_le_left (Int.add a c) (Int.add b c))) a " +
			"(Int.add_neg_cancel_right a c)) (Eq.rec.{0, 1} Int (Int.add (Int.add b c) (Int.neg c)) (fun (w : Int) (_ : Eq.{1} Int (Int.add (Int.add b c) " +
			"(Int.neg c)) w) => (Int.le (Int.add (Int.min (Int.add a c) (Int.add b c)) (Int.neg c)) w)) (Int.add_le_add_right (Int.min (Int.add a c) " +
			"(Int.add b c)) (Int.add b c) (Int.neg c) (Int.min_le_right (Int.add a c) (Int.add b c))) b (Int.add_neg_cancel_right b c))) (Int.add " +
			"(Int.add (Int.min a b) c) (Int.neg c)) (Eq.symm.{1} Int (Int.add (Int.add (Int.min a b) c) (Int.neg c)) (Int.min a b) " +
			"(Int.add_neg_cancel_right (Int.min a b) c)))) (Int.le_min (Int.add a c) (Int.add b c) (Int.add (Int.min a b) c) (Int.add_le_add_right " +
			"(Int.min a b) a c (Int.min_le_left a b)) (Int.add_le_add_right (Int.min a b) b c (Int.min_le_right a b)))"},
	// `max (a + c) (b + c) = max a b + c`, mirror of `min_add_add_right`: `max_le` of two `add_le_add_right` one way; the other, `a, b <= m + -c` gives
	// `max a b <= m + -c`, and `le_of_add_le_add_right` moves `-c` across.
	valDecl{"thm", "Int.max_add_add_right", nil,
		"(a : Int) → (b : Int) → (c : Int) → Eq.{1} Int (Int.max (Int.add a c) (Int.add b c)) (Int.add (Int.max a b) c)",
		"fun (a : Int) (b : Int) (c : Int) => Int.le_antisymm (Int.max (Int.add a c) (Int.add b c)) (Int.add (Int.max a b) c) (Int.max_le (Int.add a " +
			"c) (Int.add b c) (Int.add (Int.max a b) c) (Int.add_le_add_right a (Int.max a b) c (Int.le_max_left a b)) (Int.add_le_add_right b (Int.max a " +
			"b) c (Int.le_max_right a b))) (Int.le_of_add_le_add_right (Int.add (Int.max a b) c) (Int.max (Int.add a c) (Int.add b c)) (Int.neg c) " +
			"(Eq.rec.{0, 1} Int (Int.max a b) (fun (w : Int) (_ : Eq.{1} Int (Int.max a b) w) => (Int.le w (Int.add (Int.max (Int.add a c) (Int.add b c)) " +
			"(Int.neg c)))) (Int.max_le a b (Int.add (Int.max (Int.add a c) (Int.add b c)) (Int.neg c)) (Eq.rec.{0, 1} Int (Int.add (Int.add a c) " +
			"(Int.neg c)) (fun (w : Int) (_ : Eq.{1} Int (Int.add (Int.add a c) (Int.neg c)) w) => (Int.le w (Int.add (Int.max (Int.add a c) (Int.add b " +
			"c)) (Int.neg c)))) (Int.add_le_add_right (Int.add a c) (Int.max (Int.add a c) (Int.add b c)) (Int.neg c) (Int.le_max_left (Int.add a c) " +
			"(Int.add b c))) a (Int.add_neg_cancel_right a c)) (Eq.rec.{0, 1} Int (Int.add (Int.add b c) (Int.neg c)) (fun (w : Int) (_ : Eq.{1} Int " +
			"(Int.add (Int.add b c) (Int.neg c)) w) => (Int.le w (Int.add (Int.max (Int.add a c) (Int.add b c)) (Int.neg c)))) (Int.add_le_add_right " +
			"(Int.add b c) (Int.max (Int.add a c) (Int.add b c)) (Int.neg c) (Int.le_max_right (Int.add a c) (Int.add b c))) b (Int.add_neg_cancel_right " +
			"b c))) (Int.add (Int.add (Int.max a b) c) (Int.neg c)) (Eq.symm.{1} Int (Int.add (Int.add (Int.max a b) c) (Int.neg c)) (Int.max a b) " +
			"(Int.add_neg_cancel_right (Int.max a b) c))))"},
	// The left forms `min (c + a) (c + b) = c + min a b` (and `max`): `add_comm` inside both arguments, `min_add_add_right`, then `add_comm` outside.
	valDecl{"thm", "Int.min_add_add_left", nil,
		"(a : Int) → (b : Int) → (c : Int) → Eq.{1} Int (Int.min (Int.add c a) (Int.add c b)) (Int.add c (Int.min a b))",
		"fun (a : Int) (b : Int) (c : Int) => Eq.trans.{1} Int (Int.min (Int.add c a) (Int.add c b)) (Int.min (Int.add a c) (Int.add b c)) (Int.add c " +
			"(Int.min a b)) (Eq.trans.{1} Int (Int.min (Int.add c a) (Int.add c b)) (Int.min (Int.add a c) (Int.add c b)) (Int.min (Int.add a c) (Int.add " +
			"b c)) (congrArg.{1, 1} Int Int (fun (x : Int) => Int.min x (Int.add c b)) (Int.add c a) (Int.add a c) (Int.add_comm c a)) (congrArg.{1, 1} " +
			"Int Int (fun (y : Int) => Int.min (Int.add a c) y) (Int.add c b) (Int.add b c) (Int.add_comm c b))) (Eq.trans.{1} Int (Int.min (Int.add a c) " +
			"(Int.add b c)) (Int.add (Int.min a b) c) (Int.add c (Int.min a b)) (Int.min_add_add_right a b c) (Int.add_comm (Int.min a b) c))"},
	valDecl{"thm", "Int.max_add_add_left", nil,
		"(a : Int) → (b : Int) → (c : Int) → Eq.{1} Int (Int.max (Int.add c a) (Int.add c b)) (Int.add c (Int.max a b))",
		"fun (a : Int) (b : Int) (c : Int) => Eq.trans.{1} Int (Int.max (Int.add c a) (Int.add c b)) (Int.max (Int.add a c) (Int.add b c)) (Int.add c " +
			"(Int.max a b)) (Eq.trans.{1} Int (Int.max (Int.add c a) (Int.add c b)) (Int.max (Int.add a c) (Int.add c b)) (Int.max (Int.add a c) (Int.add " +
			"b c)) (congrArg.{1, 1} Int Int (fun (x : Int) => Int.max x (Int.add c b)) (Int.add c a) (Int.add a c) (Int.add_comm c a)) (congrArg.{1, 1} " +
			"Int Int (fun (y : Int) => Int.max (Int.add a c) y) (Int.add c b) (Int.add b c) (Int.add_comm c b))) (Eq.trans.{1} Int (Int.max (Int.add a c) " +
			"(Int.add b c)) (Int.add (Int.max a b) c) (Int.add c (Int.max a b)) (Int.max_add_add_right a b c) (Int.add_comm (Int.max a b) c))"},
	// `-(min a b) = max (-a) (-b)`: `le_antisymm`. `max_le` with `neg_le_neg` of `min_le_left`/`min_le_right` gives `>=`; for `<=` split on `decide (a <= b)`, `min` is `a` or `b`,
	// and `-a`, `-b` is `le_max_left`/`le_max_right`.
	valDecl{"thm", "Int.neg_min", nil,
		"(a : Int) → (b : Int) → Eq.{1} Int (Int.neg (Int.min a b)) (Int.max (Int.neg a) (Int.neg b))",
		"fun (a : Int) (b : Int) => Int.le_antisymm (Int.neg (Int.min a b)) (Int.max (Int.neg a) (Int.neg b)) (Decidable.rec.{0} (Int.le a b) (fun (_ " +
			": Decidable (Int.le a b)) => (Int.le (Int.neg (Int.min a b)) (Int.max (Int.neg a) (Int.neg b)))) (fun (h : Not (Int.le a b)) => Eq.rec.{0, " +
			"1} Int b (fun (w : Int) (_ : Eq.{1} Int b w) => (Int.le (Int.neg w) (Int.max (Int.neg a) (Int.neg b)))) (Int.le_max_right (Int.neg a) " +
			"(Int.neg b)) (Int.min a b) (Eq.symm.{1} Int (Int.min a b) b (Int.min_eq_right_of_not_le a b h))) (fun (h : (Int.le a b)) => Eq.rec.{0, 1} " +
			"Int a (fun (w : Int) (_ : Eq.{1} Int a w) => (Int.le (Int.neg w) (Int.max (Int.neg a) (Int.neg b)))) (Int.le_max_left (Int.neg a) (Int.neg " +
			"b)) (Int.min a b) (Eq.symm.{1} Int (Int.min a b) a (Int.min_eq_left a b h))) (Int.decLe a b)) (Int.max_le (Int.neg a) (Int.neg b) (Int.neg " +
			"(Int.min a b)) (Int.neg_le_neg (Int.min a b) a (Int.min_le_left a b)) (Int.neg_le_neg (Int.min a b) b (Int.min_le_right a b)))"},
	// `-(max a b) = min (-a) (-b)`: `neg_min` at `-a`, `-b` reads `-(min (-a) (-b)) = max (--a) (--b)`; rewrite `--x = x` and negate both sides.
	valDecl{"thm", "Int.neg_max", nil,
		"(a : Int) → (b : Int) → Eq.{1} Int (Int.neg (Int.max a b)) (Int.min (Int.neg a) (Int.neg b))",
		"fun (a : Int) (b : Int) => Eq.trans.{1} Int (Int.neg (Int.max a b)) (Int.neg (Int.max (Int.neg (Int.neg a)) (Int.neg (Int.neg b)))) (Int.min " +
			"(Int.neg a) (Int.neg b)) (congrArg.{1, 1} Int Int Int.neg (Int.max a b) (Int.max (Int.neg (Int.neg a)) (Int.neg (Int.neg b))) (Eq.trans.{1} " +
			"Int (Int.max a b) (Int.max (Int.neg (Int.neg a)) b) (Int.max (Int.neg (Int.neg a)) (Int.neg (Int.neg b))) (congrArg.{1, 1} Int Int (fun (x : " +
			"Int) => Int.max x b) a (Int.neg (Int.neg a)) (Eq.symm.{1} Int (Int.neg (Int.neg a)) a (Int.neg_neg a))) (congrArg.{1, 1} Int Int (Int.max " +
			"(Int.neg (Int.neg a))) b (Int.neg (Int.neg b)) (Eq.symm.{1} Int (Int.neg (Int.neg b)) b (Int.neg_neg b))))) (Eq.trans.{1} Int (Int.neg " +
			"(Int.max (Int.neg (Int.neg a)) (Int.neg (Int.neg b)))) (Int.neg (Int.neg (Int.min (Int.neg a) (Int.neg b)))) (Int.min (Int.neg a) (Int.neg " +
			"b)) (congrArg.{1, 1} Int Int Int.neg (Int.max (Int.neg (Int.neg a)) (Int.neg (Int.neg b))) (Int.neg (Int.min (Int.neg a) (Int.neg b))) " +
			"(Eq.symm.{1} Int (Int.neg (Int.min (Int.neg a) (Int.neg b))) (Int.max (Int.neg (Int.neg a)) (Int.neg (Int.neg b))) (Int.neg_min (Int.neg a) " +
			"(Int.neg b)))) (Int.neg_neg (Int.min (Int.neg a) (Int.neg b))))"},
	// `a < b + 1` iff `a <= b`: `a + 1 <= b + 1` against `add_le_add_right` and `le_of_add_le_add_right`. So `a < a + 1`; and `a - 1 < a` since `(a - 1) + 1 = a`.
	valDecl{"thm", "Int.lt_add_one_iff", nil,
		"(a : Int) → (b : Int) → Iff (Int.lt a (Int.add b (Int.ofNat 1))) (Int.le a b)",
		"fun (a : Int) (b : Int) => Iff.intro (Int.lt a (Int.add b (Int.ofNat 1))) (Int.le a b) (fun (h : (Int.lt a (Int.add b (Int.ofNat 1)))) => " +
			"Int.le_of_add_le_add_right a b (Int.ofNat 1) h) (fun (h : (Int.le a b)) => Int.add_le_add_right a b (Int.ofNat 1) h)"},
	valDecl{"thm", "Int.lt_add_one", nil,
		"(a : Int) → (Int.lt a (Int.add a (Int.ofNat 1)))",
		"fun (a : Int) => Iff.mpr (Int.lt a (Int.add a (Int.ofNat 1))) (Int.le a a) (Int.lt_add_one_iff a a) (Int.le_refl a)"},
	valDecl{"thm", "Int.sub_one_lt", nil,
		"(a : Int) → (Int.lt (Int.sub a (Int.ofNat 1)) a)",
		"fun (a : Int) => Eq.rec.{0, 1} Int a (fun (w : Int) (_ : Eq.{1} Int a w) => (Int.le w a)) (Int.le_refl a) (Int.add (Int.sub a (Int.ofNat 1)) " +
			"(Int.ofNat 1)) (Eq.symm.{1} Int (Int.add (Int.sub a (Int.ofNat 1)) (Int.ofNat 1)) a (Int.sub_add_cancel a (Int.ofNat 1)))"},
	// `mul_lt_mul_of_pos_left`: `c * a + 1 <= c * a + c` (`1 <= c`), and `c * a + c = c * (a + 1) <= c * b` by `mul_add`, `mul_one` and `mul_le_mul_of_nonneg_left`.
	valDecl{"thm", "Int.mul_lt_mul_of_pos_left", nil,
		"(a : Int) → (b : Int) → (c : Int) → (Int.lt a b) → (Int.lt (Int.ofNat 0) c) → (Int.lt (Int.mul c a) (Int.mul c b))",
		"fun (a : Int) (b : Int) (c : Int) (h : (Int.lt a b)) (hc : (Int.lt (Int.ofNat 0) c)) => Int.le_trans (Int.add (Int.mul c a) (Int.ofNat 1)) " +
			"(Int.add (Int.mul c a) c) (Int.mul c b) (Int.add_le_add_left (Int.ofNat 1) c (Int.mul c a) hc) (Eq.rec.{0, 1} Int (Int.mul c (Int.add a " +
			"(Int.ofNat 1))) (fun (w : Int) (_ : Eq.{1} Int (Int.mul c (Int.add a (Int.ofNat 1))) w) => (Int.le w (Int.mul c b))) " +
			"(Int.mul_le_mul_of_nonneg_left (Int.add a (Int.ofNat 1)) b c (Int.nonneg_of_le_zero c (Int.le_of_lt (Int.ofNat 0) c hc)) h) (Int.add " +
			"(Int.mul c a) c) (Eq.trans.{1} Int (Int.mul c (Int.add a (Int.ofNat 1))) (Int.add (Int.mul c a) (Int.mul c (Int.ofNat 1))) (Int.add (Int.mul " +
			"c a) c) (Int.mul_add c a (Int.ofNat 1)) (congrArg.{1, 1} Int Int (Int.add (Int.mul c a)) (Int.mul c (Int.ofNat 1)) c (Int.mul_one c))))"},
	// The right form `a * c < b * c`: `mul_lt_mul_of_pos_left` with `mul_comm` on both products.
	valDecl{"thm", "Int.mul_lt_mul_of_pos_right", nil,
		"(a : Int) → (b : Int) → (c : Int) → (Int.lt a b) → (Int.lt (Int.ofNat 0) c) → (Int.lt (Int.mul a c) (Int.mul b c))",
		"fun (a : Int) (b : Int) (c : Int) (h : (Int.lt a b)) (hc : (Int.lt (Int.ofNat 0) c)) => Eq.rec.{0, 1} Int (Int.mul c a) (fun (w : Int) (_ : " +
			"Eq.{1} Int (Int.mul c a) w) => (Int.lt w (Int.mul b c))) (Eq.rec.{0, 1} Int (Int.mul c b) (fun (v : Int) (_ : Eq.{1} Int (Int.mul c b) v) => " +
			"(Int.lt (Int.mul c a) v)) (Int.mul_lt_mul_of_pos_left a b c h hc) (Int.mul b c) (Int.mul_comm c b)) (Int.mul a c) (Int.mul_comm c a)"},
	// Cancelling a positive factor in `<`: `c * a < c * b` rules out `b <= a` (`mul_le_mul_of_nonneg_left`, `not_le_of_lt`), so `le_total` leaves `a <= b` and `lt_of_le_of_not_le` gives `a < b`.
	// The right form is the left one after `mul_comm` on both products.
	valDecl{"thm", "Int.lt_of_mul_lt_mul_left", nil,
		"(a : Int) → (b : Int) → (c : Int) → (Int.lt (Int.mul c a) (Int.mul c b)) → (Int.lt (Int.ofNat 0) c) → (Int.lt a b)",
		"fun (a : Int) (b : Int) (c : Int) (h : (Int.lt (Int.mul c a) (Int.mul c b))) (hc : (Int.lt (Int.ofNat 0) c)) => Or.rec (Int.le a b) (Int.le " +
			"b a) (fun (_ : Or (Int.le a b) (Int.le b a)) => (Int.lt a b)) (fun (h1 : (Int.le a b)) => Int.lt_of_le_of_not_le a b h1 (fun (hba : (Int.le " +
			"b a)) => Int.not_le_of_lt (Int.mul c a) (Int.mul c b) h (Int.mul_le_mul_of_nonneg_left b a c (Int.nonneg_of_le_zero c (Int.le_of_lt " +
			"(Int.ofNat 0) c hc)) hba))) (fun (h2 : (Int.le b a)) => False.elim.{0} (Int.lt a b) ((fun (hba : (Int.le b a)) => Int.not_le_of_lt (Int.mul " +
			"c a) (Int.mul c b) h (Int.mul_le_mul_of_nonneg_left b a c (Int.nonneg_of_le_zero c (Int.le_of_lt (Int.ofNat 0) c hc)) hba)) h2)) " +
			"(Int.le_total a b)"},
	valDecl{"thm", "Int.lt_of_mul_lt_mul_right", nil,
		"(a : Int) → (b : Int) → (c : Int) → (Int.lt (Int.mul a c) (Int.mul b c)) → (Int.lt (Int.ofNat 0) c) → (Int.lt a b)",
		"fun (a : Int) (b : Int) (c : Int) (h : (Int.lt (Int.mul a c) (Int.mul b c))) (hc : (Int.lt (Int.ofNat 0) c)) => Int.lt_of_mul_lt_mul_left a " +
			"b c (Eq.rec.{0, 1} Int (Int.mul b c) (fun (v : Int) (_ : Eq.{1} Int (Int.mul b c) v) => (Int.lt (Int.mul c a) v)) (Eq.rec.{0, 1} Int " +
			"(Int.mul a c) (fun (w : Int) (_ : Eq.{1} Int (Int.mul a c) w) => (Int.lt w (Int.mul b c))) h (Int.mul c a) (Int.mul_comm a c)) (Int.mul c b) " +
			"(Int.mul_comm b c)) hc"},
	// Two-sided monotonicity: `a <= b`, `c <= d`, `0 <= c`, `0 <= b` give `a * c <= b * c <= b * d` by the right and left one-sided forms and `le_trans`.
	valDecl{"thm", "Int.mul_le_mul", nil,
		"(a : Int) → (b : Int) → (c : Int) → (d : Int) → (Int.le a b) → (Int.le c d) → (Int.le (Int.ofNat 0) c) → (Int.le (Int.ofNat 0) b) → (Int.le " +
			"(Int.mul a c) (Int.mul b d))",
		"fun (a : Int) (b : Int) (c : Int) (d : Int) (h1 : (Int.le a b)) (h2 : (Int.le c d)) (hc : (Int.le (Int.ofNat 0) c)) (hb : (Int.le (Int.ofNat " +
			"0) b)) => Int.le_trans (Int.mul a c) (Int.mul b c) (Int.mul b d) (Int.mul_le_mul_of_nonneg_right a b c (Int.nonneg_of_le_zero c hc) h1) " +
			"(Int.mul_le_mul_of_nonneg_left c d b (Int.nonneg_of_le_zero b hb) h2)"},
	// `(-a) * (-b) = a * b` from `neg_mul`, `mul_neg`, `neg_neg`. `0 <= a * a`: if `0 <= a` it is `mul_nonneg`; if `a <= 0` then `0 <= -a` (`neg_le_neg`) and `(-a) * (-a) = a * a`.
	valDecl{"thm", "Int.neg_mul_neg", nil,
		"(a : Int) → (b : Int) → Eq.{1} Int (Int.mul (Int.neg a) (Int.neg b)) (Int.mul a b)",
		"fun (a : Int) (b : Int) => Eq.trans.{1} Int (Int.mul (Int.neg a) (Int.neg b)) (Int.neg (Int.mul a (Int.neg b))) (Int.mul a b) (Int.neg_mul a " +
			"(Int.neg b)) (Eq.trans.{1} Int (Int.neg (Int.mul a (Int.neg b))) (Int.neg (Int.neg (Int.mul a b))) (Int.mul a b) (congrArg.{1, 1} Int Int " +
			"Int.neg (Int.mul a (Int.neg b)) (Int.neg (Int.mul a b)) (Int.mul_neg a b)) (Int.neg_neg (Int.mul a b)))"},
	valDecl{"thm", "Int.mul_self_nonneg", nil,
		"(a : Int) → (Int.le (Int.ofNat 0) (Int.mul a a))",
		"fun (a : Int) => Or.rec (Int.le (Int.ofNat 0) a) (Int.le a (Int.ofNat 0)) (fun (_ : Or (Int.le (Int.ofNat 0) a) (Int.le a (Int.ofNat 0))) => " +
			"(Int.le (Int.ofNat 0) (Int.mul a a))) (fun (h : (Int.le (Int.ofNat 0) a)) => Int.mul_nonneg a a h h) (fun (h : (Int.le a (Int.ofNat 0))) => " +
			"Eq.rec.{0, 1} Int (Int.mul (Int.neg a) (Int.neg a)) (fun (w : Int) (_ : Eq.{1} Int (Int.mul (Int.neg a) (Int.neg a)) w) => (Int.le " +
			"(Int.ofNat 0) w)) (Int.mul_nonneg (Int.neg a) (Int.neg a) (Int.neg_le_neg a (Int.ofNat 0) h) (Int.neg_le_neg a (Int.ofNat 0) h)) (Int.mul a " +
			"a) (Int.neg_mul_neg a a)) (Int.le_total (Int.ofNat 0) a)"},
	// `a <= natAbs a`: `le_refl` for `ofNat k`; for `negSucc k` the difference `succ k + succ k` is a `NonNeg.mk`.
	// `-natAbs a <= a`: `le_natAbs (-a)`, then `neg_le_neg`, with `natAbs_neg` and `neg_neg` rewriting the two ends.
	valDecl{"thm", "Int.le_natAbs", nil,
		"(a : Int) → (Int.le a (Int.ofNat (Int.natAbs a)))",
		"fun (a : Int) => Int.rec.{0} (fun (x : Int) => (Int.le x (Int.ofNat (Int.natAbs x)))) (fun (k : Nat) => Int.le_refl (Int.ofNat k)) (fun (k : " +
			"Nat) => Int.NonNeg.mk (Nat.add (Nat.succ k) (Nat.succ k))) a"},
	valDecl{"thm", "Int.neg_natAbs_le", nil,
		"(a : Int) → (Int.le (Int.neg (Int.ofNat (Int.natAbs a))) a)",
		"fun (a : Int) => Eq.rec.{0, 1} Int (Int.neg (Int.neg a)) (fun (w : Int) (_ : Eq.{1} Int (Int.neg (Int.neg a)) w) => (Int.le (Int.neg " +
			"(Int.ofNat (Int.natAbs a))) w)) (Eq.rec.{0, 1} Nat (Int.natAbs (Int.neg a)) (fun (w : Nat) (_ : Eq.{1} Nat (Int.natAbs (Int.neg a)) w) => " +
			"(Int.le (Int.neg (Int.ofNat w)) (Int.neg (Int.neg a)))) (Int.neg_le_neg (Int.neg a) (Int.ofNat (Int.natAbs (Int.neg a))) (Int.le_natAbs " +
			"(Int.neg a))) (Int.natAbs a) (Int.natAbs_neg a)) a (Int.neg_neg a)"},
	// `0 <= c` gives `c * min a b = min (c * a) (c * b)` and the same for `max`, by `le_antisymm`: one side is `le_min` / `max_le` of
	// `mul_le_mul_of_nonneg_left`; the other splits on `decide (a <= b)`, rewrites `min a b` / `max a b` to `a` or `b`, and closes with `min_le_*` / `le_max_*`.
	valDecl{"thm", "Int.mul_min_of_nonneg", nil,
		"(c : Int) → (a : Int) → (b : Int) → (Int.le (Int.ofNat 0) c) → Eq.{1} Int (Int.mul c (Int.min a b)) (Int.min (Int.mul c a) (Int.mul c b))",
		"fun (c : Int) (a : Int) (b : Int) (hc : (Int.le (Int.ofNat 0) c)) => Int.le_antisymm (Int.mul c (Int.min a b)) (Int.min (Int.mul c a) " +
			"(Int.mul c b)) (Int.le_min (Int.mul c a) (Int.mul c b) (Int.mul c (Int.min a b)) (Int.mul_le_mul_of_nonneg_left (Int.min a b) a c " +
			"(Int.nonneg_of_le_zero c hc) (Int.min_le_left a b)) (Int.mul_le_mul_of_nonneg_left (Int.min a b) b c (Int.nonneg_of_le_zero c hc) " +
			"(Int.min_le_right a b))) (Decidable.rec.{0} (Int.le a b) (fun (_ : Decidable (Int.le a b)) => (Int.le (Int.min (Int.mul c a) (Int.mul c b)) " +
			"(Int.mul c (Int.min a b)))) (fun (h : Not (Int.le a b)) => Eq.rec.{0, 1} Int b (fun (w : Int) (_ : Eq.{1} Int b w) => (Int.le (Int.min " +
			"(Int.mul c a) (Int.mul c b)) (Int.mul c w))) (Int.min_le_right (Int.mul c a) (Int.mul c b)) (Int.min a b) (Eq.symm.{1} Int (Int.min a b) b " +
			"(Int.min_eq_right_of_not_le a b h))) (fun (h : (Int.le a b)) => Eq.rec.{0, 1} Int a (fun (w : Int) (_ : Eq.{1} Int a w) => (Int.le (Int.min " +
			"(Int.mul c a) (Int.mul c b)) (Int.mul c w))) (Int.min_le_left (Int.mul c a) (Int.mul c b)) (Int.min a b) (Eq.symm.{1} Int (Int.min a b) a " +
			"(Int.min_eq_left a b h))) (Int.decLe a b))"},
	valDecl{"thm", "Int.mul_max_of_nonneg", nil,
		"(c : Int) → (a : Int) → (b : Int) → (Int.le (Int.ofNat 0) c) → Eq.{1} Int (Int.mul c (Int.max a b)) (Int.max (Int.mul c a) (Int.mul c b))",
		"fun (c : Int) (a : Int) (b : Int) (hc : (Int.le (Int.ofNat 0) c)) => Int.le_antisymm (Int.mul c (Int.max a b)) (Int.max (Int.mul c a) " +
			"(Int.mul c b)) (Decidable.rec.{0} (Int.le a b) (fun (_ : Decidable (Int.le a b)) => (Int.le (Int.mul c (Int.max a b)) (Int.max (Int.mul c a) " +
			"(Int.mul c b)))) (fun (h : Not (Int.le a b)) => Eq.rec.{0, 1} Int a (fun (w : Int) (_ : Eq.{1} Int a w) => (Int.le (Int.mul c w) (Int.max " +
			"(Int.mul c a) (Int.mul c b)))) (Int.le_max_left (Int.mul c a) (Int.mul c b)) (Int.max a b) (Eq.symm.{1} Int (Int.max a b) a " +
			"(Int.max_eq_left_of_not_le a b h))) (fun (h : (Int.le a b)) => Eq.rec.{0, 1} Int b (fun (w : Int) (_ : Eq.{1} Int b w) => (Int.le (Int.mul c " +
			"w) (Int.max (Int.mul c a) (Int.mul c b)))) (Int.le_max_right (Int.mul c a) (Int.mul c b)) (Int.max a b) (Eq.symm.{1} Int (Int.max a b) b " +
			"(Int.max_eq_right a b h))) (Int.decLe a b)) (Int.max_le (Int.mul c a) (Int.mul c b) (Int.mul c (Int.max a b)) (Int.mul_le_mul_of_nonneg_left " +
			"a (Int.max a b) c (Int.nonneg_of_le_zero c hc) (Int.le_max_left a b)) (Int.mul_le_mul_of_nonneg_left b (Int.max a b) c " +
			"(Int.nonneg_of_le_zero c hc) (Int.le_max_right a b)))"},
	// Sign rules from the order lemmas; `0 + 0` and `-0` reduce to `0`, so `add_le_add` / `add_lt_add` / `neg_le_neg` / `neg_lt_neg` apply at `0` as they are.
	valDecl{"thm", "Int.add_nonneg", nil,
		"(a : Int) → (b : Int) → (Int.le (Int.ofNat 0) a) → (Int.le (Int.ofNat 0) b) → (Int.le (Int.ofNat 0) (Int.add a b))",
		"fun (a : Int) (b : Int) (ha : (Int.le (Int.ofNat 0) a)) (hb : (Int.le (Int.ofNat 0) b)) => Int.add_le_add (Int.ofNat 0) a (Int.ofNat 0) b ha " +
			"hb"},
	valDecl{"thm", "Int.add_pos", nil,
		"(a : Int) → (b : Int) → (Int.lt (Int.ofNat 0) a) → (Int.lt (Int.ofNat 0) b) → (Int.lt (Int.ofNat 0) (Int.add a b))",
		"fun (a : Int) (b : Int) (ha : (Int.lt (Int.ofNat 0) a)) (hb : (Int.lt (Int.ofNat 0) b)) => Int.add_lt_add (Int.ofNat 0) a (Int.ofNat 0) b ha " +
			"hb"},
	valDecl{"thm", "Int.neg_nonneg_of_nonpos", nil,
		"(a : Int) → (Int.le a (Int.ofNat 0)) → (Int.le (Int.ofNat 0) (Int.neg a))",
		"fun (a : Int) (h : (Int.le a (Int.ofNat 0))) => Int.neg_le_neg a (Int.ofNat 0) h"},
	valDecl{"thm", "Int.neg_nonpos_of_nonneg", nil,
		"(a : Int) → (Int.le (Int.ofNat 0) a) → (Int.le (Int.neg a) (Int.ofNat 0))",
		"fun (a : Int) (h : (Int.le (Int.ofNat 0) a)) => Int.neg_le_neg (Int.ofNat 0) a h"},
	valDecl{"thm", "Int.neg_pos_of_neg", nil,
		"(a : Int) → (Int.lt a (Int.ofNat 0)) → (Int.lt (Int.ofNat 0) (Int.neg a))",
		"fun (a : Int) (h : (Int.lt a (Int.ofNat 0))) => Int.neg_lt_neg a (Int.ofNat 0) h"},
	valDecl{"thm", "Int.neg_neg_of_pos", nil,
		"(a : Int) → (Int.lt (Int.ofNat 0) a) → (Int.lt (Int.neg a) (Int.ofNat 0))",
		"fun (a : Int) (h : (Int.lt (Int.ofNat 0) a)) => Int.neg_lt_neg (Int.ofNat 0) a h"},
	// Product signs. `0 <= (-a) * (-b) = a * b` (`neg_mul_neg`). For mixed signs, `0 <= a * -b = -(a * b)` (`mul_neg`, `neg_mul`), and `neg_le_neg` / `neg_lt_neg`
	// plus `neg_neg` flip it to `a * b <= 0` (`< 0` with `mul_pos`).
	valDecl{"thm", "Int.mul_nonneg_of_nonpos_of_nonpos", nil,
		"(a : Int) → (b : Int) → (Int.le a (Int.ofNat 0)) → (Int.le b (Int.ofNat 0)) → (Int.le (Int.ofNat 0) (Int.mul a b))",
		"fun (a : Int) (b : Int) (ha : (Int.le a (Int.ofNat 0))) (hb : (Int.le b (Int.ofNat 0))) => Eq.rec.{0, 1} Int (Int.mul (Int.neg a) (Int.neg " +
			"b)) (fun (w : Int) (_ : Eq.{1} Int (Int.mul (Int.neg a) (Int.neg b)) w) => (Int.le (Int.ofNat 0) w)) (Int.mul_nonneg (Int.neg a) (Int.neg b) " +
			"(Int.neg_nonneg_of_nonpos a ha) (Int.neg_nonneg_of_nonpos b hb)) (Int.mul a b) (Int.neg_mul_neg a b)"},
	valDecl{"thm", "Int.mul_nonpos_of_nonneg_of_nonpos", nil,
		"(a : Int) → (b : Int) → (Int.le (Int.ofNat 0) a) → (Int.le b (Int.ofNat 0)) → (Int.le (Int.mul a b) (Int.ofNat 0))",
		"fun (a : Int) (b : Int) (ha : (Int.le (Int.ofNat 0) a)) (hb : (Int.le b (Int.ofNat 0))) => Eq.rec.{0, 1} Int (Int.neg (Int.neg (Int.mul a " +
			"b))) (fun (w : Int) (_ : Eq.{1} Int (Int.neg (Int.neg (Int.mul a b))) w) => (Int.le w (Int.ofNat 0))) (Int.neg_le_neg (Int.ofNat 0) (Int.neg " +
			"(Int.mul a b)) (Eq.rec.{0, 1} Int (Int.mul a (Int.neg b)) (fun (w : Int) (_ : Eq.{1} Int (Int.mul a (Int.neg b)) w) => (Int.le (Int.ofNat 0) " +
			"w)) (Int.mul_nonneg a (Int.neg b) ha (Int.neg_nonneg_of_nonpos b hb)) (Int.neg (Int.mul a b)) (Int.mul_neg a b))) (Int.mul a b) (Int.neg_neg " +
			"(Int.mul a b))"},
	valDecl{"thm", "Int.mul_nonpos_of_nonpos_of_nonneg", nil,
		"(a : Int) → (b : Int) → (Int.le a (Int.ofNat 0)) → (Int.le (Int.ofNat 0) b) → (Int.le (Int.mul a b) (Int.ofNat 0))",
		"fun (a : Int) (b : Int) (ha : (Int.le a (Int.ofNat 0))) (hb : (Int.le (Int.ofNat 0) b)) => Eq.rec.{0, 1} Int (Int.neg (Int.neg (Int.mul a " +
			"b))) (fun (w : Int) (_ : Eq.{1} Int (Int.neg (Int.neg (Int.mul a b))) w) => (Int.le w (Int.ofNat 0))) (Int.neg_le_neg (Int.ofNat 0) (Int.neg " +
			"(Int.mul a b)) (Eq.rec.{0, 1} Int (Int.mul (Int.neg a) b) (fun (w : Int) (_ : Eq.{1} Int (Int.mul (Int.neg a) b) w) => (Int.le (Int.ofNat 0) " +
			"w)) (Int.mul_nonneg (Int.neg a) b (Int.neg_nonneg_of_nonpos a ha) hb) (Int.neg (Int.mul a b)) (Int.neg_mul a b))) (Int.mul a b) (Int.neg_neg " +
			"(Int.mul a b))"},
	valDecl{"thm", "Int.mul_neg_of_pos_of_neg", nil,
		"(a : Int) → (b : Int) → (Int.lt (Int.ofNat 0) a) → (Int.lt b (Int.ofNat 0)) → (Int.lt (Int.mul a b) (Int.ofNat 0))",
		"fun (a : Int) (b : Int) (ha : (Int.lt (Int.ofNat 0) a)) (hb : (Int.lt b (Int.ofNat 0))) => Eq.rec.{0, 1} Int (Int.neg (Int.neg (Int.mul a " +
			"b))) (fun (w : Int) (_ : Eq.{1} Int (Int.neg (Int.neg (Int.mul a b))) w) => (Int.lt w (Int.ofNat 0))) (Int.neg_lt_neg (Int.ofNat 0) (Int.neg " +
			"(Int.mul a b)) (Eq.rec.{0, 1} Int (Int.mul a (Int.neg b)) (fun (w : Int) (_ : Eq.{1} Int (Int.mul a (Int.neg b)) w) => (Int.lt (Int.ofNat 0) " +
			"w)) (Int.mul_pos a (Int.neg b) ha (Int.neg_pos_of_neg b hb)) (Int.neg (Int.mul a b)) (Int.mul_neg a b))) (Int.mul a b) (Int.neg_neg (Int.mul " +
			"a b))"},
	// Strict order: `lt_asymm`, `ne_of_lt`, `le_of_not_lt` (`le_total` and `eq_or_lt_of_le`), `lt_trichotomy` (`le_total` then `eq_or_lt_of_le` on each side).
	// Strict products `0 < a * b` (both negative, `mul_pos` on the negations) and `a * b < 0` (negative times positive), as the non-strict ones above.
	valDecl{"thm", "Int.lt_asymm", nil,
		"(a : Int) → (b : Int) → (Int.lt a b) → Not (Int.lt b a)",
		"fun (a : Int) (b : Int) (h : (Int.lt a b)) => Int.not_lt_of_le b a (Int.le_of_lt a b h)"},
	valDecl{"thm", "Int.ne_of_lt", nil,
		"(a : Int) → (b : Int) → (Int.lt a b) → Not (Eq.{1} Int a b)",
		"fun (a : Int) (b : Int) (h : (Int.lt a b)) (e : (Eq.{1} Int a b)) => Int.lt_irrefl a (Eq.rec.{0, 1} Int b (fun (w : Int) (_ : (Eq.{1} Int b " +
			"w)) => (Int.lt a w)) h a (Eq.symm.{1} Int a b e))"},
	valDecl{"thm", "Int.le_of_not_lt", nil,
		"(a : Int) → (b : Int) → Not (Int.lt a b) → (Int.le b a)",
		"fun (a : Int) (b : Int) (hn : Not (Int.lt a b)) => Or.rec (Int.le b a) (Int.le a b) (fun (_ : (Or (Int.le b a) (Int.le a b))) => (Int.le b " +
			"a)) (fun (h : (Int.le b a)) => h) (fun (h : (Int.le a b)) => Or.rec (Eq.{1} Int a b) (Int.lt a b) (fun (_ : (Or (Eq.{1} Int a b) (Int.lt a " +
			"b))) => (Int.le b a)) (fun (e : (Eq.{1} Int a b)) => Eq.rec.{0, 1} Int a (fun (w : Int) (_ : (Eq.{1} Int a w)) => (Int.le w a)) (Int.le_refl " +
			"a) b e) (fun (l : (Int.lt a b)) => False.elim.{0} (Int.le b a) (hn l)) (Int.eq_or_lt_of_le a b h)) (Int.le_total b a)"},
	valDecl{"thm", "Int.lt_trichotomy", nil,
		"(a : Int) → (b : Int) → (Or (Int.lt a b) (Or (Eq.{1} Int a b) (Int.lt b a)))",
		"fun (a : Int) (b : Int) => Or.rec (Int.le a b) (Int.le b a) (fun (_ : (Or (Int.le a b) (Int.le b a))) => (Or (Int.lt a b) (Or (Eq.{1} Int a " +
			"b) (Int.lt b a)))) (fun (h : (Int.le a b)) => Or.rec (Eq.{1} Int a b) (Int.lt a b) (fun (_ : (Or (Eq.{1} Int a b) (Int.lt a b))) => (Or " +
			"(Int.lt a b) (Or (Eq.{1} Int a b) (Int.lt b a)))) (fun (e : (Eq.{1} Int a b)) => Or.inr (Int.lt a b) (Or (Eq.{1} Int a b) (Int.lt b a)) " +
			"(Or.inl (Eq.{1} Int a b) (Int.lt b a) e)) (fun (l : (Int.lt a b)) => Or.inl (Int.lt a b) (Or (Eq.{1} Int a b) (Int.lt b a)) l) " +
			"(Int.eq_or_lt_of_le a b h)) (fun (h : (Int.le b a)) => Or.rec (Eq.{1} Int b a) (Int.lt b a) (fun (_ : (Or (Eq.{1} Int b a) (Int.lt b a))) => " +
			"(Or (Int.lt a b) (Or (Eq.{1} Int a b) (Int.lt b a)))) (fun (e : (Eq.{1} Int b a)) => Or.inr (Int.lt a b) (Or (Eq.{1} Int a b) (Int.lt b a)) " +
			"(Or.inl (Eq.{1} Int a b) (Int.lt b a) (Eq.symm.{1} Int b a e))) (fun (l : (Int.lt b a)) => Or.inr (Int.lt a b) (Or (Eq.{1} Int a b) (Int.lt " +
			"b a)) (Or.inr (Eq.{1} Int a b) (Int.lt b a) l)) (Int.eq_or_lt_of_le b a h)) (Int.le_total a b)"},
	valDecl{"thm", "Int.mul_pos_of_neg_of_neg", nil,
		"(a : Int) → (b : Int) → (Int.lt a (Int.ofNat 0)) → (Int.lt b (Int.ofNat 0)) → (Int.lt (Int.ofNat 0) (Int.mul a b))",
		"fun (a : Int) (b : Int) (ha : (Int.lt a (Int.ofNat 0))) (hb : (Int.lt b (Int.ofNat 0))) => Eq.rec.{0, 1} Int (Int.mul (Int.neg a) (Int.neg " +
			"b)) (fun (w : Int) (_ : (Eq.{1} Int (Int.mul (Int.neg a) (Int.neg b)) w)) => (Int.lt (Int.ofNat 0) w)) (Int.mul_pos (Int.neg a) (Int.neg b) " +
			"(Int.neg_pos_of_neg a ha) (Int.neg_pos_of_neg b hb)) (Int.mul a b) (Int.neg_mul_neg a b)"},
	valDecl{"thm", "Int.mul_neg_of_neg_of_pos", nil,
		"(a : Int) → (b : Int) → (Int.lt a (Int.ofNat 0)) → (Int.lt (Int.ofNat 0) b) → (Int.lt (Int.mul a b) (Int.ofNat 0))",
		"fun (a : Int) (b : Int) (ha : (Int.lt a (Int.ofNat 0))) (hb : (Int.lt (Int.ofNat 0) b)) => Eq.rec.{0, 1} Int (Int.neg (Int.neg (Int.mul a " +
			"b))) (fun (w : Int) (_ : (Eq.{1} Int (Int.neg (Int.neg (Int.mul a b))) w)) => (Int.lt w (Int.ofNat 0))) (Int.neg_lt_neg (Int.ofNat 0) " +
			"(Int.neg (Int.mul a b)) (Eq.rec.{0, 1} Int (Int.mul (Int.neg a) b) (fun (w : Int) (_ : (Eq.{1} Int (Int.mul (Int.neg a) b) w)) => (Int.lt " +
			"(Int.ofNat 0) w)) (Int.mul_pos (Int.neg a) b (Int.neg_pos_of_neg a ha) hb) (Int.neg (Int.mul a b)) (Int.neg_mul a b))) (Int.mul a b) " +
			"(Int.neg_neg (Int.mul a b))"},
	// From `mul_eq_zero`: `a * b != 0` for nonzero factors, `a * a = 0 -> a = 0`. `0 < a * a` for `a != 0`: `lt_trichotomy a 0` leaves `a < 0` (`mul_pos_of_neg_of_neg`),
	// `a = 0` (contradiction) or `0 < a` (`mul_pos`).
	valDecl{"thm", "Int.mul_ne_zero", nil,
		"(a : Int) → (b : Int) → Not (Eq.{1} Int a (Int.ofNat 0)) → Not (Eq.{1} Int b (Int.ofNat 0)) → Not (Eq.{1} Int (Int.mul a b) (Int.ofNat 0))",
		"fun (a : Int) (b : Int) (ha : Not (Eq.{1} Int a (Int.ofNat 0))) (hb : Not (Eq.{1} Int b (Int.ofNat 0))) (h : (Eq.{1} Int (Int.mul a b) " +
			"(Int.ofNat 0))) => Or.rec (Eq.{1} Int a (Int.ofNat 0)) (Eq.{1} Int b (Int.ofNat 0)) (fun (_ : (Or (Eq.{1} Int a (Int.ofNat 0)) (Eq.{1} Int b " +
			"(Int.ofNat 0)))) => False) (fun (e : (Eq.{1} Int a (Int.ofNat 0))) => ha e) (fun (e : (Eq.{1} Int b (Int.ofNat 0))) => hb e) " +
			"(Int.mul_eq_zero a b h)"},
	valDecl{"thm", "Int.mul_self_eq_zero", nil,
		"(a : Int) → (Eq.{1} Int (Int.mul a a) (Int.ofNat 0)) → (Eq.{1} Int a (Int.ofNat 0))",
		"fun (a : Int) (h : (Eq.{1} Int (Int.mul a a) (Int.ofNat 0))) => Or.rec (Eq.{1} Int a (Int.ofNat 0)) (Eq.{1} Int a (Int.ofNat 0)) (fun (_ : " +
			"(Or (Eq.{1} Int a (Int.ofNat 0)) (Eq.{1} Int a (Int.ofNat 0)))) => (Eq.{1} Int a (Int.ofNat 0))) (fun (e : (Eq.{1} Int a (Int.ofNat 0))) => " +
			"e) (fun (e : (Eq.{1} Int a (Int.ofNat 0))) => e) (Int.mul_eq_zero a a h)"},
	valDecl{"thm", "Int.mul_self_pos", nil,
		"(a : Int) → Not (Eq.{1} Int a (Int.ofNat 0)) → (Int.lt (Int.ofNat 0) (Int.mul a a))",
		"fun (a : Int) (ha : Not (Eq.{1} Int a (Int.ofNat 0))) => Or.rec (Int.lt a (Int.ofNat 0)) (Or (Eq.{1} Int a (Int.ofNat 0)) (Int.lt (Int.ofNat " +
			"0) a)) (fun (_ : (Or (Int.lt a (Int.ofNat 0)) (Or (Eq.{1} Int a (Int.ofNat 0)) (Int.lt (Int.ofNat 0) a)))) => (Int.lt (Int.ofNat 0) (Int.mul " +
			"a a))) (fun (h : (Int.lt a (Int.ofNat 0))) => Int.mul_pos_of_neg_of_neg a a h h) (fun (h2 : (Or (Eq.{1} Int a (Int.ofNat 0)) (Int.lt " +
			"(Int.ofNat 0) a))) => Or.rec (Eq.{1} Int a (Int.ofNat 0)) (Int.lt (Int.ofNat 0) a) (fun (_ : (Or (Eq.{1} Int a (Int.ofNat 0)) (Int.lt " +
			"(Int.ofNat 0) a))) => (Int.lt (Int.ofNat 0) (Int.mul a a))) (fun (e : (Eq.{1} Int a (Int.ofNat 0))) => False.elim.{0} (Int.lt (Int.ofNat 0) " +
			"(Int.mul a a)) (ha e)) (fun (h : (Int.lt (Int.ofNat 0) a)) => Int.mul_pos a a h h) h2) (Int.lt_trichotomy a (Int.ofNat 0))"},
	// `0 < a - b <-> b < a`, `a - b < 0 <-> a < b`, `a - b <= 0 <-> a <= b`: add `b` (`add_lt_add_right`) and use `sub_add_cancel` / `zero_add`; back, add `-b` and use `add_neg_cancel`.
	valDecl{"thm", "Int.sub_pos", nil,
		"(a : Int) → (b : Int) → Iff (Int.lt (Int.ofNat 0) (Int.sub a b)) (Int.lt b a)",
		"fun (a : Int) (b : Int) => Iff.intro (Int.lt (Int.ofNat 0) (Int.sub a b)) (Int.lt b a) (fun (h : (Int.lt (Int.ofNat 0) (Int.sub a b))) => " +
			"Eq.rec.{0, 1} Int (Int.add (Int.sub a b) b) (fun (w : Int) (_ : (Eq.{1} Int (Int.add (Int.sub a b) b) w)) => (Int.lt b w)) (Eq.rec.{0, 1} " +
			"Int (Int.add (Int.ofNat 0) b) (fun (w : Int) (_ : (Eq.{1} Int (Int.add (Int.ofNat 0) b) w)) => (Int.lt w (Int.add (Int.sub a b) b))) " +
			"(Int.add_lt_add_right (Int.ofNat 0) (Int.sub a b) b h) b (Int.zero_add b)) a (Int.sub_add_cancel a b)) (fun (h : (Int.lt b a)) => Eq.rec.{0, " +
			"1} Int (Int.add b (Int.neg b)) (fun (w : Int) (_ : (Eq.{1} Int (Int.add b (Int.neg b)) w)) => (Int.lt w (Int.add a (Int.neg b)))) " +
			"(Int.add_lt_add_right b a (Int.neg b) h) (Int.ofNat 0) (Int.add_neg_cancel b))"},
	valDecl{"thm", "Int.sub_neg", nil,
		"(a : Int) → (b : Int) → Iff (Int.lt (Int.sub a b) (Int.ofNat 0)) (Int.lt a b)",
		"fun (a : Int) (b : Int) => Iff.intro (Int.lt (Int.sub a b) (Int.ofNat 0)) (Int.lt a b) (fun (h : (Int.lt (Int.sub a b) (Int.ofNat 0))) => " +
			"Eq.rec.{0, 1} Int (Int.add (Int.ofNat 0) b) (fun (w : Int) (_ : (Eq.{1} Int (Int.add (Int.ofNat 0) b) w)) => (Int.lt a w)) (Eq.rec.{0, 1} " +
			"Int (Int.add (Int.sub a b) b) (fun (w : Int) (_ : (Eq.{1} Int (Int.add (Int.sub a b) b) w)) => (Int.lt w (Int.add (Int.ofNat 0) b))) " +
			"(Int.add_lt_add_right (Int.sub a b) (Int.ofNat 0) b h) a (Int.sub_add_cancel a b)) b (Int.zero_add b)) (fun (h : (Int.lt a b)) => Eq.rec.{0, " +
			"1} Int (Int.add b (Int.neg b)) (fun (w : Int) (_ : (Eq.{1} Int (Int.add b (Int.neg b)) w)) => (Int.lt (Int.add a (Int.neg b)) w)) " +
			"(Int.add_lt_add_right a b (Int.neg b) h) (Int.ofNat 0) (Int.add_neg_cancel b))"},
	valDecl{"thm", "Int.sub_nonpos", nil,
		"(a : Int) → (b : Int) → Iff (Int.le (Int.sub a b) (Int.ofNat 0)) (Int.le a b)",
		"fun (a : Int) (b : Int) => Iff.intro (Int.le (Int.sub a b) (Int.ofNat 0)) (Int.le a b) (fun (h : (Int.le (Int.sub a b) (Int.ofNat 0))) => " +
			"Eq.rec.{0, 1} Int (Int.add (Int.ofNat 0) b) (fun (w : Int) (_ : (Eq.{1} Int (Int.add (Int.ofNat 0) b) w)) => (Int.le a w)) (Eq.rec.{0, 1} " +
			"Int (Int.add (Int.sub a b) b) (fun (w : Int) (_ : (Eq.{1} Int (Int.add (Int.sub a b) b) w)) => (Int.le w (Int.add (Int.ofNat 0) b))) " +
			"(Int.add_le_add_right (Int.sub a b) (Int.ofNat 0) b h) a (Int.sub_add_cancel a b)) b (Int.zero_add b)) (fun (h : (Int.le a b)) => Eq.rec.{0, " +
			"1} Int (Int.add b (Int.neg b)) (fun (w : Int) (_ : (Eq.{1} Int (Int.add b (Int.neg b)) w)) => (Int.le (Int.add a (Int.neg b)) w)) " +
			"(Int.add_le_add_right a b (Int.neg b) h) (Int.ofNat 0) (Int.add_neg_cancel b))"},
	// Adding a nonneg / positive term: `a <= a + b`, `a < a + b` (and with `b + a`) by `add_le_add_*` / `add_lt_add_*` at `0` and `add_zero` / `zero_add`; `a - b <= a`, `a - b < a`
	// the same way through `-b <= 0` / `-b < 0`. Mixed sums: `add_lt_add_of_le_of_lt` and `_of_lt_of_le` chain one strict and one non-strict step with `lt_of_lt_of_le`.
	valDecl{"thm", "Int.le_add_of_nonneg_right", nil,
		"(a : Int) → (b : Int) → (Int.le (Int.ofNat 0) b) → (Int.le a (Int.add a b))",
		"fun (a : Int) (b : Int) (h : (Int.le (Int.ofNat 0) b)) => Eq.rec.{0, 1} Int (Int.add a (Int.ofNat 0)) (fun (w : Int) (_ : (Eq.{1} Int " +
			"(Int.add a (Int.ofNat 0)) w)) => (Int.le w (Int.add a b))) (Int.add_le_add_left (Int.ofNat 0) b a h) a (Int.add_zero a)"},
	valDecl{"thm", "Int.le_add_of_nonneg_left", nil,
		"(a : Int) → (b : Int) → (Int.le (Int.ofNat 0) b) → (Int.le a (Int.add b a))",
		"fun (a : Int) (b : Int) (h : (Int.le (Int.ofNat 0) b)) => Eq.rec.{0, 1} Int (Int.add (Int.ofNat 0) a) (fun (w : Int) (_ : (Eq.{1} Int " +
			"(Int.add (Int.ofNat 0) a) w)) => (Int.le w (Int.add b a))) (Int.add_le_add_right (Int.ofNat 0) b a h) a (Int.zero_add a)"},
	valDecl{"thm", "Int.lt_add_of_pos_right", nil,
		"(a : Int) → (b : Int) → (Int.lt (Int.ofNat 0) b) → (Int.lt a (Int.add a b))",
		"fun (a : Int) (b : Int) (h : (Int.lt (Int.ofNat 0) b)) => Eq.rec.{0, 1} Int (Int.add a (Int.ofNat 0)) (fun (w : Int) (_ : (Eq.{1} Int " +
			"(Int.add a (Int.ofNat 0)) w)) => (Int.lt w (Int.add a b))) (Int.add_lt_add_left (Int.ofNat 0) b a h) a (Int.add_zero a)"},
	valDecl{"thm", "Int.lt_add_of_pos_left", nil,
		"(a : Int) → (b : Int) → (Int.lt (Int.ofNat 0) b) → (Int.lt a (Int.add b a))",
		"fun (a : Int) (b : Int) (h : (Int.lt (Int.ofNat 0) b)) => Eq.rec.{0, 1} Int (Int.add (Int.ofNat 0) a) (fun (w : Int) (_ : (Eq.{1} Int " +
			"(Int.add (Int.ofNat 0) a) w)) => (Int.lt w (Int.add b a))) (Int.add_lt_add_right (Int.ofNat 0) b a h) a (Int.zero_add a)"},
	valDecl{"thm", "Int.sub_le_self", nil,
		"(a : Int) → (b : Int) → (Int.le (Int.ofNat 0) b) → (Int.le (Int.sub a b) a)",
		"fun (a : Int) (b : Int) (h : (Int.le (Int.ofNat 0) b)) => Eq.rec.{0, 1} Int (Int.add a (Int.ofNat 0)) (fun (w : Int) (_ : (Eq.{1} Int " +
			"(Int.add a (Int.ofNat 0)) w)) => (Int.le (Int.sub a b) w)) (Int.add_le_add_left (Int.neg b) (Int.ofNat 0) a (Int.neg_nonpos_of_nonneg b h)) " +
			"a (Int.add_zero a)"},
	valDecl{"thm", "Int.sub_lt_self", nil,
		"(a : Int) → (b : Int) → (Int.lt (Int.ofNat 0) b) → (Int.lt (Int.sub a b) a)",
		"fun (a : Int) (b : Int) (h : (Int.lt (Int.ofNat 0) b)) => Eq.rec.{0, 1} Int (Int.add a (Int.ofNat 0)) (fun (w : Int) (_ : (Eq.{1} Int " +
			"(Int.add a (Int.ofNat 0)) w)) => (Int.lt (Int.sub a b) w)) (Int.add_lt_add_left (Int.neg b) (Int.ofNat 0) a (Int.neg_neg_of_pos b h)) a " +
			"(Int.add_zero a)"},
	valDecl{"thm", "Int.add_lt_add_of_le_of_lt", nil,
		"(a : Int) → (b : Int) → (c : Int) → (d : Int) → (Int.le a b) → (Int.lt c d) → (Int.lt (Int.add a c) (Int.add b d))",
		"fun (a : Int) (b : Int) (c : Int) (d : Int) (h1 : (Int.le a b)) (h2 : (Int.lt c d)) => Int.lt_of_lt_of_le (Int.add a c) (Int.add a d) " +
			"(Int.add b d) (Int.add_lt_add_left c d a h2) (Int.add_le_add_right a b d h1)"},
	valDecl{"thm", "Int.add_lt_add_of_lt_of_le", nil,
		"(a : Int) → (b : Int) → (c : Int) → (d : Int) → (Int.lt a b) → (Int.le c d) → (Int.lt (Int.add a c) (Int.add b d))",
		"fun (a : Int) (b : Int) (c : Int) (d : Int) (h1 : (Int.lt a b)) (h2 : (Int.le c d)) => Int.lt_of_lt_of_le (Int.add a c) (Int.add b c) " +
			"(Int.add b d) (Int.add_lt_add_right a b c h1) (Int.add_le_add_left c d b h2)"},
	// `0 <= a` gives `ofNat (natAbs a) = a`: `ofNat k` is `refl`, `negSucc k` contradicts `not_nonneg_negSucc`. For `a <= 0` apply it to `-a` and rewrite with `natAbs_neg`.
	valDecl{"thm", "Int.ofNat_natAbs_of_nonneg", nil,
		"(a : Int) → (Int.le (Int.ofNat 0) a) → (Eq.{1} Int (Int.ofNat (Int.natAbs a)) a)",
		"fun (a : Int) => Int.rec.{0} (fun (x : Int) => (Int.le (Int.ofNat 0) x) → (Eq.{1} Int (Int.ofNat (Int.natAbs x)) x)) (fun (k : Nat) (_ : " +
			"(Int.le (Int.ofNat 0) (Int.ofNat k))) => Eq.refl.{1} Int (Int.ofNat k)) (fun (k : Nat) (h : (Int.le (Int.ofNat 0) (Int.negSucc k))) => " +
			"False.elim.{0} (Eq.{1} Int (Int.ofNat (Int.natAbs (Int.negSucc k))) (Int.negSucc k)) (Int.not_nonneg_negSucc k (Int.nonneg_of_le_zero " +
			"(Int.negSucc k) h))) a"},
	valDecl{"thm", "Int.ofNat_natAbs_of_nonpos", nil,
		"(a : Int) → (Int.le a (Int.ofNat 0)) → (Eq.{1} Int (Int.ofNat (Int.natAbs a)) (Int.neg a))",
		"fun (a : Int) (h : (Int.le a (Int.ofNat 0))) => Eq.rec.{0, 1} Nat (Int.natAbs (Int.neg a)) (fun (w : Nat) (_ : (Eq.{1} Nat (Int.natAbs " +
			"(Int.neg a)) w)) => (Eq.{1} Int (Int.ofNat w) (Int.neg a))) (Int.ofNat_natAbs_of_nonneg (Int.neg a) (Int.neg_nonneg_of_nonpos a h)) " +
			"(Int.natAbs a) (Int.natAbs_neg a)"},
	// `a = natAbs a` or `a = -natAbs a`: `le_total 0 a`, then the two `ofNat_natAbs_of_*` lemmas (the second after `neg_neg`).
	valDecl{"thm", "Int.natAbs_eq", nil,
		"(a : Int) → (Or (Eq.{1} Int a (Int.ofNat (Int.natAbs a))) (Eq.{1} Int a (Int.neg (Int.ofNat (Int.natAbs a)))))",
		"fun (a : Int) => Or.rec (Int.le (Int.ofNat 0) a) (Int.le a (Int.ofNat 0)) (fun (_ : (Or (Int.le (Int.ofNat 0) a) (Int.le a (Int.ofNat 0)))) " +
			"=> (Or (Eq.{1} Int a (Int.ofNat (Int.natAbs a))) (Eq.{1} Int a (Int.neg (Int.ofNat (Int.natAbs a)))))) (fun (h : (Int.le (Int.ofNat 0) a)) " +
			"=> Or.inl (Eq.{1} Int a (Int.ofNat (Int.natAbs a))) (Eq.{1} Int a (Int.neg (Int.ofNat (Int.natAbs a)))) (Eq.symm.{1} Int (Int.ofNat " +
			"(Int.natAbs a)) a (Int.ofNat_natAbs_of_nonneg a h))) (fun (h : (Int.le a (Int.ofNat 0))) => Or.inr (Eq.{1} Int a (Int.ofNat (Int.natAbs a))) " +
			"(Eq.{1} Int a (Int.neg (Int.ofNat (Int.natAbs a)))) (Eq.rec.{0, 1} Int (Int.neg a) (fun (w : Int) (_ : (Eq.{1} Int (Int.neg a) w)) => " +
			"(Eq.{1} Int a (Int.neg w))) (Eq.symm.{1} Int (Int.neg (Int.neg a)) a (Int.neg_neg a)) (Int.ofNat (Int.natAbs a)) (Eq.symm.{1} Int (Int.ofNat " +
			"(Int.natAbs a)) (Int.neg a) (Int.ofNat_natAbs_of_nonpos a h)))) (Int.le_total (Int.ofNat 0) a)"},
	// `-a <= natAbs a` (`le_natAbs (-a)` with `natAbs_neg`). Triangle inequality `natAbs (a + b) <= natAbs a + natAbs b`: in `Int`, `a + b <= |a| + |b|` if `0 <= a + b`,
	// and `-(a + b) = -a + -b <= |a| + |b|` if `a + b <= 0`; `ofNat_le` moves it to `Nat`.
	valDecl{"thm", "Int.neg_le_natAbs", nil,
		"(a : Int) → (Int.le (Int.neg a) (Int.ofNat (Int.natAbs a)))",
		"fun (a : Int) => Eq.rec.{0, 1} Nat (Int.natAbs (Int.neg a)) (fun (w : Nat) (_ : (Eq.{1} Nat (Int.natAbs (Int.neg a)) w)) => (Int.le (Int.neg " +
			"a) (Int.ofNat w))) (Int.le_natAbs (Int.neg a)) (Int.natAbs a) (Int.natAbs_neg a)"},
	valDecl{"thm", "Int.natAbs_add_le", nil,
		"(a : Int) → (b : Int) → (Nat.le (Int.natAbs (Int.add a b)) (Nat.add (Int.natAbs a) (Int.natAbs b)))",
		"fun (a : Int) (b : Int) => Iff.mp (Int.le (Int.ofNat (Int.natAbs (Int.add a b))) (Int.ofNat (Nat.add (Int.natAbs a) (Int.natAbs b)))) " +
			"(Nat.le (Int.natAbs (Int.add a b)) (Nat.add (Int.natAbs a) (Int.natAbs b))) (Int.ofNat_le (Int.natAbs (Int.add a b)) (Nat.add (Int.natAbs a) " +
			"(Int.natAbs b))) (Or.rec (Int.le (Int.ofNat 0) (Int.add a b)) (Int.le (Int.add a b) (Int.ofNat 0)) (fun (_ : (Or (Int.le (Int.ofNat 0) " +
			"(Int.add a b)) (Int.le (Int.add a b) (Int.ofNat 0)))) => (Int.le (Int.ofNat (Int.natAbs (Int.add a b))) (Int.add (Int.ofNat (Int.natAbs a)) " +
			"(Int.ofNat (Int.natAbs b))))) (fun (hs : (Int.le (Int.ofNat 0) (Int.add a b))) => Eq.rec.{0, 1} Int (Int.add a b) (fun (w : Int) (_ : " +
			"(Eq.{1} Int (Int.add a b) w)) => (Int.le w (Int.add (Int.ofNat (Int.natAbs a)) (Int.ofNat (Int.natAbs b))))) (Int.add_le_add a (Int.ofNat " +
			"(Int.natAbs a)) b (Int.ofNat (Int.natAbs b)) (Int.le_natAbs a) (Int.le_natAbs b)) (Int.ofNat (Int.natAbs (Int.add a b))) (Eq.symm.{1} Int " +
			"(Int.ofNat (Int.natAbs (Int.add a b))) (Int.add a b) (Int.ofNat_natAbs_of_nonneg (Int.add a b) hs))) (fun (hs : (Int.le (Int.add a b) " +
			"(Int.ofNat 0))) => Eq.rec.{0, 1} Int (Int.neg (Int.add a b)) (fun (w : Int) (_ : (Eq.{1} Int (Int.neg (Int.add a b)) w)) => (Int.le w " +
			"(Int.add (Int.ofNat (Int.natAbs a)) (Int.ofNat (Int.natAbs b))))) (Eq.rec.{0, 1} Int (Int.add (Int.neg a) (Int.neg b)) (fun (w : Int) (_ : " +
			"(Eq.{1} Int (Int.add (Int.neg a) (Int.neg b)) w)) => (Int.le w (Int.add (Int.ofNat (Int.natAbs a)) (Int.ofNat (Int.natAbs b))))) " +
			"(Int.add_le_add (Int.neg a) (Int.ofNat (Int.natAbs a)) (Int.neg b) (Int.ofNat (Int.natAbs b)) (Int.neg_le_natAbs a) (Int.neg_le_natAbs b)) " +
			"(Int.neg (Int.add a b)) (Eq.symm.{1} Int (Int.neg (Int.add a b)) (Int.add (Int.neg a) (Int.neg b)) (Int.neg_add a b))) (Int.ofNat " +
			"(Int.natAbs (Int.add a b))) (Eq.symm.{1} Int (Int.ofNat (Int.natAbs (Int.add a b))) (Int.neg (Int.add a b)) (Int.ofNat_natAbs_of_nonpos " +
			"(Int.add a b) hs))) (Int.le_total (Int.ofNat 0) (Int.add a b)))"},
	// `natAbs (a - b) <= natAbs a + natAbs b`: `a - b` is `a + -b`, so `natAbs_add_le` with `natAbs_neg`. `natAbs (a - b) = natAbs (b - a)`: `natAbs_neg` and `neg_sub`.
	valDecl{"thm", "Int.natAbs_sub_le", nil,
		"(a : Int) → (b : Int) → (Nat.le (Int.natAbs (Int.sub a b)) (Nat.add (Int.natAbs a) (Int.natAbs b)))",
		"fun (a : Int) (b : Int) => Eq.rec.{0, 1} Nat (Int.natAbs (Int.neg b)) (fun (w : Nat) (_ : (Eq.{1} Nat (Int.natAbs (Int.neg b)) w)) => " +
			"(Nat.le (Int.natAbs (Int.sub a b)) (Nat.add (Int.natAbs a) w))) (Int.natAbs_add_le a (Int.neg b)) (Int.natAbs b) (Int.natAbs_neg b)"},
	valDecl{"thm", "Int.natAbs_sub_comm", nil,
		"(a : Int) → (b : Int) → (Eq.{1} Nat (Int.natAbs (Int.sub a b)) (Int.natAbs (Int.sub b a)))",
		"fun (a : Int) (b : Int) => Eq.trans.{1} Nat (Int.natAbs (Int.sub a b)) (Int.natAbs (Int.neg (Int.sub a b))) (Int.natAbs (Int.sub b a)) " +
			"(Eq.symm.{1} Nat (Int.natAbs (Int.neg (Int.sub a b))) (Int.natAbs (Int.sub a b)) (Int.natAbs_neg (Int.sub a b))) (congrArg.{1, 1} Int Nat " +
			"Int.natAbs (Int.neg (Int.sub a b)) (Int.sub b a) (Int.neg_sub a b))"},
	// Lean's `Int.gcd_dvd_left`/`right` (`Init/Data/Int/Gcd.lean`: `gcd_dvd_left (a b) : ↑(gcd a b) ∣ a`). No new
	// multiplication lemma is needed: `Int.natAbs_dvd_natAbs` already turns Nat divisibility into `Int.dvd` of the same
	// two values, and `natAbs (ofNat (gcd a b))` is `gcd a b` by definition, so `Nat.gcd_dvd_left` is what is applied.
	valDecl{"thm", "Int.gcd_dvd_left", nil, "(a : Int) → (b : Int) → Int.dvd (Int.ofNat (Int.gcd a b)) a",
		"fun (a : Int) (b : Int) => Iff.mp (Nat.dvd (Int.natAbs (Int.ofNat (Int.gcd a b))) (Int.natAbs a)) " +
			"(Int.dvd (Int.ofNat (Int.gcd a b)) a) (Int.natAbs_dvd_natAbs (Int.ofNat (Int.gcd a b)) a) " +
			"(Nat.gcd_dvd_left (Int.natAbs a) (Int.natAbs b))"},
	valDecl{"thm", "Int.gcd_dvd_right", nil, "(a : Int) → (b : Int) → Int.dvd (Int.ofNat (Int.gcd a b)) b",
		"fun (a : Int) (b : Int) => Iff.mp (Nat.dvd (Int.natAbs (Int.ofNat (Int.gcd a b))) (Int.natAbs b)) " +
			"(Int.dvd (Int.ofNat (Int.gcd a b)) b) (Int.natAbs_dvd_natAbs (Int.ofNat (Int.gcd a b)) b) " +
			"(Nat.gcd_dvd_right (Int.natAbs a) (Int.natAbs b))"},
	valDecl{"thm", "Int.gcd_dvd", nil,
		"(a : Int) → (b : Int) → And (Int.dvd (Int.ofNat (Int.gcd a b)) a) (Int.dvd (Int.ofNat (Int.gcd a b)) b)",
		"fun (a : Int) (b : Int) => And.intro (Int.dvd (Int.ofNat (Int.gcd a b)) a) " +
			"(Int.dvd (Int.ofNat (Int.gcd a b)) b) (Int.gcd_dvd_left a b) (Int.gcd_dvd_right a b)"},

	valDecl{"def", "Int.decDvd", nil, "(a : Int) → (b : Int) → Decidable (Int.dvd a b)",
		"fun (a : Int) (b : Int) => Decidable.rec.{1} (Nat.dvd (Int.natAbs a) (Int.natAbs b)) " +
			"(fun (_ : Decidable (Nat.dvd (Int.natAbs a) (Int.natAbs b))) => Decidable (Int.dvd a b)) " +
			"(fun (h : Not (Nat.dvd (Int.natAbs a) (Int.natAbs b))) => Decidable.isFalse (Int.dvd a b) " +
			"(fun (hd : Int.dvd a b) => h (Iff.mpr (Nat.dvd (Int.natAbs a) (Int.natAbs b)) (Int.dvd a b) " +
			"(Int.natAbs_dvd_natAbs a b) hd))) " +
			"(fun (h : Nat.dvd (Int.natAbs a) (Int.natAbs b)) => Decidable.isTrue (Int.dvd a b) " +
			"(Iff.mp (Nat.dvd (Int.natAbs a) (Int.natAbs b)) (Int.dvd a b) (Int.natAbs_dvd_natAbs a b) h)) " +
			"(Nat.decDvd (Int.natAbs a) (Int.natAbs b))"},
}

var quotDecls = []valDecl{
	{"quot", "Quot", []string{"u"}, "(α : Sort u) → (α → α → Prop) → Sort u", ""},
	{"quot", "Quot.mk", []string{"u"}, "(α : Sort u) → (r : α → α → Prop) → α → Quot.{u} α r", ""},
	{"quot", "Quot.lift", []string{"u", "v"},
		"(α : Sort u) → (r : α → α → Prop) → (β : Sort v) → (f : α → β) → ((a : α) → (b : α) → r a b → Eq.{v} β (f a) (f b)) → Quot.{u} α r → β", ""},
	{"quot", "Quot.ind", []string{"u"},
		"(α : Sort u) → (r : α → α → Prop) → (β : Quot.{u} α r → Prop) → ((a : α) → β (Quot.mk.{u} α r a)) → (q : Quot.{u} α r) → β q", ""},
}

// AddQuot declares the quotient primitives (quot.cpp). Eq must already exist.
func (env *Env) AddQuot() error {
	for _, q := range quotDecls {
		ty, err := ParseTerm(q.ty, q.lparams)
		if err != nil {
			return err
		}
		tc, err := env.checkHeader(q.name, q.lparams, ty)
		if err != nil {
			return err
		}
		_ = tc
		env.put(&Decl{Kind: DQuot, Name: q.name, LParams: q.lparams, Type: ty, QuotKind: q.name})
	}
	return nil
}

func (env *Env) addPrelude(d any) error {
	switch d := d.(type) {
	case indDecl:
		ty, err := ParseTerm(d.ty, d.lparams)
		if err != nil {
			return err
		}
		spec := InductiveSpec{Name: d.name, LParams: d.lparams, NumParams: d.nparams, Type: ty}
		for _, c := range d.ctors {
			ct, err := ParseTerm(c[1], d.lparams)
			if err != nil {
				return fmt.Errorf("%s: %w", c[0], err)
			}
			spec.Ctors = append(spec.Ctors, CtorSpec{c[0], ct})
		}
		return env.AddInductive(spec)
	case valDecl:
		ty, err := ParseTerm(d.ty, d.lparams)
		if err != nil {
			return fmt.Errorf("%s: %w", d.name, err)
		}
		if d.kind == "axiom" {
			return env.AddAxiom(d.name, d.lparams, ty)
		}
		val, err := ParseTerm(d.val, d.lparams)
		if err != nil {
			return fmt.Errorf("%s: %w", d.name, err)
		}
		switch d.kind {
		case "def":
			return env.AddDefn(d.name, d.lparams, ty, val)
		case "thm":
			return env.AddTheorem(d.name, d.lparams, ty, val)
		case "opaque":
			return env.AddOpaque(d.name, d.lparams, ty, val)
		}
	}
	return fmt.Errorf("bad prelude entry %v", d)
}

// BuildPrelude constructs a fresh environment: Eq/And/Or/Iff/Exists/Decidable, Nat with its operations, the decision
// procedures and `of_decide_eq_true`, all checked by this package's kernel.
func BuildPrelude() (*Env, error) {
	env := NewEnv()
	for _, d := range append(append([]any{}, preludeDecls...), preludeExtra...) {
		if err := env.addPrelude(d); err != nil {
			name := ""
			switch d := d.(type) {
			case indDecl:
				name = d.name
			case valDecl:
				name = d.name
			}
			return nil, fmt.Errorf("prelude %s: %w", name, err)
		}
	}
	if err := env.AddQuot(); err != nil {
		return nil, fmt.Errorf("prelude Quot: %w", err)
	}
	if got, want := strings.Join(env.Axioms(), ","), strings.Join(sortedCopy(PreludeAxioms), ","); got != want {
		return nil, fmt.Errorf("prelude axioms %s differ from the declared list %s", got, want)
	}
	// Every constant is a definition or a theorem: an opaque declaration would keep a body from the kernel, and the
	// prelude needs none (`Nat.gcd` was the last one, and it is now the countdown in `Nat.gcdIter`).
	for _, n := range env.order {
		if d := env.decls[n]; d != nil && d.Kind == DOpaque {
			return nil, fmt.Errorf("prelude %s is opaque", n)
		}
	}
	return env, nil
}

var (
	preludeOnce sync.Once
	preludeEnv  *Env
	preludeErr  error
)

// Prelude returns the shared, immutable prelude environment (Clone it before adding declarations).
func Prelude() (*Env, error) {
	preludeOnce.Do(func() { preludeEnv, preludeErr = BuildPrelude() })
	return preludeEnv, preludeErr
}
