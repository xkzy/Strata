package lean

import (
	"strings"
	"testing"
)

func checkTerm(t *testing.T, env *Env, lparams []string, src string) (*Expr, error) {
	t.Helper()
	e, err := ParseTerm(src, lparams)
	if err != nil {
		t.Fatalf("parse %q: %v", src, err)
	}
	return NewTC(env, lparams).Check(e)
}

func wantReject(t *testing.T, env *Env, lparams []string, src, why string) {
	t.Helper()
	if ty, err := checkTerm(t, env, lparams, src); err == nil {
		t.Errorf("kernel accepted %q (type %v); %s", src, ty, why)
	}
}

func wantAccept(t *testing.T, env *Env, lparams []string, src string) {
	t.Helper()
	if _, err := checkTerm(t, env, lparams, src); err != nil {
		t.Errorf("kernel rejected %q: %v", src, err)
	}
}

func TestKernelRejectsBadTerms(t *testing.T) {
	env := mustPrelude(t)
	// Eq.refl true is not a proof of true = false
	wantReject(t, env, nil, "(fun (h : Eq.{1} Bool Bool.true Bool.false) => h) (Eq.refl.{1} Bool Bool.true)", "wrong argument type")
	wantReject(t, env, nil, "Nat.succ Bool.true", "application type mismatch")
	wantReject(t, env, nil, "Nat.succ Nat.zero Nat.zero", "applying a non-function")
	wantReject(t, env, nil, "Eq.refl.{0} Nat 0", "Nat : Type is not Sort 0, so level 0 is a universe error")
	wantReject(t, env, nil, "Eq.refl Nat 0", "missing universe level")
	wantReject(t, env, nil, "Nat.zero.{1}", "constant without universe parameters")
	wantReject(t, env, nil, "Foo.bar", "unknown constant")
	wantReject(t, env, nil, "Sort u", "undeclared universe parameter")
	wantReject(t, env, nil, "fun (x : Nat) => x x", "self application")
	wantReject(t, env, nil, "(fun (h : Eq.{1} Nat 0 1) => h) (Eq.refl.{1} Nat 0)", "refl proves 0 = 0, not 0 = 1")
	wantReject(t, env, nil, "Nat.rec.{1} (fun (_ : Nat) => Nat) Nat.zero Nat.zero Nat.zero", "bad minor premise")
	wantReject(t, env, nil, "Or.rec.{1} Prop Prop (fun (_ : Or Prop Prop) => Nat)", "Or eliminates only into Prop")
	// False.rec is the only way to get anything from False: proving 0 = 1 from nothing must fail
	wantAccept(t, env, nil, "False.elim.{1} Nat")
}

func TestKernelAcceptsGoodTerms(t *testing.T) {
	env := mustPrelude(t)
	for _, src := range []string{
		"Nat.add 2 2", "Eq.refl.{1} Nat 4", "fun (x : Nat) => Nat.succ x",
		"(fun (x : Nat) => x) 3", "Eq.refl.{1} Nat (Nat.add 2 2)",
		"of_decide_eq_true (Eq.{1} Nat 4 4) (Nat.decEq 4 4) (Eq.refl.{1} Bool Bool.true)",
		"Eq.rec.{1, 1} Nat 0 (fun (x : Nat) (_ : Eq.{1} Nat 0 x) => Nat) 5 0 (Eq.refl.{1} Nat 0)",
	} {
		wantAccept(t, env, nil, src)
	}
}

func defEq(t *testing.T, env *Env, lparams []string, a, b string) bool {
	t.Helper()
	ea, err := ParseTerm(a, lparams)
	if err != nil {
		t.Fatal(err)
	}
	eb, err := ParseTerm(b, lparams)
	if err != nil {
		t.Fatal(err)
	}
	tc := NewTC(env, lparams)
	if _, err := tc.Check(ea); err != nil {
		t.Fatalf("%q: %v", a, err)
	}
	if _, err := tc.Check(eb); err != nil {
		t.Fatalf("%q: %v", b, err)
	}
	ok, err := tc.IsDefEq(ea, eb)
	if err != nil {
		t.Fatal(err)
	}
	return ok
}

func TestDefEqRules(t *testing.T) {
	env := mustPrelude(t)
	yes := [][2]string{
		{"(fun (x : Nat) => Nat.succ x) 1", "2"}, // beta + Nat literal vs succ
		{"Nat.succ Nat.zero", "1"},               // constructor vs literal
		{"Nat.add 3 4", "7"},                     // GMP acceleration
		{"Nat.mul 6 7", "42"},                    // GMP acceleration
		{"Nat.pow 2 10", "1024"},                 // GMP acceleration
		{"Nat.sub 3 5", "0"},                     // truncated subtraction
		{"Nat.beq 3 3", "Bool.true"},             // predicate extension
		{"Nat.ble 4 3", "Bool.false"},            // predicate extension
		{"(fun (f : Nat → Nat) => fun (x : Nat) => f x)", "(fun (f : Nat → Nat) => f)"},            // eta
		{"let x : Nat := 5; Nat.add x x", "10"},                                                    // zeta
		{"Decidable.decide (Eq.{1} Nat 2 2) (Nat.decEq 2 2)", "Bool.true"},                         // delta + iota
		{"Nat.rec.{1} (fun (_ : Nat) => Nat) 7 (fun (_ : Nat) (ih : Nat) => Nat.succ ih) 3", "10"}, // recursor on a literal
	}
	for _, c := range yes {
		if !defEq(t, env, nil, c[0], c[1]) {
			t.Errorf("%s should be definitionally equal to %s", c[0], c[1])
		}
	}
	no := [][2]string{{"Nat.add 3 4", "8"}, {"Nat.succ 1", "1"}, {"Bool.true", "Bool.false"}, {"Nat.pow 2 3", "9"}}
	for _, c := range no {
		if defEq(t, env, nil, c[0], c[1]) {
			t.Errorf("%s must not be definitionally equal to %s", c[0], c[1])
		}
	}
	// proof irrelevance: two proofs of 1 = 1 are equal
	if !defEq(t, env, nil,
		"Eq.refl.{1} Nat 1",
		"Eq.symm.{1} Nat 1 1 (Eq.refl.{1} Nat 1)") {
		t.Error("proof irrelevance")
	}
	// structure eta for And: h = And.intro (And.left h) (And.right h)
	if !defEq(t, env, nil, "fun (p : Prop) (q : Prop) (h : And p q) => h",
		"fun (p : Prop) (q : Prop) (h : And p q) => And.intro p q (And.left p q h) (And.right p q h)") {
		t.Error("structure eta for And")
	}
	// K-like reduction: Eq.rec on a hypothesis that is not literally refl
	if !defEq(t, env, nil,
		"fun (a : Nat) (h : Eq.{1} Nat a a) => Eq.rec.{1, 1} Nat a (fun (x : Nat) (_ : Eq.{1} Nat a x) => Nat) 5 a h",
		"fun (a : Nat) (h : Eq.{1} Nat a a) => 5") {
		t.Error("K reduction for Eq.rec")
	}
}

func TestUniverseLevels(t *testing.T) {
	u, v := Param("u"), Param("v")
	eq := [][2]*Level{
		{Max(u, v), Max(v, u)}, {Max(u, Zero), u}, {Succ(Max(u, v)), Max(Succ(u), Succ(v))},
		{IMax(u, Succ(v)), Max(u, Succ(v))}, {IMax(u, Zero), Zero}, {IMax(Zero, u), u}, {Max(u, u), u},
		{Max(Succ(Zero), Succ(u)), Succ(Max(Zero, u))},
	}
	for _, p := range eq {
		if !LevelEquiv(p[0], p[1]) {
			t.Errorf("%s should equal %s", p[0], p[1])
		}
	}
	ne := [][2]*Level{{u, v}, {Succ(u), u}, {IMax(u, v), Max(u, v)}, {Max(u, Succ(Zero)), u}}
	for _, p := range ne {
		if LevelEquiv(p[0], p[1]) {
			t.Errorf("%s should differ from %s", p[0], p[1])
		}
	}
	if !LevelGeq(Succ(u), u) || !LevelGeq(Max(u, v), v) || LevelGeq(u, Succ(u)) || LevelGeq(u, v) || !LevelGeq(Succ(Zero), Zero) {
		t.Error("LevelGeq")
	}
}

func mustTerm(t *testing.T, src string, lps []string) *Expr {
	t.Helper()
	e, err := ParseTerm(src, lps)
	if err != nil {
		t.Fatalf("%q: %v", src, err)
	}
	return e
}

func TestInductiveAcceptance(t *testing.T) {
	base := mustPrelude(t)
	try := func(spec InductiveSpec) error { return base.Clone().AddInductive(spec) }
	mk := func(name string, np int, ty string, lps []string, ctors ...[2]string) InductiveSpec {
		s := InductiveSpec{Name: name, LParams: lps, NumParams: np, Type: mustTerm(t, ty, lps)}
		for _, c := range ctors {
			s.Ctors = append(s.Ctors, CtorSpec{c[0], mustTerm(t, c[1], lps)})
		}
		return s
	}
	// a well formed one: lists
	if err := try(mk("List", 1, "Type → Type", nil,
		[2]string{"List.nil", "(α : Type) → List α"},
		[2]string{"List.cons", "(α : Type) → (head : α) → (tail : List α) → List α"})); err != nil {
		t.Errorf("List rejected: %v", err)
	}
	// negative occurrence
	if err := try(mk("Bad", 0, "Type", nil, [2]string{"Bad.mk", "(f : Bad → Nat) → Bad"})); err == nil || !strings.Contains(err.Error(), "non positive") {
		t.Errorf("non-positive occurrence accepted: %v", err)
	}
	// positive occurrence under a Pi is fine (Nat → Tree)
	if err := try(mk("Tree", 0, "Type", nil, [2]string{"Tree.leaf", "Tree"}, [2]string{"Tree.node", "(f : Nat → Tree) → Tree"})); err != nil {
		t.Errorf("Tree rejected: %v", err)
	}
	// universe: a Type-valued field in a Type-valued inductive is too big
	if err := try(mk("Big", 0, "Type", nil, [2]string{"Big.mk", "(x : Type) → Big"})); err == nil || !strings.Contains(err.Error(), "universe") {
		t.Errorf("universe-too-big accepted: %v", err)
	}
	// the same field is fine one universe up
	if err := try(mk("Big1", 0, "Type 1", nil, [2]string{"Big1.mk", "(x : Type) → Big1"})); err != nil {
		t.Errorf("Big1 rejected: %v", err)
	}
	// constructor must return the type
	if err := try(mk("Wrong", 0, "Type", nil, [2]string{"Wrong.mk", "Nat"})); err == nil {
		t.Error("constructor returning another type accepted")
	}
	// parameters must be repeated verbatim in the result
	if err := try(mk("P", 1, "Type → Type", nil, [2]string{"P.mk", "(α : Type) → P Nat"})); err == nil {
		t.Error("constructor changing a parameter accepted")
	}
	// a name that is already taken
	if err := try(mk("Nat", 0, "Type", nil)); err == nil {
		t.Error("redeclaring Nat accepted")
	}
	// theorem whose type is not a proposition
	if err := base.Clone().AddTheorem("thmData", nil, mustTerm(t, "Nat", nil), mustTerm(t, "0", nil)); err == nil {
		t.Error("theorem about data accepted")
	}
	// definition whose body has the wrong type
	if err := base.Clone().AddDefn("badDef", nil, mustTerm(t, "Nat", nil), mustTerm(t, "Bool.true", nil)); err == nil {
		t.Error("ill-typed definition accepted")
	}
}

func TestRecursorsMatchLean(t *testing.T) {
	env := mustPrelude(t)
	same := func(name string, lps []string, want string) {
		t.Helper()
		d := env.Get(name)
		if d == nil {
			t.Fatalf("%s missing", name)
		}
		w := mustTerm(t, want, d.LParams)
		if !Equal(d.Type, w) {
			t.Errorf("%s : %v\n  want %v", name, d.Type, w)
		}
		if strings.Join(d.LParams, ",") != strings.Join(lps, ",") {
			t.Errorf("%s universe params %v, want %v", name, d.LParams, lps)
		}
	}
	same("Nat.rec", []string{"u"}, "(motive : Nat → Sort u) → motive Nat.zero → ((n : Nat) → motive n → motive (Nat.succ n)) → (t : Nat) → motive t")
	same("Bool.rec", []string{"u"}, "(motive : Bool → Sort u) → motive Bool.false → motive Bool.true → (t : Bool) → motive t")
	same("False.rec", []string{"u"}, "(motive : False → Sort u) → (t : False) → motive t")
	same("Eq.rec", []string{"u_1", "u"},
		"(α : Sort u) → (a : α) → (motive : (b : α) → Eq.{u} α a b → Sort u_1) → motive a (Eq.refl.{u} α a) → (b : α) → (t : Eq.{u} α a b) → motive b t")
	// a Prop with two constructors eliminates only into Prop, so there is no universe parameter
	same("Nat.le.rec", nil,
		"(n : Nat) → (motive : (a : Nat) → Nat.le n a → Prop) → motive n (Nat.le.refl n) → "+
			"((m : Nat) → (a : Nat.le n m) → motive m a → motive (Nat.succ m) (Nat.le.step n m a)) → (a : Nat) → (t : Nat.le n a) → motive a t")
	same("Or.rec", nil, "(a : Prop) → (b : Prop) → (motive : Or a b → Prop) → ((h : a) → motive (Or.inl a b h)) → ((h : b) → motive (Or.inr a b h)) → (t : Or a b) → motive t")
	same("Exists.rec", []string{"u"}, "(α : Sort u) → (p : α → Prop) → (motive : Exists.{u} α p → Prop) → ((w : α) → (h : p w) → motive (Exists.intro.{u} α p w h)) → (t : Exists.{u} α p) → motive t")
	same("And.rec", []string{"u"}, "(a : Prop) → (b : Prop) → (motive : And a b → Sort u) → ((left : a) → (right : b) → motive (And.intro a b left right)) → (t : And a b) → motive t")
	same("Int.rec", []string{"u"},
		"(motive : Int → Sort u) → ((n : Nat) → motive (Int.ofNat n)) → ((n : Nat) → motive (Int.negSucc n)) → (t : Int) → motive t")
	// an indexed Prop family with one index: the motive takes the index and the proof, and elimination is into Prop only
	same("Int.NonNeg.rec", nil,
		"(motive : (t : Int) → Int.NonNeg t → Prop) → ((n : Nat) → motive (Int.ofNat n) (Int.NonNeg.mk n)) → "+
			"(t : Int) → (h : Int.NonNeg t) → motive t h")
	same("Decidable.rec", []string{"u"},
		"(p : Prop) → (motive : Decidable p → Sort u) → ((h : Not p) → motive (Decidable.isFalse p h)) → ((h : p) → motive (Decidable.isTrue p h)) → (t : Decidable p) → motive t")
	if !env.Get("Eq.rec").K || env.Get("Nat.rec").K || env.Get("Or.rec").K {
		t.Error("K flags: only Eq.rec is a K-like recursor here")
	}
}
