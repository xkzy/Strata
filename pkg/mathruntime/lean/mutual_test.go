package lean

import (
	"strings"
	"testing"
)

func mkSpec(t *testing.T, name string, np int, ty string, ctors ...[2]string) InductiveSpec {
	s := InductiveSpec{Name: name, NumParams: np, Type: mustTerm(t, ty, nil)}
	for _, c := range ctors {
		s.Ctors = append(s.Ctors, CtorSpec{c[0], mustTerm(t, c[1], nil)})
	}
	return s
}

func TestMutualTreeForest(t *testing.T) {
	env := mustPrelude(t).Clone()
	err := env.AddMutual([]InductiveSpec{
		mkSpec(t, "Tree", 0, "Type", [2]string{"Tree.node", "(f : Forest) → Tree"}),
		mkSpec(t, "Forest", 0, "Type", [2]string{"Forest.nil", "Forest"}, [2]string{"Forest.cons", "(t : Tree) → (f : Forest) → Forest"}),
	})
	if err != nil {
		t.Fatal(err)
	}
	// Lean: Tree.rec and Forest.rec each take both motives and all three minor premises
	minors := "((f : Forest) → motive_2 f → motive_1 (Tree.node f)) → motive_2 Forest.nil → " +
		"((t : Tree) → (f : Forest) → motive_1 t → motive_2 f → motive_2 (Forest.cons t f)) → "
	for name, want := range map[string]string{
		"Tree.rec":   "(motive_1 : Tree → Sort u) → (motive_2 : Forest → Sort u) → " + minors + "(t : Tree) → motive_1 t",
		"Forest.rec": "(motive_1 : Tree → Sort u) → (motive_2 : Forest → Sort u) → " + minors + "(t : Forest) → motive_2 t",
	} {
		d := env.Get(name)
		if d == nil || !Equal(d.Type, mustTerm(t, want, []string{"u"})) {
			t.Errorf("%s : %v", name, d)
		}
	}
	// computation: count nodes. Tree.node (Forest.cons (Tree.node Forest.nil) Forest.nil) has 2 nodes
	count := func(arg string) string {
		return "Tree.rec.{1} (fun (_ : Tree) => Nat) (fun (_ : Forest) => Nat) (fun (_ : Forest) (n : Nat) => Nat.succ n) 0 " +
			"(fun (_ : Tree) (_ : Forest) (a : Nat) (b : Nat) => Nat.add a b) (" + arg + ")"
	}
	tree := "Tree.node (Forest.cons (Tree.node Forest.nil) Forest.nil)"
	wantAccept(t, env, nil, count(tree))
	tc := NewTC(env, nil)
	got, err := tc.Whnf(mustTerm(t, count(tree), nil))
	if err != nil {
		t.Fatal(err)
	}
	if ok, _ := tc.IsDefEq(got, mustTerm(t, "2", nil)); !ok {
		t.Errorf("node count reduced to %v, want 2", got)
	}
}

func TestMutualEvenOdd(t *testing.T) {
	env := mustPrelude(t).Clone()
	err := env.AddMutual([]InductiveSpec{
		mkSpec(t, "Even", 0, "Nat → Prop", [2]string{"Even.zero", "Even 0"}, [2]string{"Even.succ", "(n : Nat) → (h : Odd n) → Even (Nat.succ n)"}),
		mkSpec(t, "Odd", 0, "Nat → Prop", [2]string{"Odd.succ", "(n : Nat) → (h : Even n) → Odd (Nat.succ n)"}),
	})
	if err != nil {
		t.Fatal(err)
	}
	// a mutual Prop eliminates only into Prop: no universe parameter
	if d := env.Get("Even.rec"); d == nil || len(d.LParams) != 0 || d.NumMotives != 2 || d.NumMinors != 3 {
		t.Fatalf("Even.rec = %+v", d)
	}
	want := "(motive_1 : (a : Nat) → Even a → Prop) → (motive_2 : (a : Nat) → Odd a → Prop) → motive_1 0 Even.zero → " +
		"((n : Nat) → (h : Odd n) → motive_2 n h → motive_1 (Nat.succ n) (Even.succ n h)) → " +
		"((n : Nat) → (h : Even n) → motive_1 n h → motive_2 (Nat.succ n) (Odd.succ n h)) → (a : Nat) → (t : Even a) → motive_1 a t"
	if !Equal(env.Get("Even.rec").Type, mustTerm(t, want, nil)) {
		t.Errorf("Even.rec : %v", env.Get("Even.rec").Type)
	}
	// Even 2 is provable, built from the constructors
	wantAccept(t, env, nil, "Even.succ 1 (Odd.succ 0 Even.zero)")
	wantReject(t, env, nil, "Even.succ 1 Even.zero", "Even.zero : Even 0 is not Odd 1")
}

func TestMutualRejections(t *testing.T) {
	base := mustPrelude(t)
	try := func(specs ...InductiveSpec) error { return base.Clone().AddMutual(specs) }
	if err := try(mkSpec(t, "A", 0, "Type", [2]string{"A.mk", "(f : B → Nat) → A"}), mkSpec(t, "B", 0, "Type", [2]string{"B.mk", "B"})); err == nil || !strings.Contains(err.Error(), "non positive") {
		t.Errorf("negative occurrence of a sibling accepted: %v", err)
	}
	if err := try(mkSpec(t, "A", 0, "Type", [2]string{"A.mk", "A"}), mkSpec(t, "B", 0, "Type 1", [2]string{"B.mk", "B"})); err == nil || !strings.Contains(err.Error(), "same universe") {
		t.Errorf("different universes accepted: %v", err)
	}
	if err := try(mkSpec(t, "A", 0, "Type", [2]string{"A.mk", "A"}), mkSpec(t, "A", 0, "Type", [2]string{"A2.mk", "A"})); err == nil {
		t.Error("duplicate type name accepted")
	}
	if err := try(mkSpec(t, "A", 0, "Type", [2]string{"A.mk", "B"}), mkSpec(t, "B", 0, "Type", [2]string{"B.mk", "B"})); err == nil {
		t.Error("constructor returning a sibling accepted")
	}
}
