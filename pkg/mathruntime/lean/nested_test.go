package lean

import (
	"strings"
	"testing"
)

func envWithList(t *testing.T) *Env {
	env := mustPrelude(t).Clone()
	if err := env.AddInductive(InductiveSpec{Name: "List", LParams: []string{"u"}, NumParams: 1,
		Type: mustTerm(t, "Sort (u+1) → Sort (u+1)", []string{"u"}),
		Ctors: []CtorSpec{
			{"List.nil", mustTerm(t, "(α : Type u) → List.{u} α", []string{"u"})},
			{"List.cons", mustTerm(t, "(α : Type u) → (head : α) → (tail : List.{u} α) → List.{u} α", []string{"u"})}}}); err != nil {
		t.Fatal(err)
	}
	return env
}

func TestNestedRose(t *testing.T) {
	env := envWithList(t)
	err := env.AddInductive(mkSpec(t, "Rose", 0, "Type", [2]string{"Rose.node", "(children : List.{0} Rose) → Rose"}))
	if err != nil {
		t.Fatal(err)
	}
	if env.Get("Rose._nested_1") != nil || env.Get("Rose._nested_1.rec") != nil {
		t.Error("auxiliary type leaked into the environment")
	}
	if !env.Get("Rose").IsRec {
		t.Error("Rose should be recursive")
	}
	minors := "((children : List.{0} Rose) → motive_2 children → motive_1 (Rose.node children)) → motive_2 (List.nil.{0} Rose) → " +
		"((head : Rose) → (tail : List.{0} Rose) → motive_1 head → motive_2 tail → motive_2 (List.cons.{0} Rose head tail)) → "
	for name, want := range map[string]string{
		"Rose.rec":   "(motive_1 : Rose → Sort u) → (motive_2 : List.{0} Rose → Sort u) → " + minors + "(t : Rose) → motive_1 t",
		"Rose.rec_1": "(motive_1 : Rose → Sort u) → (motive_2 : List.{0} Rose → Sort u) → " + minors + "(t : List.{0} Rose) → motive_2 t",
	} {
		d := env.Get(name)
		if d == nil {
			t.Fatalf("%s missing", name)
		}
		if !Equal(d.Type, mustTerm(t, want, []string{"u"})) {
			t.Errorf("%s : %v", name, d.Type)
		}
	}
	// computation: count the nodes of node [node [], node []] = 3 through Rose.rec_1 on the child list
	size := "Rose.rec.{1} (fun (_ : Rose) => Nat) (fun (_ : List.{0} Rose) => Nat) (fun (_ : List.{0} Rose) (n : Nat) => Nat.succ n) 0 " +
		"(fun (_ : Rose) (_ : List.{0} Rose) (a : Nat) (b : Nat) => Nat.add a b) "
	leaf := "Rose.node (List.nil.{0} Rose)"
	tree := "Rose.node (List.cons.{0} Rose (" + leaf + ") (List.cons.{0} Rose (" + leaf + ") (List.nil.{0} Rose)))"
	wantAccept(t, env, nil, size+"("+tree+")")
	tc := NewTC(env, nil)
	got, err := tc.Whnf(mustTerm(t, size+"("+tree+")", nil))
	if err != nil {
		t.Fatal(err)
	}
	if ok, _ := tc.IsDefEq(got, mustTerm(t, "3", nil)); !ok {
		t.Errorf("Rose size reduced to %v, want 3", got)
	}
	// the list recursor reduces on List's own constructors
	listSize := "Rose.rec_1.{1} (fun (_ : Rose) => Nat) (fun (_ : List.{0} Rose) => Nat) (fun (_ : List.{0} Rose) (n : Nat) => Nat.succ n) 0 " +
		"(fun (_ : Rose) (_ : List.{0} Rose) (a : Nat) (b : Nat) => Nat.add a b) (List.cons.{0} Rose (" + leaf + ") (List.nil.{0} Rose))"
	got, err = tc.Whnf(mustTerm(t, listSize, nil))
	if err != nil {
		t.Fatal(err)
	}
	if ok, _ := tc.IsDefEq(got, mustTerm(t, "1", nil)); !ok {
		t.Errorf("list size reduced to %v, want 1", got)
	}
}

func TestNestedRejections(t *testing.T) {
	env := envWithList(t)
	// a nested occurrence in a negative position is still negative
	if err := env.Clone().AddInductive(mkSpec(t, "Bad", 0, "Type", [2]string{"Bad.mk", "(f : Bad → List.{0} Nat) → Bad"})); err == nil || !strings.Contains(err.Error(), "non positive") {
		t.Error("negative occurrence next to a nested one accepted")
	}
	// the nested type's parameter may not mention a field
	if err := env.Clone().AddInductive(mkSpec(t, "Dep", 0, "Type", [2]string{"Dep.mk", "(n : Nat) → (xs : List.{0} (Eq.{1} Nat n n → Dep)) → Dep"})); err == nil || !strings.Contains(err.Error(), "local variables") {
		t.Error("parameter mentioning a field accepted")
	}
}

func TestNestedWithParameter(t *testing.T) {
	env := envWithList(t)
	err := env.AddInductive(mkSpec(t, "T", 1, "Type → Type",
		[2]string{"T.node", "(α : Type) → (a : α) → (cs : List.{0} (T α)) → T α"}))
	if err != nil {
		t.Fatal(err)
	}
	minors := "((a : α) → (cs : List.{0} (T α)) → motive_2 cs → motive_1 (T.node α a cs)) → motive_2 (List.nil.{0} (T α)) → " +
		"((head : T α) → (tail : List.{0} (T α)) → motive_1 head → motive_2 tail → motive_2 (List.cons.{0} (T α) head tail)) → "
	head := "(α : Type) → (motive_1 : T α → Sort u) → (motive_2 : List.{0} (T α) → Sort u) → "
	for name, want := range map[string]string{
		"T.rec":   head + minors + "(t : T α) → motive_1 t",
		"T.rec_1": head + minors + "(t : List.{0} (T α)) → motive_2 t",
	} {
		d := env.Get(name)
		if d == nil || !Equal(d.Type, mustTerm(t, want, []string{"u"})) {
			t.Errorf("%s : %v", name, d)
		}
	}
	// count the elements stored in node 7 [node 8 [], node 9 []] over α := Nat: 3
	rec := "T.rec.{1} Nat (fun (_ : T Nat) => Nat) (fun (_ : List.{0} (T Nat)) => Nat) (fun (_ : Nat) (_ : List.{0} (T Nat)) (n : Nat) => Nat.succ n) 0 " +
		"(fun (_ : T Nat) (_ : List.{0} (T Nat)) (a : Nat) (b : Nat) => Nat.add a b) "
	leaf := func(n string) string { return "T.node Nat " + n + " (List.nil.{0} (T Nat))" }
	tree := "T.node Nat 7 (List.cons.{0} (T Nat) (" + leaf("8") + ") (List.cons.{0} (T Nat) (" + leaf("9") + ") (List.nil.{0} (T Nat))))"
	wantAccept(t, env, nil, rec+"("+tree+")")
	tc := NewTC(env, nil)
	got, err := tc.Whnf(mustTerm(t, rec+"("+tree+")", nil))
	if err != nil {
		t.Fatal(err)
	}
	if ok, _ := tc.IsDefEq(got, mustTerm(t, "3", nil)); !ok {
		t.Errorf("T size reduced to %v, want 3", got)
	}
}

func TestNestedInsideMutual(t *testing.T) {
	env := envWithList(t)
	err := env.AddMutual([]InductiveSpec{
		mkSpec(t, "A", 0, "Type", [2]string{"A.mk", "(bs : List.{0} B) → A"}),
		mkSpec(t, "B", 0, "Type", [2]string{"B.mk", "(a : A) → B"}),
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range []string{"A.rec", "B.rec", "A.rec_1"} {
		d := env.Get(n)
		if d == nil || d.NumMotives != 3 || d.NumMinors != 4 {
			t.Errorf("%s = %+v", n, d)
		}
	}
	if env.Get("A.rec_2") != nil {
		t.Error("only one auxiliary type expected")
	}
	// the motive of the auxiliary type comes last, after A's and B's
	want := "(motive_1 : A → Sort u) → (motive_2 : B → Sort u) → (motive_3 : List.{0} B → Sort u) → " +
		"((bs : List.{0} B) → motive_3 bs → motive_1 (A.mk bs)) → ((a : A) → motive_1 a → motive_2 (B.mk a)) → motive_3 (List.nil.{0} B) → " +
		"((head : B) → (tail : List.{0} B) → motive_2 head → motive_3 tail → motive_3 (List.cons.{0} B head tail)) → (t : A) → motive_1 t"
	if d := env.Get("A.rec"); !Equal(d.Type, mustTerm(t, want, []string{"u"})) {
		t.Errorf("A.rec : %v", d.Type)
	}
}
