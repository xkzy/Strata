package lean

import "testing"

// A mutual block whose types nest a sibling: `A` holds a `List B` and `B` holds an `A`. The nested family (`List`) is
// outside the block, but its parameter is a type the block defines, so the auxiliary joins the block and each recursor
// of the block grows a third motive for `List B`.
func TestMutualBlockThatNestsASibling(t *testing.T) {
	env := envWithList(t)
	err := env.AddMutual([]InductiveSpec{
		mkSpec(t, "A", 0, "Type", [2]string{"A.mk", "(kids : List.{0} B) → A"}),
		mkSpec(t, "B", 0, "Type", [2]string{"B.mk", "(a : A) → B"}),
	})
	if err != nil {
		t.Fatal(err)
	}
	if env.Get("A._nested_1") != nil || env.Get("B._nested_1") != nil {
		t.Error("auxiliary type leaked into the environment")
	}
	// Lean: one motive per type in the block plus one for the nested family, and the minor premises in the order the
	// types were declared (then the auxiliary's constructors).
	minors := "(mk : (kids : List.{0} B) → motive_3 kids → motive_1 (A.mk kids)) → " +
		"(mk : (a : A) → motive_1 a → motive_2 (B.mk a)) → " +
		"(nil : motive_3 (List.nil.{0} B)) → " +
		"(cons : (head : B) → (tail : List.{0} B) → motive_2 head → motive_3 tail → " +
		"motive_3 (List.cons.{0} B head tail)) → "
	for name, concl := range map[string]string{
		"A.rec":   "(t : A) → motive_1 t",
		"B.rec":   "(t : B) → motive_2 t",
		"A.rec_1": "(t : List.{0} B) → motive_3 t",
	} {
		want := "(motive_1 : A → Sort u) → (motive_2 : B → Sort u) → (motive_3 : List.{0} B → Sort u) → " +
			minors + concl
		d := env.Get(name)
		if d == nil {
			t.Fatalf("%s missing", name)
		}
		if !Equal(d.Type, mustTerm(t, want, []string{"u"})) {
			t.Errorf("%s : %v", name, d.Type)
		}
	}
	// computation: A.mk [B.mk (A.mk [])] has two A/B nodes, counted through all three motives
	rec := "A.rec.{1} (fun (_ : A) => Nat) (fun (_ : B) => Nat) (fun (_ : List.{0} B) => Nat) " +
		"(fun (kids : List.{0} B) (n : Nat) => Nat.succ n) " +
		"(fun (a : A) (n : Nat) => n) 0 " +
		"(fun (head : B) (tail : List.{0} B) (a : Nat) (b : Nat) => Nat.add a b) "
	value := "A.mk (List.cons.{0} B (B.mk (A.mk (List.nil.{0} B))) (List.nil.{0} B))"
	wantAccept(t, env, nil, rec+"("+value+")")
	tc := NewTC(env, nil)
	got, err := tc.Whnf(mustTerm(t, rec+"("+value+")", nil))
	if err != nil {
		t.Fatal(err)
	}
	if ok, _ := tc.IsDefEq(got, mustTerm(t, "2", nil)); !ok {
		t.Errorf("the count reduced to %v, want 2", got)
	}
}

// The same through `B.rec`, whose conclusion is the second motive, so the entry point of the elimination can be either
// type of the block.
func TestMutualNestingEliminatesFromEitherType(t *testing.T) {
	env := envWithList(t)
	err := env.AddMutual([]InductiveSpec{
		mkSpec(t, "A", 0, "Type", [2]string{"A.mk", "(kids : List.{0} B) → A"}),
		mkSpec(t, "B", 0, "Type", [2]string{"B.mk", "(a : A) → B"}),
	})
	if err != nil {
		t.Fatal(err)
	}
	// the `List B` argument of A.mk is passed on unchanged: only A and B are counted
	rec := "B.rec.{1} (fun (_ : A) => Nat) (fun (_ : B) => Nat) (fun (_ : List.{0} B) => Nat) " +
		"(fun (kids : List.{0} B) (n : Nat) => n) " +
		"(fun (a : A) (n : Nat) => Nat.succ n) 0 " +
		"(fun (head : B) (tail : List.{0} B) (a : Nat) (b : Nat) => Nat.add a b) "
	value := "B.mk (A.mk (List.cons.{0} B (B.mk (A.mk (List.nil.{0} B))) (List.nil.{0} B)))"
	wantAccept(t, env, nil, rec+"("+value+")")
	tc := NewTC(env, nil)
	got, err := tc.Whnf(mustTerm(t, rec+"("+value+")", nil))
	if err != nil {
		t.Fatal(err)
	}
	// only the two `B` nodes are counted: the `A` minor passes the nested count on unchanged
	if ok, _ := tc.IsDefEq(got, mustTerm(t, "2", nil)); !ok {
		t.Errorf("the count reduced to %v, want 2", got)
	}
}
