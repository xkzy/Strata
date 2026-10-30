package lean

import "testing"

// A family with three indices: `Cube α i j k`. Its recursor is Lean's - one motive taking the three indices before the
// value, and the constructor minors binding them - and it reduces on values whose indices are fixed.
func envWithCube(t *testing.T) *Env {
	t.Helper()
	env := envWithList(t)
	err := env.AddInductive(InductiveSpec{Name: "Cube", NumParams: 1,
		Type: mustTerm(t, "(α : Type) → Nat → Nat → Nat → Type", nil),
		Ctors: []CtorSpec{
			{"Cube.empty", mustTerm(t, "(α : Type) → (i : Nat) → (j : Nat) → (k : Nat) → Cube α i j k", nil)},
			{"Cube.point", mustTerm(t, "(α : Type) → (i : Nat) → (j : Nat) → (k : Nat) → (a : α) → Cube α i j k", nil)},
			{"Cube.stack", mustTerm(t, "(α : Type) → (i : Nat) → (j : Nat) → (k : Nat) → (below : Cube α i j k) → (top : Cube α i j k) → Cube α i j (Nat.succ k)", nil)},
		}})
	if err != nil {
		t.Fatal(err)
	}
	return env
}

func TestThreeIndexFamily(t *testing.T) {
	env := envWithCube(t)
	want := "(α : Type) → (motive : (i : Nat) → (j : Nat) → (k : Nat) → Cube α i j k → Sort u) → " +
		"(empty : (i : Nat) → (j : Nat) → (k : Nat) → motive i j k (Cube.empty α i j k)) → " +
		"(point : (i : Nat) → (j : Nat) → (k : Nat) → (a : α) → motive i j k (Cube.point α i j k a)) → " +
		"(stack : (i : Nat) → (j : Nat) → (k : Nat) → (below : Cube α i j k) → (top : Cube α i j k) → " +
		"motive i j k below → motive i j k top → motive i j (Nat.succ k) (Cube.stack α i j k below top)) → " +
		"(i : Nat) → (j : Nat) → (k : Nat) → (t : Cube α i j k) → motive i j k t"
	d := env.Get("Cube.rec")
	if d == nil {
		t.Fatal("Cube.rec missing")
	}
	if !Equal(d.Type, mustTerm(t, want, []string{"u"})) {
		t.Errorf("Cube.rec : %v", d.Type)
	}
	// computation at fixed indices: a leaf counts 1, a stack of two leaves 2, and the empty cube 0
	rec := "Cube.rec.{1} Nat (fun (i : Nat) (j : Nat) (k : Nat) (t : Cube Nat i j k) => Nat) " +
		"(fun (i : Nat) (j : Nat) (k : Nat) => 0) " +
		"(fun (i : Nat) (j : Nat) (k : Nat) (a : Nat) => 1) " +
		"(fun (i : Nat) (j : Nat) (k : Nat) (below : Cube Nat i j k) (top : Cube Nat i j k) (a : Nat) (b : Nat) => " +
		"Nat.add a b) "
	for _, c := range []struct {
		name string
		src  string
		want string
	}{
		{"empty", rec + "2 3 4 (Cube.empty Nat 2 3 4)", "0"},
		{"point", rec + "2 3 0 (Cube.point Nat 2 3 0 7)", "1"},
		{"stack", rec + "0 0 1 (Cube.stack Nat 0 0 0 (Cube.point Nat 0 0 0 1) (Cube.point Nat 0 0 0 2))", "2"},
	} {
		wantAccept(t, env, nil, c.src)
		tc := NewTC(env, nil)
		got, err := tc.Whnf(mustTerm(t, c.src, nil))
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if ok, _ := tc.IsDefEq(got, mustTerm(t, c.want, nil)); !ok {
			t.Errorf("%s reduced to %v, want %s", c.name, got, c.want)
		}
	}
	// the index is part of the type: a value at one index is not a value at another
	wantReject(t, env, nil, "(fun (t : Cube Nat 2 3 4) => t) (Cube.point Nat 2 3 5 7)", "wrong third index")
}

// A three-index family nested inside another type: `Slice.mk` holds a `Cube Slice 2 3 4`. The auxiliary carries all
// three indices, so its motive takes them before the value, and the parameter `Slice` is recursive, so the auxiliary's
// own `point` minor takes an induction hypothesis for it.
func TestNestedThreeIndexFamily(t *testing.T) {
	env := envWithCube(t)
	if err := env.AddInductive(InductiveSpec{Name: "Slice", NumParams: 0,
		Type: mustTerm(t, "Type", nil),
		Ctors: []CtorSpec{
			{"Slice.mk", mustTerm(t, "(cells : Cube Slice 2 3 4) → Slice", nil)},
		}}); err != nil {
		t.Fatal(err)
	}
	if env.Get("Slice._nested_1") != nil {
		t.Error("auxiliary type leaked into the environment")
	}
	minors := "(mk : (cells : Cube Slice 2 3 4) → motive_2 2 3 4 cells → motive_1 (Slice.mk cells)) → " +
		"(empty : (i : Nat) → (j : Nat) → (k : Nat) → motive_2 i j k (Cube.empty Slice i j k)) → " +
		"(point : (i : Nat) → (j : Nat) → (k : Nat) → (a : Slice) → motive_1 a → " +
		"motive_2 i j k (Cube.point Slice i j k a)) → " +
		"(stack : (i : Nat) → (j : Nat) → (k : Nat) → (below : Cube Slice i j k) → (top : Cube Slice i j k) → " +
		"motive_2 i j k below → motive_2 i j k top → motive_2 i j (Nat.succ k) (Cube.stack Slice i j k below top)) → "
	for name, concl := range map[string]string{
		"Slice.rec":   "(t : Slice) → motive_1 t",
		"Slice.rec_1": "(i : Nat) → (j : Nat) → (k : Nat) → (t : Cube Slice i j k) → motive_2 i j k t",
	} {
		want := "(motive_1 : Slice → Sort u) → (motive_2 : (i : Nat) → (j : Nat) → (k : Nat) → Cube Slice i j k → Sort u) → " +
			minors + concl
		d := env.Get(name)
		if d == nil {
			t.Fatalf("%s missing", name)
		}
		if !Equal(d.Type, mustTerm(t, want, []string{"u"})) {
			t.Errorf("%s : %v", name, d.Type)
		}
	}
	// computation: one slice holding a point holding an empty slice counts 3 (the two slices and the point)
	rec := "Slice.rec.{1} (fun (_ : Slice) => Nat) " +
		"(fun (i : Nat) (j : Nat) (k : Nat) (t : Cube Slice i j k) => Nat) " +
		"(fun (cells : Cube Slice 2 3 4) (n : Nat) => Nat.succ n) " +
		"(fun (i : Nat) (j : Nat) (k : Nat) => 0) " +
		"(fun (i : Nat) (j : Nat) (k : Nat) (a : Slice) (n : Nat) => Nat.succ n) " +
		"(fun (i : Nat) (j : Nat) (k : Nat) (below : Cube Slice i j k) (top : Cube Slice i j k) (a : Nat) (b : Nat) => " +
		"Nat.add a b) "
	value := "(Slice.mk (Cube.point Slice 2 3 4 (Slice.mk (Cube.empty Slice 2 3 4))))"
	wantAccept(t, env, nil, rec+value)
	tc := NewTC(env, nil)
	got, err := tc.Whnf(mustTerm(t, rec+value, nil))
	if err != nil {
		t.Fatal(err)
	}
	if ok, _ := tc.IsDefEq(got, mustTerm(t, "3", nil)); !ok {
		t.Errorf("the count reduced to %v, want 3", got)
	}
	// and through the auxiliary's own recursor, from the nested value directly
	aux := "Slice.rec_1.{1} (fun (_ : Slice) => Nat) " +
		"(fun (i : Nat) (j : Nat) (k : Nat) (t : Cube Slice i j k) => Nat) " +
		"(fun (cells : Cube Slice 2 3 4) (n : Nat) => Nat.succ n) " +
		"(fun (i : Nat) (j : Nat) (k : Nat) => 0) " +
		"(fun (i : Nat) (j : Nat) (k : Nat) (a : Slice) (n : Nat) => Nat.succ n) " +
		"(fun (i : Nat) (j : Nat) (k : Nat) (below : Cube Slice i j k) (top : Cube Slice i j k) (a : Nat) (b : Nat) => " +
		"Nat.add a b) "
	wantAccept(t, env, nil, aux+"2 3 4 (Cube.point Slice 2 3 4 (Slice.mk (Cube.empty Slice 2 3 4)))")
	got, err = tc.Whnf(mustTerm(t, aux+"2 3 4 (Cube.point Slice 2 3 4 (Slice.mk (Cube.empty Slice 2 3 4)))", nil))
	if err != nil {
		t.Fatal(err)
	}
	if ok, _ := tc.IsDefEq(got, mustTerm(t, "2", nil)); !ok {
		t.Errorf("the auxiliary count reduced to %v, want 2", got)
	}
}
