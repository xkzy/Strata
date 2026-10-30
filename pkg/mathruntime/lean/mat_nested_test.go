package lean

import "testing"

// Vector has one index; a family with two (a matrix whose rows all have the same width) is the next step up: the
// auxiliary of a nested occurrence then carries both indices, its motive takes them before the value, and the
// recursor's minor for each of the family's constructors has to line them up with the constructor's result. The
// matrix's own constructor holds a `Vector α n`, so the auxiliary for the matrix nests Vector in turn and the
// recursor grows a third motive for it.
func envWithMat(t *testing.T) *Env {
	t.Helper()
	env := envWithVector(t)
	mat := InductiveSpec{
		Name:      "Mat",
		LParams:   []string{"u"},
		NumParams: 1,
		Type:      mustTerm(t, "(α : Sort (u+1)) → Nat → Nat → Sort (u+1)", []string{"u"}),
		Ctors: []CtorSpec{
			{"Mat.empty", mustTerm(t, "(α : Type u) → (n : Nat) → Mat.{u} α 0 n", []string{"u"})},
			{"Mat.cons", mustTerm(t, "(α : Type u) → (n : Nat) → (m : Nat) → (row : Vector.{u} α n) → "+
				"(rest : Mat.{u} α m n) → Mat.{u} α (Nat.succ m) n", []string{"u"})},
		},
	}
	if err := env.AddInductive(mat); err != nil {
		t.Fatal(err)
	}
	return env
}

func TestMatrixRecursorHasTwoIndices(t *testing.T) {
	env := envWithMat(t)
	want := "(α : Sort (u+1)) → (motive : (r : Nat) → (c : Nat) → Mat.{u} α r c → Sort u_1) → " +
		"((n : Nat) → motive 0 n (Mat.empty.{u} α n)) → " +
		"((n : Nat) → (m : Nat) → (row : Vector.{u} α n) → (rest : Mat.{u} α m n) → motive m n rest → " +
		"motive (Nat.succ m) n (Mat.cons.{u} α n m row rest)) → " +
		"(r : Nat) → (c : Nat) → (t : Mat.{u} α r c) → motive r c t"
	d := env.Get("Mat.rec")
	if d == nil || !Equal(d.Type, mustTerm(t, want, []string{"u", "u_1"})) {
		t.Fatalf("Mat.rec : %v", d)
	}
}

// A type that nests a fixed 2x2 occurrence: the auxiliary is indexed twice, and its own nested Vector gets an
// auxiliary too, so the recursor has one motive per family involved.
func TestNestedTwoIndexFamily(t *testing.T) {
	env := envWithMat(t)
	spec := mkSpec(t, "Block", 0, "Type",
		[2]string{"Block.leaf", "Block"},
		[2]string{"Block.grid", "(kids : Mat.{0} Block 2 2) → Block"})
	if err := env.AddInductive(spec); err != nil {
		t.Fatalf("a nested two-index family: %v", err)
	}
	common := "(motive_1 : Block → Sort u) → (motive_2 : (r : Nat) → (c : Nat) → Mat.{0} Block r c → Sort u) → " +
		"(motive_3 : (n : Nat) → Vector.{0} Block n → Sort u) → " +
		"motive_1 Block.leaf → " +
		"((kids : Mat.{0} Block 2 2) → motive_2 2 2 kids → motive_1 (Block.grid kids)) → " +
		"((n : Nat) → motive_2 0 n (Mat.empty.{0} Block n)) → " +
		"((n : Nat) → (m : Nat) → (row : Vector.{0} Block n) → (rest : Mat.{0} Block m n) → " +
		"motive_3 n row → motive_2 m n rest → motive_2 (Nat.succ m) n (Mat.cons.{0} Block n m row rest)) → " +
		"motive_3 0 (Vector.nil.{0} Block) → " +
		"((n : Nat) → (head : Block) → (tail : Vector.{0} Block n) → motive_1 head → motive_3 n tail → " +
		"motive_3 (Nat.succ n) (Vector.cons.{0} Block n head tail)) → "
	for name, concl := range map[string]string{
		"Block.rec":   "(t : Block) → motive_1 t",
		"Block.rec_1": "(r : Nat) → (c : Nat) → (t : Mat.{0} Block r c) → motive_2 r c t",
		"Block.rec_2": "(n : Nat) → (t : Vector.{0} Block n) → motive_3 n t",
	} {
		d := env.Get(name)
		if d == nil {
			t.Fatalf("%s missing", name)
		}
		if !Equal(d.Type, mustTerm(t, common+concl, []string{"u"})) {
			t.Errorf("%s : %v", name, d.Type)
		}
	}

	// A 2x2 grid of leaves counts 1 for the grid plus one for each of its four leaves.
	row := "Vector.cons.{0} Block 1 Block.leaf (Vector.cons.{0} Block 0 Block.leaf (Vector.nil.{0} Block))"
	mat := "Mat.cons.{0} Block 2 1 (" + row + ") (Mat.cons.{0} Block 2 0 (" + row + ") (Mat.empty.{0} Block 2))"
	term := "Block.rec.{1} (fun (_ : Block) => Nat) " +
		"(fun (r : Nat) (c : Nat) (_ : Mat.{0} Block r c) => Nat) " +
		"(fun (n : Nat) (_ : Vector.{0} Block n) => Nat) 1 " +
		"(fun (kids : Mat.{0} Block 2 2) (k : Nat) => Nat.succ k) " +
		"(fun (n : Nat) => Nat.zero) " +
		"(fun (n : Nat) (m : Nat) (row : Vector.{0} Block n) (rest : Mat.{0} Block m n) (rh : Nat) (ih : Nat) => " +
		"Nat.add rh ih) 0 " +
		"(fun (n : Nat) (head : Block) (tail : Vector.{0} Block n) (h : Nat) (t : Nat) => Nat.add h t) " +
		"(Block.grid (" + mat + "))"
	wantAccept(t, env, nil, term)
	got, err := NewTC(env, nil).Whnf(mustTerm(t, term, nil))
	if err != nil {
		t.Fatal(err)
	}
	if ok, _ := NewTC(env, nil).IsDefEq(got, mustTerm(t, "5", nil)); !ok {
		t.Errorf("leaf count reduced to %v, want 5", got)
	}
}

// Both indices bound in the constructor (`Mat Grid r c`), the two-index analogue of `Vector Forest n`.
func TestNestedTwoIndexFamilyWithBoundIndices(t *testing.T) {
	env := envWithMat(t)
	spec := mkSpec(t, "Grid", 0, "Type",
		[2]string{"Grid.leaf", "Grid"},
		[2]string{"Grid.node", "(r : Nat) → (c : Nat) → (kids : Mat.{0} Grid r c) → Grid"})
	if err := env.AddInductive(spec); err != nil {
		t.Fatalf("a nested two-index family with bound indices: %v", err)
	}
	want := "(motive_1 : Grid → Sort u) → (motive_2 : (r : Nat) → (c : Nat) → Mat.{0} Grid r c → Sort u) → " +
		"(motive_3 : (n : Nat) → Vector.{0} Grid n → Sort u) → " +
		"motive_1 Grid.leaf → " +
		"((r : Nat) → (c : Nat) → (kids : Mat.{0} Grid r c) → motive_2 r c kids → motive_1 (Grid.node r c kids)) → " +
		"((n : Nat) → motive_2 0 n (Mat.empty.{0} Grid n)) → " +
		"((n : Nat) → (m : Nat) → (row : Vector.{0} Grid n) → (rest : Mat.{0} Grid m n) → " +
		"motive_3 n row → motive_2 m n rest → motive_2 (Nat.succ m) n (Mat.cons.{0} Grid n m row rest)) → " +
		"motive_3 0 (Vector.nil.{0} Grid) → " +
		"((n : Nat) → (head : Grid) → (tail : Vector.{0} Grid n) → motive_1 head → motive_3 n tail → " +
		"motive_3 (Nat.succ n) (Vector.cons.{0} Grid n head tail)) → " +
		"(t : Grid) → motive_1 t"
	d := env.Get("Grid.rec")
	if d == nil || !Equal(d.Type, mustTerm(t, want, []string{"u"})) {
		t.Fatalf("Grid.rec : %v", d)
	}
	// A 1x3 grid of leaves: one for the grid, one for each leaf of its only row.
	row := "Vector.cons.{0} Grid 2 Grid.leaf (Vector.cons.{0} Grid 1 Grid.leaf " +
		"(Vector.cons.{0} Grid 0 Grid.leaf (Vector.nil.{0} Grid)))"
	term := "Grid.rec.{1} (fun (_ : Grid) => Nat) " +
		"(fun (r : Nat) (c : Nat) (_ : Mat.{0} Grid r c) => Nat) " +
		"(fun (n : Nat) (_ : Vector.{0} Grid n) => Nat) 1 " +
		"(fun (r : Nat) (c : Nat) (kids : Mat.{0} Grid r c) (k : Nat) => Nat.succ k) " +
		"(fun (n : Nat) => Nat.zero) " +
		"(fun (n : Nat) (m : Nat) (row : Vector.{0} Grid n) (rest : Mat.{0} Grid m n) (rh : Nat) (ih : Nat) => " +
		"Nat.add rh ih) 0 " +
		"(fun (n : Nat) (head : Grid) (tail : Vector.{0} Grid n) (h : Nat) (t : Nat) => Nat.add h t) " +
		"(Grid.node 1 3 (Mat.cons.{0} Grid 3 0 (" + row + ") (Mat.empty.{0} Grid 3)))"
	wantAccept(t, env, nil, term)
	got, err := NewTC(env, nil).Whnf(mustTerm(t, term, nil))
	if err != nil {
		t.Fatal(err)
	}
	if ok, _ := NewTC(env, nil).IsDefEq(got, mustTerm(t, "4", nil)); !ok {
		t.Errorf("leaf count reduced to %v, want 4", got)
	}
}
