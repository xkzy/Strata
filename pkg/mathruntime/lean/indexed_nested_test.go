package lean

import "testing"

// Vector is Lean's indexed family (`Vector α n`), not a nested one. Above it, a type that nests an INDEXED occurrence
// - `Vector Tree 2`, or `Vector Forest n` with the index bound in the constructor - is what Lean lifts into an
// indexed auxiliary type and eliminates together with the main type. These tests pin that path: the auxiliary's
// motive takes the index before the value, one auxiliary serves every index of the same family, and computation
// through it reduces on small finite values.

func envWithVector(t *testing.T) *Env {
	t.Helper()
	env := mustPrelude(t).Clone()
	spec := InductiveSpec{
		Name:      "Vector",
		LParams:   []string{"u"},
		NumParams: 1,
		Type:      mustTerm(t, "(α : Sort (u+1)) → Nat → Sort (u+1)", []string{"u"}),
	}
	spec.Ctors = []CtorSpec{
		{"Vector.nil", mustTerm(t, "(α : Type u) → Vector.{u} α 0", []string{"u"})},
		{"Vector.cons", mustTerm(t, "(α : Type u) → (n : Nat) → (head : α) → (tail : Vector.{u} α n) → Vector.{u} α (Nat.succ n)", []string{"u"})},
	}
	if err := env.AddInductive(spec); err != nil {
		t.Fatal(err)
	}
	return env
}

func TestVectorRecursor(t *testing.T) {
	env := envWithVector(t)
	// Lean: the motive takes the index and the value, `nil` is the n = 0 case, and `cons` takes the tail's motive.
	want := "(α : Sort (u+1)) → (motive : (n : Nat) → Vector.{u} α n → Sort u_1) → " +
		"motive 0 (Vector.nil.{u} α) → " +
		"((n : Nat) → (head : α) → (tail : Vector.{u} α n) → motive n tail → " +
		"motive (Nat.succ n) (Vector.cons.{u} α n head tail)) → " +
		"(n : Nat) → (t : Vector.{u} α n) → motive n t"
	d := env.Get("Vector.rec")
	if d == nil {
		t.Fatal("Vector.rec missing")
	}
	if !Equal(d.Type, mustTerm(t, want, []string{"u", "u_1"})) {
		t.Fatalf("Vector.rec : %v", d.Type)
	}
}

// sizeTerm is a node count through the auxiliary recursor: the tree motive is a Nat, the vector motive is a Nat (the
// sum of the trees in it), and the `cons` minor adds the head's count to the tail's. nodeMinor is the new type's own
// constructor minor, which takes the index only when the constructor binds it.
func sizeTerm(main, nodeMinor string) string {
	return "(fun (_ : " + main + ") => Nat) (fun (n : Nat) (_ : Vector.{0} " + main + " n) => Nat) 1 " +
		nodeMinor + " 0 " +
		"(fun (n : Nat) (head : " + main + ") (tail : Vector.{0} " + main + " n) (h : Nat) (t : Nat) => Nat.add h t) "
}

func TestNestedThroughIndexedFamily(t *testing.T) {
	env := envWithVector(t)
	spec := mkSpec(t, "Tree", 0, "Type",
		[2]string{"Tree.leaf", "Tree"},
		[2]string{"Tree.node", "(kids : Vector.{0} Tree 2) → Tree"})
	if err := env.AddInductive(spec); err != nil {
		t.Fatalf("a nested occurrence of an indexed family: %v", err)
	}
	if env.Get("Tree._nested_1") != nil {
		t.Error("the auxiliary indexed type leaked into the environment")
	}
	// Both motives: the tree, and the auxiliary vector indexed by its length.
	rec := "(motive_1 : Tree → Sort u) → (motive_2 : (n : Nat) → Vector.{0} Tree n → Sort u) → " +
		"motive_1 Tree.leaf → " +
		"((kids : Vector.{0} Tree 2) → motive_2 2 kids → motive_1 (Tree.node kids)) → " +
		"motive_2 0 (Vector.nil.{0} Tree) → " +
		"((n : Nat) → (head : Tree) → (tail : Vector.{0} Tree n) → motive_1 head → motive_2 n tail → " +
		"motive_2 (Nat.succ n) (Vector.cons.{0} Tree n head tail)) → "
	for name, ty := range map[string]string{
		"Tree.rec":   rec + "(t : Tree) → motive_1 t",
		"Tree.rec_1": rec + "(n : Nat) → (t : Vector.{0} Tree n) → motive_2 n t",
	} {
		d := env.Get(name)
		if d == nil {
			t.Fatalf("%s missing", name)
		}
		if !Equal(d.Type, mustTerm(t, ty, []string{"u"})) {
			t.Errorf("%s : %v", name, d.Type)
		}
	}

	// node [leaf, node [leaf, leaf]] = 1 + 1 + 3
	two := "Tree.node (Vector.cons.{0} Tree 1 Tree.leaf (Vector.cons.{0} Tree 0 Tree.leaf (Vector.nil.{0} Tree)))"
	tree := "Tree.node (Vector.cons.{0} Tree 1 Tree.leaf (Vector.cons.{0} Tree 0 (" + two + ") (Vector.nil.{0} Tree)))"
	term := "Tree.rec.{1} " +
		sizeTerm("Tree", "(fun (kids : Vector.{0} Tree 2) (s : Nat) => Nat.succ s)") + "(" + tree + ")"
	wantAccept(t, env, nil, term)
	got, err := NewTC(env, nil).Whnf(mustTerm(t, term, nil))
	if err != nil {
		t.Fatal(err)
	}
	if ok, _ := NewTC(env, nil).IsDefEq(got, mustTerm(t, "5", nil)); !ok {
		t.Errorf("node count reduced to %v, want 5", got)
	}
}

// The index of the nested occurrence may be bound in the constructor (`Vector Forest n`), which is the shape Lean's
// own `Vector` users write; the auxiliary is the same indexed type.
func TestNestedIndexedWithABoundIndex(t *testing.T) {
	env := envWithVector(t)
	spec := mkSpec(t, "Forest", 0, "Type",
		[2]string{"Forest.leaf", "Forest"},
		[2]string{"Forest.node", "(n : Nat) → (kids : Vector.{0} Forest n) → Forest"})
	if err := env.AddInductive(spec); err != nil {
		t.Fatalf("a nested occurrence with a bound index: %v", err)
	}
	rec := "(motive_1 : Forest → Sort u) → (motive_2 : (n : Nat) → Vector.{0} Forest n → Sort u) → " +
		"motive_1 Forest.leaf → " +
		"((n : Nat) → (kids : Vector.{0} Forest n) → motive_2 n kids → motive_1 (Forest.node n kids)) → " +
		"motive_2 0 (Vector.nil.{0} Forest) → " +
		"((n : Nat) → (head : Forest) → (tail : Vector.{0} Forest n) → motive_1 head → motive_2 n tail → " +
		"motive_2 (Nat.succ n) (Vector.cons.{0} Forest n head tail)) → "
	if d := env.Get("Forest.rec"); d == nil || !Equal(d.Type, mustTerm(t, rec+"(t : Forest) → motive_1 t", []string{"u"})) {
		t.Fatalf("Forest.rec : %v", d)
	}
	// Forest.node 2 [leaf, Forest.node 0 []] has three nodes
	inner := "Forest.node 0 (Vector.nil.{0} Forest)"
	forest := "Forest.node 2 (Vector.cons.{0} Forest 1 Forest.leaf (Vector.cons.{0} Forest 0 (" + inner + ") (Vector.nil.{0} Forest)))"
	term := "Forest.rec.{1} " +
		sizeTerm("Forest", "(fun (n : Nat) (kids : Vector.{0} Forest n) (s : Nat) => Nat.succ s)") + "(" + forest + ")"
	wantAccept(t, env, nil, term)
	got, err := NewTC(env, nil).Whnf(mustTerm(t, term, nil))
	if err != nil {
		t.Fatal(err)
	}
	if ok, _ := NewTC(env, nil).IsDefEq(got, mustTerm(t, "3", nil)); !ok {
		t.Errorf("node count reduced to %v, want 3", got)
	}
}

// Two occurrences of the same family at different indices share ONE auxiliary type (it is indexed, so one serves
// every length) - Lean's `elim_nested_inductive` does the same, and a kernel that made two would duplicate the
// recursor's motives.
func TestTwoNestedIndicesShareOneAuxiliary(t *testing.T) {
	env := envWithVector(t)
	spec := mkSpec(t, "Expr", 0, "Type",
		[2]string{"Expr.leaf", "Expr"},
		[2]string{"Expr.pair", "(kids : Vector.{0} Expr 2) → (more : Vector.{0} Expr 1) → Expr"})
	if err := env.AddInductive(spec); err != nil {
		t.Fatalf("two nested occurrences: %v", err)
	}
	if env.Get("Expr.rec_2") != nil {
		t.Error("two auxiliary recursors for one family: an indexed auxiliary serves every index")
	}
	want := "(motive_1 : Expr → Sort u) → (motive_2 : (n : Nat) → Vector.{0} Expr n → Sort u) → " +
		"motive_1 Expr.leaf → " +
		"((kids : Vector.{0} Expr 2) → (more : Vector.{0} Expr 1) → motive_2 2 kids → motive_2 1 more → " +
		"motive_1 (Expr.pair kids more)) → " +
		"motive_2 0 (Vector.nil.{0} Expr) → " +
		"((n : Nat) → (head : Expr) → (tail : Vector.{0} Expr n) → motive_1 head → motive_2 n tail → " +
		"motive_2 (Nat.succ n) (Vector.cons.{0} Expr n head tail)) → " +
		"(t : Expr) → motive_1 t"
	d := env.Get("Expr.rec")
	if d == nil || !Equal(d.Type, mustTerm(t, want, []string{"u"})) {
		t.Fatalf("Expr.rec : %v", d)
	}
}

// Two different families nested in one declaration: an indexed one (`Vector Node 2`) and a plain one (`List Node`).
// Each becomes its own auxiliary type, numbered in the order the constructor mentions them.
func TestTwoDifferentNestedFamilies(t *testing.T) {
	env := envWithList(t)
	if err := env.AddInductive(InductiveSpec{Name: "Vector", LParams: []string{"u"}, NumParams: 1,
		Type: mustTerm(t, "(α : Sort (u+1)) → Nat → Sort (u+1)", []string{"u"}),
		Ctors: []CtorSpec{
			{"Vector.nil", mustTerm(t, "(α : Type u) → Vector.{u} α 0", []string{"u"})},
			{"Vector.cons", mustTerm(t, "(α : Type u) → (n : Nat) → (head : α) → (tail : Vector.{u} α n) → Vector.{u} α (Nat.succ n)", []string{"u"})}}}); err != nil {
		t.Fatal(err)
	}
	spec := mkSpec(t, "Node", 0, "Type",
		[2]string{"Node.leaf", "Node"},
		[2]string{"Node.node", "(kids : Vector.{0} Node 2) → (more : List.{0} Node) → Node"})
	if err := env.AddInductive(spec); err != nil {
		t.Fatalf("two nested families: %v", err)
	}
	common := "(motive_1 : Node → Sort u) → (motive_2 : (n : Nat) → Vector.{0} Node n → Sort u) → " +
		"(motive_3 : List.{0} Node → Sort u) → " +
		"motive_1 Node.leaf → " +
		"((kids : Vector.{0} Node 2) → (more : List.{0} Node) → motive_2 2 kids → motive_3 more → " +
		"motive_1 (Node.node kids more)) → " +
		"motive_2 0 (Vector.nil.{0} Node) → " +
		"((n : Nat) → (head : Node) → (tail : Vector.{0} Node n) → motive_1 head → motive_2 n tail → " +
		"motive_2 (Nat.succ n) (Vector.cons.{0} Node n head tail)) → " +
		"motive_3 (List.nil.{0} Node) → " +
		"((head : Node) → (tail : List.{0} Node) → motive_1 head → motive_3 tail → " +
		"motive_3 (List.cons.{0} Node head tail)) → "
	for name, concl := range map[string]string{
		"Node.rec":   "(t : Node) → motive_1 t",
		"Node.rec_1": "(n : Nat) → (t : Vector.{0} Node n) → motive_2 n t",
		"Node.rec_2": "(t : List.{0} Node) → motive_3 t",
	} {
		d := env.Get(name)
		if d == nil {
			t.Fatalf("%s missing", name)
		}
		if !Equal(d.Type, mustTerm(t, common+concl, []string{"u"})) {
			t.Errorf("%s : %v", name, d.Type)
		}
	}

	// node [leaf, leaf] [leaf] = 1 + (1 + 1) + 1, through both auxiliaries at once
	kids := "Vector.cons.{0} Node 1 Node.leaf (Vector.cons.{0} Node 0 Node.leaf (Vector.nil.{0} Node))"
	more := "List.cons.{0} Node Node.leaf (List.nil.{0} Node)"
	term := "Node.rec.{1} (fun (_ : Node) => Nat) " +
		"(fun (n : Nat) (_ : Vector.{0} Node n) => Nat) (fun (_ : List.{0} Node) => Nat) 1 " +
		"(fun (kids : Vector.{0} Node 2) (more : List.{0} Node) (k : Nat) (m : Nat) => Nat.succ (Nat.add k m)) 0 " +
		"(fun (n : Nat) (head : Node) (tail : Vector.{0} Node n) (h : Nat) (t : Nat) => Nat.add h t) 0 " +
		"(fun (head : Node) (tail : List.{0} Node) (h : Nat) (t : Nat) => Nat.add h t) " +
		"(Node.node (" + kids + ") (" + more + "))"
	wantAccept(t, env, nil, term)
	got, err := NewTC(env, nil).Whnf(mustTerm(t, term, nil))
	if err != nil {
		t.Fatal(err)
	}
	if ok, _ := NewTC(env, nil).IsDefEq(got, mustTerm(t, "4", nil)); !ok {
		t.Errorf("node count reduced to %v, want 4", got)
	}
}

// The index of the nested occurrence may be a function of a constructor argument (`Vector Bar (Nat.succ n)`): the
// auxiliary is still the same indexed family, and its constructor minor receives `n` and the vector at `n+1`.
func TestNestedIndexIsAFunctionOfAnArgument(t *testing.T) {
	env := envWithVector(t)
	spec := mkSpec(t, "Bar", 0, "Type",
		[2]string{"Bar.leaf", "Bar"},
		[2]string{"Bar.node", "(n : Nat) → (kids : Vector.{0} Bar (Nat.succ n)) → Bar"})
	if err := env.AddInductive(spec); err != nil {
		t.Fatalf("a nested index that mentions a constructor argument: %v", err)
	}
	want := "(motive_1 : Bar → Sort u) → (motive_2 : (n : Nat) → Vector.{0} Bar n → Sort u) → " +
		"motive_1 Bar.leaf → " +
		"((n : Nat) → (kids : Vector.{0} Bar (Nat.succ n)) → motive_2 (Nat.succ n) kids → motive_1 (Bar.node n kids)) → " +
		"motive_2 0 (Vector.nil.{0} Bar) → " +
		"((n : Nat) → (head : Bar) → (tail : Vector.{0} Bar n) → motive_1 head → motive_2 n tail → " +
		"motive_2 (Nat.succ n) (Vector.cons.{0} Bar n head tail)) → " +
		"(t : Bar) → motive_1 t"
	d := env.Get("Bar.rec")
	if d == nil || !Equal(d.Type, mustTerm(t, want, []string{"u"})) {
		t.Fatalf("Bar.rec : %v", d)
	}
	kids := "Vector.cons.{0} Bar 1 Bar.leaf (Vector.cons.{0} Bar 0 Bar.leaf (Vector.nil.{0} Bar))"
	term := "Bar.rec.{1} (fun (_ : Bar) => Nat) (fun (n : Nat) (_ : Vector.{0} Bar n) => Nat) 1 " +
		"(fun (n : Nat) (kids : Vector.{0} Bar (Nat.succ n)) (k : Nat) => Nat.succ k) 0 " +
		"(fun (n : Nat) (head : Bar) (tail : Vector.{0} Bar n) (h : Nat) (t : Nat) => Nat.add h t) " +
		"(Bar.node 1 (" + kids + "))"
	wantAccept(t, env, nil, term)
	got, err := NewTC(env, nil).Whnf(mustTerm(t, term, nil))
	if err != nil {
		t.Fatal(err)
	}
	if ok, _ := NewTC(env, nil).IsDefEq(got, mustTerm(t, "3", nil)); !ok {
		t.Errorf("node count reduced to %v, want 3", got)
	}
}
