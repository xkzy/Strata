package lean

import (
	"fmt"
	"sort"
)

type DeclKind uint8

const (
	DAxiom DeclKind = iota
	DDefn
	DThm
	DOpaque
	DInduct
	DCtor
	DRec
	DQuot
)

type HintKind uint8

const (
	HOpaque HintKind = iota
	HAbbrev
	HRegular
)

type Hints struct {
	Kind   HintKind
	Height int
}

type RecRule struct {
	Ctor    string
	NFields int
	RHS     *Expr
}

type Decl struct {
	Kind    DeclKind
	Name    string
	LParams []string
	Type    *Expr
	Value   *Expr
	Hints   Hints
	// inductive
	NumParams, NumIndices int
	Ctors                 []string
	IsRec                 bool
	// constructor
	Induct    string
	CIdx      int
	NumFields int
	// recursor
	NumMotives, NumMinors int
	Rules                 []RecRule
	K                     bool
	// quotient: "Quot", "Quot.mk", "Quot.lift", "Quot.ind"
	QuotKind string
}

func (d *Decl) HasValue() bool { return d.Kind == DDefn || d.Kind == DThm }

type Env struct {
	decls map[string]*Decl
	order []string
}

func NewEnv() *Env { return &Env{decls: map[string]*Decl{}} }

func (e *Env) Get(n string) *Decl { return e.decls[n] }

// Clone shares the declarations (they are immutable once added) so a request can add its own without touching the prelude.
func (e *Env) Clone() *Env {
	c := &Env{decls: make(map[string]*Decl, len(e.decls)+8), order: append([]string(nil), e.order...)}
	for k, v := range e.decls {
		c.decls[k] = v
	}
	return c
}

func (e *Env) put(d *Decl) {
	e.decls[d.Name] = d
	e.order = append(e.order, d.Name)
}

// Axioms lists the declarations the kernel accepted without a proof, sorted.
func (e *Env) Axioms() []string {
	var out []string
	for n, d := range e.decls {
		if d.Kind == DAxiom {
			out = append(out, n)
		}
	}
	sort.Strings(out)
	return out
}

func (e *Env) checkHeader(name string, lparams []string, ty *Expr) (*TC, error) {
	if _, dup := e.decls[name]; dup {
		return nil, fmt.Errorf("already declared: %s", name)
	}
	seen := map[string]bool{}
	for _, p := range lparams {
		if seen[p] {
			return nil, fmt.Errorf("%s: duplicate universe parameter %s", name, p)
		}
		seen[p] = true
	}
	if ty.HasLooseBVars() || ty.HasFVar() {
		return nil, fmt.Errorf("%s: type is not closed", name)
	}
	tc := NewTC(e, lparams)
	return tc, tc.protect(func() { tc.ensureSort(tc.infer(ty, false)) })
}

func (e *Env) AddAxiom(name string, lparams []string, ty *Expr) error {
	if _, err := e.checkHeader(name, lparams, ty); err != nil {
		return err
	}
	e.put(&Decl{Kind: DAxiom, Name: name, LParams: lparams, Type: ty})
	return nil
}

func (e *Env) height(v *Expr) int {
	h := 0
	var walk func(x *Expr)
	walk = func(x *Expr) {
		switch x.Kind {
		case Const:
			if d := e.decls[x.Name]; d != nil && d.Kind == DDefn && d.Hints.Height > h {
				h = d.Hints.Height
			}
		case App, Lam, Pi:
			walk(x.A)
			walk(x.B)
		case Let:
			walk(x.A)
			walk(x.B)
			walk(x.C)
		case Proj:
			walk(x.A)
		}
	}
	walk(v)
	return h + 1
}

func (e *Env) addValued(kind DeclKind, name string, lparams []string, ty, val *Expr, hints *Hints) error {
	tc, err := e.checkHeader(name, lparams, ty)
	if err != nil {
		return err
	}
	if val.HasLooseBVars() || val.HasFVar() {
		return fmt.Errorf("%s: value is not closed", name)
	}
	err = tc.protect(func() {
		if kind == DThm && !tc.isProp(ty) {
			tc.fail("theorem %s: type is not a proposition", name)
		}
		vt := tc.infer(val, false)
		if !tc.isDefEq(vt, ty) {
			tc.fail("%s: value has type %s, expected %s", name, vt, ty)
		}
	})
	if err != nil {
		return err
	}
	d := &Decl{Kind: kind, Name: name, LParams: lparams, Type: ty, Value: val, Hints: Hints{Kind: HOpaque}}
	if kind == DDefn {
		d.Hints = Hints{Kind: HRegular, Height: e.height(val)}
		if hints != nil {
			d.Hints = *hints
		}
	}
	e.put(d)
	return nil
}

func (e *Env) AddDefn(name string, lparams []string, ty, val *Expr) error {
	return e.addValued(DDefn, name, lparams, ty, val, nil)
}

func (e *Env) AddAbbrev(name string, lparams []string, ty, val *Expr) error {
	return e.addValued(DDefn, name, lparams, ty, val, &Hints{Kind: HAbbrev})
}

func (e *Env) AddTheorem(name string, lparams []string, ty, val *Expr) error {
	return e.addValued(DThm, name, lparams, ty, val, nil)
}

// AddOpaque declares a constant with a checked body the kernel never unfolds.
func (e *Env) AddOpaque(name string, lparams []string, ty, val *Expr) error {
	return e.addValued(DOpaque, name, lparams, ty, val, nil)
}

func sortedCopy(a []string) []string {
	b := append([]string(nil), a...)
	sort.Strings(b)
	return b
}
