package lean

import (
	"fmt"
	"strings"
)

// Nested inductives (inductive.cpp elim_nested_inductive): an occurrence `J As` of an existing inductive type J whose
// parameters mention the types being declared (`inductive Rose | node : List Rose → Rose`) is replaced by an
// auxiliary type with J's constructors instantiated at As. The enlarged block is checked as a mutual one, then the
// auxiliary types are translated back to `J As`. The auxiliary types' recursors are kept as `Main.rec_1`, `Main.rec_2`...

type auxType struct {
	name  string
	j     *Expr   // J.{levels} As
	jName string  // J
	as    []*Expr // As, over the block's parameter fvars
	ctors map[string]string
}

type nestedElim struct {
	tc     *TC
	np     int
	params []*Expr
	lvls   []*Level
	names  map[string]bool
	main   string
	aux    []*auxType
	// auxSpecs[i] belongs to aux[i]; the slot is reserved before the constructors are processed so that numbering
	// follows discovery order even when constructors nest further
	auxSpecs []InductiveSpec
}

func (n *nestedElim) occurs(e *Expr) bool {
	for nm := range n.names {
		if constOccurs(e, nm) {
			return true
		}
	}
	return false
}

// replaceNested rewrites every nested occurrence in body (a constructor type with the parameters instantiated).
func (n *nestedElim) replaceNested(body *Expr) *Expr {
	tc := n.tc
	return replace(body, 0, func(x *Expr, _ int) *Expr {
		if x.Kind != App {
			return nil
		}
		head := GetAppFn(x)
		if head.Kind != Const || n.names[head.Name] {
			return nil
		}
		d := tc.env.Get(head.Name)
		if d == nil || d.Kind != DInduct {
			return nil
		}
		args := GetAppArgs(x)
		if len(args) < d.NumParams {
			return nil
		}
		as := args[:d.NumParams]
		nested := false
		for _, a := range as {
			if n.occurs(a) {
				nested = true
			}
		}
		if !nested {
			return nil
		}
		for _, a := range as {
			if a.HasLooseBVars() {
				tc.fail("invalid nested inductive datatype '%s', nested inductive datatypes parameters cannot contain local variables", head.Name)
			}
		}
		key := MkApp(head, as...)
		var at *auxType
		for _, a := range n.aux {
			if Equal(a.j, key) {
				at = a
			}
		}
		if at == nil {
			at = n.newAux(d, head, as, key)
		}
		return MkApp(MkApp(MkConst(at.name, n.lvls...), n.params...), args[d.NumParams:]...)
	})
}

func (n *nestedElim) newAux(d *Decl, head *Expr, as []*Expr, key *Expr) *auxType {
	tc := n.tc
	at := &auxType{name: fmt.Sprintf("%s._nested_%d", n.main, len(n.aux)+1), j: key, jName: d.Name, as: as, ctors: map[string]string{}}
	n.aux = append(n.aux, at)
	n.names[at.name] = true
	inst := func(ty *Expr, lps []string) *Expr {
		ty = InstantiateLParams(ty, lps, head.Lvls)
		for i := 0; i < d.NumParams; i++ {
			w := ty
			if w.Kind != Pi {
				w = tc.whnf(ty)
			}
			if w.Kind != Pi {
				tc.fail("nested inductive %s: ill-formed parameters", d.Name)
			}
			ty = Instantiate(w.B, as[i])
		}
		return ty
	}
	slot := len(n.auxSpecs)
	spec := InductiveSpec{Name: at.name, NumParams: n.np}
	spec.Type = tc.mkBinding(true, n.params, inst(d.Type, d.LParams))
	n.auxSpecs = append(n.auxSpecs, spec)
	for _, cn := range d.Ctors {
		cd := tc.env.Get(cn)
		short := cn[strings.LastIndex(cn, ".")+1:]
		body := n.replaceNested(inst(cd.Type, cd.LParams))
		an := at.name + "." + short
		at.ctors[an] = cn
		spec.Ctors = append(spec.Ctors, CtorSpec{an, tc.mkBinding(true, n.params, body)})
	}
	n.auxSpecs[slot] = spec
	return at
}

// elimNested returns the block with nested occurrences replaced and the auxiliary types appended; with no nested
// occurrence it returns specs unchanged.
func (tc *TC) elimNested(specs []InductiveSpec) ([]InductiveSpec, *nestedElim) {
	np := specs[0].NumParams
	n := &nestedElim{tc: tc, np: np, names: map[string]bool{}, main: specs[0].Name}
	for _, sp := range specs {
		n.names[sp.Name] = true
	}
	for _, p := range specs[0].LParams {
		n.lvls = append(n.lvls, Param(p))
	}
	ty := specs[0].Type
	for i := 0; i < np; i++ {
		w := ty
		if w.Kind != Pi {
			w = tc.whnf(ty)
		}
		if w.Kind != Pi {
			return specs, nil // addMutual reports the malformed type
		}
		p := tc.mkLocal(w.Name, w.A)
		n.params = append(n.params, p)
		ty = Instantiate(w.B, p)
	}
	out := make([]InductiveSpec, len(specs))
	for k, sp := range specs {
		out[k] = sp
		out[k].Ctors = make([]CtorSpec, len(sp.Ctors))
		for i, c := range sp.Ctors {
			out[k].Ctors[i] = c
			t := c.Type
			ok := true
			for j := 0; j < np && ok; j++ {
				if t.Kind != Pi {
					ok = false
					break
				}
				t = Instantiate(t.B, n.params[j])
			}
			if !ok || !n.occurs(t) {
				continue
			}
			if r := n.replaceNested(t); r != t {
				out[k].Ctors[i].Type = tc.mkBinding(true, n.params, r)
			}
		}
	}
	if len(n.aux) == 0 {
		return specs, nil
	}
	return append(out, n.auxSpecs...), n
}

// back translates a term over the augmented block into the original one.
func (n *nestedElim) back(e *Expr) *Expr {
	byName := map[string]*auxType{}
	byCtor := map[string]*auxType{}
	for _, a := range n.aux {
		byName[a.name] = a
		for an := range a.ctors {
			byCtor[an] = a
		}
	}
	var tr func(*Expr) *Expr
	tr = func(e *Expr) *Expr {
		return replace(e, 0, func(x *Expr, _ int) *Expr {
			head := GetAppFn(x)
			if head.Kind != Const {
				return nil
			}
			args := GetAppArgs(x)
			if a, ok := byName[head.Name]; ok && len(args) >= n.np {
				return MkApp(n.instAs(a.j, args[:n.np]), mapExprs(args[n.np:], tr)...)
			}
			if a, ok := byCtor[head.Name]; ok && len(args) >= n.np {
				c := MkConst(a.ctors[head.Name], GetAppFn(a.j).Lvls...)
				return MkApp(n.instAs(MkApp(c, a.as...), args[:n.np]), mapExprs(args[n.np:], tr)...)
			}
			if strings.HasSuffix(head.Name, ".rec") {
				if a, ok := byName[strings.TrimSuffix(head.Name, ".rec")]; ok {
					return MkApp(MkConst(n.recName(a), head.Lvls...), mapExprs(args, tr)...)
				}
			}
			return nil
		})
	}
	return tr(e)
}

func mapExprs(a []*Expr, f func(*Expr) *Expr) []*Expr {
	out := make([]*Expr, len(a))
	for i, x := range a {
		out[i] = f(x)
	}
	return out
}

// instAs replaces the block parameters in e by args.
func (n *nestedElim) instAs(e *Expr, args []*Expr) *Expr {
	return InstantiateRev(Abstract(e, n.params), args)
}

func (n *nestedElim) recName(a *auxType) string {
	for i, x := range n.aux {
		if x == a {
			return fmt.Sprintf("%s.rec_%d", n.main, i+1)
		}
	}
	return ""
}

// translate maps the declarations of the augmented block back to the original block's: the auxiliary types and their
// constructors disappear, their recursors become Main.rec_i over J's own constructors.
func (n *nestedElim) translate(decls []*Decl, orig []InductiveSpec) []*Decl {
	origCtor := map[string]*Expr{}
	for _, sp := range orig {
		for _, c := range sp.Ctors {
			origCtor[c.Name] = c.Type
		}
	}
	auxByName := map[string]*auxType{}
	for _, a := range n.aux {
		auxByName[a.name] = a
	}
	var out []*Decl
	for _, d := range decls {
		switch d.Kind {
		case DInduct:
			if auxByName[d.Name] != nil {
				continue
			}
		case DCtor:
			if auxByName[d.Induct] != nil {
				continue
			}
			c := *d
			c.Type = origCtor[d.Name]
			out = append(out, &c)
			continue
		case DRec:
			c := *d
			c.Type = n.back(d.Type)
			c.Rules = make([]RecRule, len(d.Rules))
			for i, r := range d.Rules {
				c.Rules[i] = r
				c.Rules[i].RHS = n.back(r.RHS)
			}
			if a := auxByName[d.Induct]; a != nil {
				c.Name = n.recName(a)
				c.Induct = a.jName
				for i := range c.Rules {
					c.Rules[i].Ctor = a.ctors[c.Rules[i].Ctor]
				}
			}
			out = append(out, &c)
			continue
		}
		out = append(out, d)
	}
	return out
}
