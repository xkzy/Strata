package lean

import (
	"fmt"
	"strings"
)

type CtorSpec struct {
	Name string
	Type *Expr
}

// InductiveSpec describes one inductive type of a (non-nested) block: `Type` is Π params, Π indices, Sort u and
// each constructor type starts with the same parameters.
type InductiveSpec struct {
	Name      string
	LParams   []string
	NumParams int
	Type      *Expr
	Ctors     []CtorSpec
}

func constOccurs(e *Expr, name string) bool {
	found := false
	replace(e, 0, func(x *Expr, _ int) *Expr {
		if found {
			return x
		}
		if x.Kind == Const && x.Name == name {
			found = true
			return x
		}
		return nil
	})
	return found
}

type ctorInfo struct {
	name    string
	fields  []*Expr // fvars
	result  *Expr   // I params indices
	indices []*Expr
}

// AddInductive declares one inductive type; see AddMutual.
func (env *Env) AddInductive(spec InductiveSpec) error { return env.AddMutual([]InductiveSpec{spec}) }

// AddMutual checks a block of mutually inductive types (same level parameters, same parameters, same universe) for
// universe levels and strict positivity, then declares the types, their constructors and one recursor per type, each
// taking every type's motive and every constructor's minor premise (inductive.cpp add_inductive).
func (env *Env) AddMutual(specs []InductiveSpec) (err error) {
	if len(specs) == 0 {
		return fmt.Errorf("empty inductive block")
	}
	seen := map[string]bool{}
	claim := func(n string) error {
		if _, dup := env.decls[n]; dup || seen[n] {
			return fmt.Errorf("already declared: %s", n)
		}
		seen[n] = true
		return nil
	}
	name := specs[0].Name
	for _, sp := range specs {
		names := []string{sp.Name, sp.Name + ".rec"}
		for _, c := range sp.Ctors {
			names = append(names, c.Name)
		}
		for _, nm := range names {
			if err := claim(nm); err != nil {
				return err
			}
		}
		if sp.NumParams != specs[0].NumParams || strings.Join(sp.LParams, ",") != strings.Join(specs[0].LParams, ",") {
			return fmt.Errorf("%s: mutual types must share level parameters and parameters", sp.Name)
		}
	}
	work := env.Clone()
	tc := NewTC(work, specs[0].LParams)
	var newDecls []*Decl
	err = tc.protect(func() {
		aug, nest := tc.elimNested(specs)
		newDecls = tc.addMutual(work, aug)
		if nest != nil {
			newDecls = nest.translate(newDecls, specs)
			// the translated recursors must be well typed in an environment that has only the original types
			chk := env.Clone()
			for _, d := range newDecls {
				chk.put(d)
			}
			ctc := NewTC(chk, nil)
			for _, d := range newDecls {
				if d.Kind == DRec {
					ctc.lparams = d.LParams
					ctc.ensureSort(ctc.infer(d.Type, false))
				}
			}
		}
	})
	if err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	for _, d := range newDecls {
		env.put(d)
	}
	return nil
}

type indType struct {
	spec    InductiveSpec
	indices []*Expr
	major   *Expr
	motive  *Expr
	decl    *Decl
	ctors   []ctorInfo
	rec     *Decl
}

func (tc *TC) addMutual(work *Env, specs []InductiveSpec) []*Decl {
	np := specs[0].NumParams
	lparams := specs[0].LParams
	lvls := make([]*Level, len(lparams))
	for i, p := range lparams {
		lvls[i] = Param(p)
	}
	n := len(specs)
	names := map[string]int{}
	for k, sp := range specs {
		names[sp.Name] = k
	}
	occursAny := func(e *Expr) bool {
		for nm := range names {
			if constOccurs(e, nm) {
				return true
			}
		}
		return false
	}

	// parameters (shared) and indices of each type former
	var params []*Expr
	types := make([]*indType, n)
	var resLevel *Level
	for k, sp := range specs {
		tc.ensureSort(tc.infer(sp.Type, false))
		it := &indType{spec: sp}
		types[k] = it
		ty := sp.Type
		pi := 0
		for {
			w := ty
			if w.Kind != Pi {
				w = tc.whnf(InstantiateRev(ty, nil))
			}
			if w.Kind != Pi {
				break
			}
			if pi < np {
				if k == 0 {
					params = append(params, tc.mkLocal(w.Name, w.A))
				} else if !tc.isDefEq(w.A, tc.lctx[params[pi].Name].ty) {
					tc.fail("the parameters of %s differ from those of %s", sp.Name, specs[0].Name)
				}
				ty = Instantiate(w.B, params[pi])
				pi++
				continue
			}
			fv := tc.mkLocal(w.Name, w.A)
			it.indices = append(it.indices, fv)
			ty = Instantiate(w.B, fv)
		}
		if pi != np {
			tc.fail("number of parameters exceeds the arity of the type")
		}
		rs := tc.whnf(ty)
		if rs.Kind != Sort {
			tc.fail("the type of an inductive must end in a Sort")
		}
		if k == 0 {
			resLevel = rs.Lvl
		} else if !LevelEquiv(rs.Lvl, resLevel) {
			tc.fail("mutually inductive types must live in the same universe")
		}
		ctorNames := make([]string, len(sp.Ctors))
		for i, c := range sp.Ctors {
			ctorNames[i] = c.Name
		}
		it.decl = &Decl{Kind: DInduct, Name: sp.Name, LParams: lparams, Type: sp.Type,
			NumParams: np, NumIndices: len(it.indices), Ctors: ctorNames}
		work.put(it.decl)
	}
	isProp := resLevel.IsZero()            // check_constructors: is_zero
	normZero := LevelEquiv(resLevel, Zero) // elim rule and K target: normalizes_to_zero
	indApp := func(it *indType, idx []*Expr) *Expr {
		return MkApp(MkApp(MkConst(it.spec.Name, lvls...), params...), idx...)
	}

	// recursive field detection: Π xs, I_k params ts
	recTarget := func(t *Expr) (xs, ts []*Expr, k int, ok bool) {
		t = tc.whnf(t)
		for t.Kind == Pi {
			x := tc.mkLocal(t.Name, t.A)
			xs = append(xs, x)
			t = tc.whnf(Instantiate(t.B, x))
		}
		if f := GetAppFn(t); f.Kind == Const {
			if k, in := names[f.Name]; in {
				a := GetAppArgs(t)
				if len(a) == np+len(types[k].indices) {
					return xs, a[np:], k, true
				}
			}
		}
		return nil, nil, 0, false
	}

	var checkPositive func(t *Expr)
	checkPositive = func(t *Expr) {
		t = tc.whnf(t)
		if !occursAny(t) {
			return
		}
		if t.Kind == Pi {
			if occursAny(t.A) {
				tc.fail("arg #%d of '%s' has a non positive occurrence of the datatypes being declared", 0, "constructor")
			}
			x := tc.mkLocal(t.Name, t.A)
			checkPositive(Instantiate(t.B, x))
			return
		}
		f := GetAppFn(t)
		a := GetAppArgs(t)
		k, in := names[f.Name]
		if f.Kind != Const || !in || len(a) != np+len(types[k].indices) {
			tc.fail("invalid nested occurrence of the datatype being declared")
		}
		for i := 0; i < np; i++ {
			if !tc.isDefEq(a[i], params[i]) {
				tc.fail("invalid occurrence: the parameters of the datatype must stay fixed")
			}
		}
		for _, ix := range a[np:] {
			if occursAny(ix) {
				tc.fail("invalid occurrence: an index contains the datatype being declared")
			}
		}
	}

	var newDecls []*Decl
	for _, it := range types {
		I := it.spec.Name
		for ci, c := range it.spec.Ctors {
			tc.ensureSort(tc.infer(c.Type, false))
			t := c.Type
			info := ctorInfo{name: c.Name}
			for i := 0; ; i++ {
				w := t
				if w.Kind != Pi {
					w = tc.whnf(t)
				}
				if w.Kind != Pi {
					t = w
					break
				}
				if i < np {
					if !tc.isDefEq(w.A, tc.lctx[params[i].Name].ty) {
						tc.fail("constructor %s: parameter %d does not match the inductive type's", c.Name, i+1)
					}
					t = Instantiate(w.B, params[i])
					continue
				}
				s := tc.ensureSort(tc.infer(w.A, false))
				if !isProp && !LevelGeq(resLevel, s) {
					tc.fail("constructor %s: universe level %s of field '%s' is too big for the inductive type's level %s", c.Name, s, w.Name, resLevel)
				}
				checkPositive(w.A)
				fv := tc.mkLocal(w.Name, w.A)
				info.fields = append(info.fields, fv)
				t = Instantiate(w.B, fv)
			}
			if h := GetAppFn(t); h.Kind != Const || h.Name != I {
				tc.fail("constructor %s must return the inductive type", c.Name)
			}
			ra := GetAppArgs(t)
			if len(ra) != np+len(it.indices) {
				tc.fail("constructor %s: wrong number of arguments to the inductive type", c.Name)
			}
			for i := 0; i < np; i++ {
				if !Equal(ra[i], params[i]) {
					tc.fail("constructor %s: the result must use the declared parameters", c.Name)
				}
			}
			info.indices = ra[np:]
			for _, ix := range info.indices {
				if occursAny(ix) {
					tc.fail("constructor %s: an index contains the datatype being declared", c.Name)
				}
			}
			info.result = t
			it.ctors = append(it.ctors, info)
			newDecls = append(newDecls, &Decl{Kind: DCtor, Name: c.Name, LParams: lparams, Type: c.Type,
				Induct: I, CIdx: ci, NumFields: len(info.fields)})
		}
	}
	isRec := false
	for _, it := range types {
		for _, ci := range it.ctors {
			for _, f := range ci.fields {
				if _, _, _, ok := recTarget(tc.lctx[f.Name].ty); ok {
					isRec = true
				}
			}
		}
	}
	for _, it := range types {
		it.decl.IsRec = isRec
	}

	// elimination universe (elim_only_at_universe_zero)
	elimProp := false
	switch {
	case resLevel.IsNeverZero():
	case n > 1:
		elimProp = true
	case len(types[0].ctors) > 1:
		elimProp = true
	case len(types[0].ctors) == 1:
		c0 := types[0].ctors[0]
		for _, f := range c0.fields {
			if tc.isProp(tc.lctx[f.Name].ty) {
				continue
			}
			// it must be one of the result type's arguments itself, not merely occur inside one
			inIdx := false
			for _, ix := range c0.indices {
				if ix.Kind == FVar && ix.Name == f.Name {
					inIdx = true
				}
			}
			if !inIdx {
				elimProp = true
			}
		}
	}
	var elimLevel *Level
	recLParams := lparams
	if elimProp {
		elimLevel = Zero
	} else {
		u := "u"
		for k := 1; ; k++ {
			clash := false
			for _, p := range lparams {
				if p == u {
					clash = true
				}
			}
			if !clash {
				break
			}
			u = fmt.Sprintf("u_%d", k)
		}
		elimLevel = Param(u)
		recLParams = append([]string{u}, lparams...)
	}
	recLvls := make([]*Level, 0, len(recLParams))
	if !elimProp {
		recLvls = append(recLvls, elimLevel)
	}
	recLvls = append(recLvls, lvls...)

	// one motive per type: C_k : Π indices, Π (t : I_k params indices), Sort elim
	for k, it := range types {
		it.major = tc.mkLocal("t", indApp(it, it.indices))
		motiveTy := tc.mkBinding(true, append(append([]*Expr{}, it.indices...), it.major), MkSort(elimLevel))
		mname := "motive"
		if n > 1 {
			mname = fmt.Sprintf("motive_%d", k+1)
		}
		it.motive = tc.mkLocal(mname, motiveTy)
	}

	// minor premises, in constructor order across the block
	type recField struct {
		field, ih *Expr
		xs, ts    []*Expr
		k         int
	}
	type minorInfo struct {
		it    *indType
		ci    ctorInfo
		minor *Expr
		ihs   []recField
	}
	var minors []*minorInfo
	for _, it := range types {
		for _, ci := range it.ctors {
			mi := &minorInfo{it: it, ci: ci}
			var ihFvars []*Expr
			for _, f := range ci.fields {
				xs, ts, k, ok := recTarget(tc.lctx[f.Name].ty)
				if !ok {
					continue
				}
				capp := MkApp(MkApp(types[k].motive, ts...), MkApp(f, xs...))
				ihTy := tc.mkBinding(true, xs, capp)
				ih := tc.mkLocal(tc.lctx[f.Name].name+"_ih", ihTy)
				ihFvars = append(ihFvars, ih)
				mi.ihs = append(mi.ihs, recField{f, ih, xs, ts, k})
			}
			cApp := MkApp(MkApp(MkConst(ci.name, lvls...), params...), ci.fields...)
			concl := MkApp(MkApp(it.motive, ci.indices...), cApp)
			short := ci.name[strings.LastIndex(ci.name, ".")+1:]
			mi.minor = tc.mkLocal(short, tc.mkBinding(true, append(append([]*Expr{}, ci.fields...), ihFvars...), concl))
			minors = append(minors, mi)
		}
	}

	all := append([]*Expr{}, params...)
	for _, it := range types {
		all = append(all, it.motive)
	}
	for _, mi := range minors {
		all = append(all, mi.minor)
	}
	recConst := func(it *indType) *Expr { return MkConst(it.spec.Name+".rec", recLvls...) }

	k := n == 1 && normZero && len(types[0].ctors) == 1 && len(types[0].ctors[0].fields) == 0
	for _, it := range types {
		recTyBody := MkApp(MkApp(it.motive, it.indices...), it.major)
		recTy := tc.mkBinding(true, append(append(append([]*Expr{}, all...), it.indices...), it.major), recTyBody)
		var rules []RecRule
		for _, mi := range minors {
			if mi.it != it {
				continue
			}
			var ihVals []*Expr
			for _, rf := range mi.ihs {
				call := MkApp(MkApp(MkApp(recConst(types[rf.k]), all...), rf.ts...), MkApp(rf.field, rf.xs...))
				ihVals = append(ihVals, tc.mkBinding(false, rf.xs, call))
			}
			rhsBody := MkApp(MkApp(mi.minor, mi.ci.fields...), ihVals...)
			rules = append(rules, RecRule{Ctor: mi.ci.name, NFields: len(mi.ci.fields),
				RHS: tc.mkBinding(false, append(append([]*Expr{}, all...), mi.ci.fields...), rhsBody)})
		}
		it.rec = &Decl{Kind: DRec, Name: it.spec.Name + ".rec", LParams: recLParams, Type: recTy, Induct: it.spec.Name,
			NumParams: np, NumIndices: len(it.indices), NumMotives: n, NumMinors: len(minors), Rules: rules, K: k}
	}
	// the generated recursors must themselves be well typed
	for _, d := range newDecls {
		work.put(d)
	}
	for _, it := range types {
		work.put(it.rec)
	}
	tc.lparams = recLParams
	for _, it := range types {
		tc.ensureSort(tc.infer(it.rec.Type, false))
	}
	tc.lparams = lparams
	for _, it := range types {
		newDecls = append(newDecls, it.rec)
	}
	out := make([]*Decl, 0, len(newDecls)+n)
	for _, it := range types {
		out = append(out, it.decl)
	}
	return append(out, newDecls...)
}
