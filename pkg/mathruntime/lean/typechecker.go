package lean

import (
	"fmt"
	"math/big"
	"time"
)

type kernelError struct {
	msg    string
	budget bool // the checker gave up (step or time budget), which says nothing about the claim
}

func (k kernelError) Error() string { return "kernel: " + k.msg }

// IsBudgetError reports whether err is the checker running out of steps or time.
func IsBudgetError(err error) bool {
	k, ok := err.(kernelError)
	return ok && k.budget
}

type localDecl struct {
	name string
	ty   *Expr
}

// TC is the kernel type checker (type_checker.cpp). One TC serves one declaration or request.
type TC struct {
	env      *Env
	lctx     map[string]localDecl
	lparams  []string
	inferC   [2]map[*Expr]*Expr
	whnfCore map[*Expr]*Expr
	whnfC    map[*Expr]*Expr
	eqC      map[[2]*Expr]bool
	failC    map[[2]*Expr]bool
	nextID   int

	MaxSteps int       // 0 = unlimited
	Deadline time.Time // zero = none
	steps    int
}

// natMaxBits mirrors LEAN_NAT_MAX_SIZE (128MB) scaled down: a request is not the place to compute 100MB numerals.
const natMaxBits = 1 << 22

func NewTC(env *Env, lparams []string) *TC {
	return &TC{env: env, lctx: map[string]localDecl{}, lparams: lparams,
		inferC:   [2]map[*Expr]*Expr{{}, {}},
		whnfCore: map[*Expr]*Expr{}, whnfC: map[*Expr]*Expr{}, eqC: map[[2]*Expr]bool{}, failC: map[[2]*Expr]bool{}}
}

func (tc *TC) fail(f string, a ...any) { panic(kernelError{msg: fmt.Sprintf(f, a...)}) }

func (tc *TC) tick() {
	tc.steps++
	if tc.MaxSteps > 0 && tc.steps > tc.MaxSteps {
		panic(kernelError{"step budget exceeded", true})
	}
	if !tc.Deadline.IsZero() && time.Now().After(tc.Deadline) {
		panic(kernelError{"deadline exceeded", true})
	}
}

func (tc *TC) protect(f func()) (err error) {
	defer func() {
		if r := recover(); r != nil {
			if k, ok := r.(kernelError); ok {
				err = k
				return
			}
			panic(r)
		}
	}()
	f()
	return nil
}

// Public entry points.

func (tc *TC) Check(e *Expr) (ty *Expr, err error) {
	err = tc.protect(func() { ty = tc.infer(e, false) })
	return
}

func (tc *TC) InferOnly(e *Expr) (ty *Expr, err error) {
	err = tc.protect(func() { ty = tc.infer(e, true) })
	return
}

func (tc *TC) IsDefEq(a, b *Expr) (ok bool, err error) {
	err = tc.protect(func() { ok = tc.isDefEq(a, b) })
	return
}

func (tc *TC) Whnf(e *Expr) (r *Expr, err error) {
	err = tc.protect(func() { r = tc.whnf(e) })
	return
}

// Steps reports how much work the checker did, for budgets and metrics.
func (tc *TC) Steps() int { return tc.steps }

func (tc *TC) mkLocal(name string, ty *Expr) *Expr {
	tc.nextID++
	id := fmt.Sprintf("_kernel_fresh.%d", tc.nextID)
	tc.lctx[id] = localDecl{name, ty}
	return MkFVar(id)
}

func (tc *TC) mkBinding(pi bool, fvars []*Expr, body *Expr) *Expr {
	r := Abstract(body, fvars)
	for i := len(fvars) - 1; i >= 0; i-- {
		d := tc.lctx[fvars[i].Name]
		dom := Abstract(d.ty, fvars[:i])
		if pi {
			r = MkPi(d.name, dom, r)
		} else {
			r = MkLam(d.name, dom, r)
		}
	}
	return r
}

func (tc *TC) ensureSort(e *Expr) *Level {
	if e.Kind == Sort {
		return e.Lvl
	}
	if w := tc.whnf(e); w.Kind == Sort {
		return w.Lvl
	}
	tc.fail("type expected, got %s", e)
	return nil
}

func (tc *TC) ensurePi(e *Expr) *Expr {
	if e.Kind == Pi {
		return e
	}
	if w := tc.whnf(e); w.Kind == Pi {
		return w
	}
	tc.fail("function expected, got %s", e)
	return nil
}

func (tc *TC) isProp(ty *Expr) bool {
	t := tc.whnf(tc.infer(ty, true))
	return t.Kind == Sort && LevelEquiv(t.Lvl, Zero)
}

func (tc *TC) infer(e *Expr, inferOnly bool) *Expr {
	if e.HasLooseBVars() {
		tc.fail("loose bound variable in %s", e)
	}
	i := 0
	if inferOnly {
		i = 1
	}
	if r, ok := tc.inferC[i][e]; ok {
		return r
	}
	tc.tick()
	var r *Expr
	switch e.Kind {
	case FVar:
		d, ok := tc.lctx[e.Name]
		if !ok {
			tc.fail("unknown free variable %s", e.Name)
		}
		r = d.ty
	case Sort:
		if !inferOnly {
			tc.checkLevel(e.Lvl)
		}
		r = MkSort(Succ(e.Lvl))
	case Const:
		r = tc.inferConst(e, inferOnly)
	case Lam:
		r = tc.inferLambda(e, inferOnly)
	case Pi:
		r = tc.inferPi(e, inferOnly)
	case App:
		r = tc.inferApp(e, inferOnly)
	case Let:
		if !inferOnly {
			tc.ensureSort(tc.infer(e.A, false))
			if vt := tc.infer(e.B, false); !tc.isDefEq(vt, e.A) {
				tc.fail("let value has type %s, expected %s", vt, e.A)
			}
		}
		r = tc.infer(Instantiate(e.C, e.B), inferOnly)
	case NatLit:
		if e.Nat.BitLen() > natMaxBits {
			tc.fail("numeral too large")
		}
		r = MkConst("Nat")
	case Proj:
		r = tc.inferProj(e, inferOnly)
	default:
		tc.fail("cannot infer the type of %s", e)
	}
	tc.inferC[i][e] = r
	return r
}

func (tc *TC) checkLevel(l *Level) {
	if n, ok := levelParamsOK(l, tc.lparams); !ok {
		tc.fail("undeclared universe parameter %s", n)
	}
}

func (tc *TC) inferConst(e *Expr, inferOnly bool) *Expr {
	d := tc.env.Get(e.Name)
	if d == nil {
		tc.fail("unknown constant %s", e.Name)
	}
	if len(d.LParams) != len(e.Lvls) {
		tc.fail("constant %s expects %d universe levels, got %d", e.Name, len(d.LParams), len(e.Lvls))
	}
	if !inferOnly {
		for _, l := range e.Lvls {
			tc.checkLevel(l)
		}
	}
	return InstantiateLParams(d.Type, d.LParams, e.Lvls)
}

func (tc *TC) inferLambda(e *Expr, inferOnly bool) *Expr {
	var fvars []*Expr
	for e.Kind == Lam {
		dom := InstantiateRev(e.A, fvars)
		if !inferOnly {
			tc.ensureSort(tc.infer(dom, false))
		}
		fvars = append(fvars, tc.mkLocal(e.Name, dom))
		e = e.B
	}
	body := tc.infer(InstantiateRev(e, fvars), inferOnly)
	return tc.mkBinding(true, fvars, body)
}

func (tc *TC) inferPi(e *Expr, inferOnly bool) *Expr {
	var fvars []*Expr
	var us []*Level
	for e.Kind == Pi {
		dom := InstantiateRev(e.A, fvars)
		us = append(us, tc.ensureSort(tc.infer(dom, inferOnly)))
		fvars = append(fvars, tc.mkLocal(e.Name, dom))
		e = e.B
	}
	r := tc.ensureSort(tc.infer(InstantiateRev(e, fvars), inferOnly))
	for i := len(us) - 1; i >= 0; i-- {
		r = IMaxS(us[i], r)
	}
	return MkSort(r)
}

func (tc *TC) inferApp(e *Expr, inferOnly bool) *Expr {
	args := GetAppArgs(e)
	fty := tc.infer(GetAppFn(e), inferOnly)
	if !inferOnly {
		for _, a := range args {
			fty = tc.ensurePi(fty)
			aty := tc.infer(a, false)
			if !tc.isDefEq(aty, fty.A) {
				tc.fail("application type mismatch: argument %s has type %s but is expected to have type %s", a, aty, fty.A)
			}
			fty = Instantiate(fty.B, a)
		}
		return fty
	}
	// infer-only: keep the body open while it stays a Pi and instantiate the pending arguments in one pass
	j := 0
	for i := range args {
		if fty.Kind != Pi {
			fty = tc.ensurePi(InstantiateRev(fty, args[j:i]))
			j = i
		}
		fty = fty.B
	}
	return InstantiateRev(fty, args[j:])
}

func (tc *TC) inferProj(e *Expr, inferOnly bool) *Expr {
	st := tc.whnf(tc.infer(e.A, inferOnly))
	I := GetAppFn(st)
	if I.Kind != Const {
		tc.fail("invalid projection: %s is not a structure", st)
	}
	d := tc.env.Get(I.Name)
	if d == nil || d.Kind != DInduct || len(d.Ctors) != 1 || d.NumIndices != 0 || I.Name != e.Name {
		tc.fail("invalid projection: %s is not a structure", I.Name)
	}
	args := GetAppArgs(st)
	if len(args) != d.NumParams {
		tc.fail("invalid projection: wrong number of parameters")
	}
	c := tc.env.Get(d.Ctors[0])
	ct := InstantiateLParams(c.Type, c.LParams, I.Lvls)
	for _, a := range args {
		ct = tc.ensurePi(ct)
		ct = Instantiate(ct.B, a)
	}
	isProp := tc.isProp(st)
	for i := 0; i < e.Idx; i++ {
		ct = tc.ensurePi(ct)
		if ct.B.HasLooseBVars() && isProp && !tc.isProp(ct.A) {
			tc.fail("invalid projection: data field of a proposition is used")
		}
		ct = Instantiate(ct.B, MkProj(I.Name, i, e.A))
	}
	ct = tc.ensurePi(ct)
	if isProp && !tc.isProp(ct.A) {
		tc.fail("invalid projection: cannot project data out of a proposition")
	}
	return ct.A
}

// ---------------------------------------------------------------- reduction

func (tc *TC) natConst(n string) bool { return n == "Nat.zero" }

func isNatZero(e *Expr) bool {
	return (e.Kind == Const && e.Name == "Nat.zero") || (e.Kind == NatLit && e.Nat.Sign() == 0)
}

func isNatLitExt(e *Expr) bool { return e.Kind == NatLit || (e.Kind == Const && e.Name == "Nat.zero") }

func natVal(e *Expr) *big.Int {
	if e.Kind == Const {
		return new(big.Int)
	}
	return e.Nat
}

func natLitToCtor(e *Expr) *Expr {
	if e.Nat.Sign() == 0 {
		return MkConst("Nat.zero")
	}
	return MkApp(MkConst("Nat.succ"), MkNat(new(big.Int).Sub(e.Nat, big.NewInt(1))))
}

func (tc *TC) whnfCoreFn(e *Expr, cheapProj bool) *Expr {
	switch e.Kind {
	case BVar, Sort, Pi, Lam, NatLit, Const, FVar:
		return e
	}
	if !cheapProj {
		if r, ok := tc.whnfCore[e]; ok {
			return r
		}
	}
	tc.tick()
	var r *Expr
	switch e.Kind {
	case Let:
		r = tc.whnfCoreFn(Instantiate(e.C, e.B), cheapProj)
	case Proj:
		if m := tc.reduceProj(e, cheapProj); m != nil {
			r = tc.whnfCoreFn(m, cheapProj)
		} else {
			r = e
		}
	case App:
		f0 := GetAppFn(e)
		f := tc.whnfCoreFn(f0, cheapProj)
		switch {
		case f.Kind == Lam:
			r = tc.whnfCoreFn(betaApp(f, GetAppArgs(e)), cheapProj)
		case f != f0:
			r = tc.whnfCoreFn(MkApp(f, GetAppArgs(e)...), cheapProj)
		default:
			if red := tc.reduceRecursor(e); red != nil {
				r = tc.whnfCoreFn(red, cheapProj)
			} else {
				r = e
			}
		}
	}
	if !cheapProj {
		tc.whnfCore[e] = r
	}
	return r
}

func (tc *TC) reduceProj(e *Expr, cheap bool) *Expr {
	var c *Expr
	if cheap {
		c = tc.whnfCoreFn(e.A, cheap)
	} else {
		c = tc.whnf(e.A)
	}
	return tc.reduceProjCore(c, e.Idx)
}

func (tc *TC) reduceProjCore(c *Expr, idx int) *Expr {
	f := GetAppFn(c)
	if f.Kind != Const {
		return nil
	}
	d := tc.env.Get(f.Name)
	if d == nil || d.Kind != DCtor {
		return nil
	}
	args := GetAppArgs(c)
	np := tc.env.Get(d.Induct).NumParams
	if np+idx >= len(args) {
		return nil
	}
	return args[np+idx]
}

func (tc *TC) reduceRecursor(e *Expr) *Expr {
	f := GetAppFn(e)
	if f.Kind != Const {
		return nil
	}
	d := tc.env.Get(f.Name)
	if d == nil {
		return nil
	}
	switch d.Kind {
	case DQuot:
		return tc.quotReduce(e, d)
	case DRec:
		return tc.inductiveReduce(e, f, d)
	}
	return nil
}

func (tc *TC) quotReduce(e *Expr, d *Decl) *Expr {
	var mkPos, argPos int
	switch d.QuotKind {
	case "Quot.lift":
		mkPos, argPos = 5, 3
	case "Quot.ind":
		mkPos, argPos = 4, 3
	default:
		return nil
	}
	args := GetAppArgs(e)
	if len(args) <= mkPos {
		return nil
	}
	mk := tc.whnf(args[mkPos])
	mf := GetAppFn(mk)
	if mf.Kind != Const || mf.Name != "Quot.mk" || appNumArgs(mk) != 3 {
		return nil
	}
	r := MkApp(args[argPos], GetAppArgs(mk)[2])
	return MkApp(r, args[mkPos+1:]...)
}

func (tc *TC) isStructureLike(name string) bool {
	d := tc.env.Get(name)
	return d != nil && d.Kind == DInduct && len(d.Ctors) == 1 && d.NumIndices == 0 && !d.IsRec
}

// toCtorWhenK: for a K-like recursor (Eq), a major premise of type I params indices is treated as the nullary constructor.
func (tc *TC) toCtorWhenK(rec *Decl, major *Expr) *Expr {
	appType := tc.whnf(tc.infer(major, true))
	I := GetAppFn(appType)
	if I.Kind != Const || I.Name != rec.Induct {
		return major
	}
	id := tc.env.Get(I.Name)
	targs := GetAppArgs(appType)
	if len(targs) < id.NumParams {
		return major
	}
	ctor := MkApp(MkConst(id.Ctors[0], I.Lvls...), targs[:id.NumParams]...)
	nty := tc.infer(ctor, true)
	if !tc.isDefEq(appType, nty) {
		return major
	}
	return ctor
}

func (tc *TC) expandEtaStruct(ty *Expr, e *Expr) *Expr {
	I := GetAppFn(ty)
	id := tc.env.Get(I.Name)
	args := GetAppArgs(ty)
	c := tc.env.Get(id.Ctors[0])
	r := MkApp(MkConst(c.Name, I.Lvls...), args[:id.NumParams]...)
	for i := 0; i < c.NumFields; i++ {
		r = MkApp(r, MkProj(I.Name, i, e))
	}
	return r
}

func (tc *TC) toCtorWhenStructure(induct string, e *Expr) *Expr {
	if !tc.isStructureLike(induct) {
		return e
	}
	if f := GetAppFn(e); f.Kind == Const {
		if d := tc.env.Get(f.Name); d != nil && d.Kind == DCtor {
			return e
		}
	}
	ty := tc.whnf(tc.infer(e, true))
	if f := GetAppFn(ty); f.Kind != Const || f.Name != induct {
		return e
	}
	if tc.isProp(ty) {
		return e
	}
	return tc.expandEtaStruct(ty, e)
}

func (tc *TC) inductiveReduce(e, f *Expr, rec *Decl) *Expr {
	args := GetAppArgs(e)
	majorIdx := rec.NumParams + rec.NumMotives + rec.NumMinors + rec.NumIndices
	if majorIdx >= len(args) {
		return nil
	}
	major := args[majorIdx]
	if rec.K {
		major = tc.toCtorWhenK(rec, major)
	}
	major = tc.whnf(major)
	if major.Kind == NatLit {
		major = natLitToCtor(major)
	} else {
		major = tc.toCtorWhenStructure(rec.Induct, major)
	}
	mf := GetAppFn(major)
	if mf.Kind != Const {
		return nil
	}
	var rule *RecRule
	for i := range rec.Rules {
		if rec.Rules[i].Ctor == mf.Name {
			rule = &rec.Rules[i]
		}
	}
	if rule == nil {
		return nil
	}
	margs := GetAppArgs(major)
	if len(margs) < rule.NFields {
		return nil
	}
	if len(f.Lvls) != len(rec.LParams) {
		return nil
	}
	rhs := InstantiateLParams(rule.RHS, rec.LParams, f.Lvls)
	rhs = MkApp(rhs, args[:rec.NumParams+rec.NumMotives+rec.NumMinors]...)
	rhs = MkApp(rhs, margs[len(margs)-rule.NFields:]...)
	rhs = MkApp(rhs, args[majorIdx+1:]...)
	return rhs
}

func (tc *TC) isDelta(e *Expr) *Decl {
	f := GetAppFn(e)
	if f.Kind != Const {
		return nil
	}
	if d := tc.env.Get(f.Name); d != nil && d.HasValue() {
		return d
	}
	return nil
}

func (tc *TC) unfoldDefinition(e *Expr) *Expr {
	f := GetAppFn(e)
	d := tc.isDelta(e)
	if d == nil || len(f.Lvls) != len(d.LParams) {
		return nil
	}
	v := InstantiateLParams(d.Value, d.LParams, f.Lvls)
	if e.Kind == App {
		return betaApp(v, GetAppArgs(e))
	}
	return v
}

func (tc *TC) checkNatSize(v *big.Int) {
	if v.BitLen() > natMaxBits {
		tc.fail("the kernel refused a Nat numeral because its size exceeds the maximum")
	}
}

var natBinOps = map[string]func(a, b *big.Int) *big.Int{
	"Nat.add": func(a, b *big.Int) *big.Int { return new(big.Int).Add(a, b) },
	"Nat.sub": func(a, b *big.Int) *big.Int {
		if a.Cmp(b) < 0 {
			return new(big.Int)
		}
		return new(big.Int).Sub(a, b)
	},
	"Nat.mul": func(a, b *big.Int) *big.Int { return new(big.Int).Mul(a, b) },
	"Nat.div": func(a, b *big.Int) *big.Int {
		if b.Sign() == 0 {
			return new(big.Int)
		}
		return new(big.Int).Div(a, b)
	},
	"Nat.mod": func(a, b *big.Int) *big.Int {
		if b.Sign() == 0 {
			return new(big.Int).Set(a)
		}
		return new(big.Int).Mod(a, b)
	},
	"Nat.gcd":  func(a, b *big.Int) *big.Int { return new(big.Int).GCD(nil, nil, a, b) },
	"Nat.land": func(a, b *big.Int) *big.Int { return new(big.Int).And(a, b) },
	"Nat.lor":  func(a, b *big.Int) *big.Int { return new(big.Int).Or(a, b) },
	"Nat.xor":  func(a, b *big.Int) *big.Int { return new(big.Int).Xor(a, b) },
	"Nat.shiftRight": func(a, b *big.Int) *big.Int {
		if !b.IsUint64() {
			return new(big.Int)
		}
		return new(big.Int).Rsh(a, uint(b.Uint64()))
	},
}

// reduceNat is the kernel's Nat extension: arithmetic on literals runs on big integers instead of unary recursors.
func (tc *TC) reduceNat(e *Expr) *Expr {
	if e.HasFVar() {
		return nil
	}
	switch appNumArgs(e) {
	case 1:
		if f := e.A; f.Kind == Const && f.Name == "Nat.succ" {
			a := tc.whnf(e.B)
			if !isNatLitExt(a) {
				return nil
			}
			v := new(big.Int).Add(natVal(a), big.NewInt(1))
			tc.checkNatSize(v)
			return MkNat(v)
		}
	case 2:
		f := e.A.A
		if f.Kind != Const {
			return nil
		}
		op, isBin := natBinOps[f.Name]
		isPow := f.Name == "Nat.pow"
		isShl := f.Name == "Nat.shiftLeft"
		isPred := f.Name == "Nat.beq" || f.Name == "Nat.ble"
		if !isBin && !isPow && !isShl && !isPred {
			return nil
		}
		a := tc.whnf(e.A.B)
		if !isNatLitExt(a) {
			return nil
		}
		b := tc.whnf(e.B)
		if !isNatLitExt(b) {
			return nil
		}
		x, y := natVal(a), natVal(b)
		switch {
		case isPred:
			c := x.Cmp(y)
			if (f.Name == "Nat.beq" && c == 0) || (f.Name == "Nat.ble" && c <= 0) {
				return MkConst("Bool.true")
			}
			return MkConst("Bool.false")
		case isPow:
			if !y.IsUint64() || y.Uint64() > 1<<32-1 {
				tc.fail("the kernel refused to evaluate Nat.pow: exponent does not fit in 32 bits")
			}
			// base^exp has about bits(base)*exp bits; base <= 1 stays small
			if x.Cmp(big.NewInt(1)) > 0 && y.Uint64() != 0 && uint64(x.BitLen())*y.Uint64() > natMaxBits {
				tc.fail("the kernel refused to evaluate Nat.pow: result exceeds the maximum numeral size")
			}
			return MkNat(new(big.Int).Exp(x, y, nil))
		case isShl:
			if x.Sign() != 0 {
				if !y.IsUint64() || y.Uint64()+uint64(x.BitLen()) > natMaxBits {
					tc.fail("the kernel refused to evaluate Nat.shiftLeft: result exceeds the maximum numeral size")
				}
				return MkNat(new(big.Int).Lsh(x, uint(y.Uint64())))
			}
			return MkNat(new(big.Int))
		default:
			r := op(x, y)
			if f.Name == "Nat.add" || f.Name == "Nat.sub" || f.Name == "Nat.mul" {
				tc.checkNatSize(r)
			}
			return MkNat(r)
		}
	}
	return nil
}

func (tc *TC) whnf(e *Expr) *Expr {
	switch e.Kind {
	case BVar, Sort, Pi, NatLit, FVar:
		return e
	}
	if r, ok := tc.whnfC[e]; ok {
		return r
	}
	t := e
	for {
		t1 := tc.whnfCoreFn(t, false)
		if v := tc.reduceNat(t1); v != nil {
			tc.whnfC[e] = v
			return v
		}
		if next := tc.unfoldDefinition(t1); next != nil {
			t = next
			continue
		}
		tc.whnfC[e] = t1
		return t1
	}
}

func newBigDec(n *big.Int) *big.Int { return new(big.Int).Sub(n, big.NewInt(1)) }
