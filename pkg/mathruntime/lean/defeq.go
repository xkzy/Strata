package lean

// Definitional equality (type_checker.cpp is_def_eq_core and friends).

type lbool int8

const (
	lUndef lbool = iota
	lTrue
	lFalse
)

func toL(b bool) lbool {
	if b {
		return lTrue
	}
	return lFalse
}

func (tc *TC) isDefEq(t, s *Expr) bool {
	if tc.isDefEqCore(t, s) {
		tc.eqC[[2]*Expr{t, s}] = true
		return true
	}
	return false
}

func (tc *TC) isDefEqBinding(t, s *Expr) bool {
	k := t.Kind
	var fvars []*Expr
	for t.Kind == k && s.Kind == k {
		td, sd := InstantiateRev(t.A, fvars), InstantiateRev(s.A, fvars)
		if !Equal(td, sd) && !tc.isDefEq(td, sd) {
			return false
		}
		fvars = append(fvars, tc.mkLocal(t.Name, td))
		t, s = t.B, s.B
	}
	return tc.isDefEq(InstantiateRev(t, fvars), InstantiateRev(s, fvars))
}

func (tc *TC) quickIsDefEq(t, s *Expr) lbool {
	if t == s || tc.eqC[[2]*Expr{t, s}] || Equal(t, s) {
		return lTrue
	}
	if t.Kind == s.Kind {
		switch t.Kind {
		case Lam, Pi:
			return toL(tc.isDefEqBinding(t, s))
		case Sort:
			return toL(LevelEquiv(t.Lvl, s.Lvl))
		case NatLit:
			return toL(t.Nat.Cmp(s.Nat) == 0)
		}
	}
	return lUndef
}

func (tc *TC) isDefEqArgs(t, s *Expr) bool {
	for t.Kind == App && s.Kind == App {
		if !tc.isDefEq(t.B, s.B) {
			return false
		}
		t, s = t.A, s.A
	}
	return t.Kind != App && s.Kind != App
}

func (tc *TC) isDefEqApp(t, s *Expr) bool {
	if t.Kind != App || s.Kind != App || appNumArgs(t) != appNumArgs(s) {
		return false
	}
	if !tc.isDefEq(GetAppFn(t), GetAppFn(s)) {
		return false
	}
	return tc.isDefEqArgs(t, s)
}

func (tc *TC) proofIrrel(t, s *Expr) lbool {
	tt := tc.infer(t, true)
	if !tc.isProp(tt) {
		return lUndef
	}
	return toL(tc.isDefEq(tt, tc.infer(s, true)))
}

func natSuccPred(t *Expr) *Expr {
	if t.Kind == NatLit {
		if t.Nat.Sign() != 0 {
			return MkNat(newBigDec(t.Nat))
		}
		return nil
	}
	if t.Kind == App && t.A.Kind == Const && t.A.Name == "Nat.succ" {
		return t.B
	}
	return nil
}

func (tc *TC) isDefEqOffset(t, s *Expr) lbool {
	if isNatZero(t) && isNatZero(s) {
		return lTrue
	}
	pt, ps := natSuccPred(t), natSuccPred(s)
	if pt != nil && ps != nil {
		return toL(tc.isDefEqCore(pt, ps))
	}
	return lUndef
}

func hintCompare(a, b Hints) int {
	if a.Kind == b.Kind {
		if a.Kind == HRegular {
			switch {
			case a.Height == b.Height:
				return 0
			case a.Height > b.Height:
				return -1
			}
			return 1
		}
		return 0
	}
	switch {
	case a.Kind == HOpaque:
		return 1
	case b.Kind == HOpaque:
		return -1
	case a.Kind == HAbbrev:
		return -1
	}
	return 1
}

type reduction int

const (
	rContinue reduction = iota
	rUnknown
	rEqual
	rDiff
)

func (tc *TC) lazyDeltaStep(t, s **Expr) reduction {
	dt, ds := tc.isDelta(*t), tc.isDelta(*s)
	switch {
	case dt == nil && ds == nil:
		return rUnknown
	case dt != nil && ds == nil:
		*t = tc.whnfCoreFn(tc.unfoldDefinition(*t), true)
	case dt == nil:
		*s = tc.whnfCoreFn(tc.unfoldDefinition(*s), true)
	default:
		c := hintCompare(dt.Hints, ds.Hints)
		switch {
		case c < 0:
			*t = tc.whnfCoreFn(tc.unfoldDefinition(*t), true)
		case c > 0:
			*s = tc.whnfCoreFn(tc.unfoldDefinition(*s), true)
		default:
			if (*t).Kind == App && (*s).Kind == App && dt == ds && dt.Hints.Kind == HRegular {
				key := [2]*Expr{*t, *s}
				if !tc.failC[key] {
					if levelsEq(GetAppFn(*t).Lvls, GetAppFn(*s).Lvls) && tc.isDefEqArgs(*t, *s) {
						return rEqual
					}
					tc.failC[key] = true
				}
			}
			*t = tc.whnfCoreFn(tc.unfoldDefinition(*t), true)
			*s = tc.whnfCoreFn(tc.unfoldDefinition(*s), true)
		}
	}
	switch tc.quickIsDefEq(*t, *s) {
	case lTrue:
		return rEqual
	case lFalse:
		return rDiff
	}
	return rContinue
}

func (tc *TC) lazyDelta(t, s **Expr) lbool {
	for {
		tc.tick()
		if r := tc.isDefEqOffset(*t, *s); r != lUndef {
			return r
		}
		if !(*t).HasFVar() && !(*s).HasFVar() {
			if v := tc.reduceNat(*t); v != nil {
				return toL(tc.isDefEqCore(v, *s))
			} else if v := tc.reduceNat(*s); v != nil {
				return toL(tc.isDefEqCore(*t, v))
			}
		}
		switch tc.lazyDeltaStep(t, s) {
		case rContinue:
		case rUnknown:
			return lUndef
		case rEqual:
			return lTrue
		case rDiff:
			return lFalse
		}
	}
}

func (tc *TC) tryEtaExpansion(t, s *Expr) bool {
	for i := 0; i < 2; i++ {
		if t.Kind == Lam && s.Kind != Lam {
			if st := tc.whnf(tc.infer(s, true)); st.Kind == Pi {
				ns := MkLam(st.Name, st.A, MkApp(lift(s, 0, 1), MkBVar(0)))
				return tc.isDefEq(t, ns)
			}
		}
		t, s = s, t
	}
	return false
}

func (tc *TC) tryEtaStructCore(t, s *Expr) bool {
	f := GetAppFn(s)
	if f.Kind != Const {
		return false
	}
	cd := tc.env.Get(f.Name)
	if cd == nil || cd.Kind != DCtor {
		return false
	}
	id := tc.env.Get(cd.Induct)
	args := GetAppArgs(s)
	if len(args) != id.NumParams+cd.NumFields || !tc.isStructureLike(cd.Induct) {
		return false
	}
	if !tc.isDefEq(tc.infer(t, true), tc.infer(s, true)) {
		return false
	}
	for i := id.NumParams; i < len(args); i++ {
		if !tc.isDefEq(MkProj(cd.Induct, i-id.NumParams, t), args[i]) {
			return false
		}
	}
	return true
}

func (tc *TC) isDefEqUnitLike(t, s *Expr) bool {
	tt := tc.whnf(tc.infer(t, true))
	I := GetAppFn(tt)
	if I.Kind != Const || !tc.isStructureLike(I.Name) {
		return false
	}
	if tc.env.Get(tc.env.Get(I.Name).Ctors[0]).NumFields != 0 {
		return false
	}
	return tc.isDefEq(tt, tc.infer(s, true))
}

func (tc *TC) isDefEqCore(t, s *Expr) bool {
	tc.tick()
	if r := tc.quickIsDefEq(t, s); r != lUndef {
		return r == lTrue
	}
	// closed t against Bool.true: evaluate t (the `decide` fast path)
	if !t.HasFVar() && s.Kind == Const && s.Name == "Bool.true" {
		if w := tc.whnf(t); w.Kind == Const && w.Name == "Bool.true" {
			return true
		}
	}
	tn, sn := tc.whnfCoreFn(t, true), tc.whnfCoreFn(s, true)
	if tn != t || sn != s {
		if r := tc.quickIsDefEq(tn, sn); r != lUndef {
			return r == lTrue
		}
	}
	if r := tc.proofIrrel(tn, sn); r != lUndef {
		return r == lTrue
	}
	if r := tc.lazyDelta(&tn, &sn); r != lUndef {
		return r == lTrue
	}
	if tn.Kind == Const && sn.Kind == Const && tn.Name == sn.Name && levelsEq(tn.Lvls, sn.Lvls) {
		return true
	}
	if tn.Kind == FVar && sn.Kind == FVar && tn.Name == sn.Name {
		return true
	}
	if tn.Kind == Proj && sn.Kind == Proj && tn.Idx == sn.Idx && tc.isDefEq(tn.A, sn.A) {
		return true
	}
	tnn, snn := tc.whnfCoreFn(tn, false), tc.whnfCoreFn(sn, false)
	if tnn != tn || snn != sn {
		return tc.isDefEqCore(tnn, snn)
	}
	if tc.isDefEqApp(tn, sn) {
		return true
	}
	if tc.tryEtaExpansion(tn, sn) {
		return true
	}
	if tc.tryEtaStructCore(tn, sn) || tc.tryEtaStructCore(sn, tn) {
		return true
	}
	return tc.isDefEqUnitLike(tn, sn)
}
