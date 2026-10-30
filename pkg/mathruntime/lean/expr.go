package lean

import (
	"math/big"
	"strconv"
	"strings"
)

type Kind uint8

const (
	BVar Kind = iota
	FVar
	Sort
	Const
	App
	Lam
	Pi
	Let
	NatLit
	Proj
)

// Expr is immutable. Binder names are cosmetic; de Bruijn indices carry the meaning.
type Expr struct {
	Kind  Kind
	Idx   int    // BVar index; Proj field
	Name  string // FVar id, Const name, binder name, Proj structure name
	Lvl   *Level // Sort
	Lvls  []*Level
	A, B  *Expr // App fn/arg; Lam/Pi domain/body; Let type/value; Proj struct expr in A
	C     *Expr // Let body
	Nat   *big.Int
	loose int // 1 + the largest loose bvar index, 0 if closed
	fv    bool
}

func max2(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func MkBVar(i int) *Expr                   { return &Expr{Kind: BVar, Idx: i, loose: i + 1} }
func MkFVar(id string) *Expr               { return &Expr{Kind: FVar, Name: id, fv: true} }
func MkSort(l *Level) *Expr                { return &Expr{Kind: Sort, Lvl: l} }
func MkConst(n string, ls ...*Level) *Expr { return &Expr{Kind: Const, Name: n, Lvls: ls} }
func MkNat(n *big.Int) *Expr               { return &Expr{Kind: NatLit, Nat: n} }
func MkNatU(n uint64) *Expr                { return MkNat(new(big.Int).SetUint64(n)) }
func Prop() *Expr                          { return MkSort(Zero) }
func Type0() *Expr                         { return MkSort(One()) }

func MkApp(f *Expr, args ...*Expr) *Expr {
	for _, a := range args {
		f = &Expr{Kind: App, A: f, B: a, loose: max2(f.loose, a.loose), fv: f.fv || a.fv}
	}
	return f
}

func mkBinding(k Kind, name string, dom, body *Expr) *Expr {
	return &Expr{Kind: k, Name: name, A: dom, B: body, loose: max2(dom.loose, body.loose-1), fv: dom.fv || body.fv}
}
func MkLam(name string, dom, body *Expr) *Expr { return mkBinding(Lam, name, dom, body) }
func MkPi(name string, dom, body *Expr) *Expr  { return mkBinding(Pi, name, dom, body) }
func MkArrow(dom, cod *Expr) *Expr             { return MkPi("_", dom, lift(cod, 0, 1)) }

func MkLet(name string, ty, val, body *Expr) *Expr {
	return &Expr{Kind: Let, Name: name, A: ty, B: val, C: body,
		loose: max2(max2(ty.loose, val.loose), body.loose-1), fv: ty.fv || val.fv || body.fv}
}

func MkProj(sname string, idx int, e *Expr) *Expr {
	return &Expr{Kind: Proj, Name: sname, Idx: idx, A: e, loose: e.loose, fv: e.fv}
}

func (e *Expr) HasLooseBVars() bool { return e.loose > 0 }
func (e *Expr) HasFVar() bool       { return e.fv }

func GetAppFn(e *Expr) *Expr {
	for e.Kind == App {
		e = e.A
	}
	return e
}

func GetAppArgs(e *Expr) []*Expr {
	n := 0
	for x := e; x.Kind == App; x = x.A {
		n++
	}
	args := make([]*Expr, n)
	for x := e; x.Kind == App; x = x.A {
		n--
		args[n] = x.B
	}
	return args
}

func appNumArgs(e *Expr) int {
	n := 0
	for ; e.Kind == App; e = e.A {
		n++
	}
	return n
}

func levelsEq(a, b []*Level) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !LevelEquiv(a[i], b[i]) {
			return false
		}
	}
	return true
}

// Equal is structural equality up to binder names and level equivalence.
func Equal(a, b *Expr) bool {
	if a == b {
		return true
	}
	if a.Kind != b.Kind || a.loose != b.loose || a.fv != b.fv {
		return false
	}
	switch a.Kind {
	case BVar:
		return a.Idx == b.Idx
	case FVar:
		return a.Name == b.Name
	case Sort:
		return LevelEquiv(a.Lvl, b.Lvl)
	case Const:
		return a.Name == b.Name && levelsEq(a.Lvls, b.Lvls)
	case App, Lam, Pi:
		return Equal(a.A, b.A) && Equal(a.B, b.B)
	case Let:
		return Equal(a.A, b.A) && Equal(a.B, b.B) && Equal(a.C, b.C)
	case NatLit:
		return a.Nat.Cmp(b.Nat) == 0
	case Proj:
		return a.Idx == b.Idx && a.Name == b.Name && Equal(a.A, b.A)
	}
	return false
}

// replace rebuilds e bottom-up; f returns a replacement or nil to descend. off counts the binders passed.
func replace(e *Expr, off int, f func(*Expr, int) *Expr) *Expr {
	if r := f(e, off); r != nil {
		return r
	}
	switch e.Kind {
	case App:
		a, b := replace(e.A, off, f), replace(e.B, off, f)
		if a == e.A && b == e.B {
			return e
		}
		return MkApp(a, b)
	case Lam, Pi:
		a, b := replace(e.A, off, f), replace(e.B, off+1, f)
		if a == e.A && b == e.B {
			return e
		}
		return mkBinding(e.Kind, e.Name, a, b)
	case Let:
		a, b, c := replace(e.A, off, f), replace(e.B, off, f), replace(e.C, off+1, f)
		if a == e.A && b == e.B && c == e.C {
			return e
		}
		return MkLet(e.Name, a, b, c)
	case Proj:
		a := replace(e.A, off, f)
		if a == e.A {
			return e
		}
		return MkProj(e.Name, e.Idx, a)
	}
	return e
}

// lift adds d to every loose bvar index >= s.
func lift(e *Expr, s, d int) *Expr {
	if d == 0 || e.loose <= s {
		return e
	}
	return replace(e, 0, func(x *Expr, off int) *Expr {
		if x.loose <= s+off {
			return x
		}
		if x.Kind == BVar {
			return MkBVar(x.Idx + d)
		}
		return nil
	})
}

// lower subtracts d from every loose bvar index >= s (d <= s).
func lower(e *Expr, s, d int) *Expr {
	if d == 0 || e.loose <= s {
		return e
	}
	return replace(e, 0, func(x *Expr, off int) *Expr {
		if x.loose <= s+off {
			return x
		}
		if x.Kind == BVar {
			return MkBVar(x.Idx - d)
		}
		return nil
	})
}

// Instantiate replaces loose bvar i by subst[i] (bvars above len(subst) shift down).
func Instantiate(e *Expr, subst ...*Expr) *Expr {
	n := len(subst)
	if e.loose == 0 || n == 0 {
		return e
	}
	return replace(e, 0, func(x *Expr, off int) *Expr {
		if x.loose <= off {
			return x
		}
		if x.Kind == BVar {
			if j := x.Idx - off; j < n {
				return lift(subst[j], 0, off)
			}
			return MkBVar(x.Idx - n)
		}
		return nil
	})
}

// InstantiateRev is Instantiate with the substitution read right to left (last element is bvar 0).
func InstantiateRev(e *Expr, subst []*Expr) *Expr {
	rev := make([]*Expr, len(subst))
	for i, s := range subst {
		rev[len(subst)-1-i] = s
	}
	return Instantiate(e, rev...)
}

// Abstract turns the given free variables into bvars: fvars[i] becomes bvar (n-1-i), so fvars[0] binds outermost.
func Abstract(e *Expr, fvars []*Expr) *Expr {
	if !e.fv || len(fvars) == 0 {
		return e
	}
	n := len(fvars)
	return replace(e, 0, func(x *Expr, off int) *Expr {
		if !x.fv {
			return x
		}
		if x.Kind == FVar {
			for i := n - 1; i >= 0; i-- {
				if fvars[i].Name == x.Name {
					return MkBVar(off + n - 1 - i)
				}
			}
			return x
		}
		return nil
	})
}

func InstantiateLParams(e *Expr, names []string, ls []*Level) *Expr {
	if len(names) == 0 {
		return e
	}
	return replace(e, 0, func(x *Expr, _ int) *Expr {
		switch x.Kind {
		case Sort:
			return MkSort(InstantiateLevel(x.Lvl, names, ls))
		case Const:
			if len(x.Lvls) == 0 {
				return x
			}
			nl := make([]*Level, len(x.Lvls))
			for i, l := range x.Lvls {
				nl[i] = InstantiateLevel(l, names, ls)
			}
			return MkConst(x.Name, nl...)
		case BVar, FVar, NatLit:
			return x
		}
		return nil
	})
}

// HeadBeta applies a lambda head to its arguments.
func HeadBeta(e *Expr) *Expr {
	f := GetAppFn(e)
	if f.Kind != Lam || e.Kind != App {
		return e
	}
	return betaApp(f, GetAppArgs(e))
}

func betaApp(f *Expr, args []*Expr) *Expr {
	i := 0
	for f.Kind == Lam && i < len(args) {
		f = f.B
		i++
	}
	return MkApp(Instantiate(f, reversed(args[:i])...), args[i:]...)
}

func reversed(a []*Expr) []*Expr {
	r := make([]*Expr, len(a))
	for i, x := range a {
		r[len(a)-1-i] = x
	}
	return r
}

func (e *Expr) String() string {
	var sb strings.Builder
	e.write(&sb, 0)
	return sb.String()
}

func (e *Expr) write(sb *strings.Builder, depth int) {
	switch e.Kind {
	case BVar:
		sb.WriteString("#" + strconv.Itoa(e.Idx))
	case FVar:
		sb.WriteString(e.Name)
	case Sort:
		if e.Lvl.IsZero() {
			sb.WriteString("Prop")
		} else {
			sb.WriteString("Sort " + e.Lvl.String())
		}
	case Const:
		sb.WriteString(e.Name)
	case NatLit:
		sb.WriteString(e.Nat.String())
	case App:
		sb.WriteString("(")
		for i, a := range append([]*Expr{GetAppFn(e)}, GetAppArgs(e)...) {
			if i > 0 {
				sb.WriteString(" ")
			}
			a.write(sb, depth)
		}
		sb.WriteString(")")
	case Lam, Pi:
		if e.Kind == Lam {
			sb.WriteString("(fun (" + e.Name + " : ")
		} else {
			sb.WriteString("((" + e.Name + " : ")
		}
		e.A.write(sb, depth)
		if e.Kind == Lam {
			sb.WriteString(") => ")
		} else {
			sb.WriteString(") -> ")
		}
		e.B.write(sb, depth+1)
		sb.WriteString(")")
	case Let:
		sb.WriteString("(let " + e.Name + " := ")
		e.B.write(sb, depth)
		sb.WriteString("; ")
		e.C.write(sb, depth+1)
		sb.WriteString(")")
	case Proj:
		e.A.write(sb, depth)
		sb.WriteString("." + strconv.Itoa(e.Idx+1))
	}
}
