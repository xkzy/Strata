// Package lean is a Go port of the Lean 4 kernel (src/kernel in leanprover/lean4): universe levels, expressions,
// the environment, inductive types with generated recursors, quotients, and the type checker with Nat literal
// acceleration. It is derived from Lean 4, Copyright (c) Microsoft Corporation and the Lean FRO, licensed under the
// Apache License 2.0 (the license text is in LICENSE next to this file).
//
// Not ported: String literals, the elaborator, tactics and the compiler.
package lean

import (
	"sort"
	"strconv"
	"strings"
)

type LevelKind uint8

const (
	LZero LevelKind = iota
	LSucc
	LMax
	LIMax
	LParam
)

type Level struct {
	Kind LevelKind
	A, B *Level
	Name string
}

var Zero = &Level{Kind: LZero}

func Succ(a *Level) *Level    { return &Level{Kind: LSucc, A: a} }
func Max(a, b *Level) *Level  { return &Level{Kind: LMax, A: a, B: b} }
func IMax(a, b *Level) *Level { return &Level{Kind: LIMax, A: a, B: b} }
func Param(n string) *Level   { return &Level{Kind: LParam, Name: n} }
func One() *Level             { return Succ(Zero) }
func (l *Level) IsZero() bool { return l.Kind == LZero }

func (l *Level) String() string {
	switch l.Kind {
	case LZero:
		return "0"
	case LSucc:
		n, b := l.offset()
		if b.Kind == LZero {
			return strconv.Itoa(n)
		}
		return b.String() + "+" + strconv.Itoa(n)
	case LMax:
		return "max(" + l.A.String() + "," + l.B.String() + ")"
	case LIMax:
		return "imax(" + l.A.String() + "," + l.B.String() + ")"
	}
	return l.Name
}

func (l *Level) offset() (int, *Level) {
	n := 0
	for l.Kind == LSucc {
		n++
		l = l.A
	}
	return n, l
}

// IsNeverZero: the level is >= 1 for every assignment of its parameters.
func (l *Level) IsNeverZero() bool {
	switch l.Kind {
	case LSucc:
		return true
	case LMax:
		return l.A.IsNeverZero() || l.B.IsNeverZero()
	case LIMax:
		return l.B.IsNeverZero()
	}
	return false
}

func (l *Level) HasParam() bool {
	switch l.Kind {
	case LParam:
		return true
	case LSucc:
		return l.A.HasParam()
	case LMax, LIMax:
		return l.A.HasParam() || l.B.HasParam()
	}
	return false
}

// InstantiateLevel replaces the named parameters; a name without a replacement stays.
func InstantiateLevel(l *Level, names []string, ls []*Level) *Level {
	switch l.Kind {
	case LSucc:
		return Succ(InstantiateLevel(l.A, names, ls))
	case LMax:
		return Max(InstantiateLevel(l.A, names, ls), InstantiateLevel(l.B, names, ls))
	case LIMax:
		return IMax(InstantiateLevel(l.A, names, ls), InstantiateLevel(l.B, names, ls))
	case LParam:
		for i, n := range names {
			if n == l.Name && i < len(ls) {
				return ls[i]
			}
		}
	}
	return l
}

// levelParamsOK reports whether every parameter in l is in ps.
func levelParamsOK(l *Level, ps []string) (string, bool) {
	switch l.Kind {
	case LSucc:
		return levelParamsOK(l.A, ps)
	case LMax, LIMax:
		if n, ok := levelParamsOK(l.A, ps); !ok {
			return n, false
		}
		return levelParamsOK(l.B, ps)
	case LParam:
		for _, p := range ps {
			if p == l.Name {
				return "", true
			}
		}
		return l.Name, false
	}
	return "", true
}

// A normal form is a max of (base + offset) terms; base is a parameter, an imax atom, or nil for the constant.
type lterm struct {
	key  string
	base *Level
	off  int
}

func normTerms(l *Level) []lterm {
	var ts []lterm
	switch l.Kind {
	case LZero:
		ts = []lterm{{}}
	case LParam:
		ts = []lterm{{key: l.Name, base: l}}
	case LSucc:
		ts = normTerms(l.A)
		for i := range ts {
			ts[i].off++
		}
	case LMax:
		ts = append(normTerms(l.A), normTerms(l.B)...)
	case LIMax:
		switch {
		case l.B.IsNeverZero():
			ts = append(normTerms(l.A), normTerms(l.B)...)
		case l.B.Kind == LZero:
			ts = []lterm{{}}
		case l.A.Kind == LZero:
			ts = normTerms(l.B)
		default:
			a, b := fromTerms(simplify(normTerms(l.A))), fromTerms(simplify(normTerms(l.B)))
			if levelKey(a) == levelKey(b) {
				ts = normTerms(b)
			} else {
				at := IMax(a, b)
				ts = []lterm{{key: "i(" + levelKey(a) + "," + levelKey(b) + ")", base: at}}
			}
		}
	}
	return ts
}

func levelKey(l *Level) string { return l.String() }

func simplify(ts []lterm) []lterm {
	best := map[string]lterm{}
	for _, t := range ts {
		if o, ok := best[t.key]; !ok || o.off < t.off {
			best[t.key] = t
		}
	}
	maxOff := -1
	for _, t := range best {
		if t.off > maxOff {
			maxOff = t.off
		}
	}
	out := make([]lterm, 0, len(best))
	for _, t := range best {
		// the constant k is below every other term whose offset is >= k
		if t.key == "" && len(best) > 1 {
			big := false
			for _, u := range best {
				if u.key != "" && u.off >= t.off {
					big = true
				}
			}
			if big {
				continue
			}
		}
		out = append(out, t)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].key != out[j].key {
			return out[i].key < out[j].key
		}
		return out[i].off < out[j].off
	})
	return out
}

func fromTerms(ts []lterm) *Level {
	var r *Level
	for _, t := range ts {
		l := t.base
		if l == nil {
			l = Zero
		}
		for i := 0; i < t.off; i++ {
			l = Succ(l)
		}
		if r == nil {
			r = l
		} else {
			r = Max(r, l)
		}
	}
	return r
}

func normKey(l *Level) string {
	ts := simplify(normTerms(l))
	var sb strings.Builder
	for _, t := range ts {
		sb.WriteString(t.key + "+" + strconv.Itoa(t.off) + ";")
	}
	return sb.String()
}

// LevelEquiv: a and b denote the same level for every parameter assignment.
func LevelEquiv(a, b *Level) bool { return a == b || normKey(a) == normKey(b) }

// LevelGeq: a >= b for every parameter assignment.
func LevelGeq(a, b *Level) bool {
	ta, tb := simplify(normTerms(a)), simplify(normTerms(b))
	for _, y := range tb {
		ok := false
		for _, x := range ta {
			if x.off >= y.off && (x.key == y.key || y.key == "") {
				ok = true
			}
		}
		if !ok {
			return false
		}
	}
	return true
}

// MaxS and IMaxS are the simplifying constructors the kernel uses when it infers sorts (mk_max / mk_imax).
func MaxS(a, b *Level) *Level {
	switch {
	case a.IsZero():
		return b
	case b.IsZero(), levelStructEq(a, b):
		return a
	}
	na, ba := a.offset()
	nb, bb := b.offset()
	if levelStructEq(ba, bb) {
		if na >= nb {
			return a
		}
		return b
	}
	if ba.IsZero() && na <= nb && bb.Kind != LZero {
		return b
	}
	if bb.IsZero() && nb <= na {
		return a
	}
	return Max(a, b)
}

func IMaxS(a, b *Level) *Level {
	switch {
	case b.IsNeverZero():
		return MaxS(a, b)
	case b.IsZero(), a.IsZero():
		return b
	case levelStructEq(a, b):
		return a
	}
	return IMax(a, b)
}

func levelStructEq(a, b *Level) bool {
	if a == b {
		return true
	}
	if a.Kind != b.Kind {
		return false
	}
	switch a.Kind {
	case LSucc:
		return levelStructEq(a.A, b.A)
	case LMax, LIMax:
		return levelStructEq(a.A, b.A) && levelStructEq(a.B, b.B)
	case LParam:
		return a.Name == b.Name
	}
	return true
}
