package lean

import (
	"fmt"
	"math/big"
	"strings"
	"time"
)

// Surface syntax for closed Nat and Int claims and the elaborator that turns one into kernel terms.
//
//	prop := conj ('∨' conj)*     conj := neg ('∧' neg)*     neg := '¬' neg | '(' prop ')' | expr rel expr
//	rel  := = ≠ < > ≤ ≥ ∣        expr := term (('+'|'-') term)*    term := pow (('*'|'/'|'%') pow)*
//	pow  := atom ('^' pow)?      atom := number | negative number | variable | f '(' expr (',' expr)* ')' | '(' expr ')'
//	f    := gcd lcm min max (two arguments) | factorial fact (one argument)
//	neg  also: ('∀'|'∃') x ('<'|'≤') number ',' prop      (a bounded quantifier)
//
// A bounded quantifier is expanded: `∀ x < 3, P x` is elaborated as `P 0 ∧ P 1 ∧ P 2` (and `∃` as `∨`), up to
// maxBound values of x and maxExpansions copies of a body in all, and the kernel checks that expanded proposition.
// The variable stands for a numeral inside the body; no other variables exist.
//
// ASCII spellings are accepted: <= >= != /\ \/ ~ and `|` for divides.
// The operators elaborate to what Lean's instances unfold to (`a + b` to `Nat.add a b`, `a ≤ b` to `Nat.le a b`);
// the HAdd/LE/Dvd class layer is not ported.
// A claim that contains a negative numeral (`-3`, or `(-2)`) is an Int claim: every numeral in it is an Int literal
// (`Int.ofNat n` / `Int.negSucc (n-1)`, as Lean writes `-3`), `+ - * ^ / %` are `Int.add sub mul pow ediv emod`, the
// relations are `Int.le`/`Int.lt` (that is, `Int.NonNeg (b - a)`, Lean's definition) and the quantifier variables
// range over `Int.ofNat k`. `/` and `%` follow Lean's E-rounding: `-12 / 7 = -2` and `-12 % 7 = 2`, and `a ∣ b` is
// `Int.dvd`, Lean's `Exists (fun c => b = a * c)`. A claim without a negative numeral stays a Nat claim, where `-` is
// the truncated `Nat.sub`: `3 - 5 = 0` is a Nat claim (true) and `3 - 5 = -2` is an Int claim (true).
// `gcd`, `lcm`, `min` and `max` follow their Int ports (`Int.gcd`/`Int.lcm` are Nat-valued in Lean, so an Int claim
// reads them as `Int.ofNat (...)`); `factorial` has no Int port and is rejected there rather than read as Nat.

type claimTok struct {
	kind string // num, sym, eof
	s    string
}

func lexClaim(src string) ([]claimTok, error) {
	rs := []rune(src)
	var ts []claimTok
	for i := 0; i < len(rs); {
		r := rs[i]
		two := ""
		if i+1 < len(rs) {
			two = string(rs[i : i+2])
		}
		switch {
		case r == ' ' || r == '\t' || r == '\n' || r == '\r':
			i++
		case r >= '0' && r <= '9':
			j := i
			for j < len(rs) && rs[j] >= '0' && rs[j] <= '9' {
				j++
			}
			ts = append(ts, claimTok{"num", string(rs[i:j])})
			i = j
		case r == '-' && unaryMinus(ts) && i+1 < len(rs) && rs[i+1] >= '0' && rs[i+1] <= '9':
			// a sign, not a subtraction: `-3`, `2 * -3`, `2 - -3`
			j := i + 1
			for j < len(rs) && rs[j] >= '0' && rs[j] <= '9' {
				j++
			}
			ts = append(ts, claimTok{"num", string(rs[i:j])})
			i = j
		case two == "<=" || two == ">=" || two == "!=":
			ts = append(ts, claimTok{"sym", map[string]string{"<=": "≤", ">=": "≥", "!=": "≠"}[two]})
			i += 2
		case two == "/\\":
			ts = append(ts, claimTok{"sym", "∧"})
			i += 2
		case two == "\\/":
			ts = append(ts, claimTok{"sym", "∨"})
			i += 2
		case r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r == '_':
			j := i
			for j < len(rs) && (rs[j] >= 'a' && rs[j] <= 'z' || rs[j] >= 'A' && rs[j] <= 'Z' || rs[j] == '_' || rs[j] >= '0' && rs[j] <= '9') {
				j++
			}
			w := string(rs[i:j])
			switch w {
			case "forall":
				ts = append(ts, claimTok{"sym", "∀"})
			case "exists":
				ts = append(ts, claimTok{"sym", "∃"})
			default:
				ts = append(ts, claimTok{"id", w})
			}
			i = j
		case strings.ContainsRune("+-*/%^()=<>≤≥≠∣∧∨¬∀∃,", r):
			ts = append(ts, claimTok{"sym", string(r)})
			i++
		case r == '~':
			ts = append(ts, claimTok{"sym", "¬"})
			i++
		case r == '|':
			ts = append(ts, claimTok{"sym", "∣"})
			i++
		default:
			return nil, fmt.Errorf("unsupported character %q", r)
		}
	}
	return append(ts, claimTok{kind: "eof"}), nil
}

// unaryMinus reports whether a '-' after the tokens so far starts a number rather than subtracting one term from
// another: it does at the start, after an operator or an opening bracket, but not after a number, a variable or ')'.
func unaryMinus(ts []claimTok) bool {
	if len(ts) == 0 {
		return true
	}
	t := ts[len(ts)-1]
	return t.kind == "sym" && t.s != ")"
}

// hasNegativeLiteral selects the Int fragment for a claim: a negative numeral has no Nat reading.
func hasNegativeLiteral(ts []claimTok) bool {
	for _, t := range ts {
		if t.kind == "num" && strings.HasPrefix(t.s, "-") {
			return true
		}
	}
	return false
}

// claimFunc is the named-function table: the Nat definition every claim uses, the Int port when there is one, and
// whether that port is Nat-valued in Lean (`Int.gcd`/`Int.lcm` are). A claim has one type and the coercion layer is
// not ported, so in an Int claim a Nat-valued result is read as `Int.ofNat (Int.gcd a b)`: ofNat is injective and
// preserves the order, so the verdict is Lean's.
type claimFunc struct {
	nat     string
	intName string // "" when the function has no Int port
	arity   int
	intCast bool
}

var claimFuncs = map[string]claimFunc{
	"gcd":       {"Nat.gcd", "Int.gcd", 2, true},
	"lcm":       {"Nat.lcm", "Int.lcm", 2, true},
	"min":       {"Nat.min", "Int.min", 2, false},
	"max":       {"Nat.max", "Int.max", 2, false},
	"factorial": {"Nat.factorial", "", 1, false},
	"fact":      {"Nat.factorial", "", 1, false},
}

// Decided pairs a proposition with a Decidable instance for it.
type decided struct{ prop, inst *Expr }

type claimParser struct {
	ts         []claimTok
	pos        int
	depth      int
	vars       map[string]int
	expansions int
	intMode    bool
	// unsupported holds the first Int-unsupported construct, so it survives the backtracking in neg()
	unsupported string
}

const (
	maxBound      = 256
	maxExpansions = 4096
)

type claimError string

func (e claimError) Error() string { return string(e) }

func (p *claimParser) fail(f string, a ...any) { panic(claimError(fmt.Sprintf(f, a...))) }

// failUnsupported rejects an Int claim that uses a construct only the Nat fragment has. The reason is kept on the
// parser because neg() backtracks over a failed comparison and would otherwise report "expected a comparison".
func (p *claimParser) failUnsupported(f string, a ...any) {
	p.unsupported = fmt.Sprintf(f, a...)
	p.fail("%s", p.unsupported)
}

// litIn builds a numeral in the given type: a Nat literal, or the Int literal Lean writes for the same number
// (`Int.ofNat n` for n ≥ 0, `Int.negSucc (n-1)` for n < 0, since `-3` is `negSucc 2`).
func (p *claimParser) litIn(n *big.Int, intMode bool) *Expr {
	if !intMode {
		return MkNat(n)
	}
	if n.Sign() >= 0 {
		return MkApp(c("Int.ofNat"), MkNat(n))
	}
	return MkApp(c("Int.negSucc"), MkNat(new(big.Int).Sub(new(big.Int).Neg(n), big.NewInt(1))))
}

// varValueIn is the value of a quantifier variable k in the given type.
func (p *claimParser) varValueIn(k uint64, intMode bool) *Expr {
	if intMode {
		return MkApp(c("Int.ofNat"), MkNatU(k))
	}
	return MkNatU(k)
}

func (p *claimParser) peek() claimTok { return p.ts[p.pos] }
func (p *claimParser) eat(s string) bool {
	if t := p.peek(); t.kind == "sym" && t.s == s {
		p.pos++
		return true
	}
	return false
}

const maxClaimDepth = 100

func (p *claimParser) enter() {
	if p.depth++; p.depth > maxClaimDepth {
		p.fail("claim is nested too deeply")
	}
}

func c(n string) *Expr         { return MkConst(n) }
func natT() *Expr              { return c("Nat") }
func intT() *Expr              { return c("Int") }
func eqIn(t, a, b *Expr) *Expr { return MkApp(MkConst("Eq", One()), t, a, b) }
func eqNat(a, b *Expr) *Expr   { return eqIn(natT(), a, b) }
func opIn(intMode bool, nat, intOp string) string {
	if intMode {
		return intOp
	}
	return nat
}

func (p *claimParser) prop() decided {
	p.enter()
	defer func() { p.depth-- }()
	v := p.conj()
	for p.eat("∨") {
		w := p.conj()
		v = decided{MkApp(c("Or"), v.prop, w.prop), MkApp(c("instDecidableOr"), v.prop, w.prop, v.inst, w.inst)}
	}
	return v
}

func (p *claimParser) conj() decided {
	v := p.neg()
	for p.eat("∧") {
		w := p.neg()
		v = decided{MkApp(c("And"), v.prop, w.prop), MkApp(c("instDecidableAnd"), v.prop, w.prop, v.inst, w.inst)}
	}
	return v
}

func (p *claimParser) neg() decided {
	p.enter()
	defer func() { p.depth-- }()
	if p.eat("¬") {
		v := p.neg()
		return decided{MkApp(c("Not"), v.prop), MkApp(c("instDecidableNot"), v.prop, v.inst)}
	}
	if t := p.peek(); t.kind == "sym" && (t.s == "∀" || t.s == "∃") {
		p.pos++
		return p.quantified(t.s == "∀")
	}
	if t := p.peek(); t.kind == "id" && t.s == "coprime" && p.ts[p.pos+1].kind == "sym" && p.ts[p.pos+1].s == "(" {
		if p.intMode {
			p.failUnsupported("coprime is a Nat predicate; it is not ported for Int (use a claim without a negative numeral)")
		}
		p.pos += 2
		a := p.expr()
		if !p.eat(",") {
			p.fail("coprime takes two arguments")
		}
		b := p.expr()
		if !p.eat(")") {
			p.fail("missing )")
		}
		return decided{MkApp(c("Nat.Coprime"), a, b), MkApp(c("Nat.decCoprime"), a, b)}
	}
	// "(2+3)*4 = 20" and "(1 = 1) ∧ ..." both start with '(': try a comparison, then a parenthesised prop
	save := p.pos
	if v, ok := p.tryCmp(); ok {
		return v
	}
	p.pos = save
	if p.eat("(") {
		v := p.prop()
		if !p.eat(")") {
			p.fail("missing )")
		}
		return v
	}
	p.fail("expected a comparison")
	return decided{}
}

// quantified parses `x < N, body` (after the quantifier) by parsing body once per value of x.
func (p *claimParser) quantified(all bool) decided {
	id := p.peek()
	if id.kind != "id" {
		p.fail("expected a variable after the quantifier")
	}
	p.pos++
	var upto int
	switch {
	case p.eat("<"):
		upto = 0
	case p.eat("≤"):
		upto = 1
	default:
		p.fail("a quantifier needs a bound: ∀ x < n, ...")
	}
	nt := p.peek()
	if nt.kind != "num" {
		p.fail("the bound of a quantifier must be a number")
	}
	p.pos++
	n, _ := new(big.Int).SetString(nt.s, 10)
	if !n.IsInt64() || n.Int64()+int64(upto) > maxBound {
		p.fail("the bound of a quantifier is limited to %d", maxBound)
	}
	count := int(n.Int64()) + upto
	if !p.eat(",") {
		p.fail("missing , after the bound")
	}
	start := p.pos
	old, had := p.vars[id.s]
	if p.vars == nil {
		p.vars = map[string]int{}
	}
	defer func() {
		if had {
			p.vars[id.s] = old
		} else {
			delete(p.vars, id.s)
		}
	}()
	var parts []decided
	end := start
	// the body is parsed at least once so that an empty range still consumes it
	for k := 0; k < count || k == 0; k++ {
		if p.expansions++; p.expansions > maxExpansions {
			p.fail("quantifiers expand to more than %d cases", maxExpansions)
		}
		p.vars[id.s] = k
		p.pos = start
		d := p.prop()
		end = p.pos
		if k < count {
			parts = append(parts, d)
		}
	}
	p.pos = end
	if len(parts) == 0 {
		if all {
			return decided{c("True"), MkApp(c("Decidable.isTrue"), c("True"), c("True.intro"))}
		}
		return decided{c("False"), MkApp(c("Decidable.isFalse"), c("False"), MkLam("h", c("False"), MkBVar(0)))}
	}
	v := parts[0]
	for _, w := range parts[1:] {
		if all {
			v = decided{MkApp(c("And"), v.prop, w.prop), MkApp(c("instDecidableAnd"), v.prop, w.prop, v.inst, w.inst)}
		} else {
			v = decided{MkApp(c("Or"), v.prop, w.prop), MkApp(c("instDecidableOr"), v.prop, w.prop, v.inst, w.inst)}
		}
	}
	return v
}

func (p *claimParser) tryCmp() (v decided, ok bool) {
	defer func() {
		if r := recover(); r != nil {
			if _, isClaim := r.(claimError); isClaim {
				ok = false
				return
			}
			panic(r)
		}
	}()
	a := p.expr()
	t := p.peek()
	if t.kind != "sym" || !strings.Contains("=≠<>≤≥∣", t.s) {
		return decided{}, false
	}
	p.pos++
	b := p.expr()
	app := func(f string, args ...*Expr) *Expr { return MkApp(c(f), args...) }
	if p.intMode {
		eq := eqIn(intT(), a, b)
		switch t.s {
		case "=":
			return decided{eq, app("Int.decEq", a, b)}, true
		case "≠":
			return decided{MkApp(MkConst("Ne", One()), intT(), a, b), app("instDecidableNot", eq, app("Int.decEq", a, b))}, true
		case "<":
			return decided{app("Int.lt", a, b), app("Int.decLt", a, b)}, true
		case ">":
			return decided{app("Int.lt", b, a), app("Int.decLt", b, a)}, true
		case "≤":
			return decided{app("Int.le", a, b), app("Int.decLe", a, b)}, true
		case "≥":
			return decided{app("Int.le", b, a), app("Int.decLe", b, a)}, true
		case "∣":
			return decided{app("Int.dvd", a, b), app("Int.decDvd", a, b)}, true
		}
		p.fail("unexpected relation %q", t.s)
	}
	switch t.s {
	case "=":
		return decided{eqNat(a, b), app("Nat.decEq", a, b)}, true
	case "≠":
		return decided{MkApp(MkConst("Ne", One()), natT(), a, b), app("instDecidableNot", eqNat(a, b), app("Nat.decEq", a, b))}, true
	case "<":
		return decided{app("Nat.lt", a, b), app("Nat.decLt", a, b)}, true
	case ">":
		return decided{app("Nat.lt", b, a), app("Nat.decLt", b, a)}, true
	case "≤":
		return decided{app("Nat.le", a, b), app("Nat.decLe", a, b)}, true
	case "≥":
		return decided{app("Nat.le", b, a), app("Nat.decLe", b, a)}, true
	}
	return decided{app("Nat.dvd", a, b), app("Nat.decDvd", a, b)}, true
}

func (p *claimParser) expr() *Expr { return p.exprAt(p.intMode) }

func (p *claimParser) exprAt(intMode bool) *Expr {
	v := p.termAt(intMode)
	for {
		switch {
		case p.eat("+"):
			v = MkApp(c(opIn(intMode, "Nat.add", "Int.add")), v, p.termAt(intMode))
		case p.eat("-"):
			v = MkApp(c(opIn(intMode, "Nat.sub", "Int.sub")), v, p.termAt(intMode))
		default:
			return v
		}
	}
}

func (p *claimParser) termAt(intMode bool) *Expr {
	v := p.powAt(intMode)
	for {
		switch {
		case p.eat("*"):
			v = MkApp(c(opIn(intMode, "Nat.mul", "Int.mul")), v, p.powAt(intMode))
		case p.eat("/"):
			v = MkApp(c(opIn(intMode, "Nat.div", "Int.ediv")), v, p.powAt(intMode))
		case p.eat("%"):
			v = MkApp(c(opIn(intMode, "Nat.mod", "Int.emod")), v, p.powAt(intMode))
		default:
			return v
		}
	}
}

// powAt parses `atom ('^' pow)?`. The exponent of an Int power is a Nat (`Int.pow : Int → Nat → Int`), so the
// exponent's sub-expression is parsed in Nat mode: `(-2)^3^2` is `Int.pow (-2) (Nat.pow 3 2)`.
func (p *claimParser) powAt(intMode bool) *Expr {
	b := p.atom(intMode)
	if p.eat("^") {
		if intMode {
			return MkApp(c("Int.pow"), b, p.powAt(false))
		}
		return MkApp(c("Nat.pow"), b, p.powAt(false))
	}
	return b
}

// atom parses a numeral, a quantifier variable, a Nat function application or a parenthesised expression in the
// given mode (the claim's mode, or Nat inside the exponent of an Int power).
func (p *claimParser) atom(intMode bool) *Expr {
	p.enter()
	defer func() { p.depth-- }()
	t := p.peek()
	if t.kind == "num" {
		p.pos++
		n, _ := new(big.Int).SetString(t.s, 10)
		return p.litIn(n, intMode)
	}
	if t.kind == "id" && p.ts[p.pos+1].kind == "sym" && p.ts[p.pos+1].s == "(" {
		fn, ok := claimFuncs[t.s]
		if !ok {
			p.fail("unknown function %s", t.s)
		}
		if intMode && fn.intName == "" {
			p.failUnsupported("%s is a Nat function; it is not ported for Int (use a claim without a negative numeral)", t.s)
		}
		p.pos += 2
		var args []*Expr
		for {
			args = append(args, p.exprAt(intMode))
			if !p.eat(",") {
				break
			}
		}
		if !p.eat(")") {
			p.fail("missing ) after the arguments of %s", t.s)
		}
		if len(args) != fn.arity {
			p.fail("%s takes %d argument(s), got %d", t.s, fn.arity, len(args))
		}
		if !intMode {
			return MkApp(c(fn.nat), args...)
		}
		call := MkApp(c(fn.intName), args...)
		if fn.intCast {
			call = MkApp(c("Int.ofNat"), call)
		}
		return call
	}
	if t.kind == "id" {
		v, ok := p.vars[t.s]
		if !ok {
			p.fail("unbound variable %s (only quantifier variables exist)", t.s)
		}
		p.pos++
		return p.varValueIn(uint64(v), intMode)
	}
	if p.eat("(") {
		v := p.exprAt(intMode)
		if !p.eat(")") {
			p.fail("missing )")
		}
		return v
	}
	p.fail("expected a number or (")
	return nil
}

// ElabClaim parses a claim into its proposition and a Decidable instance (both kernel terms over the prelude).
func ElabClaim(claim string) (prop, inst *Expr, err error) {
	ts, err := lexClaim(claim)
	if err != nil {
		return nil, nil, err
	}
	p := &claimParser{ts: ts, intMode: hasNegativeLiteral(ts)}
	defer func() {
		if r := recover(); r != nil {
			if ce, ok := r.(claimError); ok {
				if p.unsupported != "" {
					ce = claimError(p.unsupported)
				}
				prop, inst, err = nil, nil, ce
				return
			}
			panic(r)
		}
	}()
	d := p.prop()
	if p.peek().kind != "eof" {
		p.fail("unexpected %q", p.peek().s)
	}
	return d.prop, d.inst, nil
}

// Options bound the kernel's work; both are enforced inside whnf and is_def_eq.
type Options struct {
	MaxSteps int
	Deadline time.Time
}

// Proof is a kernel-checked certificate: Term has type Prop in the prelude environment.
type Proof struct {
	Prop, Term *Expr
	Steps      int
}

// ErrNotProved means the claim parsed but the kernel rejected the proof of it. For a closed decidable claim that
// means it is false; Refuted says the kernel also accepted a proof of its negation.
type ErrNotProved struct {
	Reason  string
	Refuted bool
	Proof   *Proof // the kernel-checked proof of the negation, when Refuted
}

func (e ErrNotProved) Error() string { return "claim is false: " + e.Reason }

// ErrUnsupported means the text is outside the supported fragment.
type ErrUnsupported struct{ Reason string }

func (e ErrUnsupported) Error() string { return e.Reason }

// ProveClaim elaborates the claim, builds `of_decide_eq_true prop inst (Eq.refl true)` and has the kernel check
// that this term has type `prop`. Success is the kernel's verdict, not an evaluator's. When the check fails it
// tries the same for `Not prop` so a false claim comes back with a certificate too.
func ProveClaim(claim string, opt Options) (*Proof, error) {
	env, err := Prelude()
	if err != nil {
		return nil, err
	}
	prop, inst, err := ElabClaim(claim)
	if err != nil {
		return nil, ErrUnsupported{err.Error()}
	}
	p, err := checkDecided(env, prop, inst, opt)
	if err == nil || IsBudgetError(err) {
		return p, err
	}
	np, ni := MkApp(c("Not"), prop), MkApp(c("instDecidableNot"), prop, inst)
	if neg, nerr := checkDecided(env, np, ni, opt); nerr == nil {
		return nil, ErrNotProved{Reason: err.Error(), Refuted: true, Proof: neg}
	}
	return nil, ErrNotProved{Reason: err.Error()}
}

func checkDecided(env *Env, prop, inst *Expr, opt Options) (*Proof, error) {
	term := MkApp(c("of_decide_eq_true"), prop, inst, MkApp(MkConst("Eq.refl", One()), c("Bool"), c("Bool.true")))
	tc := NewTC(env, nil)
	tc.MaxSteps, tc.Deadline = opt.MaxSteps, opt.Deadline
	ty, err := tc.Check(term)
	if err == nil {
		var ok bool
		if ok, err = tc.IsDefEq(ty, prop); err == nil && !ok {
			err = kernelError{msg: "proof has the wrong type"}
		}
	}
	if err != nil {
		return nil, err
	}
	return &Proof{Prop: prop, Term: term, Steps: tc.Steps()}, nil
}
