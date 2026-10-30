package lean

import (
	"fmt"
	"math/big"
	"strings"
	"unicode"
)

// Parser for fully explicit kernel terms (no implicit arguments, no inference): binders carry their types and
// universe levels are written `Nat.rec.{1}`. Bound names become de Bruijn indices; anything else is a constant.
//
//	term   := 'fun' binder+ '=>' term | '∀' binder+ ',' term | 'let' x ':' term ':=' term ';' term | arrow
//	arrow  := ('(' x+ ':' term ')')+ '→' arrow | app ('→' arrow)?
//	app    := atom+
//	atom   := ident ('.{' level,* '}')? | number | '(' term ')' | 'Prop' | 'Type' | 'Sort' level
type tok struct {
	kind string // id, num, sym, eof
	s    string
}

type parser struct {
	toks    []tok
	pos     int
	scope   []string
	lparams map[string]bool
}

func tokenize(src string) ([]tok, error) {
	var ts []tok
	rs := []rune(src)
	for i := 0; i < len(rs); {
		r := rs[i]
		switch {
		case unicode.IsSpace(r):
			i++
		case r == '-' && i+1 < len(rs) && rs[i+1] == '-': // comment to end of line
			for i < len(rs) && rs[i] != '\n' {
				i++
			}
		case unicode.IsDigit(r):
			j := i
			for j < len(rs) && unicode.IsDigit(rs[j]) {
				j++
			}
			ts = append(ts, tok{"num", string(rs[i:j])})
			i = j
		case unicode.IsLetter(r) || r == '_':
			j := i
			for j < len(rs) {
				c := rs[j]
				if unicode.IsLetter(c) || unicode.IsDigit(c) || c == '_' || c == '\'' {
					j++
				} else if c == '.' && j+1 < len(rs) && (unicode.IsLetter(rs[j+1]) || unicode.IsDigit(rs[j+1]) || rs[j+1] == '_') {
					j++
				} else {
					break
				}
			}
			ts = append(ts, tok{"id", string(rs[i:j])})
			i = j
		case r == '.' && i+1 < len(rs) && rs[i+1] == '{':
			ts = append(ts, tok{"sym", ".{"})
			i += 2
		case r == '=' && i+1 < len(rs) && rs[i+1] == '>':
			ts = append(ts, tok{"sym", "=>"})
			i += 2
		case r == ':' && i+1 < len(rs) && rs[i+1] == '=':
			ts = append(ts, tok{"sym", ":="})
			i += 2
		case r == '-' && i+1 < len(rs) && rs[i+1] == '>':
			ts = append(ts, tok{"sym", "→"})
			i += 2
		case strings.ContainsRune("()},:;+→∀λ", r):
			s := string(r)
			if r == 'λ' {
				s = "fun"
			}
			ts = append(ts, tok{"sym", s})
			i++
		default:
			return nil, fmt.Errorf("unexpected character %q", r)
		}
	}
	return append(ts, tok{kind: "eof"}), nil
}

func newParser(src string, lparams []string) (*parser, error) {
	ts, err := tokenize(src)
	if err != nil {
		return nil, err
	}
	p := &parser{toks: ts, lparams: map[string]bool{}}
	for _, l := range lparams {
		p.lparams[l] = true
	}
	return p, nil
}

// ParseTerm parses a closed kernel term; lparams are the declared universe parameter names.
func ParseTerm(src string, lparams []string) (e *Expr, err error) {
	p, err := newParser(src, lparams)
	if err != nil {
		return nil, err
	}
	defer func() {
		if r := recover(); r != nil {
			if pe, ok := r.(parseError); ok {
				e, err = nil, pe
				return
			}
			panic(r)
		}
	}()
	e = p.term()
	if p.peek().kind != "eof" {
		p.fail("unexpected %q", p.peek().s)
	}
	return e, nil
}

type parseError string

func (e parseError) Error() string { return "parse: " + string(e) }

func (p *parser) fail(f string, a ...any) { panic(parseError(fmt.Sprintf(f, a...))) }
func (p *parser) peek() tok               { return p.toks[p.pos] }
func (p *parser) peekAt(n int) tok {
	if p.pos+n < len(p.toks) {
		return p.toks[p.pos+n]
	}
	return tok{kind: "eof"}
}
func (p *parser) next() tok           { t := p.toks[p.pos]; p.pos++; return t }
func (p *parser) isSym(s string) bool { t := p.peek(); return t.kind == "sym" && t.s == s }
func (p *parser) isKw(s string) bool  { t := p.peek(); return t.kind == "id" && t.s == s }
func (p *parser) expect(s string) {
	if !p.isSym(s) {
		p.fail("expected %q, got %q", s, p.peek().s)
	}
	p.pos++
}

func (p *parser) term() *Expr {
	switch {
	case p.isKw("fun"), p.isSym("fun"):
		p.pos++
		return p.binders(false, "=>")
	case p.isSym("∀"), p.isKw("forall"):
		p.pos++
		return p.binders(true, ",")
	case p.isKw("let"):
		p.pos++
		name := p.next().s
		p.expect(":")
		ty := p.term()
		p.expect(":=")
		val := p.term()
		p.expect(";")
		p.scope = append(p.scope, name)
		body := p.term()
		p.scope = p.scope[:len(p.scope)-1]
		return MkLet(name, ty, val, body)
	}
	return p.arrow()
}

type binder struct {
	name string
	ty   *Expr
}

// groups parses ('(' x+ ':' term ')')+ and pushes every name into scope; the caller pops them.
func (p *parser) groups() []binder {
	var bs []binder
	for p.isSym("(") && p.peekAt(1).kind == "id" {
		save := p.pos
		p.pos++
		var names []string
		for p.peek().kind == "id" {
			names = append(names, p.next().s)
		}
		if !p.isSym(":") {
			p.pos = save
			break
		}
		p.pos++
		ty := p.term()
		p.expect(")")
		for _, n := range names {
			bs = append(bs, binder{n, ty})
			// the type was parsed before these names; shift is handled by lifting for later names in the group
			ty = lift(ty, 0, 1)
			p.scope = append(p.scope, n)
		}
	}
	return bs
}

func (p *parser) binders(pi bool, sep string) *Expr {
	n0 := len(p.scope)
	bs := p.groups()
	if len(bs) == 0 {
		p.fail("expected a binder")
	}
	p.expect(sep)
	body := p.term()
	p.scope = p.scope[:n0]
	for i := len(bs) - 1; i >= 0; i-- {
		if pi {
			body = MkPi(bs[i].name, bs[i].ty, body)
		} else {
			body = MkLam(bs[i].name, bs[i].ty, body)
		}
	}
	return body
}

func (p *parser) arrow() *Expr {
	if p.isSym("(") && p.peekAt(1).kind == "id" {
		// a dependent arrow starts with a binder group; a parenthesised term cannot contain `x : T`
		save, n0 := p.pos, len(p.scope)
		if bs := p.groups(); len(bs) > 0 {
			if p.isSym("→") {
				p.pos++
				body := p.arrow()
				p.scope = p.scope[:n0]
				for i := len(bs) - 1; i >= 0; i-- {
					body = MkPi(bs[i].name, bs[i].ty, body)
				}
				return body
			}
		}
		p.scope = p.scope[:n0]
		p.pos = save
	}
	dom := p.app()
	if p.isSym("→") {
		p.pos++
		p.scope = append(p.scope, "_")
		cod := p.arrow()
		p.scope = p.scope[:len(p.scope)-1]
		return MkPi("_", dom, cod)
	}
	return dom
}

func (p *parser) startsAtom() bool {
	t := p.peek()
	switch t.kind {
	case "num":
		return true
	case "id":
		switch t.s {
		case "fun", "forall", "let":
			return false
		}
		return true
	case "sym":
		return t.s == "("
	}
	return false
}

func (p *parser) app() *Expr {
	f := p.atom()
	for p.startsAtom() {
		f = MkApp(f, p.atom())
	}
	return f
}

func (p *parser) atom() *Expr {
	t := p.next()
	switch t.kind {
	case "num":
		n, _ := new(big.Int).SetString(t.s, 10)
		return MkNat(n)
	case "id":
		switch t.s {
		case "Prop":
			return Prop()
		case "Type":
			if k := p.peek(); k.kind == "num" || (k.kind == "id" && p.lparams[k.s]) {
				return MkSort(Succ(p.levelAtom()))
			}
			return Type0()
		case "Sort":
			return MkSort(p.levelAtom())
		}
		for i := len(p.scope) - 1; i >= 0; i-- {
			if p.scope[i] == t.s {
				return MkBVar(len(p.scope) - 1 - i)
			}
		}
		var ls []*Level
		if p.isSym(".{") {
			p.pos++
			for {
				ls = append(ls, p.level())
				if p.isSym(",") {
					p.pos++
					continue
				}
				break
			}
			p.expect("}")
		}
		return MkConst(t.s, ls...)
	case "sym":
		if t.s == "(" {
			e := p.term()
			p.expect(")")
			return e
		}
	}
	p.fail("unexpected %q", t.s)
	return nil
}

func (p *parser) level() *Level {
	if p.isKw("max") || p.isKw("imax") {
		k := p.next().s
		a, b := p.levelAtom(), p.levelAtom()
		if k == "max" {
			return Max(a, b)
		}
		return IMax(a, b)
	}
	l := p.levelAtom()
	for p.isSym("+") {
		p.pos++
		t := p.next()
		n := 0
		fmt.Sscanf(t.s, "%d", &n)
		for i := 0; i < n; i++ {
			l = Succ(l)
		}
	}
	return l
}

func (p *parser) levelAtom() *Level {
	t := p.next()
	switch {
	case t.kind == "num":
		n := 0
		fmt.Sscanf(t.s, "%d", &n)
		l := Zero
		for i := 0; i < n; i++ {
			l = Succ(l)
		}
		return l
	case t.kind == "id":
		return Param(t.s)
	case t.kind == "sym" && t.s == "(":
		l := p.level()
		p.expect(")")
		return l
	}
	p.fail("bad universe level near %q", t.s)
	return nil
}
