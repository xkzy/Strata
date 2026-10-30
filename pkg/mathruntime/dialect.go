package mathruntime

import (
	"fmt"
	"regexp"
	"strings"
)

// Dialect is a notation a request can be written in and a result can be returned in. All dialects run on the one
// StrataCAS engine: Parse rewrites the source to the engine's canonical form, Format rewrites the canonical result.
// Backend and Version are the labels reported in MathResult for work done on behalf of that dialect.
type Dialect struct {
	Name    string
	Backend string
	Version string
	Detect  func(src string) bool // true when src is unmistakably this dialect (only used when no language is named)
	Parse   func(src string) string
	Format  func(s string, op MathOperation) string
}

func ident(s string) string { return s }
func never(string) bool     { return false }

const (
	LangInfix   = "infix"
	LangWolfram = "wolfram"
	LangSage    = "sage"
	LangLaTeX   = "latex"
	LangLean    = "lean"
)

var sageTokens = []string{"var(", ".diff(", ".derivative(", ".integrate(", ".integral(", ".factor(", ".expand(", ".simplify(",
	".roots(", ".det(", ".inverse(", ".transpose(", "matrix([", "is_prime(", "euler_phi(", "fibonacci(", "power_mod(", "xgcd("}

// dialects is the registry. Order matters only for Detect: the first match wins.
var dialects = []Dialect{
	{Name: LangSage, Backend: "SageMath", Version: "10.4", Parse: sageParse, Format: sageFormat,
		Detect: func(src string) bool {
			for _, t := range sageTokens {
				if strings.Contains(src, t) {
					return true
				}
			}
			return false
		}},
	{Name: LangWolfram, Backend: "Mathics3-Core", Version: "3.0.0", Parse: wolframParse, Format: func(s string, _ MathOperation) string { return s }, Detect: never},
	{Name: LangLaTeX, Backend: "Mathics3-Core", Version: "3.0.0", Parse: latexParse, Format: latexFormat, Detect: never},
	{Name: LangLean, Backend: "Mathics3-Core", Version: "3.0.0", Parse: ident, Format: func(s string, _ MathOperation) string { return s }, Detect: never},
	{Name: LangInfix, Backend: "Mathics3-Core", Version: "3.0.0", Parse: ident, Format: func(s string, _ MathOperation) string { return s }, Detect: never},
}

// Languages lists the notations a request or result can use, each with the backend label it reports.
func Languages() map[string]string {
	m := make(map[string]string, len(dialects))
	for _, d := range dialects {
		m[d.Name] = d.Backend + " " + d.Version
	}
	return m
}

func dialectByName(name string) (Dialect, bool) {
	name = strings.ToLower(strings.TrimSpace(name))
	for _, d := range dialects {
		if d.Name == name {
			return d, true
		}
	}
	return Dialect{}, false
}

// inputDialect is the dialect a request is written in: the named one, else the one Detect recognises (the "sage"
// tenant counts as naming Sage), else plain infix. An unknown name is an error, not a silent fallback.
func inputDialect(req MathRequest) (Dialect, error) {
	if req.InputLanguage != "" {
		d, ok := dialectByName(req.InputLanguage)
		if !ok {
			return Dialect{}, fmt.Errorf("unknown input_language %q", req.InputLanguage)
		}
		return d, nil
	}
	if strings.ToLower(req.TenantID) == LangSage {
		d, _ := dialectByName(LangSage)
		return d, nil
	}
	for _, d := range dialects {
		if d.Detect(req.Expression) {
			return d, nil
		}
	}
	d, _ := dialectByName(LangInfix)
	return d, nil
}

// outputDialect is the named output dialect, else the input's.
func outputDialect(req MathRequest, in Dialect) (Dialect, error) {
	if req.OutputLanguage == "" {
		return in, nil
	}
	d, ok := dialectByName(req.OutputLanguage)
	if !ok {
		return Dialect{}, fmt.Errorf("unknown output_language %q", req.OutputLanguage)
	}
	return d, nil
}

// ---- Sage ----

// sageParse converts SageMath Pythonic mathematical expressions into canonical CAS form.
func sageParse(sageExpr string) string {
	if sageExpr == "" {
		return ""
	}

	text := sageExpr

	// 1. Strip var declarations: var('x'), var('x y z'), x, y = var('x y')
	varDeclRe := regexp.MustCompile(`(?:(?:[a-zA-Z_][a-zA-Z0-9_,\s]*=)?\s*var\s*\(\s*['"][^'"]*['"]\s*\)\s*(?:;|\n)?)`)
	text = varDeclRe.ReplaceAllString(text, "")

	// 2. Strip PolynomialRing declarations: R.<x> = PolynomialRing(QQ)
	ringDeclRe := regexp.MustCompile(`(?:[a-zA-Z_][a-zA-Z0-9_]*\.<[^>]+>\s*=\s*(?:PolynomialRing\([^)]+\)|[a-zA-Z0-9_]+\[[^\]]+\])\s*(?:;|\n)?)`)
	text = ringDeclRe.ReplaceAllString(text, "")

	text = strings.TrimSpace(text)

	// 3. Dot-method transformations:
	// expr.diff(x) -> diff(expr, x)
	// expr.factor() -> factor(expr)
	// expr.roots(x) -> solve(expr == 0, x)
	dotMethodRe := regexp.MustCompile(`\(?([a-zA-Z0-9_\+\-\*\/\^\s\(\)\[\],]+?)\)?\.(diff|derivative|integrate|integral|factor|expand|simplify|roots|det|inverse|transpose|rank|trace|solve|linearsolve|qr|cholesky)\s*\(([^)]*)\)`)
	for {
		m := dotMethodRe.FindStringSubmatch(text)
		if len(m) == 0 {
			break
		}
		target := m[1]
		method := m[2]
		args := m[3]

		var replacement string
		if method == "roots" {
			v := "x"
			if strings.TrimSpace(args) != "" {
				v = strings.TrimSpace(args)
			}
			replacement = fmt.Sprintf("solve(%s == 0, %s)", target, v)
		} else if strings.TrimSpace(args) == "" {
			replacement = fmt.Sprintf("%s(%s)", method, target)
		} else {
			replacement = fmt.Sprintf("%s(%s, %s)", method, target, args)
		}
		text = strings.Replace(text, m[0], replacement, 1)
	}

	// 4. Translate matrix([[...]]) to [[...]]
	matrixRe := regexp.MustCompile(`\bmatrix\s*\(\s*(\[\[[\s\S]*?\]\])\s*\)`)
	text = matrixRe.ReplaceAllString(text, "$1")

	return strings.TrimSpace(text)
}

func sageFormat(s string, op MathOperation) string {
	if op == OpSolve {
		s = strings.ReplaceAll(s, "{", "[")
		s = strings.ReplaceAll(s, "}", "]")
		return strings.ReplaceAll(s, "->", "==")
	}
	if strings.HasPrefix(s, "{") && strings.HasSuffix(s, "}") {
		return "[" + s[1:len(s)-1] + "]"
	}
	return s
}

// ---- Wolfram ----

var wolframHeads = map[string]string{"D": "diff", "Integrate": "integrate", "Solve": "solve", "Simplify": "simplify",
	"Factor": "factor", "Expand": "expand", "Limit": "limit", "Det": "det", "Inverse": "inverse", "Transpose": "transpose",
	"Sin": "sin", "Cos": "cos", "Tan": "tan", "Exp": "exp", "Log": "log", "Sqrt": "sqrt"}

var wolframIdentRe = regexp.MustCompile(`^[A-Za-z][A-Za-z0-9]*`)

// wolframParse rewrites Head[args] calls of the known heads to head(args). Other brackets (lists, {} and [[ ]]) stay.
func wolframParse(src string) string {
	var out strings.Builder
	var calls []bool // one entry per open '[': true when it opened a rewritten call
	for i := 0; i < len(src); {
		c := src[i]
		switch {
		case (c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z') && (i == 0 || !isWordByte(src[i-1])):
			id := wolframIdentRe.FindString(src[i:])
			i += len(id)
			if fn, ok := wolframHeads[id]; ok && i < len(src) && src[i] == '[' {
				out.WriteString(fn + "(")
				calls = append(calls, true)
				i++
			} else {
				out.WriteString(id)
			}
		case c == '[':
			calls = append(calls, false)
			out.WriteByte(c)
			i++
		case c == ']':
			if n := len(calls); n > 0 {
				if calls[n-1] {
					out.WriteByte(')')
				} else {
					out.WriteByte(']')
				}
				calls = calls[:n-1]
			} else {
				out.WriteByte(c)
			}
			i++
		default:
			out.WriteByte(c)
			i++
		}
	}
	return out.String()
}

func isWordByte(b byte) bool {
	return b == '_' || b >= '0' && b <= '9' || b >= 'A' && b <= 'Z' || b >= 'a' && b <= 'z'
}

// ---- LaTeX ----

var (
	latexFracRe = regexp.MustCompile(`\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}`)
	latexSqrtRe = regexp.MustCompile(`\\sqrt\s*\{([^{}]*)\}`)
	latexPowRe  = regexp.MustCompile(`\^\{([^{}]*)\}`)
	latexFnRe   = regexp.MustCompile(`\\(sin|cos|tan|exp|log|ln)\b`)
	latexCmdRe  = regexp.MustCompile(`\\(left|right|,|;|!)`)
)

// latexParse reads the common subset: \frac, \sqrt, ^{..}, \cdot, \times, \pi, \sin-style functions, \left \right.
// \int, \sum, \lim and other commands are not read; they stay in the text and the validator rejects them.
func latexParse(src string) string {
	s := latexCmdRe.ReplaceAllString(src, "")
	for { // nested \frac{\frac{1}{2}}{3} unwinds from the inside out
		t := latexFracRe.ReplaceAllString(s, "(($1)/($2))")
		t = latexSqrtRe.ReplaceAllString(t, "sqrt($1)")
		if t == s {
			break
		}
		s = t
	}
	s = latexPowRe.ReplaceAllString(s, "^($1)")
	s = latexFnRe.ReplaceAllString(s, "$1")
	s = strings.NewReplacer(`\cdot`, "*", `\times`, "*", `\div`, "/", `\pi`, "pi", "{", "(", "}", ")").Replace(s)
	return strings.TrimSpace(s)
}

var (
	latexOutPowRe = regexp.MustCompile(`\^\(?(-?[A-Za-z0-9]+)\)?`)
	latexOutDivRe = regexp.MustCompile(`\(([A-Za-z0-9]+)/([A-Za-z0-9]+)\)|([A-Za-z0-9]+)/([A-Za-z0-9]+)`)
)

// latexFormat writes a result as LaTeX: a/b as \frac, ^n as ^{n}, * as \cdot. Longer fractions are left as a/b.
func latexFormat(s string, _ MathOperation) string {
	s = latexOutDivRe.ReplaceAllStringFunc(s, func(m string) string {
		p := latexOutDivRe.FindStringSubmatch(m)
		if p[1] != "" {
			return `\frac{` + p[1] + `}{` + p[2] + `}`
		}
		return `\frac{` + p[3] + `}{` + p[4] + `}`
	})
	s = latexOutPowRe.ReplaceAllString(s, `^{$1}`)
	return strings.ReplaceAll(s, "*", ` \cdot `)
}
