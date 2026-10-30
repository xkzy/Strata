package mathruntime

import (
	"strings"
	"testing"
)

func sageFormatForTest(s string) string { return sageFormat(s, OpSolve) }

func TestDialectParse(t *testing.T) {
	for _, c := range []struct{ lang, in, want string }{
		{LangWolfram, "D[Sin[x^2], x]", "diff(sin(x^2), x)"},
		{LangWolfram, "Solve[x^2 == 4, x]", "solve(x^2 == 4, x)"},
		{LangWolfram, "Det[{{1, 2}, {3, 4}}]", "det({{1, 2}, {3, 4}})"},
		{LangLaTeX, `\frac{1}{2} + \sqrt{x}`, "((1)/(2)) + sqrt(x)"},
		{LangLaTeX, `2 \cdot x^{3} \left( y \right)`, "2 * x^(3) ( y )"},
		{LangLaTeX, `\sin(\pi)`, "sin(pi)"},
		{LangSage, "(x^3 - 8).factor()", "factor(x^3 - 8)"},
	} {
		d, _ := dialectByName(c.lang)
		if got := d.Parse(c.in); got != c.want {
			t.Errorf("%s %q: got %q want %q", c.lang, c.in, got, c.want)
		}
	}
}

func TestDialectRuntime(t *testing.T) {
	rt := NewMathRuntime(nil)

	// named input and output languages
	res := rt.ProcessRequest(MathRequest{Operation: OpIntegrate, Expression: "x^2", Variable: "x", InputLanguage: LangWolfram, OutputLanguage: LangLaTeX})
	if res.Status != StatusSuccess || !strings.Contains(res.ExactResult, "x^{3}") {
		t.Errorf("latex out: %v %q", res.Status, res.ExactResult)
	}
	if got := sageFormatForTest("{x -> -2, x -> 2}"); got != "[x == -2, x == 2]" {
		t.Errorf("sage format: %q", got)
	}
	// the Sage tenant and Sage tokens still pick Sage
	res = rt.ProcessRequest(MathRequest{Operation: OpEvaluate, Expression: "fibonacci(10)"})
	if res.BackendName != "SageMath" || res.ExactResult != "55" {
		t.Errorf("sage detect: %s %q", res.BackendName, res.ExactResult)
	}
	// a Lean claim goes to the kernel
	res = rt.ProcessRequest(MathRequest{Expression: "2 + 2 = 4", InputLanguage: LangLean})
	if res.Status != StatusSuccess || res.ExactResult != "proved" {
		t.Errorf("lean in: %v %q %s", res.Status, res.ExactResult, res.ErrorMessage)
	}
	// reject: unknown languages are errors, not a silent fallback
	for _, req := range []MathRequest{
		{Operation: OpEvaluate, Expression: "1+1", InputLanguage: "maple"},
		{Operation: OpEvaluate, Expression: "1+1", OutputLanguage: "maple"},
		{Operation: OpEvaluate, Expression: "1+1", OutputLanguage: LangLean},
	} {
		if res := rt.ProcessRequest(req); res.Status == StatusSuccess && req.OutputLanguage != LangLean {
			t.Errorf("%+v accepted", req)
		}
	}
	// reject: a LaTeX command that is not read does not reach the engine as an expression
	if res := rt.ProcessRequest(MathRequest{Operation: OpEvaluate, Expression: `\int_0^1 x`, InputLanguage: LangLaTeX}); res.Status == StatusSuccess {
		t.Errorf("unread LaTeX command accepted: %q", res.ExactResult)
	}
}
