package mathruntime

import (
	"strings"
	"testing"
	"time"
)

// Every input here reaches the runtime over HTTP (/v1/strata/cas/solve, /math/evaluate, /math/theorems/verify, the MCP
// tools) from callers who are not trusted: one request must not pin a CPU, grow without end, or panic.

// bounded runs f and fails when it panics or has not returned within d.
func bounded(t *testing.T, name string, d time.Duration, f func()) bool {
	t.Helper()
	done := make(chan interface{}, 1)
	go func() {
		defer func() { done <- recover() }()
		f()
	}()
	select {
	case p := <-done:
		if p != nil {
			t.Errorf("%s: panicked: %v", name, p)
			return false
		}
		return true
	case <-time.After(d):
		t.Errorf("%s: still running after %s (unbounded work)", name, d)
		return false
	}
}

func fastExec(t *testing.T, op MathOperation, expr string) (res MathResult, ok bool) {
	t.Helper()
	ok = bounded(t, string(op)+" "+expr, 2*time.Second, func() {
		res = NewFastNumericBackend().Execute(MathRequest{Operation: op, Expression: expr, Mode: ModeExact, PrecisionDigits: 15})
	})
	return
}

func TestModPowNeedsAPositiveModulus(t *testing.T) {
	// big.Int.Exp treats a zero modulus as "no modulus": 2^16000000 in full (and 2^(10^14) would take the process down)
	for _, expr := range []string{"2^16000000 mod 0", "2^16000000 mod abc", "2^16000000 mod -5"} {
		if res, ok := fastExec(t, OpModPow, expr); ok && res.Status == StatusSuccess {
			t.Errorf("%q must be refused, got %q", expr, res.ExactResult)
		}
	}
	if res, ok := fastExec(t, OpModPow, "4^13 mod 497"); !ok || res.ExactResult != "445" {
		t.Errorf("4^13 mod 497 = 445, got %q (%s)", res.ExactResult, res.ErrorMessage)
	}
}

func TestModInverseNeverPanics(t *testing.T) {
	for _, expr := range []string{"5 mod 0", "1 mod 0", "abc mod 7", "1 mod abc", "0 mod 0"} {
		if res, ok := fastExec(t, OpModInverse, expr); ok && res.Status == StatusSuccess {
			t.Errorf("%q must be refused, got %q", expr, res.ExactResult)
		}
	}
	if res, ok := fastExec(t, OpModInverse, "3 mod 11"); !ok || res.ExactResult != "4" {
		t.Errorf("3^-1 mod 11 = 4, got %q (%s)", res.ExactResult, res.ErrorMessage)
	}
}

func TestBinomialIsBounded(t *testing.T) {
	if res, ok := fastExec(t, OpProbability, "C(100000,50000)"); ok && res.Status == StatusSuccess {
		t.Errorf("C(100000,50000) is a 100000-bit number: must be refused, got %d digits", len(res.ExactResult))
	}
	if res, ok := fastExec(t, OpProbability, "C(9223372036854775807,4611686018427387903)"); ok && res.Status == StatusSuccess {
		t.Error("an out-of-range C(n,k) must be refused")
	}
	if res, ok := fastExec(t, OpProbability, "C(10,3)"); !ok || res.ExactResult != "120" {
		t.Errorf("C(10,3) = 120, got %q", res.ExactResult)
	}
}

func TestLCMOfZerosDoesNotPanic(t *testing.T) {
	if res, ok := fastExec(t, OpEvaluate, "LCM(0,0)"); ok && res.Status == StatusSuccess && res.ExactResult != "0" {
		t.Errorf("lcm(0,0) = 0, got %q", res.ExactResult)
	}
	if res, ok := fastExec(t, OpEvaluate, "LCM(4,6)"); !ok || res.ExactResult != "12" {
		t.Errorf("lcm(4,6) = 12, got %q", res.ExactResult)
	}
}

func TestExactRationalPowersCannotGrowWithoutBound(t *testing.T) {
	// each ^100 multiplies the size by 100: a few nested levels (a few hundred characters) would take the process down
	if res, ok := fastExec(t, OpEvaluate, "((9^100)^100)^100"); ok && res.Status == StatusSuccess {
		t.Errorf("a 1,000,000-digit result must be refused, got %d digits", len(res.ExactResult))
	}
	if res, ok := fastExec(t, OpEvaluate, "2^10"); !ok || res.ExactResult != "1024" {
		t.Errorf("2^10 = 1024, got %q", res.ExactResult)
	}
}

func TestHugePowerIsNotReportedAsZero(t *testing.T) {
	// 10^400 overflows a float64: the old code turned that into the exact result "0"
	res, ok := fastExec(t, OpEvaluate, "10^400")
	if !ok {
		return
	}
	want := "1" + strings.Repeat("0", 400)
	if res.Status == StatusSuccess && res.ExactResult != want {
		t.Errorf("10^400 must be exact or refused, got %q", res.ExactResult)
	}
}

func verify(t *testing.T, name string, args ...string) (res MathVerificationResult, ok bool) {
	t.Helper()
	ok = bounded(t, name, 2*time.Second, func() {
		res = GetTheoremRegistry().VerifyTheorem(name, args, "")
	})
	return
}

func TestTheoremVerifiersAreBounded(t *testing.T) {
	// Wilson loops p times; the Binomial Theorem n times; Euler's trial division sqrt(n) times
	if res, ok := verify(t, "Wilson's Theorem", "1000000000039"); ok && res.Matches {
		t.Errorf("Wilson with p = 10^12+39 cannot be checked exhaustively: %+v", res)
	}
	if res, ok := verify(t, "Binomial Theorem", "1", "1", "1099511627776"); ok && res.Matches {
		t.Errorf("Binomial Theorem with n = 2^40 cannot be expanded term by term: %+v", res)
	}
	if res, ok := verify(t, "Euler's Totient Theorem", "3", "2305843009213693951"); ok && res.Matches {
		t.Errorf("Euler's totient of 2^61-1 by trial division: %+v", res)
	}
	// and the ones that fit still work
	if res, ok := verify(t, "Wilson's Theorem", "13"); !ok || !res.Matches {
		t.Errorf("Wilson p=13: %+v", res)
	}
	if res, ok := verify(t, "Binomial Theorem", "2", "3", "5"); !ok || !res.Matches {
		t.Errorf("Binomial (2+3)^5: %+v", res)
	}
	if res, ok := verify(t, "Euler's Totient Theorem", "3", "10"); !ok || !res.Matches {
		t.Errorf("Euler a=3 n=10: %+v", res)
	}
}

func TestRSAVerifierRefusesWhatItCannotCompute(t *testing.T) {
	// N = 0 makes Exp compute m^e in full (and Mod divide by zero); a non-number made a nil *big.Int
	for _, args := range [][]string{{"2", "4000000", "3", "0"}, {"x", "3", "5", "7"}, {"2", "3", "5", "-7"}} {
		if res, ok := verify(t, "RSA Correctness Theorem", args...); ok && res.Matches {
			t.Errorf("RSA %v must be refused: %+v", args, res)
		}
	}
	if res, ok := verify(t, "RSA Correctness Theorem", "42", "17", "2753", "3233"); !ok || !res.Matches {
		t.Errorf("the textbook RSA key (3233, e=17, d=2753) must verify: %+v", res)
	}
}

func TestFermatVerifierCapsTheNumberSize(t *testing.T) {
	huge := "1" + strings.Repeat("0", 3000) + "7"
	res, ok := verify(t, "Fermat's Little Theorem", "3", huge)
	if ok && !strings.Contains(strings.ToLower(res.DiscrepancyDetails), "too large") {
		t.Errorf("a %d-digit modulus must be refused as too large: %+v", len(huge), res)
	}
	if res, ok := verify(t, "Fermat's Little Theorem", "3", "101"); !ok || !res.Matches {
		t.Errorf("3^100 mod 101: %+v", res)
	}
}

func TestUnifiedBackendBoundsItsNumberTheory(t *testing.T) {
	run := func(expr string) (res MathResult, ok bool) {
		ok = bounded(t, expr, 2*time.Second, func() {
			res = NewUnifiedCasBackend().Execute(MathRequest{Operation: OpEvaluate, Expression: expr})
		})
		return
	}
	for _, expr := range []string{
		"fib(1000000000000)",
		"euler_phi(2305843009213693951)",
		"is_prime(" + strings.Repeat("9", 1) + strings.Repeat("3", 5000) + ")",
		"power_mod(3, " + strings.Repeat("7", 5000) + ", 1000003)",
	} {
		short := expr
		if len(short) > 40 {
			short = short[:40] + "..."
		}
		if res, ok := run(expr); ok && res.Status == StatusSuccess {
			t.Errorf("%s must be refused, got %.40q", short, res.ExactResult)
		}
	}
	for expr, want := range map[string]string{"fib(10)": "55", "euler_phi(36)": "12", "is_prime(97)": "True", "power_mod(4, 13, 497)": "445"} {
		if res, ok := run(expr); !ok || res.ExactResult != want {
			t.Errorf("%s = %s, got %q (%s)", expr, want, res.ExactResult, res.ErrorMessage)
		}
	}
}

func TestSessionStateIsBounded(t *testing.T) {
	c := NewMathResultCache(100)
	for i := 0; i < 5000; i++ {
		c.SetSessionVar("session-"+strings.Repeat("x", i%7)+string(rune('a'+i%26))+string(rune('a'+(i/26)%26))+string(rune('a'+(i/676)%26)), "v", "1")
	}
	if n := len(c.sessionVars); n > 1024 {
		t.Errorf("%d sessions kept: a caller choosing session ids must not grow memory without end", n)
	}
	for i := 0; i < 1000; i++ {
		c.SetSessionVar("one", "var"+strings.Repeat("v", i%5)+string(rune('a'+i%26))+string(rune('a'+(i/26)%26)), "1")
	}
	if n := len(c.sessionVars["one"]); n > 64 {
		t.Errorf("%d variables kept in one session", n)
	}
	c.SetSessionVar("big", "x", strings.Repeat("9", 100000))
	if v, ok := c.GetSessionVar("big", "x"); ok && len(v) > 4096 {
		t.Errorf("a %d-char value was kept", len(v))
	}
	c.SetSessionVar("two", "keep", "5")
	if v, ok := c.GetSessionVar("two", "keep"); !ok || v != "5" {
		t.Errorf("a normal variable must be kept, got %q %v", v, ok)
	}
}

func TestResultCacheDoesNotLeakItsIndexes(t *testing.T) {
	c := NewMathResultCache(50)
	for i := 0; i < 500; i++ {
		req := MathRequest{TenantID: "t", Operation: OpEvaluate, Expression: strings.Repeat("1+", i%400) + "1"}
		req.Variable = strings.Repeat("v", i)
		c.Put(req, MathResult{ExactResult: "x"})
	}
	if n := len(c.tenantMap["t"]); n > 50 {
		t.Errorf("the tenant index holds %d keys after the cache evicted all but 50", n)
	}
	same := MathRequest{TenantID: "t2", Operation: OpEvaluate, Expression: "2+2"}
	c.Put(same, MathResult{ExactResult: "4"})
	before := len(c.lru)
	c.Put(same, MathResult{ExactResult: "4"})
	if len(c.lru) != before {
		t.Errorf("storing a key twice grew the eviction list from %d to %d", before, len(c.lru))
	}
	c.Put(MathRequest{Operation: OpEvaluate, Expression: "huge"}, MathResult{ExactResult: strings.Repeat("9", 1<<20)})
	if _, ok := c.Get(MathRequest{Operation: OpEvaluate, Expression: "huge"}); ok {
		t.Error("a 1 MB result must not be cached")
	}
}

func TestSessionVariableSubstitutionIsBoundedAndDeterministic(t *testing.T) {
	long := strings.Repeat("9", 4000)
	if _, ok := substituteSessionVars(strings.Repeat("x", 4000), map[string]string{"x": long}, 4096); ok {
		t.Error("4000 x's with a 4000-digit value is a 16 MB expression: must be refused before it is built")
	}
	got, ok := substituteSessionVars("ab + a", map[string]string{"a": "1", "ab": "2"}, 4096)
	if !ok || got != "2 + 1" {
		t.Errorf("the longest name goes first: want %q, got %q", "2 + 1", got)
	}
	for i := 0; i < 20; i++ {
		if again, _ := substituteSessionVars("ab + a", map[string]string{"a": "1", "ab": "2"}, 4096); again != got {
			t.Fatalf("the result must not depend on map order: %q then %q", got, again)
		}
	}
}

func TestOverlongExpressionIsRefusedBeforeItIsRead(t *testing.T) {
	rt := NewMathRuntime(nil)
	rt.Cache.SetSessionVar("s", "x", strings.Repeat("9", 4000))
	for _, req := range []MathRequest{
		{Operation: OpEvaluate, Expression: strings.Repeat("1+", 3_000_000) + "1"},
		{Operation: OpEvaluate, Expression: strings.Repeat("x", 4000), SessionID: "s"},
	} {
		var res MathResult
		if !bounded(t, "evaluate a huge expression", 2*time.Second, func() { res = rt.ProcessRequest(req) }) {
			continue
		}
		if res.Status != StatusInvalidExpression {
			t.Errorf("want invalid_expression, got %q (%s)", res.Status, res.ErrorMessage)
		}
	}
}
