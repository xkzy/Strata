package mathruntime

import (
	"fmt"
	"math/big"
	"sort"
	"strings"
)

// Limits on the work one request can ask for. The runtime is reachable over HTTP (/v1/strata/cas/solve, /math/evaluate,
// /math/theorems/verify, the MCP tools) by callers who are not trusted, and nothing here is cancelled once it started,
// so each operation is refused up front when its cost or its result would not be small. The numbers are generous for
// real use (a 4096-bit RSA key, a 20,000-digit exact result, 10,000 terms of a binomial expansion), and each one at
// its limit takes at most 74 ms (fib(100000); the others 60 ms or less), measured on a Ryzen 9 9950X3D.
const (
	maxRationalBits     = 1 << 16                  // an exact rational result: numerator + denominator bits (about 19,700 digits)
	maxBinomialN        = 20000                    // C(n, k): the result has up to n bits
	maxFibonacciN       = 100000                   // fib(n) has about 0.69 n bits and costs n additions
	maxTrialDivisionN   = int64(1_000_000_000_000) // euler_phi(n): trial division takes sqrt(n) steps
	maxWilsonP          = int64(10_000_000)        // Wilson's theorem is checked by multiplying 1..p-1
	maxBinomialTheoremN = 10000                    // (a+b)^n expanded term by term
	maxPrimalityBits    = 2048                     // ProbablyPrime(20): 20 modular exponentiations of this size
	maxModExpBits       = 4096                     // modular exponentiation operands (RSA-4096)
	maxPrecisionDigits  = 1000                     // digits of a numeric result

	maxTheoremArgs       = 16   // arguments of a theorem check
	maxTheoremArgLen     = 2048 // characters of one argument (a 4096-bit number has 1234 digits)
	maxTheoremAssumption = 8192 // characters of the stated assumptions

	maxSessions        = 1024  // sessions holding variables (the oldest is dropped for a new one)
	maxSessionVars     = 64    // variables in one session
	maxSessionValueLen = 4096  // characters of one variable's value
	maxCachedResultLen = 65536 // characters of a result worth caching
)

// ratBits is the size of an exact rational: its numerator's bits plus its denominator's.
func ratBits(r *big.Rat) int { return r.Num().BitLen() + r.Denom().BitLen() }

// bigArg parses a decimal integer argument of at most maxBits bits; the message says what is wrong when it is not one.
func bigArg(s string, maxBits int) (*big.Int, string) {
	if len(s) > maxTheoremArgLen {
		return nil, "argument too large"
	}
	n, ok := new(big.Int).SetString(s, 10)
	if !ok {
		return nil, "Invalid integer arguments"
	}
	if n.BitLen() > maxBits {
		return nil, fmt.Sprintf("argument too large: at most %d bits", maxBits)
	}
	return n, ""
}

// substituteSessionVars replaces each variable's name in expr by its value, longest names first (so "ab" is not
// broken by a variable "a") and in a fixed order (map order made the result depend on chance). It gives up, ok false,
// as soon as the result would be longer than maxLen: a short expression of one repeated variable name with a long value
// would otherwise grow to megabytes before anything checks its length.
func substituteSessionVars(expr string, vars map[string]string, maxLen int) (string, bool) {
	names := make([]string, 0, len(vars))
	for k := range vars {
		if k != "" {
			names = append(names, k)
		}
	}
	sort.Slice(names, func(i, j int) bool {
		if len(names[i]) != len(names[j]) {
			return len(names[i]) > len(names[j])
		}
		return names[i] < names[j]
	})
	for _, k := range names {
		v := vars[k]
		if n := strings.Count(expr, k); n > 0 {
			if maxLen > 0 && len(expr)+n*(len(v)-len(k)) > maxLen {
				return "", false
			}
			expr = strings.ReplaceAll(expr, k, v)
		}
	}
	return expr, true
}
