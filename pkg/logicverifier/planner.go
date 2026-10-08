// pkg/logicverifier/planner.go - Verification Routing Planner & Fast Numeric Solver
package logicverifier

import (
	"fmt"
	"math"
	"math/big"
	"strconv"
	"strings"
	"time"
)

type VerificationPlanner struct {
	cache            *VerificationCache
	ruleEngine       *RuleEngine
	constraintEngine *ConstraintEngine
	unitVerifier     *UnitVerifier
	schemaVerifier   *SchemaVerifier
	metrics          *MetricsCollector
}

func NewVerificationPlanner(cache *VerificationCache, metrics *MetricsCollector) *VerificationPlanner {
	if cache == nil {
		cache = NewVerificationCache(10000, 1*time.Hour)
	}
	if metrics == nil {
		metrics = NewMetricsCollector()
	}
	return &VerificationPlanner{
		cache:            cache,
		ruleEngine:       NewRuleEngine(),
		constraintEngine: NewConstraintEngine(),
		unitVerifier:     NewUnitVerifier(),
		schemaVerifier:   NewSchemaVerifier(),
		metrics:          metrics,
	}
}

func (vp *VerificationPlanner) Verify(claim VerificationClaim) VerificationResult {
	startTime := time.Now()
	hash := ComputeCanonicalHash(claim)

	// 1. Check Cache
	if cached, hit := vp.cache.Get(hash); hit {
		cached.CanonicalHash = hash
		cached.ExecutionTimeMs = float64(time.Since(startTime).Microseconds()) / 1000.0
		vp.metrics.Record(cached)
		return cached
	}

	var result VerificationResult

	switch claim.Type {
	case ClaimArithmetic, ClaimSymbolic, ClaimEquation:
		result = vp.verifyArithmetic(claim)
	case ClaimProposition, ClaimConsistency:
		result = vp.ruleEngine.Verify(claim)
	case ClaimConstraint:
		result = vp.constraintEngine.Verify(claim)
	case ClaimUnitDimension:
		result = vp.unitVerifier.Verify(claim)
	case ClaimSchemaType, ClaimCodeSyntax:
		result = vp.schemaVerifier.Verify(claim)
	case ClaimComposite:
		result = vp.verifyComposite(claim)
	default:
		// Conservative fallback: NEVER guess. Always return UNKNOWN.
		result = VerificationResult{
			ClaimID:            claim.ClaimID,
			Type:               ClaimUnknown,
			BackendUsed:        "conservative_fallback",
			Status:             StatusUnknown,
			Evidence:           "no deterministic backend available for claim type",
			CompactObservation: "Verification: UNKNOWN | Unsupported claim type",
		}
	}

	result.CanonicalHash = hash
	result.RawStatement = claim.RawStatement
	result.ExecutionTimeMs = float64(time.Since(startTime).Microseconds()) / 1000.0
	result.Timestamp = time.Now()
	if result.ObservationTokens == 0 && result.CompactObservation != "" {
		result.ObservationTokens = int64(len(strings.Fields(result.CompactObservation)))
	}

	// Cache successful PASS or definitive FAIL results
	if result.Status == StatusPass || result.Status == StatusFail {
		vp.cache.Put(hash, result)
	}

	vp.metrics.Record(result)
	return result
}

// verifyArithmetic handles exact rational and floating point evaluation
func (vp *VerificationPlanner) verifyArithmetic(claim VerificationClaim) VerificationResult {
	res := VerificationResult{
		ClaimID:     claim.ClaimID,
		Type:        claim.Type,
		BackendUsed: "deterministic_numeric_engine",
		Status:      StatusUnknown,
	}

	expr := strings.TrimSpace(claim.Expression)
	claimed := strings.TrimSpace(claim.ClaimedValue)

	if expr == "" || claimed == "" {
		res.Status = StatusUnknown
		res.Evidence = "empty expression or claimed value"
		return res
	}

	// 1. Exact rational check e.g. 1/3 + 1/6 == 1/2 or 0.5
	actualRat, errRat := evalRationalExpr(expr)
	claimedRat, errClaimedRat := evalRationalExpr(claimed)

	if errRat == nil && errClaimedRat == nil {
		if actualRat.Cmp(claimedRat) == 0 {
			res.Status = StatusPass
			res.ActualValue = claimedRat.RatString()
			res.ExpectedValue = actualRat.RatString()
			res.Evidence = fmt.Sprintf("exact rational equivalence: %s == %s", expr, actualRat.RatString())
			res.CompactObservation = fmt.Sprintf("Verification: PASS | %s = %s", expr, actualRat.RatString())
			return res
		}
	}

	// 2. High precision big.Float / float64 evaluation
	actualFloat, errFloat := evalFloatExpr(expr)
	claimedFloat, errClaimFloat := evalFloatExpr(claimed)

	if errFloat != nil || errClaimFloat != nil {
		res.Status = StatusUnknown
		res.Evidence = fmt.Sprintf("expression evaluation failed: %v / %v", errFloat, errClaimFloat)
		return res
	}

	res.ExpectedValue = fmt.Sprintf("%g", actualFloat)
	res.ActualValue = fmt.Sprintf("%g", claimedFloat)

	tol := claim.Tolerance.AbsTol
	if tol == 0 {
		tol = 1e-6
	}
	relTol := claim.Tolerance.RelTol
	if relTol == 0 {
		relTol = 1e-6
	}

	diff := math.Abs(actualFloat - claimedFloat)
	maxMag := math.Max(math.Abs(actualFloat), math.Abs(claimedFloat))
	if maxMag == 0 {
		maxMag = 1.0
	}

	if diff <= tol || (diff/maxMag) <= relTol {
		res.Status = StatusPass
		res.Evidence = fmt.Sprintf("numeric evaluation matches within tolerance: expected %g, claimed %g (diff: %g)", actualFloat, claimedFloat, diff)
		res.CompactObservation = fmt.Sprintf("Verification: PASS | %s = %g", expr, actualFloat)
	} else {
		res.Status = StatusFail
		res.FailureReason = fmt.Sprintf("numeric discrepancy: expected %g, claimed %g (diff: %g)", actualFloat, claimedFloat, diff)
		res.Evidence = fmt.Sprintf("calculated %g != claimed %g", actualFloat, claimedFloat)
		res.CompactObservation = fmt.Sprintf("Verification: FAIL | Expected %g, got %g", actualFloat, claimedFloat)
	}

	return res
}

func (vp *VerificationPlanner) verifyComposite(claim VerificationClaim) VerificationResult {
	res := VerificationResult{
		ClaimID:     claim.ClaimID,
		Type:        ClaimComposite,
		BackendUsed: "composite_verifier",
		Status:      StatusPass,
	}

	if len(claim.SubClaims) == 0 {
		res.Status = StatusUnknown
		res.Evidence = "composite claim contains no sub-claims"
		return res
	}

	passCount := 0
	failCount := 0
	unknownCount := 0

	for _, sub := range claim.SubClaims {
		subRes := vp.Verify(sub)
		subDesc := sub.ClaimID
		if subDesc == "" {
			subDesc = string(sub.Type) + ":" + sub.Expression
		}

		switch subRes.Status {
		case StatusPass:
			passCount++
			res.VerifiedComponents = append(res.VerifiedComponents, subDesc)
		case StatusFail:
			failCount++
			res.FailedComponents = append(res.FailedComponents, fmt.Sprintf("%s (%s)", subDesc, subRes.FailureReason))
		default:
			unknownCount++
			res.UnknownComponents = append(res.UnknownComponents, subDesc)
		}
	}

	if failCount > 0 && passCount == 0 {
		res.Status = StatusFail
		res.FailureReason = fmt.Sprintf("all sub-claims failed (%d/%d)", failCount, len(claim.SubClaims))
		res.CompactObservation = fmt.Sprintf("Verification: FAIL | All %d components failed", failCount)
	} else if failCount > 0 && passCount > 0 {
		res.Status = StatusPartial
		res.FailureReason = fmt.Sprintf("%d sub-claims failed, %d passed", failCount, passCount)
		res.CompactObservation = fmt.Sprintf("Verification: PARTIAL | %d passed, %d failed", passCount, failCount)
	} else if unknownCount > 0 && passCount > 0 {
		res.Status = StatusPartial
		res.CompactObservation = fmt.Sprintf("Verification: PARTIAL | %d passed, %d unknown", passCount, unknownCount)
	} else if unknownCount > 0 && passCount == 0 {
		res.Status = StatusUnknown
		res.Evidence = "could not verify any sub-claim"
		res.CompactObservation = "Verification: UNKNOWN | Unable to verify sub-claims"
	} else {
		res.Status = StatusPass
		res.Evidence = fmt.Sprintf("all %d sub-claims verified successfully", passCount)
		res.CompactObservation = fmt.Sprintf("Verification: PASS | All %d components verified", passCount)
	}

	return res
}

// Simple recursive-descent evaluator for arithmetic expressions
func evalFloatExpr(expr string) (float64, error) {
	expr = strings.ReplaceAll(expr, " ", "")
	expr = strings.ReplaceAll(expr, "×", "*")
	expr = strings.ReplaceAll(expr, "÷", "/")
	expr = strings.ReplaceAll(expr, "**", "^")

	p := &mathParser{src: expr, pos: 0}
	val, err := p.parseExpression()
	if err != nil {
		return 0, err
	}
	if p.pos < len(p.src) {
		return 0, fmt.Errorf("unexpected token at pos %d: %s", p.pos, p.src[p.pos:])
	}
	return val, nil
}

type mathParser struct {
	src string
	pos int
}

func (p *mathParser) peek() byte {
	if p.pos < len(p.src) {
		return p.src[p.pos]
	}
	return 0
}

func (p *mathParser) next() byte {
	ch := p.peek()
	if ch != 0 {
		p.pos++
	}
	return ch
}

func (p *mathParser) parseExpression() (float64, error) {
	val, err := p.parseTerm()
	if err != nil {
		return 0, err
	}

	for p.peek() == '+' || p.peek() == '-' {
		op := p.next()
		right, err := p.parseTerm()
		if err != nil {
			return 0, err
		}
		if op == '+' {
			val += right
		} else {
			val -= right
		}
	}
	return val, nil
}

func (p *mathParser) parseTerm() (float64, error) {
	val, err := p.parseFactor()
	if err != nil {
		return 0, err
	}

	for p.peek() == '*' || p.peek() == '/' || p.peek() == '%' {
		op := p.next()
		right, err := p.parseFactor()
		if err != nil {
			return 0, err
		}
		if op == '*' {
			val *= right
		} else if op == '/' {
			if right == 0 {
				return 0, fmt.Errorf("division by zero")
			}
			val /= right
		} else {
			val = math.Mod(val, right)
		}
	}
	return val, nil
}

func (p *mathParser) parseFactor() (float64, error) {
	val, err := p.parsePrimary()
	if err != nil {
		return 0, err
	}

	if p.peek() == '^' {
		p.next()
		exponent, err := p.parseFactor()
		if err != nil {
			return 0, err
		}
		val = math.Pow(val, exponent)
	}
	return val, nil
}

func (p *mathParser) parsePrimary() (float64, error) {
	// Handle unary plus/minus
	if p.peek() == '+' {
		p.next()
		return p.parsePrimary()
	}
	if p.peek() == '-' {
		p.next()
		val, err := p.parsePrimary()
		return -val, err
	}

	// Handle parentheses
	if p.peek() == '(' {
		p.next()
		val, err := p.parseExpression()
		if err != nil {
			return 0, err
		}
		if p.peek() != ')' {
			return 0, fmt.Errorf("expected closing parenthesis")
		}
		p.next()
		return val, nil
	}

	// Handle functions like sqrt(...)
	if strings.HasPrefix(p.src[p.pos:], "sqrt(") {
		p.pos += 5
		val, err := p.parseExpression()
		if err != nil {
			return 0, err
		}
		if p.peek() != ')' {
			return 0, fmt.Errorf("expected closing parenthesis after sqrt")
		}
		p.next()
		if val < 0 {
			return 0, fmt.Errorf("sqrt of negative number")
		}
		return math.Sqrt(val), nil
	}

	// Parse number
	start := p.pos
	for p.pos < len(p.src) && ((p.src[p.pos] >= '0' && p.src[p.pos] <= '9') || p.src[p.pos] == '.' || p.src[p.pos] == 'e' || p.src[p.pos] == 'E') {
		p.pos++
	}

	if start == p.pos {
		return 0, fmt.Errorf("expected number at pos %d", p.pos)
	}

	numStr := p.src[start:p.pos]
	val, err := strconv.ParseFloat(numStr, 64)
	if err != nil {
		return 0, err
	}
	return val, nil
}

func evalRationalExpr(expr string) (*big.Rat, error) {
	expr = strings.TrimSpace(expr)
	// Try parsing simple fraction "1/2" or sum "1/3 + 1/6"
	parts := strings.Split(expr, "+")
	if len(parts) > 1 {
		sum := new(big.Rat)
		for _, part := range parts {
			r, err := evalRationalExpr(strings.TrimSpace(part))
			if err != nil {
				return nil, err
			}
			sum.Add(sum, r)
		}
		return sum, nil
	}

	// Single term: either "1/3" or "0.5" or "10"
	if slashParts := strings.Split(expr, "/"); len(slashParts) == 2 {
		num, ok1 := new(big.Int).SetString(strings.TrimSpace(slashParts[0]), 10)
		den, ok2 := new(big.Int).SetString(strings.TrimSpace(slashParts[1]), 10)
		if ok1 && ok2 && den.Sign() != 0 {
			return big.NewRat(0, 1).SetFrac(num, den), nil
		}
	}

	r := new(big.Rat)
	if _, ok := r.SetString(expr); ok {
		return r, nil
	}

	return nil, fmt.Errorf("cannot parse as exact rational")
}
