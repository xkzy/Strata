package mathruntime

import (
	"fmt"
	"math"
	"math/big"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
)

type MathRuntime struct {
	Limits         MathSecurityLimits
	Validator      *ExpressionValidator
	Parser         *ExpressionParser
	Cache          *MathResultCache
	FastBackend    *FastNumericBackend
	CasBackend     *UnifiedCasBackend
	MathicsBackend *MathicsBackend
	SageBackend    *SageBackend

	stats struct {
		totalCalculations    uint64
		cacheHits            uint64
		cacheMisses          uint64
		fastPathCount        uint64
		mathicsCount         uint64
		verificationCount    uint64
		verificationFailures uint64
		timeouts             uint64
		securityRejections   uint64
		totalExecTimeMs      float64
	}
	statsMu sync.Mutex
}

func NewMathRuntime(limits *MathSecurityLimits) *MathRuntime {
	lim := DefaultSecurityLimits()
	if limits != nil {
		lim = *limits
	}

	cas := NewUnifiedCasBackend()
	return &MathRuntime{
		Limits:         lim,
		Validator:      NewExpressionValidator(lim),
		Parser:         NewExpressionParser(),
		Cache:          NewMathResultCache(10000),
		FastBackend:    NewFastNumericBackend(),
		CasBackend:     cas,
		MathicsBackend: NewMathicsBackend(),
		SageBackend:    NewSageBackend(),
	}
}

func (r *MathRuntime) RouteBackend(req MathRequest, complexity int) bool {
	if req.Mode == ModeSymbolic {
		return false
	}
	switch req.Operation {
	case OpMatrixInverse, OpMatrixMultiply, OpMatrixTrace, OpEigenvalues, OpLinearSystem,
		OpStatistics, OpModInverse, OpModPow, OpDeterminant, OpProbability, OpEvaluate, OpNumericEvaluate:
		return true
	case OpSimplify, OpFactor, OpExpand, OpSolve, OpDifferentiate, OpIntegrate, OpLimit:
		return true
	}
	return false
}

func (r *MathRuntime) ProcessRequest(req MathRequest) MathResult {
	atomic.AddUint64(&r.stats.totalCalculations, 1)

	// Step 0: Check for Variable Assignment (e.g. x = 10)
	assignParts := strings.Split(req.Expression, "=")
	if len(assignParts) == 2 && !strings.ContainsAny(assignParts[0], "+-*/^()[]") {
		varName := strings.TrimSpace(assignParts[0])
		if varName != "" && !strings.Contains(varName, "==") {
			valReq := req
			valReq.Expression = strings.TrimSpace(assignParts[1])
			valRes := r.ProcessRequest(valReq)
			if valRes.Status == StatusSuccess {
				if req.SessionID != "" {
					r.Cache.SetSessionVar(req.SessionID, varName, valRes.ExactResult)
				}
				return MathResult{
					RequestID:           req.RequestID,
					Status:              StatusSuccess,
					BackendName:         "MathSessionStore",
					BackendVersion:      "1.0.0",
					CanonicalExpression: req.Expression,
					ExactResult:         valRes.ExactResult,
					NumericResult:       valRes.NumericResult,
					RawResult:           valRes.ExactResult,
					CompactObservation:  fmt.Sprintf("[MathAssignment: %s = %s]", varName, valRes.ExactResult),
					ExecutionTimeMs:     0.01,
				}
			}
		}
	}

	// Step 0b: Substitute Session Variables
	if req.SessionID != "" {
		if val, ok := r.Cache.GetSessionVar(req.SessionID, strings.TrimSpace(req.Expression)); ok {
			return MathResult{
				RequestID:           req.RequestID,
				Status:              StatusSuccess,
				BackendName:         "MathSessionStore",
				BackendVersion:      "1.0.0",
				CanonicalExpression: req.Expression,
				ExactResult:         val,
				NumericResult:       val,
				RawResult:           val,
				CompactObservation:  fmt.Sprintf("[MathVar: %s = %s]", req.Expression, val),
				ExecutionTimeMs:     0.01,
			}
		}
		vars := r.Cache.GetSessionVars(req.SessionID)
		for k, v := range vars {
			req.Expression = strings.ReplaceAll(req.Expression, k, v)
		}
	}

	// Step 1: Validation
	valid, err := r.Validator.Validate(req.Expression)
	if !valid {
		atomic.AddUint64(&r.stats.securityRejections, 1)
		return MathResult{
			RequestID:          req.RequestID,
			Status:             StatusInvalidExpression,
			ErrorMessage:       err.Error(),
			CompactObservation: fmt.Sprintf("[MathError: Invalid expression: %s]", err.Error()),
		}
	}

	complexity := r.Validator.ComputeComplexity(req.Expression)

	// Step 2: Cache Lookup
	if cached, ok := r.Cache.Get(req); ok {
		atomic.AddUint64(&r.stats.cacheHits, 1)
		cached.ComplexityScore = complexity
		return cached
	}
	atomic.AddUint64(&r.stats.cacheMisses, 1)

	// Step 3: Calculation Routing
	expr := req.Expression
	hasSage := strings.Contains(expr, "var(") ||
		strings.Contains(expr, ".diff(") ||
		strings.Contains(expr, ".derivative(") ||
		strings.Contains(expr, ".integrate(") ||
		strings.Contains(expr, ".integral(") ||
		strings.Contains(expr, ".factor(") ||
		strings.Contains(expr, ".expand(") ||
		strings.Contains(expr, ".simplify(") ||
		strings.Contains(expr, ".roots(") ||
		strings.Contains(expr, ".det(") ||
		strings.Contains(expr, ".inverse(") ||
		strings.Contains(expr, ".transpose(") ||
		strings.Contains(expr, "matrix([") ||
		strings.Contains(expr, "is_prime(") ||
		strings.Contains(expr, "euler_phi(") ||
		strings.Contains(expr, "fibonacci(") ||
		strings.Contains(expr, "power_mod(") ||
		strings.Contains(expr, "xgcd(") ||
		strings.ToLower(req.TenantID) == "sage"

	useFast := r.RouteBackend(req, complexity)
	var res MathResult
	if hasSage && r.SageBackend.Available {
		res = r.SageBackend.Execute(req)
	} else if useFast {
		atomic.AddUint64(&r.stats.fastPathCount, 1)
		res = r.FastBackend.Execute(req)
	} else {
		atomic.AddUint64(&r.stats.mathicsCount, 1)
		res = r.MathicsBackend.Execute(req)
	}

	res.ComplexityScore = complexity
	if res.CompactObservation == "" {
		res.CompactObservation = r.Parser.FormatCompactObservation(req.Expression, res.ExactResult, res.NumericResult, req.Operation)
	}
	res.ObservationTokens = int(math.Max(1, float64(len(res.CompactObservation)+3)/4.0))

	r.statsMu.Lock()
	r.stats.totalExecTimeMs += res.ExecutionTimeMs
	r.statsMu.Unlock()

	// Step 4: Cache Put
	if res.Status == StatusSuccess {
		r.Cache.Put(req, res)
	}

	return res
}

func (r *MathRuntime) Evaluate(expression string, mode MathMode, tenantID, sessionID string) MathResult {
	req := MathRequest{
		Operation:  OpEvaluate,
		Expression: expression,
		Mode:       mode,
		TenantID:   tenantID,
		SessionID:  sessionID,
	}
	return r.ProcessRequest(req)
}

func (r *MathRuntime) Simplify(expression, tenantID, sessionID string) MathResult {
	req := MathRequest{
		Operation:  OpSimplify,
		Expression: expression,
		Mode:       ModeSymbolic,
		TenantID:   tenantID,
		SessionID:  sessionID,
	}
	return r.ProcessRequest(req)
}

func (r *MathRuntime) Expand(expression, tenantID, sessionID string) MathResult {
	req := MathRequest{
		Operation:  OpExpand,
		Expression: expression,
		Mode:       ModeSymbolic,
		TenantID:   tenantID,
		SessionID:  sessionID,
	}
	return r.ProcessRequest(req)
}

func (r *MathRuntime) Factor(expression, tenantID, sessionID string) MathResult {
	req := MathRequest{
		Operation:  OpFactor,
		Expression: expression,
		Mode:       ModeSymbolic,
		TenantID:   tenantID,
		SessionID:  sessionID,
	}
	return r.ProcessRequest(req)
}

func (r *MathRuntime) Solve(equation, variable, tenantID, sessionID string) MathResult {
	req := MathRequest{
		Operation:  OpSolve,
		Expression: equation,
		Variable:   variable,
		Mode:       ModeSymbolic,
		TenantID:   tenantID,
		SessionID:  sessionID,
	}
	return r.ProcessRequest(req)
}

func (r *MathRuntime) Differentiate(expression, variable string, order int, tenantID, sessionID string) MathResult {
	req := MathRequest{
		Operation:  OpDifferentiate,
		Expression: expression,
		Variable:   variable,
		Order:      order,
		Mode:       ModeSymbolic,
		TenantID:   tenantID,
		SessionID:  sessionID,
	}
	return r.ProcessRequest(req)
}

func (r *MathRuntime) Integrate(expression, variable, tenantID, sessionID string) MathResult {
	req := MathRequest{
		Operation:  OpIntegrate,
		Expression: expression,
		Variable:   variable,
		Mode:       ModeSymbolic,
		TenantID:   tenantID,
		SessionID:  sessionID,
	}
	return r.ProcessRequest(req)
}

func (r *MathRuntime) Limit(expression, variable, point, tenantID, sessionID string) MathResult {
	req := MathRequest{
		Operation:  OpLimit,
		Expression: expression,
		Variable:   variable,
		Point:      point,
		Mode:       ModeSymbolic,
		TenantID:   tenantID,
		SessionID:  sessionID,
	}
	return r.ProcessRequest(req)
}

func (r *MathRuntime) Series(expression, variable, point string, order int, tenantID, sessionID string) MathResult {
	req := MathRequest{
		Operation:  OpSeries,
		Expression: expression,
		Variable:   variable,
		Point:      point,
		Order:      order,
		Mode:       ModeSymbolic,
		TenantID:   tenantID,
		SessionID:  sessionID,
	}
	return r.ProcessRequest(req)
}

func (r *MathRuntime) NumericEvaluate(expression string, precisionDigits int, tenantID, sessionID string) MathResult {
	req := MathRequest{
		Operation:       OpNumericEvaluate,
		Expression:      expression,
		Mode:            ModeNumeric,
		PrecisionDigits: precisionDigits,
		TenantID:        tenantID,
		SessionID:       sessionID,
	}
	return r.ProcessRequest(req)
}

func (r *MathRuntime) VerifyCalculation(llmOutput, expectedExpr string) MathVerificationResult {
	atomic.AddUint64(&r.stats.verificationCount, 1)

	ver := MathVerificationResult{}
	gt := r.Evaluate(expectedExpr, ModeExact, "", "")
	if gt.Status != StatusSuccess {
		ver.Matches = false
		ver.DiscrepancyDetails = fmt.Sprintf("Ground truth calculation failed: %s", gt.ErrorMessage)
		atomic.AddUint64(&r.stats.verificationFailures, 1)
		return ver
	}

	ver.GroundTruthResult = gt.ExactResult

	// Extract claimed result from LLM output
	numRe := regexp.MustCompile(`(?i)(?:=|\bis\b|\bresult\b|\bequals\b)\s*([0-9\.\,\-\/]+)`)
	claimed := ""
	if m := numRe.FindStringSubmatch(llmOutput); len(m) > 1 {
		claimed = r.Parser.Canonicalize(m[1])
	} else {
		claimed = r.Parser.Canonicalize(llmOutput)
	}
	ver.LLMClaimedResult = claimed

	canonGT := r.Parser.Canonicalize(gt.ExactResult)
	if claimed == canonGT {
		ver.Matches = true
		ver.RelativeError = 0.0
		return ver
	}

	// Try numeric tolerance check
	ratClaimed, _, err1 := new(FastNumericBackend).evalExactRational(claimed)
	ratGT, _, err2 := new(FastNumericBackend).evalExactRational(canonGT)
	if err1 == nil && err2 == nil {
		diff := new(big.Rat).Sub(ratClaimed, ratGT)
		fDiff, _ := diff.Float64()
		fDiff = math.Abs(fDiff)
		fGT, _ := ratGT.Float64()
		denom := math.Max(math.Abs(fGT), 1e-9)
		ver.RelativeError = fDiff / denom
		if fDiff < 1e-6 {
			ver.Matches = true
			return ver
		}
	} else {
		fClaimed, errC := strconv.ParseFloat(claimed, 64)
		fGTVal, errG := strconv.ParseFloat(canonGT, 64)
		if errC == nil && errG == nil {
			fDiff := math.Abs(fClaimed - fGTVal)
			denom := math.Max(math.Abs(fGTVal), 1e-9)
			ver.RelativeError = fDiff / denom
			if fDiff < 1e-6 {
				ver.Matches = true
				return ver
			}
		}
	}

	ver.Matches = false
	ver.DiscrepancyDetails = fmt.Sprintf("LLM output '%s' does not match verified mathematical result '%s'", claimed, gt.ExactResult)
	atomic.AddUint64(&r.stats.verificationFailures, 1)
	return ver
}

func (r *MathRuntime) InterceptAndEvaluate(text string) (string, []MathResult) {
	intents := r.Parser.DetectCalculationIntents(text)
	if len(intents) == 0 {
		return text, nil
	}

	var results []MathResult
	transformed := text

	for _, intent := range intents {
		req := MathRequest{
			Operation:  intent.Operation,
			Expression: intent.Expression,
			Variable:   intent.Variable,
			Point:      intent.Point,
			Order:      intent.Order,
			Mode:       intent.Mode,
		}
		res := r.ProcessRequest(req)
		results = append(results, res)

		if res.Status == StatusSuccess && intent.RawMatch != "" {
			obs := " " + res.CompactObservation
			transformed = strings.Replace(transformed, intent.RawMatch, intent.RawMatch+obs, 1)
		}
	}

	return transformed, results
}

func (r *MathRuntime) InterceptAndVerifyIntent(text string) *MathResult {
	if text == "" {
		return nil
	}

	// 1. Check for theorem verification intent
	reg := GetTheoremRegistry()
	for _, thm := range reg.theorems {
		if strings.Contains(strings.ToLower(text), strings.ToLower(thm.Name)) || strings.Contains(strings.ToLower(text), strings.ToLower(thm.ID)) {
			numRe := regexp.MustCompile(`[-+]?[0-9]+(?:\.[0-9]+)?`)
			matches := numRe.FindAllString(text, -1)
			ver := reg.VerifyTheorem(thm.ID, matches, "")
			if ver.Matches {
				return &MathResult{
					Status:              StatusSuccess,
					BackendName:         "StrataTheoremEngine",
					BackendVersion:      "1.0.0",
					CanonicalExpression: thm.Name,
					ExactResult:         ver.GroundTruthResult,
					CompactObservation:  fmt.Sprintf("[Theorem Verified: %s => %s]", thm.Name, ver.GroundTruthResult),
				}
			}
		}
	}

	// 2. Check for special Sage/NumberTheory functions (euler_phi, is_prime, xgcd, fibonacci, etc.)
	sageFuncRe := regexp.MustCompile(`(?i)\b(euler_phi|is_prime|xgcd|fibonacci|power_mod|divisors)\s*\(([0-9\s\,\-\+]+)\)`)
	if m := sageFuncRe.FindStringSubmatch(text); len(m) > 0 {
		res := r.ProcessRequest(MathRequest{
			Operation:  OpEvaluate,
			Expression: m[0],
			Mode:       ModeExact,
		})
		if res.Status == StatusSuccess && (res.ExactResult != "" || res.NumericResult != "") {
			return &res
		}
	}

	// 3. Check for Matrix Determinant
	detRe := regexp.MustCompile(`(?i)\b(?:det|determinant(?:\s+of)?)\s*\(?\s*(\[\[[0-9\s\,\-\+\.\/\[\]]+\]\])\s*\)?`)
	if m := detRe.FindStringSubmatch(text); len(m) > 1 {
		res := r.ProcessRequest(MathRequest{
			Operation:  OpDeterminant,
			Expression: m[1],
			Mode:       ModeExact,
		})
		if res.Status == StatusSuccess && (res.ExactResult != "" || res.NumericResult != "") {
			return &res
		}
	}

	// 4. Check for general calculation intents
	intents := r.Parser.DetectCalculationIntents(text)
	if len(intents) > 0 {
		intent := intents[0]
		req := MathRequest{
			Operation:  intent.Operation,
			Expression: intent.Expression,
			Variable:   intent.Variable,
			Point:      intent.Point,
			Order:      intent.Order,
			Mode:       ModeExact,
		}
		res := r.ProcessRequest(req)
		if res.Status == StatusSuccess && (res.ExactResult != "" || res.NumericResult != "") {
			return &res
		}
	}

	return nil
}

func (r *MathRuntime) GetStats() MathRuntimeStats {
	total := atomic.LoadUint64(&r.stats.totalCalculations)
	hits := atomic.LoadUint64(&r.stats.cacheHits)
	misses := atomic.LoadUint64(&r.stats.cacheMisses)
	fast := atomic.LoadUint64(&r.stats.fastPathCount)
	mathics := atomic.LoadUint64(&r.stats.mathicsCount)
	ver := atomic.LoadUint64(&r.stats.verificationCount)
	verFail := atomic.LoadUint64(&r.stats.verificationFailures)
	timeouts := atomic.LoadUint64(&r.stats.timeouts)
	secRej := atomic.LoadUint64(&r.stats.securityRejections)

	r.statsMu.Lock()
	totalTime := r.stats.totalExecTimeMs
	r.statsMu.Unlock()

	hitRate := 0.0
	if total > 0 {
		hitRate = float64(hits) / float64(total)
	}
	avgTime := 0.0
	if total > 0 {
		avgTime = totalTime / float64(total)
	}

	return MathRuntimeStats{
		TotalCalculations:    total,
		CacheHits:            hits,
		CacheMisses:          misses,
		CacheHitRate:         hitRate,
		FastPathCount:        fast,
		MathicsCount:         mathics,
		VerificationCount:    ver,
		VerificationFailures: verFail,
		Timeouts:             timeouts,
		SecurityRejections:   secRej,
		AvgExecutionTimeMs:   avgTime,
		TotalExecutionTimeMs: totalTime,
	}
}
