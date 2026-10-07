// Package mathruntime provides the Unified CAS Engine for Strata across General Mathematics, Physics, and Engineering.
package mathruntime

import (
	"context"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

// CASProblem describes a mathematical, physical, or engineering problem to be solved or verified
type CASProblem struct {
	ID          string            `json:"id"`
	Domain      CASDomainType     `json:"domain"`
	Expression  string            `json:"expression"`
	FormulaID   string            `json:"formula_id,omitempty"`
	TargetVar   string            `json:"target_var,omitempty"`
	Variables   map[string]string `json:"variables,omitempty"` // e.g. {"m": "10 kg", "a": "2 m/s^2"}
	Assumptions string            `json:"assumptions,omitempty"`
	Precision   int               `json:"precision,omitempty"`
	SessionID   string            `json:"session_id,omitempty"`
	TenantID    string            `json:"tenant_id,omitempty"`
}

// CASResult represents the authoritative deterministic outcome with complete physical provenance
type CASResult struct {
	ProblemID           string          `json:"problem_id,omitempty"`
	Status              MathStatus      `json:"status"`
	Domain              CASDomainType   `json:"domain"`
	CanonicalExpression string          `json:"canonical_expression"`
	ExactResult         string          `json:"exact_result"`
	NumericResult       string          `json:"numeric_result,omitempty"`
	Unit                string          `json:"unit,omitempty"`
	Dimension           string          `json:"dimension,omitempty"`
	FormulaUsed         string          `json:"formula_used,omitempty"`
	ConstantsUsed       []string        `json:"constants_used,omitempty"`
	BackendName         string          `json:"backend_name"`
	BackendVersion      string          `json:"backend_version"`
	ExecutionTimeMs     float64         `json:"execution_time_ms"`
	CompactObservation  string          `json:"compact_observation"`
	ErrorMessage        string          `json:"error_message,omitempty"`
	VerificationState   string          `json:"verification_state,omitempty"` // VERIFIED, CONTRADICTED, UNKNOWN, DIMENSION_ERROR
	CacheHit            bool            `json:"cache_hit"`
	Provenance          string          `json:"provenance"`
}

// UnifiedCASEngine defines the contract for Strata's shared CAS substrate
type UnifiedCASEngine interface {
	Solve(ctx context.Context, prob CASProblem) (CASResult, error)
	VerifyPhysicalClaim(ctx context.Context, claimStr string) (MathVerificationResult, error)
	InterceptScientificIntent(text string) *CASResult
	GetConstants() *ConstantRegistry
	GetFormulas() *FormulaRegistry
	GetTheorems() *TheoremRegistry
}

// UnifiedCASEngineImpl is the authoritative Go-native CAS engine
type UnifiedCASEngineImpl struct {
	mathRuntime *MathRuntime
	constants   *ConstantRegistry
	formulas    *FormulaRegistry
	theorems    *TheoremRegistry
	mu          sync.RWMutex
}

var (
	defaultCASEngine *UnifiedCASEngineImpl
	casOnce          sync.Once
)

func GetUnifiedCASEngine() *UnifiedCASEngineImpl {
	casOnce.Do(func() {
		defaultCASEngine = NewUnifiedCASEngine(nil)
	})
	return defaultCASEngine
}

func NewUnifiedCASEngine(mr *MathRuntime) *UnifiedCASEngineImpl {
	if mr == nil {
		mr = NewMathRuntime(nil)
	}
	return &UnifiedCASEngineImpl{
		mathRuntime: mr,
		constants:   GetConstantRegistry(),
		formulas:    GetFormulaRegistry(),
		theorems:    GetTheoremRegistry(),
	}
}

func (e *UnifiedCASEngineImpl) GetConstants() *ConstantRegistry {
	return e.constants
}

func (e *UnifiedCASEngineImpl) GetFormulas() *FormulaRegistry {
	return e.formulas
}

func (e *UnifiedCASEngineImpl) GetTheorems() *TheoremRegistry {
	return e.theorems
}

// Solve executes deterministic resolution for physical and mathematical problems
func (e *UnifiedCASEngineImpl) Solve(ctx context.Context, prob CASProblem) (CASResult, error) {
	start := time.Now()

	// 1. If formula is specified or recognized
	if prob.FormulaID != "" {
		if formula, ok := e.formulas.Get(prob.FormulaID); ok {
			parsedVars := make(map[string]float64)
			for k, vStr := range prob.Variables {
				q, err := parseQuantityString(vStr)
				if err == nil {
					parsedVars[k] = q.NumericValue
				} else {
					if fVal, err2 := strconv.ParseFloat(strings.TrimSpace(vStr), 64); err2 == nil {
						parsedVars[k] = fVal
					}
				}
			}

			if formula.SolveFunc != nil {
				val, err := formula.SolveFunc(parsedVars)
				if err == nil {
					dur := float64(time.Since(start).Microseconds()) / 1000.0
					valStr := formatPhysicalResult(val, formula.Unit)
					return CASResult{
						ProblemID:           prob.ID,
						Status:              StatusSuccess,
						Domain:              formula.Domain,
						CanonicalExpression: formula.Equation,
						ExactResult:         valStr,
						NumericResult:       fmt.Sprintf("%g", val),
						Unit:                formula.Unit,
						Dimension:           formula.Dimensions.String(),
						FormulaUsed:         formula.Name,
						BackendName:         "UnifiedCASPhysicsEngine",
						BackendVersion:      "1.0.0",
						ExecutionTimeMs:     dur,
						CompactObservation:  fmt.Sprintf("[CAS Physics: %s => %s]", formula.Name, valStr),
						VerificationState:   "VERIFIED",
						Provenance:          fmt.Sprintf("Formula: %s (%s)", formula.Name, formula.Literature),
					}, nil
				}
			}
		}
	}

	// 2. Delegate to general mathematical/CAS runtime
	mReq := MathRequest{
		Expression: prob.Expression,
		Operation:  OpEvaluate,
		Mode:       ModeExact,
		SessionID:  prob.SessionID,
		TenantID:   prob.TenantID,
	}
	mRes := e.mathRuntime.ProcessRequest(mReq)
	dur := float64(time.Since(start).Microseconds()) / 1000.0

	return CASResult{
		ProblemID:           prob.ID,
		Status:              mRes.Status,
		Domain:              DomainMathematics,
		CanonicalExpression: mRes.CanonicalExpression,
		ExactResult:         mRes.ExactResult,
		NumericResult:       mRes.NumericResult,
		BackendName:         mRes.BackendName,
		BackendVersion:      mRes.BackendVersion,
		ExecutionTimeMs:     dur,
		CompactObservation:  mRes.CompactObservation,
		VerificationState:   "VERIFIED",
		Provenance:          "UnifiedCASMathCore",
	}, nil
}

// VerifyPhysicalClaim deterministically checks a physical claim
func (e *UnifiedCASEngineImpl) VerifyPhysicalClaim(ctx context.Context, claimStr string) (MathVerificationResult, error) {
	// 1. Dimensional Inconsistency Check: e.g. "5 m + 3 s" or "velocity = distance + time"
	if dimErr := checkDimensionalAdditionError(claimStr); dimErr != "" {
		return MathVerificationResult{
			Matches:            false,
			DiscrepancyDetails: fmt.Sprintf("DIMENSION_ERROR: %s", dimErr),
			Confidence:         1.0,
		}, nil
	}

	// 2. Physical Equation Verification: e.g. "10 kg * 2 m/s^2 = 20 N" or "50 kg * 3 m/s^2 = 150 N"
	if parts := strings.Split(claimStr, "="); len(parts) == 2 {
		lhs := strings.TrimSpace(parts[0])
		rhs := strings.TrimSpace(parts[1])

		lhsQ, err1 := parseQuantityExpression(lhs)
		rhsQ, err2 := parseQuantityString(rhs)

		if err1 == nil && err2 == nil {
			// Check dimensions
			if lhsQ.Dimension != rhsQ.Dimension {
				return MathVerificationResult{
					Matches:            false,
					DiscrepancyDetails: fmt.Sprintf("DIMENSION_ERROR: LHS dimension %s != RHS dimension %s", lhsQ.Dimension.String(), rhsQ.Dimension.String()),
					Confidence:         1.0,
				}, nil
			}

			// Check values within tolerance
			fDiff := math.Abs(lhsQ.NumericValue - rhsQ.NumericValue)
			denom := math.Max(math.Abs(rhsQ.NumericValue), 1e-9)
			relErr := fDiff / denom
			if relErr < 1e-4 {
				return MathVerificationResult{
					Matches:           true,
					GroundTruthResult: rhsQ.String(),
					Confidence:        1.0,
				}, nil
			}

			return MathVerificationResult{
				Matches:            false,
				LLMClaimedResult:   rhsQ.String(),
				GroundTruthResult:  lhsQ.String(),
				RelativeError:      relErr,
				DiscrepancyDetails: fmt.Sprintf("Calculated %s != claimed %s", lhsQ.String(), rhsQ.String()),
				Confidence:         1.0,
			}, nil
		}
	}

	// 3. Fallback to Mathematical Verification
	mRes := e.mathRuntime.VerifyCalculation(claimStr, claimStr)
	return mRes, nil
}

// InterceptScientificIntent detects and solves physics and mathematical questions in natural text
func (e *UnifiedCASEngineImpl) InterceptScientificIntent(text string) *CASResult {
	if text == "" {
		return nil
	}

	trimmed := strings.TrimSpace(text)

	// 1. Check for Newton's 2nd Law question: e.g. "force on 50 kg object at 3 m/s^2" or "accelerate 10 kg at 2 m/s^2"
	forceRe := regexp.MustCompile(`(?i)(?:force|accelerat(?:e|ing)|experience)\s+(?:on\s+)?([0-9\.]+)\s*(?:kg|g)\s+(?:at|with)\s+([0-9\.]+)\s*(?:m\/s\^?2|m\/s2)`)
	if m := forceRe.FindStringSubmatch(trimmed); len(m) == 3 {
		massVal, _ := strconv.ParseFloat(m[1], 64)
		accVal, _ := strconv.ParseFloat(m[2], 64)
		fVal := massVal * accVal
		return &CASResult{
			Status:              StatusSuccess,
			Domain:              DomainMechanics,
			CanonicalExpression: fmt.Sprintf("F = %g kg * %g m/s^2", massVal, accVal),
			ExactResult:         fmt.Sprintf("%g N", fVal),
			NumericResult:       fmt.Sprintf("%g", fVal),
			Unit:                "N",
			Dimension:           ForceDimension.String(),
			FormulaUsed:         "Newton's Second Law (F = m*a)",
			BackendName:         "UnifiedCASPhysicsEngine",
			BackendVersion:      "1.0.0",
			CompactObservation:  fmt.Sprintf("[Physics Ground Truth: F = %g kg * %g m/s^2 = %g N]", massVal, accVal, fVal),
			VerificationState:   "VERIFIED",
			Provenance:          "Newton's Second Law of Motion",
		}
	}

	// 2. Check for Kinetic Energy: e.g. "kinetic energy of an 8 kg mass moving at 15 m/s"
	keRe := regexp.MustCompile(`(?i)(?:kinetic\s+energy)\s+(?:of\s+(?:an?\s+)?)?([0-9\.]+)\s*(?:kg|g)\s+(?:mass\s+)?(?:moving\s+at|at)\s+([0-9\.]+)\s*(?:m\/s)`)
	if m := keRe.FindStringSubmatch(trimmed); len(m) == 3 {
		massVal, _ := strconv.ParseFloat(m[1], 64)
		velVal, _ := strconv.ParseFloat(m[2], 64)
		keVal := 0.5 * massVal * velVal * velVal
		return &CASResult{
			Status:              StatusSuccess,
			Domain:              DomainMechanics,
			CanonicalExpression: fmt.Sprintf("KE = 0.5 * %g kg * (%g m/s)^2", massVal, velVal),
			ExactResult:         fmt.Sprintf("%g J", keVal),
			NumericResult:       fmt.Sprintf("%g", keVal),
			Unit:                "J",
			Dimension:           EnergyDimension.String(),
			FormulaUsed:         "Kinetic Energy (KE = 0.5 * m * v^2)",
			BackendName:         "UnifiedCASPhysicsEngine",
			BackendVersion:      "1.0.0",
			CompactObservation:  fmt.Sprintf("[Physics Ground Truth: KE = 0.5 * %g kg * (%g m/s)^2 = %g J]", massVal, velVal, keVal),
			VerificationState:   "VERIFIED",
			Provenance:          "Classical Mechanics",
		}
	}

	// 3. Check for Ohm's Law / Electric Power:
	// 3a. Voltage across R with I current: e.g. "voltage across 220 ohm with 4.5 A"
	voltRe := regexp.MustCompile(`(?i)(?:voltage|potential\s+difference)\s+(?:across\s+)?([0-9\.]+)\s*(?:kohm|kΩ|ohm|ohms|Ω)\s+(?:with|at|and)\s+([0-9\.]+)\s*(?:a|amp|amps|amperes)`)
	if m := voltRe.FindStringSubmatch(trimmed); len(m) == 3 {
		rVal, _ := strconv.ParseFloat(m[1], 64)
		iVal, _ := strconv.ParseFloat(m[2], 64)
		vVal := iVal * rVal
		return &CASResult{
			Status:              StatusSuccess,
			Domain:              DomainCircuits,
			CanonicalExpression: fmt.Sprintf("V = %g A * %g ohm", iVal, rVal),
			ExactResult:         fmt.Sprintf("%g V", vVal),
			NumericResult:       fmt.Sprintf("%g", vVal),
			Unit:                "V",
			Dimension:           "Voltage",
			FormulaUsed:         "Ohm's Law (V = I*R)",
			BackendName:         "UnifiedCASCircuitsEngine",
			BackendVersion:      "1.0.0",
			CompactObservation:  fmt.Sprintf("[Circuit Ground Truth: V = %g A * %g ohm = %g V]", iVal, rVal, vVal),
			VerificationState:   "VERIFIED",
			Provenance:          "Ohm's Law",
		}
	}

	// 3b. Current from V and R: e.g. "12 V and 4 ohm"
	ohmRe := regexp.MustCompile(`(?i)([0-9\.]+)\s*V(?:\s+and|\s*,|\s+through)\s+([0-9\.]+)\s*(?:ohm|ohms|Ω)`)
	if m := ohmRe.FindStringSubmatch(trimmed); len(m) == 3 {
		vVal, _ := strconv.ParseFloat(m[1], 64)
		rVal, _ := strconv.ParseFloat(m[2], 64)
		if rVal > 0 {
			iVal := vVal / rVal
			pVal := vVal * iVal
			return &CASResult{
				Status:              StatusSuccess,
				Domain:              DomainCircuits,
				CanonicalExpression: fmt.Sprintf("I = %g V / %g ohm", vVal, rVal),
				ExactResult:         fmt.Sprintf("I = %g A, P = %g W", iVal, pVal),
				NumericResult:       fmt.Sprintf("%g", iVal),
				Unit:                "A",
				Dimension:           CurrentDimension.String(),
				FormulaUsed:         "Ohm's Law (V = I*R, P = V*I)",
				BackendName:         "UnifiedCASCircuitsEngine",
				BackendVersion:      "1.0.0",
				CompactObservation:  fmt.Sprintf("[Circuit Ground Truth: V=%g V, R=%g ohm => I=%g A, P=%g W]", vVal, rVal, iVal, pVal),
				VerificationState:   "VERIFIED",
				Provenance:          "Ohm's Law & Joule Heating",
			}
		}
	}

	// 4. Check for RC time constant: e.g. "time constant of 10 kohm and 100 uF"
	rcRe := regexp.MustCompile(`(?i)(?:time\s+constant|tau)\s+(?:of\s+)?([0-9\.]+)\s*(?:kohm|kΩ|ohm|Ω)\s+(?:and\s+)?([0-9\.]+)\s*(?:uf|microfarad|nf|pf|f)`)
	if m := rcRe.FindStringSubmatch(trimmed); len(m) == 3 {
		rQ, _ := NewQuantity(mustParseFloat(m[1]), "kohm")
		cQ, _ := NewQuantity(mustParseFloat(m[2]), "uF")
		tauQ := rQ.Multiply(cQ)
		return &CASResult{
			Status:              StatusSuccess,
			Domain:              DomainCircuits,
			CanonicalExpression: "tau = R * C",
			ExactResult:         tauQ.String(),
			NumericResult:       fmt.Sprintf("%g", tauQ.NumericValue),
			Unit:                "s",
			Dimension:           TimeDimension.String(),
			FormulaUsed:         "RC Time Constant",
			BackendName:         "UnifiedCASCircuitsEngine",
			BackendVersion:      "1.0.0",
			CompactObservation:  fmt.Sprintf("[Circuit Ground Truth: tau = %s]", tauQ.String()),
			VerificationState:   "VERIFIED",
			Provenance:          "First-order RC Filter Theory",
		}
	}

	// 5. Check for Mass-Energy: e.g. "Using E = m*c^2 with mass = 2.5 kg" or "energy of 2 kg"
	emcRe := regexp.MustCompile(`(?i)(?:mass\s+energy|energy\s+of|e\s*=\s*m\*?c\^?2.*?)\b(?:mass\s*(?:=|is)?\s*)?([0-9\.]+)\s*(?:kg|g)`)
	if m := emcRe.FindStringSubmatch(trimmed); len(m) == 2 {
		mVal, _ := strconv.ParseFloat(m[1], 64)
		c := 299792458.0
		eVal := mVal * c * c
		return &CASResult{
			Status:              StatusSuccess,
			Domain:              DomainRelativity,
			CanonicalExpression: fmt.Sprintf("E = %g kg * c^2", mVal),
			ExactResult:         fmt.Sprintf("%g J", eVal),
			NumericResult:       fmt.Sprintf("%g", eVal),
			Unit:                "J",
			Dimension:           EnergyDimension.String(),
			FormulaUsed:         "Mass-Energy Equivalence (E = mc^2)",
			BackendName:         "UnifiedCASRelativityEngine",
			BackendVersion:      "1.0.0",
			CompactObservation:  fmt.Sprintf("[Relativity Ground Truth: E = %g kg * c^2 = %g J]", mVal, eVal),
			VerificationState:   "VERIFIED",
			Provenance:          "Albert Einstein (1905)",
		}
	}

	// 6. Check for Photon Energy: E = h * f, e.g. "photon energy E = h * f for light with frequency f = 6.0e14 Hz"
	photonRe := regexp.MustCompile(`(?i)(?:photon\s+energy|energy\s+of\s+photon|e\s*=\s*h\s*\*?\s*f).*?([0-9\.]+(?:e[+-]?[0-9]+)?)\s*(?:hz|khz|mhz|ghz|thz)`)
	if m := photonRe.FindStringSubmatch(trimmed); len(m) == 2 {
		fVal, _ := strconv.ParseFloat(m[1], 64)
		h := 6.62607015e-34
		eVal := h * fVal
		return &CASResult{
			Status:              StatusSuccess,
			Domain:              DomainQuantumMechanics,
			CanonicalExpression: fmt.Sprintf("E = %g * %g Hz", h, fVal),
			ExactResult:         fmt.Sprintf("%g J", eVal),
			NumericResult:       fmt.Sprintf("%g", eVal),
			Unit:                "J",
			Dimension:           EnergyDimension.String(),
			FormulaUsed:         "Planck-Einstein Relation (E = h*f)",
			BackendName:         "UnifiedCASQuantumEngine",
			BackendVersion:      "1.0.0",
			CompactObservation:  fmt.Sprintf("[Quantum Ground Truth: E = h * %g Hz = %g J]", fVal, eVal),
			VerificationState:   "VERIFIED",
			Provenance:          "Max Planck & Albert Einstein",
		}
	}

	// 7. Check for Carnot Efficiency: e.g. "carnot efficiency between 300 K and 600 K" or "carnot efficiency with Th = 600 K and Tc = 300 K"
	carnotRe := regexp.MustCompile(`(?i)(?:carnot\s+efficiency).*?([0-9\.]+)\s*k.*?([0-9\.]+)\s*k`)
	if m := carnotRe.FindStringSubmatch(trimmed); len(m) == 3 {
		t1, _ := strconv.ParseFloat(m[1], 64)
		t2, _ := strconv.ParseFloat(m[2], 64)
		tc := math.Min(t1, t2)
		th := math.Max(t1, t2)
		if th > 0 {
			eta := 1.0 - (tc / th)
			return &CASResult{
				Status:              StatusSuccess,
				Domain:              DomainThermodynamics,
				CanonicalExpression: fmt.Sprintf("eta = 1 - (%g K / %g K)", tc, th),
				ExactResult:         fmt.Sprintf("%g (%.1f%%)", eta, eta*100.0),
				NumericResult:       fmt.Sprintf("%g", eta),
				Dimension:           "Dimensionless",
				FormulaUsed:         "Carnot Cycle Efficiency (η = 1 - Tc/Th)",
				BackendName:         "UnifiedCASThermodynamicsEngine",
				BackendVersion:      "1.0.0",
				CompactObservation:  fmt.Sprintf("[Thermodynamics Ground Truth: Carnot Efficiency η = 1 - %g/%g = %g (%.1f%%)]", tc, th, eta, eta*100.0),
				VerificationState:   "VERIFIED",
				Provenance:          "Nicolas Léonard Sadi Carnot (1824)",
			}
		}
	}

	// 8. Check for Ideal Gas Law: PV = nRT (R = 8.314462618)
	gasRe := regexp.MustCompile(`(?i)(?:ideal\s+gas).*?([0-9\.]+)\s*mol.*?([0-9\.]+)\s*k.*?([0-9\.]+)\s*m\^?3`)
	if m := gasRe.FindStringSubmatch(trimmed); len(m) == 4 {
		nVal, _ := strconv.ParseFloat(m[1], 64)
		tVal, _ := strconv.ParseFloat(m[2], 64)
		vVal, _ := strconv.ParseFloat(m[3], 64)
		rConst := 8.314462618
		if vVal > 0 {
			pVal := (nVal * rConst * tVal) / vVal
			return &CASResult{
				Status:              StatusSuccess,
				Domain:              DomainThermodynamics,
				CanonicalExpression: fmt.Sprintf("P = (%g mol * %g * %g K) / %g m^3", nVal, rConst, tVal, vVal),
				ExactResult:         fmt.Sprintf("%g Pa", pVal),
				NumericResult:       fmt.Sprintf("%g", pVal),
				Unit:                "Pa",
				Dimension:           PressureDimension.String(),
				FormulaUsed:         "Ideal Gas Law (P = n*R*T / V)",
				BackendName:         "UnifiedCASThermodynamicsEngine",
				BackendVersion:      "1.0.0",
				CompactObservation:  fmt.Sprintf("[Thermodynamics Ground Truth: P = %g Pa]", pVal),
				VerificationState:   "VERIFIED",
				Provenance:          "Clapeyron Ideal Gas Law",
			}
		}
	}

	// 9. Check for Coulomb's Law: F = ke * q1 * q2 / r^2 (ke = 8.9875517923e9)
	coulombRe := regexp.MustCompile(`(?i)(?:coulomb\s+force|electrostatic\s+force).*?([0-9\.]+(?:e[+-]?[0-9]+)?)\s*c.*?([0-9\.]+(?:e[+-]?[0-9]+)?)\s*c.*?([0-9\.]+)\s*m`)
	if m := coulombRe.FindStringSubmatch(trimmed); len(m) == 4 {
		q1, _ := strconv.ParseFloat(m[1], 64)
		q2, _ := strconv.ParseFloat(m[2], 64)
		rVal, _ := strconv.ParseFloat(m[3], 64)
		ke := 8.9875517923e9
		if rVal > 0 {
			fVal := (ke * math.Abs(q1*q2)) / (rVal * rVal)
			return &CASResult{
				Status:              StatusSuccess,
				Domain:              DomainElectromagnetism,
				CanonicalExpression: fmt.Sprintf("F = (%g * %g C * %g C) / (%g m)^2", ke, q1, q2, rVal),
				ExactResult:         fmt.Sprintf("%g N", fVal),
				NumericResult:       fmt.Sprintf("%g", fVal),
				Unit:                "N",
				Dimension:           ForceDimension.String(),
				FormulaUsed:         "Coulomb's Law (F = ke * |q1*q2| / r^2)",
				BackendName:         "UnifiedCASEMEngine",
				BackendVersion:      "1.0.0",
				CompactObservation:  fmt.Sprintf("[Electromagnetism Ground Truth: F = %g N]", fVal),
				VerificationState:   "VERIFIED",
				Provenance:          "Charles-Augustin de Coulomb (1785)",
			}
		}
	}

	// 10. Check for LC Resonant Frequency: f0 = 1 / (2*pi*sqrt(L*C))
	lcRe := regexp.MustCompile(`(?i)(?:resonant\s+frequency).*?([0-9\.]+)\s*(?:mh|h|uh).*?([0-9\.]+)\s*(?:nf|uf|pf|f)`)
	if m := lcRe.FindStringSubmatch(trimmed); len(m) == 3 {
		lVal, _ := strconv.ParseFloat(m[1], 64)
		if strings.Contains(strings.ToLower(m[0]), "mh") {
			lVal *= 1e-3
		} else if strings.Contains(strings.ToLower(m[0]), "uh") {
			lVal *= 1e-6
		}
		cVal, _ := strconv.ParseFloat(m[2], 64)
		if strings.Contains(strings.ToLower(m[0]), "nf") {
			cVal *= 1e-9
		} else if strings.Contains(strings.ToLower(m[0]), "uf") {
			cVal *= 1e-6
		} else if strings.Contains(strings.ToLower(m[0]), "pf") {
			cVal *= 1e-12
		}
		if lVal > 0 && cVal > 0 {
			f0 := 1.0 / (2.0 * math.Pi * math.Sqrt(lVal*cVal))
			return &CASResult{
				Status:              StatusSuccess,
				Domain:              DomainCircuits,
				CanonicalExpression: fmt.Sprintf("f0 = 1 / (2*pi*sqrt(%g H * %g F))", lVal, cVal),
				ExactResult:         fmt.Sprintf("%g Hz", f0),
				NumericResult:       fmt.Sprintf("%g", f0),
				Unit:                "Hz",
				Dimension:           FrequencyDimension.String(),
				FormulaUsed:         "LC Resonant Frequency (f0 = 1 / (2*π*√(L*C)))",
				BackendName:         "UnifiedCASCircuitsEngine",
				BackendVersion:      "1.0.0",
				CompactObservation:  fmt.Sprintf("[Circuit Ground Truth: Resonant Frequency f0 = %g Hz]", f0),
				VerificationState:   "VERIFIED",
				Provenance:          "Thomson LC Resonance Formula",
			}
		}
	}

	// 11. Check for Snell's Law: n1*sin(th1) = n2*sin(th2) -> sin(th2) = (n1/n2)*sin(th1)
	snellRe := regexp.MustCompile(`(?i)(?:snell|refraction).*?n1\s*=\s*([0-9\.]+).*?theta1\s*=\s*([0-9\.]+)\s*(?:deg|°).*?n2\s*=\s*([0-9\.]+)`)
	if m := snellRe.FindStringSubmatch(trimmed); len(m) == 4 {
		n1, _ := strconv.ParseFloat(m[1], 64)
		deg1, _ := strconv.ParseFloat(m[2], 64)
		n2, _ := strconv.ParseFloat(m[3], 64)
		rad1 := deg1 * math.Pi / 180.0
		sin2 := (n1 / n2) * math.Sin(rad1)
		if sin2 >= -1.0 && sin2 <= 1.0 {
			deg2 := math.Asin(sin2) * 180.0 / math.Pi
			return &CASResult{
				Status:              StatusSuccess,
				Domain:              DomainOptics,
				CanonicalExpression: fmt.Sprintf("theta2 = asin((%g/%g)*sin(%g deg))", n1, n2, deg1),
				ExactResult:         fmt.Sprintf("%.2f deg", deg2),
				NumericResult:       fmt.Sprintf("%.4f", deg2),
				Unit:                "deg",
				Dimension:           "Angle",
				FormulaUsed:         "Snell's Law of Refraction (n1*sin(θ1) = n2*sin(θ2))",
				BackendName:         "UnifiedCASOpticsEngine",
				BackendVersion:      "1.0.0",
				CompactObservation:  fmt.Sprintf("[Optics Ground Truth: Refraction Angle θ2 = %.2f deg]", deg2),
				VerificationState:   "VERIFIED",
				Provenance:          "Willebrord Snellius (1621)",
			}
		}
	}

	// 12. Delegate to mathematical/symbolic intent interception
	if mRes := e.mathRuntime.InterceptAndVerifyIntent(text); mRes != nil {
		return &CASResult{
			Status:              mRes.Status,
			Domain:              DomainMathematics,
			CanonicalExpression: mRes.CanonicalExpression,
			ExactResult:         mRes.ExactResult,
			NumericResult:       mRes.NumericResult,
			BackendName:         mRes.BackendName,
			BackendVersion:      mRes.BackendVersion,
			CompactObservation:  mRes.CompactObservation,
			VerificationState:   "VERIFIED",
			Provenance:          "UnifiedCASMathCore",
		}
	}

	return nil
}

func mustParseFloat(s string) float64 {
	v, _ := strconv.ParseFloat(strings.TrimSpace(s), 64)
	return v
}

func parseQuantityString(s string) (PhysicalQuantity, error) {
	s = strings.TrimSpace(s)
	numRe := regexp.MustCompile(`^([-+]?[0-9\.]+)\s*([a-zA-Z\^\/_0-9]*)$`)
	m := numRe.FindStringSubmatch(s)
	if len(m) == 3 {
		val, err := strconv.ParseFloat(m[1], 64)
		if err != nil {
			return PhysicalQuantity{}, err
		}
		return NewQuantity(val, m[2])
	}
	return PhysicalQuantity{}, fmt.Errorf("cannot parse quantity %q", s)
}

func parseQuantityExpression(expr string) (PhysicalQuantity, error) {
	expr = strings.TrimSpace(expr)
	// Try single quantity first
	if q, err := parseQuantityString(expr); err == nil {
		return q, nil
	}

	// Product of two quantities: e.g. "10 kg * 2 m/s^2"
	if parts := strings.Split(expr, "*"); len(parts) == 2 {
		q1, err1 := parseQuantityString(parts[0])
		q2, err2 := parseQuantityString(parts[1])
		if err1 == nil && err2 == nil {
			return q1.Multiply(q2), nil
		}
	}

	// Division of two quantities: e.g. "100 J / 5 s"
	if parts := strings.Split(expr, "/"); len(parts) == 2 {
		q1, err1 := parseQuantityString(parts[0])
		q2, err2 := parseQuantityString(parts[1])
		if err1 == nil && err2 == nil {
			return q1.Divide(q2)
		}
	}

	return PhysicalQuantity{}, fmt.Errorf("unsupported quantity expression: %s", expr)
}

func checkDimensionalAdditionError(expr string) string {
	// Detect invalid additions like "5 m + 3 s"
	addRe := regexp.MustCompile(`([0-9\.]+)\s*([a-zA-Z]+)\s*\+\s*([0-9\.]+)\s*([a-zA-Z]+)`)
	if m := addRe.FindStringSubmatch(expr); len(m) == 5 {
		q1, err1 := NewQuantity(mustParseFloat(m[1]), m[2])
		q2, err2 := NewQuantity(mustParseFloat(m[3]), m[4])
		if err1 == nil && err2 == nil && q1.Dimension != q2.Dimension {
			return fmt.Sprintf("Cannot add %s (%s) and %s (%s): incompatible physical dimensions",
				m[2], q1.Dimension.String(), m[4], q2.Dimension.String())
		}
	}
	return ""
}

func formatPhysicalResult(val float64, unit string) string {
	if unit == "" || unit == "1" {
		return fmt.Sprintf("%g", val)
	}
	return fmt.Sprintf("%g %s", val, unit)
}
