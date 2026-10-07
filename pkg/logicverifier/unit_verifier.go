// pkg/logicverifier/unit_verifier.go - SI Dimensional Analysis & Homogeneity Verifier
package logicverifier

import (
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
)

// SIDimensions represents the exponents of the 7 SI base units:
// [L] Length (meter)
// [M] Mass (kilogram)
// [T] Time (second)
// [I] Electric Current (ampere)
// [Theta] Temperature (kelvin)
// [N] Amount of Substance (mole)
// [J] Luminous Intensity (candela)
type SIDimensions struct {
	L     int `json:"L"`
	M     int `json:"M"`
	T     int `json:"T"`
	I     int `json:"I"`
	Theta int `json:"Theta"`
	N     int `json:"N"`
	J     int `json:"J"`
}

func (d SIDimensions) Equal(other SIDimensions) bool {
	return d.L == other.L &&
		d.M == other.M &&
		d.T == other.T &&
		d.I == other.I &&
		d.Theta == other.Theta &&
		d.N == other.N &&
		d.J == other.J
}

func (d SIDimensions) String() string {
	var parts []string
	if d.L != 0 {
		parts = append(parts, fmt.Sprintf("m^%d", d.L))
	}
	if d.M != 0 {
		parts = append(parts, fmt.Sprintf("kg^%d", d.M))
	}
	if d.T != 0 {
		parts = append(parts, fmt.Sprintf("s^%d", d.T))
	}
	if d.I != 0 {
		parts = append(parts, fmt.Sprintf("A^%d", d.I))
	}
	if d.Theta != 0 {
		parts = append(parts, fmt.Sprintf("K^%d", d.Theta))
	}
	if d.N != 0 {
		parts = append(parts, fmt.Sprintf("mol^%d", d.N))
	}
	if d.J != 0 {
		parts = append(parts, fmt.Sprintf("cd^%d", d.J))
	}
	if len(parts) == 0 {
		return "dimensionless"
	}
	return strings.Join(parts, "*")
}

type PhysicalQuantity struct {
	Value      float64
	ScaleToSI  float64
	Dimensions SIDimensions
}

type UnitVerifier struct {
	baseUnits map[string]PhysicalQuantity
	prefixes  map[string]float64
}

func NewUnitVerifier() *UnitVerifier {
	uv := &UnitVerifier{
		baseUnits: make(map[string]PhysicalQuantity),
		prefixes:  make(map[string]float64),
	}

	// SI Base Units
	uv.baseUnits["m"] = PhysicalQuantity{Value: 1.0, ScaleToSI: 1.0, Dimensions: SIDimensions{L: 1}}
	uv.baseUnits["meter"] = uv.baseUnits["m"]
	uv.baseUnits["meters"] = uv.baseUnits["m"]
	uv.baseUnits["metre"] = uv.baseUnits["m"]
	uv.baseUnits["metres"] = uv.baseUnits["m"]

	uv.baseUnits["g"] = PhysicalQuantity{Value: 1.0, ScaleToSI: 1e-3, Dimensions: SIDimensions{M: 1}}
	uv.baseUnits["gram"] = uv.baseUnits["g"]
	uv.baseUnits["grams"] = uv.baseUnits["g"]
	uv.baseUnits["kg"] = PhysicalQuantity{Value: 1.0, ScaleToSI: 1.0, Dimensions: SIDimensions{M: 1}}
	uv.baseUnits["kilogram"] = uv.baseUnits["kg"]
	uv.baseUnits["kilograms"] = uv.baseUnits["kg"]

	uv.baseUnits["s"] = PhysicalQuantity{Value: 1.0, ScaleToSI: 1.0, Dimensions: SIDimensions{T: 1}}
	uv.baseUnits["sec"] = uv.baseUnits["s"]
	uv.baseUnits["second"] = uv.baseUnits["s"]
	uv.baseUnits["seconds"] = uv.baseUnits["s"]
	uv.baseUnits["min"] = PhysicalQuantity{Value: 1.0, ScaleToSI: 60.0, Dimensions: SIDimensions{T: 1}}
	uv.baseUnits["minute"] = uv.baseUnits["min"]
	uv.baseUnits["minutes"] = uv.baseUnits["min"]
	uv.baseUnits["h"] = PhysicalQuantity{Value: 1.0, ScaleToSI: 3600.0, Dimensions: SIDimensions{T: 1}}
	uv.baseUnits["hr"] = uv.baseUnits["h"]
	uv.baseUnits["hour"] = uv.baseUnits["h"]
	uv.baseUnits["hours"] = uv.baseUnits["h"]

	uv.baseUnits["A"] = PhysicalQuantity{Value: 1.0, ScaleToSI: 1.0, Dimensions: SIDimensions{I: 1}}
	uv.baseUnits["amp"] = uv.baseUnits["A"]
	uv.baseUnits["ampere"] = uv.baseUnits["A"]

	uv.baseUnits["K"] = PhysicalQuantity{Value: 1.0, ScaleToSI: 1.0, Dimensions: SIDimensions{Theta: 1}}
	uv.baseUnits["kelvin"] = uv.baseUnits["K"]

	uv.baseUnits["mol"] = PhysicalQuantity{Value: 1.0, ScaleToSI: 1.0, Dimensions: SIDimensions{N: 1}}
	uv.baseUnits["mole"] = uv.baseUnits["mol"]

	uv.baseUnits["cd"] = PhysicalQuantity{Value: 1.0, ScaleToSI: 1.0, Dimensions: SIDimensions{J: 1}}
	uv.baseUnits["candela"] = uv.baseUnits["cd"]

	// Common non-SI length
	uv.baseUnits["km"] = PhysicalQuantity{Value: 1.0, ScaleToSI: 1000.0, Dimensions: SIDimensions{L: 1}}
	uv.baseUnits["cm"] = PhysicalQuantity{Value: 1.0, ScaleToSI: 0.01, Dimensions: SIDimensions{L: 1}}
	uv.baseUnits["mm"] = PhysicalQuantity{Value: 1.0, ScaleToSI: 0.001, Dimensions: SIDimensions{L: 1}}
	uv.baseUnits["in"] = PhysicalQuantity{Value: 1.0, ScaleToSI: 0.0254, Dimensions: SIDimensions{L: 1}}
	uv.baseUnits["inch"] = uv.baseUnits["in"]
	uv.baseUnits["ft"] = PhysicalQuantity{Value: 1.0, ScaleToSI: 0.3048, Dimensions: SIDimensions{L: 1}}
	uv.baseUnits["foot"] = uv.baseUnits["ft"]
	uv.baseUnits["feet"] = uv.baseUnits["ft"]
	uv.baseUnits["mi"] = PhysicalQuantity{Value: 1.0, ScaleToSI: 1609.344, Dimensions: SIDimensions{L: 1}}
	uv.baseUnits["mile"] = uv.baseUnits["mi"]
	uv.baseUnits["miles"] = uv.baseUnits["mi"]

	// Derived Units
	// N = kg*m/s^2
	uv.baseUnits["N"] = PhysicalQuantity{Value: 1.0, ScaleToSI: 1.0, Dimensions: SIDimensions{L: 1, M: 1, T: -2}}
	uv.baseUnits["newton"] = uv.baseUnits["N"]
	// J = N*m = kg*m^2/s^2
	uv.baseUnits["J"] = PhysicalQuantity{Value: 1.0, ScaleToSI: 1.0, Dimensions: SIDimensions{L: 2, M: 1, T: -2}}
	uv.baseUnits["joule"] = uv.baseUnits["J"]
	// W = J/s = kg*m^2/s^3
	uv.baseUnits["W"] = PhysicalQuantity{Value: 1.0, ScaleToSI: 1.0, Dimensions: SIDimensions{L: 2, M: 1, T: -3}}
	uv.baseUnits["watt"] = uv.baseUnits["W"]
	// Pa = N/m^2 = kg/(m*s^2)
	uv.baseUnits["Pa"] = PhysicalQuantity{Value: 1.0, ScaleToSI: 1.0, Dimensions: SIDimensions{L: -1, M: 1, T: -2}}
	uv.baseUnits["pascal"] = uv.baseUnits["Pa"]
	// Hz = 1/s
	uv.baseUnits["Hz"] = PhysicalQuantity{Value: 1.0, ScaleToSI: 1.0, Dimensions: SIDimensions{T: -1}}
	uv.baseUnits["hertz"] = uv.baseUnits["Hz"]
	// C = A*s
	uv.baseUnits["C"] = PhysicalQuantity{Value: 1.0, ScaleToSI: 1.0, Dimensions: SIDimensions{T: 1, I: 1}}
	uv.baseUnits["coulomb"] = uv.baseUnits["C"]
	// V = W/A = kg*m^2/(A*s^3)
	uv.baseUnits["V"] = PhysicalQuantity{Value: 1.0, ScaleToSI: 1.0, Dimensions: SIDimensions{L: 2, M: 1, T: -3, I: -1}}
	uv.baseUnits["volt"] = uv.baseUnits["V"]

	return uv
}

func (uv *UnitVerifier) ParseExpression(expr string) (PhysicalQuantity, error) {
	expr = strings.TrimSpace(expr)

	// Check for division of two quantities: e.g. "10 m / 2 s"
	if parts := strings.Split(expr, "/"); len(parts) == 2 {
		q1, err1 := uv.ParseExpression(parts[0])
		q2, err2 := uv.ParseExpression(parts[1])
		if err1 == nil && err2 == nil && q2.Value != 0 {
			resDim := SIDimensions{
				L:     q1.Dimensions.L - q2.Dimensions.L,
				M:     q1.Dimensions.M - q2.Dimensions.M,
				T:     q1.Dimensions.T - q2.Dimensions.T,
				I:     q1.Dimensions.I - q2.Dimensions.I,
				Theta: q1.Dimensions.Theta - q2.Dimensions.Theta,
				N:     q1.Dimensions.N - q2.Dimensions.N,
				J:     q1.Dimensions.J - q2.Dimensions.J,
			}
			return PhysicalQuantity{
				Value:      q1.Value / q2.Value,
				ScaleToSI:  q1.ScaleToSI / q2.ScaleToSI,
				Dimensions: resDim,
			}, nil
		}
	}

	// Check for multiplication of two quantities: e.g. "5 kg * 2 m"
	if parts := strings.Split(expr, "*"); len(parts) == 2 {
		q1, err1 := uv.ParseExpression(parts[0])
		q2, err2 := uv.ParseExpression(parts[1])
		if err1 == nil && err2 == nil {
			resDim := SIDimensions{
				L:     q1.Dimensions.L + q2.Dimensions.L,
				M:     q1.Dimensions.M + q2.Dimensions.M,
				T:     q1.Dimensions.T + q2.Dimensions.T,
				I:     q1.Dimensions.I + q2.Dimensions.I,
				Theta: q1.Dimensions.Theta + q2.Dimensions.Theta,
				N:     q1.Dimensions.N + q2.Dimensions.N,
				J:     q1.Dimensions.J + q2.Dimensions.J,
			}
			return PhysicalQuantity{
				Value:      q1.Value * q2.Value,
				ScaleToSI:  q1.ScaleToSI * q2.ScaleToSI,
				Dimensions: resDim,
			}, nil
		}
	}

	return uv.ParseQuantity(expr)
}

var quantityParserRe = regexp.MustCompile(`^\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*(.*)$`)

func (uv *UnitVerifier) ParseQuantity(str string) (PhysicalQuantity, error) {
	str = strings.TrimSpace(str)
	m := quantityParserRe.FindStringSubmatch(str)
	if len(m) < 3 {
		return PhysicalQuantity{}, fmt.Errorf("cannot parse quantity string: %s", str)
	}

	val, err := strconv.ParseFloat(m[1], 64)
	if err != nil {
		return PhysicalQuantity{}, err
	}

	unitStr := strings.TrimSpace(m[2])
	if unitStr == "" {
		return PhysicalQuantity{Value: val, ScaleToSI: 1.0, Dimensions: SIDimensions{}}, nil
	}

	dim, scale, err := uv.parseUnitExpression(unitStr)
	if err != nil {
		return PhysicalQuantity{}, err
	}

	return PhysicalQuantity{
		Value:      val,
		ScaleToSI:  scale,
		Dimensions: dim,
	}, nil
}

func (uv *UnitVerifier) parseUnitExpression(uStr string) (SIDimensions, float64, error) {
	uStr = strings.TrimSpace(uStr)
	// Check for simple slash "m/s", "km/h"
	if parts := strings.Split(uStr, "/"); len(parts) == 2 {
		numDim, numScale, err1 := uv.parseProduct(parts[0])
		if err1 != nil {
			return SIDimensions{}, 0, err1
		}
		denDim, denScale, err2 := uv.parseProduct(parts[1])
		if err2 != nil {
			return SIDimensions{}, 0, err2
		}

		resDim := SIDimensions{
			L:     numDim.L - denDim.L,
			M:     numDim.M - denDim.M,
			T:     numDim.T - denDim.T,
			I:     numDim.I - denDim.I,
			Theta: numDim.Theta - denDim.Theta,
			N:     numDim.N - denDim.N,
			J:     numDim.J - denDim.J,
		}
		return resDim, numScale / denScale, nil
	}

	return uv.parseProduct(uStr)
}

func (uv *UnitVerifier) parseProduct(uStr string) (SIDimensions, float64, error) {
	uStr = strings.TrimSpace(uStr)
	tokens := strings.FieldsFunc(uStr, func(r rune) bool {
		return r == '*' || r == ' ' || r == '·'
	})

	totalDim := SIDimensions{}
	totalScale := 1.0

	for _, token := range tokens {
		token = strings.TrimSpace(token)
		if token == "" {
			continue
		}

		// Check for power: e.g. "m^2", "s^2"
		power := 1
		unitName := token
		if pIdx := strings.Index(token, "^"); pIdx != -1 {
			unitName = token[:pIdx]
			powStr := token[pIdx+1:]
			p, err := strconv.Atoi(powStr)
			if err != nil {
				return SIDimensions{}, 0, fmt.Errorf("invalid exponent in unit: %s", token)
			}
			power = p
		}

		bu, ok := uv.baseUnits[unitName]
		if !ok {
			return SIDimensions{}, 0, fmt.Errorf("unknown unit: %s", unitName)
		}

		totalDim.L += bu.Dimensions.L * power
		totalDim.M += bu.Dimensions.M * power
		totalDim.T += bu.Dimensions.T * power
		totalDim.I += bu.Dimensions.I * power
		totalDim.Theta += bu.Dimensions.Theta * power
		totalDim.N += bu.Dimensions.N * power
		totalDim.J += bu.Dimensions.J * power

		totalScale *= math.Pow(bu.ScaleToSI, float64(power))
	}

	return totalDim, totalScale, nil
}

func (uv *UnitVerifier) Verify(claim VerificationClaim) VerificationResult {
	res := VerificationResult{
		ClaimID:     claim.ClaimID,
		Type:        ClaimUnitDimension,
		BackendUsed: "deterministic_si_unit_verifier",
		Status:      StatusUnknown,
	}

	exprStr := claim.Expression
	if exprStr == "" {
		exprStr = claim.UnitExpression
	}
	targetStr := claim.ClaimedValue
	if targetStr == "" {
		targetStr = claim.ClaimedUnit
	}

	q1, err1 := uv.ParseExpression(exprStr)
	q2, err2 := uv.ParseExpression(targetStr)

	if err1 != nil || err2 != nil {
		res.Status = StatusUnknown
		res.Evidence = fmt.Sprintf("failed to parse physical quantities (expr: %v, target: %v)", err1, err2)
		return res
	}

	// 1. Dimensional Homogeneity Check
	if !q1.Dimensions.Equal(q2.Dimensions) {
		res.Status = StatusFail
		res.FailureReason = fmt.Sprintf("dimensional mismatch: expression has dimensions [%s] while target has [%s]",
			q1.Dimensions.String(), q2.Dimensions.String())
		res.Evidence = "quantities are dimensionally incompatible (cannot equate or convert)"
		res.CompactObservation = fmt.Sprintf("Verification: FAIL | Dimensional mismatch [%s] != [%s]",
			q1.Dimensions.String(), q2.Dimensions.String())
		return res
	}

	// 2. Numerical Homogeneity Check (if values are provided)
	siVal1 := q1.Value * q1.ScaleToSI
	siVal2 := q2.Value * q2.ScaleToSI

	tol := claim.Tolerance.AbsTol
	if tol == 0 {
		tol = 1e-4
	}

	relTol := claim.Tolerance.RelTol
	if relTol == 0 {
		relTol = 1e-4
	}

	diff := math.Abs(siVal1 - siVal2)
	maxMag := math.Max(math.Abs(siVal1), math.Abs(siVal2))
	if maxMag == 0 {
		maxMag = 1.0
	}

	if diff <= tol || (diff/maxMag) <= relTol {
		res.Status = StatusPass
		res.ActualValue = fmt.Sprintf("%g (SI: %g)", q2.Value, siVal2)
		res.ExpectedValue = fmt.Sprintf("%g (SI: %g)", q1.Value, siVal1)
		res.Evidence = fmt.Sprintf("dimensions match [%s] and SI values are equivalent (%g SI == %g SI)",
			q1.Dimensions.String(), siVal1, siVal2)
		res.CompactObservation = fmt.Sprintf("Verification: PASS | Unit & value match (%s)", q1.Dimensions.String())
	} else {
		res.Status = StatusFail
		res.ActualValue = fmt.Sprintf("%g (SI: %g)", q2.Value, siVal2)
		res.ExpectedValue = fmt.Sprintf("%g (SI: %g)", q1.Value, siVal1)
		res.FailureReason = fmt.Sprintf("conversion discrepancy: %g in base SI != %g in base SI", siVal1, siVal2)
		res.Evidence = fmt.Sprintf("dimensions match [%s] but numerical values do not match after unit conversion", q1.Dimensions.String())
		res.CompactObservation = fmt.Sprintf("Verification: FAIL | Unit value discrepancy (%g SI != %g SI)", siVal1, siVal2)
	}

	return res
}
