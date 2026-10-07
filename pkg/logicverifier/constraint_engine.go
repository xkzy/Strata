// pkg/logicverifier/constraint_engine.go - Deterministic 1D Interval Constraint Solver
package logicverifier

import (
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
)

type Interval struct {
	Min           float64
	Max           float64
	MinInclusive  bool
	MaxInclusive  bool
	IsEmpty       bool
	ExcludedVals  []float64
}

func NewUnboundedInterval() Interval {
	return Interval{
		Min:          math.Inf(-1),
		Max:          math.Inf(1),
		MinInclusive: false,
		MaxInclusive: false,
		IsEmpty:      false,
		ExcludedVals: make([]float64, 0),
	}
}

func (i *Interval) Intersect(other Interval) {
	if i.IsEmpty || other.IsEmpty {
		i.IsEmpty = true
		return
	}

	// Update Min
	if other.Min > i.Min {
		i.Min = other.Min
		i.MinInclusive = other.MinInclusive
	} else if other.Min == i.Min {
		i.MinInclusive = i.MinInclusive && other.MinInclusive
	}

	// Update Max
	if other.Max < i.Max {
		i.Max = other.Max
		i.MaxInclusive = other.MaxInclusive
	} else if other.Max == i.Max {
		i.MaxInclusive = i.MaxInclusive && other.MaxInclusive
	}

	for _, v := range other.ExcludedVals {
		i.ExcludedVals = append(i.ExcludedVals, v)
	}

	// Check emptiness
	if i.Min > i.Max {
		i.IsEmpty = true
	} else if i.Min == i.Max && (!i.MinInclusive || !i.MaxInclusive) {
		i.IsEmpty = true
	}
}

func (i *Interval) Contains(val float64) bool {
	if i.IsEmpty {
		return false
	}
	if val < i.Min || (val == i.Min && !i.MinInclusive) {
		return false
	}
	if val > i.Max || (val == i.Max && !i.MaxInclusive) {
		return false
	}
	for _, excl := range i.ExcludedVals {
		if math.Abs(val-excl) < 1e-12 {
			return false
		}
	}
	return true
}

type ConstraintEngine struct{}

func NewConstraintEngine() *ConstraintEngine {
	return &ConstraintEngine{}
}

// Regex patterns for constraint extraction: e.g. "x > 5", "x <= 10", "x == 7", "x != 3", "5 < x <= 10"
var (
	singleConstraintRe = regexp.MustCompile(`^([a-zA-Z_][a-zA-Z0-9_]*)\s*(<=|>=|<|>|==|=|!=)\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)$`)
	reversedConstraintRe = regexp.MustCompile(`^([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*(<=|>=|<|>)\s*([a-zA-Z_][a-zA-Z0-9_]*)$`)
	assignmentRe       = regexp.MustCompile(`^([a-zA-Z_][a-zA-Z0-9_]*)\s*(=|==)\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)$`)
)

func (ce *ConstraintEngine) Verify(claim VerificationClaim) VerificationResult {
	res := VerificationResult{
		ClaimID:     claim.ClaimID,
		Type:        ClaimConstraint,
		BackendUsed: "deterministic_interval_solver",
		Status:      StatusUnknown,
	}

	domain := make(map[string]Interval)

	parseSingle := func(cStr string) (string, Interval, error) {
		cStr = strings.TrimSpace(cStr)
		if m := singleConstraintRe.FindStringSubmatch(cStr); len(m) == 4 {
			varName := m[1]
			op := m[2]
			val, err := strconv.ParseFloat(m[3], 64)
			if err != nil {
				return "", Interval{}, err
			}
			iv := NewUnboundedInterval()
			switch op {
			case ">":
				iv.Min = val
				iv.MinInclusive = false
			case ">=":
				iv.Min = val
				iv.MinInclusive = true
			case "<":
				iv.Max = val
				iv.MaxInclusive = false
			case "<=":
				iv.Max = val
				iv.MaxInclusive = true
			case "==", "=":
				iv.Min = val
				iv.Max = val
				iv.MinInclusive = true
				iv.MaxInclusive = true
			case "!=":
				iv.ExcludedVals = append(iv.ExcludedVals, val)
			}
			return varName, iv, nil
		}

		if m := reversedConstraintRe.FindStringSubmatch(cStr); len(m) == 4 {
			val, err := strconv.ParseFloat(m[1], 64)
			if err != nil {
				return "", Interval{}, err
			}
			op := m[2]
			varName := m[3]
			iv := NewUnboundedInterval()
			// val < x  =>  x > val
			switch op {
			case "<":
				iv.Min = val
				iv.MinInclusive = false
			case "<=":
				iv.Min = val
				iv.MinInclusive = true
			case ">":
				iv.Max = val
				iv.MaxInclusive = false
			case ">=":
				iv.Max = val
				iv.MaxInclusive = true
			}
			return varName, iv, nil
		}

		return "", Interval{}, fmt.Errorf("unsupported constraint format: %s", cStr)
	}

	// Add all constraints
	for _, c := range claim.Constraints {
		varName, iv, err := parseSingle(c)
		if err != nil {
			// Try splitting on comma or semicolon
			subParts := strings.FieldsFunc(c, func(r rune) bool {
				return r == ',' || r == ';'
			})
			if len(subParts) > 1 {
				for _, sp := range subParts {
					vn, siv, err2 := parseSingle(sp)
					if err2 != nil {
						res.Status = StatusUnknown
						res.Evidence = fmt.Sprintf("failed to parse constraint: %s", sp)
						return res
					}
					curr, exists := domain[vn]
					if !exists {
						curr = NewUnboundedInterval()
					}
					curr.Intersect(siv)
					domain[vn] = curr
				}
				continue
			}
			res.Status = StatusUnknown
			res.Evidence = fmt.Sprintf("unsupported constraint syntax: %s", c)
			return res
		}

		curr, exists := domain[varName]
		if !exists {
			curr = NewUnboundedInterval()
		}
		curr.Intersect(iv)
		domain[varName] = curr
	}

	// Check if constraint system is inherently unsatisfiable (contradiction in premises)
	for varName, iv := range domain {
		if iv.IsEmpty {
			res.Status = StatusFail
			res.FailureReason = fmt.Sprintf("constraint system has empty feasible region for '%s'", varName)
			res.Evidence = "constraints are mutually contradictory (unsatisfiable)"
			res.CompactObservation = fmt.Sprintf("Verification: FAIL | Contradictory constraints on %s", varName)
			return res
		}
	}

	// Now check claimed value or assignment
	claimTarget := claim.ClaimedValue
	if claimTarget == "" {
		claimTarget = claim.Expression
	}
	claimTarget = strings.TrimSpace(claimTarget)

	if m := assignmentRe.FindStringSubmatch(claimTarget); len(m) == 4 {
		varName := m[1]
		val, err := strconv.ParseFloat(m[3], 64)
		if err != nil {
			res.Status = StatusUnknown
			res.Evidence = "unable to parse numeric value from claim"
			return res
		}

		iv, exists := domain[varName]
		if !exists {
			res.Status = StatusUnknown
			res.Evidence = fmt.Sprintf("variable '%s' has no constraining bounds defined", varName)
			return res
		}

		if iv.Contains(val) {
			res.Status = StatusPass
			res.ActualValue = fmt.Sprintf("%g", val)
			res.Evidence = fmt.Sprintf("value %g is within feasible interval [%g, %g]", val, iv.Min, iv.Max)
			res.CompactObservation = fmt.Sprintf("Verification: PASS | %s = %g in bounds", varName, val)
		} else {
			res.Status = StatusFail
			res.ActualValue = fmt.Sprintf("%g", val)
			res.FailureReason = fmt.Sprintf("value %g violates constraints on '%s' (bounds: [%g, %g])", val, varName, iv.Min, iv.Max)
			res.CompactObservation = fmt.Sprintf("Verification: FAIL | %s = %g outside [%g, %g]", varName, val, iv.Min, iv.Max)
		}
		return res
	}

	// If the claim itself is another constraint: e.g. "x > 15"
	varName, claimIv, err := parseSingle(claimTarget)
	if err == nil {
		iv, exists := domain[varName]
		if !exists {
			res.Status = StatusUnknown
			res.Evidence = fmt.Sprintf("no constraints defined for variable '%s'", varName)
			return res
		}
		// If claimIv is satisfied by all points in iv
		if iv.Min >= claimIv.Min && iv.Max <= claimIv.Max {
			res.Status = StatusPass
			res.Evidence = fmt.Sprintf("interval [%g, %g] is a subset of claimed range [%g, %g]", iv.Min, iv.Max, claimIv.Min, claimIv.Max)
			res.CompactObservation = fmt.Sprintf("Verification: PASS | %s satisfies bounds", varName)
		} else if (claimIv.Min > iv.Max) || (claimIv.Max < iv.Min) {
			res.Status = StatusFail
			res.FailureReason = fmt.Sprintf("claimed range [%g, %g] is completely disjoint from feasible region [%g, %g]", claimIv.Min, claimIv.Max, iv.Min, iv.Max)
			res.CompactObservation = fmt.Sprintf("Verification: FAIL | %s disjoint from bounds", varName)
		} else {
			res.Status = StatusFail
			res.FailureReason = fmt.Sprintf("feasible region [%g, %g] contains values violating claimed range [%g, %g]", iv.Min, iv.Max, claimIv.Min, claimIv.Max)
			res.CompactObservation = fmt.Sprintf("Verification: FAIL | %s exceeds bounds", varName)
		}
		return res
	}

	res.Status = StatusUnknown
	res.Evidence = "could not deterministically parse constraint claim format"
	return res
}
