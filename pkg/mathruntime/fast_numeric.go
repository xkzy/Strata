package mathruntime

import (
	"fmt"
	"math"
	"math/big"
	"regexp"
	"strconv"
	"strings"
	"time"
)

type FastNumericBackend struct {
	Name    string
	Version string
}

func NewFastNumericBackend() *FastNumericBackend {
	return &FastNumericBackend{
		Name:    "FastNumericNative",
		Version: "1.0.0",
	}
}

func (b *FastNumericBackend) SupportsOperation(op MathOperation, mode MathMode) bool {
	switch op {
	case OpEvaluate, OpNumericEvaluate, OpDeterminant, OpProbability,
		OpMatrixInverse, OpMatrixMultiply, OpMatrixTrace, OpEigenvalues, OpLinearSystem,
		OpStatistics, OpModInverse, OpModPow, OpDifferentiate, OpIntegrate, OpLimit, OpSolve:
		return true
	case OpSimplify, OpFactor, OpExpand:
		return true
	default:
		return false
	}
}

func (b *FastNumericBackend) Execute(req MathRequest) MathResult {
	startTime := time.Now()
	res := MathResult{
		RequestID:           req.RequestID,
		BackendName:         b.Name,
		BackendVersion:      b.Version,
		CanonicalExpression: req.Expression,
	}

	parser := NewExpressionParser()
	cleanExpr := parser.Canonicalize(req.Expression)

	switch req.Operation {
	case OpDifferentiate:
		v := req.Variable
		if v == "" {
			v = "x"
		}
		res.ExactResult = DifferentiateSymbolic(cleanExpr, v)
		res.NumericResult = res.ExactResult
		res.RawResult = res.ExactResult
		res.Status = StatusSuccess

	case OpIntegrate:
		v := req.Variable
		if v == "" {
			v = "x"
		}
		res.ExactResult = IntegrateSymbolic(cleanExpr, v)
		res.NumericResult = res.ExactResult
		res.RawResult = res.ExactResult
		res.Status = StatusSuccess

	case OpLimit:
		v := req.Variable
		if v == "" {
			v = "x"
		}
		res.ExactResult = ComputeLimit(cleanExpr, v, req.Point)
		res.NumericResult = res.ExactResult
		res.RawResult = res.ExactResult
		res.Status = StatusSuccess

	case OpSolve:
		if quad, err := ParseAndSolveQuadratic(cleanExpr); err == nil {
			res.ExactResult = quad.ExactStr
			res.NumericResult = quad.ExactStr
			res.RawResult = quad.ExactStr
			res.Status = StatusSuccess
		} else {
			res.Status = StatusExecutionError
			res.ErrorMessage = err.Error()
		}

	case OpMatrixInverse:
		mat, err := ParseMatrix(cleanExpr)
		if err == nil {
			inv, err := mat.Inverse()
			if err == nil {
				res.ExactResult = inv.String()
				res.NumericResult = inv.String()
				res.RawResult = inv.String()
				res.Status = StatusSuccess
			} else {
				res.Status = StatusExecutionError
				res.ErrorMessage = err.Error()
			}
		} else {
			res.Status = StatusExecutionError
			res.ErrorMessage = err.Error()
		}

	case OpMatrixMultiply:
		parts := strings.Split(cleanExpr, "*")
		if len(parts) == 2 {
			mA, errA := ParseMatrix(strings.TrimSpace(parts[0]))
			mB, errB := ParseMatrix(strings.TrimSpace(parts[1]))
			if errA == nil && errB == nil {
				prod, err := mA.Multiply(mB)
				if err == nil {
					res.ExactResult = prod.String()
					res.NumericResult = prod.String()
					res.RawResult = prod.String()
					res.Status = StatusSuccess
				} else {
					res.Status = StatusExecutionError
					res.ErrorMessage = err.Error()
				}
			} else {
				res.Status = StatusExecutionError
				res.ErrorMessage = "failed to parse input matrices"
			}
		} else {
			res.Status = StatusExecutionError
			res.ErrorMessage = "matrix multiplication requires two matrices separated by *"
		}

	case OpMatrixTrace:
		mat, err := ParseMatrix(cleanExpr)
		if err == nil {
			tr, err := mat.Trace()
			if err == nil {
				res.ExactResult = fmt.Sprintf("%g", tr)
				res.NumericResult = res.ExactResult
				res.RawResult = res.ExactResult
				res.Status = StatusSuccess
			} else {
				res.Status = StatusExecutionError
				res.ErrorMessage = err.Error()
			}
		} else {
			res.Status = StatusExecutionError
			res.ErrorMessage = err.Error()
		}

	case OpEigenvalues:
		mat, err := ParseMatrix(cleanExpr)
		if err == nil {
			l1, l2, err := mat.Eigenvalues2x2()
			if err == nil {
				res.ExactResult = fmt.Sprintf("λ1 = %g, λ2 = %g", l1, l2)
				res.NumericResult = res.ExactResult
				res.RawResult = res.ExactResult
				res.Status = StatusSuccess
			} else {
				res.Status = StatusExecutionError
				res.ErrorMessage = err.Error()
			}
		} else {
			res.Status = StatusExecutionError
			res.ErrorMessage = err.Error()
		}

	case OpLinearSystem:
		parts := strings.Split(cleanExpr, "=")
		if len(parts) == 2 {
			mA, errA := ParseMatrix(strings.TrimSpace(parts[0]))
			bVec, errB := ParseNumbers(strings.TrimSpace(parts[1]))
			if errA == nil && errB == nil {
				x, err := SolveLinearSystem(mA, bVec)
				if err == nil {
					var xStrs []string
					for _, v := range x {
						xStrs = append(xStrs, fmt.Sprintf("%g", v))
					}
					res.ExactResult = "[" + strings.Join(xStrs, ", ") + "]"
					res.NumericResult = res.ExactResult
					res.RawResult = res.ExactResult
					res.Status = StatusSuccess
				} else {
					res.Status = StatusExecutionError
					res.ErrorMessage = err.Error()
				}
			} else {
				res.Status = StatusExecutionError
				res.ErrorMessage = "failed to parse matrix A or vector b"
			}
		} else {
			res.Status = StatusExecutionError
			res.ErrorMessage = "linear system requires format A = b"
		}

	case OpStatistics:
		nums, err := ParseNumbers(cleanExpr)
		if err == nil {
			stats, err := ComputeStats(nums)
			if err == nil {
				switch strings.ToLower(req.Variable) {
				case "mean", "average":
					res.ExactResult = fmt.Sprintf("%g", stats.Mean)
				case "median":
					res.ExactResult = fmt.Sprintf("%g", stats.Median)
				case "stddev", "standard deviation", "standard_deviation":
					res.ExactResult = fmt.Sprintf("%g", stats.StdDev)
				case "variance":
					res.ExactResult = fmt.Sprintf("%g", stats.Variance)
				default:
					res.ExactResult = fmt.Sprintf("mean=%g, median=%g, stddev=%g", stats.Mean, stats.Median, stats.StdDev)
				}
				res.NumericResult = res.ExactResult
				res.RawResult = res.ExactResult
				res.Status = StatusSuccess
			} else {
				res.Status = StatusExecutionError
				res.ErrorMessage = err.Error()
			}
		} else {
			res.Status = StatusExecutionError
			res.ErrorMessage = err.Error()
		}

	case OpModInverse:
		parts := strings.Split(cleanExpr, "mod")
		if len(parts) == 2 {
			a, errA := strconv.ParseInt(strings.TrimSpace(parts[0]), 10, 64)
			m, errM := strconv.ParseInt(strings.TrimSpace(parts[1]), 10, 64)
			var inv int64
			err := errA
			if err == nil {
				err = errM
			}
			if err != nil {
				err = fmt.Errorf("mod_inverse needs two integers that fit in 64 bits: %v", err)
			} else {
				inv, err = ModInverse(a, m)
			}
			if err == nil {
				res.ExactResult = fmt.Sprintf("%d", inv)
				res.NumericResult = res.ExactResult
				res.RawResult = res.ExactResult
				res.Status = StatusSuccess
			} else {
				res.Status = StatusExecutionError
				res.ErrorMessage = err.Error()
			}
		} else {
			res.Status = StatusExecutionError
			res.ErrorMessage = "invalid mod_inverse format"
		}

	case OpModPow:
		// e.g. "a^b mod m"
		powParts := strings.Split(cleanExpr, "^")
		if len(powParts) == 2 {
			modParts := strings.Split(powParts[1], "mod")
			if len(modParts) == 2 {
				a, errA := strconv.ParseInt(strings.TrimSpace(powParts[0]), 10, 64)
				bExp, errE := strconv.ParseInt(strings.TrimSpace(modParts[0]), 10, 64)
				m, errM := strconv.ParseInt(strings.TrimSpace(modParts[1]), 10, 64)
				for _, e := range []error{errA, errE, errM} {
					if e != nil && res.ErrorMessage == "" {
						res.ErrorMessage = fmt.Sprintf("mod_pow needs three integers that fit in 64 bits: %v", e)
					}
				}
				if res.ErrorMessage == "" {
					val, err := ModPow(a, bExp, m)
					if err != nil {
						res.ErrorMessage = err.Error()
					} else {
						res.ExactResult = fmt.Sprintf("%d", val)
						res.NumericResult = res.ExactResult
						res.RawResult = res.ExactResult
						res.Status = StatusSuccess
					}
				}
			}
		}
		if res.Status != StatusSuccess {
			res.Status = StatusExecutionError
			if res.ErrorMessage == "" {
				res.ErrorMessage = "invalid mod_pow format: use a^b mod m"
			}
		}

	case OpDeterminant:
		// 2x2 determinant: [[a, b], [c, d]]
		re := regexp.MustCompile(`\[\[\s*([-+]?[0-9\.]+)\s*,\s*([-+]?[0-9\.]+)\s*\]\s*,\s*\[\s*([-+]?[0-9\.]+)\s*,\s*([-+]?[0-9\.]+)\s*\]\]`)
		if m := re.FindStringSubmatch(cleanExpr); len(m) == 5 {
			a, _ := strconv.ParseFloat(m[1], 64)
			bVal, _ := strconv.ParseFloat(m[2], 64)
			c, _ := strconv.ParseFloat(m[3], 64)
			d, _ := strconv.ParseFloat(m[4], 64)
			det := a*d - bVal*c
			if math.Floor(det) == det {
				res.ExactResult = fmt.Sprintf("%.0f", det)
			} else {
				res.ExactResult = fmt.Sprintf("%f", det)
			}
			res.NumericResult = res.ExactResult
			res.RawResult = res.ExactResult
			res.Status = StatusSuccess
		} else {
			// 3x3 determinant: [[a,b,c],[d,e,f],[g,h,i]]
			re3 := regexp.MustCompile(`\[\[\s*([-+]?[0-9\.]+)\s*,\s*([-+]?[0-9\.]+)\s*,\s*([-+]?[0-9\.]+)\s*\]\s*,\s*\[\s*([-+]?[0-9\.]+)\s*,\s*([-+]?[0-9\.]+)\s*,\s*([-+]?[0-9\.]+)\s*\]\s*,\s*\[\s*([-+]?[0-9\.]+)\s*,\s*([-+]?[0-9\.]+)\s*,\s*([-+]?[0-9\.]+)\s*\]\]`)
			if m3 := re3.FindStringSubmatch(cleanExpr); len(m3) == 10 {
				a, _ := strconv.ParseFloat(m3[1], 64)
				bVal, _ := strconv.ParseFloat(m3[2], 64)
				c, _ := strconv.ParseFloat(m3[3], 64)
				d, _ := strconv.ParseFloat(m3[4], 64)
				e, _ := strconv.ParseFloat(m3[5], 64)
				f, _ := strconv.ParseFloat(m3[6], 64)
				g, _ := strconv.ParseFloat(m3[7], 64)
				h, _ := strconv.ParseFloat(m3[8], 64)
				iVal, _ := strconv.ParseFloat(m3[9], 64)
				det := a*(e*iVal-f*h) - bVal*(d*iVal-f*g) + c*(d*h-e*g)
				if math.Floor(det) == det {
					res.ExactResult = fmt.Sprintf("%.0f", det)
				} else {
					res.ExactResult = fmt.Sprintf("%f", det)
				}
				res.NumericResult = res.ExactResult
				res.RawResult = res.ExactResult
				res.Status = StatusSuccess
			} else {
				res.Status = StatusExecutionError
				res.ErrorMessage = "invalid matrix format for determinant"
			}
		}

	case OpProbability:
		// Factorial or Combinations/Permutations
		if strings.HasSuffix(cleanExpr, "!") {
			nStr := strings.TrimSuffix(cleanExpr, "!")
			n, err := strconv.ParseInt(nStr, 10, 64)
			if err == nil && n >= 0 && n <= 100 {
				val := big.NewInt(1)
				for i := int64(2); i <= n; i++ {
					val.Mul(val, big.NewInt(i))
				}
				res.ExactResult = val.String()
				res.NumericResult = val.String()
				res.RawResult = val.String()
				res.Status = StatusSuccess
			} else {
				res.Status = StatusExecutionError
				res.ErrorMessage = "factorial parameter out of range"
			}
		} else if strings.HasPrefix(cleanExpr, "C(") || strings.HasPrefix(cleanExpr, "comb(") {
			re := regexp.MustCompile(`\d+`)
			nums := re.FindAllString(cleanExpr, 2)
			if len(nums) == 2 {
				n, errN := strconv.ParseInt(nums[0], 10, 64)
				k, errK := strconv.ParseInt(nums[1], 10, 64)
				switch {
				case errN != nil || errK != nil || n > maxBinomialN:
					res.Status = StatusExecutionError
					res.ErrorMessage = fmt.Sprintf("C(n, k) is limited to n <= %d (the result has up to n bits)", maxBinomialN)
				default:
					val := big.NewInt(0).Binomial(n, k)
					res.ExactResult = val.String()
					res.NumericResult = val.String()
					res.RawResult = val.String()
					res.Status = StatusSuccess
				}
			}
		} else {
			res.Status = StatusExecutionError
			res.ErrorMessage = "unsupported probability expression"
		}

	default:
		if strings.HasPrefix(strings.ToUpper(cleanExpr), "GCD(") {
			re := regexp.MustCompile(`\d+`)
			nums := re.FindAllString(cleanExpr, 2)
			if len(nums) == 2 {
				aBig, _ := new(big.Int).SetString(nums[0], 10)
				bBig, _ := new(big.Int).SetString(nums[1], 10)
				g := new(big.Int).GCD(nil, nil, aBig, bBig)
				res.ExactResult = g.String()
				res.NumericResult = g.String()
				res.RawResult = g.String()
				res.Status = StatusSuccess
			}
		} else if strings.HasPrefix(strings.ToUpper(cleanExpr), "LCM(") {
			re := regexp.MustCompile(`\d+`)
			nums := re.FindAllString(cleanExpr, 2)
			if len(nums) == 2 {
				aBig, _ := new(big.Int).SetString(nums[0], 10)
				bBig, _ := new(big.Int).SetString(nums[1], 10)
				g := new(big.Int).GCD(nil, nil, aBig, bBig)
				l := new(big.Int) // lcm(0, 0) = 0: gcd(0, 0) = 0 must not divide
				if g.Sign() != 0 {
					l.Quo(new(big.Int).Mul(aBig, bBig), g)
				}
				res.ExactResult = l.String()
				res.NumericResult = l.String()
				res.RawResult = l.String()
				res.Status = StatusSuccess
			}
		} else if strings.HasPrefix(cleanExpr, "sqrt(") && strings.HasSuffix(cleanExpr, ")") {
			inner := strings.TrimSuffix(strings.TrimPrefix(cleanExpr, "sqrt("), ")")
			val, err := strconv.ParseFloat(inner, 64)
			if err == nil && val >= 0 {
				s := math.Sqrt(val)
				if math.Floor(s) == s && math.Floor(val) == val {
					res.ExactResult = fmt.Sprintf("%.0f", s)
				} else {
					res.ExactResult = fmt.Sprintf("sqrt(%s)", inner)
				}
				res.NumericResult = fmt.Sprintf("%f", s)
				if req.Mode == ModeNumeric {
					res.RawResult = res.NumericResult
				} else {
					res.RawResult = res.ExactResult
				}
				res.Status = StatusSuccess
			} else {
				res.Status = StatusExecutionError
				res.ErrorMessage = "invalid sqrt operand"
			}
		} else {
			// Exact rational evaluation
			rat, dbl, err := b.evalExactRational(cleanExpr)
			if err == nil {
				res.Status = StatusSuccess
				if rat.IsInt() {
					res.ExactResult = rat.Num().String()
				} else {
					res.ExactResult = rat.RatString()
				}
				res.NumericResult = fmt.Sprintf("%.*g", min(max(req.PrecisionDigits, 1), maxPrecisionDigits), dbl)
				if req.Mode == ModeNumeric {
					res.RawResult = res.NumericResult
				} else {
					res.RawResult = res.ExactResult
				}
			} else {
				res.Status = StatusExecutionError
				res.ErrorMessage = err.Error()
			}
		}
	}

	res.ExecutionTimeMs = float64(time.Since(startTime).Microseconds()) / 1000.0
	res.CompactObservation = parser.FormatCompactObservation(req.Expression, res.ExactResult, res.NumericResult, req.Operation)
	return res
}

func getPrecedence(op rune) int {
	switch op {
	case '+', '-':
		return 1
	case '*', '/', '%':
		return 2
	case '^':
		return 3
	}
	return 0
}

func (b *FastNumericBackend) evalExactRational(expr string) (*big.Rat, float64, error) {
	clean := strings.ReplaceAll(expr, " ", "")
	var values []*big.Rat
	var ops []rune

	applyOp := func(op rune) error {
		if len(values) < 2 {
			return fmt.Errorf("invalid expression")
		}
		val2 := values[len(values)-1]
		val1 := values[len(values)-2]
		values = values[:len(values)-2]

		res := new(big.Rat)
		switch op {
		case '+':
			res.Add(val1, val2)
		case '-':
			res.Sub(val1, val2)
		case '*':
			res.Mul(val1, val2)
		case '/':
			if val2.Sign() == 0 {
				return fmt.Errorf("division by zero")
			}
			res.Quo(val1, val2)
		case '^':
			if val2.IsInt() && val2.Num().IsInt64() && val2.Sign() >= 0 {
				// exact, when the result stays small: each ^n multiplies the size by n, and a few nested ones would
				// otherwise ask for gigabytes
				p := val2.Num().Int64()
				if p > maxRationalBits || int64(ratBits(val1))*p > maxRationalBits {
					return fmt.Errorf("result too large (more than %d bits)", maxRationalBits)
				}
				num := new(big.Int).Exp(val1.Num(), big.NewInt(p), nil)
				denom := new(big.Int).Exp(val1.Denom(), big.NewInt(p), nil)
				res.SetFrac(num, denom)
			} else {
				f1, _ := val1.Float64()
				f2, _ := val2.Float64()
				fRes := math.Pow(f1, f2)
				if math.IsNaN(fRes) || math.IsInf(fRes, 0) {
					return fmt.Errorf("result is not a finite number")
				}
				res.SetFloat64(fRes)
			}
		}
		if ratBits(res) > maxRationalBits {
			return fmt.Errorf("result too large (more than %d bits)", maxRationalBits)
		}
		values = append(values, res)
		return nil
	}

	for i := 0; i < len(clean); i++ {
		c := rune(clean[i])
		if c == '(' {
			ops = append(ops, c)
		} else if c >= '0' && c <= '9' {
			j := i
			for j < len(clean) && ((clean[j] >= '0' && clean[j] <= '9') || clean[j] == '.') {
				j++
			}
			numStr := clean[i:j]
			rat := new(big.Rat)
			if _, ok := rat.SetString(numStr); !ok {
				return nil, 0, fmt.Errorf("invalid number '%s'", numStr)
			}
			values = append(values, rat)
			i = j - 1
		} else if c == ')' {
			for len(ops) > 0 && ops[len(ops)-1] != '(' {
				if err := applyOp(ops[len(ops)-1]); err != nil {
					return nil, 0, err
				}
				ops = ops[:len(ops)-1]
			}
			if len(ops) > 0 {
				ops = ops[:len(ops)-1] // pop '('
			}
		} else if c == '+' || c == '-' || c == '*' || c == '/' || c == '^' {
			if c == '-' && (i == 0 || clean[i-1] == '(' || clean[i-1] == '+' || clean[i-1] == '-' || clean[i-1] == '*' || clean[i-1] == '/') {
				values = append(values, big.NewRat(0, 1))
			}
			for len(ops) > 0 && getPrecedence(ops[len(ops)-1]) >= getPrecedence(c) {
				if err := applyOp(ops[len(ops)-1]); err != nil {
					return nil, 0, err
				}
				ops = ops[:len(ops)-1]
			}
			ops = append(ops, c)
		}
	}

	for len(ops) > 0 {
		if err := applyOp(ops[len(ops)-1]); err != nil {
			return nil, 0, err
		}
		ops = ops[:len(ops)-1]
	}

	if len(values) == 0 {
		return nil, 0, fmt.Errorf("no computed value")
	}

	finalRat := values[len(values)-1]
	f, _ := finalRat.Float64()
	return finalRat, f, nil
}
