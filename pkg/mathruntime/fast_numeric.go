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
	case OpEvaluate, OpNumericEvaluate, OpDeterminant, OpProbability:
		return true
	case OpSimplify, OpFactor, OpExpand:
		return mode != ModeSymbolic
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
	case OpDeterminant:
		// 2x2 determinant: [[a,b],[c,d]]
		re := regexp.MustCompile(`\[\[([0-9\.\-]+),([0-9\.\-]+)\],\[([0-9\.\-]+),([0-9\.\-]+)\]\]`)
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
			res.Status = StatusExecutionError
			res.ErrorMessage = "invalid 2x2 matrix format for FastNumeric determinant"
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
				n, _ := strconv.ParseInt(nums[0], 10, 64)
				k, _ := strconv.ParseInt(nums[1], 10, 64)
				val := big.NewInt(0).Binomial(n, k)
				res.ExactResult = val.String()
				res.NumericResult = val.String()
				res.RawResult = val.String()
				res.Status = StatusSuccess
			}
		} else {
			res.Status = StatusExecutionError
			res.ErrorMessage = "unsupported probability expression"
		}

	default:
		if strings.HasPrefix(cleanExpr, "sqrt(") && strings.HasSuffix(cleanExpr, ")") {
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
				res.NumericResult = fmt.Sprintf("%.*g", req.PrecisionDigits, dbl)
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
			if !val2.IsInt() {
				f1, _ := val1.Float64()
				f2, _ := val2.Float64()
				fRes := math.Pow(f1, f2)
				res.SetFloat64(fRes)
			} else {
				p := val2.Num().Int64()
				if p >= 0 && p <= 100 {
					num := new(big.Int).Exp(val1.Num(), big.NewInt(p), nil)
					denom := new(big.Int).Exp(val1.Denom(), big.NewInt(p), nil)
					res.SetFrac(num, denom)
				} else {
					f1, _ := val1.Float64()
					f2, _ := val2.Float64()
					res.SetFloat64(math.Pow(f1, f2))
				}
			}
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
