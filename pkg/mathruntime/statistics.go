package mathruntime

import (
	"fmt"
	"math"
	"math/big"
	"sort"
	"strconv"
	"strings"
)

// DescriptiveStats holds summary statistics of a dataset.
type DescriptiveStats struct {
	Count    int     `json:"count"`
	Sum      float64 `json:"sum"`
	Mean     float64 `json:"mean"`
	Median   float64 `json:"median"`
	Variance float64 `json:"variance"`
	StdDev   float64 `json:"std_dev"`
	Min      float64 `json:"min"`
	Max      float64 `json:"max"`
}

// ComputeStats calculates descriptive statistics for a slice of numbers.
func ComputeStats(data []float64) (DescriptiveStats, error) {
	n := len(data)
	if n == 0 {
		return DescriptiveStats{}, fmt.Errorf("dataset is empty")
	}

	sum := 0.0
	minVal := data[0]
	maxVal := data[0]

	sorted := make([]float64, n)
	copy(sorted, data)
	sort.Float64s(sorted)

	for _, v := range data {
		sum += v
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}

	mean := sum / float64(n)

	// Median
	var median float64
	if n%2 == 1 {
		median = sorted[n/2]
	} else {
		median = (sorted[n/2-1] + sorted[n/2]) / 2.0
	}

	// Variance & StdDev (sample variance if n > 1, else 0)
	var variance float64
	if n > 1 {
		sqSum := 0.0
		for _, v := range data {
			diff := v - mean
			sqSum += diff * diff
		}
		variance = sqSum / float64(n-1)
	}
	stdDev := math.Sqrt(variance)

	return DescriptiveStats{
		Count:    n,
		Sum:      sum,
		Mean:     mean,
		Median:   median,
		Variance: variance,
		StdDev:   stdDev,
		Min:      minVal,
		Max:      maxVal,
	}, nil
}

// ParseNumbers parses comma/space-separated numbers e.g. "[10, 20, 30, 40, 50]"
func ParseNumbers(s string) ([]float64, error) {
	s = strings.Trim(s, "[](){}")
	parts := strings.FieldsFunc(s, func(r rune) bool {
		return r == ',' || r == ' ' || r == '\t'
	})

	var nums []float64
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		v, err := strconv.ParseFloat(p, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid number '%s': %w", p, err)
		}
		nums = append(nums, v)
	}
	return nums, nil
}

// ModPow computes (base^exp) mod modulus efficiently using big.Int. The modulus must be positive and the exponent not
// negative: big.Int.Exp reads a zero modulus as "no modulus" and would compute base^exp in full.
func ModPow(base, exp, mod int64) (int64, error) {
	if mod <= 0 {
		return 0, fmt.Errorf("modulus must be a positive integer, got %d", mod)
	}
	if exp < 0 {
		return 0, fmt.Errorf("exponent must not be negative, got %d", exp)
	}
	res := new(big.Int).Exp(big.NewInt(base), big.NewInt(exp), big.NewInt(mod))
	return res.Int64(), nil
}

// ModInverse computes x such that (a * x) % m == 1 using Extended Euclidean Algorithm.
func ModInverse(a, m int64) (int64, error) {
	if m <= 0 {
		return 0, fmt.Errorf("modulus must be a positive integer, got %d", m)
	}
	g, x, _ := extendedGCD(a, m)
	if g != 1 {
		return 0, fmt.Errorf("modular inverse does not exist (gcd(%d, %d) = %d != 1)", a, m, g)
	}
	res := (x%m + m) % m
	return res, nil
}

func extendedGCD(a, b int64) (int64, int64, int64) {
	if a == 0 {
		return b, 0, 1
	}
	g, x1, y1 := extendedGCD(b%a, a)
	x := y1 - (b/a)*x1
	y := x1
	return g, x, y
}
