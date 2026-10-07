package mathruntime

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// Matrix represents a 2D matrix of float64 numbers.
type Matrix struct {
	Rows int
	Cols int
	Data [][]float64
}

// NewMatrix creates a new Matrix.
func NewMatrix(rows, cols int) *Matrix {
	data := make([][]float64, rows)
	for i := range data {
		data[i] = make([]float64, cols)
	}
	return &Matrix{Rows: rows, Cols: cols, Data: data}
}

// ParseMatrix parses string representation e.g. "[[1, 2], [3, 4]]"
func ParseMatrix(s string) (*Matrix, error) {
	s = strings.TrimSpace(s)
	if !strings.HasPrefix(s, "[[") || !strings.HasSuffix(s, "]]") {
		return nil, fmt.Errorf("invalid matrix format, expected [[...], [...]]")
	}

	inner := s[2 : len(s)-2]
	rowStrs := strings.Split(inner, "],")
	var rows [][]float64

	for _, rStr := range rowStrs {
		rStr = strings.TrimPrefix(strings.TrimSpace(rStr), "[")
		rStr = strings.TrimSuffix(strings.TrimSpace(rStr), "]")
		elemStrs := strings.Split(rStr, ",")
		var row []float64
		for _, eStr := range elemStrs {
			eStr = strings.TrimSpace(eStr)
			if eStr == "" {
				continue
			}
			val, err := strconv.ParseFloat(eStr, 64)
			if err != nil {
				return nil, fmt.Errorf("failed to parse element '%s': %w", eStr, err)
			}
			row = append(row, val)
		}
		if len(row) > 0 {
			rows = append(rows, row)
		}
	}

	if len(rows) == 0 {
		return nil, fmt.Errorf("empty matrix")
	}

	cols := len(rows[0])
	for i, r := range rows {
		if len(r) != cols {
			return nil, fmt.Errorf("inconsistent row length at row %d", i)
		}
	}

	return &Matrix{Rows: len(rows), Cols: cols, Data: rows}, nil
}

func (m *Matrix) String() string {
	var sb strings.Builder
	sb.WriteString("[")
	for i, r := range m.Data {
		if i > 0 {
			sb.WriteString(", ")
		}
		sb.WriteString("[")
		for j, v := range r {
			if j > 0 {
				sb.WriteString(", ")
			}
			sb.WriteString(fmt.Sprintf("%g", v))
		}
		sb.WriteString("]")
	}
	sb.WriteString("]")
	return sb.String()
}

// Multiply computes matrix product A * B.
func (a *Matrix) Multiply(b *Matrix) (*Matrix, error) {
	if a.Cols != b.Rows {
		return nil, fmt.Errorf("cannot multiply %dx%d with %dx%d", a.Rows, a.Cols, b.Rows, b.Cols)
	}
	res := NewMatrix(a.Rows, b.Cols)
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < b.Cols; j++ {
			sum := 0.0
			for k := 0; k < a.Cols; k++ {
				sum += a.Data[i][k] * b.Data[k][j]
			}
			res.Data[i][j] = sum
		}
	}
	return res, nil
}

// Transpose returns the transpose of matrix M.
func (m *Matrix) Transpose() *Matrix {
	res := NewMatrix(m.Cols, m.Rows)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			res.Data[j][i] = m.Data[i][j]
		}
	}
	return res
}

// Trace returns the trace (sum of diagonal elements) for square matrices.
func (m *Matrix) Trace() (float64, error) {
	if m.Rows != m.Cols {
		return 0, fmt.Errorf("trace requires a square matrix")
	}
	sum := 0.0
	for i := 0; i < m.Rows; i++ {
		sum += m.Data[i][i]
	}
	return sum, nil
}

// Determinant computes the determinant for 1x1, 2x2, 3x3 matrices.
func (m *Matrix) Determinant() (float64, error) {
	if m.Rows != m.Cols {
		return 0, fmt.Errorf("determinant requires a square matrix")
	}
	if m.Rows == 1 {
		return m.Data[0][0], nil
	}
	if m.Rows == 2 {
		return m.Data[0][0]*m.Data[1][1] - m.Data[0][1]*m.Data[1][0], nil
	}
	if m.Rows == 3 {
		a, b, c := m.Data[0][0], m.Data[0][1], m.Data[0][2]
		d, e, f := m.Data[1][0], m.Data[1][1], m.Data[1][2]
		g, h, i := m.Data[2][0], m.Data[2][1], m.Data[2][2]
		return a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g), nil
	}
	return 0, fmt.Errorf("higher order determinant not yet implemented")
}

// Inverse computes the inverse of a 2x2 or 3x3 matrix.
func (m *Matrix) Inverse() (*Matrix, error) {
	det, err := m.Determinant()
	if err != nil {
		return nil, err
	}
	if math.Abs(det) < 1e-12 {
		return nil, fmt.Errorf("matrix is singular (det=0), inverse does not exist")
	}

	if m.Rows == 2 {
		inv := NewMatrix(2, 2)
		inv.Data[0][0] = m.Data[1][1] / det
		inv.Data[0][1] = -m.Data[0][1] / det
		inv.Data[1][0] = -m.Data[1][0] / det
		inv.Data[1][1] = m.Data[0][0] / det
		return inv, nil
	}

	if m.Rows == 3 {
		a, b, c := m.Data[0][0], m.Data[0][1], m.Data[0][2]
		d, e, f := m.Data[1][0], m.Data[1][1], m.Data[1][2]
		g, h, i := m.Data[2][0], m.Data[2][1], m.Data[2][2]

		inv := NewMatrix(3, 3)
		inv.Data[0][0] = (e*i - f*h) / det
		inv.Data[0][1] = -(b*i - c*h) / det
		inv.Data[0][2] = (b*f - c*e) / det

		inv.Data[1][0] = -(d*i - f*g) / det
		inv.Data[1][1] = (a*i - c*g) / det
		inv.Data[1][2] = -(a*f - c*d) / det

		inv.Data[2][0] = (d*h - e*g) / det
		inv.Data[2][1] = -(a*h - b*g) / det
		inv.Data[2][2] = (a*e - b*d) / det
		return inv, nil
	}

	return nil, fmt.Errorf("inversion for dimension %d not implemented", m.Rows)
}

// Eigenvalues2x2 computes eigenvalues of a 2x2 matrix: det(A - lambda*I) = 0
func (m *Matrix) Eigenvalues2x2() (float64, float64, error) {
	if m.Rows != 2 || m.Cols != 2 {
		return 0, 0, fmt.Errorf("eigenvalues2x2 requires a 2x2 matrix")
	}
	tr := m.Data[0][0] + m.Data[1][1]
	det, _ := m.Determinant()
	disc := tr*tr - 4*det
	if disc < 0 {
		return 0, 0, fmt.Errorf("complex eigenvalues")
	}
	l1 := (tr + math.Sqrt(disc)) / 2.0
	l2 := (tr - math.Sqrt(disc)) / 2.0
	return l1, l2, nil
}

// SolveLinearSystem solves Ax = b using Gaussian elimination with partial pivoting.
func SolveLinearSystem(A *Matrix, b []float64) ([]float64, error) {
	if A.Rows != A.Cols {
		return nil, fmt.Errorf("system matrix A must be square")
	}
	n := A.Rows
	if len(b) != n {
		return nil, fmt.Errorf("dimension mismatch: A is %dx%d but b has length %d", n, n, len(b))
	}

	// Augment matrix
	aug := make([][]float64, n)
	for i := range aug {
		aug[i] = make([]float64, n+1)
		copy(aug[i], A.Data[i])
		aug[i][n] = b[i]
	}

	// Forward elimination
	for p := 0; p < n; p++ {
		// Find pivot
		maxRow := p
		maxVal := math.Abs(aug[p][p])
		for i := p + 1; i < n; i++ {
			if math.Abs(aug[i][p]) > maxVal {
				maxVal = math.Abs(aug[i][p])
				maxRow = i
			}
		}
		if maxVal < 1e-12 {
			return nil, fmt.Errorf("system is singular or has infinite solutions")
		}

		// Swap rows
		aug[p], aug[maxRow] = aug[maxRow], aug[p]

		// Eliminate below
		for i := p + 1; i < n; i++ {
			factor := aug[i][p] / aug[p][p]
			for j := p; j <= n; j++ {
				aug[i][j] -= factor * aug[p][j]
			}
		}
	}

	// Back substitution
	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		sum := 0.0
		for j := i + 1; j < n; j++ {
			sum += aug[i][j] * x[j]
		}
		x[i] = (aug[i][n] - sum) / aug[i][i]
	}

	return x, nil
}
