//go:build cuda

package densecpu

import (
	"math"
	"math/rand"
	"testing"
)

// Every CUDA kernel must agree with the CPU kernel on random matrices, on every card, for 1 and several inputs.
func TestCUDAKernelsMatchCPU(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for dev := 0; dev < 2; dev++ {
		for _, c := range []struct {
			typ   uint32
			f16at []int
		}{{tQ4_K, []int{0, 2}}, {tQ5_K, []int{0, 2}}, {tQ6_K, []int{208}}, {tQ8_0, []int{0}}, {tBF16, nil}} {
			elems, bytes, _ := blockInfo(c.typ)
			rows, cols := 37, 1024
			w := make([]byte, rows*cols/elems*bytes)
			rng.Read(w)
			for b := 0; b < rows*cols/elems; b++ {
				for _, o := range c.f16at {
					w[b*bytes+o], w[b*bytes+o+1] = byte(rng.Intn(256)), 0x2c+byte(rng.Intn(8))
				}
				if c.typ == tBF16 {
					w[b*2+1] &= 0x3f
				}
			}
			for _, n := range []int{1, 5} {
				X := make([][]float32, n)
				for i := range X {
					X[i] = make([]float32, cols)
					for j := range X[i] {
						X[i][j] = rng.Float32()*2 - 1
					}
				}
				Y, err := rawMatvec(dev, c.typ, w, rows, cols, X)
				if err != nil {
					t.Skipf("no CUDA device %d: %v", dev, err)
				}
				rowBytes := cols / elems * bytes
				for i := range X {
					xs := sums32(X[i])
					for r := 0; r < rows; r++ {
						want := float64(fusedDot(c.typ)(w[r*rowBytes:(r+1)*rowBytes], X[i], xs))
						if math.Abs(float64(Y[i][r])-want) > 1e-3*(1+math.Abs(want)) {
							t.Fatalf("dev %d type %d n=%d row %d: gpu %v cpu %v", dev, c.typ, n, r, Y[i][r], want)
						}
					}
				}
			}
		}
	}
}
