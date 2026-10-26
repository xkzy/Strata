package densecpu

import (
	"math"
	"math/rand"
	"testing"
)

// Each fused kernel must equal "dequantize the row, then dot" on random blocks.
func TestFusedDotMatchesDequant(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, c := range []struct {
		typ   uint32
		f16at []int // offsets of f16 scale fields, overwritten with sane values
	}{{tQ4_K, []int{0, 2}}, {tQ5_K, []int{0, 2}}, {tQ6_K, []int{208}}, {tQ8_0, []int{0}}, {tBF16, nil}} {
		elems, bytes, _ := blockInfo(c.typ)
		cols := 768 // three K-blocks, 24 Q8_0 blocks
		row := make([]byte, cols/elems*bytes)
		rng.Read(row)
		for b := 0; b < cols/elems; b++ {
			for _, o := range c.f16at {
				row[b*bytes+o], row[b*bytes+o+1] = byte(rng.Intn(256)), 0x2c+byte(rng.Intn(8)) // f16 in [~0.06, ~1]
			}
			if c.typ == tBF16 {
				row[b*2+1] &= 0x3f // keep the exponent finite
			}
		}
		x := make([]float32, cols)
		for i := range x {
			x[i] = rng.Float32()*2 - 1
		}
		deq := make([]float32, cols)
		if err := DequantRow(c.typ, row, cols, deq); err != nil {
			t.Fatal(err)
		}
		var want float64
		for i := range x {
			want += float64(deq[i]) * float64(x[i])
		}
		got := float64(fusedDot(c.typ)(row, x, sums32(x)))
		if math.Abs(got-want) > 1e-3*(1+math.Abs(want)) {
			t.Errorf("type %d: fused %v, dequant+dot %v", c.typ, got, want)
		}
	}
}
