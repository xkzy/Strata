package densecpu

import (
	"math"
	"math/rand"
	"testing"
)

func TestHalfRoundTrip(t *testing.T) {
	for i := 0; i < 65536; i++ {
		v := halfTab[i]
		if v != v { // nan
			continue
		}
		if i&0x7fff < 0x7c00 && h2f32(uint16(i)) != v {
			t.Fatalf("h2f32(%#04x) = %v, table says %v", i, h2f32(uint16(i)), v)
		}
		if got := toHalf(v); got != uint16(i) {
			t.Fatalf("half %#04x -> %v -> %#04x", i, v, got)
		}
	}
	r := rand.New(rand.NewSource(1))
	for n := 0; n < 100000; n++ {
		x := float32(r.NormFloat64()) * float32(math.Pow(10, float64(r.Intn(8)-4)))
		got := halfTab[toHalf(x)]
		if a := math.Abs(float64(x)); a > 1e-4 && a < 6e4 && math.Abs(float64(got-x)) > a*(1.0/2048)*1.001 {
			t.Fatalf("%v -> %v: error above half a unit in the last place", x, got)
		}
	}
}
