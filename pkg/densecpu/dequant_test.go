package densecpu

import "testing"

func TestDequantQ8_0(t *testing.T) {
	b := make([]byte, 34)
	b[0], b[1] = 0x00, 0x38 // f16 0.5
	for i := 0; i < 32; i++ {
		b[2+i] = byte(int8(i - 16))
	}
	out := make([]float32, 32)
	dequantBlock(tQ8_0, b, out)
	for i, v := range out {
		if want := float32(i-16) * 0.5; v != want {
			t.Fatalf("q8_0[%d]=%v want %v", i, v, want)
		}
	}
}

func TestDequantQ4K(t *testing.T) {
	b := make([]byte, 144)
	b[0], b[1] = 0x00, 0x3c // d = 1.0, dmin = 0
	for _, i := range []int{4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15} {
		b[i] = 0
	}
	for j := 0; j < 4; j++ {
		b[4+j] = 1 // scales of groups 0..3
	}
	for j := 8; j < 12; j++ {
		b[4+j] = 1 // low 4 bits of the scales of groups 4..7
	}
	for i := 16; i < 144; i++ {
		b[i] = 0x21 // low nibble 1, high nibble 2
	}
	out := make([]float32, 256)
	dequantBlock(tQ4_K, b, out)
	for i, v := range out {
		want := float32(1)
		if (i/32)%2 == 1 {
			want = 2
		}
		if v != want {
			t.Fatalf("q4_k[%d]=%v want %v", i, v, want)
		}
	}
}

func TestDequantQ6K(t *testing.T) {
	b := make([]byte, 210)
	for i := 192; i < 208; i++ {
		b[i] = 1 // scales
	}
	b[208], b[209] = 0x00, 0x3c // d = 1.0
	out := make([]float32, 256)
	dequantBlock(tQ6_K, b, out)
	for i, v := range out {
		if v != -32 {
			t.Fatalf("q6_k[%d]=%v want -32", i, v)
		}
	}
}

func TestF16(t *testing.T) {
	for _, c := range []struct {
		b    [2]byte
		want float32
	}{{[2]byte{0x00, 0x3c}, 1}, {[2]byte{0x00, 0xc0}, -2}, {[2]byte{0x01, 0x00}, 5.9604645e-8}} {
		if got := f16(c.b[:]); got != c.want {
			t.Fatalf("f16(%x)=%v want %v", c.b, got, c.want)
		}
	}
}
