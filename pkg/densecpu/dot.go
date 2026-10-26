package densecpu

import (
	"encoding/binary"
	"math"
)

// rowDot is the dot product of one quantized weight row with an activation vector x. xs holds the sums of x over
// each run of 32 elements (the K-quants' minimum term needs them, and they are the same for every weight row).
type rowDot func(row []byte, x, xs []float32) float32

// fusedDot returns the dequantize-and-dot kernel for type t, or nil when only DequantRow supports it.
// shortcut: plain Go loops, no SIMD; upgrade to the CUDA path (or assembly) when CPU decode speed matters.
func fusedDot(t uint32) rowDot {
	switch t {
	case tQ4_K:
		return dotQ4K
	case tQ5_K:
		return dotQ5K
	case tQ6_K:
		return dotQ6K
	case tQ8_0:
		return dotQ8_0
	case tBF16:
		return dotBF16
	}
	return nil
}

func sums32(x []float32) []float32 {
	out := make([]float32, len(x)/32)
	for g := range out {
		var s float32
		for _, v := range x[g*32 : g*32+32] {
			s += v
		}
		out[g] = s
	}
	return out
}

func dotQ4K(row []byte, x, xs []float32) float32 {
	var total float32
	for b := 0; b < len(x)/256; b++ {
		blk := row[b*144 : b*144+144]
		d, dmin := f16(blk), f16(blk[2:])
		sc, q := blk[4:16], blk[16:144]
		xb, xsb := x[b*256:b*256+256], xs[b*8:b*8+8]
		var sumq, summ float32
		for j := 0; j < 4; j++ {
			s1, m1 := scaleMinK4(2*j, sc)
			s2, m2 := scaleMinK4(2*j+1, sc)
			qq, x1, x2 := q[j*32:j*32+32], xb[j*64:j*64+32], xb[j*64+32:j*64+64]
			var a1, a2 float32
			for l := 0; l < 32; l++ {
				v := qq[l]
				a1 += float32(v&0xf) * x1[l]
				a2 += float32(v>>4) * x2[l]
			}
			sumq += float32(s1)*a1 + float32(s2)*a2
			summ += float32(m1)*xsb[2*j] + float32(m2)*xsb[2*j+1]
		}
		total += d*sumq - dmin*summ
	}
	return total
}

func dotQ5K(row []byte, x, xs []float32) float32 {
	var total float32
	for b := 0; b < len(x)/256; b++ {
		blk := row[b*176 : b*176+176]
		d, dmin := f16(blk), f16(blk[2:])
		sc, qh, q := blk[4:16], blk[16:48], blk[48:176]
		xb, xsb := x[b*256:b*256+256], xs[b*8:b*8+8]
		var sumq, summ float32
		for j := 0; j < 4; j++ {
			s1, m1 := scaleMinK4(2*j, sc)
			s2, m2 := scaleMinK4(2*j+1, sc)
			u1, u2 := byte(1)<<(2*j), byte(2)<<(2*j)
			qq, x1, x2 := q[j*32:j*32+32], xb[j*64:j*64+32], xb[j*64+32:j*64+64]
			var a1, a2 float32
			for l := 0; l < 32; l++ {
				v1, v2 := float32(qq[l]&0xf), float32(qq[l]>>4)
				if qh[l]&u1 != 0 {
					v1 += 16
				}
				if qh[l]&u2 != 0 {
					v2 += 16
				}
				a1 += v1 * x1[l]
				a2 += v2 * x2[l]
			}
			sumq += float32(s1)*a1 + float32(s2)*a2
			summ += float32(m1)*xsb[2*j] + float32(m2)*xsb[2*j+1]
		}
		total += d*sumq - dmin*summ
	}
	return total
}

func dotQ6K(row []byte, x, _ []float32) float32 {
	var total float32
	for b := 0; b < len(x)/256; b++ {
		blk := row[b*210 : b*210+210]
		ql, qh, sc := blk[0:128], blk[128:192], blk[192:208]
		d := f16(blk[208:])
		xb := x[b*256 : b*256+256]
		var acc float32
		for n := 0; n < 2; n++ {
			qln, qhn, scn, xn := ql[n*64:n*64+64], qh[n*32:n*32+32], sc[n*8:n*8+8], xb[n*128:n*128+128]
			var a [8]float32 // one accumulator per 16-element scale group
			for l := 0; l < 32; l++ {
				is := l / 16
				q1 := float32(int(qln[l]&0xf|((qhn[l]>>0)&3)<<4) - 32)
				q2 := float32(int(qln[l+32]&0xf|((qhn[l]>>2)&3)<<4) - 32)
				q3 := float32(int(qln[l]>>4|((qhn[l]>>4)&3)<<4) - 32)
				q4 := float32(int(qln[l+32]>>4|((qhn[l]>>6)&3)<<4) - 32)
				a[is] += q1 * xn[l]
				a[is+2] += q2 * xn[l+32]
				a[is+4] += q3 * xn[l+64]
				a[is+6] += q4 * xn[l+96]
			}
			for k := 0; k < 8; k++ {
				acc += float32(int8(scn[k])) * a[k]
			}
		}
		total += d * acc
	}
	return total
}

func dotQ8_0(row []byte, x, _ []float32) float32 {
	var total float32
	for b := 0; b < len(x)/32; b++ {
		blk := row[b*34 : b*34+34]
		xb := x[b*32 : b*32+32]
		var a float32
		for l := 0; l < 32; l++ {
			a += float32(int8(blk[2+l])) * xb[l]
		}
		total += f16(blk) * a
	}
	return total
}

func dotBF16(row []byte, x, _ []float32) float32 {
	var a float32
	for i, v := range x {
		a += math.Float32frombits(uint32(binary.LittleEndian.Uint16(row[2*i:]))<<16) * v
	}
	return a
}
