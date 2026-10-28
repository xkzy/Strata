// Package densecpu is a slow, plain-Go reference forward pass for the dense qwen35 and gemma4 GGUFs.
// It exists to be checked against (does Gemma continue "The capital of France is" with " Paris"?) before any CUDA
// kernel is written; it is not a serving path.
package densecpu

import (
	"encoding/binary"
	"fmt"
	"math"
)

// GGML tensor type ids used by the two files.
const (
	tF32    = 0
	tF16    = 1
	tQ8_0   = 8
	tQ4_K   = 12
	tQ5_K   = 13
	tQ6_K   = 14
	tIQ3_S  = 21
	tIQ4_XS = 23
	tBF16   = 30
)

// blockInfo returns elements per block and bytes per block for a supported type.
func blockInfo(t uint32) (elems, bytes int, err error) {
	switch t {
	case tF32:
		return 1, 4, nil
	case tF16, tBF16:
		return 1, 2, nil
	case tQ8_0:
		return 32, 34, nil
	case tQ4_K:
		return 256, 144, nil
	case tQ5_K:
		return 256, 176, nil
	case tQ6_K:
		return 256, 210, nil
	case tIQ3_S:
		return 256, 110, nil
	case tIQ4_XS:
		return 256, 136, nil
	}
	return 0, 0, fmt.Errorf("densecpu: tensor type %d not supported", t)
}

func f16(b []byte) float32 {
	h := uint32(binary.LittleEndian.Uint16(b))
	sign, exp, man := h>>15, (h>>10)&0x1f, h&0x3ff
	var f float32
	switch exp {
	case 0:
		f = float32(math.Ldexp(float64(man), -24))
	case 31:
		if man == 0 {
			f = float32(math.Inf(1))
		} else {
			f = float32(math.NaN())
		}
	default:
		f = float32(math.Ldexp(float64(man|0x400), int(exp)-25))
	}
	if sign == 1 {
		return -f
	}
	return f
}

func scaleMinK4(j int, q []byte) (d, m byte) {
	if j < 4 {
		return q[j] & 63, q[j+4] & 63
	}
	return (q[j+4] & 0xf) | ((q[j-4] >> 6) << 4), (q[j+4] >> 4) | ((q[j] >> 6) << 4)
}

// dequantBlock expands one block of type t from src into dst (len = elements per block).
func dequantBlock(t uint32, src []byte, dst []float32) {
	switch t {
	case tF32:
		dst[0] = math.Float32frombits(binary.LittleEndian.Uint32(src))
	case tF16:
		dst[0] = f16(src)
	case tBF16:
		dst[0] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(src)) << 16)
	case tIQ3_S:
		dequantIQ3S(src, dst)
	case tIQ4_XS:
		dequantIQ4XS(src, dst)
	case tQ8_0:
		d := f16(src)
		for j := 0; j < 32; j++ {
			dst[j] = float32(int8(src[2+j])) * d
		}
	case tQ4_K:
		d, dmin := f16(src), f16(src[2:])
		scales, q := src[4:16], src[16:]
		is, y := 0, 0
		for j := 0; j < 256; j += 64 {
			sc, m := scaleMinK4(is, scales)
			d1, m1 := d*float32(sc), dmin*float32(m)
			sc, m = scaleMinK4(is+1, scales)
			d2, m2 := d*float32(sc), dmin*float32(m)
			for l := 0; l < 32; l++ {
				dst[y+l] = d1*float32(q[l]&0xf) - m1
				dst[y+32+l] = d2*float32(q[l]>>4) - m2
			}
			y += 64
			q = q[32:]
			is += 2
		}
	case tQ5_K:
		d, dmin := f16(src), f16(src[2:])
		scales, qh, ql := src[4:16], src[16:48], src[48:]
		is, y := 0, 0
		u1, u2 := byte(1), byte(2)
		for j := 0; j < 256; j += 64 {
			sc, m := scaleMinK4(is, scales)
			d1, m1 := d*float32(sc), dmin*float32(m)
			sc, m = scaleMinK4(is+1, scales)
			d2, m2 := d*float32(sc), dmin*float32(m)
			for l := 0; l < 32; l++ {
				lo, hi := float32(ql[l]&0xf), float32(ql[l]>>4)
				if qh[l]&u1 != 0 {
					lo += 16
				}
				if qh[l]&u2 != 0 {
					hi += 16
				}
				dst[y+l] = d1*lo - m1
				dst[y+32+l] = d2*hi - m2
			}
			y += 64
			ql = ql[32:]
			is += 2
			u1 <<= 2
			u2 <<= 2
		}
	case tQ6_K:
		ql, qh, sc := src[0:128], src[128:192], src[192:208]
		d := f16(src[208:])
		y := 0
		for n := 0; n < 256; n += 128 {
			for l := 0; l < 32; l++ {
				is := l / 16
				q1 := int(ql[l]&0xf|((qh[l]>>0)&3)<<4) - 32
				q2 := int(ql[l+32]&0xf|((qh[l]>>2)&3)<<4) - 32
				q3 := int(ql[l]>>4|((qh[l]>>4)&3)<<4) - 32
				q4 := int(ql[l+32]>>4|((qh[l]>>6)&3)<<4) - 32
				dst[y+l] = d * float32(int8(sc[is])) * float32(q1)
				dst[y+l+32] = d * float32(int8(sc[is+2])) * float32(q2)
				dst[y+l+64] = d * float32(int8(sc[is+4])) * float32(q3)
				dst[y+l+96] = d * float32(int8(sc[is+6])) * float32(q4)
			}
			y += 128
			ql, qh, sc = ql[64:], qh[32:], sc[8:]
		}
	}
}

// DequantRow expands one row of cols elements (cols must be a whole number of blocks) into dst.
func DequantRow(t uint32, row []byte, cols int, dst []float32) error {
	elems, bytes, err := blockInfo(t)
	if err != nil {
		return err
	}
	if cols%elems != 0 || len(row) < cols/elems*bytes {
		return fmt.Errorf("densecpu: row of %d elements / %d bytes does not fit type %d", cols, len(row), t)
	}
	if t == tBF16 { // the output head is a 1.3G-element BF16 matrix: no per-element dispatch
		for i := 0; i < cols; i++ {
			dst[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(row[2*i:])) << 16)
		}
		return nil
	}
	for b := 0; b < cols/elems; b++ {
		dequantBlock(t, row[b*bytes:], dst[b*elems:(b+1)*elems])
	}
	return nil
}
