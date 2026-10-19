package pack

import (
	"encoding/binary"
	"math"
)

// TargetFloatForm returns the required engine format for dense pack tensors
func TargetFloatForm(name string) string {
	switch {
	case containsAny(name, "ffn_gate_inp", "hc_attn", "hc_ffn", "output_hc", "indexer", "ssm_alpha", "ssm_beta", "ple_key", "ple_value"):
		return "BF16"
	case containsAny(name, "ple_conv1d"):
		return "F16"
	case containsAny(name, "norm", "ln_1", "ln_2", "ln_f"):
		return "F32"
	default:
		return "AS_STORED"
	}
}

func containsAny(s string, substrings ...string) bool {
	for _, sub := range substrings {
		if len(s) >= len(sub) && (s == sub || hasSubstr(s, sub)) {
			return true
		}
	}
	return false
}

func hasSubstr(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

// ConvertF32ToBF16 converts IEEE 754 32-bit floats to bfloat16 using round-to-nearest-even
func ConvertF32ToBF16(src []byte) []byte {
	numFloats := len(src) / 4
	dst := make([]byte, numFloats*2)
	le := binary.LittleEndian

	for i := 0; i < numFloats; i++ {
		u32 := le.Uint32(src[i*4 : (i+1)*4])
		// Round-to-nearest-even
		lsb := (u32 >> 16) & 1
		bias := uint32(0x7fff) + lsb
		bf16 := uint16((u32 + bias) >> 16)
		le.PutUint16(dst[i*2:(i+1)*2], bf16)
	}
	return dst
}

// ConvertF16ToF32 widens half-precision float16 to single-precision float32
func ConvertF16ToF32(src []byte) []byte {
	numFloats := len(src) / 2
	dst := make([]byte, numFloats*4)
	le := binary.LittleEndian

	for i := 0; i < numFloats; i++ {
		h := le.Uint16(src[i*2 : (i+1)*2])
		sign := uint32(h>>15) << 31
		exp := uint32((h >> 10) & 0x1f)
		mant := uint32(h & 0x3ff)

		var f32Bits uint32
		if exp == 0 {
			if mant != 0 {
				// Subnormal
				val := float32(mant) / 1024.0 * float32(math.Pow(2, -14))
				if sign != 0 {
					val = -val
				}
				f32Bits = math.Float32bits(val)
			} else {
				f32Bits = sign
			}
		} else if exp == 31 {
			f32Bits = sign | 0x7f800000 | (mant << 13)
		} else {
			f32Bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13)
		}
		le.PutUint32(dst[i*4:(i+1)*4], f32Bits)
	}
	return dst
}
