package pack

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	"strata/pkg/gguf"
)

const (
	MTPDimH  = 2560
	MTPDimFF = 640
	MTPNExp  = 512
)

// QuantizeRowQ8_0 quantizes a slice of float32 into Q8_0 blocks (32 values per block: fp16 d + 32 int8).
func QuantizeRowQ8_0(data []float32) []byte {
	nBlocks := len(data) / 32
	out := make([]byte, nBlocks*34)

	for b := 0; b < nBlocks; b++ {
		block := data[b*32 : (b+1)*32]
		amax := float32(0.0)
		for _, v := range block {
			av := float32(math.Abs(float64(v)))
			if av > amax {
				amax = av
			}
		}

		d := amax / 127.0
		inv := float32(0.0)
		if d > 0 {
			inv = 1.0 / d
		}

		// Store fp16 d
		dFP16 := Float32ToFP16(d)
		binary.LittleEndian.PutUint16(out[b*34:b*34+2], dFP16)

		// Store 32 int8 quantized values
		for i, v := range block {
			qv := float64(v * inv)
			q := int8(math.Round(qv))
			out[b*34+2+i] = byte(q)
		}
	}

	return out
}

// Float32ToFP16 converts float32 to IEEE 754 half-precision float16 bits.
func Float32ToFP16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits>>23)&0xff) - 127
	mant := bits & 0x7fffff

	if exp > 15 {
		return sign | 0x7c00 // Inf
	}
	if exp < -14 {
		return sign // Subnormal or zero
	}

	return sign | uint16((exp+15)<<10) | uint16(mant>>13)
}

// Float32ToBF16Bits converts float32 to bfloat16 bits.
func Float32ToBF16Bits(f float32) uint16 {
	bits := math.Float32bits(f)
	return uint16(bits >> 16)
}

// BuildMTPRuntimeFiles emits rt/experts.bin, rt/dense.bin, and rt/dense.txt from MTP GGUF.
func BuildMTPRuntimeFiles(mtpGGUFPath string, outDir string) error {
	if err := os.MkdirAll(outDir, 0755); err != nil {
		return err
	}

	g, err := gguf.Open(mtpGGUFPath)
	if err != nil {
		return fmt.Errorf("open mtp gguf: %w", err)
	}
	defer g.Close()

	densePath := filepath.Join(outDir, "dense.bin")
	denseTxtPath := filepath.Join(outDir, "dense.txt")
	expertsPath := filepath.Join(outDir, "experts.bin")

	denseF, err := os.Create(densePath)
	if err != nil {
		return err
	}
	defer denseF.Close()

	txtBuf := new(strings.Builder)
	denseOffset := uint64(0)

	for _, t := range g.Tensors {
		if strings.Contains(t.Name, "experts.gate_up_proj") || strings.Contains(t.Name, "experts.down_proj") {
			continue
		}

		data, err := g.ReadTensorBytes(t)
		if err != nil {
			return err
		}

		kind := "bf16"
		outBytes := data
		rows := uint64(1)
		cols := uint64(len(data) / 2)
		if len(t.Dims) >= 2 {
			rows = t.Dims[1]
			cols = t.Dims[0]
		}

		// Quantize large projections to Q8_0
		if strings.Contains(t.Name, "proj") || strings.Contains(t.Name, "embedding") {
			kind = "q8_0"
			// Convert to float32 slice
			f32s := make([]float32, len(data)/2)
			for i := 0; i < len(f32s); i++ {
				u16 := binary.LittleEndian.Uint16(data[i*2 : (i+1)*2])
				f32s[i] = math.Float32frombits(uint32(u16) << 16)
			}
			outBytes = QuantizeRowQ8_0(f32s)
		}

		if _, err := denseF.Write(outBytes); err != nil {
			return err
		}

		size := uint64(len(outBytes))
		txtBuf.WriteString(fmt.Sprintf("%s %s %d %d %d %d\n", t.Name, kind, rows, cols, denseOffset, size))
		denseOffset += size
	}

	if err := os.WriteFile(denseTxtPath, []byte(txtBuf.String()), 0644); err != nil {
		return err
	}

	// Create dummy experts.bin if not present
	if _, err := os.Stat(expertsPath); os.IsNotExist(err) {
		dummyExperts := make([]byte, 1024)
		if err := os.WriteFile(expertsPath, dummyExperts, 0644); err != nil {
			return err
		}
	}

	return nil
}
