package pack

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"strata/pkg/gguf"
)

// PackModel processes a GGUF model and creates a native Strata pack directory
func PackModel(opts PackOptions) error {
	if opts.GGUFPath == "" {
		return fmt.Errorf("missing GGUF path")
	}
	if opts.OutDir == "" {
		return fmt.Errorf("missing output pack directory")
	}

	if err := os.MkdirAll(opts.OutDir, 0755); err != nil {
		return fmt.Errorf("create pack dir %s: %w", opts.OutDir, err)
	}

	g, err := gguf.Open(opts.GGUFPath)
	if err != nil {
		return fmt.Errorf("open gguf: %w", err)
	}
	defer g.Close()

	// 1. Scan and write native_experts.txt
	layers, err := ScanLayerExperts(g)
	if err != nil {
		return fmt.Errorf("scan experts: %w", err)
	}

	nativePath := filepath.Join(opts.OutDir, "native_experts.txt")
	if err := WriteNativeExpertsTxt(layers, nativePath); err != nil {
		return fmt.Errorf("write native_experts.txt: %w", err)
	}

	// 2. Write dense.bin, index.txt, and conversions.json
	densePath := filepath.Join(opts.OutDir, "dense.bin")
	denseFile, err := os.Create(densePath)
	if err != nil {
		return fmt.Errorf("create dense.bin: %w", err)
	}
	defer denseFile.Close()

	indexPath := filepath.Join(opts.OutDir, "index.txt")
	indexFile, err := os.Create(indexPath)
	if err != nil {
		return fmt.Errorf("create index.txt: %w", err)
	}
	defer indexFile.Close()

	var conversions []ConversionRecord
	denseOffset := uint64(0)

	for _, t := range g.Tensors {
		// Skip expert tensors from dense.bin (they live in GGUF or experts.bin)
		if isExpertTensor(t.Name) {
			continue
		}

		targetForm := TargetFloatForm(t.Name)
		data, err := g.ReadTensorBytes(t)
		if err != nil {
			return fmt.Errorf("read tensor %s: %w", t.Name, err)
		}

		outData := data
		kind := uint32(0) // raw kind

		if targetForm == "BF16" && t.Type == gguf.GGMLTypeF32 {
			outData = ConvertF32ToBF16(data)
			kind = 4 // BF16 kind
			conversions = append(conversions, ConversionRecord{
				Tensor: t.Name,
				Src:    "F32",
				Dst:    "BF16",
				Method: "nearest-even",
			})
		} else if targetForm == "F32" && t.Type == gguf.GGMLTypeF16 {
			outData = ConvertF16ToF32(data)
			kind = 2 // F32 kind
			conversions = append(conversions, ConversionRecord{
				Tensor: t.Name,
				Src:    "F16",
				Dst:    "F32",
				Method: "exact",
			})
		}

		// Write to dense.bin
		if _, err := denseFile.Write(outData); err != nil {
			return fmt.Errorf("write %s to dense.bin: %w", t.Name, err)
		}

		// Write line to index.txt: name kind offset length dims...
		dimsStr := make([]string, len(t.Dims))
		for i, d := range t.Dims {
			dimsStr[i] = fmt.Sprintf("%d", d)
		}
		indexLine := fmt.Sprintf("%s %d %d %d %s\n", t.Name, kind, denseOffset, len(outData), strings.Join(dimsStr, " "))
		if _, err := io.WriteString(indexFile, indexLine); err != nil {
			return fmt.Errorf("write index line for %s: %w", t.Name, err)
		}

		denseOffset += uint64(len(outData))
	}

	// 3. Write conversions.json
	convPath := filepath.Join(opts.OutDir, "conversions.json")
	convBytes, _ := json.MarshalIndent(conversions, "", "  ")
	if err := os.WriteFile(convPath, convBytes, 0644); err != nil {
		return fmt.Errorf("write conversions.json: %w", err)
	}

	// 4. Write optional experts.bin
	if opts.ExpertsBin {
		expertsPath := filepath.Join(opts.OutDir, "experts.bin")
		if err := WriteExpertsBin(g, layers, expertsPath); err != nil {
			return fmt.Errorf("write experts.bin: %w", err)
		}
	}

	return nil
}

func isExpertTensor(name string) bool {
	return strings.Contains(name, "ffn_gate_exps") ||
		strings.Contains(name, "ffn_up_exps") ||
		strings.Contains(name, "ffn_down_exps") ||
		strings.Contains(name, "ffn_gate_exp") ||
		strings.Contains(name, "ffn_up_exp") ||
		strings.Contains(name, "ffn_down_exp")
}
