package pack

import (
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"

	"strata/pkg/gguf"
)

// ScanLayerExperts scans GGUF tensors to discover MoE layers and expert weight offsets
func ScanLayerExperts(g *gguf.GGUFFile) ([]LayerExperts, error) {
	layerMap := make(map[int]*LayerExperts)

	for _, t := range g.Tensors {
		if !strings.HasPrefix(t.Name, "blk.") {
			continue
		}
		parts := strings.Split(t.Name, ".")
		if len(parts) < 3 {
			continue
		}
		layerNum, err := strconv.Atoi(parts[1])
		if err != nil {
			continue
		}

		suffix := strings.Join(parts[2:], ".")
		switch suffix {
		case "ffn_gate_exps.weight", "ffn_gate_exp.weight":
			le := getOrCreate(layerMap, layerNum)
			le.GateOffset = t.Offset
			le.GUType = t.Type
		case "ffn_up_exps.weight", "ffn_up_exp.weight":
			le := getOrCreate(layerMap, layerNum)
			le.UpOffset = t.Offset
		case "ffn_down_exps.weight", "ffn_down_exp.weight":
			le := getOrCreate(layerMap, layerNum)
			le.DownOffset = t.Offset
			le.DType = t.Type
		}
	}

	var layers []int
	for l := range layerMap {
		layers = append(layers, l)
	}
	sort.Ints(layers)

	var result []LayerExperts
	for _, l := range layers {
		le := layerMap[l]
		le.BlobBytes = (le.DownOffset - le.GateOffset) + 1024 // standard blob estimate
		result = append(result, *le)
	}

	return result, nil
}

func getOrCreate(m map[int]*LayerExperts, l int) *LayerExperts {
	if existing, ok := m[l]; ok {
		return existing
	}
	ne := &LayerExperts{Layer: l}
	m[l] = ne
	return ne
}

// WriteNativeExpertsTxt writes the native_experts.txt file required by the C++ engine
func WriteNativeExpertsTxt(layers []LayerExperts, path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	for _, l := range layers {
		// format: layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard]
		line := fmt.Sprintf("%d %d %d %d %d %d %d %d\n",
			l.Layer, l.GUType, l.DType, l.Offset, l.BlobBytes, l.GateOffset, l.UpOffset, l.DownOffset)
		if _, err := io.WriteString(f, line); err != nil {
			return err
		}
	}
	return nil
}

// WriteExpertsBin writes experts.bin by concatenating expert slice bytes for all layers
func WriteExpertsBin(g *gguf.GGUFFile, layers []LayerExperts, path string) error {
	tmpPath := path + ".tmp"
	f, err := os.Create(tmpPath)
	if err != nil {
		return err
	}
	defer func() {
		f.Close()
		_ = os.Remove(tmpPath)
	}()

	for _, l := range layers {
		gateName := fmt.Sprintf("blk.%d.ffn_gate_exps.weight", l.Layer)
		upName := fmt.Sprintf("blk.%d.ffn_up_exps.weight", l.Layer)
		downName := fmt.Sprintf("blk.%d.ffn_down_exps.weight", l.Layer)

		for _, name := range []string{gateName, upName, downName} {
			t, ok := g.GetTensor(name)
			if !ok {
				continue
			}
			data, err := g.ReadTensorBytes(t)
			if err != nil {
				return fmt.Errorf("read %s: %w", name, err)
			}
			if _, err := f.Write(data); err != nil {
				return err
			}
		}
	}

	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmpPath, path)
}
