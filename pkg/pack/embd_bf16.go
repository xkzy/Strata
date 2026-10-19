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

// PackEmbdBF16Options defines options for packaging token embedding from safetensors into GGUF.
type PackEmbdBF16Options struct {
	ModelDir string // Directory containing model.safetensors.index.json
	OutGGUF  string // Destination GGUF path
}

// PackEmbdBF16 extracts embed_tokens.weight from safetensors and emits a GGUF with token_embd.weight.
func PackEmbdBF16(opts PackEmbdBF16Options) error {
	indexPath := filepath.Join(opts.ModelDir, "model.safetensors.index.json")
	indexBytes, err := os.ReadFile(indexPath)
	if err != nil {
		return fmt.Errorf("read safetensors index: %w", err)
	}

	var indexData struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(indexBytes, &indexData); err != nil {
		return fmt.Errorf("unmarshal safetensors index: %w", err)
	}

	var embdName, embdShard string
	for name, shard := range indexData.WeightMap {
		if strings.HasSuffix(name, "embed_tokens.weight") && !strings.Contains(name, "visual") && !strings.Contains(name, "mtp") {
			embdName = name
			embdShard = shard
			break
		}
	}

	if embdName == "" || embdShard == "" {
		return fmt.Errorf("embed_tokens.weight not found in %s", opts.ModelDir)
	}

	shardPath := filepath.Join(opts.ModelDir, embdShard)
	hdr, base, err := ReadSafetensorsHeader(shardPath)
	if err != nil {
		return fmt.Errorf("read safetensors header: %w", err)
	}

	tMeta, ok := hdr[embdName]
	if !ok {
		return fmt.Errorf("tensor %s not found in %s", embdName, shardPath)
	}

	if len(tMeta.Shape) != 2 {
		return fmt.Errorf("expected 2D tensor shape, got %v", tMeta.Shape)
	}
	vocab := tMeta.Shape[0]
	dim := tMeta.Shape[1]

	f, err := os.Open(shardPath)
	if err != nil {
		return err
	}
	defer f.Close()

	if _, err := f.Seek(base+tMeta.DataOffsets[0], io.SeekStart); err != nil {
		return err
	}

	dataBytes := (tMeta.DataOffsets[1] - tMeta.DataOffsets[0])
	tensorData := make([]byte, dataBytes)
	if _, err := io.ReadFull(f, tensorData); err != nil {
		return fmt.Errorf("read tensor data: %w", err)
	}

	gw := gguf.NewWriter()
	gw.SetMetadata("general.architecture", "strata-embd")
	// Add token_embd.weight as BF16 (GGML_TYPE_BF16 = 30)
	gw.AddTensor("token_embd.weight", []uint64{dim, vocab}, 30, tensorData)

	outF, err := os.Create(opts.OutGGUF)
	if err != nil {
		return err
	}
	defer outF.Close()

	return gw.Write(outF)
}
