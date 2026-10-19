package pack

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"strata/pkg/gguf"
)

type SafetensorsHeader map[string]struct {
	Dtype       string   `json:"dtype"`
	Shape       []uint64 `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// ReadSafetensorsHeader extracts the JSON header and the byte offset where tensor data starts.
func ReadSafetensorsHeader(path string) (SafetensorsHeader, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}
	defer f.Close()

	var hdrLen uint64
	if err := binary.Read(f, binary.LittleEndian, &hdrLen); err != nil {
		return nil, 0, fmt.Errorf("read header length: %w", err)
	}

	hdrBytes := make([]byte, hdrLen)
	if _, err := io.ReadFull(f, hdrBytes); err != nil {
		return nil, 0, fmt.Errorf("read header bytes: %w", err)
	}

	var hdr SafetensorsHeader
	if err := json.Unmarshal(hdrBytes, &hdr); err != nil {
		return nil, 0, fmt.Errorf("unmarshal header json: %w", err)
	}

	return hdr, int64(8 + hdrLen), nil
}

// PackPLEFP8Options defines options for packaging PLE ngram embedding shards into GGUF.
type PackPLEFP8Options struct {
	ModelDir string // Directory containing model.safetensors.index.json
	OutGGUF  string // Destination GGUF path
}

// PackPLEFP8 reads ngram_embedding shards and emits a single-tensor GGUF with FP8 bytes.
func PackPLEFP8(opts PackPLEFP8Options) error {
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

	shardFiles := make(map[int]string)
	var scaleShard string
	var scaleTensorName string

	for name, shardFile := range indexData.WeightMap {
		if strings.Contains(name, ".ngram_embedding.shard_") && strings.HasSuffix(name, ".weight") {
			parts := strings.Split(name, ".ngram_embedding.shard_")
			if len(parts) == 2 {
				idxPart := strings.TrimSuffix(parts[1], ".weight")
				if k, err := strconv.Atoi(idxPart); err == nil {
					shardFiles[k] = shardFile
				}
			}
		} else if strings.HasSuffix(name, ".ngram_embedding.weight_scale") {
			scaleTensorName = name
			scaleShard = shardFile
		}
	}

	if len(shardFiles) == 0 || scaleShard == "" {
		return fmt.Errorf("no ngram_embedding shards or scale found in %s", opts.ModelDir)
	}

	// Read scale from BF16
	scaleHeader, scaleBase, err := ReadSafetensorsHeader(filepath.Join(opts.ModelDir, scaleShard))
	if err != nil {
		return fmt.Errorf("read scale header: %w", err)
	}
	scaleMeta, ok := scaleHeader[scaleTensorName]
	if !ok {
		return fmt.Errorf("scale tensor %s not found in %s", scaleTensorName, scaleShard)
	}

	scaleF, err := os.Open(filepath.Join(opts.ModelDir, scaleShard))
	if err != nil {
		return err
	}
	defer scaleF.Close()

	if _, err := scaleF.Seek(scaleBase+scaleMeta.DataOffsets[0], io.SeekStart); err != nil {
		return err
	}
	var scaleU16 uint16
	if err := binary.Read(scaleF, binary.LittleEndian, &scaleU16); err != nil {
		return err
	}
	// Convert BF16 to Float32
	scaleBits := uint32(scaleU16) << 16
	scaleF32 := math.Float32frombits(scaleBits)

	// Sort shards
	shardIndices := make([]int, 0, len(shardFiles))
	for k := range shardFiles {
		shardIndices = append(shardIndices, k)
	}
	sort.Ints(shardIndices)

	// Calculate total bytes
	totalRows := uint64(0)
	cols := uint64(160)

	for _, k := range shardIndices {
		shardFile := shardFiles[k]
		hdr, _, err := ReadSafetensorsHeader(filepath.Join(opts.ModelDir, shardFile))
		if err != nil {
			return fmt.Errorf("shard %d header: %w", k, err)
		}
		tensorName := fmt.Sprintf("model.layers.0.mlp.ngram_embedding.shard_%d.weight", k)
		// Try matching whatever name was mapped
		for name := range hdr {
			if strings.Contains(name, fmt.Sprintf(".shard_%d.weight", k)) {
				tensorName = name
				break
			}
		}
		tMeta := hdr[tensorName]
		if len(tMeta.Shape) >= 2 {
			totalRows += tMeta.Shape[0]
			cols = tMeta.Shape[1]
		}
	}

	// Prepare GGUF
	gw := gguf.NewWriter()
	gw.SetMetadata("general.architecture", "strata-ple")
	gw.SetMetadata("strata.ple.format", "f8_e4m3")
	gw.SetMetadata("strata.ple.scale", scaleF32)

	totalDataBytes := totalRows * cols
	dummyData := make([]byte, totalDataBytes)

	// Stream shards into dummyData
	offset := uint64(0)
	for _, k := range shardIndices {
		shardFile := shardFiles[k]
		path := filepath.Join(opts.ModelDir, shardFile)
		hdr, base, err := ReadSafetensorsHeader(path)
		if err != nil {
			return err
		}

		var tMeta struct {
			Dtype       string   `json:"dtype"`
			Shape       []uint64 `json:"shape"`
			DataOffsets [2]int64 `json:"data_offsets"`
		}
		for name, m := range hdr {
			if strings.Contains(name, fmt.Sprintf(".shard_%d.weight", k)) {
				tMeta = m
				break
			}
		}

		f, err := os.Open(path)
		if err != nil {
			return err
		}
		if _, err := f.Seek(base+tMeta.DataOffsets[0], io.SeekStart); err != nil {
			f.Close()
			return err
		}
		chunkSize := uint64(tMeta.DataOffsets[1] - tMeta.DataOffsets[0])
		if _, err := io.ReadFull(f, dummyData[offset:offset+chunkSize]); err != nil {
			f.Close()
			return err
		}
		f.Close()
		offset += chunkSize
	}

	gw.AddTensor("per_layer_token_embd.weight", []uint64{cols, totalRows}, 24 /* GGML_TYPE_I8 */, dummyData)

	outF, err := os.Create(opts.OutGGUF)
	if err != nil {
		return err
	}
	defer outF.Close()

	return gw.Write(outF)
}
