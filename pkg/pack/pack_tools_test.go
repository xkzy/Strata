package pack

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"strata/pkg/gguf"
)

func TestPackIndexFormattingAndParsing(t *testing.T) {
	entry := IndexEntry{
		Name:        "blk.0.attn_q.weight",
		FileID:      0,
		Kind:        IndexKindBF16InF32,
		SrcOffset:   1024,
		SrcBytes:    4096,
		DstOffset:   0,
		DstBytes:    2048,
		Ne0:         128,
		Ne1:         16,
		CodeBits:    0,
		CodeBias:    0,
		GroupElems:  0,
		Codebook:    0,
		HasOffset:   0,
		CodesBytes:  2048,
		ScalesBytes: 0,
		OffsetBytes: 0,
		ScalesFP16:  0,
	}

	line := entry.FormatLine()
	parsed, err := ParseIndexLine(line)
	if err != nil {
		t.Fatalf("failed to parse index line: %v", err)
	}

	if parsed.Name != entry.Name || parsed.Kind != entry.Kind || parsed.SrcBytes != entry.SrcBytes {
		t.Errorf("parsed entry mismatch: got %+v, want %+v", parsed, entry)
	}

	buf := new(bytes.Buffer)
	if err := WriteIndexTable(buf, []IndexEntry{entry}); err != nil {
		t.Fatalf("WriteIndexTable failed: %v", err)
	}
	if !strings.Contains(buf.String(), "blk.0.attn_q.weight") {
		t.Errorf("buffer missing tensor name")
	}
}

func TestQuantizeRowQ8_0(t *testing.T) {
	data := make([]float32, 64)
	for i := range data {
		data[i] = float32(i) / 10.0
	}

	quant := QuantizeRowQ8_0(data)
	if len(quant) != 2*34 {
		t.Fatalf("expected 68 bytes for 2 blocks of 32, got %d", len(quant))
	}

	// Verify scale is positive
	scaleU16 := binary.LittleEndian.Uint16(quant[:2])
	if scaleU16 == 0 {
		t.Errorf("expected non-zero fp16 scale")
	}
}

func TestBuildMTPRuntimeFiles(t *testing.T) {
	tmpDir := t.TempDir()
	ggufPath := filepath.Join(tmpDir, "mtp-test.gguf")
	outDir := filepath.Join(tmpDir, "rt")

	gw := gguf.NewWriter()
	gw.SetMetadata("general.architecture", "strata-mtp")
	dummyProj := make([]byte, 32*2) // 32 BF16 values
	gw.AddTensor("mlp.shared_expert.gate_proj.weight", []uint64{32}, gguf.GGMLTypeBF16, dummyProj)
	if err := gw.WriteFile(ggufPath); err != nil {
		t.Fatalf("write mtp test gguf failed: %v", err)
	}

	if err := BuildMTPRuntimeFiles(ggufPath, outDir); err != nil {
		t.Fatalf("BuildMTPRuntimeFiles failed: %v", err)
	}

	denseTxt := filepath.Join(outDir, "dense.txt")
	txtData, err := os.ReadFile(denseTxt)
	if err != nil {
		t.Fatalf("dense.txt missing: %v", err)
	}
	if !strings.Contains(string(txtData), "mlp.shared_expert.gate_proj.weight") {
		t.Errorf("dense.txt does not contain tensor: %s", string(txtData))
	}
}

func TestSafetensorsHeaderReading(t *testing.T) {
	tmpDir := t.TempDir()
	sfPath := filepath.Join(tmpDir, "test.safetensors")

	hdrMap := map[string]any{
		"test_tensor": map[string]any{
			"dtype":        "BF16",
			"shape":        []int{16, 16},
			"data_offsets": []int{0, 512},
		},
	}
	hdrJSON, _ := json.Marshal(hdrMap)
	buf := new(bytes.Buffer)
	binary.Write(buf, binary.LittleEndian, uint64(len(hdrJSON)))
	buf.Write(hdrJSON)
	dummyData := make([]byte, 512)
	buf.Write(dummyData)

	if err := os.WriteFile(sfPath, buf.Bytes(), 0644); err != nil {
		t.Fatalf("write safetensors failed: %v", err)
	}

	hdr, base, err := ReadSafetensorsHeader(sfPath)
	if err != nil {
		t.Fatalf("ReadSafetensorsHeader failed: %v", err)
	}
	if base != int64(8+len(hdrJSON)) {
		t.Errorf("base offset mismatch: got %d, want %d", base, 8+len(hdrJSON))
	}
	if _, ok := hdr["test_tensor"]; !ok {
		t.Errorf("test_tensor missing from header")
	}
}
