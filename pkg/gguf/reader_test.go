package gguf

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func createSyntheticGGUF() []byte {
	buf := new(bytes.Buffer)
	le := binary.LittleEndian

	// 1. Magic "GGUF"
	buf.WriteString("GGUF")
	// 2. Version 3
	_ = binary.Write(buf, le, uint32(3))
	// 3. Tensor count: 1
	_ = binary.Write(buf, le, uint64(1))
	// 4. Metadata KV count: 3
	_ = binary.Write(buf, le, uint64(3))

	// KV 1: "general.architecture" -> "gpt-oss" (string)
	writeString(buf, "general.architecture")
	_ = binary.Write(buf, le, uint32(TypeString))
	writeString(buf, "gpt-oss")

	// KV 2: "general.alignment" -> uint32(32)
	writeString(buf, "general.alignment")
	_ = binary.Write(buf, le, uint32(TypeUint32))
	_ = binary.Write(buf, le, uint32(32))

	// KV 3: "expert_count" -> uint32(32)
	writeString(buf, "expert_count")
	_ = binary.Write(buf, le, uint32(TypeUint32))
	_ = binary.Write(buf, le, uint32(32))

	// Tensor 1: "blk.0.ffn_gate.weight"
	writeString(buf, "blk.0.ffn_gate.weight")
	_ = binary.Write(buf, le, uint32(2)) // 2 dimensions
	_ = binary.Write(buf, le, uint64(64))
	_ = binary.Write(buf, le, uint64(128))
	_ = binary.Write(buf, le, uint32(0))  // GGML_TYPE_F32
	_ = binary.Write(buf, le, uint64(0))  // offset 0 in tensor data section

	// Align to 32 bytes for data start
	pos := buf.Len()
	align := 32
	pad := (align - (pos % align)) % align
	buf.Write(make([]byte, pad))

	// Tensor data payload: 64 * 128 * 4 = 32768 bytes
	tensorBytes := make([]byte, 64*128*4)
	for i := range tensorBytes {
		tensorBytes[i] = byte(i % 256)
	}
	buf.Write(tensorBytes)

	return buf.Bytes()
}

func TestGGUFReader_OpenSynthetic(t *testing.T) {
	data := createSyntheticGGUF()
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "test.gguf")
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatalf("failed to write test file: %v", err)
	}

	g, err := Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer g.Close()

	if g.Version != 3 {
		t.Errorf("expected version 3, got %d", g.Version)
	}
	if len(g.Tensors) != 1 {
		t.Errorf("expected 1 tensor, got %d", len(g.Tensors))
	}

	arch, ok := g.Metadata["general.architecture"].(string)
	if !ok || arch != "gpt-oss" {
		t.Errorf("expected architecture 'gpt-oss', got %v", g.Metadata["general.architecture"])
	}

	align, ok := g.Metadata["general.alignment"].(uint32)
	if !ok || align != 32 {
		t.Errorf("expected alignment 32, got %v", g.Metadata["general.alignment"])
	}

	tensor, ok := g.GetTensor("blk.0.ffn_gate.weight")
	if !ok {
		t.Fatal("failed to find tensor blk.0.ffn_gate.weight")
	}
	if len(tensor.Dims) != 2 || tensor.Dims[0] != 64 || tensor.Dims[1] != 128 {
		t.Errorf("unexpected dims: %v", tensor.Dims)
	}

	raw, err := g.ReadTensorBytes(tensor)
	if err != nil {
		t.Fatalf("failed to read tensor bytes: %v", err)
	}
	if len(raw) != 64*128*4 {
		t.Errorf("expected %d bytes, got %d", 64*128*4, len(raw))
	}
	if raw[1] != 1 {
		t.Errorf("expected byte[1]=1, got %d", raw[1])
	}
}

func TestGGUFWriter_RoundTrip(t *testing.T) {
	w := NewWriter()
	w.SetMetadata("test.name", "StrataGoGGUF")
	w.SetMetadata("test.count", uint32(42))

	dummyData := []byte("HELLO_GGUF_DATA_BYTES")
	w.AddTensor("layer.0.weight", []uint64{uint64(len(dummyData))}, GGMLTypeF32, dummyData)

	tmpDir := t.TempDir()
	outPath := filepath.Join(tmpDir, "written.gguf")
	if err := w.WriteFile(outPath); err != nil {
		t.Fatalf("WriteFile failed: %v", err)
	}

	r, err := Open(outPath)
	if err != nil {
		t.Fatalf("Open written file failed: %v", err)
	}
	defer r.Close()

	if r.Metadata["test.name"] != "StrataGoGGUF" {
		t.Errorf("unexpected name: %v", r.Metadata["test.name"])
	}
	if r.Metadata["test.count"] != uint32(42) {
		t.Errorf("unexpected count: %v", r.Metadata["test.count"])
	}

	ti, ok := r.GetTensor("layer.0.weight")
	if !ok {
		t.Fatal("tensor not found")
	}
	data, err := r.ReadTensorBytes(ti)
	if err != nil {
		t.Fatalf("ReadTensorBytes error: %v", err)
	}
	if string(data[:len(dummyData)]) != string(dummyData) {
		t.Errorf("expected data %s, got %s", dummyData, data)
	}
}

