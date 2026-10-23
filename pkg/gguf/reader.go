package gguf

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

const (
	GGUFMagic = 0x46554747 // "GGUF" in little-endian
)

// GGUFFile represents an opened GGUF file with its metadata and tensor catalogue
type GGUFFile struct {
	file      *os.File
	Version   uint32
	Alignment uint32
	Metadata  map[string]any
	Tensors   []*TensorInfo
	tensorMap map[string]*TensorInfo
	DataStart int64
}

// Open opens and parses the GGUF file header, metadata, and tensor directory
func Open(path string) (*GGUFFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	gf := &GGUFFile{
		file:      f,
		Metadata:  make(map[string]any),
		Tensors:   make([]*TensorInfo, 0),
		tensorMap: make(map[string]*TensorInfo),
		Alignment: 32, // default
	}

	if err := gf.parse(f); err != nil {
		f.Close()
		return nil, err
	}

	return gf, nil
}

// Close closes the underlying GGUF file
func (g *GGUFFile) Close() error {
	if g.file != nil {
		return g.file.Close()
	}
	return nil
}

// GetTensor looks up a tensor by its exact name
func (g *GGUFFile) GetTensor(name string) (*TensorInfo, bool) {
	t, ok := g.tensorMap[name]
	return t, ok
}

// ReadTensorBytes reads the binary tensor payload into memory
func (g *GGUFFile) ReadTensorBytes(t *TensorInfo) ([]byte, error) {
	absOffset := g.DataStart + int64(t.Offset)
	size := t.ByteSize()
	buf := make([]byte, size)

	_, err := g.file.ReadAt(buf, absOffset)
	if err != nil && err != io.EOF {
		return nil, fmt.Errorf("read tensor %s at offset %d: %w", t.Name, absOffset, err)
	}
	return buf, nil
}

// parse reads header, metadata key-values, and tensor descriptors
func (g *GGUFFile) parse(r io.ReadSeeker) error {
	var magic uint32
	if err := binary.Read(r, binary.LittleEndian, &magic); err != nil {
		return fmt.Errorf("read magic: %w", err)
	}
	if magic != GGUFMagic {
		return fmt.Errorf("invalid GGUF magic: 0x%x (expected 0x%x)", magic, GGUFMagic)
	}

	if err := binary.Read(r, binary.LittleEndian, &g.Version); err != nil {
		return fmt.Errorf("read version: %w", err)
	}
	if g.Version < 2 || g.Version > 3 {
		return fmt.Errorf("unsupported GGUF version: %d", g.Version)
	}

	var tensorCount uint64
	var kvCount uint64
	if err := binary.Read(r, binary.LittleEndian, &tensorCount); err != nil {
		return fmt.Errorf("read tensor count: %w", err)
	}
	if err := binary.Read(r, binary.LittleEndian, &kvCount); err != nil {
		return fmt.Errorf("read kv count: %w", err)
	}

	// 1. Read metadata key-values
	for i := uint64(0); i < kvCount; i++ {
		key, err := readString(r)
		if err != nil {
			return fmt.Errorf("read kv key %d: %w", i, err)
		}
		var valType uint32
		if err := binary.Read(r, binary.LittleEndian, &valType); err != nil {
			return fmt.Errorf("read kv %s valType: %w", key, err)
		}
		val, err := readValue(r, valType)
		if err != nil {
			return fmt.Errorf("read kv %s value: %w", key, err)
		}
		g.Metadata[key] = val
	}

	// Check alignment
	if al, ok := g.Metadata["general.alignment"].(uint32); ok && al > 0 {
		g.Alignment = al
	}

	// 2. Read tensor descriptors
	for i := uint64(0); i < tensorCount; i++ {
		name, err := readString(r)
		if err != nil {
			return fmt.Errorf("read tensor %d name: %w", i, err)
		}
		var nDims uint32
		if err := binary.Read(r, binary.LittleEndian, &nDims); err != nil {
			return fmt.Errorf("read tensor %s nDims: %w", name, err)
		}
		if nDims > maxDims {
			return fmt.Errorf("read tensor %s: %d dimensions is not plausible", name, nDims)
		}
		dims := make([]uint64, nDims)
		for d := uint32(0); d < nDims; d++ {
			if err := binary.Read(r, binary.LittleEndian, &dims[d]); err != nil {
				return fmt.Errorf("read tensor %s dim %d: %w", name, d, err)
			}
		}
		var tType uint32
		if err := binary.Read(r, binary.LittleEndian, &tType); err != nil {
			return fmt.Errorf("read tensor %s type: %w", name, err)
		}
		var offset uint64
		if err := binary.Read(r, binary.LittleEndian, &offset); err != nil {
			return fmt.Errorf("read tensor %s offset: %w", name, err)
		}

		ti := &TensorInfo{
			Name:   name,
			Dims:   dims,
			Type:   tType,
			Offset: offset,
		}
		g.Tensors = append(g.Tensors, ti)
		g.tensorMap[name] = ti
	}

	// 3. Align to DataStart
	currentPos, err := r.Seek(0, io.SeekCurrent)
	if err != nil {
		return fmt.Errorf("seek current position: %w", err)
	}

	align := int64(g.Alignment)
	pad := (align - (currentPos % align)) % align
	g.DataStart = currentPos + pad

	return nil
}

// Sizes come from the file; these caps turn a corrupt or hostile header into an error instead of a huge allocation.
const (
	maxString = 1 << 30
	maxDims   = 8
	maxArray  = 1 << 32
)

func readString(r io.Reader) (string, error) {
	var length uint64
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return "", err
	}
	if length > maxString {
		return "", fmt.Errorf("gguf: string of %d bytes is not plausible", length)
	}
	var buf bytes.Buffer // grows with the bytes actually present, so a lying length costs nothing
	if n, err := io.CopyN(&buf, r, int64(length)); err != nil {
		return "", fmt.Errorf("gguf: string cut short (%d of %d bytes): %w", n, length, err)
	}
	return buf.String(), nil
}

func readValue(r io.Reader, valType uint32) (any, error) {
	le := binary.LittleEndian
	switch valType {
	case TypeUint8:
		var v uint8
		err := binary.Read(r, le, &v)
		return v, err
	case TypeInt8:
		var v int8
		err := binary.Read(r, le, &v)
		return v, err
	case TypeUint16:
		var v uint16
		err := binary.Read(r, le, &v)
		return v, err
	case TypeInt16:
		var v int16
		err := binary.Read(r, le, &v)
		return v, err
	case TypeUint32:
		var v uint32
		err := binary.Read(r, le, &v)
		return v, err
	case TypeInt32:
		var v int32
		err := binary.Read(r, le, &v)
		return v, err
	case TypeFloat32:
		var v float32
		err := binary.Read(r, le, &v)
		return v, err
	case TypeBool:
		var b uint8
		err := binary.Read(r, le, &b)
		return b != 0, err
	case TypeString:
		return readString(r)
	case TypeUint64:
		var v uint64
		err := binary.Read(r, le, &v)
		return v, err
	case TypeInt64:
		var v int64
		err := binary.Read(r, le, &v)
		return v, err
	case TypeFloat64:
		var v float64
		err := binary.Read(r, le, &v)
		return v, err
	case TypeArray:
		var elemType uint32
		if err := binary.Read(r, le, &elemType); err != nil {
			return nil, err
		}
		var count uint64
		if err := binary.Read(r, le, &count); err != nil {
			return nil, err
		}
		if count > maxArray {
			return nil, fmt.Errorf("gguf: array of %d elements is not plausible", count)
		}
		arr := make([]any, 0, min(count, 1<<16)) // grown as elements are read, so a lying count costs nothing
		for i := uint64(0); i < count; i++ {
			elem, err := readValue(r, elemType)
			if err != nil {
				return nil, err
			}
			arr = append(arr, elem)
		}
		return arr, nil
	default:
		return nil, fmt.Errorf("unknown GGUF value type %d", valType)
	}
}
