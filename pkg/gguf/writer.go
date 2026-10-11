package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

type TensorEntry struct {
	Name string
	Dims []uint64
	Type uint32
	Data []byte
}

type GGUFWriter struct {
	Alignment uint32
	metadata  map[string]any
	tensors   []TensorEntry
}

func NewWriter() *GGUFWriter {
	return &GGUFWriter{
		Alignment: 32,
		metadata:  make(map[string]any),
		tensors:   make([]TensorEntry, 0),
	}
}

func (w *GGUFWriter) SetMetadata(key string, val any) {
	w.metadata[key] = val
}

func (w *GGUFWriter) AddTensor(name string, dims []uint64, tType uint32, data []byte) {
	w.tensors = append(w.tensors, TensorEntry{
		Name: name,
		Dims: dims,
		Type: tType,
		Data: data,
	})
}

func (w *GGUFWriter) WriteFile(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return w.Write(f)
}

func (w *GGUFWriter) Write(out io.WriteSeeker) error {
	le := binary.LittleEndian

	// 1. Magic & Version 3
	if err := binary.Write(out, le, uint32(GGUFMagic)); err != nil {
		return err
	}
	if err := binary.Write(out, le, uint32(3)); err != nil {
		return err
	}

	// 2. Counts
	if err := binary.Write(out, le, uint64(len(w.tensors))); err != nil {
		return err
	}
	if err := binary.Write(out, le, uint64(len(w.metadata))); err != nil {
		return err
	}

	// 3. Write metadata key-values
	for k, v := range w.metadata {
		if err := writeString(out, k); err != nil {
			return err
		}
		if err := writeValue(out, v); err != nil {
			return fmt.Errorf("write metadata %s: %w", k, err)
		}
	}

	// 4. Calculate tensor offsets
	currentOffset := uint64(0)
	tensorOffsets := make([]uint64, len(w.tensors))
	for i, t := range w.tensors {
		tensorOffsets[i] = currentOffset
		pad := (uint64(w.Alignment) - (uint64(len(t.Data)) % uint64(w.Alignment))) % uint64(w.Alignment)
		currentOffset += uint64(len(t.Data)) + pad
	}

	// 5. Write tensor descriptors
	for i, t := range w.tensors {
		if err := writeString(out, t.Name); err != nil {
			return err
		}
		if err := binary.Write(out, le, uint32(len(t.Dims))); err != nil {
			return err
		}
		for _, d := range t.Dims {
			if err := binary.Write(out, le, d); err != nil {
				return err
			}
		}
		if err := binary.Write(out, le, t.Type); err != nil {
			return err
		}
		if err := binary.Write(out, le, tensorOffsets[i]); err != nil {
			return err
		}
	}

	// 6. Align to DataStart
	pos, err := out.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	align := int64(w.Alignment)
	pad := (align - (pos % align)) % align
	if pad > 0 {
		if _, err := out.Write(make([]byte, pad)); err != nil {
			return err
		}
	}

	// 7. Write tensor data payloads with alignment padding
	for _, t := range w.tensors {
		if _, err := out.Write(t.Data); err != nil {
			return err
		}
		tPad := (int64(w.Alignment) - (int64(len(t.Data)) % int64(w.Alignment))) % int64(w.Alignment)
		if tPad > 0 {
			if _, err := out.Write(make([]byte, tPad)); err != nil {
				return err
			}
		}
	}

	return nil
}

func writeString(w io.Writer, s string) error {
	if err := binary.Write(w, binary.LittleEndian, uint64(len(s))); err != nil {
		return err
	}
	_, err := io.WriteString(w, s)
	return err
}

func writeValue(w io.Writer, val any) error {
	le := binary.LittleEndian
	switch v := val.(type) {
	case uint8:
		if err := binary.Write(w, le, TypeUint8); err != nil {
			return err
		}
		return binary.Write(w, le, v)
	case int8:
		if err := binary.Write(w, le, TypeInt8); err != nil {
			return err
		}
		return binary.Write(w, le, v)
	case uint16:
		if err := binary.Write(w, le, TypeUint16); err != nil {
			return err
		}
		return binary.Write(w, le, v)
	case int16:
		if err := binary.Write(w, le, TypeInt16); err != nil {
			return err
		}
		return binary.Write(w, le, v)
	case uint32:
		if err := binary.Write(w, le, TypeUint32); err != nil {
			return err
		}
		return binary.Write(w, le, v)
	case int32:
		if err := binary.Write(w, le, TypeInt32); err != nil {
			return err
		}
		return binary.Write(w, le, v)
	case float32:
		if err := binary.Write(w, le, TypeFloat32); err != nil {
			return err
		}
		return binary.Write(w, le, v)
	case bool:
		if err := binary.Write(w, le, TypeBool); err != nil {
			return err
		}
		var b uint8
		if v {
			b = 1
		}
		return binary.Write(w, le, b)
	case string:
		if err := binary.Write(w, le, TypeString); err != nil {
			return err
		}
		return writeString(w, v)
	case uint64:
		if err := binary.Write(w, le, TypeUint64); err != nil {
			return err
		}
		return binary.Write(w, le, v)
	case int64:
		if err := binary.Write(w, le, TypeInt64); err != nil {
			return err
		}
		return binary.Write(w, le, v)
	case float64:
		if err := binary.Write(w, le, TypeFloat64); err != nil {
			return err
		}
		return binary.Write(w, le, v)
	default:
		return fmt.Errorf("unsupported value type: %T", val)
	}
}
