package gguf

// GGUF metadata value types
const (
	TypeUint8   uint32 = 0
	TypeInt8    uint32 = 1
	TypeUint16  uint32 = 2
	TypeInt16   uint32 = 3
	TypeUint32  uint32 = 4
	TypeInt32   uint32 = 5
	TypeFloat32 uint32 = 6
	TypeBool    uint32 = 7
	TypeString  uint32 = 8
	TypeArray   uint32 = 9
	TypeUint64  uint32 = 10
	TypeInt64   uint32 = 11
	TypeFloat64 uint32 = 12
)

// GGML tensor data types
const (
	GGMLTypeF32     uint32 = 0
	GGMLTypeF16     uint32 = 1
	GGMLTypeQ4_0    uint32 = 2
	GGMLTypeQ4_1    uint32 = 3
	GGMLTypeQ5_0    uint32 = 6
	GGMLTypeQ5_1    uint32 = 7
	GGMLTypeQ8_0    uint32 = 8
	GGMLTypeQ8_1    uint32 = 9
	GGMLTypeQ2_K    uint32 = 10
	GGMLTypeQ3_K    uint32 = 11
	GGMLTypeQ4_K    uint32 = 12
	GGMLTypeQ5_K    uint32 = 13
	GGMLTypeQ6_K    uint32 = 14
	GGMLTypeQ8_K    uint32 = 15
	GGMLTypeIQ2_XXS uint32 = 16
	GGMLTypeIQ2_XS  uint32 = 17
	GGMLTypeIQ3_XXS uint32 = 18
	GGMLTypeIQ1_S   uint32 = 19
	GGMLTypeIQ4_NL  uint32 = 20
	GGMLTypeIQ3_S   uint32 = 21
	GGMLTypeIQ2_S   uint32 = 22
	GGMLTypeIQ4_XS  uint32 = 23
	GGMLTypeI8      uint32 = 24
	GGMLTypeI16     uint32 = 25
	GGMLTypeI32     uint32 = 26
	GGMLTypeI64     uint32 = 27
	GGMLTypeF64     uint32 = 28
	GGMLTypeIQ1_M   uint32 = 29
	GGMLTypeBF16    uint32 = 30
	GGMLTypeMXFP4   uint32 = 39
)

// TensorInfo represents the metadata of a tensor inside a GGUF file
type TensorInfo struct {
	Name   string
	Dims   []uint64
	Type   uint32
	Offset uint64 // offset relative to DataStart
}

// ByteSize calculates the total expected byte size for this tensor
func (t *TensorInfo) ByteSize() uint64 {
	elements := uint64(1)
	for _, d := range t.Dims {
		elements *= d
	}
	switch t.Type {
	case GGMLTypeF32:
		return elements * 4
	case GGMLTypeF16, GGMLTypeBF16:
		return elements * 2
	case GGMLTypeF64:
		return elements * 8
	case GGMLTypeQ4_0, GGMLTypeQ4_1:
		return (elements / 32) * 18
	case GGMLTypeQ8_0:
		return (elements / 32) * 34
	case GGMLTypeQ4_K:
		return (elements / 256) * 144
	case GGMLTypeQ2_K:
		return (elements / 256) * 82
	case GGMLTypeQ3_K:
		return (elements / 256) * 110
	case GGMLTypeQ5_K:
		return (elements / 256) * 176
	case GGMLTypeQ6_K:
		return (elements / 256) * 210
	case GGMLTypeIQ1_M:
		return (elements / 256) * 56
	case GGMLTypeIQ3_XXS:
		return (elements / 256) * 98
	case GGMLTypeIQ2_XS:
		return (elements / 256) * 74
	case GGMLTypeIQ4_XS:
		return (elements / 256) * 136
	default:
		// Fallback conservative 2 bytes per element
		return elements * 2
	}
}
