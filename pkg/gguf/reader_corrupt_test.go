package gguf

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

// A header whose key length claims 2^40 bytes must give an error, not a huge allocation.
func TestOpenRejectsLyingLengths(t *testing.T) {
	for _, n := range []uint64{1 << 40, 1 << 20} { // over the cap; under it but past the end of the file
		b := binary.LittleEndian.AppendUint32(nil, GGUFMagic)
		b = binary.LittleEndian.AppendUint32(b, 3)
		b = binary.LittleEndian.AppendUint64(b, 0) // tensors
		b = binary.LittleEndian.AppendUint64(b, 1) // kvs
		b = binary.LittleEndian.AppendUint64(b, n) // key length
		p := filepath.Join(t.TempDir(), "bad.gguf")
		os.WriteFile(p, b, 0o600)
		if _, err := Open(p); err == nil {
			t.Fatalf("length %d: want an error", n)
		}
	}
}
