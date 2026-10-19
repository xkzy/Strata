package pack

import (
	"fmt"
	"io"
	"strings"
)

// IndexKind specifies tensor layout transformation in index.txt
type IndexKind int

const (
	IndexKindVerbatim   IndexKind = 0 // Copy verbatim
	IndexKindBF16InF32  IndexKind = 1 // High 16 bits of promoted float32
	IndexKindF32Copy    IndexKind = 2 // Direct 4-byte copy
	IndexKindF16InF32   IndexKind = 3 // Convert F32 to F16 with nearest-even
)

// IndexEntry represents one flat line in index.txt
type IndexEntry struct {
	Name         string
	FileID       int    // 0: dense.bin, 1: embd.bin, 2: experts.bin
	Kind         IndexKind
	SrcOffset    uint64
	SrcBytes     uint64
	DstOffset    uint64
	DstBytes     uint64
	Ne0          uint64
	Ne1          uint64
	CodeBits     int
	CodeBias     int
	GroupElems   int
	Codebook     int
	HasOffset    int
	CodesBytes   uint64
	ScalesBytes  uint64
	OffsetBytes  uint64
	ScalesFP16   int // 1 if fp16 scales (widened to f32), 0 if f32
}

// FormatLine formats the index entry as a whitespace-separated line
func (e IndexEntry) FormatLine() string {
	return fmt.Sprintf("%s %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",
		e.Name, e.FileID, int(e.Kind),
		e.SrcOffset, e.SrcBytes, e.DstOffset, e.DstBytes,
		e.Ne0, e.Ne1,
		e.CodeBits, e.CodeBias, e.GroupElems, e.Codebook,
		e.HasOffset, e.CodesBytes, e.ScalesBytes, e.OffsetBytes, e.ScalesFP16,
	)
}

// ParseIndexLine parses a single line of index.txt
func ParseIndexLine(line string) (*IndexEntry, error) {
	parts := strings.Fields(line)
	if len(parts) < 18 {
		return nil, fmt.Errorf("invalid index line: expected 18 columns, got %d", len(parts))
	}

	var e IndexEntry
	var kind int
	e.Name = parts[0]
	n, err := fmt.Sscanf(strings.Join(parts[1:], " "), "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d",
		&e.FileID, &kind,
		&e.SrcOffset, &e.SrcBytes, &e.DstOffset, &e.DstBytes,
		&e.Ne0, &e.Ne1,
		&e.CodeBits, &e.CodeBias, &e.GroupElems, &e.Codebook,
		&e.HasOffset, &e.CodesBytes, &e.ScalesBytes, &e.OffsetBytes, &e.ScalesFP16,
	)
	if err != nil || n < 17 {
		return nil, fmt.Errorf("failed to parse index fields: %w", err)
	}
	e.Kind = IndexKind(kind)
	return &e, nil
}

// WriteIndexTable writes a collection of IndexEntries to a writer
func WriteIndexTable(w io.Writer, entries []IndexEntry) error {
	for _, e := range entries {
		if _, err := io.WriteString(w, e.FormatLine()); err != nil {
			return err
		}
	}
	return nil
}
