package pack

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"sort"
)

const (
	ProfileMagic   = "STRP"
	ProfileVersion = 1
	DefaultNLayers = 48
	DefaultNExpert = 512
)

// ExpertPair represents a (layer, expert) index pair.
type ExpertPair struct {
	Layer  uint16
	Expert uint16
}

// ProfileOptions specifies parameters for building an expert cache profile.
type ProfileOptions struct {
	BaseProfilePath string
	NoBase          bool
	Reorder         bool
	OutPath         string
	NLayers         int
	NExperts        int
	TracePaths      []string
}

// ReadProfile reads a binary STRP profile file.
func ReadProfile(path string, nLayers, nExperts int) ([]ExpertPair, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read profile file: %w", err)
	}
	if len(data) < 24 {
		return nil, fmt.Errorf("profile too short (<24 bytes)")
	}
	if string(data[:4]) != ProfileMagic {
		return nil, fmt.Errorf("invalid profile magic %q", string(data[:4]))
	}

	reader := bytes.NewReader(data[4:])
	var version, nl, ne, slots, n uint32
	if err := binary.Read(reader, binary.LittleEndian, &version); err != nil {
		return nil, err
	}
	if err := binary.Read(reader, binary.LittleEndian, &nl); err != nil {
		return nil, err
	}
	if err := binary.Read(reader, binary.LittleEndian, &ne); err != nil {
		return nil, err
	}
	if err := binary.Read(reader, binary.LittleEndian, &slots); err != nil {
		return nil, err
	}
	if err := binary.Read(reader, binary.LittleEndian, &n); err != nil {
		return nil, err
	}

	if int(nl) != nLayers || int(ne) != nExperts {
		return nil, fmt.Errorf("profile geometry mismatch: got %dx%d, want %dx%d", nl, ne, nLayers, nExperts)
	}

	pairs := make([]ExpertPair, n)
	for i := uint32(0); i < n; i++ {
		var p ExpertPair
		if err := binary.Read(reader, binary.LittleEndian, &p.Layer); err != nil {
			return nil, err
		}
		if err := binary.Read(reader, binary.LittleEndian, &p.Expert); err != nil {
			return nil, err
		}
		pairs[i] = p
	}
	return pairs, nil
}

// ReadTrace reads a routing trace file produced by --dump-routing.
func ReadTrace(path string, nLayers, nExperts int) (map[ExpertPair]int, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read trace file: %w", err)
	}

	freq := make(map[ExpertPair]int)
	reader := bytes.NewReader(data)

	for reader.Len() >= 8 {
		var layer, k int32
		if err := binary.Read(reader, binary.LittleEndian, &layer); err != nil {
			break
		}
		if err := binary.Read(reader, binary.LittleEndian, &k); err != nil {
			break
		}
		if k <= 0 || int(k)*8 > reader.Len() {
			break
		}

		for i := int32(0); i < k; i++ {
			var expertID int32
			if err := binary.Read(reader, binary.LittleEndian, &expertID); err != nil {
				break
			}
			if int(layer) >= 0 && int(layer) < nLayers && int(expertID) >= 0 && int(expertID) < nExperts {
				pair := ExpertPair{Layer: uint16(layer), Expert: uint16(expertID)}
				freq[pair]++
			}
		}

		// Skip weights (k * float32)
		if _, err := reader.Seek(int64(k*4), io.SeekCurrent); err != nil {
			break
		}
	}

	return freq, nil
}

// RankProfile combines base ranking, trace frequencies, and full layer fill.
func RankProfile(basePairs []ExpertPair, traceFreq map[ExpertPair]int, nLayers, nExperts int, noBase, reorder bool) ([]ExpertPair, map[string]int) {
	ranked := make([]ExpertPair, 0, nLayers*nExperts)
	seen := make(map[ExpertPair]bool)
	counts := map[string]int{"base": 0, "trace": 0, "fill": 0}

	take := func(pairs []ExpertPair, key string) {
		for _, p := range pairs {
			if !seen[p] {
				seen[p] = true
				ranked = append(ranked, p)
				counts[key]++
			}
		}
	}

	// Sort traces by frequency descending, then layer/expert ascending
	type traceEntry struct {
		pair ExpertPair
		freq int
	}
	traceList := make([]traceEntry, 0, len(traceFreq))
	for p, f := range traceFreq {
		traceList = append(traceList, traceEntry{pair: p, freq: f})
	}
	sort.Slice(traceList, func(i, j int) bool {
		if traceList[i].freq != traceList[j].freq {
			return traceList[i].freq > traceList[j].freq
		}
		if traceList[i].pair.Layer != traceList[j].pair.Layer {
			return traceList[i].pair.Layer < traceList[j].pair.Layer
		}
		return traceList[i].pair.Expert < traceList[j].pair.Expert
	})

	tracePairs := make([]ExpertPair, len(traceList))
	for i, te := range traceList {
		tracePairs[i] = te.pair
	}

	if noBase || reorder {
		take(tracePairs, "trace")
	}
	if !noBase {
		take(basePairs, "base")
	}
	if !noBase && !reorder {
		take(tracePairs, "trace")
	}

	// Fill remaining
	fillPairs := make([]ExpertPair, 0, nLayers*nExperts)
	for e := 0; e < nExperts; e++ {
		for l := 0; l < nLayers; l++ {
			fillPairs = append(fillPairs, ExpertPair{Layer: uint16(l), Expert: uint16(e)})
		}
	}
	take(fillPairs, "fill")

	return ranked, counts
}

// WriteProfile writes the binary STRP profile file.
func WriteProfile(path string, ranked []ExpertPair, nLayers, nExperts int) error {
	buf := new(bytes.Buffer)
	buf.WriteString(ProfileMagic)

	header := []uint32{
		ProfileVersion,
		uint32(nLayers),
		uint32(nExperts),
		uint32(len(ranked)),
		uint32(len(ranked)),
	}
	if err := binary.Write(buf, binary.LittleEndian, header); err != nil {
		return err
	}

	for _, p := range ranked {
		if err := binary.Write(buf, binary.LittleEndian, p.Layer); err != nil {
			return err
		}
		if err := binary.Write(buf, binary.LittleEndian, p.Expert); err != nil {
			return err
		}
	}

	// Lookup table
	table := make([][]int32, nLayers)
	for l := 0; l < nLayers; l++ {
		table[l] = make([]int32, nExperts)
		for e := 0; e < nExperts; e++ {
			table[l][e] = -1
		}
	}
	for slot, p := range ranked {
		if int(p.Layer) < nLayers && int(p.Expert) < nExperts {
			table[p.Layer][p.Expert] = int32(slot)
		}
	}

	for l := 0; l < nLayers; l++ {
		if err := binary.Write(buf, binary.LittleEndian, table[l]); err != nil {
			return err
		}
	}

	return os.WriteFile(path, buf.Bytes(), 0644)
}
