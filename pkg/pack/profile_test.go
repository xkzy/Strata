package pack

import (
	"path/filepath"
	"testing"
)

func TestProfileRoundTrip(t *testing.T) {
	tmpDir := t.TempDir()
	outPath := filepath.Join(tmpDir, "test-profile.bin")

	nl := 4
	ne := 8

	basePairs := []ExpertPair{
		{Layer: 0, Expert: 1},
		{Layer: 1, Expert: 3},
		{Layer: 2, Expert: 5},
	}

	traceFreq := map[ExpertPair]int{
		{Layer: 3, Expert: 2}: 10,
		{Layer: 1, Expert: 3}: 5,
		{Layer: 0, Expert: 7}: 2,
	}

	ranked, counts := RankProfile(basePairs, traceFreq, nl, ne, false, true)
	if len(ranked) != nl*ne {
		t.Fatalf("expected %d ranked pairs, got %d", nl*ne, len(ranked))
	}
	if counts["trace"] == 0 || counts["base"] == 0 || counts["fill"] == 0 {
		t.Errorf("unexpected counts: %+v", counts)
	}

	if err := WriteProfile(outPath, ranked, nl, ne); err != nil {
		t.Fatalf("failed to write profile: %v", err)
	}

	readBack, err := ReadProfile(outPath, nl, ne)
	if err != nil {
		t.Fatalf("failed to read profile back: %v", err)
	}

	if len(readBack) != len(ranked) {
		t.Fatalf("length mismatch: got %d, want %d", len(readBack), len(ranked))
	}

	for i := range ranked {
		if readBack[i] != ranked[i] {
			t.Fatalf("pair mismatch at %d: got %+v, want %+v", i, readBack[i], ranked[i])
		}
	}
}
