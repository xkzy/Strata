package generationloop

import (
	"testing"
)

func TestGenerationLoopDetector(t *testing.T) {
	detector := NewGenerationLoopDetector(nil)

	// Normal tokens
	for i := 1; i <= 10; i++ {
		v := detector.Step(i, "word ")
		if v.ShouldStop {
			t.Errorf("unexpected stop on normal sequence")
		}
	}

	// Single token repetition loop
	var finalV GenerationLoopVerdict
	for i := 0; i < 35; i++ {
		finalV = detector.Step(999, "...")
	}
	if !finalV.ShouldStop || finalV.LoopType != LoopSingleToken {
		t.Errorf("expected single token loop stop, got %+v", finalV)
	}

	// Reset
	detector.Reset()

	// Arithmetic loop recovery
	var arithV GenerationLoopVerdict
	for i := 0; i < 20; i++ {
		arithV = detector.Step(100+i%3, "12345 * 67890 = ")
	}
	if !arithV.ShouldStop || arithV.LoopType != LoopMathCalculationLoop {
		t.Errorf("expected math calculation loop detection, got %+v", arithV)
	}
}
