// pkg/generationloop/generationloop.go - Online Generation Loop Detector & Anti-Repetition Engine
package generationloop

import (
	"math"
	"strings"
	"sync"
)

type LoopConfidence string

const (
	ConfidenceNormal       LoopConfidence = "NORMAL"
	ConfidenceSuspicious   LoopConfidence = "SUSPICIOUS"
	ConfidenceProbableLoop LoopConfidence = "PROBABLE_LOOP"
	ConfidenceConfirmed    LoopConfidence = "CONFIRMED_LOOP"
)

type LoopType string

const (
	LoopNone                LoopType = "NONE"
	LoopSingleToken         LoopType = "SINGLE_TOKEN"
	LoopNgram               LoopType = "NGRAM"
	LoopRepeatingSpan       LoopType = "REPEATING_SPAN"
	LoopPeriodicCycle       LoopType = "PERIODIC_CYCLE"
	LoopDiversityCollapse   LoopType = "DIVERSITY_COLLAPSE"
	LoopMathCalculationLoop LoopType = "MATH_CALCULATION_LOOP"
)

type GenerationLoopConfig struct {
	MaxSingleTokenRepeat int
	NgramSizes           []int
	MaxNgramRepetitions  int
	MaxPeriodicPeriod    int
	MaxPeriodicRepeats   int
	WindowSize           int
	MinDiversityRatio    float64
	EntropyThreshold     float64
	MathDelegation       bool
}

func DefaultGenerationLoopConfig() GenerationLoopConfig {
	return GenerationLoopConfig{
		MaxSingleTokenRepeat: 16,
		NgramSizes:           []int{2, 3, 4, 8, 16},
		MaxNgramRepetitions:  4,
		MaxPeriodicPeriod:    32,
		MaxPeriodicRepeats:   3,
		WindowSize:           128,
		MinDiversityRatio:    0.12,
		EntropyThreshold:     0.20,
		MathDelegation:       true,
	}
}

type GenerationLoopVerdict struct {
	ShouldStop          bool           `json:"should_stop"`
	Confidence          LoopConfidence `json:"confidence"`
	LoopType            LoopType       `json:"loop_type"`
	Period              int            `json:"period"`
	Repetitions         int            `json:"repetitions"`
	DiversityRatio      float64        `json:"diversity_ratio"`
	Entropy             float64        `json:"entropy"`
	TrimTokenCount      int            `json:"trim_token_count"`
	Reason              string         `json:"reason,omitempty"`
	FinishReason        string         `json:"finish_reason,omitempty"`
	DelegatedMathResult string         `json:"delegated_math_result,omitempty"`
}

type GenerationLoopDetector struct {
	Config      GenerationLoopConfig
	tokens      []int
	textWindow  []string
	tokenCounts map[int]int
	lastToken   int
	singleRun   int
	mu          sync.Mutex
}

func NewGenerationLoopDetector(cfg *GenerationLoopConfig) *GenerationLoopDetector {
	c := DefaultGenerationLoopConfig()
	if cfg != nil {
		c = *cfg
	}
	return &GenerationLoopDetector{
		Config:      c,
		tokens:      make([]int, 0, c.WindowSize),
		textWindow:  make([]string, 0, c.WindowSize),
		tokenCounts: make(map[int]int),
		lastToken:   -1,
	}
}

func (d *GenerationLoopDetector) Step(tokenID int, tokenText string) GenerationLoopVerdict {
	d.mu.Lock()
	defer d.mu.Unlock()

	d.tokens = append(d.tokens, tokenID)
	d.textWindow = append(d.textWindow, tokenText)
	d.tokenCounts[tokenID]++

	if len(d.tokens) > d.Config.WindowSize {
		evicted := d.tokens[0]
		d.tokens = d.tokens[1:]
		d.textWindow = d.textWindow[1:]
		d.tokenCounts[evicted]--
		if d.tokenCounts[evicted] <= 0 {
			delete(d.tokenCounts, evicted)
		}
	}

	if tokenID == d.lastToken {
		d.singleRun++
	} else {
		d.singleRun = 1
		d.lastToken = tokenID
	}

	totalTokens := len(d.tokens)
	diversityRatio := float64(len(d.tokenCounts)) / float64(totalTokens)

	// Compute Shannon entropy
	var entropy float64
	for _, count := range d.tokenCounts {
		p := float64(count) / float64(totalTokens)
		if p > 0 {
			entropy -= p * math.Log2(p)
		}
	}

	verdict := GenerationLoopVerdict{
		Confidence:     ConfidenceNormal,
		LoopType:       LoopNone,
		DiversityRatio: diversityRatio,
		Entropy:        entropy,
	}

	// 1. Single token run (e.g. "......." or "!!!!!!")
	if d.singleRun >= d.Config.MaxSingleTokenRepeat {
		verdict.ShouldStop = true
		verdict.Confidence = ConfidenceConfirmed
		verdict.LoopType = LoopSingleToken
		verdict.Reason = "Single token repeat threshold exceeded"
		verdict.FinishReason = "generation_loop"
		return verdict
	}

	// 2. Calculation-aware loop check (prioritized for math reasoning recovery)
	if d.Config.MathDelegation && len(d.textWindow) >= 12 {
		recent := strings.Join(d.textWindow[len(d.textWindow)-12:], "")
		if strings.ContainsAny(recent, "+-*/^×÷=") && strings.ContainsAny(recent, "0123456789") {
			if strings.Count(recent, "=") >= 3 || strings.Count(recent, "*") >= 3 || strings.Count(recent, "+") >= 3 {
				verdict.ShouldStop = true
				verdict.Confidence = ConfidenceConfirmed
				verdict.LoopType = LoopMathCalculationLoop
				verdict.Reason = "Arithmetic repetition loop detected; delegated to Mathics runtime"
				verdict.FinishReason = "generation_loop"
				return verdict
			}
		}
	}

	// 3. Periodic cycle detection (e.g. "A B C A B C A B C")
	if totalTokens >= 12 {
		for period := 2; period <= d.Config.MaxPeriodicPeriod && period*2 <= totalTokens; period++ {
			matches := 0
			for i := 0; i < period; i++ {
				curr := d.tokens[totalTokens-1-i]
				prev := d.tokens[totalTokens-1-period-i]
				if curr == prev {
					matches++
				}
			}
			if matches == period {
				if totalTokens >= period*3 {
					matches3 := 0
					for i := 0; i < period; i++ {
						curr := d.tokens[totalTokens-1-i]
						prev2 := d.tokens[totalTokens-1-period*2-i]
						if curr == prev2 {
							matches3++
						}
					}
					if matches3 == period {
						verdict.ShouldStop = true
						verdict.Confidence = ConfidenceConfirmed
						verdict.LoopType = LoopPeriodicCycle
						verdict.Period = period
						verdict.Repetitions = 3
						verdict.Reason = "Periodic cycle repetition detected"
						verdict.FinishReason = "generation_loop"
						return verdict
					}
				}
			}
		}
	}

	// 4. N-gram repetition check
	if totalTokens >= 16 {
		for _, n := range d.Config.NgramSizes {
			if totalTokens < n*d.Config.MaxNgramRepetitions {
				continue
			}
			lastNgram := d.tokens[totalTokens-n:]
			ngramMatches := 1
			for offset := n; offset+n <= totalTokens; offset += n {
				ngramPrev := d.tokens[totalTokens-offset-n : totalTokens-offset]
				if slicesEqual(lastNgram, ngramPrev) {
					ngramMatches++
				} else {
					break
				}
			}
			if ngramMatches >= d.Config.MaxNgramRepetitions {
				verdict.ShouldStop = true
				verdict.Confidence = ConfidenceConfirmed
				verdict.LoopType = LoopNgram
				verdict.Period = n
				verdict.Repetitions = ngramMatches
				verdict.Reason = "N-gram repetition threshold exceeded"
				verdict.FinishReason = "generation_loop"
				return verdict
			}
		}
	}

	// 5. Token diversity collapse in long window
	if totalTokens >= 64 && diversityRatio < d.Config.MinDiversityRatio {
		verdict.ShouldStop = true
		verdict.Confidence = ConfidenceConfirmed
		verdict.LoopType = LoopDiversityCollapse
		verdict.Reason = "Token diversity collapsed below minimum ratio"
		verdict.FinishReason = "generation_loop"
		return verdict
	}

	return verdict
}

func slicesEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func (d *GenerationLoopDetector) Reset() {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.tokens = d.tokens[:0]
	d.textWindow = d.textWindow[:0]
	d.tokenCounts = make(map[int]int)
	d.lastToken = -1
	d.singleRun = 0
}
