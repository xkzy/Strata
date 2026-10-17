package tune

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"strings"
)

// A random search over the engine's numeric options, after the placement is chosen: from the best setting so far,
// try a random nearby one, keep it when it is faster. The size of the jump shrinks every round (a decaying learning
// rate), and a slower setting is still taken now and then while the "temperature" is high (simulated annealing), so
// the search can leave a local peak early and settles down late. The functions are pure; the caller measures.

// Knob is one numeric engine option the search may change.
type Knob struct {
	Flag    string
	Lo, Hi  float64
	Whole   bool
	Default float64 // used when the run config does not set the flag
}

// DefaultKnobs are the options that moved the speed in measurements: CPU worker threads, the share of offloaded experts
// sent over PCIe, and the draft length of speculative decoding.
func DefaultKnobs() []Knob {
	return []Knob{
		{Flag: "--pool-workers", Lo: 4, Hi: 32, Whole: true, Default: 16},
		{Flag: "--pcie-frac", Lo: 0, Hi: 1, Default: 0.5},
		{Flag: "--spec", Lo: 1, Hi: 8, Whole: true, Default: 4},
	}
}

// Decay is value * rate^round: the step size and the temperature both shrink this way.
func Decay(value, rate float64, round int) float64 { return value * math.Pow(rate, float64(round)) }

// Score ranks a result: decode speed first, prompt speed second (weighted geometric mean, 0.8 and 0.2).
func Score(r Result) float64 {
	if r.Err != "" || r.DecodeTokS <= 0 || r.PrefillTokS <= 0 {
		return 0
	}
	return math.Pow(r.DecodeTokS, 0.8) * math.Pow(r.PrefillTokS, 0.2)
}

// Propose changes a random non-empty subset of the knobs in args by a Gaussian jump of step*(Hi-Lo), clamped to the
// range. The caller can compare Settings of the result to skip a setting it already measured.
func Propose(rng *rand.Rand, args []string, knobs []Knob, step float64) []string {
	out := append([]string(nil), args...)
	chosen := make([]bool, len(knobs))
	any := false
	for i := range knobs {
		if rng.Float64() < 0.5 {
			chosen[i], any = true, true
		}
	}
	if !any {
		chosen[rng.Intn(len(knobs))] = true
	}
	for i, k := range knobs {
		cur := k.Default
		if v, err := strconv.ParseFloat(lookup(out, k.Flag), 64); err == nil {
			cur = v
		}
		if chosen[i] {
			cur += rng.NormFloat64() * step * (k.Hi - k.Lo)
			if k.Whole {
				cur = math.Round(cur)
			} else {
				cur = math.Round(cur*20) / 20 // 0.05 steps: finer differences are lost in the measurement noise
			}
			cur = math.Min(k.Hi, math.Max(k.Lo, cur))
		}
		s := strconv.FormatFloat(cur, 'f', -1, 64)
		out = SetArg(out, k.Flag, &s)
	}
	return out
}

// Accept decides whether the search moves to a measured setting: always when it scores higher, else with probability
// exp(-relative loss / temperature).
func Accept(rng *rand.Rand, cur, got, temperature float64) bool {
	if got >= cur {
		return true
	}
	if temperature <= 0 || cur <= 0 {
		return false
	}
	return rng.Float64() < math.Exp(-(cur-got)/cur/temperature)
}

// ArgsOf is the "args" of a candidate's run config.
func ArgsOf(c Candidate) []string { return argsOf(c.Config) }

// WithArgs is a copy of c with args as its "args" and a new name; c is not changed.
func WithArgs(c Candidate, name string, args []string) Candidate {
	b, _ := json.Marshal(c.Config)
	cfg := map[string]interface{}{}
	_ = json.Unmarshal(b, &cfg)
	withArgs(cfg, args)
	return Candidate{Name: name, Config: cfg}
}

// Settings lists the knob values of a candidate, for printing.
func Settings(c Candidate, knobs []Knob) string {
	var parts []string
	for _, k := range knobs {
		v := lookup(argsOf(c.Config), k.Flag)
		if v == "" {
			v = fmt.Sprint(k.Default) + " (default)"
		}
		parts = append(parts, k.Flag+" "+v)
	}
	return strings.Join(parts, ", ")
}
