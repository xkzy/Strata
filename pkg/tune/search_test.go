package tune

import (
	"math/rand"
	"strconv"
	"strings"
	"testing"
)

func TestProposeStaysInRangeAndChangesSomething(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	knobs := DefaultKnobs()
	args := argsOf(base())
	changed := 0
	for i := 0; i < 300; i++ {
		got := Propose(rng, args, knobs, 0.3)
		for _, k := range knobs {
			if lookup(got, k.Flag) == "" {
				continue
			}
			v, err := strconv.ParseFloat(lookup(got, k.Flag), 64)
			if err != nil || v < k.Lo || v > k.Hi || (k.Whole && v != float64(int(v))) {
				t.Fatalf("%s = %q out of range %v..%v", k.Flag, lookup(got, k.Flag), k.Lo, k.Hi)
			}
		}
		if strings.Join(got, " ") != strings.Join(args, " ") {
			changed++
		}
		if !strings.Contains(strings.Join(got, " "), "--abliterate") {
			t.Fatal("other options must stay")
		}
		if lookup(got, "--kv-resident") != "" {
			t.Fatal("--kv-resident switches KV streaming on; the search must not add it")
		}
	}
	if changed < 200 {
		t.Errorf("only %d of 300 proposals differ from the start", changed)
	}
}

func TestSmallerStepMovesLess(t *testing.T) {
	knobs := DefaultKnobs()[:1] // --pool-workers
	spread := func(step float64) float64 {
		rng := rand.New(rand.NewSource(2))
		sum := 0.0
		for i := 0; i < 400; i++ {
			v, _ := strconv.ParseFloat(lookup(Propose(rng, []string{"--pool-workers", "16"}, knobs, step), "--pool-workers"), 64)
			sum += (v - 16) * (v - 16)
		}
		return sum
	}
	if spread(0.05) >= spread(0.3) {
		t.Error("a smaller step should move the value less")
	}
	if Decay(0.3, 0.5, 2) != 0.075 {
		t.Errorf("Decay = %v", Decay(0.3, 0.5, 2))
	}
}

func TestAcceptAndScore(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	if !Accept(rng, 10, 11, 0) || Accept(rng, 10, 9, 0) {
		t.Error("at temperature 0 only an improvement is taken")
	}
	hot, cold := 0, 0
	for i := 0; i < 1000; i++ {
		if Accept(rng, 10, 9.5, 0.2) {
			hot++
		}
		if Accept(rng, 10, 9.5, 0.01) {
			cold++
		}
	}
	if hot <= cold {
		t.Errorf("a hotter search should take more slower settings: %d vs %d", hot, cold)
	}
	fast := Result{DecodeTokS: 40, PrefillTokS: 300}
	if Score(fast) <= Score(Result{DecodeTokS: 30, PrefillTokS: 300}) || Score(Result{Err: "x", DecodeTokS: 99, PrefillTokS: 99}) != 0 {
		t.Error("score order")
	}
}

func TestWithArgsDoesNotChangeTheOriginal(t *testing.T) {
	c := Candidates(base(), 1)[0]
	n := WithArgs(c, "search-1", []string{"--pool-workers", "9"})
	if lookup(ArgsOf(c), "--pool-workers") != "" || lookup(ArgsOf(n), "--pool-workers") != "9" || n.Name != "search-1" {
		t.Errorf("orig %v new %v", ArgsOf(c), ArgsOf(n))
	}
	if !strings.Contains(Settings(n, DefaultKnobs()), "--pool-workers 9") {
		t.Error(Settings(n, DefaultKnobs()))
	}
}

func TestProposeChangesAnOptionalKnobOnlyWhereSet(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	args := []string{"--kv-resident", "20480"}
	moved := false
	for i := 0; i < 200; i++ {
		got := Propose(rng, args, DefaultKnobs(), 0.3)
		v, err := strconv.Atoi(lookup(got, "--kv-resident"))
		if err != nil || v < 20480 || v > 49152 || v%1024 != 0 {
			t.Fatalf("--kv-resident = %q", lookup(got, "--kv-resident"))
		}
		moved = moved || v != 20480
	}
	if !moved {
		t.Error("--kv-resident never changed though the run config sets it")
	}
}
