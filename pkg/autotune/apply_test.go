package autotune

import (
	"reflect"
	"testing"
)

func TestApplyArgsOnlyWritesMeasuredKnobs(t *testing.T) {
	args := []string{"--pack", "/p", "--pool-workers", "10", "--pcie-frac", "0.35", "--spec-min-p", "0.5", "--kv", "int8"}
	sel := Candidate{Placement: "cpu", PoolWorkers: 6, PCIeFrac: 0.55, SpecMinP: 0.3}

	// No evidence: the arguments must be untouched.
	if got := ApplyArgs(args, sel, ApplyEvidence{}); !reflect.DeepEqual(got, args) {
		t.Errorf("unmeasured change: %v", got)
	}
	// Worker evidence replaces the existing value in place.
	got := ApplyArgs(args, sel, ApplyEvidence{PoolWorkersMeasured: true})
	if got[3] != "6" {
		t.Errorf("worker not replaced: %v", got)
	}
	if len(got) != len(args) {
		t.Errorf("in-place replacement changed the length: %v", got)
	}
	// PCIe and spec replace in place too.
	got = ApplyArgs(args, sel, ApplyEvidence{PoolWorkersMeasured: true, PCIeMeasured: true, SpecMeasured: true})
	if got[5] != "0.55" || got[7] != "0.30" {
		t.Errorf("pcie/spec: %v", got)
	}
	// A missing flag is appended, never duplicated.
	got = ApplyArgs([]string{"--pack", "/p"}, sel, ApplyEvidence{PoolWorkersMeasured: true})
	if len(got) != 4 || got[2] != "--pool-workers" || got[3] != "6" {
		t.Errorf("append: %v", got)
	}
	// Formatting is stable for the engine's atof parsing.
	if s := format2(0.07); s != "0.07" {
		t.Errorf("format2: %s", s)
	}
	if s := format2(1.0); s != "1.00" {
		t.Errorf("format2: %s", s)
	}
}

func TestEvidenceComesFromTheMeasuredKey(t *testing.T) {
	c := Candidate{Placement: "gpu", Devices: []int{0}, PoolWorkers: 4, PCIeFrac: 0.55, SpecMinP: 0.5, Batch: 1}
	m := Measurements{
		DecodeTokS: map[string]float64{c.Key(): 100},
		CPUScaling: []CPUScalePoint{{1, 10, 0}, {4, 40, 0}},
	}
	ev := EvidenceFromMeasurements(c, m)
	if !ev.PoolWorkersMeasured || !ev.PCIeMeasured || !ev.SpecMeasured {
		t.Errorf("evidence: %+v", ev)
	}
	other := Candidate{Placement: "gpu", Devices: []int{1}, PoolWorkers: 4, PCIeFrac: 0.55, SpecMinP: 0.5, Batch: 1}
	ev = EvidenceFromMeasurements(other, m)
	if ev.PCIeMeasured || ev.SpecMeasured {
		t.Errorf("evidence must be per candidate: %+v", ev)
	}
}
