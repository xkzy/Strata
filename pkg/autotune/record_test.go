package autotune

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func makeRecord(t *testing.T) (Record, Capabilities, Workload) {
	t.Helper()
	caps, w := testCaps(), testWorkload()
	m := testMeasurements()
	cons := DefaultConstraints(caps, w)
	plan := Solve(w, caps, m, cons, Options{Objective: ObjDecodeTokS})
	rec := NewRecord(caps, w, m, plan, cons, Methodology{Tool: "test"})
	rec.Validation = Validation{Validated: true, Correct: true, Detail: "served", At: time.Now()}
	return rec, caps, w
}

func TestRecordRoundTripIsAtomicAndUsable(t *testing.T) {
	rec, caps, w := makeRecord(t)
	path := filepath.Join(t.TempDir(), "tuning.json")
	if err := Save(path, rec); err != nil {
		t.Fatal(err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Selected.Key() != rec.Selected.Key() || loaded.Fingerprint != caps.Fingerprint() {
		t.Fatalf("round trip changed the record: %+v vs %+v", loaded.Selected, rec.Selected)
	}
	if ok, reason := loaded.Usable(caps, w, time.Now(), RecordMaxAge); !ok {
		t.Fatalf("a fresh record for the same machine must be usable: %s", reason)
	}
	if !loaded.Validated() {
		t.Error("validation must survive the round trip")
	}
	// Saving twice replaces atomically and leaves no temporary file behind.
	if err := Save(path, rec); err != nil {
		t.Fatal(err)
	}
	entries, _ := os.ReadDir(filepath.Dir(path))
	for _, e := range entries {
		if len(e.Name()) > 0 && e.Name()[0] == '.' {
			t.Errorf("temporary file left behind: %s", e.Name())
		}
	}
}

func TestRecordMissingAndCorrupt(t *testing.T) {
	if _, err := Load(filepath.Join(t.TempDir(), "absent.json")); !errors.Is(err, ErrNoRecord) {
		t.Errorf("missing record: %v", err)
	}
	path := filepath.Join(t.TempDir(), "bad.json")
	if err := os.WriteFile(path, []byte("{not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(path); !errors.Is(err, ErrCorruptRecord) {
		t.Errorf("corrupt record: %v", err)
	}
	if err := os.WriteFile(path, []byte(`{"version":1}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(path); !errors.Is(err, ErrCorruptRecord) {
		t.Errorf("a record without a fingerprint is corrupt: %v", err)
	}
}

func TestRecordInvalidation(t *testing.T) {
	rec, caps, w := makeRecord(t)
	now := time.Now()
	if ok, _ := rec.Usable(caps, w, now, RecordMaxAge); !ok {
		t.Fatal("baseline record must be usable")
	}
	other := caps
	other.Memory.TotalBytes += 1 << 30
	if ok, reason := rec.Usable(other, w, now, RecordMaxAge); ok || reason == "" {
		t.Error("a different machine must invalidate the record")
	}
	otherGPU := caps
	otherGPU.Accelerators = append([]Accelerator(nil), caps.Accelerators...)
	otherGPU.Accelerators[0].Name = "Different card"
	if ok, _ := rec.Usable(otherGPU, w, now, RecordMaxAge); ok {
		t.Error("a changed device must invalidate the record")
	}
	otherModel := w
	otherModel.MaxContext = 16384
	if ok, _ := rec.Usable(caps, otherModel, now, RecordMaxAge); ok {
		t.Error("a different context must invalidate the record")
	}
	if ok, reason := rec.Usable(caps, w, now.Add(RecordMaxAge+time.Hour), RecordMaxAge); ok || reason == "" {
		t.Error("an old record must expire")
	}
	if ok, reason := rec.Usable(caps, w, now, RecordMaxAge); !ok {
		t.Errorf("re-checking must not consume the record: %s", reason)
	}
	// A stored configuration that is no longer feasible is rejected even with a matching fingerprint.
	broken := rec
	broken.Selected = Candidate{Placement: "gpu", Devices: []int{9}, PCIeFrac: -1, Batch: 1}
	if ok, _ := broken.Usable(caps, w, now, RecordMaxAge); ok {
		t.Error("a device index that no longer exists must invalidate the record")
	}
}

func TestRecordHintsAreMeasurementsOnly(t *testing.T) {
	rec, caps, w := makeRecord(t)
	path := filepath.Join(t.TempDir(), "tuning.json")
	if err := Save(path, rec); err != nil {
		t.Fatal(err)
	}
	m, reason := HintMeasurements(path, caps, w, time.Now())
	if reason != "" {
		t.Fatalf("hint refused: %s", reason)
	}
	if m.Source != "record" || len(m.DecodeTokS) != len(rec.Measurements.DecodeTokS) {
		t.Errorf("hints: %+v", m)
	}
	if _, reason := HintMeasurements(path, caps, w, time.Now().Add(RecordMaxAge+time.Hour)); reason == "" {
		t.Error("an expired record must not lend hints")
	}
}

func TestRecordContainsNoLiveState(t *testing.T) {
	rec, _, _ := makeRecord(t)
	// The record is plain data: marshalling it must not fail on a channel, function, pointer or handle.
	if _, err := marshalRecord(rec); err != nil {
		t.Errorf("record is not plain data: %v", err)
	}
}

func marshalRecord(rec Record) ([]byte, error) {
	return json.Marshal(rec)
}

func TestCachedStartKeepsValidationAndFailureHistory(t *testing.T) {
	rec, caps, w := makeRecord(t)
	rec.Failures = []Failure{{Error: "first causal error"}}
	path := filepath.Join(t.TempDir(), "tuning.json")
	if err := Save(path, rec); err != nil {
		t.Fatal(err)
	}
	// No probe: this is the cached server path. It must not erase what the same configuration already proved.
	pr := PlanStartup(context.Background(), caps, PlanOptions{
		Geometry: w.Geometry, Envelope: w.Envelope, MaxContext: w.MaxContext, RecordPath: path, Save: true,
	})
	if !pr.Saved {
		t.Fatalf("record not saved: %v", pr.Err)
	}
	got, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if !got.Validation.Validated || !got.Validation.Correct {
		t.Errorf("a cached start erased the validation: %+v", got.Validation)
	}
	if len(got.Failures) == 0 || got.Failures[0].Error != "first causal error" {
		t.Errorf("failure history was lost: %+v", got.Failures)
	}
}
