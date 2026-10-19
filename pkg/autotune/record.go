// pkg/autotune/record.go - persisted tuning decisions, treated as hints.
//
// A record is data only: no pointers, handles, stream IDs or allocator state. It is written atomically and a
// missing or corrupt file is a normal condition (the caller tunes again or uses defaults), never a startup
// failure. A record is usable only after its fingerprint, workload hash, schema and feasibility all match, and
// after a fresh validation sample; an old record is not trusted just because it parsed.
package autotune

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// RecordVersion invalidates records whose layout or meaning changed.
const RecordVersion = 1

// DefaultRecordName is the file the server and the CLI use unless Options.RecordPath overrides it.
const DefaultRecordName = "strata-tuning.json"

// RecordMaxAge is the longest a record is considered without fresh validation. Hardware and driver changes are
// caught by the fingerprint; this age only covers measurement drift (other software, thermal state, drivers).
const RecordMaxAge = 30 * 24 * time.Hour

// ErrNoRecord means there is nothing to load; it is not an error condition for startup.
var ErrNoRecord = errors.New("no tuning record")

// ErrCorruptRecord means the file exists but cannot be used; startup falls back to tuning/defaults.
var ErrCorruptRecord = errors.New("tuning record is corrupt")

// Methodology describes how a record was produced, so a later reader can judge it.
type Methodology struct {
	Tool    string        `json:"tool"`
	Probe   string        `json:"probe,omitempty"`
	Limits  Limits        `json:"limits,omitempty"`
	Started time.Time     `json:"started,omitempty"`
	Ended   time.Time     `json:"ended,omitempty"`
	Phases  []PhaseRecord `json:"phases,omitempty"`
	Samples int           `json:"samples,omitempty"`
	Notes   []string      `json:"notes,omitempty"`
}

// Validation records the correctness/resource check that must precede applying a plan.
type Validation struct {
	Validated     bool      `json:"validated"`
	Correct       bool      `json:"correct"`
	Detail        string    `json:"detail,omitempty"`
	EngineVersion string    `json:"engine_version,omitempty"`
	At            time.Time `json:"at,omitempty"`
}

// Failure is a preserved first causal error from a tuning attempt.
type Failure struct {
	At        time.Time `json:"at"`
	Phase     string    `json:"phase,omitempty"`
	Candidate string    `json:"candidate,omitempty"`
	Error     string    `json:"error"`
}

// Record is the versioned tuning record. It deliberately contains no live handles.
type Record struct {
	Version        int          `json:"version"`
	Fingerprint    string       `json:"fingerprint"`
	WorkloadHash   string       `json:"workload_hash"`
	Capabilities   Capabilities `json:"capabilities"`
	Workload       Workload     `json:"workload"`
	Measurements   Measurements `json:"measurements"`
	Selected       Candidate    `json:"selected"`
	Prediction     Prediction   `json:"prediction"`
	Objective      string       `json:"objective"`
	Classification string       `json:"classification"`
	Coverage       string       `json:"coverage"`
	Gap            float64      `json:"gap,omitempty"`
	Constraints    Constraints  `json:"constraints"`
	Validation     Validation   `json:"validation"`
	Failures       []Failure    `json:"failures,omitempty"`
	Methodology    Methodology  `json:"methodology"`
	CreatedAt      time.Time    `json:"created_at"`
	UpdatedAt      time.Time    `json:"updated_at"`
}

// NewRecord builds a record from a plan and the measurements behind it.
func NewRecord(caps Capabilities, wl Workload, m Measurements, plan Plan, cons Constraints, meth Methodology) Record {
	now := time.Now()
	return Record{
		Version: RecordVersion, Fingerprint: caps.Fingerprint(), WorkloadHash: wl.Fingerprint(),
		Capabilities: caps, Workload: wl, Measurements: m,
		Selected: plan.Selected, Prediction: plan.Prediction, Objective: plan.Objective,
		Classification: plan.Classification, Coverage: plan.Coverage, Gap: plan.Gap,
		Constraints: cons, Methodology: meth, CreatedAt: now, UpdatedAt: now,
	}
}

// DefaultRecordPath returns $STRATA_TUNING_RECORD when set, else DefaultRecordName in the working directory.
func DefaultRecordPath() string {
	if p := os.Getenv("STRATA_TUNING_RECORD"); p != "" {
		return p
	}
	return DefaultRecordName
}

// Save writes the record atomically: a temporary file in the same directory, fsync, rename, then a directory
// sync where the platform supports it. A crash leaves either the old record or the new one, never a half file.
func Save(path string, rec Record) error {
	if path == "" {
		path = DefaultRecordPath()
	}
	rec.Version = RecordVersion
	rec.UpdatedAt = time.Now()
	if rec.CreatedAt.IsZero() {
		rec.CreatedAt = rec.UpdatedAt
	}
	raw, err := json.MarshalIndent(rec, "", "  ")
	if err != nil {
		return err
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	f, err := os.CreateTemp(dir, ".strata-tuning-*")
	if err != nil {
		return err
	}
	tmp := f.Name()
	defer os.Remove(tmp) // no-op after a successful rename
	if _, err := f.Write(raw); err != nil {
		f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return err
	}
	if err := f.Chmod(0o644); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		// Windows refuses to replace an existing file; the old record is removed first, and a crash in between
		// simply leaves no record (startup tolerates that).
		if removeErr := os.Remove(path); removeErr != nil && !os.IsNotExist(removeErr) {
			return err
		}
		if err := os.Rename(tmp, path); err != nil {
			return err
		}
	}
	if d, err := os.Open(dir); err == nil {
		_ = d.Sync()
		_ = d.Close()
	}
	return nil
}

// Load reads a record. Missing is ErrNoRecord; unreadable or invalid JSON is ErrCorruptRecord.
func Load(path string) (Record, error) {
	if path == "" {
		path = DefaultRecordPath()
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return Record{}, ErrNoRecord
		}
		return Record{}, fmt.Errorf("%w: %v", ErrCorruptRecord, err)
	}
	var rec Record
	if err := json.Unmarshal(raw, &rec); err != nil {
		return Record{}, fmt.Errorf("%w: %v", ErrCorruptRecord, err)
	}
	if rec.Version == 0 || rec.Fingerprint == "" || rec.Selected.Placement == "" {
		return Record{}, ErrCorruptRecord
	}
	return rec, nil
}

// Usable says whether a record may be used as a hint for this machine and model. It checks the schema, the
// hardware/driver fingerprint, the model/workload hash and the stored constraint facts. It does not run any
// benchmark: the caller validates a small sample before applying the result.
func (r Record) Usable(caps Capabilities, wl Workload, now time.Time, maxAge time.Duration) (bool, string) {
	if r.Version != RecordVersion {
		return false, fmt.Sprintf("record version %d, expected %d", r.Version, RecordVersion)
	}
	if r.Fingerprint != caps.Fingerprint() {
		return false, "hardware, driver or backend changed"
	}
	if r.WorkloadHash != wl.Fingerprint() {
		return false, "model geometry, context or KV format changed"
	}
	if now.IsZero() {
		now = time.Now()
	}
	if maxAge <= 0 {
		maxAge = RecordMaxAge
	}
	if !r.UpdatedAt.IsZero() && now.Sub(r.UpdatedAt) > maxAge {
		return false, "record is older than " + maxAge.String()
	}
	if r.Selected.Placement == "" {
		return false, "record has no selected configuration"
	}
	if f := Check(r.Selected, wl, caps, r.Constraints); !f.OK {
		return false, "the stored configuration no longer satisfies the constraints: " + violationSummary(f)
	}
	return true, ""
}

// Validated reports whether the record's selected configuration was smoke-tested on the running engine. An
// unvalidated record may still lend its CPU measurements, but engine-side knobs are only applied from a
// validated one.
func (r Record) Validated() bool { return r.Validation.Validated && r.Validation.Correct }

func violationSummary(f Feasibility) string {
	if len(f.Reasons) == 0 {
		return "rejected"
	}
	s := f.Reasons[0].Code
	for i := 1; i < len(f.Reasons) && i < 3; i++ {
		s += ", " + f.Reasons[i].Code
	}
	return s
}

// HintMeasurements returns the record's measurements if the record is usable, else nil. Measurements are always
// hints: the solver re-checks every candidate against the current constraints.
func HintMeasurements(path string, caps Capabilities, wl Workload, now time.Time) (Measurements, string) {
	rec, err := Load(path)
	if err != nil {
		return Measurements{}, err.Error()
	}
	if ok, reason := rec.Usable(caps, wl, now, RecordMaxAge); !ok {
		return Measurements{}, reason
	}
	m := rec.Measurements
	m.Source = "record"
	return m, ""
}
