package mathruntime

import (
	"fmt"
	"regexp"
	"strings"
	"sync/atomic"
	"time"

	"strata/pkg/mathruntime/lean"
)

// LeanBackend proves or refutes closed Nat claims ("2^10 = 1024", "3 ∣ 12 ∧ 5 < 7") and Int claims (a claim with a
// negative numeral, "3 - 5 = -2") with the Go port of the Lean 4 kernel in the lean package. The verdict is a
// kernel-checked proof term, so it does not rest on this runtime's own CAS. Nothing is executed outside the process.
type LeanBackend struct {
	Name      string
	Version   string
	Available bool
	Timeout   time.Duration
	MaxSteps  int
}

func NewLeanBackend(timeout time.Duration) *LeanBackend {
	b := &LeanBackend{Name: "Lean4-Kernel-Go", Version: "4.35.0", Timeout: timeout, MaxSteps: 5_000_000}
	_, err := lean.Prelude()
	b.Available = err == nil
	return b
}

// Prove returns StatusSuccess with ExactResult "proved" or "refuted" (each backed by a kernel-checked term).
// A claim outside the supported fragment is StatusUnsupportedOp; running out of budget is StatusTimeout.
func (b *LeanBackend) Prove(claim string) MathResult {
	res := MathResult{BackendName: b.Name, BackendVersion: b.Version, CanonicalExpression: claim}
	if !b.Available {
		res.Status, res.ErrorMessage = StatusExecutionError, "lean prelude failed to build"
		return res
	}
	start := time.Now()
	opt := lean.Options{MaxSteps: b.MaxSteps}
	if b.Timeout > 0 {
		opt.Deadline = start.Add(b.Timeout)
	}
	proof, err := lean.ProveClaim(claim, opt)
	res.ExecutionTimeMs = float64(time.Since(start).Microseconds()) / 1000
	verdict := ""
	switch e := err.(type) {
	case nil:
		res.Status, res.ExactResult, verdict = StatusSuccess, "proved", "proved"
	case lean.ErrNotProved:
		if e.Refuted {
			res.Status, res.ExactResult, verdict, proof = StatusSuccess, "refuted", "refuted", e.Proof
		} else {
			res.Status, res.ErrorMessage = StatusExecutionError, e.Error()
		}
	case lean.ErrUnsupported:
		res.Status, res.ErrorMessage = StatusUnsupportedOp, e.Error()
	default:
		if lean.IsBudgetError(err) {
			res.Status, res.ErrorMessage = StatusTimeout, err.Error()
		} else {
			res.Status, res.ErrorMessage = StatusExecutionError, err.Error()
		}
	}
	res.RawResult = res.ExactResult
	if proof != nil {
		term := proof.Term.String()
		if len(term) > 400 {
			term = term[:400] + "…"
		}
		res.Metadata = map[string]interface{}{"kernel_steps": proof.Steps, "proof_term": term, "proposition": proof.Prop.String()}
	}
	if verdict != "" {
		res.CompactObservation = fmt.Sprintf("[Lean kernel: %s => %s]", claim, verdict)
	} else {
		res.CompactObservation = fmt.Sprintf("[Lean kernel: %s => %s]", claim, res.Status)
	}
	return res
}

// leanClaimInText finds `example/theorem/lemma ... : <claim> := by decide|rfl|norm_num|simp|omega` in model output
// and returns the claim between the colon and `:=`.
var leanDeclRe = regexp.MustCompile(`(?s)\b(?:example|theorem|lemma)\b[^:\n]*:\s*(.+?)\s*:=\s*by\s+(?:decide|rfl|norm_num|simp|omega|native_decide)\b`)

func leanClaimInText(text string) (string, bool) {
	if m := leanDeclRe.FindStringSubmatch(text); len(m) > 1 {
		return strings.TrimSpace(m[1]), true
	}
	return "", false
}

// proveRequest runs an OpProve request. The claim is the request's Expression.
func (r *MathRuntime) proveRequest(req MathRequest) MathResult {
	atomicAdd(&r.stats.leanCount)
	res := r.LeanBackend.Prove(req.Expression)
	res.RequestID = req.RequestID
	res.ObservationTokens = max(1, (len(res.CompactObservation)+3)/4)
	if res.Status == StatusSuccess {
		if res.ExactResult == "refuted" {
			atomicAdd(&r.stats.leanRefuted)
		}
	}
	return res
}

// ProveWithLean is the Go-API entry point: it goes through ProcessRequest, so the length limit and stats apply.
func (r *MathRuntime) ProveWithLean(claim string) MathResult {
	return r.ProcessRequest(MathRequest{Operation: OpProve, Expression: claim, Mode: ModeExact})
}

// natFaithful reports whether an expression means the same in Lean's Nat and in this runtime's rational
// arithmetic: only + * ^ and parentheses over unsigned integers (no - or / or %, which differ on Nat).
var natFaithfulRe = regexp.MustCompile(`^[0-9+*^()\s]+$`)
var plainIntRe = regexp.MustCompile(`^[0-9]+$`)

// certify asks the Lean kernel whether `expr = value` holds, independently of the CAS that produced value.
// It returns (checked, agrees): checked is false when the expression is outside the fragment where both engines
// mean the same thing.
func (r *MathRuntime) certify(expr, value string) (checked, agrees bool) {
	expr, value = strings.TrimSpace(expr), strings.TrimSpace(value)
	if !natFaithfulRe.MatchString(expr) || !plainIntRe.MatchString(value) || len(expr) > 512 {
		return false, false
	}
	res := r.LeanBackend.Prove(expr + " = " + value)
	if res.Status != StatusSuccess {
		return false, false
	}
	return true, res.ExactResult == "proved"
}

func atomicAdd(p *uint64) { atomic.AddUint64(p, 1) }
