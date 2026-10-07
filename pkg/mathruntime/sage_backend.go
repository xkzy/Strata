package mathruntime

import (
	"strings"
)

// SageBackend is a SageMath-compatible CAS backend adapter wrapping the unified StrataCAS engine.
type SageBackend struct {
	*UnifiedCasBackend
}

// NewSageBackend creates a new SageMath-compatible CAS adapter.
func NewSageBackend() *SageBackend {
	cas := NewUnifiedCasBackend()
	cas.Name = "SageMath"
	cas.Version = "10.4"
	return &SageBackend{
		UnifiedCasBackend: cas,
	}
}

// Execute executes a math request using the underlying unified CAS engine and applies Sage formatting if needed.
func (b *SageBackend) Execute(req MathRequest) MathResult {
	res := b.UnifiedCasBackend.Execute(req)
	res.BackendName = b.Name
	res.BackendVersion = b.Version

	if req.Operation == OpSolve {
		res.ExactResult = formatSageString(res.ExactResult, req.Operation)
		res.RawResult = res.ExactResult
	}
	return res
}

func formatSageString(s string, op MathOperation) string {
	if op == OpSolve {
		s = strings.ReplaceAll(s, "{", "[")
		s = strings.ReplaceAll(s, "}", "]")
		s = strings.ReplaceAll(s, "->", "==")
		return s
	}
	if strings.HasPrefix(s, "{") && strings.HasSuffix(s, "}") {
		return "[" + s[1:len(s)-1] + "]"
	}
	return s
}
