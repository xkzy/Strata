package mathruntime

// MathicsBackend is a Wolfram-compatible CAS backend adapter wrapping the unified StrataCAS engine.
type MathicsBackend struct {
	*UnifiedCasBackend
}

// NewMathicsBackend creates a new Mathics-compatible CAS adapter.
func NewMathicsBackend() *MathicsBackend {
	cas := NewUnifiedCasBackend()
	cas.Name = "Mathics3-Core"
	cas.Version = "3.0.0"
	return &MathicsBackend{
		UnifiedCasBackend: cas,
	}
}

// Execute executes a math request using the underlying unified CAS engine.
func (b *MathicsBackend) Execute(req MathRequest) MathResult {
	res := b.UnifiedCasBackend.Execute(req)
	res.BackendName = b.Name
	res.BackendVersion = b.Version
	return res
}
