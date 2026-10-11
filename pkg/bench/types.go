package bench

// BenchmarkOptions configures an inference benchmark execution
type BenchmarkOptions struct {
	ServerURL   string
	Prompt      string
	MaxTokens   int
	Concurrence int
	Iterations  int
	ModelName   string
}

// BenchmarkReport contains measured performance statistics
type BenchmarkReport struct {
	Iterations       int     `json:"iterations"`
	PromptTokens     int     `json:"prompt_tokens"`
	GenTokens        int     `json:"gen_tokens"`
	TTFTMs           float64 `json:"ttft_ms"`
	PromptTokensSec  float64 `json:"prompt_tokens_sec"`
	GenTokensPerSec  float64 `json:"gen_tokens_sec"`
	TotalLatencyMs   float64 `json:"total_latency_ms"`
	P50LatencyMs     float64 `json:"p50_latency_ms"`
	P95LatencyMs     float64 `json:"p95_latency_ms"`
}
