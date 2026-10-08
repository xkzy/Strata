package resourcemanager

import (
	"sync"
)

const (
	ShortInteractive   = "SHORT_INTERACTIVE"
	LongContext        = "LONG_CONTEXT"
	HighThroughput     = "HIGH_THROUGHPUT"
	LowLatency         = "LOW_LATENCY"
	MemoryBound        = "MEMORY_BOUND"
	ComputeBound       = "COMPUTE_BOUND"
	TransferBound      = "TRANSFER_BOUND"
	MoeSparse          = "MOE_SPARSE"
	MoeDense           = "MOE_DENSE"
	PrefillHeavy       = "PREFILL_HEAVY"
	DecodeHeavy        = "DECODE_HEAVY"
	MathEvaluation     = "MATH_EVALUATION"
	MathicsCalculation = "MATHICS_CALCULATION"
)

type ResourceManager struct {
	mu            sync.Mutex
	concurrency   int
	queueLimit    int
	cacheTier     string
	powerPolicy   string
	activeWork    int
	totalAdmitted int
	totalRejected int
	decisions     []map[string]interface{}
}

func NewResourceManager() *ResourceManager {
	return &ResourceManager{
		concurrency: 4,
		queueLimit:  32,
		cacheTier:   "balanced",
		powerPolicy: "BALANCED",
		decisions:   make([]map[string]interface{}, 0, 64),
	}
}

func (rm *ResourceManager) AdmitRequest(workloadClass string) (bool, int) {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	if rm.activeWork >= rm.concurrency+rm.queueLimit {
		rm.totalRejected++
		return false, 2 // retry_after 2s
	}

	rm.activeWork++
	rm.totalAdmitted++
	return true, 0
}

func (rm *ResourceManager) FinishRequest() {
	rm.mu.Lock()
	defer rm.mu.Unlock()
	if rm.activeWork > 0 {
		rm.activeWork--
	}
}

func (rm *ResourceManager) Metrics() map[string]interface{} {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	return map[string]interface{}{
		"concurrency":    rm.concurrency,
		"queue_limit":    rm.queueLimit,
		"cache_tier":     rm.cacheTier,
		"power_policy":   rm.powerPolicy,
		"active_work":    rm.activeWork,
		"total_admitted": rm.totalAdmitted,
		"total_rejected": rm.totalRejected,
		"workloads": []string{
			ShortInteractive, LongContext, HighThroughput, LowLatency,
			MathEvaluation, MathicsCalculation,
		},
	}
}
