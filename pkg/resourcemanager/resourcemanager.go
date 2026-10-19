// pkg/resourcemanager/resourcemanager.go - Advanced Multi-Resource Task Distribution Policy
package resourcemanager

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// Workload Classification constants
const (
	ShortInteractive    = "SHORT_INTERACTIVE"
	LongContext         = "LONG_CONTEXT"
	HighThroughput      = "HIGH_THROUGHPUT"
	LowLatency          = "LOW_LATENCY"
	MemoryBound         = "MEMORY_BOUND"
	ComputeBound        = "COMPUTE_BOUND"
	TransferBound       = "TRANSFER_BOUND"
	MoeSparse           = "MOE_SPARSE"
	MoeDense            = "MOE_DENSE"
	PrefillHeavy        = "PREFILL_HEAVY"
	DecodeHeavy         = "DECODE_HEAVY"
	MathEvaluation      = "MATH_EVALUATION"
	MathicsCalculation  = "MATHICS_CALCULATION"
	LogicVerification   = "LOGIC_VERIFICATION"
	SpeculativeDrafting = "SPECULATIVE_DRAFTING"
)

// Hardware Target Device enum
type TargetDevice string

const (
	DeviceGPU           TargetDevice = "GPU"             // Dedicated Discrete GPU (Tensor Cores / High Bandwidth)
	DeviceCPUWorkerPool TargetDevice = "CPU_WORKER_POOL" // Multi-core CPU thread pool (AVX2/AVX-512)
	DeviceAPUShared     TargetDevice = "APU_SHARED"      // Integrated GPU / APU shared memory
	DeviceHybridAsync   TargetDevice = "HYBRID_ASYNC"    // Pipelined GPU + CPU Co-execution
)

// Execution Strategy enum
type ExecutionStrategy string

const (
	StrategyDirectGPU         ExecutionStrategy = "DIRECT_GPU"         // Standard full forward pass on GPU
	StrategyChunkedPrefill    ExecutionStrategy = "CHUNKED_PREFILL"    // Interleave prompt chunks with decode tokens
	StrategyActivationOffload ExecutionStrategy = "ACTIVATION_OFFLOAD" // Send small activation vectors over PCIe to CPU experts
	StrategyAsyncSpeculative  ExecutionStrategy = "ASYNC_SPECULATIVE"  // Asynchronous CPU drafting with parallel GPU verification
	StrategyIndependentCPU    ExecutionStrategy = "INDEPENDENT_CPU"    // Pure CPU execution without GPU contention
)

// ResourceSnapshot captures current hardware utilization
type ResourceSnapshot struct {
	VRAMUsedBytes     int64     `json:"vram_used_bytes"`
	VRAMTotalBytes    int64     `json:"vram_total_bytes"`
	HostRAMUsedBytes  int64     `json:"host_ram_used_bytes"`
	HostRAMTotalBytes int64     `json:"host_ram_total_bytes"`
	PCIeTransferMBs   float64   `json:"pcie_transfer_mbs"`
	PCIeMaxBWMBs      float64   `json:"pcie_max_bw_mbs"`
	CPULoadPercent    float64   `json:"cpu_load_percent"`
	GPULoadPercent    float64   `json:"gpu_load_percent"`
	ActiveQueueLength int       `json:"active_queue_length"`
	Timestamp         time.Time `json:"timestamp"`
}

// TaskDescriptor describes an incoming task for dispatching
type TaskDescriptor struct {
	TaskID        string `json:"task_id"`
	WorkloadClass string `json:"workload_class"`
	PromptTokens  int    `json:"prompt_tokens"`
	MaxTokens     int    `json:"max_tokens"`
	ModelID       string `json:"model_id"`
	Priority      int    `json:"priority"` // Higher is more urgent (0-10)
	IsMoE         bool   `json:"is_moe"`
	IsPureCPU     bool   `json:"is_pure_cpu"`
}

// AdmitDecision contains routing and execution directives
type AdmitDecision struct {
	Admitted          bool              `json:"admitted"`
	TargetDevice      TargetDevice      `json:"target_device"`
	Strategy          ExecutionStrategy `json:"strategy"`
	ChunkSize         int               `json:"chunk_size,omitempty"`
	RetryAfterSeconds int               `json:"retry_after_seconds,omitempty"`
	Reason            string            `json:"reason"`
}

// DecisionRecord logs historical scheduling events
type DecisionRecord struct {
	Timestamp     time.Time         `json:"timestamp"`
	TaskID        string            `json:"task_id"`
	WorkloadClass string            `json:"workload_class"`
	Device        TargetDevice      `json:"device"`
	Strategy      ExecutionStrategy `json:"strategy"`
	DominantRatio float64           `json:"dominant_ratio"`
	Reason        string            `json:"reason"`
}

// ResourceManager implements the multi-resource task distribution policy
type ResourceManager struct {
	// Adapt is the runtime controller: admission, queueing and memory-tier policy. It is separate from the
	// legacy RouteTask counters below, which remain the task-routing policy.
	Adapt *Controller

	mu                sync.RWMutex
	concurrency       int
	queueLimit        int
	cacheTier         string
	powerPolicy       string
	activeWork        int
	activeCPUWork     int
	totalAdmitted     int
	totalRejected     int
	lastSnapshot      ResourceSnapshot
	decisions         []DecisionRecord
	maxDecisionsLog   int
	recentCompletions []time.Time
}

// NewResourceManager initializes the resource controller with balanced defaults
func NewResourceManager() *ResourceManager {
	return &ResourceManager{
		Adapt:             NewController(DefaultControllerConfig()),
		concurrency:       4,
		queueLimit:        32,
		cacheTier:         "balanced",
		powerPolicy:       "BALANCED",
		decisions:         make([]DecisionRecord, 0, 128),
		maxDecisionsLog:   128,
		recentCompletions: make([]time.Time, 0, 64),
		lastSnapshot: ResourceSnapshot{
			VRAMTotalBytes:    8 << 30,  // 8 GiB default
			HostRAMTotalBytes: 32 << 30, // 32 GiB default
			PCIeMaxBWMBs:      15754.0,  // PCIe 3.0 x16 ~ 15.75 GB/s
			Timestamp:         time.Now(),
		},
	}
}

// UpdateTelemetry updates real-time hardware measurements
func (rm *ResourceManager) UpdateTelemetry(snap ResourceSnapshot) {
	rm.mu.Lock()
	defer rm.mu.Unlock()
	if snap.Timestamp.IsZero() {
		snap.Timestamp = time.Now()
	}
	rm.lastSnapshot = snap
}

// DominantResourceRatio computes max utilization ratio across VRAM, RAM, PCIe, CPU
func (rm *ResourceManager) DominantResourceRatio() float64 {
	rm.mu.RLock()
	defer rm.mu.RUnlock()

	return rm.dominantRatioLocked()
}

func (rm *ResourceManager) dominantRatioLocked() float64 {
	snap := rm.lastSnapshot
	vramRatio := 0.0
	if snap.VRAMTotalBytes > 0 {
		vramRatio = float64(snap.VRAMUsedBytes) / float64(snap.VRAMTotalBytes)
	}

	ramRatio := 0.0
	if snap.HostRAMTotalBytes > 0 {
		ramRatio = float64(snap.HostRAMUsedBytes) / float64(snap.HostRAMTotalBytes)
	}

	pcieRatio := 0.0
	if snap.PCIeMaxBWMBs > 0 {
		pcieRatio = snap.PCIeTransferMBs / snap.PCIeMaxBWMBs
	}

	cpuRatio := snap.CPULoadPercent / 100.0

	return math.Max(vramRatio, math.Max(ramRatio, math.Max(pcieRatio, cpuRatio)))
}

// RouteTask determines optimal device and execution strategy based on workload and hardware pressure
func (rm *ResourceManager) RouteTask(task TaskDescriptor) AdmitDecision {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	domRatio := rm.dominantRatioLocked()

	// 1. Independent CPU Tasks (CAS, Logic Verification, Mathics)
	if task.IsPureCPU ||
		task.WorkloadClass == MathEvaluation ||
		task.WorkloadClass == MathicsCalculation ||
		task.WorkloadClass == LogicVerification {

		rm.activeCPUWork++
		rm.totalAdmitted++
		rm.logDecision(task.TaskID, task.WorkloadClass, DeviceCPUWorkerPool, StrategyIndependentCPU, domRatio, "routed to CPU worker pool without GPU slot consumption")
		return AdmitDecision{
			Admitted:     true,
			TargetDevice: DeviceCPUWorkerPool,
			Strategy:     StrategyIndependentCPU,
			Reason:       "Independent CPU CAS/Logic task executed on thread pool",
		}
	}

	// 2. Admission Control & Backpressure Check
	capacity := rm.concurrency + rm.queueLimit
	if rm.activeWork >= capacity || domRatio >= 0.95 {
		rm.totalRejected++
		retryAfter := rm.estimateRetryAfterLocked()
		rm.logDecision(task.TaskID, task.WorkloadClass, DeviceGPU, StrategyDirectGPU, domRatio, fmt.Sprintf("rejected: saturation ratio %.2f, active %d >= %d", domRatio, rm.activeWork, capacity))
		return AdmitDecision{
			Admitted:          false,
			TargetDevice:      DeviceGPU,
			Strategy:          StrategyDirectGPU,
			RetryAfterSeconds: retryAfter,
			Reason:            fmt.Sprintf("Server overloaded (dominant saturation: %.1f%%)", domRatio*100),
		}
	}

	// 3. Routing Strategy Determination
	var targetDevice TargetDevice = DeviceGPU
	var strategy ExecutionStrategy = StrategyDirectGPU
	chunkSize := 0
	reason := "standard GPU forward pass"

	switch {
	case task.WorkloadClass == SpeculativeDrafting:
		targetDevice = DeviceCPUWorkerPool
		strategy = StrategyAsyncSpeculative
		reason = "asynchronous CPU speculative drafting"

	case task.IsMoE || task.WorkloadClass == MoeSparse:
		// If PCIe is congested or VRAM is tight, use activation-driven CPU offload
		if domRatio > 0.60 || rm.lastSnapshot.PCIeTransferMBs > 8000.0 {
			targetDevice = DeviceHybridAsync
			strategy = StrategyActivationOffload
			reason = "heterogeneous activation-driven MoE compute to preserve PCIe bandwidth"
		} else {
			targetDevice = DeviceGPU
			strategy = StrategyDirectGPU
			reason = "fast GPU-resident MoE execution"
		}

	case task.PromptTokens > 512 || task.WorkloadClass == PrefillHeavy || task.WorkloadClass == LongContext:
		targetDevice = DeviceGPU
		strategy = StrategyChunkedPrefill
		chunkSize = 256
		if domRatio > 0.75 {
			chunkSize = 128 // throttle chunk size under pressure to maintain decode latency
		}
		reason = fmt.Sprintf("chunked prefill (%d tokens/step) co-batched with decodes", chunkSize)

	case task.WorkloadClass == ShortInteractive || task.WorkloadClass == LowLatency:
		targetDevice = DeviceGPU
		strategy = StrategyDirectGPU
		reason = "prioritized low-latency interactive decode"
	}

	rm.activeWork++
	rm.totalAdmitted++
	rm.logDecision(task.TaskID, task.WorkloadClass, targetDevice, strategy, domRatio, reason)

	return AdmitDecision{
		Admitted:     true,
		TargetDevice: targetDevice,
		Strategy:     strategy,
		ChunkSize:    chunkSize,
		Reason:       reason,
	}
}

// AdmitRequest provides backward compatibility with legacy server calls
func (rm *ResourceManager) AdmitRequest(workloadClass string) (bool, int) {
	decision := rm.RouteTask(TaskDescriptor{
		TaskID:        fmt.Sprintf("req-%d", time.Now().UnixNano()),
		WorkloadClass: workloadClass,
		PromptTokens:  128,
		MaxTokens:     128,
	})
	return decision.Admitted, decision.RetryAfterSeconds
}

// FinishRequest decrements active in-flight count and records completion velocity
func (rm *ResourceManager) FinishRequest() {
	rm.mu.Lock()
	defer rm.mu.Unlock()
	if rm.activeWork > 0 {
		rm.activeWork--
	} else if rm.activeCPUWork > 0 {
		rm.activeCPUWork--
	}
	now := time.Now()
	rm.recentCompletions = append(rm.recentCompletions, now)
	if len(rm.recentCompletions) > 64 {
		rm.recentCompletions = rm.recentCompletions[1:]
	}
}

// FinishCPURequest decrements active CPU-only task count
func (rm *ResourceManager) FinishCPURequest() {
	rm.mu.Lock()
	defer rm.mu.Unlock()
	if rm.activeCPUWork > 0 {
		rm.activeCPUWork--
	}
}

func (rm *ResourceManager) estimateRetryAfterLocked() int {
	now := time.Now()
	// Calculate completions in last 10 seconds
	recentCount := 0
	for _, t := range rm.recentCompletions {
		if now.Sub(t) <= 10*time.Second {
			recentCount++
		}
	}
	if recentCount > 0 {
		rate := float64(recentCount) / 10.0 // completions per second
		if rate > 0 {
			sec := int(math.Ceil(float64(rm.activeWork) / rate))
			if sec < 1 {
				sec = 1
			}
			if sec > 60 {
				sec = 60
			}
			return sec
		}
	}
	return 2
}

func (rm *ResourceManager) logDecision(taskID, wClass string, dev TargetDevice, strat ExecutionStrategy, ratio float64, reason string) {
	record := DecisionRecord{
		Timestamp:     time.Now(),
		TaskID:        taskID,
		WorkloadClass: wClass,
		Device:        dev,
		Strategy:      strat,
		DominantRatio: ratio,
		Reason:        reason,
	}
	if len(rm.decisions) >= rm.maxDecisionsLog {
		rm.decisions = rm.decisions[1:]
	}
	rm.decisions = append(rm.decisions, record)
}

// ConfigureRuntime applies the run config's power policy and resource_adapt switch to the runtime controller.
func (rm *ResourceManager) ConfigureRuntime(policy string, enabled bool) {
	rm.mu.Lock()
	rm.powerPolicy = normalizePolicy(policy)
	rm.Adapt.Configure(policy, enabled)
	rm.mu.Unlock()
}

// Metrics returns observability snapshot for /metrics and /status endpoints
func (rm *ResourceManager) Metrics() map[string]interface{} {
	rm.mu.RLock()
	defer rm.mu.RUnlock()

	decList := make([]map[string]interface{}, 0, len(rm.decisions))
	for _, d := range rm.decisions {
		decList = append(decList, map[string]interface{}{
			"time":           d.Timestamp.Unix(),
			"task_id":        d.TaskID,
			"workload_class": d.WorkloadClass,
			"device":         string(d.Device),
			"strategy":       string(d.Strategy),
			"dominant_ratio": d.DominantRatio,
			"reason":         d.Reason,
		})
	}

	return map[string]interface{}{
		"concurrency":      rm.concurrency,
		"queue_limit":      rm.queueLimit,
		"cache_tier":       rm.cacheTier,
		"power_policy":     rm.powerPolicy,
		"active_work":      rm.activeWork + rm.activeCPUWork,
		"active_gpu_work":  rm.activeWork,
		"active_cpu_work":  rm.activeCPUWork,
		"total_admitted":   rm.totalAdmitted,
		"total_rejected":   rm.totalRejected,
		"dominant_ratio":   rm.dominantRatioLocked(),
		"adapt":            rm.Adapt.Stats(),
		"last_snapshot":    rm.lastSnapshot,
		"recent_decisions": decList,
		"workloads": []string{
			ShortInteractive, LongContext, HighThroughput, LowLatency,
			MemoryBound, ComputeBound, TransferBound, MoeSparse, MoeDense,
			PrefillHeavy, DecodeHeavy, MathEvaluation, MathicsCalculation,
			LogicVerification, SpeculativeDrafting,
		},
	}
}
