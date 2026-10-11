// pkg/resourcemanager/resourcemanager_test.go - Unit Tests for Advanced Task Distribution Policy
package resourcemanager

import (
	"testing"
	"time"
)

func TestResourceManager_LegacyLifecycle(t *testing.T) {
	rm := NewResourceManager()

	ok, retry := rm.AdmitRequest(MathEvaluation)
	if !ok || retry != 0 {
		t.Fatalf("expected admission for initial work")
	}

	met := rm.Metrics()
	if met["active_work"].(int) != 1 {
		t.Errorf("expected active work 1, got %v", met["active_work"])
	}

	rm.FinishRequest()
	met2 := rm.Metrics()
	if met2["active_work"].(int) != 0 {
		t.Errorf("expected active work 0, got %v", met2["active_work"])
	}
}

func TestResourceManager_DominantResourceRatio(t *testing.T) {
	rm := NewResourceManager()

	snap := ResourceSnapshot{
		VRAMUsedBytes:     6 << 30, // 6 GiB of 8 GiB = 75%
		VRAMTotalBytes:    8 << 30,
		HostRAMUsedBytes:  16 << 30, // 16 GiB of 32 GiB = 50%
		HostRAMTotalBytes: 32 << 30,
		PCIeTransferMBs:   3000.0,
		PCIeMaxBWMBs:      15000.0, // 20%
		CPULoadPercent:    40.0,    // 40%
		Timestamp:         time.Now(),
	}

	rm.UpdateTelemetry(snap)
	ratio := rm.DominantResourceRatio()

	if ratio < 0.74 || ratio > 0.76 {
		t.Fatalf("expected dominant ratio ~0.75 (from VRAM), got %f", ratio)
	}

	// Change CPU to 90%
	snap.CPULoadPercent = 90.0
	rm.UpdateTelemetry(snap)
	ratio2 := rm.DominantResourceRatio()

	if ratio2 < 0.89 || ratio2 > 0.91 {
		t.Fatalf("expected dominant ratio ~0.90 (from CPU), got %f", ratio2)
	}
}

func TestResourceManager_RoutingPolicies(t *testing.T) {
	rm := NewResourceManager()

	// 1. Pure CPU Task: Mathics / CAS
	mathTask := TaskDescriptor{
		TaskID:        "cas-1",
		WorkloadClass: MathEvaluation,
		IsPureCPU:     true,
	}
	decMath := rm.RouteTask(mathTask)
	if !decMath.Admitted {
		t.Fatalf("expected admission for pure CPU task")
	}
	if decMath.TargetDevice != DeviceCPUWorkerPool || decMath.Strategy != StrategyIndependentCPU {
		t.Fatalf("expected CPU worker pool independent strategy, got: device=%s, strategy=%s", decMath.TargetDevice, decMath.Strategy)
	}

	// 2. Prefill-heavy / Long Context task -> Chunked prefill
	longTask := TaskDescriptor{
		TaskID:        "long-1",
		WorkloadClass: LongContext,
		PromptTokens:  2048,
		MaxTokens:     256,
	}
	decLong := rm.RouteTask(longTask)
	if !decLong.Admitted {
		t.Fatalf("expected admission for long context task")
	}
	if decLong.Strategy != StrategyChunkedPrefill || decLong.ChunkSize != 256 {
		t.Fatalf("expected chunked prefill with size 256, got: strategy=%s, chunk=%d", decLong.Strategy, decLong.ChunkSize)
	}

	// 3. Speculative Drafting -> Asynchronous CPU drafting
	specTask := TaskDescriptor{
		TaskID:        "spec-1",
		WorkloadClass: SpeculativeDrafting,
	}
	decSpec := rm.RouteTask(specTask)
	if !decSpec.Admitted {
		t.Fatalf("expected admission for speculative drafting task")
	}
	if decSpec.TargetDevice != DeviceCPUWorkerPool || decSpec.Strategy != StrategyAsyncSpeculative {
		t.Fatalf("expected async speculative drafting strategy, got: device=%s, strategy=%s", decSpec.TargetDevice, decSpec.Strategy)
	}

	// 4. MoE under PCIe / memory pressure -> Activation offload
	rm.UpdateTelemetry(ResourceSnapshot{
		VRAMUsedBytes:     7 << 30,
		VRAMTotalBytes:    8 << 30, // 87.5% VRAM pressure
		HostRAMTotalBytes: 32 << 30,
		PCIeTransferMBs:   9000.0,
		PCIeMaxBWMBs:      15000.0,
		Timestamp:         time.Now(),
	})
	moeTask := TaskDescriptor{
		TaskID:        "moe-1",
		WorkloadClass: MoeSparse,
		IsMoE:         true,
	}
	decMoE := rm.RouteTask(moeTask)
	if !decMoE.Admitted {
		t.Fatalf("expected admission for MoE task")
	}
	if decMoE.TargetDevice != DeviceHybridAsync || decMoE.Strategy != StrategyActivationOffload {
		t.Fatalf("expected hybrid async activation offload, got: device=%s, strategy=%s", decMoE.TargetDevice, decMoE.Strategy)
	}
}

func TestResourceManager_BackpressureAndRetryAfter(t *testing.T) {
	rm := NewResourceManager()

	// Saturate system to 98%
	rm.UpdateTelemetry(ResourceSnapshot{
		VRAMUsedBytes:     7900 << 20,
		VRAMTotalBytes:    8000 << 20, // 98.7%
		HostRAMTotalBytes: 32 << 30,
		PCIeMaxBWMBs:      15000.0,
		Timestamp:         time.Now(),
	})

	normalTask := TaskDescriptor{
		TaskID:        "blocked-1",
		WorkloadClass: ShortInteractive,
	}
	dec := rm.RouteTask(normalTask)
	if dec.Admitted {
		t.Fatalf("expected rejection under 98%% dominant saturation")
	}
	if dec.RetryAfterSeconds <= 0 {
		t.Fatalf("expected positive retry_after seconds, got %d", dec.RetryAfterSeconds)
	}

	met := rm.Metrics()
	if met["total_rejected"].(int) != 1 {
		t.Fatalf("expected total_rejected == 1, got %v", met["total_rejected"])
	}
}
