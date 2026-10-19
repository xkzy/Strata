package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"strata/pkg/resourcemanager"
)

func TestRuntimeAdmissionClosedReturns429WithRetry(t *testing.T) {
	srv := NewStrataServer(ServerConfig{Port: 0})
	srv.ResourceManager.ConfigureRuntime("BALANCED", true)
	c := srv.ResourceManager.Adapt
	c.Observe(resourcemanager.Sample{At: time.Now(), DeviceFailed: true, EngineState: "failed", HealthReason: "Xid 79"})

	ticket, aerr := srv.admitRequest(context.Background(), 10)
	if aerr == nil || aerr.Status != http.StatusTooManyRequests {
		t.Fatalf("closed admission must return 429: ticket=%d err=%+v", ticket, aerr)
	}
	if aerr.RetryAfter <= 0 || aerr.Type != "server_overloaded" {
		t.Errorf("admission error must say why and when to retry: %+v", aerr)
	}
}

func TestRuntimeAdmissionDisabledKeepsOldBehaviour(t *testing.T) {
	srv := NewStrataServer(ServerConfig{Port: 0})
	srv.ResourceManager.ConfigureRuntime("BALANCED", false)
	ticket, aerr := srv.admitRequest(context.Background(), 10)
	if aerr != nil || ticket == 0 {
		t.Fatalf("disabled adaptation must admit: %d %+v", ticket, aerr)
	}
	srv.releaseRequest(ticket)
}

func TestRuntimeMetricsExposeController(t *testing.T) {
	srv := NewStrataServer(ServerConfig{Port: 0})
	rec := httptest.NewRecorder()
	srv.handleMetrics(rec, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("metrics status %d", rec.Code)
	}
	var body map[string]interface{}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	res, ok := body["resource"].(map[string]interface{})
	if !ok {
		t.Fatalf("no resource section: %v", body)
	}
	adapt, ok := res["adapt"].(map[string]interface{})
	if !ok {
		t.Fatalf("no adaptation section: %v", res)
	}
	if _, ok := adapt["knobs"]; !ok {
		t.Errorf("adaptation section must show the knobs: %v", adapt)
	}
	if _, ok := adapt["decisions"]; !ok {
		t.Errorf("adaptation section must show decisions: %v", adapt)
	}
}

func TestOverloadErrorSetsRetryAfterForBothAPIs(t *testing.T) {
	rec := httptest.NewRecorder()
	writeAPIError(rec, &apiError{Status: http.StatusTooManyRequests, Type: "server_overloaded", Msg: "busy", RetryAfter: 7})
	if got := rec.Header().Get("Retry-After"); got != "7" {
		t.Errorf("openai Retry-After = %q", got)
	}
	rec = httptest.NewRecorder()
	anError(rec, &apiError{Status: http.StatusTooManyRequests, Type: "server_overloaded", Msg: "busy", RetryAfter: 9})
	if got := rec.Header().Get("Retry-After"); got != "9" {
		t.Errorf("anthropic Retry-After = %q", got)
	}
}

func TestStopRuntimeCancelsBackgroundExperiments(t *testing.T) {
	srv := NewStrataServer(ServerConfig{Port: 0})
	srv.StopRuntime()
	select {
	case <-srv.lifecycleCtx.Done():
	case <-time.After(time.Second):
		t.Fatal("StopRuntime must cancel a running startup/refinement experiment")
	}
	srv.StopRuntime() // idempotent: a second shutdown must not panic
}

// A GPU whose driver stopped answering publishes no live values (see pkg/telemetry); that absence must read as
// unknown, not as "0% VRAM used", which would be a permission to grow.
func TestApplyTelemetryTreatsMissingVRAMAsUnknown(t *testing.T) {
	from := func(now map[string]interface{}) resourcemanager.Sample {
		var st resourcemanager.Sample
		applyTelemetry(&st, map[string]interface{}{"now": now})
		return st
	}
	if st := from(map[string]interface{}{"gpu_mem_total": uint64(16 << 30)}); st.KnownVRAM {
		t.Errorf("a total without a live reading is unknown, not empty: %+v", st)
	}
	st := from(map[string]interface{}{"gpu_mem_total": uint64(16 << 30), "gpu_mem_used": uint64(12 << 30)})
	if !st.KnownVRAM || st.VRAMUsedPercent != 75 || st.VRAMFreeBytes != 4<<30 {
		t.Errorf("a complete reading must be known: %+v", st)
	}
	st = from(map[string]interface{}{"ram_total": uint64(64 << 30), "ram_used": uint64(16 << 30), "ram_system": true})
	if !st.KnownRAM || st.RAMUsedPercent != 25 || st.RAMAvailableByte != 48<<30 {
		t.Errorf("system RAM must be known: %+v", st)
	}
	if st := from(map[string]interface{}{"ram_total": uint64(64 << 30), "ram_used": uint64(16 << 30), "ram_system": false}); st.KnownRAM {
		t.Errorf("the Go heap must not be read as host RAM: %+v", st)
	}
	if st := from(map[string]interface{}{}); st.KnownVRAM || st.KnownRAM || st.KnownCPU {
		t.Errorf("an empty sample knows nothing: %+v", st)
	}
}

// The engine's ready-time plan is the only place that says how much VRAM it meant to leave free: the 2026-10-19
// 5060 Ti run left 334 MiB, read 97% used, and the controller refused every request until the plan was taken into
// account (see resourcemanager's TestPlannedVRAMFillIsNotPressure for the admission outcome).
func TestApplyEnginePlan(t *testing.T) {
	var st resourcemanager.Sample
	applyEnginePlan(&st, map[string]string{"vram_free_mib": "334", "engine": "0.1.40"})
	if !st.KnownVRAMPlan || st.VRAMPlanFreeByte != 334<<20 {
		t.Fatalf("a ready engine's plan must be recorded: %+v", st)
	}
	for _, info := range []map[string]string{
		{},
		{"vram_free_mib": ""},
		{"vram_free_mib": "n/a"},
		{"vram_free_mib": "-1"},
	} {
		st := resourcemanager.Sample{}
		applyEnginePlan(&st, info)
		if st.KnownVRAMPlan {
			t.Errorf("no usable vram_free_mib in %v must leave the plan unknown: %+v", info, st)
		}
	}
}
