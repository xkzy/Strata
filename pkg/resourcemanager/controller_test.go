package resourcemanager

import (
	"context"
	"testing"
	"time"
)

func healthy(at time.Time, concurrency int) Sample {
	return Sample{
		At: at, KnownCPU: true, CPUPercent: 10, KnownRAM: true, RAMUsedPercent: 25,
		RAMAvailableByte: 16 << 30, KnownVRAM: true, VRAMUsedPercent: 20, VRAMFreeBytes: 6 << 30,
		EngineState: "ready", P95TTFTms: 100, PromptP95: 256, DecodeTokS: 40,
	}
}

func newTestController() *Controller {
	cfg := ConfigForPolicy(PolicyBalanced)
	cfg.Enabled = true
	cfg.FastInterval = time.Second
	cfg.MediumInterval = 5 * time.Second
	cfg.SlowInterval = 30 * time.Second
	cfg.MinDwell = 10 * time.Second
	cfg.MonitorWindow = 15 * time.Second
	return NewController(cfg)
}

func mustAdmit(t *testing.T, c *Controller, prompt int) uint64 {
	t.Helper()
	a := c.Admit(context.Background(), prompt)
	if !a.Admitted {
		t.Fatalf("admission refused: %+v", a)
	}
	return a.Ticket
}

func timedAdmit(c *Controller, prompt int, out chan<- Admission) {
	out <- c.Admit(context.Background(), prompt)
}

func TestDisabledControllerAdmitsEverything(t *testing.T) {
	c := NewController(DefaultControllerConfig())
	c.Configure(PolicyBalanced, false)
	for i := 0; i < 100; i++ {
		a := c.Admit(context.Background(), 100000)
		if !a.Admitted {
			t.Fatalf("disabled controller refused request %d: %+v", i, a)
		}
		c.Release(a.Ticket, time.Now())
	}
}

func TestDemandGrowsConcurrencyWithHysteresis(t *testing.T) {
	c := newTestController()
	now := time.Unix(1000, 0)
	var tickets []uint64
	for i := 0; i < 4; i++ {
		tickets = append(tickets, mustAdmit(t, c, 100))
	}
	ch := make(chan Admission, 1)
	go timedAdmit(c, 100, ch)
	waitFor(t, func() bool { return c.Stats().Queued == 1 })

	d := c.Observe(healthy(now, 4))
	if len(d) == 0 || c.Knobs().Concurrency != 5 {
		t.Fatalf("demand with no pressure must grow concurrency: knobs=%+v decisions=%+v", c.Knobs(), d)
	}
	// The waiting request receives the new slot and must be released.
	select {
	case a := <-ch:
		if !a.Admitted {
			t.Fatalf("waiting request refused: %+v", a)
		}
		tickets = append(tickets, a.Ticket)
	case <-time.After(2 * time.Second):
		t.Fatal("waiting request was not woken")
	}
	// A second identical evaluation inside the dwell time does not grow again.
	before := c.Knobs().Concurrency
	if d := c.Observe(healthy(now.Add(2*time.Second), 5)); len(d) != 0 || c.Knobs().Concurrency != before {
		t.Fatalf("dwell time must hold growth: %+v %+v", c.Knobs(), d)
	}
	for _, tk := range tickets {
		c.Release(tk, now)
	}
}

func TestCPUPressureShrinksAndClosesOnlyOnCriticalMemory(t *testing.T) {
	c := newTestController()
	now := time.Unix(2000, 0)
	for i := 0; i < 4; i++ {
		mustAdmit(t, c, 100)
	}
	cpu := healthy(now, 4)
	cpu.CPUPercent = 97
	d := c.Observe(cpu)
	if len(d) == 0 || c.Knobs().Concurrency != 2 {
		t.Fatalf("CPU saturation must halve concurrency: %+v %+v", c.Knobs(), d)
	}
	if !c.Knobs().AdmissionOpen {
		t.Fatal("CPU pressure alone must not close admission")
	}
	ram := healthy(now.Add(2*time.Second), 2)
	ram.RAMUsedPercent = 96
	d = c.Observe(ram)
	if c.Knobs().AdmissionOpen {
		t.Fatalf("critical RAM pressure must close admission: %+v %+v", c.Knobs(), d)
	}
	if a := c.Admit(context.Background(), 10); a.Admitted || a.Reason != "memory_critical" {
		t.Fatalf("closed admission accepted work: %+v", a)
	}
}

func TestQueueLimitAndRetryAfter(t *testing.T) {
	cfg := ConfigForPolicy(PolicyLowLatency)
	cfg.Enabled = true
	cfg.StartConcurrency, cfg.MinConcurrency, cfg.MaxConcurrency = 1, 1, 2
	cfg.StartQueue, cfg.MinQueue, cfg.MaxQueue = 1, 1, 4
	cfg.FastInterval = time.Second
	c := NewController(cfg)
	now := time.Unix(3000, 0)
	first := mustAdmit(t, c, 10)
	ch := make(chan Admission, 1)
	ctx, cancel := context.WithCancel(context.Background())
	go func() { ch <- c.Admit(ctx, 10) }()
	waitFor(t, func() bool { return c.Stats().Queued == 1 })
	third := c.Admit(context.Background(), 10)
	if third.Admitted || third.Reason != "server_overloaded" || third.RetryAfter <= 0 {
		t.Fatalf("queue must be bounded and report retry_after: %+v", third)
	}
	cancel()
	select {
	case a := <-ch:
		if a.Admitted || a.Reason != "request canceled while queued" {
			t.Fatalf("queued request did not observe cancellation: %+v", a)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("queued request did not return after cancellation")
	}
	c.Release(first, now)
	_ = now
}

func TestDeviceFailureClosesAndRecoveryReopens(t *testing.T) {
	c := newTestController()
	now := time.Unix(4000, 0)
	c.Observe(healthy(now, 0)) // establish a healthy sample
	bad := healthy(now.Add(time.Second), 0)
	bad.DeviceFailed = true
	bad.EngineState = "failed"
	bad.HealthReason = "Xid 79"
	c.Observe(bad)
	if c.Knobs().AdmissionOpen {
		t.Fatalf("fatal device failure must close admission: %+v", c.Knobs())
	}
	a := c.Admit(context.Background(), 10)
	if a.Admitted || a.Reason != "device_failure" {
		t.Fatalf("failed device accepted work: %+v", a)
	}
	// Recovery requires a fresh healthy sample; it is not inferred from time passing.
	c.Observe(healthy(now.Add(2*time.Second), 0))
	if !c.Knobs().AdmissionOpen {
		t.Fatalf("a healthy engine sample must reopen admission: %+v", c.Knobs())
	}
}

func TestMonitorRollsBackAWorseGrowth(t *testing.T) {
	c := newTestController()
	now := time.Unix(5000, 0)
	var tickets []uint64
	for i := 0; i < 4; i++ {
		tickets = append(tickets, mustAdmit(t, c, 100))
	}
	ch := make(chan Admission, 1)
	go timedAdmit(c, 100, ch)
	waitFor(t, func() bool { return c.Stats().Queued == 1 })
	c.Observe(healthy(now, 4))
	select {
	case a := <-ch:
		tickets = append(tickets, a.Ticket)
	case <-time.After(2 * time.Second):
		t.Fatal("waiting request was not woken")
	}
	if c.Knobs().Concurrency != 5 {
		t.Fatalf("precondition: expected growth, got %+v", c.Knobs())
	}
	for _, tk := range tickets {
		c.Release(tk, now)
	}
	// The monitor window ends and the measured TTFT is worse than the baseline captured at commit.
	bad := healthy(now.Add(16*time.Second), 0)
	bad.P95TTFTms = 400
	bad.PromptP95 = 256
	d := c.Observe(bad)
	if c.Knobs().Concurrency != 4 || len(d) == 0 || !d[len(d)-1].Rollback {
		t.Fatalf("worse latency after growth must roll back: knobs=%+v decisions=%+v", c.Knobs(), d)
	}
}

func TestStableStateDoesNotOscillate(t *testing.T) {
	c := newTestController()
	now := time.Unix(6000, 0)
	c.Observe(healthy(now, 0))
	for i := 1; i <= 20; i++ {
		d := c.Observe(healthy(now.Add(time.Duration(i)*time.Second), 0))
		if len(d) != 0 {
			t.Fatalf("stable samples produced changes at step %d: %+v", i, d)
		}
	}
}

func TestLongPromptShiftRequestsARetune(t *testing.T) {
	c := newTestController()
	now := time.Unix(7000, 0)
	c.Observe(healthy(now, 0)) // baseline prompt p95 = 256
	later := healthy(now.Add(31*time.Second), 0)
	later.PromptP95 = 1024
	c.Observe(later)
	if !c.RetuneDue() {
		t.Fatalf("a persistent prompt shift must request a bounded retune: %+v", c.Stats())
	}
	if c.RetuneDue() != true {
		t.Fatal("RetuneDue must not consume the request")
	}
	if !c.TakeRetuneDue() {
		t.Fatal("TakeRetuneDue must report and clear a pending request")
	}
	if c.RetuneDue() {
		t.Fatal("TakeRetuneDue must clear the request")
	}
}

func TestPolicyBoundsAndValidator(t *testing.T) {
	c := NewController(ConfigForPolicy(PolicyMaxThroughput))
	if c.Knobs().Concurrency != 8 || c.Knobs().QueueLimit != 128 {
		t.Fatalf("max-throughput policy: %+v", c.Knobs())
	}
	c.SetValidator(func(k Knobs) error { return errTest("engine not ready") })
	now := time.Unix(8000, 0)
	for i := 0; i < 8; i++ {
		mustAdmit(t, c, 10)
	}
	ch := make(chan Admission, 1)
	go timedAdmit(c, 10, ch)
	waitFor(t, func() bool { return c.Stats().Queued == 1 })
	before := c.Knobs().Concurrency
	c.Observe(healthy(now, 8))
	if c.Knobs().Concurrency != before {
		t.Fatalf("validator must block a change: %+v", c.Knobs())
	}
	// Release the admitted work so the queued goroutine can finish instead of leaking into the next test.
	for i := uint64(1); i <= 8; i++ {
		c.Release(i, now)
	}
	select {
	case a := <-ch:
		if !a.Admitted {
			t.Fatalf("queued request after release: %+v", a)
		}
		c.Release(a.Ticket, now)
	case <-time.After(2 * time.Second):
		t.Fatal("queued request was not released")
	}
}

type errTest string

func (e errTest) Error() string { return string(e) }

func waitFor(t *testing.T, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("condition was not reached")
}

func TestUnknownTelemetryDoesNotInventSafetyOrPressure(t *testing.T) {
	c := newTestController()
	now := time.Unix(9000, 0)
	for i := 0; i < 4; i++ {
		mustAdmit(t, c, 10)
	}
	ch := make(chan Admission, 1)
	go timedAdmit(c, 10, ch)
	waitFor(t, func() bool { return c.Stats().Queued == 1 })
	unknown := Sample{At: now, EngineState: "ready"}
	d := c.Observe(unknown)
	if c.Knobs().Concurrency != 4 {
		t.Fatalf("growth must not be inferred without a memory reading: %+v %+v", c.Knobs(), d)
	}
	if !c.Knobs().AdmissionOpen {
		t.Fatal("missing telemetry must not close admission")
	}
	// A later complete reading after the dwell time permits the growth.
	healthyNow := healthy(now.Add(11*time.Second), 4)
	d = c.Observe(healthyNow)
	if c.Knobs().Concurrency != 5 || len(d) == 0 {
		t.Fatalf("a complete reading must allow the pending growth: %+v %+v", c.Knobs(), d)
	}
	select {
	case a := <-ch:
		if a.Admitted {
			c.Release(a.Ticket, now)
		}
	case <-time.After(time.Second):
	}
	for i := uint64(1); i <= 4; i++ {
		c.Release(i, now)
	}
}

func TestVRAMCriticalPressureClosesAdmission(t *testing.T) {
	c := newTestController()
	now := time.Unix(10000, 0)
	c.Observe(healthy(now, 0))
	hot := healthy(now.Add(time.Second), 0)
	hot.VRAMUsedPercent = 97
	c.Observe(hot)
	if c.Knobs().AdmissionOpen {
		t.Fatalf("critical VRAM pressure must stop new work: %+v", c.Knobs())
	}
}

// The 2026-10-19 run against the RTX 5060 Ti alone: the engine's plan filled 15.5 of 16 GiB and left 334 MiB, the
// collector read 97%, and every request was refused with 429 memory_critical. That percentage is what the plan
// allocated, not pressure, and the engine's own floor (256 MiB) was still free.
func TestPlannedVRAMFillIsNotPressure(t *testing.T) {
	now := time.Unix(12000, 0)
	for _, tc := range []struct {
		name    string
		known   bool
		freeMiB uint64
		open    bool
	}{
		{"met plan with 334 MiB free", true, 334, true},
		{"met plan exactly at the 256 MiB floor", true, 256, true},
		{"plan below the engine's stall floor", true, 200, false},
		{"no plan reported: the percentage keeps its old meaning", false, 0, false},
	} {
		c := newTestController()
		c.Observe(healthy(now, 0))
		st := healthy(now.Add(time.Second), 0)
		st.VRAMUsedPercent = 97
		st.KnownVRAMPlan, st.VRAMPlanFreeByte = tc.known, tc.freeMiB<<20
		c.Observe(st)
		if c.Knobs().AdmissionOpen != tc.open {
			t.Errorf("%s: admission open = %v, want %v (%+v)", tc.name, c.Knobs().AdmissionOpen, tc.open, c.Knobs())
		}
	}
}

func TestInsufficientHostMemoryBlocksGrowth(t *testing.T) {
	c := newTestController()
	now := time.Unix(11000, 0)
	for i := 0; i < 4; i++ {
		mustAdmit(t, c, 10)
	}
	ch := make(chan Admission, 1)
	go timedAdmit(c, 10, ch)
	waitFor(t, func() bool { return c.Stats().Queued == 1 })
	tight := healthy(now, 4)
	tight.RAMAvailableByte = 100 << 20 // less than one request + reserve
	c.Observe(tight)
	if c.Knobs().Concurrency != 4 {
		t.Fatalf("growth without host memory is unsafe: %+v", c.Knobs())
	}
	for i := uint64(1); i <= 4; i++ {
		c.Release(i, now)
	}
	select {
	case <-ch:
	case <-time.After(time.Second):
	}
}

func TestLargePromptSlotsAndMarkLarge(t *testing.T) {
	cfg := ConfigForPolicy(PolicyBalanced)
	cfg.Enabled = true
	cfg.StartConcurrency, cfg.MinConcurrency, cfg.MaxConcurrency = 2, 1, 2
	cfg.StartQueue, cfg.MinQueue, cfg.MaxQueue = 8, 1, 8
	cfg.LargePromptTokens = 4096
	cfg.FastInterval, cfg.MediumInterval, cfg.SlowInterval = time.Hour, time.Hour, time.Hour
	c := NewController(cfg)
	now := time.Unix(12000, 0)
	a := mustAdmit(t, c, 100)
	b := mustAdmit(t, c, 100)
	// The runtime path admits before it knows the prompt length, then upgrades the ticket.
	c.MarkLarge(a, 5000)
	c.MarkLarge(b, 5000)
	if c.Stats().InFlight != 2 {
		t.Fatalf("marking must not change the active count: %+v", c.Stats())
	}
	refused := c.Admit(context.Background(), 5000)
	if refused.Admitted || refused.Reason != "prefill_slot_limit" {
		t.Fatalf("all large slots in use must refuse another large prefill: %+v", refused)
	}
	c.Release(a, now)
	next := c.Admit(context.Background(), 5000)
	if !next.Admitted {
		t.Fatalf("a freed large slot must admit the next large prompt: %+v", next)
	}
	c.Release(b, now)
	c.Release(next.Ticket, now)
}

func TestConfigureWakesQueuedRequests(t *testing.T) {
	cfg := ConfigForPolicy(PolicyBalanced)
	cfg.Enabled = true
	cfg.StartConcurrency, cfg.MinConcurrency, cfg.MaxConcurrency = 1, 1, 1
	cfg.StartQueue, cfg.MinQueue, cfg.MaxQueue = 4, 1, 4
	c := NewController(cfg)
	first := mustAdmit(t, c, 10)
	ch := make(chan Admission, 1)
	go timedAdmit(c, 10, ch)
	waitFor(t, func() bool { return c.Stats().Queued == 1 })

	c.Configure(PolicyBalanced, true) // resets the knobs: the waiter must observe the new channel, not hang
	select {
	case a := <-ch:
		if !a.Admitted {
			t.Fatalf("a reconfigured controller must let the queued request proceed: %+v", a)
		}
		c.Release(a.Ticket, time.Now())
	case <-time.After(2 * time.Second):
		t.Fatal("Configure left a waiter asleep on the old notification channel")
	}
	c.Release(first, time.Now())
}

func TestClosingAdmissionRejectsQueuedRequests(t *testing.T) {
	cfg := ConfigForPolicy(PolicyBalanced)
	cfg.Enabled = true
	cfg.StartConcurrency, cfg.MinConcurrency, cfg.MaxConcurrency = 1, 1, 1
	cfg.StartQueue, cfg.MinQueue, cfg.MaxQueue = 4, 1, 4
	c := NewController(cfg)
	first := mustAdmit(t, c, 10)
	ch := make(chan Admission, 1)
	go timedAdmit(c, 10, ch)
	waitFor(t, func() bool { return c.Stats().Queued == 1 })

	hot := healthy(time.Unix(13000, 0), 1)
	hot.RAMUsedPercent = 96
	c.Observe(hot)
	select {
	case a := <-ch:
		if a.Admitted || a.Reason != "memory_critical" {
			t.Fatalf("queued work must be refused when admission closes: %+v", a)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("closing admission did not wake the queued request")
	}
	c.Release(first, time.Unix(13000, 0))
}

func TestRecoveryRequestsBoundedRetune(t *testing.T) {
	c := newTestController()
	now := time.Unix(14000, 0)
	c.Observe(healthy(now, 0))
	bad := healthy(now.Add(time.Second), 0)
	bad.DeviceFailed = true
	bad.EngineState = "failed"
	bad.HealthReason = "Xid 79"
	c.Observe(bad)
	if c.RetuneDue() {
		t.Fatal("a failure alone is not a workload change: no retune yet")
	}
	c.Observe(healthy(now.Add(31*time.Second), 0))
	_ = c.TakeRetuneDue() // may also be set by the prompt-shift rule; clear and check the recovery record
	var sawRecovery bool
	for _, d := range c.Stats().Decisions {
		if d.Reason == "engine recovered to ready: bounded re-measurement requested for the next start" {
			sawRecovery = true
		}
	}
	if !sawRecovery {
		t.Fatalf("a failed -> ready transition must request a bounded re-measurement: %+v", c.Stats().Decisions)
	}
}

func TestMonitorRollsBackAWorseDecodeGrowth(t *testing.T) {
	c := newTestController()
	now := time.Unix(15000, 0)
	var tickets []uint64
	for i := 0; i < 4; i++ {
		tickets = append(tickets, mustAdmit(t, c, 100))
	}
	ch := make(chan Admission, 1)
	go timedAdmit(c, 100, ch)
	waitFor(t, func() bool { return c.Stats().Queued == 1 })
	c.Observe(healthy(now, 4))
	select {
	case a := <-ch:
		tickets = append(tickets, a.Ticket)
	case <-time.After(2 * time.Second):
		t.Fatal("waiting request was not woken")
	}
	if c.Knobs().Concurrency != 5 {
		t.Fatalf("precondition: expected growth, got %+v", c.Knobs())
	}
	for _, tk := range tickets {
		c.Release(tk, now)
	}
	bad := healthy(now.Add(16*time.Second), 0)
	bad.P95TTFTms = 100 // unchanged; only decode throughput regressed
	bad.DecodeTokS = 20 // half of the 40 tok/s captured at commit
	d := c.Observe(bad)
	if c.Knobs().Concurrency != 4 || len(d) == 0 || !d[len(d)-1].Rollback {
		t.Fatalf("a growth that halved decode throughput must roll back: knobs=%+v decisions=%+v", c.Knobs(), d)
	}
}

func TestGradualRAMPressureDegradesBeforeClosing(t *testing.T) {
	c := newTestController()
	now := time.Unix(16000, 0)
	c.Observe(healthy(now, 0)) // 25%: retain

	balanced := healthy(now.Add(5*time.Second), 0)
	balanced.RAMUsedPercent = 75
	c.Observe(balanced)
	if c.Knobs().Tier != TierBalanced || !c.Knobs().AdmissionOpen {
		t.Fatalf("75%% RAM must be a balanced tier, not a shutdown: %+v", c.Knobs())
	}

	for i := 0; i < 4; i++ {
		mustAdmit(t, c, 100)
	}
	tight := healthy(now.Add(10*time.Second), 4)
	tight.RAMUsedPercent = 87
	c.Observe(tight)
	if !c.Knobs().AdmissionOpen {
		t.Fatalf("tight RAM must reduce capacity before closing admission: %+v", c.Knobs())
	}
	if c.Knobs().Concurrency != 2 || c.Knobs().QueueLimit > 32 {
		t.Fatalf("tight RAM must halve concurrency and the queue: %+v", c.Knobs())
	}

	critical := healthy(now.Add(15*time.Second), 2)
	critical.RAMUsedPercent = 94
	c.Observe(critical)
	if c.Knobs().AdmissionOpen {
		t.Fatalf("critical RAM must close admission after the gradual steps: %+v", c.Knobs())
	}
	for i := uint64(1); i <= 4; i++ {
		c.Release(i, now)
	}
}

func TestConcurrentObserveAdmitRelease(t *testing.T) {
	c := newTestController()
	cfg := ConfigForPolicy(PolicyBalanced)
	cfg.Enabled = true
	c.Configure(cfg.Policy, true)
	done := make(chan struct{})
	for w := 0; w < 8; w++ {
		go func(w int) {
			defer func() { done <- struct{}{} }()
			for i := 0; i < 200; i++ {
				a := c.Admit(context.Background(), 10)
				if a.Admitted {
					if i%17 == 0 {
						c.MarkLarge(a.Ticket, 5000)
					}
					c.Release(a.Ticket, time.Now())
				}
			}
		}(w)
	}
	for i := 0; i < 200; i++ {
		s := healthy(time.Unix(17000, 0).Add(time.Duration(i)*time.Second), 0)
		s.CPUPercent = 20 + float64(i%70)
		s.RAMUsedPercent = 30 + float64(i%50)
		c.Observe(s)
	}
	for w := 0; w < 8; w++ {
		select {
		case <-done:
		case <-time.After(10 * time.Second):
			t.Fatal("concurrent admission deadlocked against the control loops")
		}
	}
	if c.Stats().InFlight != 0 {
		t.Fatalf("every ticket must be released: %+v", c.Stats())
	}
}

func TestAdmissionReopenUsesLowerThreshold(t *testing.T) {
	c := newTestController()
	now := time.Unix(18000, 0)
	c.Observe(healthy(now, 0))
	hot := healthy(now.Add(time.Second), 0)
	hot.RAMUsedPercent = 94
	c.Observe(hot)
	if c.Knobs().AdmissionOpen {
		t.Fatalf("critical RAM must close admission: %+v", c.Knobs())
	}
	stillTight := healthy(now.Add(2*time.Second), 0)
	stillTight.RAMUsedPercent = 90 // below critical but still in the tight band
	c.Observe(stillTight)
	if c.Knobs().AdmissionOpen {
		t.Fatalf("reopening at the tight line would flap across the critical line: %+v", c.Knobs())
	}
	clear := healthy(now.Add(3*time.Second), 0)
	clear.RAMUsedPercent = 84
	c.Observe(clear)
	if !c.Knobs().AdmissionOpen {
		t.Fatalf("a reading below the tight line must reopen admission: %+v", c.Knobs())
	}
}

func TestLostRAMTelemetryDoesNotReopenAMemoryClose(t *testing.T) {
	c := newTestController()
	now := time.Unix(18100, 0)
	c.Observe(healthy(now, 0))
	hot := healthy(now.Add(time.Second), 0)
	hot.RAMUsedPercent = 94
	c.Observe(hot)
	if c.Knobs().AdmissionOpen {
		t.Fatalf("precondition: expected a memory close: %+v", c.Knobs())
	}
	unknown := Sample{At: now.Add(2 * time.Second), EngineState: "ready"}
	c.Observe(unknown)
	if c.Knobs().AdmissionOpen {
		t.Fatalf("missing telemetry must not count as recovery from a memory close: %+v", c.Knobs())
	}
}

func TestMonitorDoesNotAttributeAWorkloadShiftToTheReconfiguration(t *testing.T) {
	c := newTestController()
	now := time.Unix(19000, 0)
	var tickets []uint64
	for i := 0; i < 4; i++ {
		tickets = append(tickets, mustAdmit(t, c, 100))
	}
	ch := make(chan Admission, 1)
	go timedAdmit(c, 100, ch)
	waitFor(t, func() bool { return c.Stats().Queued == 1 })
	c.Observe(healthy(now, 4))
	select {
	case a := <-ch:
		tickets = append(tickets, a.Ticket)
	case <-time.After(2 * time.Second):
		t.Fatal("waiting request was not woken")
	}
	for _, tk := range tickets {
		c.Release(tk, now)
	}
	shifted := healthy(now.Add(16*time.Second), 0)
	shifted.PromptP95 = 2048 // 8x the 256-token baseline: the mix, not the extra slot, explains a change
	shifted.P95TTFTms = 400
	d := c.Observe(shifted)
	if c.Knobs().Concurrency != 5 {
		t.Fatalf("a workload shift must not be attributed to the reconfiguration: knobs=%+v decisions=%+v", c.Knobs(), d)
	}
	if len(d) == 0 || d[len(d)-1].Rollback {
		t.Fatalf("expected an explicit inconclusive record, not a rollback: %+v", d)
	}
}
