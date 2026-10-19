package telemetry

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeNvidia replaces the nvidia-smi seam for one test and records the calls it saw. It returns a valid query
// line for --query-gpu and a dmon block for dmon, unless err is set.
type fakeNvidia struct {
	mu    sync.Mutex
	calls []string
	err   error
}

func (f *fakeNvidia) run(ctx context.Context, args ...string) ([]byte, error) {
	f.mu.Lock()
	f.calls = append(f.calls, strings.Join(args, " "))
	f.mu.Unlock()
	if f.err != nil {
		return nil, f.err
	}
	if args[0] == "dmon" {
		return []byte("# gpu  rxpci  txpci\n# Idx   MB/s   MB/s\n    0     12     34\n"), nil
	}
	return []byte("1024, 16311, 7, 41, 55.5, 4, 4, 8\n"), nil
}

func (f *fakeNvidia) count(prefix string) int {
	f.mu.Lock()
	defer f.mu.Unlock()
	n := 0
	for _, c := range f.calls {
		if strings.HasPrefix(c, prefix) {
			n++
		}
	}
	return n
}

// install swaps the exec seam and clears the probe caches, restoring both when the test ends.
func installFakeNvidia(t *testing.T, fake *fakeNvidia) {
	t.Helper()
	resetNvidiaProbes()
	old := nvidiaExec
	nvidiaExec = fake.run
	t.Cleanup(func() {
		nvidiaExec = old
		resetNvidiaProbes()
	})
}

func resetNvidiaProbes() {
	nvidiaMu.Lock()
	defer nvidiaMu.Unlock()
	nvidiaQueryAt, nvidiaQuerySmp, nvidiaQueryOK = time.Time{}, nvSample{}, false
	nvidiaPcieAt, nvidiaPcieRx, nvidiaPcieTx, nvidiaPcieOK = time.Time{}, 0, 0, false
}

// Two collectors in one process must not each fork nvidia-smi inside the share window.
func TestGPUQueryIsSharedBetweenCollectors(t *testing.T) {
	fake := &fakeNvidia{}
	installFakeNvidia(t, fake)

	a, b := NewTelemetryCollector(), NewTelemetryCollector()
	a.sampleOnce()
	b.sampleOnce()

	if got := fake.count("--query-gpu"); got != 1 {
		t.Errorf("two collectors in one window ran %d nvidia-smi queries, want 1", got)
	}
}

// A sample must reach a GPU reading from the shared probe, and the collector keeps the card's PCIe link.
func TestSamplePublishesTheProbedGPUReading(t *testing.T) {
	fake := &fakeNvidia{}
	installFakeNvidia(t, fake)

	tc := NewTelemetryCollector()
	tc.sampleOnce()
	now := tc.Snapshot()["now"].(map[string]interface{})
	if got, _ := now["gpu_mem_used"].(uint64); got != 1024<<20 {
		t.Errorf("gpu_mem_used = %d, want the probed usage", got)
	}
	if got, _ := now["gpu_mem_total"].(uint64); got != 16311<<20 {
		t.Errorf("gpu_mem_total = %d, want the probed total", got)
	}
	if got, _ := now["gpu_util"].(float64); got != 7 {
		t.Errorf("gpu_util = %v, want the probed value", got)
	}
	if tc.pcie != [3]int{4, 4, 8} {
		t.Errorf("pcie link = %v, want the probed gen/width", tc.pcie)
	}
}

// A missing or faulted driver must not be re-probed every second: nvidia-smi can take seconds to fail, and the
// CPU/RAM samples have to keep arriving.
func TestFailedGPUProbeIsNotRepeatedEverySample(t *testing.T) {
	fake := &fakeNvidia{err: errors.New("nvidia-smi: driver fault")}
	installFakeNvidia(t, fake)

	base := time.Now()
	if _, ok := nvidiaGPUQuery(base); ok {
		t.Fatal("a failing probe must report not-ok")
	}
	if _, ok := nvidiaGPUQuery(base.Add(time.Second)); ok {
		t.Fatal("still failing")
	}
	if got := fake.count("--query-gpu"); got != 1 {
		t.Errorf("a failure inside the retry window ran %d queries, want 1", got)
	}
	// After the window a recovery is noticed.
	nvidiaExec = func(ctx context.Context, args ...string) ([]byte, error) {
		return []byte("2048, 16311, 12, 42, 60.0, 4, 4, 8\n"), nil
	}
	smp, ok := nvidiaGPUQuery(base.Add(6 * time.Second))
	if !ok || smp.usedBytes != 2048<<20 {
		t.Errorf("after the retry window the probe must recover: %v %v", smp, ok)
	}
}

// dmon costs about 117 ms - a whole sampling period - so it runs on its own slower cadence and reuses the last
// reading in between.
func TestPCIeProbeIsRateLimitedAndKeepsTheLastReading(t *testing.T) {
	fake := &fakeNvidia{}
	installFakeNvidia(t, fake)

	base := time.Now()
	rx, tx, ok := nvidiaPCIeThroughput(base)
	if !ok || rx != 12 || tx != 34 {
		t.Fatalf("first probe = %v %v %v", rx, tx, ok)
	}
	for s := 1; s <= 4; s++ {
		rx, tx, ok = nvidiaPCIeThroughput(base.Add(time.Duration(s) * time.Second))
		if !ok || rx != 12 || tx != 34 {
			t.Errorf("second %d reused reading = %v %v %v", s, rx, tx, ok)
		}
	}
	if got := fake.count("dmon"); got != 1 {
		t.Errorf("five samples in five seconds ran %d dmon probes, want 1", got)
	}
	if _, _, ok := nvidiaPCIeThroughput(base.Add(5 * time.Second)); !ok {
		t.Error("a probe after the interval must refresh")
	}
	if got := fake.count("dmon"); got != 2 {
		t.Errorf("after the interval dmon ran %d times, want 2", got)
	}
}

// A driver that hangs must not freeze the sampler: the probe returns at its deadline.
func TestHangingNvidiaSmiHitsItsDeadline(t *testing.T) {
	installFakeNvidia(t, &fakeNvidia{})
	oldQuery, oldPcie := nvidiaQueryTimeout, nvidiaPcieTimeout
	nvidiaQueryTimeout, nvidiaPcieTimeout = 50*time.Millisecond, 50*time.Millisecond
	t.Cleanup(func() { nvidiaQueryTimeout, nvidiaPcieTimeout = oldQuery, oldPcie })
	nvidiaExec = func(ctx context.Context, args ...string) ([]byte, error) {
		<-ctx.Done() // a wedged driver: no output, no exit
		return nil, ctx.Err()
	}

	done := make(chan time.Duration, 1)
	go func() {
		start := time.Now()
		if _, ok := nvidiaGPUQuery(time.Now()); ok {
			t.Error("a hung probe must report not-ok")
		}
		done <- time.Since(start)
	}()
	select {
	case took := <-done:
		if took > 2*time.Second {
			t.Errorf("probe took %v, want the deadline to end it", took)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("a hanging nvidia-smi blocked the sampler")
	}
}

func TestParseDmonThroughputSumsTheCards(t *testing.T) {
	block := "# gpu  rxpci  txpci\n# Idx   MB/s   MB/s\n    0     10     20\n    1      2      3\n"
	rx, tx, ok := parseDmonThroughput(block)
	if !ok || rx != 12 || tx != 23 {
		t.Errorf("parseDmonThroughput = %v %v %v", rx, tx, ok)
	}
	if _, _, ok := parseDmonThroughput("# gpu  rxpci  txpci\n"); ok {
		t.Error("a header-only block must report not-ok")
	}
}
