// pkg/telemetry/nvprobe.go - shared, rate-limited nvidia-smi probes
//
// The 1 Hz sampler used to fork nvidia-smi twice on every tick: one --query-gpu call for memory, utilisation,
// temperature and the PCIe link, and one `dmon` run for the PCIe throughput. Measured on the development PC
// (AMD Ryzen 9 9950X3D, RTX 5060 Ti + RTX 4060, Linux, 20 calls each): the query costs 33.6 ms and dmon 115.4 ms,
// so one sample spent 149 ms of its 1000 ms period in child processes - about 15% of one core, repeated by every
// collector in the process. Worst case it is worse than a delay: a driver that stops answering makes nvidia-smi
// block, which used to stall the sampler and leave the controller deciding on a frozen reading. The probes are
// shared between collectors here, rate-limited, and run under a deadline.
package telemetry

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

var errNvidiaSmiOff = errors.New("nvidia-smi probes are off (STRATA_NO_NVIDIA_SMI=1)")

var (
	nvidiaMu sync.Mutex

	nvidiaQueryAt  time.Time
	nvidiaQuerySmp nvSample
	nvidiaQueryOK  bool

	nvidiaPcieAt time.Time
	nvidiaPcieRx float64
	nvidiaPcieTx float64
	nvidiaPcieOK bool

	// nvidiaExec runs nvidia-smi under a deadline. A driver in a bad state can make nvidia-smi block for a long
	// time; without a deadline the sampler goroutine would stop producing samples entirely, and the controller
	// would keep deciding on a frozen last reading. The seam takes the context so a test can prove the deadline.
	nvidiaExec = func(ctx context.Context, args ...string) ([]byte, error) {
		cmd := exec.CommandContext(ctx, "nvidia-smi", args...)
		cmd.WaitDelay = time.Second // a call stuck in the driver can ignore the kill and keep its pipes open
		return cmd.Output()
	}

	// Deadlines, not constants, so a test can shorten them instead of waiting out a real hang.
	nvidiaQueryTimeout = 8 * time.Second
	nvidiaPcieTimeout  = 3 * time.Second
)

const (
	// A successful query is shared for a little under one sampling period. One collector still refreshes it on
	// most ticks on purpose - VRAM used is the input to the admission decision and should not be older than a
	// period - but several collectors in one process (the tests build many, a multi-model server can too) do not
	// each fork nvidia-smi.
	nvidiaQueryShare = 900 * time.Millisecond
	// A missing or faulted driver is not retried every second: nvidia-smi can take seconds to fail, and the
	// CPU/RAM samples must keep arriving meanwhile.
	nvidiaQueryFail = 5 * time.Second

	// dmon occupies a whole sampling period of its own (117 ms measured), and the PCIe sparkline does not need a
	// point per second. Between probes the last good reading is reused, so the history stays continuous.
	nvidiaPcieInterval = 5 * time.Second
)

// runNvidia executes one nvidia-smi probe under its deadline.
func runNvidia(timeout time.Duration, args ...string) ([]byte, error) {
	if os.Getenv("STRATA_NO_NVIDIA_SMI") == "1" {
		// Diagnostic switch: no nvidia-smi child while the engine runs (the driver answers each probe through the
		// same GSP firmware the compute work uses). The GPU charts stay empty.
		return nil, errNvidiaSmiOff
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	return nvidiaExec(ctx, args...)
}

// nvidiaGPUQuery returns the nvidia-smi sample, refreshing it when the cached one is older than its window. One
// probe serves the whole process: a second caller inside a window waits for the first instead of starting another
// nvidia-smi process.
func nvidiaGPUQuery(now time.Time) (nvSample, bool) {
	nvidiaMu.Lock()
	defer nvidiaMu.Unlock()

	window := nvidiaQueryShare
	if !nvidiaQueryOK {
		window = nvidiaQueryFail // no GPU or a faulted driver: probe less often, but do notice a recovery
	}
	if !nvidiaQueryAt.IsZero() && now.Sub(nvidiaQueryAt) < window {
		return nvidiaQuerySmp, nvidiaQueryOK
	}

	out, err := runNvidia(nvidiaQueryTimeout, "--query-gpu="+nvidiaSmiQuery, "--format=csv,noheader,nounits")
	nvidiaQueryAt = now
	if err != nil {
		nvidiaQueryOK = false
		return nvSample{}, false
	}
	line := strings.TrimSpace(string(out))
	if idx := strings.Index(line, "\n"); idx != -1 {
		line = line[:idx] // GPU 0: the collector reports one card's totals
	}
	smp, ok := parseNvidiaSmiLine(line)
	nvidiaQueryOK = ok
	if ok {
		nvidiaQuerySmp = smp
	}
	return smp, ok
}

// nvidiaPCIeThroughput returns the PCIe RX and TX rates in MB/s, refreshing them at most once per
// nvidiaPcieInterval. Until the first probe succeeds, ok stays false and nothing is published.
func nvidiaPCIeThroughput(now time.Time) (rxMB, txMB float64, ok bool) {
	nvidiaMu.Lock()
	defer nvidiaMu.Unlock()

	if !nvidiaPcieAt.IsZero() && now.Sub(nvidiaPcieAt) < nvidiaPcieInterval {
		return nvidiaPcieRx, nvidiaPcieTx, nvidiaPcieOK
	}
	out, err := runNvidia(nvidiaPcieTimeout, "dmon", "-s", "t", "-c", "1")
	nvidiaPcieAt = now
	if err != nil {
		nvidiaPcieOK = false
		return 0, 0, false
	}
	rxMB, txMB, ok = parseDmonThroughput(string(out))
	if ok {
		nvidiaPcieRx, nvidiaPcieTx, nvidiaPcieOK = rxMB, txMB, true
	} else {
		nvidiaPcieOK = false
	}
	return rxMB, txMB, ok
}

// parseDmonThroughput sums the per-GPU rxpci/txpci columns of one `nvidia-smi dmon -s t` block.
func parseDmonThroughput(out string) (rxMB, txMB float64, ok bool) {
	for _, line := range strings.Split(out, "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "#") || trimmed == "" {
			continue
		}
		fields := strings.Fields(trimmed)
		if len(fields) >= 3 {
			rx, errRx := strconv.ParseFloat(fields[1], 64)
			tx, errTx := strconv.ParseFloat(fields[2], 64)
			if errRx == nil && errTx == nil {
				rxMB += rx
				txMB += tx
				ok = true
			}
		}
	}
	return rxMB, txMB, ok
}
