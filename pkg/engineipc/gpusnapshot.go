// pkg/engineipc/gpusnapshot.go - what the GPUs and the kernel say at the moment the engine died, written to the
// failure log before the engine is started again. The engine's own output only says "unspecified launch failure";
// whether the card fell off the bus (Xid 79), hit a power or thermal limit, or merely lost its context is only
// visible from outside the process.
package engineipc

import (
	"context"
	"fmt"
	"os/exec"
	"regexp"
	"strings"
	"time"
)

const (
	snapshotTimeout = 6 * time.Second // nvidia-smi can hang on a GPU that is gone
	maxXidLines     = 12
)

var xidMarkers = []string{"Xid", "fallen off the bus", "GPU recovery action", "Node Reboot Required", "uvm encountered global fatal"}

// gpuSnapshot is a seam for tests; it must not need the engine lock.
var gpuSnapshot = collectGPUSnapshot

func runWithTimeout(name string, args ...string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), snapshotTimeout)
	defer cancel()
	out, err := exec.CommandContext(ctx, name, args...).CombinedOutput()
	return strings.TrimSpace(string(out)), err
}

// xidLines keeps the last maxXidLines kernel lines that name a GPU fault (the driver's per-call "rpcSendMessage
// failed" lines repeat twice a second after a fault and would bury the one that matters).
func xidLines(kernelLog string) []string {
	var keep []string
	for _, l := range strings.Split(kernelLog, "\n") {
		for _, m := range xidMarkers {
			if strings.Contains(l, m) {
				keep = append(keep, strings.TrimSpace(l))
				break
			}
		}
	}
	if len(keep) > maxXidLines {
		keep = keep[len(keep)-maxXidLines:]
	}
	return keep
}

// collectGPUSnapshot returns the log lines, each one already prefixed for the failure log.
func collectGPUSnapshot() []string {
	var lines []string
	add := func(label, text string, err error) {
		if text == "" && err != nil {
			text = "unavailable: " + err.Error()
		} else if err != nil {
			text += " (" + err.Error() + ")"
		}
		for _, l := range strings.Split(text, "\n") {
			if l = strings.TrimSpace(l); l != "" {
				lines = append(lines, label+l)
			}
		}
	}
	out, err := runWithTimeout("nvidia-smi", "--query-gpu=index,name,pci.bus_id,temperature.gpu,power.draw,memory.used,memory.total,pcie.link.gen.current,pcie.link.width.current", "--format=csv,noheader")
	if strings.Contains(out, "Unable to determine the device handle") || strings.Contains(out, "Unknown Error") {
		lines = append(lines, "gpu: driver lost device handle — try: sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm")
	}
	add("gpu: ", out, err)
	klog, kerr := runWithTimeout("journalctl", "-k", "-b", "--no-pager", "-n", "3000")
	if kerr != nil || klog == "" {
		klog, kerr = runWithTimeout("dmesg")
	}
	if x := xidLines(klog); len(x) > 0 {
		for _, l := range x {
			lines = append(lines, "kernel: "+l)
		}
	} else if kerr != nil {
		lines = append(lines, "kernel: log unreadable: "+kerr.Error())
	} else {
		lines = append(lines, "kernel: no Xid / fallen-off-the-bus lines this boot")
	}
	return lines
}

// deviceLossMarkers in a snapshot line mean the driver itself lost a card (not a bug in the engine's request).
var deviceLossMarkers = []string{"Xid (PCI:", "fallen off the bus", "Node Reboot Required", "uvm encountered global fatal", "driver lost device handle", "Unable to determine the device handle"}

var (
	pciBusRE     = regexp.MustCompile(`[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}(?:\.[0-9])?`)
	xidRE        = regexp.MustCompile(`Xid \(PCI:([0-9A-Fa-f:.]+)\):\s*(\d+)(?:,\s*(.*))?$`)
	nvidiaRowRE  = regexp.MustCompile(`^gpu:\s*\d+,\s*([^,]+),\s*([0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}(?:\.[0-9])?)`)
	summaryCards = 2
)

// sameBus compares two PCI bus ids where one may lack the domain or the function part (0000:01:00 vs
// 0000:01:00.0): the kernel writes both forms in the same boot.
func sameBus(a, b string) bool {
	if a == b {
		return true
	}
	norm := func(s string) string {
		if !strings.HasPrefix(s, "0000:") {
			s = "0000:" + s
		}
		if !strings.Contains(s, ".") {
			s += ".0"
		}
		return s
	}
	return norm(a) == norm(b)
}

// gpuFaultSummary names the card the driver lost and the first Xid it reported for that card, so the reason a
// client sees says "GPU 0000:01:00.0 (NVIDIA GeForce RTX 5060 Ti) is gone: Xid 79 (GPU has fallen off the bus)"
// instead of only the engine's own text ("cannot pin 322 MiB", "unspecified launch failure"). The first Xid is
// the cause; what follows it (154, UVM 0x60) is a consequence and stays out. A healthy card's name is never
// named. It returns "" when nothing in the snapshot shows a lost or faulted device.
func gpuFaultSummary(lines []string) string {
	type fault struct{ xid, detail string }
	byBus := map[string]fault{}
	var order []string
	nameOf := func(bus string) string { // the snapshot's own nvidia-smi row, when it listed the card at all
		for _, l := range lines {
			if m := nvidiaRowRE.FindStringSubmatch(strings.TrimSpace(l)); m != nil && sameBus(m[2], bus) {
				return strings.TrimSpace(m[1])
			}
		}
		return ""
	}
	busFrom := func(s string) string { return pciBusRE.FindString(s) }
	add := func(bus string, f fault) {
		if bus == "" {
			return
		}
		if prev, seen := byBus[bus]; seen {
			// The first Xid for this card is the cause and outranks a later one; a marker with no Xid (a lost
			// device handle) is upgraded by the Xid that explains it, which may be logged later in the snapshot.
			if prev.xid == "" && f.xid != "" {
				byBus[bus] = f
			}
			return
		}
		byBus[bus] = f
		order = append(order, bus)
	}
	for _, l := range lines {
		if !strings.HasPrefix(l, "kernel: ") && !strings.HasPrefix(l, "gpu: ") {
			continue
		}
		if m := xidRE.FindStringSubmatch(l); m != nil {
			detail := strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(m[3]), "."))
			add(m[1], fault{xid: "Xid " + m[2], detail: detail})
			continue
		}
		for _, marker := range deviceLossMarkers {
			if strings.Contains(l, marker) && !strings.HasPrefix(marker, "Xid") {
				add(busFrom(l), fault{detail: marker})
				break
			}
		}
	}
	var out []string
	for i, bus := range order {
		if i == summaryCards {
			out = append(out, fmt.Sprintf("(+%d more cards)", len(order)-summaryCards))
			break
		}
		f := byBus[bus]
		s := "GPU " + bus
		if name := nameOf(bus); name != "" {
			s += " (" + name + ")"
		}
		s += " is gone: "
		if f.xid != "" {
			s += f.xid
			if f.detail != "" {
				s += " (" + f.detail + ")"
			}
		} else {
			s += f.detail
		}
		out = append(out, s)
	}
	return strings.Join(out, "; ")
}

// deviceLost reports whether the snapshot shows a lost or fatally faulted device, and the line that says so.
func deviceLost(lines []string) (string, bool) {
	for _, l := range lines {
		if !strings.HasPrefix(l, "kernel: ") && !strings.HasPrefix(l, "gpu: ") {
			continue
		}
		for _, m := range deviceLossMarkers {
			if strings.Contains(l, m) {
				return l, true
			}
		}
	}
	return "", false
}

// startGPUSnapshot writes the snapshot to the failure log in the background: nvidia-smi can hang on a GPU that is
// gone, so its caller (afterFailure, which holds eo.mu) must not wait for it. Stop waits for it (snapshotWG), so no
// write lands after shutdown. Call it without eo.mu held.
func (eo *EngineOrchestrator) startGPUSnapshot(when string) {
	eo.snapshotMu.Lock()
	if eo.stopping {
		eo.snapshotMu.Unlock()
		return
	}
	eo.snapshotWG.Add(1)
	eo.snapshotMu.Unlock()
	go func() {
		defer eo.snapshotWG.Done()
		eo.writeGPUSnapshot(when)
	}()
}

// writeGPUSnapshot appends the snapshot to the failure log and returns its lines. Call it without eo.mu held.
func (eo *EngineOrchestrator) writeGPUSnapshot(when string) []string {
	var b strings.Builder
	b.WriteString(time.Now().Format(time.RFC3339) + " GPU state " + when + "\n")
	lines := gpuSnapshot()
	for _, l := range lines {
		b.WriteString("    | " + l + "\n")
	}
	eo.appendFailureLog(b.String())
	eo.nameLostCard(lines)
	return lines
}

// nameLostCard appends what the snapshot says about a lost card (its bus id, and the Xid when the kernel logged
// one) to the reason a client sees. The engine's own output on a fall-off is only "unspecified launch failure" or
// "cannot pin 322 MiB"; the card and the Xid are visible from outside the process. Call it without eo.mu held.
func (eo *EngineOrchestrator) nameLostCard(lines []string) {
	summary := gpuFaultSummary(lines)
	if summary == "" {
		return
	}
	eo.mu.Lock()
	if eo.state == StateFailed && !strings.Contains(eo.reason, summary) {
		if eo.reason == "" {
			eo.reason = summary
		} else {
			eo.reason += "; " + summary
		}
	}
	eo.mu.Unlock()
}
