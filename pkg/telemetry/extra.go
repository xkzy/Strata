// pkg/telemetry/extra.go - disk throughput, integrated GPU and PCIe link readings for the Monitor tab
package telemetry

import (
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
)

const sectorBytes = 512

var wholeDisk = regexp.MustCompile(`^(nvme\d+n\d+|sd[a-z]+|vd[a-z]+|xvd[a-z]+|hd[a-z]+|mmcblk\d+)$`)

// parseDiskstats sums the sectors read and written by whole physical disks. Partitions, loop, dm, md and zram
// devices are skipped: they would count the same bytes twice or are not disks.
func parseDiskstats(content string) (readSectors, writeSectors uint64) {
	for _, line := range strings.Split(content, "\n") {
		f := strings.Fields(line)
		if len(f) < 10 || !wholeDisk.MatchString(f[2]) {
			continue
		}
		r, err1 := strconv.ParseUint(f[5], 10, 64)
		w, err2 := strconv.ParseUint(f[9], 10, 64)
		if err1 != nil || err2 != nil {
			continue
		}
		readSectors += r
		writeSectors += w
	}
	return
}

// diskRates turns the running sector counters into MB/s since the previous call. ok is false on the first
// call and when a counter went backwards (a disk disappeared).
func (tc *TelemetryCollector) diskRates(readSectors, writeSectors uint64, at time.Time) (readMB, writeMB float64, ok bool) {
	havePrev, pr, pw, pt := tc.haveDisk, tc.prevDiskRead, tc.prevDiskWrite, tc.prevDiskAt
	tc.haveDisk, tc.prevDiskRead, tc.prevDiskWrite, tc.prevDiskAt = true, readSectors, writeSectors, at
	dt := at.Sub(pt).Seconds()
	if !havePrev || dt <= 0 || readSectors < pr || writeSectors < pw {
		return 0, 0, false
	}
	const mib = 1 << 20
	return float64(readSectors-pr) * sectorBytes / dt / mib, float64(writeSectors-pw) * sectorBytes / dt / mib, true
}

func (tc *TelemetryCollector) readDisk() (readMB, writeMB float64, ok bool) {
	data, err := os.ReadFile("/proc/diskstats")
	if err != nil {
		return 0, 0, false
	}
	r, w := parseDiskstats(string(data))
	return tc.diskRates(r, w, time.Now())
}

// drmGPU is one /sys/class/drm/cardN/device directory.
type drmGPU struct{ dir string }

var cardDir = regexp.MustCompile(`^card\d+$`)

func readUint(path string) (uint64, bool) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, false
	}
	v, err := strconv.ParseUint(strings.TrimSpace(string(b)), 10, 64)
	return v, err == nil
}

// discoverIGPU returns the integrated GPU under a DRM root: a card with a large GTT and a small
// (at most 2 GiB) VRAM carve-out, which is how an APU shows up. Discrete cards are not returned.
func discoverIGPU(root string) *drmGPU {
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil
	}
	for _, e := range entries {
		if !cardDir.MatchString(e.Name()) {
			continue
		}
		dir := filepath.Join(root, e.Name(), "device")
		vram, ok := readUint(filepath.Join(dir, "mem_info_vram_total"))
		if !ok {
			continue
		}
		gtt, _ := readUint(filepath.Join(dir, "mem_info_gtt_total"))
		if gtt > 0 && vram <= 2<<30 {
			return &drmGPU{dir: dir}
		}
	}
	return nil
}

func (g *drmGPU) name() string {
	if b, err := os.ReadFile(filepath.Join(g.dir, "product_name")); err == nil {
		if n := strings.TrimSpace(string(b)); n != "" {
			return n
		}
	}
	return "Integrated Graphics"
}

func (g *drmGPU) read() map[string]interface{} {
	out := map[string]interface{}{}
	if v, ok := readUint(filepath.Join(g.dir, "gpu_busy_percent")); ok {
		out["util"] = float64(v)
	}
	for key, file := range map[string]string{"mem_used": "mem_info_vram_used", "mem_total": "mem_info_vram_total",
		"gtt_used": "mem_info_gtt_used", "gtt_total": "mem_info_gtt_total"} {
		if v, ok := readUint(filepath.Join(g.dir, file)); ok {
			out[key] = v
		}
	}
	return out
}

// nvSample is one line of the nvidia-smi query in nvidiaSmiQuery.
type nvSample struct {
	usedBytes, totalBytes          uint64
	util, temp, power              float64
	hasPower                       bool
	pcieGen, pcieGenMax, pcieWidth int
}

const nvidiaSmiQuery = "memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current"

func parseNvidiaSmiLine(line string) (nvSample, bool) {
	parts := strings.Split(line, ",")
	if len(parts) < 4 {
		return nvSample{}, false
	}
	f := func(i int) string {
		if i >= len(parts) {
			return ""
		}
		return strings.TrimSpace(parts[i])
	}
	used, err1 := strconv.ParseUint(f(0), 10, 64)
	total, err2 := strconv.ParseUint(f(1), 10, 64)
	if err1 != nil || err2 != nil {
		return nvSample{}, false
	}
	s := nvSample{usedBytes: used << 20, totalBytes: total << 20}
	s.util, _ = strconv.ParseFloat(f(2), 64)
	s.temp, _ = strconv.ParseFloat(f(3), 64)
	p, err := strconv.ParseFloat(f(4), 64) // "[N/A]": the card does not report power
	s.power, s.hasPower = p, err == nil
	s.pcieGen, _ = strconv.Atoi(f(5))
	s.pcieGenMax, _ = strconv.Atoi(f(6))
	s.pcieWidth, _ = strconv.Atoi(f(7))
	return s, true
}
