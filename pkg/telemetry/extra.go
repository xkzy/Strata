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

// readAMDSysfs is the fallback card: the DRM card with the most VRAM, read from its device directory. Its
// temperature and power come from the card's hwmon when the driver exposes them (temp1_input in millidegrees,
// power1_average in microwatts); when it does not, they are reported as absent. The old path returned a fixed
// 45 C and 50 W, which is worse than no number at all: the controller then believed a measurement that nobody
// took.
func readAMDSysfs(root string) gpuReading {
	entries, err := os.ReadDir(root)
	if err != nil {
		return gpuReading{}
	}
	var best gpuReading
	for _, e := range entries {
		if !cardDir.MatchString(e.Name()) {
			continue
		}
		dir := filepath.Join(root, e.Name(), "device")
		tot, _ := readUint(filepath.Join(dir, "mem_info_vram_total"))
		if tot == 0 || tot < best.total {
			continue
		}
		r := gpuReading{present: true, total: tot}
		r.used, _ = readUint(filepath.Join(dir, "mem_info_vram_used"))
		if u, ok := readUint(filepath.Join(dir, "gpu_busy_percent")); ok {
			r.util = float64(u)
		}
		r.temp, r.hasTemp = readHwmonValue(dir, "temp1_input", 1000)     // millidegrees -> C
		r.power, r.hasPower = readHwmonValue(dir, "power1_average", 1e6) // microwatts -> W
		if !r.hasPower {
			r.power, r.hasPower = readHwmonValue(dir, "power1_input", 1e6)
		}
		best = r
	}
	return best
}

// readHwmonValue reads one of the card's hwmon files (any hwmonN under its device) and scales it to the unit
// the caller wants.
func readHwmonValue(deviceDir, file string, scale float64) (float64, bool) {
	dirs, err := filepath.Glob(filepath.Join(deviceDir, "hwmon", "hwmon*"))
	if err != nil {
		return 0, false
	}
	for _, d := range dirs {
		if v, ok := readUint(filepath.Join(d, file)); ok {
			return float64(v) / scale, true
		}
	}
	return 0, false
}

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
	hasTemp, hasPower              bool
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
	t, errT := strconv.ParseFloat(f(3), 64) // "[N/A]": the card does not report it
	s.temp, s.hasTemp = t, errT == nil
	p, err := strconv.ParseFloat(f(4), 64) // "[N/A]": the card does not report power
	s.power, s.hasPower = p, err == nil
	s.pcieGen, _ = strconv.Atoi(f(5))
	s.pcieGenMax, _ = strconv.Atoi(f(6))
	s.pcieWidth, _ = strconv.Atoi(f(7))
	return s, true
}

func parseGtToGen(gtStr string) int {
	gtStr = strings.ToLower(strings.TrimSpace(gtStr))
	if strings.Contains(gtStr, "64.0") {
		return 6
	} else if strings.Contains(gtStr, "32.0") {
		return 5
	} else if strings.Contains(gtStr, "16.0") {
		return 4
	} else if strings.Contains(gtStr, "8.0") {
		return 3
	} else if strings.Contains(gtStr, "5.0") {
		return 2
	} else if strings.Contains(gtStr, "2.5") {
		return 1
	}
	if n, err := strconv.Atoi(gtStr); err == nil && n >= 1 && n <= 6 {
		return n
	}
	return 0
}

func discoverDrmPCIe(root string) (gen, maxGen, width, maxWidth int) {
	entries, err := os.ReadDir(root)
	if err != nil {
		return 0, 0, 0, 0
	}
	bestVRAM := uint64(0)
	for _, e := range entries {
		if !cardDir.MatchString(e.Name()) {
			continue
		}
		dir := filepath.Join(root, e.Name(), "device")
		vram, _ := readUint(filepath.Join(dir, "mem_info_vram_total"))

		curSpeed, _ := os.ReadFile(filepath.Join(dir, "current_link_speed"))
		maxSpeed, _ := os.ReadFile(filepath.Join(dir, "max_link_speed"))
		curWidth, _ := readUint(filepath.Join(dir, "current_link_width"))
		maxW, _ := readUint(filepath.Join(dir, "max_link_width"))

		g := parseGtToGen(string(curSpeed))
		mg := parseGtToGen(string(maxSpeed))
		w := int(curWidth)
		mw := int(maxW)

		if g > 0 || mg > 0 || w > 0 {
			if vram >= bestVRAM || gen == 0 {
				bestVRAM = vram
				gen = g
				maxGen = mg
				width = w
				maxWidth = mw
			}
		}
	}
	return
}

func PcieBandwidthGBps(gen, width int) float64 {
	var perLaneGBps float64
	switch gen {
	case 1:
		perLaneGBps = 0.250
	case 2:
		perLaneGBps = 0.500
	case 3:
		perLaneGBps = 0.985
	case 4:
		perLaneGBps = 1.969
	case 5:
		perLaneGBps = 3.938
	case 6:
		perLaneGBps = 7.877
	default:
		perLaneGBps = 1.969
	}
	if width <= 0 {
		width = 16
	}
	return perLaneGBps * float64(width)
}

// readPcieThroughput returns live PCIe RX and TX transfer rates in MB/s. The nvidia-smi dmon probe behind it is
// rate-limited and shared (nvprobe.go), so this is usually a cached read: dmon costs about 117 ms, a whole
// sampling period, and the sparkline does not need a point per second.
func (tc *TelemetryCollector) readPcieThroughput() (rxMB, txMB float64, ok bool) {
	return nvidiaPCIeThroughput(time.Now())
}
